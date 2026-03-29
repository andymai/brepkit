//! Solid-level tessellation orchestration.

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::TriangleMesh;
use super::edge_sampling::{circle_param_range, sample_edge, segments_for_chord_deviation};
use super::mesh_ops::weld_boundary_vertices;
use super::nonplanar::{tessellate_nonplanar_cdt, tessellate_nonplanar_snap};
use super::nurbs::{compute_angular_range, compute_v_param_range};
use super::planar::{
    cdt_triangulate_simple, collect_wire_global_vertices, project_by_normal,
    remove_closing_duplicate_global, remove_closing_duplicate_ids, run_planar_cdt,
    tessellate_planar_shared_with_holes,
};
use super::{MERGE_GRID, point_merge_key};

/// Tessellate all faces of a solid into a single watertight triangle mesh.
///
/// Unlike per-face `tessellate()`, this function coordinates tessellation across
/// all faces of the solid by pre-computing shared edge tessellations. When two
/// faces share an edge, the edge is tessellated once and both faces receive
/// identical vertices along that boundary -- eliminating cracks between adjacent
/// faces and producing a guaranteed 2-manifold mesh.
///
/// # Algorithm
///
/// Based on Stoger & Kurka (2003), "Watertight Tessellation of B-rep NURBS
/// CAD-Models Using Connectivity Information":
///
/// 1. Build edge-to-face adjacency map from the solid's topology.
/// 2. Tessellate each unique edge once, producing a shared polyline.
/// 3. For each face, tessellate using cached edge points as boundary vertices.
/// 4. Merge all per-face meshes into a single mesh with shared boundary vertices.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
#[allow(clippy::too_many_lines)]
pub fn tessellate_solid(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_topology::explorer;

    // Phase 1: Collect all faces and build edge->face adjacency.
    let all_faces = explorer::solid_faces(topo, solid)?;
    let edge_face_map = explorer::edge_to_face_map(topo, solid)?;

    // Phase 2: Tessellate each unique edge once.
    let edge_indices: Vec<usize> = edge_face_map.keys().copied().collect();
    #[cfg(not(target_arch = "wasm32"))]
    let mut edge_points: HashMap<usize, Vec<Point3>> = if edge_indices.len() >= 32 {
        use rayon::prelude::*;
        let results: Vec<Result<(usize, Vec<Point3>), crate::OperationsError>> = edge_indices
            .par_iter()
            .filter_map(|&edge_idx| {
                let edge_id = topo.edge_id_from_index(edge_idx)?;
                let edge_data = match topo.edge(edge_id) {
                    Ok(d) => d,
                    Err(e) => return Some(Err(crate::OperationsError::Topology(e))),
                };
                Some(sample_edge(topo, edge_data, deflection).map(|pts| (edge_idx, pts)))
            })
            .collect();
        let mut map = HashMap::new();
        for r in results {
            let (idx, pts) = r?;
            map.insert(idx, pts);
        }
        map
    } else {
        let mut map = HashMap::new();
        for &edge_idx in &edge_indices {
            if let Some(edge_id) = topo.edge_id_from_index(edge_idx) {
                if let Ok(edge_data) = topo.edge(edge_id) {
                    let points = sample_edge(topo, edge_data, deflection)?;
                    map.insert(edge_idx, points);
                }
            }
        }
        map
    };
    #[cfg(target_arch = "wasm32")]
    let mut edge_points: HashMap<usize, Vec<Point3>> = {
        let mut map = HashMap::new();
        for &edge_idx in &edge_indices {
            if let Some(edge_id) = topo.edge_id_from_index(edge_idx) {
                if let Ok(edge_data) = topo.edge(edge_id) {
                    let points = sample_edge(topo, edge_data, deflection)?;
                    map.insert(edge_idx, points);
                }
            }
        }
        map
    };

    // Phase 2b: Synchronize circle edge samples with face grid density.
    {
        for &face_id in &all_faces {
            let face_data = topo.face(face_id)?;
            let face_nu = match face_data.surface() {
                FaceSurface::Cone(cone) => {
                    let v_range =
                        compute_v_param_range(topo, face_data, |p| cone.project_point(p).1);
                    let u_range = compute_angular_range(topo, face_data, |p| cone.project_point(p));
                    let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
                    segments_for_chord_deviation(
                        max_radius.max(0.01),
                        u_range.1 - u_range.0,
                        deflection,
                    )
                }
                FaceSurface::Cylinder(cyl) => {
                    let u_range = compute_angular_range(topo, face_data, |p| cyl.project_point(p));
                    segments_for_chord_deviation(cyl.radius(), u_range.1 - u_range.0, deflection)
                }
                _ => continue,
            };
            let expected_count = face_nu + 1;

            let mut wire_ids = vec![face_data.outer_wire()];
            wire_ids.extend_from_slice(face_data.inner_wires());
            for &wire_id in &wire_ids {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    let Some(edge_id) = topo.edge_id_from_index(edge_idx) else {
                        continue;
                    };
                    let Ok(edge_data) = topo.edge(edge_id) else {
                        continue;
                    };
                    let EdgeCurve::Circle(circle) = edge_data.curve() else {
                        continue;
                    };

                    if let Some(pts) = edge_points.get(&edge_idx) {
                        if pts.len() < expected_count {
                            let (t_start, t_end) = circle_param_range(topo, edge_data, circle)?;
                            let new_pts = brepkit_geometry::sampling::sample_uniform(
                                circle,
                                t_start,
                                t_end,
                                expected_count,
                            );
                            edge_points.insert(edge_idx, new_pts);
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Build merged mesh with shared edge vertices.
    let mut merged = TriangleMesh::default();
    let mut point_to_global: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut edge_global_indices: HashMap<usize, Vec<u32>> = HashMap::new();

    for (&edge_idx, points) in &edge_points {
        let mut global_ids = Vec::with_capacity(points.len());
        for &pt in points {
            let key = point_merge_key(pt, MERGE_GRID);
            let idx = point_to_global.entry(key).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt);
                merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                idx
            });
            global_ids.push(*idx);
        }
        edge_global_indices.insert(edge_idx, global_ids);
    }

    // Phase 3b: Circle edge refinement.
    {
        use std::collections::HashSet;

        let tol_linear = brepkit_math::tolerance::Tolerance::new().linear;
        let refine_tol = tol_linear * 10.0;

        for &edge_idx in &edge_indices {
            let Some(edge_id) = topo.edge_id_from_index(edge_idx) else {
                continue;
            };
            let Ok(edge_data) = topo.edge(edge_id) else {
                continue;
            };
            let EdgeCurve::Circle(circle) = edge_data.curve() else {
                continue;
            };

            let Ok(start_vtx) = topo.vertex(edge_data.start()) else {
                continue;
            };
            let Ok(end_vtx) = topo.vertex(edge_data.end()) else {
                continue;
            };
            let start_pos = start_vtx.point();
            let end_pos = end_vtx.point();

            let (t_min, t_max) = edge_data.curve().domain_with_endpoints(start_pos, end_pos);
            let is_closed = edge_data.start() == edge_data.end();

            let existing_gids_vec: Vec<u32> = edge_global_indices
                .get(&edge_idx)
                .cloned()
                .unwrap_or_default();
            let existing_gids: HashSet<u32> = existing_gids_vec.iter().copied().collect();

            let mut insertions: Vec<(f64, u32)> = Vec::new();
            for (gid, pos) in merged.positions.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let gid32 = gid as u32;
                if existing_gids.contains(&gid32) {
                    continue;
                }
                if (*pos - start_pos).length() < refine_tol {
                    continue;
                }
                if !is_closed && (*pos - end_pos).length() < refine_tol {
                    continue;
                }
                let t = circle.project(*pos);
                let on_circle = circle.evaluate(t);
                let dist = (*pos - on_circle).length();
                if dist >= refine_tol {
                    continue;
                }
                let in_range = if is_closed {
                    true
                } else if t_min < t_max {
                    t >= t_min - 1e-8 && t <= t_max + 1e-8
                } else {
                    t >= t_min - 1e-8 || t <= t_max + 1e-8
                };
                if in_range {
                    insertions.push((t, gid32));
                }
            }

            if insertions.is_empty() {
                continue;
            }

            insertions.sort_by(|a, b| a.0.total_cmp(&b.0));
            insertions.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-8);

            let mut all_with_t: Vec<(f64, u32)> = existing_gids_vec
                .iter()
                .map(|&gid| {
                    let pos = merged.positions[gid as usize];
                    (circle.project(pos), gid)
                })
                .collect();
            all_with_t.extend(insertions);
            all_with_t.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut seen_gids = HashSet::new();
            all_with_t.retain(|(_, gid)| seen_gids.insert(*gid));

            let refined: Vec<u32> = all_with_t.into_iter().map(|(_, gid)| gid).collect();
            edge_global_indices.insert(edge_idx, refined);
        }
    }

    // Phase 4: Tessellate each face using its boundary edge vertices.
    #[allow(clippy::items_after_statements)]
    struct CdtJob {
        pts2d: Vec<brepkit_math::vec::Point2>,
        outer_count: usize,
        inner_wire_ranges: Vec<(usize, usize)>,
        all_global_ids: Vec<Option<u32>>,
        all_positions: Vec<Point3>,
        normal: Vec3,
        is_reversed: bool,
    }
    #[allow(clippy::items_after_statements)]
    type CdtResult = Result<Vec<(usize, usize, usize)>, crate::OperationsError>;

    // Phase 4a: Collect CDT jobs for large planar faces with holes.
    let mut cdt_jobs: Vec<CdtJob> = Vec::new();
    let mut other_face_indices: Vec<usize> = Vec::new();

    for (fi, &face_id) in all_faces.iter().enumerate() {
        let face_data = topo.face(face_id)?;
        let has_inner = !face_data.inner_wires().is_empty();
        if let FaceSurface::Plane { normal, .. } = face_data.surface() {
            if has_inner {
                let normal = *normal;
                let is_reversed = face_data.is_reversed();
                let wire = topo.wire(face_data.outer_wire())?;
                let tol = 1e-10;

                let (mut all_positions, mut all_global_ids) = collect_wire_global_vertices(
                    wire,
                    &edge_global_indices,
                    &merged.positions,
                    tol,
                );
                remove_closing_duplicate_global(
                    &mut all_positions,
                    &mut all_global_ids,
                    &merged.positions,
                    tol,
                );
                let outer_count = all_positions.len();

                let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();
                for &iw_id in face_data.inner_wires() {
                    let iw = topo.wire(iw_id)?;
                    let start = all_positions.len();
                    let (inner_pos, inner_gids) = collect_wire_global_vertices(
                        iw,
                        &edge_global_indices,
                        &merged.positions,
                        tol,
                    );
                    let mut inner_flat_ids: Vec<u32> = Vec::with_capacity(inner_gids.len());
                    let mut next_sentinel = u32::MAX;
                    for (pos, gid_opt) in inner_pos.into_iter().zip(inner_gids) {
                        let gid = gid_opt.unwrap_or_else(|| {
                            debug_assert!(false, "inner wire vertex had no global ID");
                            let s = next_sentinel;
                            next_sentinel = next_sentinel.wrapping_sub(1);
                            s
                        });
                        inner_flat_ids.push(gid);
                        all_positions.push(pos);
                        all_global_ids.push(Some(gid));
                    }
                    if inner_flat_ids.len() > 2 {
                        remove_closing_duplicate_ids(&mut inner_flat_ids, &merged.positions, tol);
                        let expected_end = start + inner_flat_ids.len();
                        all_positions.truncate(expected_end);
                        all_global_ids.truncate(expected_end);
                    }
                    let end = all_positions.len();
                    inner_wire_ranges.push((start, end));
                }

                let pts2d: Vec<brepkit_math::vec::Point2> = all_positions
                    .iter()
                    .map(|&p| project_by_normal(p, normal))
                    .collect();

                cdt_jobs.push(CdtJob {
                    pts2d,
                    outer_count,
                    inner_wire_ranges,
                    all_global_ids,
                    all_positions,
                    normal,
                    is_reversed,
                });
                continue;
            }
        }
        other_face_indices.push(fi);
    }

    // Phase 4b: Run CDTs in parallel for large planar faces.
    #[cfg(not(target_arch = "wasm32"))]
    let cdt_results: Vec<CdtResult> = if cdt_jobs.len() >= 2 {
        use rayon::prelude::*;
        cdt_jobs
            .par_iter()
            .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
            .collect()
    } else {
        cdt_jobs
            .iter()
            .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
            .collect()
    };
    #[cfg(target_arch = "wasm32")]
    let cdt_results: Vec<CdtResult> = cdt_jobs
        .iter()
        .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
        .collect();

    // Phase 4c: Merge CDT results into the shared mesh (sequential).
    for (job, result) in cdt_jobs.iter().zip(cdt_results) {
        let tris = result?;

        let needs_flip = if let Some(&(i0, i1, i2)) = tris.first() {
            let p0 = job.all_positions[i0];
            let p1 = job.all_positions[i1];
            let p2 = job.all_positions[i2];
            let a = p1 - p0;
            let b = p2 - p0;
            let winding_matches = a.cross(b).dot(job.normal) > 0.0;
            winding_matches == job.is_reversed
        } else {
            false
        };

        for &(i0, i1, i2) in &tris {
            let g0 = job.all_global_ids[i0].unwrap_or(0);
            let g1 = job.all_global_ids[i1].unwrap_or(0);
            let g2 = job.all_global_ids[i2].unwrap_or(0);
            if needs_flip {
                merged.indices.push(g0);
                merged.indices.push(g2);
                merged.indices.push(g1);
            } else {
                merged.indices.push(g0);
                merged.indices.push(g1);
                merged.indices.push(g2);
            }
        }
    }

    // Phase 4d: Process remaining faces sequentially.
    for &fi in &other_face_indices {
        if let Err(e) = tessellate_face_with_shared_edges(
            topo,
            all_faces[fi],
            deflection,
            &edge_global_indices,
            &mut merged,
            &mut point_to_global,
        ) {
            log::warn!("skipping face during tessellation: {e}");
        }
    }

    // Phase 5: Surface-aware vertex normals.
    let n_verts = merged.positions.len();
    let tri_count = merged.indices.len() / 3;

    let mut needs_normal = vec![false; n_verts];
    for i in 0..n_verts {
        let n = &merged.normals[i];
        if n.x().abs() < 1e-30 && n.y().abs() < 1e-30 && n.z().abs() < 1e-30 {
            needs_normal[i] = true;
        }
    }

    {
        use std::collections::HashSet;

        let mut vertex_faces: HashMap<usize, HashSet<FaceId>> = HashMap::new();
        for (&edge_idx, global_ids) in &edge_global_indices {
            if let Some(face_ids) = edge_face_map.get(&edge_idx) {
                for &gid in global_ids {
                    let gi = gid as usize;
                    if gi < n_verts && needs_normal[gi] {
                        let entry = vertex_faces.entry(gi).or_default();
                        for &fid in face_ids {
                            entry.insert(fid);
                        }
                    }
                }
            }
        }

        let mut fallback_needed = vec![false; n_verts];
        for i in 0..n_verts {
            if !needs_normal[i] {
                continue;
            }
            let pos = merged.positions[i];
            let mut normal_sum = Vec3::new(0.0, 0.0, 0.0);
            let mut count = 0_u32;
            if let Some(faces) = vertex_faces.get(&i) {
                for &fid in faces {
                    if let Ok(face_data) = topo.face(fid) {
                        let surf = face_data.surface();
                        if let Some(n) = crate::fillet::face_surface_normal_at(surf, pos) {
                            let oriented = if face_data.is_reversed() {
                                Vec3::new(-n.x(), -n.y(), -n.z())
                            } else {
                                n
                            };
                            normal_sum += oriented;
                            count += 1;
                        }
                    }
                }
            }
            if count > 0 {
                merged.normals[i] = normal_sum.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            } else {
                fallback_needed[i] = true;
            }
        }

        if fallback_needed.iter().any(|&f| f) {
            let mut accum: Vec<Vec3> = vec![Vec3::new(0.0, 0.0, 0.0); n_verts];
            for t in 0..tri_count {
                let i0 = merged.indices[t * 3] as usize;
                let i1 = merged.indices[t * 3 + 1] as usize;
                let i2 = merged.indices[t * 3 + 2] as usize;
                let a = merged.positions[i1] - merged.positions[i0];
                let b = merged.positions[i2] - merged.positions[i0];
                let face_normal = a.cross(b);
                if fallback_needed.get(i0).copied().unwrap_or(false) {
                    accum[i0] += face_normal;
                }
                if fallback_needed.get(i1).copied().unwrap_or(false) {
                    accum[i1] += face_normal;
                }
                if fallback_needed.get(i2).copied().unwrap_or(false) {
                    accum[i2] += face_normal;
                }
            }
            for i in 0..n_verts {
                if fallback_needed[i] {
                    merged.normals[i] = accum[i].normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                }
            }
        }
    }

    // Phase 6: Weld boundary vertices.
    weld_boundary_vertices(&mut merged, deflection);

    Ok(merged)
}

/// Tessellate a single face, reusing shared edge vertices from the global mesh.
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_face_with_shared_edges(
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    let face_data = topo.face(face_id)?;
    let is_reversed = face_data.is_reversed();

    let idx_start = merged.indices.len();
    let pos_start = merged.positions.len();

    if let FaceSurface::Plane { normal, .. } = face_data.surface() {
        let normal = *normal;
        let wire = topo.wire(face_data.outer_wire())?;

        let mut boundary_global_ids: Vec<u32> = Vec::new();
        let tol = 1e-10;

        for oe in wire.edges() {
            let edge_idx = oe.edge().index();
            if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
                let is_fwd = oe.is_forward();
                let len = global_ids.len();
                for j in 0..len {
                    let gid = if is_fwd {
                        global_ids[j]
                    } else {
                        global_ids[len - 1 - j]
                    };
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if last_gid == gid {
                            continue;
                        }
                        if (last_gid as usize) < merged.positions.len()
                            && (gid as usize) < merged.positions.len()
                        {
                            let last_pos = merged.positions[last_gid as usize];
                            let this_pos = merged.positions[gid as usize];
                            if (last_pos - this_pos).length() < tol {
                                continue;
                            }
                        }
                    }
                    boundary_global_ids.push(gid);
                }
            } else {
                let edge_data = topo.edge(oe.edge())?;
                let points = sample_edge(topo, edge_data, deflection)?;
                let ordered: Vec<Point3> = if oe.is_forward() {
                    points
                } else {
                    points.into_iter().rev().collect()
                };
                for (j, pt) in ordered.iter().enumerate() {
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if (last_gid as usize) < merged.positions.len() {
                            let last_pos = merged.positions[last_gid as usize];
                            if (last_pos - *pt).length() < tol {
                                continue;
                            }
                        }
                    }
                    let key = point_merge_key(*pt, MERGE_GRID);
                    let gid = point_to_global.entry(key).or_insert_with(|| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = merged.positions.len() as u32;
                        merged.positions.push(*pt);
                        merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                        idx
                    });
                    boundary_global_ids.push(*gid);
                }
            }
        }

        remove_closing_duplicate_ids(&mut boundary_global_ids, &merged.positions, tol);

        let n = boundary_global_ids.len();
        if n < 3 {
            return Ok(());
        }

        let local_positions: Vec<Point3> = boundary_global_ids
            .iter()
            .map(|&gid| merged.positions[gid as usize])
            .collect();

        if face_data.inner_wires().is_empty() {
            let mut local_indices = cdt_triangulate_simple(&local_positions, normal);

            if local_indices.len() >= 3 {
                let i0 = local_indices[0] as usize;
                let i1 = local_indices[1] as usize;
                let i2 = local_indices[2] as usize;
                let a = local_positions[i1] - local_positions[i0];
                let b = local_positions[i2] - local_positions[i0];
                let tri_normal = a.cross(b);
                if tri_normal.dot(normal) < 0.0 {
                    for t in 0..local_indices.len() / 3 {
                        local_indices.swap(t * 3 + 1, t * 3 + 2);
                    }
                }
            }

            for &li in &local_indices {
                merged.indices.push(boundary_global_ids[li as usize]);
            }
        } else {
            tessellate_planar_shared_with_holes(
                topo,
                face_data,
                &boundary_global_ids,
                &local_positions,
                normal,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(face_data.surface(), FaceSurface::Nurbs(_)) {
        let cdt_ok = tessellate_nonplanar_cdt(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        );
        if cdt_ok.is_err() {
            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(
        face_data.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        let is_standard_rect = {
            let wire = topo.wire(face_data.outer_wire())?;
            wire.edges().len() <= 4
                && wire.edges().iter().all(|oe| {
                    topo.edge(oe.edge())
                        .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line | EdgeCurve::Circle(_)))
                })
        };

        if is_standard_rect {
            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        } else {
            let pos_save = merged.positions.len();
            let nrm_save = merged.normals.len();
            let idx_save = merged.indices.len();
            let cdt_ok = tessellate_nonplanar_cdt(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            );
            if cdt_ok.is_err() || merged.indices.len() == idx_save {
                merged.positions.truncate(pos_save);
                merged.normals.truncate(nrm_save);
                merged.indices.truncate(idx_save);
                tessellate_nonplanar_snap(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?;
            }
        }
    } else {
        let pos_save = merged.positions.len();
        let nrm_save = merged.normals.len();
        let idx_save = merged.indices.len();
        let ptg_count_save = point_to_global.len();

        let cdt_ok = tessellate_nonplanar_cdt(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        );
        let cdt_produced_tris = cdt_ok.is_ok() && merged.indices.len() > idx_save;
        if !cdt_produced_tris {
            merged.positions.truncate(pos_save);
            merged.normals.truncate(nrm_save);
            merged.indices.truncate(idx_save);
            if point_to_global.len() > ptg_count_save {
                point_to_global.retain(|_, v| (*v as usize) < pos_save);
            }

            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    }

    if is_reversed {
        let idx_end = merged.indices.len();
        let tri_count = (idx_end - idx_start) / 3;
        for t in 0..tri_count {
            let base = idx_start + t * 3;
            merged.indices.swap(base + 1, base + 2);
        }
        for n in &mut merged.normals[pos_start..] {
            *n = -*n;
        }
    }

    Ok(())
}
