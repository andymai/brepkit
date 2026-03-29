//! Non-planar CDT and fallback paths for face tessellation.

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::edge_sampling::{sample_edge, segments_for_chord_deviation};
use super::{MERGE_GRID, TriangleMesh, point_merge_key};

/// CDT-based tessellation for non-planar faces with exact boundary constraints.
///
/// Projects shared edge points into (u,v) parameter space, generates interior
/// sample points, then runs Constrained Delaunay Triangulation. Boundary
/// vertices use their pre-existing global IDs (watertight by construction).
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_nonplanar_cdt(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    use std::collections::HashSet;

    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use brepkit_topology::edge::EdgeId;

    // Step 1: Collect boundary points in wire-traversal order with global IDs.
    let wire = topo.wire(face_data.outer_wire())?;
    let tol_dup = 1e-10;

    // Fourth element: is_forward flag -- needed for seam UV assignment.
    let mut boundary_3d: Vec<(Point3, u32, EdgeId, bool)> = Vec::new();
    for oe in wire.edges() {
        let edge_id_local = oe.edge();
        let edge_idx = edge_id_local.index();
        let is_fwd = oe.is_forward();
        if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
            let ordered: Vec<u32> = if is_fwd {
                global_ids.clone()
            } else {
                global_ids.iter().rev().copied().collect()
            };
            for (j, &gid) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (_, last_gid, _, _) = boundary_3d[boundary_3d.len() - 1];
                    if last_gid == gid
                        || (merged.positions[last_gid as usize] - merged.positions[gid as usize])
                            .length()
                            < tol_dup
                    {
                        continue;
                    }
                }
                boundary_3d.push((merged.positions[gid as usize], gid, edge_id_local, is_fwd));
            }
        } else {
            // Edge not in shared pool -- insert directly.
            let edge_data = topo.edge(oe.edge())?;
            let points = sample_edge(topo, edge_data, deflection)?;
            let ordered: Vec<Point3> = if is_fwd {
                points
            } else {
                points.into_iter().rev().collect()
            };
            for (j, &pt) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (last_pos, _, _, _) = boundary_3d[boundary_3d.len() - 1];
                    if (last_pos - pt).length() < tol_dup {
                        continue;
                    }
                }
                let key = point_merge_key(pt, MERGE_GRID);
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(pt);
                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                    idx
                });
                boundary_3d.push((pt, gid, edge_id_local, is_fwd));
            }
        }
    }

    // Remove closing duplicate.
    if boundary_3d.len() > 2 {
        if let (Some(&(_, first_gid, _, _)), Some(&(_, last_gid, _, _))) =
            (boundary_3d.first(), boundary_3d.last())
        {
            if first_gid == last_gid
                || (merged.positions[first_gid as usize] - merged.positions[last_gid as usize])
                    .length()
                    < tol_dup
            {
                boundary_3d.pop();
            }
        }
    }

    let n_boundary = boundary_3d.len();
    if n_boundary < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "non-planar face has fewer than 3 boundary vertices".to_string(),
        });
    }

    // Step 2: Project boundary 3D points to (u,v) parameter space.
    let mut boundary_uv: Vec<(f64, f64)> = boundary_3d
        .iter()
        .map(|(pt, _, edge_id_local, _)| {
            // Try PCurve lookup first.
            if let Some(pcurve) = topo.pcurves().get(*edge_id_local, face_id) {
                let uv = project_via_pcurve(pcurve, *pt, face_data.surface());
                if let Some(uv) = uv {
                    return Ok(uv);
                }
            }
            // Fall back to surface projection.
            project_to_surface_uv(face_data.surface(), *pt)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 2a: Unwrap periodic u across the seam for polyline boundaries.
    {
        let is_periodic = matches!(
            face_data.surface(),
            FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_)
        );
        if is_periodic && !boundary_uv.is_empty() {
            for i in 1..boundary_uv.len() {
                let prev_u = boundary_uv[i - 1].0;
                let mut u = boundary_uv[i].0;
                let diff = u - prev_u;
                let shifts = (diff / std::f64::consts::TAU + 0.5).floor();
                u -= shifts * std::f64::consts::TAU;
                boundary_uv[i].0 = u;
            }
            let first_u = boundary_uv[0].0;
            let last_u = boundary_uv.last().map_or(first_u, |p| p.0);
            let close_diff = first_u - last_u;
            if close_diff.abs() > std::f64::consts::PI {
                let u_mid = boundary_uv.iter().map(|p| p.0).sum::<f64>() / boundary_uv.len() as f64;
                let target_mid = std::f64::consts::PI;
                let shift = target_mid - u_mid;
                for pt in &mut boundary_uv {
                    pt.0 += shift;
                }
            }
        }
    }

    // Compute (u,v) bounding box from a set of UV pairs.
    #[allow(clippy::items_after_statements)]
    fn uv_bounds(uvs: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        uvs.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(u_lo, u_hi, v_lo, v_hi), &(u, v)| {
                (u_lo.min(u), u_hi.max(u), v_lo.min(v), v_hi.max(v))
            },
        )
    }
    let (u_min, u_max, v_min, v_max) = uv_bounds(&boundary_uv);

    // Step 2b: Detect and fix degenerate seam edges.
    let (u_min, u_max, v_min, v_max) = {
        let mut wire_edge_counts: HashMap<usize, usize> = HashMap::new();
        for oe in wire.edges() {
            *wire_edge_counts.entry(oe.edge().index()).or_default() += 1;
        }
        let seam_edge_indices: HashSet<usize> = wire_edge_counts
            .iter()
            .filter(|&(_, &c)| c > 1)
            .map(|(&idx, _)| idx)
            .collect();

        if !seam_edge_indices.is_empty() {
            let non_seam_uvs: Vec<(f64, f64)> = boundary_uv
                .iter()
                .enumerate()
                .filter(|(i, _)| !seam_edge_indices.contains(&boundary_3d[*i].2.index()))
                .map(|(_, &uv)| uv)
                .collect();
            let (u_min_bnd, u_max_bnd, v_min_bnd, v_max_bnd) = if non_seam_uvs.is_empty() {
                (u_min, u_max, v_min, v_max)
            } else {
                uv_bounds(&non_seam_uvs)
            };

            #[allow(clippy::items_after_statements)]
            struct SeamRun {
                indices: Vec<usize>,
                is_forward: bool,
            }
            let mut seam_runs: Vec<SeamRun> = Vec::new();
            let mut current_indices: Vec<usize> = Vec::new();
            let mut current_fwd: Option<bool> = None;
            for i in 0..n_boundary {
                let (_, _, edge_id, is_fwd) = boundary_3d[i];
                if seam_edge_indices.contains(&edge_id.index()) {
                    current_indices.push(i);
                    if current_fwd.is_none() {
                        current_fwd = Some(is_fwd);
                    }
                } else if !current_indices.is_empty() {
                    seam_runs.push(SeamRun {
                        indices: std::mem::take(&mut current_indices),
                        is_forward: current_fwd.unwrap_or(true),
                    });
                    current_fwd = None;
                }
            }
            if !current_indices.is_empty() {
                let tail_fwd = current_fwd.unwrap_or(true);
                if !seam_runs.is_empty()
                    && seam_edge_indices.contains(&boundary_3d[0].2.index())
                    && seam_runs[0].is_forward == tail_fwd
                {
                    current_indices.extend(seam_runs.remove(0).indices);
                }
                seam_runs.push(SeamRun {
                    indices: current_indices,
                    is_forward: tail_fwd,
                });
            }

            for run in &seam_runs {
                let u_assign = if run.is_forward { u_max_bnd } else { u_min_bnd };
                let n_pts = run.indices.len();

                let v_first = boundary_uv[run.indices[0]].1;
                let (v_start, v_end) = if (v_first - v_min_bnd).abs() < (v_first - v_max_bnd).abs()
                {
                    (v_min_bnd, v_max_bnd)
                } else {
                    (v_max_bnd, v_min_bnd)
                };

                for (k, &i) in run.indices.iter().enumerate() {
                    let t = if n_pts > 1 {
                        k as f64 / (n_pts - 1) as f64
                    } else {
                        0.5
                    };
                    let v = v_start + t * (v_end - v_start);
                    boundary_uv[i] = (u_assign, v);
                }
            }
        }

        // Recompute UV bounding box after seam fix.
        uv_bounds(&boundary_uv)
    };

    let margin = 0.01;
    let bounds = (
        Point2::new(u_min - margin, v_min - margin),
        Point2::new(u_max + margin, v_max + margin),
    );
    let mut cdt = Cdt::with_capacity(bounds, n_boundary);

    // Step 3: Insert boundary points into CDT.
    let mut cdt_to_global: Vec<Option<u32>> = vec![None; 3]; // 3 super-triangle verts

    let boundary_pts: Vec<Point2> = boundary_uv
        .iter()
        .map(|&(u, v)| Point2::new(u, v))
        .collect();
    let boundary_cdt_ids = cdt
        .insert_points_hilbert(&boundary_pts)
        .map_err(crate::OperationsError::Math)?;
    let max_cdt_idx = boundary_cdt_ids.iter().copied().max().unwrap_or(2);
    if cdt_to_global.len() <= max_cdt_idx {
        cdt_to_global.resize(max_cdt_idx + 1, None);
    }
    for (i, &cdt_idx) in boundary_cdt_ids.iter().enumerate() {
        cdt_to_global[cdt_idx] = Some(boundary_3d[i].1);
    }

    // Step 4: Insert boundary constraints.
    for i in 0..n_boundary {
        let v0 = boundary_cdt_ids[i];
        let v1 = boundary_cdt_ids[(i + 1) % n_boundary];
        cdt.insert_constraint(v0, v1)
            .map_err(crate::OperationsError::Math)?;
    }

    // Step 5: Generate interior sample points.
    let du = u_max - u_min;
    let dv = v_max - v_min;
    if du > 1e-15 && dv > 1e-15 {
        let (n_u, n_v) = interior_grid_resolution(face_data.surface(), du, dv, deflection);

        let boundary_uv_ref = &boundary_uv;
        let interior_pts: Vec<Point2> = (1..n_u)
            .flat_map(|iu| {
                (1..n_v).filter_map(move |iv| {
                    let u = u_min + du * (iu as f64 / n_u as f64);
                    let v = v_min + dv * (iv as f64 / n_v as f64);
                    let pt2 = Point2::new(u, v);
                    point_in_polygon_2d(boundary_uv_ref, pt2).then_some(pt2)
                })
            })
            .collect();
        if !interior_pts.is_empty() {
            let interior_cdt_ids = cdt
                .insert_points_hilbert(&interior_pts)
                .map_err(crate::OperationsError::Math)?;
            let max_interior = interior_cdt_ids.iter().copied().max().unwrap_or(0);
            if cdt_to_global.len() <= max_interior {
                cdt_to_global.resize(max_interior + 1, None);
            }
        }
    }

    // Step 6: Remove triangles outside the boundary polygon.
    let boundary_pairs: Vec<(usize, usize)> = (0..n_boundary)
        .map(|i| (boundary_cdt_ids[i], boundary_cdt_ids[(i + 1) % n_boundary]))
        .collect();
    cdt.remove_exterior(&boundary_pairs);

    // Step 7: Assign global IDs to interior CDT vertices and emit triangles.
    let cdt_verts = cdt.vertices();
    let triangles = cdt.triangles();

    let mut final_global_ids: Vec<u32> = vec![0; cdt_to_global.len()];

    for i in 0..cdt_to_global.len() {
        if let Some(gid) = cdt_to_global[i] {
            final_global_ids[i] = gid;
        } else if i >= 3 {
            let pt2 = cdt_verts[i];
            let surface = face_data.surface();
            let pt3 = eval_surface_point(surface, pt2.x(), pt2.y());
            let nrm = surface.normal(pt2.x(), pt2.y());

            let key = point_merge_key(pt3, MERGE_GRID);
            let gid = *point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt3);
                merged.normals.push(nrm);
                idx
            });
            final_global_ids[i] = gid;
        }
    }

    // Emit triangles.
    for (i0, i1, i2) in triangles {
        if i0 < 3 || i1 < 3 || i2 < 3 {
            continue; // Skip super-triangle vertices
        }
        merged.indices.push(final_global_ids[i0]);
        merged.indices.push(final_global_ids[i1]);
        merged.indices.push(final_global_ids[i2]);
    }

    Ok(())
}

/// Project a 3D point onto a face surface, returning (u, v) parameters.
fn project_to_surface_uv(
    surface: &FaceSurface,
    pt: Point3,
) -> Result<(f64, f64), crate::OperationsError> {
    match surface {
        FaceSurface::Cylinder(cyl) => Ok(cyl.project_point(pt)),
        FaceSurface::Cone(cone) => Ok(cone.project_point(pt)),
        FaceSurface::Sphere(sphere) => Ok(sphere.project_point(pt)),
        FaceSurface::Torus(torus) => Ok(torus.project_point(pt)),
        FaceSurface::Nurbs(surface) => {
            brepkit_math::nurbs::projection::project_point_to_surface(surface, pt, 1e-6)
                .map(|proj| (proj.u, proj.v))
                .map_err(crate::OperationsError::Math)
        }
        FaceSurface::Plane { .. } => Err(crate::OperationsError::InvalidInput {
            reason: "planar faces should not use CDT tessellation".to_string(),
        }),
    }
}

/// Try to find (u,v) coordinates for a 3D point using a PCurve.
fn project_via_pcurve(
    pcurve: &brepkit_topology::pcurve::PCurve,
    pt: Point3,
    surface: &FaceSurface,
) -> Option<(f64, f64)> {
    let t_start = pcurve.t_start();
    let t_end = pcurve.t_end();
    let n_samples = 16;

    let mut best_t = t_start;
    let mut best_dist = f64::MAX;

    for i in 0..=n_samples {
        let t = t_start + (t_end - t_start) * (i as f64) / (n_samples as f64);
        let uv = pcurve.evaluate(t);
        let p_surf = eval_surface_point(surface, uv.x(), uv.y());
        let d = (p_surf - pt).length();
        if d < best_dist {
            best_dist = d;
            best_t = t;
        }
    }

    // Refine with bisection around best_t.
    let dt = (t_end - t_start) / (n_samples as f64);
    let mut lo = (best_t - dt).max(t_start);
    let mut hi = (best_t + dt).min(t_end);
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let uv_lo = pcurve.evaluate(lo);
        let uv_hi = pcurve.evaluate(hi);
        let d_lo = (eval_surface_point(surface, uv_lo.x(), uv_lo.y()) - pt).length();
        let d_hi = (eval_surface_point(surface, uv_hi.x(), uv_hi.y()) - pt).length();
        if d_lo < d_hi {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let t_final = 0.5 * (lo + hi);
    let uv = pcurve.evaluate(t_final);
    let p_final = eval_surface_point(surface, uv.x(), uv.y());

    if (p_final - pt).length() < brepkit_math::tolerance::Tolerance::default().linear {
        Some((uv.x(), uv.y()))
    } else {
        None
    }
}

/// Evaluate a non-planar surface at `(u, v)` and return a 3D point.
fn eval_surface_point(surface: &FaceSurface, u: f64, v: f64) -> Point3 {
    surface.evaluate(u, v).unwrap_or(Point3::new(0.0, 0.0, 0.0))
}

/// Estimate the effective radius of a surface for sample density calculation.
fn estimate_surface_radius(surface: &FaceSurface) -> f64 {
    match surface {
        FaceSurface::Cylinder(cyl) => cyl.radius(),
        FaceSurface::Cone(_) => 1.0,
        FaceSurface::Sphere(sphere) => sphere.radius(),
        FaceSurface::Torus(torus) => torus.major_radius() + torus.minor_radius(),
        FaceSurface::Nurbs(_) | FaceSurface::Plane { .. } => 1.0,
    }
}

/// Compute interior grid resolution for `tessellate_nonplanar_cdt`.
fn interior_grid_resolution(
    surface: &FaceSurface,
    du: f64,
    dv: f64,
    deflection: f64,
) -> (usize, usize) {
    match surface {
        FaceSurface::Sphere(sphere) => {
            let r = sphere.radius();
            let n_u = segments_for_chord_deviation(r, du, deflection).max(2);
            let n_v = segments_for_chord_deviation(r, dv, deflection).max(2);
            (n_u, n_v)
        }
        FaceSurface::Torus(torus) => {
            let n_u = segments_for_chord_deviation(torus.major_radius(), du, deflection).max(2);
            let n_v = segments_for_chord_deviation(torus.minor_radius(), dv, deflection).max(2);
            (n_u, n_v)
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_) => {
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation(r, du, deflection).max(2);
            let n_v = segments_for_chord_deviation(r, dv, deflection).max(2);
            (n_u, n_v)
        }
    }
}

/// Check if a 2D point is inside a polygon defined by (u, v) coordinates.
/// Uses the winding number algorithm for robustness.
pub(super) fn point_in_polygon_2d(polygon: &[(f64, f64)], pt: brepkit_math::vec::Point2) -> bool {
    let n = polygon.len();
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = polygon[i].1;
        let yj = polygon[j].1;
        if yi <= pt.y() {
            if yj > pt.y() {
                let cross = (polygon[j].0 - polygon[i].0) * (pt.y() - yi)
                    - (pt.x() - polygon[i].0) * (yj - yi);
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= pt.y() {
            let cross =
                (polygon[j].0 - polygon[i].0) * (pt.y() - yi) - (pt.x() - polygon[i].0) * (yj - yi);
            if cross < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

/// Snap-based fallback tessellation for non-planar faces.
pub(super) fn tessellate_nonplanar_snap(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    let mut face_mesh = super::tessellate(topo, face_id, deflection)?;

    // `tessellate()` already applies the `is_reversed` flip. The caller
    // `tessellate_face_with_shared_edges` will apply its own flip, so undo
    // the one from `tessellate()` to avoid a double-flip.
    if face_data.is_reversed() {
        let tri_count = face_mesh.indices.len() / 3;
        for t in 0..tri_count {
            face_mesh.indices.swap(t * 3 + 1, t * 3 + 2);
        }
        for n in &mut face_mesh.normals {
            *n = -*n;
        }
    }

    let mut local_to_global: Vec<u32> = Vec::with_capacity(face_mesh.positions.len());

    // Collect all edge points for this face to use as snap targets.
    let wire = topo.wire(face_data.outer_wire())?;
    let mut snap_targets: Vec<(Point3, u32)> = Vec::new();
    for oe in wire.edges() {
        if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
            for &gid in global_ids {
                if (gid as usize) < merged.positions.len() {
                    snap_targets.push((merged.positions[gid as usize], gid));
                }
            }
        }
    }
    for &inner_wire_id in face_data.inner_wires() {
        if let Ok(inner_wire) = topo.wire(inner_wire_id) {
            for oe in inner_wire.edges() {
                if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
                    for &gid in global_ids {
                        if (gid as usize) < merged.positions.len() {
                            snap_targets.push((merged.positions[gid as usize], gid));
                        }
                    }
                }
            }
        }
    }

    // Build spatial hash for O(1) snap lookups.
    let snap_tol = 1e-6;
    let inv_cell = 1.0 / snap_tol;
    let mut snap_grid: HashMap<(i64, i64, i64), Vec<u32>> =
        HashMap::with_capacity(snap_targets.len());
    for &(target_pos, gid) in &snap_targets {
        let cx = (target_pos.x() * inv_cell).round() as i64;
        let cy = (target_pos.y() * inv_cell).round() as i64;
        let cz = (target_pos.z() * inv_cell).round() as i64;
        snap_grid.entry((cx, cy, cz)).or_default().push(gid);
    }

    for (i, &pos) in face_mesh.positions.iter().enumerate() {
        let cx = (pos.x() * inv_cell).round() as i64;
        let cy = (pos.y() * inv_cell).round() as i64;
        let cz = (pos.z() * inv_cell).round() as i64;
        let mut best_gid = None;
        let mut best_dist = snap_tol;
        // Check 3x3x3 neighborhood for snap matches.
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    if let Some(gids) = snap_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &gid in gids {
                            let target_pos = merged.positions[gid as usize];
                            let dist = (pos - target_pos).length();
                            if dist < best_dist {
                                best_dist = dist;
                                best_gid = Some(gid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(gid) = best_gid {
            local_to_global.push(gid);
        } else {
            let key = point_merge_key(pos, MERGE_GRID);
            let gid = point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pos);
                merged.normals.push(
                    face_mesh
                        .normals
                        .get(i)
                        .copied()
                        .unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                );
                idx
            });
            local_to_global.push(*gid);
        }
    }

    for &li in &face_mesh.indices {
        merged.indices.push(local_to_global[li as usize]);
    }

    Ok(())
}
