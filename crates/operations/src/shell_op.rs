//! Shell (hollow/offset) operation for creating thin-walled solids.
//!
//! Offsets faces of a solid inward to create a hollow shell with
//! uniform wall thickness. Optionally removes specified faces to
//! create openings.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::{FaceSpec, assemble_solid_mixed};
use crate::dot_normal_point;

/// Compute the inner vertex position using miter-vector offset.
///
/// Given a vertex with normals from adjacent faces, solves for the offset
/// direction that satisfies `m · n_i = 1` for all non-open face normals
/// (open face normals contribute 0). The inner position is:
///   `inner = outer - thickness * m`
///
/// For 3 linearly independent normals, this is equivalent to 3-plane
/// intersection. For 2 normals, it produces the least-norm miter (the
/// shortest offset vector satisfying both constraints). For 1 normal,
/// it offsets along that normal.
fn compute_miter_offset(outer: Point3, unique_normals: &[(Vec3, bool)], thickness: f64) -> Point3 {
    // Build system: for each unique normal, m · n_i = weight_i
    // where weight_i = 1.0 for non-open faces, 0.0 for open faces.
    let mut normals: Vec<Vec3> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for &(n, is_open) in unique_normals {
        normals.push(n);
        weights.push(if is_open { 0.0 } else { 1.0 });
    }

    let miter = match normals.len() {
        0 => return outer,
        1 => {
            // Single normal: offset along it.
            normals[0] * weights[0]
        }
        2 => {
            // Two normals: least-norm solution of [n1; n2] · m = [w1; w2].
            // m = N^T (N N^T)^{-1} w
            let n1 = normals[0];
            let n2 = normals[1];
            let w1 = weights[0];
            let w2 = weights[1];

            let g11 = n1.dot(n1);
            let g12 = n1.dot(n2);
            let g22 = n2.dot(n2);
            let det = g11 * g22 - g12 * g12;

            if det.abs() < 1e-12 {
                // Nearly parallel normals: just use the first non-open one.
                if w1 > 0.5 { n1 * w1 } else { n2 * w2 }
            } else {
                let inv_det = 1.0 / det;
                let a1 = (g22 * w1 - g12 * w2) * inv_det;
                let a2 = (-g12 * w1 + g11 * w2) * inv_det;
                n1 * a1 + n2 * a2
            }
        }
        _ => {
            // Three or more normals: use the first 3 linearly independent
            // normals and solve via Cramer's rule (3-plane intersection).
            let n1 = normals[0];
            let n2 = normals[1];
            let n3 = normals[2];
            let w1 = weights[0];
            let w2 = weights[1];
            let w3 = weights[2];

            let n2_cross_n3 = n2.cross(n3);
            let det = n1.dot(n2_cross_n3);

            if det.abs() < 1e-12 {
                // Degenerate: fall back to 2-normal solution with first two.
                let g11 = n1.dot(n1);
                let g12 = n1.dot(n2);
                let g22 = n2.dot(n2);
                let d2 = g11 * g22 - g12 * g12;
                if d2.abs() < 1e-12 {
                    n1 * w1
                } else {
                    let inv = 1.0 / d2;
                    let a1 = (g22 * w1 - g12 * w2) * inv;
                    let a2 = (-g12 * w1 + g11 * w2) * inv;
                    n1 * a1 + n2 * a2
                }
            } else {
                let n3_cross_n1 = n3.cross(n1);
                let n1_cross_n2 = n1.cross(n2);
                let inv_det = 1.0 / det;
                let mx =
                    (w1 * n2_cross_n3.x() + w2 * n3_cross_n1.x() + w3 * n1_cross_n2.x()) * inv_det;
                let my =
                    (w1 * n2_cross_n3.y() + w2 * n3_cross_n1.y() + w3 * n1_cross_n2.y()) * inv_det;
                let mz =
                    (w1 * n2_cross_n3.z() + w2 * n3_cross_n1.z() + w3 * n1_cross_n2.z()) * inv_det;
                Vec3::new(mx, my, mz)
            }
        }
    };

    Point3::new(
        outer.x() - thickness * miter.x(),
        outer.y() - thickness * miter.y(),
        outer.z() - thickness * miter.z(),
    )
}

/// Create a hollow shell from a solid by offsetting faces inward.
///
/// Each face is offset inward by `thickness` along its outward normal.
/// Supports planar, NURBS, and analytic surface faces.
/// If `open_faces` is non-empty, those faces are removed from both the
/// outer and inner shells, creating openings.
///
/// # Errors
///
/// Returns an error if:
/// - `thickness` is non-positive
/// - Any face in `open_faces` is not part of the solid
/// - Face offset fails (e.g., negative radius for curved surfaces)
/// - The resulting shell is degenerate
#[allow(clippy::too_many_lines)]
pub fn shell(
    topo: &mut Topology,
    solid: SolidId,
    thickness: f64,
    open_faces: &[FaceId],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if thickness <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("shell thickness must be positive, got {thickness}"),
        });
    }

    let solid_data = topo.solid(solid)?;
    let shell_data = topo.shell(solid_data.outer_shell())?;
    let all_face_ids: Vec<FaceId> = shell_data.faces().to_vec();

    let open_set: HashSet<usize> = open_faces.iter().map(|f| f.index()).collect();

    let solid_face_set: HashSet<usize> = all_face_ids.iter().map(|f| f.index()).collect();
    for &of in open_faces {
        if !solid_face_set.contains(&of.index()) {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("face {} is not part of the solid", of.index()),
            });
        }
    }

    // Collect face vertex data (samples curved edges for proper polygons).
    let mut face_verts: Vec<(FaceId, Vec<Point3>)> = Vec::new();
    for &fid in &all_face_ids {
        let verts = crate::boolean::face_polygon(topo, fid)?;
        face_verts.push((fid, verts));
    }

    let mut result_specs: Vec<FaceSpec> = Vec::new();

    // ─── Phase 1: Build vertex→normals map using ALL face types ───────────
    //
    // For each vertex, collect the outward surface normals from ALL adjacent
    // faces (planar and non-planar). We use these to compute a miter vector
    // that gives the correct inner vertex position at the intersection of
    // all offset surfaces meeting at that vertex.
    let inv_tol = 1.0 / tol.linear;
    let quantize_pt = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * inv_tol).round() as i64,
            (p.y() * inv_tol).round() as i64,
            (p.z() * inv_tol).round() as i64,
        )
    };

    let mut vertex_normals: HashMap<(i64, i64, i64), Vec<(Vec3, bool)>> = HashMap::new();

    for &(fid, ref verts) in &face_verts {
        let face = topo.face(fid)?;
        let is_open = open_set.contains(&fid.index());

        for v in verts {
            let (u, v_param) = face.surface().project_point(*v).unwrap_or((0.0, 0.0));
            let mut normal = face.surface().normal(u, v_param);
            // Account for the face's reversal flag: when a face is reversed,
            // the native surface normal points in the wrong direction.
            if face.is_reversed() {
                normal = -normal;
            }
            vertex_normals
                .entry(quantize_pt(*v))
                .or_default()
                .push((normal, is_open));
        }
    }

    // ─── Phase 2: Compute inner vertex positions via miter vectors ────────
    //
    // The miter vector m at a vertex satisfies m · n_i = 1 for each unique
    // face normal n_i. The inner position is: inner = outer - thickness * m.
    // This correctly handles vertices where 2 or 3 offset surfaces intersect
    // (including non-planar surfaces like cylinders at tangent points).
    //
    // For open faces, the offset distance is 0 (the rim vertex stays on the
    // original plane), so we use n_i with a weight of 0 in that direction.
    let mut inner_pos: HashMap<(i64, i64, i64), Point3> = HashMap::new();

    for (&key, normals) in &vertex_normals {
        // Deduplicate nearly-parallel normals, keeping track of whether
        // each unique normal is offset (non-open) or stays (open).
        let mut unique: Vec<(Vec3, bool)> = Vec::new();
        for &(n, is_open) in normals {
            // Use cosine similarity to deduplicate nearly-parallel normals.
            // At tangent points (where a flat face meets a curved face),
            // normals can differ by small amounts that still cause near-singular
            // miter vectors if treated as independent.
            let dominated = unique.iter_mut().any(|(un, existing_open)| {
                let dot = un.dot(n);
                if dot > 0.995 {
                    // Nearly parallel — merge. Prefer the non-open (offset) variant.
                    if *existing_open && !is_open {
                        *un = n;
                        *existing_open = false;
                    }
                    true
                } else {
                    false
                }
            });
            if !dominated {
                unique.push((n, is_open));
            }
        }

        // Reconstruct the outer point from the quantized key.
        let outer_pt = Point3::new(
            key.0 as f64 / inv_tol,
            key.1 as f64 / inv_tol,
            key.2 as f64 / inv_tol,
        );

        // Build the miter offset: solve N · m = b where b_i = thickness
        // for non-open faces, 0 for open faces.
        let inner = compute_miter_offset(outer_pt, &unique, thickness);
        inner_pos.insert(key, inner);
    }

    // Outer faces: the non-open faces kept as-is.
    for &(fid, ref verts) in &face_verts {
        if open_set.contains(&fid.index()) {
            continue;
        }
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                result_specs.push(FaceSpec::Planar {
                    vertices: verts.clone(),
                    normal: *normal,
                    d: *d,
                    inner_wires: vec![],
                });
            }
            FaceSurface::Cylinder(cyl) => {
                // Use CylindricalFace to preserve arc edges (Circle EdgeCurve)
                // so that tessellation and volume computation remain accurate.
                let wire = topo.wire(face.outer_wire())?;
                let has_closed_edge = wire
                    .edges()
                    .iter()
                    .any(|oe| topo.edge(oe.edge()).is_ok_and(|e| e.start() == e.end()));
                if has_closed_edge {
                    result_specs.push(FaceSpec::Surface {
                        vertices: verts.clone(),
                        surface: FaceSurface::Cylinder(cyl.clone()),
                        reversed: false,
                        inner_wires: vec![],
                    });
                } else {
                    result_specs.push(FaceSpec::CylindricalFace {
                        vertices: verts.clone(),
                        cylinder: cyl.clone(),
                        reversed: false,
                        inner_wires: vec![],
                    });
                }
            }
            other => {
                result_specs.push(FaceSpec::Surface {
                    vertices: verts.clone(),
                    surface: other.clone(),
                    reversed: false,
                    inner_wires: vec![],
                });
            }
        }
    }

    // ─── Phase 4: Inner faces (offset of non-open faces) ──────────────────
    //
    // All inner vertex positions come from the miter vector computation in
    // Phase 2. This ensures watertight geometry at ALL junctions, including
    // where planar faces meet cylindrical faces at tangent points.

    for &(fid, ref outer_verts) in &face_verts {
        if open_set.contains(&fid.index()) {
            continue;
        }
        let face = topo.face(fid)?;

        // Reversed winding gives the inner face an inward-pointing normal.
        let inner_verts: Vec<Point3> = outer_verts
            .iter()
            .map(|v| inner_pos.get(&quantize_pt(*v)).copied().unwrap_or(*v))
            .rev()
            .collect();

        match face.surface() {
            FaceSurface::Plane { normal, .. } => {
                let inner_normal = -*normal;
                let inner_d = dot_normal_point(inner_normal, inner_verts[0]);
                result_specs.push(FaceSpec::Planar {
                    vertices: inner_verts,
                    normal: inner_normal,
                    d: inner_d,
                    inner_wires: vec![],
                });
            }
            FaceSurface::Cylinder(cyl) => {
                let new_radius = cyl.radius() - thickness;
                if new_radius > tol.linear
                    && let Ok(new_cyl) = brepkit_math::surfaces::CylindricalSurface::new(
                        cyl.origin(),
                        cyl.axis(),
                        new_radius,
                    )
                {
                    // Full-circle cylinders: use Surface (the dense sample
                    // polygon from face_polygon contains seam-duplicate
                    // vertices that CylindricalFace can't handle cleanly).
                    // Partial-arc cylinders: use CylindricalFace to create
                    // Circle edges that preserve angular range info.
                    let wire = topo.wire(face.outer_wire())?;
                    let has_closed_edge = wire
                        .edges()
                        .iter()
                        .any(|oe| topo.edge(oe.edge()).is_ok_and(|e| e.start() == e.end()));
                    if has_closed_edge {
                        result_specs.push(FaceSpec::Surface {
                            vertices: inner_verts,
                            surface: FaceSurface::Cylinder(new_cyl),
                            reversed: true,
                            inner_wires: vec![],
                        });
                    } else {
                        result_specs.push(FaceSpec::CylindricalFace {
                            vertices: inner_verts,
                            cylinder: new_cyl,
                            reversed: true,
                            inner_wires: vec![],
                        });
                    }
                }
            }
            FaceSurface::Cone(_cone) => {
                let inner_fid = crate::offset_face::offset_face(topo, fid, -thickness, 8)?;
                let inner_face = topo.face(inner_fid)?;
                result_specs.push(FaceSpec::Surface {
                    vertices: inner_verts,
                    surface: inner_face.surface().clone(),
                    reversed: true,
                    inner_wires: vec![],
                });
            }
            FaceSurface::Sphere(sphere) => {
                let new_r = sphere.radius() - thickness;
                if new_r <= 0.0 {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: format!(
                            "shell thickness ({thickness}) exceeds sphere radius ({}), \
                             resulting inner sphere would have non-positive radius ({new_r})",
                            sphere.radius(),
                        ),
                    });
                }
                let new_sph = brepkit_math::surfaces::SphericalSurface::new(sphere.center(), new_r)
                    .map_err(crate::OperationsError::Math)?;
                result_specs.push(FaceSpec::Surface {
                    vertices: inner_verts,
                    surface: FaceSurface::Sphere(new_sph),
                    reversed: true,
                    inner_wires: vec![],
                });
            }
            FaceSurface::Nurbs(_) | FaceSurface::Torus(_) => {
                let inner_fid = crate::offset_face::offset_face(topo, fid, -thickness, 8)?;
                let inner_face = topo.face(inner_fid)?;
                result_specs.push(FaceSpec::Surface {
                    vertices: inner_verts,
                    surface: inner_face.surface().clone(),
                    reversed: true,
                    inner_wires: vec![],
                });
            }
        }
    }

    // ─── Phase 5: Assemble outer + inner faces, then close rim ─────────────
    //
    // Instead of creating disconnected rim quads (which don't share edges
    // with the outer/inner faces), we first assemble the outer + inner faces
    // into a solid with open boundaries, then find the boundary edges and
    // create a single annular rim face per open face. This guarantees edge
    // sharing and produces a manifold shell.

    if result_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "shell operation produced no faces".into(),
        });
    }

    let solid = assemble_solid_mixed(topo, &result_specs, tol)?;

    let edge_face_map = brepkit_topology::explorer::edge_to_face_map(topo, solid)?;
    let mut boundary_edge_ids: Vec<brepkit_topology::edge::EdgeId> = Vec::new();
    for (&edge_idx, faces) in &edge_face_map {
        if faces.len() == 1
            && let Some(eid) = topo.edge_id_from_index(edge_idx)
        {
            boundary_edge_ids.push(eid);
        }
    }

    if boundary_edge_ids.is_empty() {
        // No open boundary — shell is already closed (no open faces, or all faces present).
        return Ok(solid);
    }

    // Determine the oriented direction of each boundary edge relative to its single face.
    // The rim face must use the OPPOSITE orientation so the edge is shared correctly.
    let mut boundary_oriented: Vec<brepkit_topology::wire::OrientedEdge> = Vec::new();
    for &eid in &boundary_edge_ids {
        let face_id = edge_face_map[&eid.index()][0];
        let face = topo.face(face_id)?;
        let wire = topo.wire(face.outer_wire())?;
        let mut found = false;
        for oe in wire.edges() {
            if oe.edge() == eid {
                boundary_oriented.push(brepkit_topology::wire::OrientedEdge::new(
                    eid,
                    !oe.is_forward(),
                ));
                found = true;
                break;
            }
        }
        if !found {
            for &iw_id in face.inner_wires() {
                let iw = topo.wire(iw_id)?;
                for oe in iw.edges() {
                    if oe.edge() == eid {
                        boundary_oriented.push(brepkit_topology::wire::OrientedEdge::new(
                            eid,
                            !oe.is_forward(),
                        ));
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                // Fallback: use forward orientation.
                boundary_oriented.push(brepkit_topology::wire::OrientedEdge::new(eid, true));
            }
        }
    }

    let loops = sort_edges_into_loops(topo, &boundary_oriented)?;

    if loops.len() < 2 {
        // Need at least 2 loops (outer + inner) for an annular face.
        // If only 1 loop, something is wrong — return the solid as-is.
        return Ok(solid);
    }

    // Classify loops: the outer loop has larger average distance from centroid.
    let mut centroid = Vec3::new(0.0, 0.0, 0.0);
    let mut vert_count = 0.0;
    let mut rim_z = 0.0_f64;
    for oe in &boundary_oriented {
        let edge = topo.edge(oe.edge())?;
        let p = topo.vertex(edge.start())?.point();
        centroid += Vec3::new(p.x(), p.y(), p.z());
        rim_z += p.z();
        vert_count += 1.0;
    }
    if vert_count > 0.0 {
        centroid = centroid * (1.0 / vert_count);
        rim_z /= vert_count;
    }

    let mut loop_radii: Vec<(usize, f64)> = Vec::new();
    for (i, lp) in loops.iter().enumerate() {
        let mut avg_r = 0.0;
        let mut n = 0.0;
        for oe in lp {
            let edge = topo.edge(oe.edge())?;
            let p = topo.vertex(edge.start())?.point();
            let dx = p.x() - centroid.x();
            let dy = p.y() - centroid.y();
            avg_r += (dx * dx + dy * dy).sqrt();
            n += 1.0;
        }
        if n > 0.0 {
            avg_r /= n;
        }
        loop_radii.push((i, avg_r));
    }
    loop_radii.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Largest loop is the outer wire, all others are inner wires (holes).
    let outer_loop_idx = loop_radii[0].0;

    let outer_wire = brepkit_topology::wire::Wire::new(loops[outer_loop_idx].clone(), true)
        .map_err(crate::OperationsError::Topology)?;
    let outer_wire_id = topo.add_wire(outer_wire);

    let mut inner_wire_ids = Vec::new();
    for &(idx, _) in &loop_radii[1..] {
        let inner_wire = brepkit_topology::wire::Wire::new(loops[idx].clone(), true)
            .map_err(crate::OperationsError::Topology)?;
        inner_wire_ids.push(topo.add_wire(inner_wire));
    }

    // Rim face normal: pointing away from solid center (outward at the rim).
    // For a top-opened shell, this is typically +Z or -Z.
    // Compute from the open face's normal.
    let rim_normal = {
        let mut n = Vec3::new(0.0, 0.0, 1.0);
        for &(fid, _) in &face_verts {
            if open_set.contains(&fid.index())
                && let Ok(f) = topo.face(fid)
                && let FaceSurface::Plane { normal, .. } = f.surface()
            {
                // The rim normal should point in the same direction as the
                // removed face's outward normal (away from solid interior).
                n = if f.is_reversed() { -*normal } else { *normal };
                break;
            }
        }
        n
    };

    let rim_d =
        rim_normal.x() * centroid.x() + rim_normal.y() * centroid.y() + rim_normal.z() * rim_z;
    let rim_face = brepkit_topology::face::Face::new(
        outer_wire_id,
        inner_wire_ids,
        FaceSurface::Plane {
            normal: rim_normal,
            d: rim_d,
        },
    );
    let rim_face_id = topo.add_face(rim_face);

    let solid_data = topo.solid(solid)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let mut new_faces: Vec<FaceId> = shell.faces().to_vec();
    new_faces.push(rim_face_id);
    let new_shell =
        brepkit_topology::shell::Shell::new(new_faces).map_err(crate::OperationsError::Topology)?;
    *topo.shell_mut(shell_id)? = new_shell;

    Ok(solid)
}

/// Sort oriented edges into connected loops.
///
/// Takes a set of oriented boundary edges and groups them into closed loops
/// by following edge connectivity (end vertex → start vertex of next edge).
fn sort_edges_into_loops(
    topo: &Topology,
    edges: &[brepkit_topology::wire::OrientedEdge],
) -> Result<Vec<Vec<brepkit_topology::wire::OrientedEdge>>, crate::OperationsError> {
    use brepkit_topology::vertex::VertexId;

    if edges.is_empty() {
        return Ok(Vec::new());
    }

    let mut start_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut edge_endpoints: Vec<(VertexId, VertexId)> = Vec::new();
    for (i, oe) in edges.iter().enumerate() {
        let edge = topo.edge(oe.edge())?;
        let (sv, ev) = if oe.is_forward() {
            (edge.start(), edge.end())
        } else {
            (edge.end(), edge.start())
        };
        start_map.entry(sv.index()).or_default().push(i);
        edge_endpoints.push((sv, ev));
    }

    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();

    while let Some(start_idx) = used.iter().position(|&u| !u) {
        let mut current_loop = Vec::new();
        let mut current = start_idx;
        let chain_start_vid = edge_endpoints[current].0.index();

        loop {
            if used[current] {
                break;
            }
            used[current] = true;
            current_loop.push(edges[current]);
            let end_vid = edge_endpoints[current].1.index();

            if end_vid == chain_start_vid {
                break; // Loop closed.
            }

            let mut found = false;
            if let Some(candidates) = start_map.get(&end_vid) {
                for &idx in candidates {
                    if !used[idx] {
                        current = idx;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break; // Broken chain — give up on this loop.
            }
        }

        if !current_loop.is_empty() {
            loops.push(current_loop);
        }
    }

    Ok(loops)
}

#[cfg(test)]
mod tests;
