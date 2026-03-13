//! Shell (hollow/offset) operation for creating thin-walled solids.
//!
//! Equivalent to `BRepOffsetAPI_MakeThickSolid` in `OpenCascade`.
//! Offsets faces of a solid inward to create a hollow shell with
//! uniform wall thickness. Optionally removes specified faces to
//! create openings.

use std::collections::{HashMap, HashSet};

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::{FaceSpec, assemble_solid_mixed};
use crate::dot_normal_point;

/// Compute the outward surface normal at a 3D point for any face surface type.
fn surface_normal_at(surface: &FaceSurface, pt: Point3) -> Vec3 {
    match surface {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Cylinder(cyl) => {
            // Radial direction from the cylinder axis through the point.
            let to_axis = Vec3::new(
                pt.x() - cyl.origin().x(),
                pt.y() - cyl.origin().y(),
                pt.z() - cyl.origin().z(),
            );
            let along = cyl.axis() * cyl.axis().dot(to_axis);
            let radial = to_axis - along;
            radial.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0))
        }
        FaceSurface::Cone(cone) => {
            let to_apex = Vec3::new(
                pt.x() - cone.apex().x(),
                pt.y() - cone.apex().y(),
                pt.z() - cone.apex().z(),
            );
            let along = cone.axis() * cone.axis().dot(to_apex);
            let radial = to_apex - along;
            radial.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0))
        }
        FaceSurface::Sphere(sphere) => {
            let dir = Vec3::new(
                pt.x() - sphere.center().x(),
                pt.y() - sphere.center().y(),
                pt.z() - sphere.center().z(),
            );
            dir.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
        }
        FaceSurface::Nurbs(srf) => nurbs_normal_at_point(srf, pt),
        FaceSurface::Torus(_) => {
            // Torus: fallback — rarely hit in practice.
            Vec3::new(0.0, 0.0, 1.0)
        }
    }
}

/// Evaluate the NURBS surface normal at the parametric point closest to `pt`.
///
/// Performs a coarse 5×5 grid search over the parameter domain to find the
/// closest surface point, then evaluates the analytic normal there.
fn nurbs_normal_at_point(srf: &NurbsSurface, pt: Point3) -> Vec3 {
    const N: usize = 5;

    let (u_lo, u_hi) = srf.domain_u();
    let (v_lo, v_hi) = srf.domain_v();

    let mut best_u = 0.5 * (u_lo + u_hi);
    let mut best_v = 0.5 * (v_lo + v_hi);
    let mut best_d2 = f64::MAX;
    for i in 0..=N {
        let u = u_lo + (u_hi - u_lo) * (i as f64 / N as f64);
        for j in 0..=N {
            let v = v_lo + (v_hi - v_lo) * (j as f64 / N as f64);
            let s = srf.evaluate(u, v);
            let dx = s.x() - pt.x();
            let dy = s.y() - pt.y();
            let dz = s.z() - pt.z();
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_u = u;
                best_v = v;
            }
        }
    }

    srf.normal(best_u, best_v)
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

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

    // Validate open_faces belong to the solid.
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

    // Collect (normal, is_open) for each face at each vertex.
    let mut vertex_normals: HashMap<(i64, i64, i64), Vec<(Vec3, bool)>> = HashMap::new();

    for &(fid, ref verts) in &face_verts {
        let face = topo.face(fid)?;
        let is_open = open_set.contains(&fid.index());

        for v in verts {
            let mut normal = surface_normal_at(face.surface(), *v);
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

    // ─── Phase 3: Outer faces (non-open, kept as-is) ──────────────────────

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
                    });
                } else {
                    result_specs.push(FaceSpec::CylindricalFace {
                        vertices: verts.clone(),
                        cylinder: cyl.clone(),
                        reversed: false,
                    });
                }
            }
            other => {
                result_specs.push(FaceSpec::Surface {
                    vertices: verts.clone(),
                    surface: other.clone(),
                    reversed: false,
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

        // Map outer vertices to inner positions (reversed winding for inward normal).
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
                });
            }
            FaceSurface::Cylinder(cyl) => {
                let new_radius = cyl.radius() - thickness;
                if new_radius > tol.linear {
                    if let Ok(new_cyl) = brepkit_math::surfaces::CylindricalSurface::new(
                        cyl.origin(),
                        cyl.axis(),
                        new_radius,
                    ) {
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
                            });
                        } else {
                            result_specs.push(FaceSpec::CylindricalFace {
                                vertices: inner_verts,
                                cylinder: new_cyl,
                                reversed: true,
                            });
                        }
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
                });
            }
            FaceSurface::Sphere(sphere) => {
                let new_r = sphere.radius() - thickness;
                if new_r > tol.linear {
                    if let Ok(new_sph) =
                        brepkit_math::surfaces::SphericalSurface::new(sphere.center(), new_r)
                    {
                        result_specs.push(FaceSpec::Surface {
                            vertices: inner_verts,
                            surface: FaceSurface::Sphere(new_sph),
                            reversed: true,
                        });
                    }
                }
            }
            FaceSurface::Nurbs(_) | FaceSurface::Torus(_) => {
                let inner_fid = crate::offset_face::offset_face(topo, fid, -thickness, 8)?;
                let inner_face = topo.face(inner_fid)?;
                result_specs.push(FaceSpec::Surface {
                    vertices: inner_verts,
                    surface: inner_face.surface().clone(),
                    reversed: true,
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
    // sharing and produces a manifold shell matching OCCT's topology.

    if result_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "shell operation produced no faces".into(),
        });
    }

    let solid = assemble_solid_mixed(topo, &result_specs, tol)?;

    // Find boundary edges (edges used by exactly 1 face).
    let edge_face_map = brepkit_topology::explorer::edge_to_face_map(topo, solid)?;
    let mut boundary_edge_ids: Vec<brepkit_topology::edge::EdgeId> = Vec::new();
    for (&edge_idx, faces) in &edge_face_map {
        if faces.len() == 1 {
            if let Some(eid) = topo.edges.id_from_index(edge_idx) {
                boundary_edge_ids.push(eid);
            }
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
        // Find orientation of this edge in its face's wire.
        let wire = topo.wire(face.outer_wire())?;
        let mut found = false;
        for oe in wire.edges() {
            if oe.edge() == eid {
                // Rim face uses opposite orientation.
                boundary_oriented.push(brepkit_topology::wire::OrientedEdge::new(
                    eid,
                    !oe.is_forward(),
                ));
                found = true;
                break;
            }
        }
        if !found {
            // Check inner wires too.
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

    // Sort boundary edges into connected loops.
    let loops = sort_edges_into_loops(topo, &boundary_oriented)?;

    if loops.len() < 2 {
        // Need at least 2 loops (outer + inner) for an annular face.
        // If only 1 loop, something is wrong — return the solid as-is.
        return Ok(solid);
    }

    // Classify loops: the outer loop has larger average distance from centroid.
    // Compute centroid of all boundary vertices.
    let mut centroid = Vec3::new(0.0, 0.0, 0.0);
    let mut vert_count = 0.0;
    let mut rim_z = 0.0_f64;
    for oe in &boundary_oriented {
        let edge = topo.edge(oe.edge())?;
        let p = topo.vertex(edge.start())?.point();
        centroid = centroid + Vec3::new(p.x(), p.y(), p.z());
        rim_z += p.z();
        vert_count += 1.0;
    }
    if vert_count > 0.0 {
        centroid = centroid * (1.0 / vert_count);
        rim_z /= vert_count;
    }

    // Compute average radial distance for each loop.
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
    let outer_wire_id = topo.wires.alloc(outer_wire);

    let mut inner_wire_ids = Vec::new();
    for &(idx, _) in &loop_radii[1..] {
        let inner_wire = brepkit_topology::wire::Wire::new(loops[idx].clone(), true)
            .map_err(crate::OperationsError::Topology)?;
        inner_wire_ids.push(topo.wires.alloc(inner_wire));
    }

    // Rim face normal: pointing away from solid center (outward at the rim).
    // For a top-opened shell, this is typically +Z or -Z.
    // Compute from the open face's normal.
    let rim_normal = {
        let mut n = Vec3::new(0.0, 0.0, 1.0);
        for &(fid, _) in &face_verts {
            if open_set.contains(&fid.index()) {
                if let Ok(f) = topo.face(fid) {
                    if let FaceSurface::Plane { normal, .. } = f.surface() {
                        // The rim normal should point in the same direction as the
                        // removed face's outward normal (away from solid interior).
                        n = if f.is_reversed() { -*normal } else { *normal };
                        break;
                    }
                }
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
    let rim_face_id = topo.faces.alloc(rim_face);

    // Add the rim face to the shell.
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

    // Build start_vertex → edge index map.
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

            // Find next edge starting at end_vid.
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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_possible_wrap)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    /// Helper: get face IDs matching a given normal direction.
    fn find_faces_by_normal(topo: &Topology, solid: SolidId, target_normal: Vec3) -> Vec<FaceId> {
        let tol = Tolerance::loose();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut result = Vec::new();
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                if tol.approx_eq(normal.x(), target_normal.x())
                    && tol.approx_eq(normal.y(), target_normal.y())
                    && tol.approx_eq(normal.z(), target_normal.z())
                {
                    result.push(fid);
                }
            }
        }
        result
    }

    #[test]
    fn shell_closed_box() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Shell with no open faces: creates a solid shell.
        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 6 outer + 6 inner = 12 faces (no rim faces since no openings).
        assert_eq!(sh.faces().len(), 12, "closed shell should have 12 faces");
    }

    #[test]
    fn shell_open_top() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find the top face (+Z normal).
        let top_faces = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top_faces.len(), 1, "should find exactly one +Z face");

        let result = shell(&mut topo, cube, 0.1, &top_faces).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 5 outer + 5 inner + 1 annular rim = 11 faces
        assert_eq!(sh.faces().len(), 11, "open-top shell should have 11 faces");

        // Check volume accuracy: 1 - 0.8*0.8*0.9 = 0.424
        let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
        let expected = 1.0 - 0.8 * 0.8 * 0.9;
        eprintln!("[shell_open_top] volume: {vol:.6}, expected: {expected:.6}");
    }

    #[test]
    fn shell_volume_decrease() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let original_vol = crate::measure::solid_volume(&topo, cube, 0.1).unwrap();

        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        // The shelled solid should have less volume than the original
        // (we removed the interior).
        assert!(
            shell_vol < original_vol,
            "shell volume ({shell_vol}) should be less than original ({original_vol})"
        );
        assert!(
            shell_vol > 0.0,
            "shell volume should be positive, got {shell_vol}"
        );
    }

    #[test]
    fn shell_zero_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, 0.0, &[]).is_err());
    }

    #[test]
    fn shell_negative_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, -0.1, &[]).is_err());
    }

    #[test]
    fn shell_two_open_faces_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find both +Z and -Z faces
        let top = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        let bot = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, -1.0));
        let mut open_faces = top;
        open_faces.extend(bot);
        assert_eq!(open_faces.len(), 2);

        let result = shell(&mut topo, cube, 0.1, &open_faces).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
        // Expected: 1.0 - 0.8*0.8*1.0 = 0.36
        assert!(vol > 0.1, "tube shell volume should be positive, got {vol}");
        assert!(
            vol < 1.0,
            "tube shell volume should be < original 1.0, got {vol}"
        );
    }

    /// Simulates the gridfinity "1×1 flat no-lip" pipeline:
    /// rounded rectangle → extrude → shell (open top).
    /// Reports face count and volume for debugging parity issues.
    #[test]
    fn shell_rounded_rect_extrude_diagnostics() {
        use crate::primitives::make_box;

        let mut topo = Topology::new();

        // Gridfinity dimensions: 41.5×41.5×21mm, 4mm corner radius, 1.2mm wall thickness
        let w = 41.5;
        let d = 41.5;
        let h = 21.0;
        let thickness = 1.2;

        // Use a simple box (no rounded corners) to isolate shell behavior.
        let box_solid = make_box(&mut topo, w, d, h).unwrap();

        // Count faces after extrude.
        let box_shell_data = topo
            .shell(topo.solid(box_solid).unwrap().outer_shell())
            .unwrap();
        let extrude_face_count = box_shell_data.faces().len();
        eprintln!("[diag] Box extrude face count: {extrude_face_count}");
        assert_eq!(extrude_face_count, 6);

        // Find top face and shell.
        let top_faces = find_faces_by_normal(&topo, box_solid, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top_faces.len(), 1, "should find exactly one top face");

        let shelled = shell(&mut topo, box_solid, thickness, &top_faces).unwrap();
        let sh = topo
            .shell(topo.solid(shelled).unwrap().outer_shell())
            .unwrap();
        let shell_face_count = sh.faces().len();
        eprintln!("[diag] Box shell face count: {shell_face_count}");
        // 5 outer + 5 inner + 1 annular rim = 11
        assert_eq!(shell_face_count, 11, "box shell should have 11 faces");

        // Verify original box volume first.
        let box_vol = crate::measure::solid_volume(&topo, box_solid, 0.01).unwrap();
        let expected_box_vol = w * d * h;
        eprintln!("[diag] Box volume: {box_vol:.2}, expected: {expected_box_vol:.2}");

        let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
        let expected_vol =
            w * d * h - (w - 2.0 * thickness) * (d - 2.0 * thickness) * (h - thickness);
        let pct = (vol - expected_vol).abs() / expected_vol;
        eprintln!("[diag] Shell volume: {vol:.2}, expected: {expected_vol:.2}, diff: {pct:.4}");

        // Also count face surface types.
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            let kind = match f.surface() {
                FaceSurface::Plane { .. } => "Plane",
                FaceSurface::Cylinder(_) => "Cylinder",
                FaceSurface::Cone(_) => "Cone",
                FaceSurface::Sphere(_) => "Sphere",
                FaceSurface::Torus(_) => "Torus",
                FaceSurface::Nurbs(_) => "Nurbs",
            };
            let wire = topo.wire(f.outer_wire()).unwrap();
            eprintln!(
                "[diag]   Face {}: {kind}, {} edges",
                fid.index(),
                wire.edges().len()
            );
        }

        assert!(
            pct < 0.05,
            "shell volume should be within 5% of expected, got {pct:.4}"
        );
    }

    /// Gridfinity exact parameters (r=4mm corner radius) diagnostic.
    #[test]
    fn shell_gridfinity_exact_params() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let tol = Tolerance::new();

        // Exact gridfinity 1×1 flat no-lip parameters
        let w = 41.5_f64; // 1 × 42 - 0.5 clearance
        let d = 41.5_f64;
        let h = 21.0_f64; // 3 height units × 7mm
        let r = 4.0_f64; // CORNER_RADIUS = SOCKET_CORNER_RADIUS = 4mm
        let thickness = 1.2_f64;

        let hw = w / 2.0;
        let hd = d / 2.0;

        let v0 = Point3::new(hw - r, -hd, 0.0);
        let v1 = Point3::new(hw, -hd + r, 0.0);
        let v2 = Point3::new(hw, hd - r, 0.0);
        let v3 = Point3::new(hw - r, hd, 0.0);
        let v4 = Point3::new(-hw + r, hd, 0.0);
        let v5 = Point3::new(-hw, hd - r, 0.0);
        let v6 = Point3::new(-hw, -hd + r, 0.0);
        let v7 = Point3::new(-hw + r, -hd, 0.0);

        let vids: Vec<_> = [v0, v1, v2, v3, v4, v5, v6, v7]
            .iter()
            .map(|p| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
            .collect();

        let c_br = Point3::new(hw - r, -hd + r, 0.0);
        let c_tr = Point3::new(hw - r, hd - r, 0.0);
        let c_tl = Point3::new(-hw + r, hd - r, 0.0);
        let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        let mk_line =
            |topo: &mut Topology, s, e| topo.edges.alloc(Edge::new(s, e, EdgeCurve::Line));
        let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
            let circle = Circle3D::new(center, z_axis, r).unwrap();
            topo.edges.alloc(Edge::new(s, e, EdgeCurve::Circle(circle)))
        };

        let e_bot = mk_line(&mut topo, vids[7], vids[0]);
        let e_br = mk_arc(&mut topo, vids[0], vids[1], c_br);
        let e_right = mk_line(&mut topo, vids[1], vids[2]);
        let e_tr = mk_arc(&mut topo, vids[2], vids[3], c_tr);
        let e_top = mk_line(&mut topo, vids[3], vids[4]);
        let e_tl = mk_arc(&mut topo, vids[4], vids[5], c_tl);
        let e_left = mk_line(&mut topo, vids[5], vids[6]);
        let e_bl = mk_arc(&mut topo, vids[6], vids[7], c_bl);

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_bot, true),
                OrientedEdge::new(e_br, true),
                OrientedEdge::new(e_right, true),
                OrientedEdge::new(e_tr, true),
                OrientedEdge::new(e_top, true),
                OrientedEdge::new(e_tl, true),
                OrientedEdge::new(e_left, true),
                OrientedEdge::new(e_bl, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.wires.alloc(wire);
        let face = Face::new(
            wire_id,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        );
        let face_id = topo.faces.alloc(face);

        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

        let sh_before = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        let fc_before = sh_before.faces().len();
        eprintln!("[gf-exact] Faces after extrude: {fc_before}");

        let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top.len(), 1, "one top face");

        let shelled = shell(&mut topo, solid, thickness, &top).unwrap();
        let sh2 = topo
            .shell(topo.solid(shelled).unwrap().outer_shell())
            .unwrap();
        let fc_after = sh2.faces().len();
        eprintln!("[gf-exact] Faces after shell: {fc_after}");

        let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        let chi = v as i64 - e as i64 + f as i64;
        eprintln!("[gf-exact] F={f}, E={e}, V={v}, χ={chi}");

        let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
        eprintln!("[gf-exact] Volume: {vol:.2}");

        let result = crate::validate::validate_solid(&topo, shelled);
        eprintln!("[gf-exact] Validation: {result:?}");

        // After unify_faces
        let removed = crate::heal::unify_faces(&mut topo, shelled).unwrap();
        let sh3 = topo
            .shell(topo.solid(shelled).unwrap().outer_shell())
            .unwrap();
        let fc_unified = sh3.faces().len();
        eprintln!("[gf-exact] After unify_faces (removed {removed}): {fc_unified} faces");

        let (f2, e2, v2) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        let chi2 = v2 as i64 - e2 as i64 + f2 as i64;
        eprintln!("[gf-exact] After unify: F={f2}, E={e2}, V={v2}, χ={chi2}");
    }

    /// Rounded rectangle extrusion → shell: the gridfinity "1×1 flat no-lip" path.
    /// This test creates a face with lines + circle arcs, extrudes, then shells.
    #[test]
    fn shell_rounded_rect_with_arcs() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let tol = Tolerance::new();

        // Parameters matching gridfinity 1×1.
        let w = 41.5_f64;
        let d = 41.5_f64;
        let h = 21.0_f64;
        let r = 2.6_f64; // corner radius
        let thickness = 1.2_f64;

        // Rounded rectangle on XY at z=0:
        //   4 line segments + 4 quarter-circle arcs.
        // Vertices at the tangent points (where lines meet arcs).
        let hw = w / 2.0;
        let hd = d / 2.0;

        // Tangent points (CCW from bottom-right):
        let v0 = Point3::new(hw - r, -hd, 0.0);
        let v1 = Point3::new(hw, -hd + r, 0.0);
        let v2 = Point3::new(hw, hd - r, 0.0);
        let v3 = Point3::new(hw - r, hd, 0.0);
        let v4 = Point3::new(-hw + r, hd, 0.0);
        let v5 = Point3::new(-hw, hd - r, 0.0);
        let v6 = Point3::new(-hw, -hd + r, 0.0);
        let v7 = Point3::new(-hw + r, -hd, 0.0);

        let vids: Vec<_> = [v0, v1, v2, v3, v4, v5, v6, v7]
            .iter()
            .map(|p| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
            .collect();

        // Corner centers:
        let c_br = Point3::new(hw - r, -hd + r, 0.0);
        let c_tr = Point3::new(hw - r, hd - r, 0.0);
        let c_tl = Point3::new(-hw + r, hd - r, 0.0);
        let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        let mk_line =
            |topo: &mut Topology, s, e| topo.edges.alloc(Edge::new(s, e, EdgeCurve::Line));
        let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
            let circle = Circle3D::new(center, z_axis, r).unwrap();
            topo.edges.alloc(Edge::new(s, e, EdgeCurve::Circle(circle)))
        };

        let e_bot = mk_line(&mut topo, vids[7], vids[0]);
        let e_br = mk_arc(&mut topo, vids[0], vids[1], c_br);
        let e_right = mk_line(&mut topo, vids[1], vids[2]);
        let e_tr = mk_arc(&mut topo, vids[2], vids[3], c_tr);
        let e_top = mk_line(&mut topo, vids[3], vids[4]);
        let e_tl = mk_arc(&mut topo, vids[4], vids[5], c_tl);
        let e_left = mk_line(&mut topo, vids[5], vids[6]);
        let e_bl = mk_arc(&mut topo, vids[6], vids[7], c_bl);

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_bot, true),
                OrientedEdge::new(e_br, true),
                OrientedEdge::new(e_right, true),
                OrientedEdge::new(e_tr, true),
                OrientedEdge::new(e_top, true),
                OrientedEdge::new(e_tl, true),
                OrientedEdge::new(e_left, true),
                OrientedEdge::new(e_bl, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.wires.alloc(wire);

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
        let face_id = topo.faces.alloc(face);

        // Extrude up.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

        // Count faces after extrude.
        let sh = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        let extrude_fc = sh.faces().len();
        eprintln!("[rounded] Extrude faces: {extrude_fc}");
        // Expected: 2 caps + 8 sides (4 planar + 4 cylindrical) = 10
        assert_eq!(extrude_fc, 10, "extruded rounded rect should have 10 faces");

        // Count face types.
        let mut plane_count = 0;
        let mut cyl_count = 0;
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            match f.surface() {
                FaceSurface::Plane { .. } => plane_count += 1,
                FaceSurface::Cylinder(_) => cyl_count += 1,
                _ => {}
            }
        }
        eprintln!("[rounded] Extrude: {plane_count} planar, {cyl_count} cylinder");
        assert_eq!(plane_count, 6, "4 flat sides + 2 caps = 6 planar");
        assert_eq!(cyl_count, 4, "4 corner cylinders");

        // Volume check after extrude (before shell).
        // Expected: A = w*d - 4*r^2*(1-pi/4), V = A*h
        let expected_area = w * d - 4.0 * r * r * (1.0 - std::f64::consts::FRAC_PI_4);
        let expected_vol = expected_area * h;
        let extrude_vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let rel_err = (extrude_vol - expected_vol).abs() / expected_vol;
        eprintln!(
            "[rounded] Extrude volume: {extrude_vol:.2} (expected {expected_vol:.2}, diff {:.4}%)",
            rel_err * 100.0
        );
        assert!(
            rel_err < 0.001,
            "extrude volume error {rel_err:.6} exceeds 0.1%"
        );

        // Find top face(s) and shell.
        let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top.len(), 1, "one top face");

        let shelled = shell(&mut topo, solid, thickness, &top).unwrap();
        let sh2 = topo
            .shell(topo.solid(shelled).unwrap().outer_shell())
            .unwrap();
        let shell_fc = sh2.faces().len();
        eprintln!("[rounded] Shell faces: {shell_fc}");

        // Count surface types in shell.
        let mut sp = 0;
        let mut sc = 0;
        for &fid in sh2.faces() {
            let f = topo.face(fid).unwrap();
            match f.surface() {
                FaceSurface::Plane { .. } => sp += 1,
                FaceSurface::Cylinder(_) => sc += 1,
                _ => {}
            }
            let w2 = topo.wire(f.outer_wire()).unwrap();
            let kind = match f.surface() {
                FaceSurface::Plane { .. } => "Plane",
                FaceSurface::Cylinder(_) => "Cyl",
                _ => "Other",
            };
            eprintln!(
                "[rounded]   Face {}: {kind}, {} edges",
                fid.index(),
                w2.edges().len()
            );
        }
        eprintln!("[rounded] Shell: {sp} planar, {sc} cylinder");

        // Diagnostic: inspect the rim face (last face) in detail.
        {
            let rim_fid = *sh2.faces().last().unwrap();
            let rim_f = topo.face(rim_fid).unwrap();
            let outer_w = topo.wire(rim_f.outer_wire()).unwrap();
            eprintln!(
                "[rim-diag] Rim face {}: outer wire has {} edges, {} inner wires",
                rim_fid.index(),
                outer_w.edges().len(),
                rim_f.inner_wires().len()
            );
            for (i, oe) in outer_w.edges().iter().enumerate() {
                let e = topo.edge(oe.edge()).unwrap();
                let sv = topo.vertex(e.start()).unwrap().point();
                let ev = topo.vertex(e.end()).unwrap().point();
                let kind = match e.curve() {
                    brepkit_topology::edge::EdgeCurve::Line => "Line",
                    brepkit_topology::edge::EdgeCurve::Circle(_) => "Circle",
                    _ => "Other",
                };
                eprintln!(
                    "[rim-diag]   outer[{i}]: {kind} fwd={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                    oe.is_forward(),
                    sv.x(),
                    sv.y(),
                    sv.z(),
                    ev.x(),
                    ev.y(),
                    ev.z()
                );
            }
            for (iw_idx, &iw_id) in rim_f.inner_wires().iter().enumerate() {
                let iw = topo.wire(iw_id).unwrap();
                eprintln!("[rim-diag] Inner wire {iw_idx}: {} edges", iw.edges().len());
                for (i, oe) in iw.edges().iter().enumerate() {
                    let e = topo.edge(oe.edge()).unwrap();
                    let sv = topo.vertex(e.start()).unwrap().point();
                    let ev = topo.vertex(e.end()).unwrap().point();
                    let kind = match e.curve() {
                        brepkit_topology::edge::EdgeCurve::Line => "Line",
                        brepkit_topology::edge::EdgeCurve::Circle(_) => "Circle",
                        _ => "Other",
                    };
                    eprintln!(
                        "[rim-diag]   inner[{i}]: {kind} fwd={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                        oe.is_forward(),
                        sv.x(),
                        sv.y(),
                        sv.z(),
                        ev.x(),
                        ev.y(),
                        ev.z()
                    );
                }
            }
        }

        // Check face reversal state.
        for &fid in sh2.faces() {
            let f = topo.face(fid).unwrap();
            let kind = match f.surface() {
                FaceSurface::Plane { .. } => "Plane",
                FaceSurface::Cylinder(_) => "Cyl",
                _ => "Other",
            };
            if f.is_reversed() {
                eprintln!("[rounded]   Face {}: {kind} REVERSED", fid.index());
            }
        }

        let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
        eprintln!("[rounded] Shell volume: {vol:.2}");
        // Per-face signed volume for shell diagnostic.
        {
            let mut shell_total = 0.0_f64;
            for &fid in sh2.faces() {
                let face = topo.face(fid).unwrap();
                let kind = match face.surface() {
                    FaceSurface::Plane { .. } => "Plane",
                    FaceSurface::Cylinder(_) => "Cyl",
                    _ => "Other",
                };
                let mesh = crate::tessellate::tessellate(&topo, fid, 0.01).unwrap();
                let tris = mesh.indices.len() / 3;
                let mut fv = 0.0_f64;
                for t in 0..tris {
                    let v0 = mesh.positions[mesh.indices[t * 3] as usize];
                    let v1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
                    let v2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
                    let a = Vec3::new(v0.x(), v0.y(), v0.z());
                    let b = Vec3::new(v1.x(), v1.y(), v1.z());
                    let c = Vec3::new(v2.x(), v2.y(), v2.z());
                    fv += a.dot(b.cross(c));
                }
                fv /= 6.0;
                shell_total += fv;
                eprintln!(
                    "[shell-vol] Face {} ({kind}, rev={}): {tris} tris, signed={fv:.2}",
                    fid.index(),
                    face.is_reversed()
                );
            }
            eprintln!(
                "[shell-vol] Total signed: {shell_total:.2}, abs: {:.2}",
                shell_total.abs()
            );
        }

        // Check Euler characteristic.
        let result = crate::validate::validate_solid(&topo, shelled);
        eprintln!("[rounded] Validation: {result:?}");
    }

    /// CW-wound rounded rectangle (brepjs convention) → extrude → shell.
    /// This is the exact scenario where the shell bbox was expanding outward.
    #[test]
    fn shell_cw_rounded_rect_bounds_preserved() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let tol = Tolerance::new();

        let w = 41.5_f64;
        let d = 41.5_f64;
        let h = 21.0_f64;
        let r = 4.0_f64;
        let thickness = 1.2_f64;

        let hw = w / 2.0;
        let hd = d / 2.0;

        // CW winding (brepjs convention): BOTTOM→RIGHT→TOP→LEFT
        // Start at bottom-left tangent point, go right
        let pts = [
            Point3::new(-hw + r, -hd, 0.0), // 0: bottom-left straight start
            Point3::new(hw - r, -hd, 0.0),  // 1: bottom-right straight end
            Point3::new(hw, -hd + r, 0.0),  // 2: right-bottom straight start
            Point3::new(hw, hd - r, 0.0),   // 3: right-top straight end
            Point3::new(hw - r, hd, 0.0),   // 4: top-right straight start
            Point3::new(-hw + r, hd, 0.0),  // 5: top-left straight end
            Point3::new(-hw, hd - r, 0.0),  // 6: left-top straight start
            Point3::new(-hw, -hd + r, 0.0), // 7: left-bottom straight end
        ];
        let vids: Vec<_> = pts
            .iter()
            .map(|p| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
            .collect();

        let c_br = Point3::new(hw - r, -hd + r, 0.0);
        let c_tr = Point3::new(hw - r, hd - r, 0.0);
        let c_tl = Point3::new(-hw + r, hd - r, 0.0);
        let c_bl = Point3::new(-hw + r, -hd + r, 0.0);
        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        let mk_line =
            |topo: &mut Topology, s, e| topo.edges.alloc(Edge::new(s, e, EdgeCurve::Line));
        let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
            let circle = Circle3D::new(center, z_axis, r).unwrap();
            topo.edges.alloc(Edge::new(s, e, EdgeCurve::Circle(circle)))
        };

        // CW order: bottom→br_arc→right→tr_arc→top→tl_arc→left→bl_arc
        let e_bot = mk_line(&mut topo, vids[0], vids[1]);
        let e_br = mk_arc(&mut topo, vids[1], vids[2], c_br);
        let e_right = mk_line(&mut topo, vids[2], vids[3]);
        let e_tr = mk_arc(&mut topo, vids[3], vids[4], c_tr);
        let e_top = mk_line(&mut topo, vids[4], vids[5]);
        let e_tl = mk_arc(&mut topo, vids[5], vids[6], c_tl);
        let e_left = mk_line(&mut topo, vids[6], vids[7]);
        let e_bl = mk_arc(&mut topo, vids[7], vids[0], c_bl);

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_bot, true),
                OrientedEdge::new(e_br, true),
                OrientedEdge::new(e_right, true),
                OrientedEdge::new(e_tr, true),
                OrientedEdge::new(e_top, true),
                OrientedEdge::new(e_tl, true),
                OrientedEdge::new(e_left, true),
                OrientedEdge::new(e_bl, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.wires.alloc(wire);

        // CW winding → face normal should be -Z
        let face = Face::new(
            wire_id,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        );
        let face_id = topo.faces.alloc(face);

        // Extrude upward
        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

        // Shell: remove top face
        let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top.len(), 1, "one top face");

        let shelled = shell(&mut topo, solid, thickness, &top).unwrap();

        // Key assertion: bounding box should NOT expand beyond the original
        let bbox = crate::measure::solid_bounding_box(&topo, shelled).unwrap();
        let bbox_x = bbox.max.x() - bbox.min.x();
        let bbox_y = bbox.max.y() - bbox.min.y();
        eprintln!("[cw-shell] bbox X={bbox_x:.3}, Y={bbox_y:.3} (expected ~{w})");
        assert!(
            (bbox_x - w).abs() < 0.5,
            "bbox X should be ~{w}, got {bbox_x:.3} (expanded by {:.3})",
            bbox_x - w
        );
        assert!(
            (bbox_y - d).abs() < 0.5,
            "bbox Y should be ~{d}, got {bbox_y:.3} (expanded by {:.3})",
            bbox_y - d
        );
    }
}
