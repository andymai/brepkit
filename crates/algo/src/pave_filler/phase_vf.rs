//! Phase VF: Vertex-on-face interference detection.
//!
//! For each (vertex, face) pair across solids, checks if the vertex
//! lies on the face surface. If so, records a VF interference and
//! adds the vertex to the face's `vertices_in` set.

use std::collections::HashSet;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Interference};
use crate::error::AlgoError;

/// Detect vertices lying on faces between the two solids.
///
/// Checks vertices of A against faces of B, and vertices of B against
/// faces of A. When a vertex lies on a face surface (within tolerance),
/// a VF interference is recorded and the vertex is added to the face's
/// `vertices_in` set.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let verts_a = brepkit_topology::explorer::solid_vertices(topo, solid_a)?;
    let verts_b = brepkit_topology::explorer::solid_vertices(topo, solid_b)?;
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Collect face-edge vertex sets to skip vertices already on face edges
    let face_edge_verts_b = collect_face_edge_vertices(topo, &faces_b)?;
    let face_edge_verts_a = collect_face_edge_vertices(topo, &faces_a)?;

    // Check vertices of A against faces of B
    check_vertex_face_pairs(topo, &verts_a, &faces_b, &face_edge_verts_b, tol, arena)?;

    // Check vertices of B against faces of A
    check_vertex_face_pairs(topo, &verts_b, &faces_a, &face_edge_verts_a, tol, arena)?;

    Ok(())
}

/// Collect the set of vertices on each face's boundary edges.
fn collect_face_edge_vertices(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<HashSet<VertexId>>, AlgoError> {
    let mut result = Vec::with_capacity(faces.len());
    for &fid in faces {
        let edges = brepkit_topology::explorer::face_edges(topo, fid)?;
        let mut verts = HashSet::new();
        for eid in edges {
            let edge = topo.edge(eid)?;
            verts.insert(edge.start());
            verts.insert(edge.end());
        }
        result.push(verts);
    }
    Ok(result)
}

/// Check each vertex against each face and record VF interferences.
#[allow(clippy::too_many_lines)]
fn check_vertex_face_pairs(
    topo: &Topology,
    vertices: &[VertexId],
    faces: &[FaceId],
    face_edge_verts: &[HashSet<VertexId>],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    for &vid in vertices {
        let resolved_vid = arena.resolve_vertex(vid);
        let vertex = topo.vertex(resolved_vid)?;
        let pos = vertex.point();
        let vtol = vertex.tolerance();

        for (face_idx, &fid) in faces.iter().enumerate() {
            // Skip if vertex is already on this face's boundary
            if face_edge_verts[face_idx].contains(&resolved_vid) {
                continue;
            }
            // Also check the unresolved vertex
            if face_edge_verts[face_idx].contains(&vid) {
                continue;
            }

            let face = topo.face(fid)?;
            let surface = face.surface();
            let combined_tol = vtol + tol.linear;

            match surface {
                FaceSurface::Plane { normal, d } => {
                    // Point-to-plane distance: |dot(pos, normal) - d|
                    let dist = (dot_point_normal(pos, *normal) - d).abs();
                    if dist <= combined_tol {
                        // Compute UV as projection onto the plane
                        // For planes we use a simple projection — pick two
                        // orthonormal axes on the plane surface.
                        let (u_axis, v_axis) = plane_local_axes(*normal)?;
                        // Project pos onto the plane's local frame.
                        // Origin on the plane: normal * d (as Point3).
                        let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);
                        let diff = pos - origin; // Vec3
                        let u = diff.dot(u_axis);
                        let v = diff.dot(v_axis);

                        record_vf(arena, resolved_vid, fid, (u, v));
                    }
                }
                _ => {
                    // Parametric surface: project and evaluate
                    if let Some((u, v)) = surface.project_point(pos) {
                        if let Some(surf_pt) = surface.evaluate(u, v) {
                            let dist = (pos - surf_pt).length();
                            if dist <= combined_tol {
                                record_vf(arena, resolved_vid, fid, (u, v));
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Record a VF interference and add vertex to face info.
fn record_vf(arena: &mut GfaArena, vertex: VertexId, face: FaceId, uv: (f64, f64)) {
    arena
        .interference
        .vf
        .push(Interference::VF { vertex, face, uv });
    arena.face_info_mut(face).vertices_in.insert(vertex);

    log::debug!(
        "VF: vertex {vertex:?} on face {face:?} at uv=({:.6}, {:.6})",
        uv.0,
        uv.1,
    );
}

/// Dot product of a point (as position vector) with a direction.
fn dot_point_normal(p: brepkit_math::vec::Point3, n: Vec3) -> f64 {
    p.x() * n.x() + p.y() * n.y() + p.z() * n.z()
}

/// Compute two orthonormal axes on a plane given its normal.
///
/// # Errors
///
/// Returns [`AlgoError`] if the normal is degenerate.
fn plane_local_axes(normal: Vec3) -> Result<(Vec3, Vec3), AlgoError> {
    // Pick an axis not parallel to normal
    let reference = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = normal.cross(reference).normalize()?;
    let v = normal.cross(u).normalize()?;
    Ok((u, v))
}
