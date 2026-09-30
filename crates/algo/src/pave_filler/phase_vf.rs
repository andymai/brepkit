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
    // A NURBS surface with positive weights lies in its control points' box,
    // so a vertex farther from that box than its tolerance is farther from
    // the surface too, and needs no projection.
    let hulls: Vec<Option<brepkit_math::aabb::Aabb3>> = faces
        .iter()
        .map(|&fid| {
            topo.face(fid).map(|f| match f.surface() {
                FaceSurface::Nurbs(n) if n.weights().iter().flatten().all(|&w| w > 0.0) => {
                    Some(n.aabb())
                }
                _ => None,
            })
        })
        .collect::<Result<_, _>>()?;
    for &vid in vertices {
        let resolved_vid = arena.resolve_vertex(vid);
        let vertex = topo.vertex(resolved_vid)?;
        let pos = vertex.point();
        let vtol = vertex.tolerance();

        for (face_idx, &fid) in faces.iter().enumerate() {
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
            if hulls[face_idx].is_some_and(|h| !h.expanded(combined_tol).contains_point(pos)) {
                continue;
            }

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
                    if let Some((u, v)) = surface.project_point(pos)
                        && let Some(surf_pt) = surface.evaluate(u, v)
                    {
                        let dist = (pos - surf_pt).length();
                        if dist <= combined_tol {
                            record_vf(arena, resolved_vid, fid, (u, v));
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// The hull bound skips only vertices farther than their tolerance from
    /// a NURBS face's control-point box: a vertex just past the box, within
    /// tolerance of the patch at a corner where the patch meets the box, is
    /// still recorded, and one well clear of the box is not.
    #[test]
    fn hull_bound_keeps_vertices_within_tolerance_of_a_nurbs_face() {
        let tol = Tolerance::new();
        let mut topo = Topology::new();
        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let vids: Vec<_> = corners
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
            .collect();
        let eids: Vec<_> = (0..4)
            .map(|i| topo.add_edge(Edge::new(vids[i], vids[(i + 1) % 4], EdgeCurve::Line)))
            .collect();
        let wire = Wire::new(
            eids.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let saddle = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![vec![corners[0], corners[3]], vec![corners[1], corners[2]]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(saddle)));
        let near = topo.add_vertex(Vertex::new(
            Point3::new(1.0 + 0.5 * tol.linear, 0.0, 1.0),
            tol.linear,
        ));
        let far = topo.add_vertex(Vertex::new(Point3::new(1.5, 0.5, 0.5), tol.linear));
        let mut arena = GfaArena::new();
        check_vertex_face_pairs(
            &topo,
            &[near, far],
            &[face],
            &[HashSet::new()],
            tol,
            &mut arena,
        )
        .unwrap();
        let recorded: Vec<VertexId> = arena
            .interference
            .vf
            .iter()
            .filter_map(|i| match i {
                crate::ds::Interference::VF { vertex, .. } => Some(*vertex),
                _ => None,
            })
            .collect();
        assert_eq!(recorded, vec![near]);
    }
}
