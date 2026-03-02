//! Wire offset: produce a parallel wire at a given distance.
//!
//! Equivalent to `BRepOffsetAPI_MakeOffset` in `OpenCascade` for 2D
//! wire offsetting. Creates a new wire that is parallel to the input
//! wire, offset by a specified distance.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::boolean::face_vertices;

/// Offset a planar wire by a given distance.
///
/// Positive `distance` offsets outward (away from interior), negative
/// offsets inward. The wire must be closed and lie on a planar face.
///
/// Returns a new `WireId` for the offset wire.
///
/// # Algorithm
///
/// For each edge of the wire, compute an offset line by shifting the
/// edge along its inward-facing normal. Then compute new vertices at
/// the intersection of adjacent offset lines.
///
/// # Errors
///
/// Returns an error if the wire is not closed, the face is not planar,
/// or offset produces degenerate geometry.
pub fn offset_wire(
    topo: &mut Topology,
    face_id: brepkit_topology::face::FaceId,
    distance: f64,
) -> Result<WireId, crate::OperationsError> {
    let tol = Tolerance::new();

    if tol.approx_eq(distance, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "offset distance is zero".into(),
        });
    }

    // Get face normal.
    let face = topo.face(face_id)?;
    let face_normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Nurbs(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "wire offset on NURBS faces is not supported".into(),
            });
        }
    };

    // Get ordered vertices of the face's outer wire.
    let verts = face_vertices(topo, face_id)?;
    let n = verts.len();
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "wire must have at least 3 vertices".into(),
        });
    }

    // Compute edge normals (outward-pointing, in the face plane).
    // For a CCW-wound wire viewed from the face normal direction,
    // the outward normal of edge (i → i+1) is edge_direction × face_normal.
    let mut edge_normals = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let edge_dir = verts[j] - verts[i];
        let outward = edge_dir.cross(face_normal);
        let len = outward.length();
        if len < tol.linear {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("degenerate edge at vertex {i}"),
            });
        }
        edge_normals.push(Vec3::new(
            outward.x() / len,
            outward.y() / len,
            outward.z() / len,
        ));
    }

    // Offset each edge and compute new vertex positions at intersections.
    let mut offset_verts = Vec::with_capacity(n);

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };

        // Offset lines for the previous edge and current edge.
        // Previous edge: from verts[prev] to verts[i], offset by edge_normals[prev].
        // Current edge:  from verts[i] to verts[(i+1)%n], offset by edge_normals[i].

        let offset_prev = edge_normals[prev] * distance;
        let offset_curr = edge_normals[i] * distance;

        let p0_prev = verts[prev] + offset_prev;
        let p1_prev = verts[i] + offset_prev;
        let p0_curr = verts[i] + offset_curr;
        let p1_curr = verts[(i + 1) % n] + offset_curr;

        // Compute intersection of the two offset lines.
        let d_prev = p1_prev - p0_prev;
        let d_curr = p1_curr - p0_curr;

        // Use the parametric intersection: p0_prev + t * d_prev = p0_curr + s * d_curr
        // Solve for t using the cross product with d_curr.
        let diff = p0_curr - p0_prev;
        let cross = d_prev.cross(d_curr);
        let cross_len_sq = cross.length_squared();

        if cross_len_sq < tol.linear * tol.linear {
            // Parallel edges — use the midpoint of the offset positions.
            #[allow(clippy::manual_midpoint)]
            let mid = Point3::new(
                (p1_prev.x() + p0_curr.x()) / 2.0,
                (p1_prev.y() + p0_curr.y()) / 2.0,
                (p1_prev.z() + p0_curr.z()) / 2.0,
            );
            offset_verts.push(mid);
        } else {
            let t = diff.cross(d_curr).dot(cross) / cross_len_sq;
            let intersection = Point3::new(
                d_prev.x().mul_add(t, p0_prev.x()),
                d_prev.y().mul_add(t, p0_prev.y()),
                d_prev.z().mul_add(t, p0_prev.z()),
            );
            offset_verts.push(intersection);
        }
    }

    // Build the offset wire.
    let vert_ids: Vec<_> = offset_verts
        .iter()
        .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
        .collect();

    let edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.edges
                .alloc(Edge::new(vert_ids[i], vert_ids[next], EdgeCurve::Line))
        })
        .collect();

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();

    let wire = Wire::new(oriented, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.wires.alloc(wire))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// Helper: make a unit square face on XY plane.
    fn make_square(topo: &mut Topology) -> brepkit_topology::face::FaceId {
        let tol_val = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol_val));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol_val));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);

        topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    #[test]
    fn offset_outward_square() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);

        let offset_wid = offset_wire(&mut topo, face, 0.1).unwrap();

        let wire = topo.wire(offset_wid).unwrap();
        assert_eq!(wire.edges().len(), 4, "offset square should have 4 edges");

        // Verify the offset wire is larger: check that all vertices are
        // outside the original [-0.1, 1.1] range.
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let tol = Tolerance::new();
            assert!(
                start.x() < -tol.linear
                    || start.x() > 1.0 + tol.linear
                    || start.y() < -tol.linear
                    || start.y() > 1.0 + tol.linear,
                "offset vertex should be outside original square: ({}, {})",
                start.x(),
                start.y()
            );
        }
    }

    #[test]
    fn offset_inward_square() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);

        let offset_wid = offset_wire(&mut topo, face, -0.1).unwrap();

        let wire = topo.wire(offset_wid).unwrap();
        assert_eq!(wire.edges().len(), 4);

        // Verify all vertices are inside the original square.
        let tol = Tolerance::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            assert!(
                start.x() > tol.linear
                    && start.x() < 1.0 - tol.linear
                    && start.y() > tol.linear
                    && start.y() < 1.0 - tol.linear,
                "inward offset vertex should be inside square: ({}, {})",
                start.x(),
                start.y()
            );
        }
    }

    #[test]
    fn offset_zero_distance_error() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);
        assert!(offset_wire(&mut topo, face, 0.0).is_err());
    }

    #[test]
    fn offset_preserves_edge_count() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);

        let offset_wid = offset_wire(&mut topo, face, 0.5).unwrap();
        let wire = topo.wire(offset_wid).unwrap();
        assert_eq!(wire.edges().len(), 4, "offset should preserve edge count");
    }
}
