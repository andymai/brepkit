//! Wire offset: produce a parallel wire at a given distance.
//!
//! Creates a new wire that is parallel to the input wire, offset by a
//! specified distance.

use brepkit_math::curves::Circle3D;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::boolean::face_polygon;

/// How to join offset edges at corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Extend adjacent offset edges until they intersect (sharp corners).
    Intersection,
    /// Insert a circular arc at each corner, centered on the original vertex.
    Arc,
    /// Connect adjacent offset edge endpoints with a straight line (bevel).
    Chamfer,
}

/// Offset a planar wire by a given distance.
///
/// Positive `distance` offsets outward (away from interior), negative
/// offsets inward. The wire must be closed and lie on a planar face.
///
/// Returns a new `WireId` for the offset wire.
///
/// This is a convenience wrapper that calls [`offset_wire_with_join`]
/// with [`JoinType::Intersection`].
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
    offset_wire_with_join(topo, face_id, distance, JoinType::Intersection)
}

/// Offset a planar wire by a given distance with a specific join type.
///
/// Positive `distance` offsets outward (away from interior), negative
/// offsets inward. The wire must be closed and lie on a planar face.
///
/// Returns a new `WireId` for the offset wire.
///
/// # Join types
///
/// - [`JoinType::Intersection`]: Extend adjacent offset edges until they
///   intersect, producing sharp corners. The result has the same number
///   of edges as the input.
/// - [`JoinType::Arc`]: Insert a circular arc at each corner, centered
///   on the original vertex with radius equal to `|distance|`. The
///   result has twice as many edges (alternating lines and arcs).
/// - [`JoinType::Chamfer`]: Connect adjacent offset edge endpoints with
///   a straight line (bevel). The result has twice as many edges
///   (alternating original-direction lines and chamfer lines).
///
/// # Errors
///
/// Returns an error if the wire is not closed, the face is not planar,
/// or offset produces degenerate geometry.
#[allow(clippy::too_many_lines)]
pub fn offset_wire_with_join(
    topo: &mut Topology,
    face_id: brepkit_topology::face::FaceId,
    distance: f64,
    join_type: JoinType,
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
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "wire offset on non-planar faces is not supported".into(),
            });
        }
    };

    // Get ordered vertices of the face's outer wire.
    let verts = face_polygon(topo, face_id)?;
    let n = verts.len();
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "wire must have at least 3 vertices".into(),
        });
    }

    // Compute edge normals (outward-pointing, in the face plane).
    // For a CCW-wound wire viewed from the face normal direction,
    // the outward normal of edge (i -> i+1) is edge_direction x face_normal.
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

    match join_type {
        JoinType::Intersection => build_intersection_wire(topo, &verts, &edge_normals, distance),
        JoinType::Arc => build_arc_wire(topo, &verts, &edge_normals, distance, face_normal),
        JoinType::Chamfer => build_chamfer_wire(topo, &verts, &edge_normals, distance),
    }
}

/// Build offset wire with intersection joins (sharp corners).
fn build_intersection_wire(
    topo: &mut Topology,
    verts: &[Point3],
    edge_normals: &[Vec3],
    distance: f64,
) -> Result<WireId, crate::OperationsError> {
    let tol = Tolerance::new();
    let n = verts.len();

    let mut offset_verts = Vec::with_capacity(n);

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };

        let offset_prev = edge_normals[prev] * distance;
        let offset_curr = edge_normals[i] * distance;

        let p0_prev = verts[prev] + offset_prev;
        let p1_prev = verts[i] + offset_prev;
        let p0_curr = verts[i] + offset_curr;
        let p1_curr = verts[(i + 1) % n] + offset_curr;

        let d_prev = p1_prev - p0_prev;
        let d_curr = p1_curr - p0_curr;

        let diff = p0_curr - p0_prev;
        let cross = d_prev.cross(d_curr);
        let cross_len_sq = cross.length_squared();

        if cross_len_sq < tol.linear * tol.linear {
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
        .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
        .collect();

    let edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.add_edge(Edge::new(vert_ids[i], vert_ids[next], EdgeCurve::Line))
        })
        .collect();

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();

    let wire = Wire::new(oriented, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

/// Build offset wire with arc joins (rounded corners).
///
/// At each corner where two offset edges meet, a circular arc is
/// inserted. The arc is centered at the original (pre-offset) vertex
/// with radius `|distance|`, sweeping from the endpoint of the
/// previous offset edge to the start of the next offset edge.
///
/// Vertices are pre-allocated so that adjacent edges share vertex IDs,
/// which is required for wire connectivity.
fn build_arc_wire(
    topo: &mut Topology,
    verts: &[Point3],
    edge_normals: &[Vec3],
    distance: f64,
    face_normal: Vec3,
) -> Result<WireId, crate::OperationsError> {
    let tol = Tolerance::new();
    let n = verts.len();
    let radius = distance.abs();

    // For each edge i, compute the two endpoints of the offset edge
    // (before any intersection trimming). The offset edge for edge i
    // goes from verts[i] + offset to verts[i+1] + offset.
    let mut offset_starts = Vec::with_capacity(n);
    let mut offset_ends = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let offset = edge_normals[i] * distance;
        offset_starts.push(verts[i] + offset);
        offset_ends.push(verts[j] + offset);
    }

    // Pre-allocate vertices so adjacent edges share IDs.
    //
    // At each corner between edge i and edge (i+1), the offset produces
    // two points: offset_ends[i] and offset_starts[(i+1)%n]. If they
    // are coincident (parallel adjacent edges), a single shared vertex
    // is used and no arc is inserted. Otherwise, two vertices are
    // created and an arc edge connects them.
    //
    // line_start_vids[i] = start vertex of line edge i
    // line_end_vids[i]   = end vertex of line edge i (= arc start at corner i+1)
    // For each corner, either:
    //   - coincident: line_end_vids[i] == line_start_vids[(i+1)%n]  (shared vertex)
    //   - non-coincident: arc from line_end_vids[i] to line_start_vids[(i+1)%n]

    // First pass: determine which corners are coincident vs need arcs.
    let mut corner_coincident = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        corner_coincident.push((offset_ends[i] - offset_starts[next]).length() < tol.linear);
    }

    // Second pass: create vertices. For each edge i we need a start and end vertex.
    // The start vertex of edge i is either:
    //   - A new vertex at offset_starts[i] (if the previous corner had an arc, the
    //     arc's end vertex will be this same ID), OR
    //   - The same ID as the previous line edge's end vertex (if previous corner was coincident).
    //
    // Strategy: create all line-start and line-end vertices, then for
    // coincident corners, make line_start[next] = line_end[i].

    let mut line_start_vids = Vec::with_capacity(n);
    let mut line_end_vids = Vec::with_capacity(n);

    // Create line-end vertices (the offset_ends points) for all edges.
    for i in 0..n {
        line_end_vids.push(topo.add_vertex(Vertex::new(offset_ends[i], tol.linear)));
    }

    // Create line-start vertices. At corner between edge (prev) and
    // edge i: if coincident, reuse line_end of prev edge; otherwise
    // create a new vertex at offset_starts[i] (arc end will use this ID).
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        if corner_coincident[prev] {
            // Coincident corner: the end of the previous line IS the
            // start of this line (no arc between them).
            line_start_vids.push(line_end_vids[prev]);
        } else {
            // Non-coincident corner: the arc's end vertex is a distinct
            // point at offset_starts[i].
            line_start_vids.push(topo.add_vertex(Vertex::new(offset_starts[i], tol.linear)));
        }
    }

    // Build edges with shared vertex IDs.
    let mut oriented_edges = Vec::with_capacity(2 * n);

    for i in 0..n {
        let next = (i + 1) % n;

        // Line edge for offset edge i.
        let line_edge = topo.add_edge(Edge::new(
            line_start_vids[i],
            line_end_vids[i],
            EdgeCurve::Line,
        ));
        oriented_edges.push(OrientedEdge::new(line_edge, true));

        // Arc edge at corner (i+1): from line_end_vids[i] to line_start_vids[next].
        if corner_coincident[i] {
            // Coincident endpoints — vertices already shared, no arc needed.
            continue;
        }

        let center = verts[next];
        let circle =
            Circle3D::new(center, face_normal, radius).map_err(crate::OperationsError::Math)?;

        let arc_edge = topo.add_edge(Edge::new(
            line_end_vids[i],
            line_start_vids[next],
            EdgeCurve::Circle(circle),
        ));
        oriented_edges.push(OrientedEdge::new(arc_edge, true));
    }

    let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

/// Build offset wire with chamfer joins (beveled corners).
///
/// At each corner, the two adjacent offset edge endpoints are
/// connected by a straight line segment instead of intersecting
/// the offset lines.
///
/// Vertices are pre-allocated so that adjacent edges share vertex IDs,
/// which is required for wire connectivity.
fn build_chamfer_wire(
    topo: &mut Topology,
    verts: &[Point3],
    edge_normals: &[Vec3],
    distance: f64,
) -> Result<WireId, crate::OperationsError> {
    let tol = Tolerance::new();
    let n = verts.len();

    // Compute offset edge endpoints (before intersection trimming).
    let mut offset_starts = Vec::with_capacity(n);
    let mut offset_ends = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let offset = edge_normals[i] * distance;
        offset_starts.push(verts[i] + offset);
        offset_ends.push(verts[j] + offset);
    }

    // Pre-allocate vertices so adjacent edges share IDs.
    // Same strategy as build_arc_wire: at each corner, if the two
    // offset endpoints are coincident, share a single vertex; otherwise
    // create two vertices connected by a chamfer edge.

    let mut corner_coincident = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        corner_coincident.push((offset_ends[i] - offset_starts[next]).length() < tol.linear);
    }

    // Create line-end vertices for all edges.
    let mut line_end_vids = Vec::with_capacity(n);
    for i in 0..n {
        line_end_vids.push(topo.add_vertex(Vertex::new(offset_ends[i], tol.linear)));
    }

    // Create line-start vertices. At the corner between edge (prev) and
    // edge i: if coincident, reuse line_end of prev; otherwise create
    // a new vertex at offset_starts[i].
    let mut line_start_vids = Vec::with_capacity(n);
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        if corner_coincident[prev] {
            line_start_vids.push(line_end_vids[prev]);
        } else {
            line_start_vids.push(topo.add_vertex(Vertex::new(offset_starts[i], tol.linear)));
        }
    }

    // Build edges with shared vertex IDs.
    let mut oriented_edges = Vec::with_capacity(2 * n);

    for i in 0..n {
        let next = (i + 1) % n;

        // Line edge for offset edge i.
        let line_edge = topo.add_edge(Edge::new(
            line_start_vids[i],
            line_end_vids[i],
            EdgeCurve::Line,
        ));
        oriented_edges.push(OrientedEdge::new(line_edge, true));

        // Chamfer edge at corner (i+1): from line_end_vids[i] to line_start_vids[next].
        if corner_coincident[i] {
            // Coincident endpoints — vertices already shared, no chamfer needed.
            continue;
        }

        let chamfer_edge = topo.add_edge(Edge::new(
            line_end_vids[i],
            line_start_vids[next],
            EdgeCurve::Line,
        ));
        oriented_edges.push(OrientedEdge::new(chamfer_edge, true));
    }

    let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

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
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol_val));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol_val));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol_val));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol_val));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.add_wire(wire);

        topo.add_face(Face::new(
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

    #[test]
    fn offset_arc_join_square() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);
        let d = 0.1;

        let offset_wid = offset_wire_with_join(&mut topo, face, d, JoinType::Arc).unwrap();

        let wire = topo.wire(offset_wid).unwrap();
        // Square has 4 edges -> 4 offset lines + 4 arcs = 8 edges.
        assert_eq!(
            wire.edges().len(),
            8,
            "arc-joined offset square should have 8 edges"
        );

        // Count line vs arc edges.
        let mut lines = 0;
        let mut arcs = 0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            match edge.curve() {
                EdgeCurve::Line => lines += 1,
                EdgeCurve::Circle(_) => arcs += 1,
                _ => {}
            }
        }
        assert_eq!(lines, 4, "should have 4 line edges");
        assert_eq!(arcs, 4, "should have 4 arc edges");
        assert_eq!(lines + arcs, 8, "all edges should be lines or arcs");

        // The total perimeter of the arc-joined offset should be
        // 4 * 1.0 (original edge length, preserved) + 4 * (pi/2 * 0.1)
        // = 4.0 + 0.2*pi ~ 4.6283
        let mut perimeter = 0.0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            match edge.curve() {
                EdgeCurve::Line => {
                    let s = topo.vertex(edge.start()).unwrap().point();
                    let e = topo.vertex(edge.end()).unwrap().point();
                    perimeter += (e - s).length();
                }
                EdgeCurve::Circle(c) => {
                    // Quarter-circle arc: pi/2 * radius.
                    perimeter += (PI / 2.0) * c.radius();
                }
                _ => {}
            }
        }
        let expected = 4.0 + 2.0 * PI * d;
        assert!(
            (perimeter - expected).abs() < 1e-6,
            "arc perimeter {perimeter} should be ~{expected}"
        );
    }

    #[test]
    fn offset_chamfer_join_square() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);
        let d = 0.1;

        let offset_wid = offset_wire_with_join(&mut topo, face, d, JoinType::Chamfer).unwrap();

        let wire = topo.wire(offset_wid).unwrap();
        // Square has 4 edges -> 4 offset lines + 4 chamfer lines = 8 edges.
        assert_eq!(
            wire.edges().len(),
            8,
            "chamfer-joined offset square should have 8 edges"
        );

        // All edges should be lines.
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            assert!(
                matches!(edge.curve(), EdgeCurve::Line),
                "chamfer offset should only contain line edges"
            );
        }

        // Each chamfer line connects two points at distance d from the
        // corner, forming a 45-degree bevel. Length = d * sqrt(2).
        // Total perimeter = 4 * 1.0 + 4 * d * sqrt(2)
        let mut perimeter = 0.0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let s = topo.vertex(edge.start()).unwrap().point();
            let e = topo.vertex(edge.end()).unwrap().point();
            perimeter += (e - s).length();
        }
        let expected = 4.0 + 4.0 * d * std::f64::consts::SQRT_2;
        assert!(
            (perimeter - expected).abs() < 1e-6,
            "chamfer perimeter {perimeter} should be ~{expected}"
        );
    }

    #[test]
    fn offset_intersection_matches_legacy() {
        // Verify that offset_wire_with_join(..., Intersection) produces
        // the same result as the legacy offset_wire function.
        let mut topo = Topology::new();
        let face = make_square(&mut topo);

        let wid = offset_wire_with_join(&mut topo, face, 0.2, JoinType::Intersection).unwrap();
        let wire = topo.wire(wid).unwrap();
        assert_eq!(wire.edges().len(), 4);

        // All vertices should be at the expected offset positions.
        let tol = Tolerance::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let pt = topo.vertex(edge.start()).unwrap().point();
            // For a unit square offset outward by 0.2, corners are at
            // (-0.2, -0.2), (1.2, -0.2), (1.2, 1.2), (-0.2, 1.2).
            let x = pt.x();
            let y = pt.y();
            assert!(
                (tol.approx_eq(x, -0.2) || tol.approx_eq(x, 1.2))
                    && (tol.approx_eq(y, -0.2) || tol.approx_eq(y, 1.2)),
                "intersection vertex at ({x}, {y}) should be a corner of offset square"
            );
        }
    }

    #[test]
    fn offset_arc_inward_square() {
        let mut topo = Topology::new();
        let face = make_square(&mut topo);

        let offset_wid = offset_wire_with_join(&mut topo, face, -0.1, JoinType::Arc).unwrap();

        let wire = topo.wire(offset_wid).unwrap();
        // Should still produce 8 edges (4 lines + 4 arcs).
        assert_eq!(wire.edges().len(), 8);

        // All line edge vertices should be inside the original square.
        let tol = Tolerance::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            assert!(
                start.x() > -tol.linear
                    && start.x() < 1.0 + tol.linear
                    && start.y() > -tol.linear
                    && start.y() < 1.0 + tol.linear,
                "inward arc offset vertex should be inside original square: ({}, {})",
                start.x(),
                start.y()
            );
        }
    }
}
