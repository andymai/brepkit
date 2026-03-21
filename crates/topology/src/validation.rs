//! Topology validation utilities.
//!
//! These functions check structural invariants of topological entities
//! such as wire closure and shell manifoldness.

use std::collections::HashMap;

use crate::Topology;
use crate::TopologyError;
use crate::shell::Shell;
use crate::wire::{Wire, WireId};

/// Validates that a wire forms a closed loop.
///
/// A closed wire requires that for each consecutive pair of oriented edges
/// the end vertex of the first equals the start vertex of the second, and
/// that the last edge connects back to the first.
///
/// # Errors
///
/// Returns [`TopologyError::WireNotClosed`] if the wire is not closed.
/// Returns [`TopologyError::EdgeNotFound`] if any edge id is invalid.
pub fn validate_wire_closed(wire: &Wire, topo: &Topology) -> Result<(), TopologyError> {
    if !wire.is_closed() {
        return Err(TopologyError::WireNotClosed);
    }

    let oriented = wire.edges();
    for window in oriented.windows(2) {
        let current = &window[0];
        let next = &window[1];

        let current_edge = topo.edge(current.edge())?;
        let next_edge = topo.edge(next.edge())?;

        if current.oriented_end(current_edge) != next.oriented_start(next_edge) {
            return Err(TopologyError::WireNotClosed);
        }
    }

    // Check last -> first closure.
    if let (Some(last), Some(first)) = (oriented.last(), oriented.first()) {
        let last_edge = topo.edge(last.edge())?;
        let first_edge = topo.edge(first.edge())?;

        if last.oriented_end(last_edge) != first.oriented_start(first_edge) {
            return Err(TopologyError::WireNotClosed);
        }
    }

    Ok(())
}

/// Collects all edge usage counts for a given wire.
fn count_wire_edges(
    wire_id: WireId,
    topo: &Topology,
    counts: &mut HashMap<usize, usize>,
) -> Result<(), TopologyError> {
    let wire = topo.wire(wire_id)?;
    for oe in wire.edges() {
        *counts.entry(oe.edge().index()).or_insert(0) += 1;
    }
    Ok(())
}

/// Validates that a shell is manifold.
///
/// A manifold shell requires that every edge is shared by at most two faces.
/// This function walks shell -> faces -> wires -> edges, counts each edge's
/// usage, and reports any edge shared by more than two faces.
///
/// Note: [`AdjacencyIndex::is_manifold()`](crate::adjacency::AdjacencyIndex::is_manifold)
/// performs the same check but also detects boundary edges (shared by only 1 face).
/// Use `AdjacencyIndex` when you need the full adjacency data; use this function
/// for a lightweight pass/fail manifold check.
///
/// # Errors
///
/// Returns [`TopologyError::NonManifold`] if any edge is shared by more
/// than two faces.
/// Returns entity-not-found errors if any referenced ID is invalid.
pub fn validate_shell_manifold(shell: &Shell, topo: &Topology) -> Result<(), TopologyError> {
    let mut edge_counts: HashMap<usize, usize> = HashMap::new();

    for &face_id in shell.faces() {
        let face = topo.face(face_id)?;

        // Outer wire.
        count_wire_edges(face.outer_wire(), topo, &mut edge_counts)?;

        // Inner wires (holes).
        for &inner_wire_id in face.inner_wires() {
            count_wire_edges(inner_wire_id, topo, &mut edge_counts)?;
        }
    }

    for (&edge_index, &count) in &edge_counts {
        if count > 2 {
            return Err(TopologyError::NonManifold {
                reason: format!(
                    "edge index {edge_index} is shared by {count} faces (max 2 for manifold)"
                ),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};

    use crate::edge::{Edge, EdgeCurve};
    use crate::face::{Face, FaceSurface};
    use crate::topology::Topology;
    use crate::wire::OrientedEdge;

    use super::*;

    /// Helper: builds a closed triangular wire from 3 vertices.
    fn make_triangle(topo: &mut Topology) -> WireId {
        use crate::vertex::Vertex;

        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

        topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e0, true),
                    OrientedEdge::new(e1, true),
                    OrientedEdge::new(e2, true),
                ],
                true,
            )
            .unwrap(),
        )
    }

    #[test]
    fn validate_wire_closed_triangle() {
        let mut topo = Topology::new();
        let wid = make_triangle(&mut topo);
        let wire = topo.wire(wid).unwrap();
        assert!(validate_wire_closed(wire, &topo).is_ok());
    }

    #[test]
    fn manifold_two_face_shell() {
        // Two triangular faces sharing one edge — each edge used at most 2 times.
        let mut topo = Topology::new();

        let v0 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));
        let v3 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));

        let shared = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let ea0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e_a1 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));
        let eb0 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let eb1 = topo.add_edge(Edge::new(v3, v1, EdgeCurve::Line));

        let w0 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(ea0, true),
                    OrientedEdge::new(shared, true),
                    OrientedEdge::new(e_a1, true),
                ],
                true,
            )
            .unwrap(),
        );
        let w1 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(shared, false),
                    OrientedEdge::new(eb0, true),
                    OrientedEdge::new(eb1, true),
                ],
                true,
            )
            .unwrap(),
        );

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let f0 = topo.add_face(Face::new(w0, vec![], FaceSurface::Plane { normal, d: 0.0 }));
        let f1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Plane { normal, d: 0.0 }));

        let shell = Shell::new(vec![f0, f1]).unwrap();
        assert!(validate_shell_manifold(&shell, &topo).is_ok());
    }

    #[test]
    fn non_manifold_three_face_shared_edge() {
        // Three faces sharing a single edge -> non-manifold.
        let mut topo = Topology::new();

        let v0 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));
        let v3 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));
        let v4 = topo.add_vertex(crate::vertex::Vertex::new(Point3::new(0.5, 0.5, 1.0), 1e-7));

        // Shared edge between all three faces.
        let shared = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        // Face 1: v0-v1-v2
        let e_a = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e_b = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));
        let w0 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(shared, true),
                    OrientedEdge::new(e_a, true),
                    OrientedEdge::new(e_b, true),
                ],
                true,
            )
            .unwrap(),
        );

        // Face 2: v0-v1-v3
        let e_c = topo.add_edge(Edge::new(v1, v3, EdgeCurve::Line));
        let e_d = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
        let w1 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(shared, true),
                    OrientedEdge::new(e_c, true),
                    OrientedEdge::new(e_d, true),
                ],
                true,
            )
            .unwrap(),
        );

        // Face 3: v0-v1-v4 — third face sharing the same edge
        let e_e = topo.add_edge(Edge::new(v1, v4, EdgeCurve::Line));
        let e_f = topo.add_edge(Edge::new(v4, v0, EdgeCurve::Line));
        let w2 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(shared, true),
                    OrientedEdge::new(e_e, true),
                    OrientedEdge::new(e_f, true),
                ],
                true,
            )
            .unwrap(),
        );

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let f0 = topo.add_face(Face::new(w0, vec![], FaceSurface::Plane { normal, d: 0.0 }));
        let f1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Plane { normal, d: 0.0 }));
        let f2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Plane { normal, d: 0.0 }));

        let shell = Shell::new(vec![f0, f1, f2]).unwrap();
        let result = validate_shell_manifold(&shell, &topo);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, TopologyError::NonManifold { .. }),
            "expected NonManifold, got {err:?}"
        );
    }
}
