//! Topological adjacency graph for face connectivity queries.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::TopologyError;
use crate::arena::Arena;
use crate::face::{Face, FaceId};
use crate::shell::Shell;
use crate::wire::Wire;

/// A precomputed adjacency graph over the faces of a shell.
///
/// Two faces are adjacent if they share at least one edge.
#[derive(Debug, Clone)]
pub struct TopologyGraph {
    /// Map from each face index to its list of adjacent faces.
    adjacency: HashMap<usize, SmallVec<[FaceId; 6]>>,
}

impl TopologyGraph {
    /// Builds an adjacency graph from a shell.
    ///
    /// Walks all faces and their wires to discover shared edges, then
    /// records mutual adjacency for each pair of faces sharing an edge.
    ///
    /// # Errors
    ///
    /// Returns entity-not-found errors if any referenced ID is invalid.
    pub fn from_shell(
        shell: &Shell,
        faces: &Arena<Face>,
        wires: &Arena<Wire>,
    ) -> Result<Self, TopologyError> {
        // Map from edge index → list of face IDs that reference it.
        let mut edge_to_faces: HashMap<usize, SmallVec<[FaceId; 2]>> = HashMap::new();

        for &face_id in shell.faces() {
            let face = faces
                .get(face_id)
                .ok_or(TopologyError::FaceNotFound(face_id))?;

            // Iterate all wires of the face: outer first, then inner (holes).
            let all_wire_ids =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());

            for wire_id in all_wire_ids {
                let wire = wires
                    .get(wire_id)
                    .ok_or(TopologyError::WireNotFound(wire_id))?;
                for oe in wire.edges() {
                    edge_to_faces
                        .entry(oe.edge().index())
                        .or_default()
                        .push(face_id);
                }
            }
        }

        // Build adjacency: for each edge shared by exactly 2 faces,
        // record mutual adjacency.
        let mut adjacency: HashMap<usize, SmallVec<[FaceId; 6]>> = HashMap::new();
        for face_ids in edge_to_faces.values() {
            if face_ids.len() == 2 {
                let a = face_ids[0];
                let b = face_ids[1];
                let a_neighbors = adjacency.entry(a.index()).or_default();
                if !a_neighbors.contains(&b) {
                    a_neighbors.push(b);
                }
                let b_neighbors = adjacency.entry(b.index()).or_default();
                if !b_neighbors.contains(&a) {
                    b_neighbors.push(a);
                }
            }
        }

        Ok(Self { adjacency })
    }

    /// Returns the faces adjacent to the given face, or an empty slice
    /// if the face is not present in the graph.
    #[must_use]
    pub fn adjacent_faces(&self, face: FaceId) -> &[FaceId] {
        self.adjacency
            .get(&face.index())
            .map_or(&[], SmallVec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};

    use crate::edge::{Edge, EdgeCurve};
    use crate::face::{Face, FaceSurface};
    use crate::vertex::Vertex;
    use crate::wire::OrientedEdge;

    use super::*;

    #[test]
    fn three_face_adjacency() {
        // Three triangular faces arranged in a strip:
        //   f0 shares edge with f1, f1 shares edge with f2.
        let mut vertices = Arena::new();
        let mut edges = Arena::new();
        let mut wires_arena = Arena::new();
        let mut faces_arena = Arena::new();

        let v0 = vertices.alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = vertices.alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = vertices.alloc(Vertex::new(Point3::new(0.5, 1.0, 0.0), 1e-7));
        let v3 = vertices.alloc(Vertex::new(Point3::new(1.5, 1.0, 0.0), 1e-7));
        let v4 = vertices.alloc(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-7));

        // f0: v0-v1-v2
        let e01 = edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e12 = edges.alloc(Edge::new(v1, v2, EdgeCurve::Line)); // shared f0-f1
        let e20 = edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

        // f1: v1-v3-v2  (shares e12 reversed with f0)
        let e13 = edges.alloc(Edge::new(v1, v3, EdgeCurve::Line));
        let e32 = edges.alloc(Edge::new(v3, v2, EdgeCurve::Line));
        // Also shares edge v1-v3 with f2.

        // f2: v1-v4-v3  (shares e13 reversed with f1)
        let e14 = edges.alloc(Edge::new(v1, v4, EdgeCurve::Line));
        let e43 = edges.alloc(Edge::new(v4, v3, EdgeCurve::Line));

        let w0 = wires_arena.alloc(
            Wire::new(
                vec![
                    OrientedEdge::new(e01, true),
                    OrientedEdge::new(e12, true),
                    OrientedEdge::new(e20, true),
                ],
                true,
            )
            .unwrap(),
        );
        let w1 = wires_arena.alloc(
            Wire::new(
                vec![
                    OrientedEdge::new(e12, false),
                    OrientedEdge::new(e13, true),
                    OrientedEdge::new(e32, true),
                ],
                true,
            )
            .unwrap(),
        );
        let w2 = wires_arena.alloc(
            Wire::new(
                vec![
                    OrientedEdge::new(e13, false),
                    OrientedEdge::new(e14, true),
                    OrientedEdge::new(e43, true),
                ],
                true,
            )
            .unwrap(),
        );

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let f0 = faces_arena.alloc(Face::new(w0, vec![], FaceSurface::Plane { normal, d: 0.0 }));
        let f1 = faces_arena.alloc(Face::new(w1, vec![], FaceSurface::Plane { normal, d: 0.0 }));
        let f2 = faces_arena.alloc(Face::new(w2, vec![], FaceSurface::Plane { normal, d: 0.0 }));

        let shell = Shell::new(vec![f0, f1, f2]).unwrap();
        let graph = TopologyGraph::from_shell(&shell, &faces_arena, &wires_arena).unwrap();

        // f0 is adjacent to f1 (via e12)
        assert_eq!(graph.adjacent_faces(f0).len(), 1);
        assert!(graph.adjacent_faces(f0).contains(&f1));

        // f1 is adjacent to f0 and f2
        assert_eq!(graph.adjacent_faces(f1).len(), 2);
        assert!(graph.adjacent_faces(f1).contains(&f0));
        assert!(graph.adjacent_faces(f1).contains(&f2));

        // f2 is adjacent to f1 (via e13)
        assert_eq!(graph.adjacent_faces(f2).len(), 1);
        assert!(graph.adjacent_faces(f2).contains(&f1));
    }
}
