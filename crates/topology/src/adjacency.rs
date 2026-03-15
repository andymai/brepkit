//! Solid-scoped adjacency index for edge-to-face and face-to-face queries.
//!
//! [`AdjacencyIndex`] precomputes adjacency relationships from a solid's shell,
//! mapping edges to their adjacent faces and detecting non-manifold or boundary
//! edges.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::Topology;
use crate::TopologyError;
use crate::edge::EdgeId;
use crate::face::FaceId;
use crate::solid::SolidId;

/// Precomputed adjacency data for a solid's topology.
///
/// Built from a solid's shell, mapping edges to their adjacent faces
/// and detecting non-manifold/boundary edges.
#[derive(Debug, Clone)]
pub struct AdjacencyIndex {
    /// Maps each edge to the faces that reference it.
    edge_faces: HashMap<EdgeId, SmallVec<[FaceId; 2]>>,
    /// Maps each face to its neighbor faces (those sharing an edge).
    face_neighbors: HashMap<FaceId, SmallVec<[FaceId; 6]>>,
    /// Edges referenced by more than 2 faces.
    non_manifold_edges: Vec<EdgeId>,
    /// Edges referenced by exactly 1 face.
    boundary_edges: Vec<EdgeId>,
}

impl AdjacencyIndex {
    /// Builds an adjacency index from a solid's outer shell.
    ///
    /// Walks all faces in the solid's outer shell, collecting edge-to-face
    /// relationships and classifying edges as manifold, non-manifold, or
    /// boundary.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] if any referenced entity (solid, shell,
    /// face, or wire) does not exist in the topology.
    pub fn build(topo: &Topology, solid: SolidId) -> Result<Self, TopologyError> {
        let shell_id = topo.solid(solid)?.outer_shell();
        let faces = topo.shell(shell_id)?.faces().to_vec();
        Self::build_from_faces(topo, &faces)
    }

    /// Builds an adjacency index from an explicit list of faces.
    ///
    /// This is useful for open shells or partial topology that is not
    /// wrapped in a solid.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] if any referenced face or wire does not
    /// exist in the topology.
    pub fn build_from_faces(topo: &Topology, faces: &[FaceId]) -> Result<Self, TopologyError> {
        let mut edge_faces: HashMap<EdgeId, SmallVec<[FaceId; 2]>> = HashMap::new();

        // Walk all faces -> wires -> oriented edges to build edge_faces map.
        for &face_id in faces {
            let face = topo.face(face_id)?;
            // Collect all wire IDs (outer + inner) to avoid borrow issues.
            let mut wire_ids = vec![face.outer_wire()];
            wire_ids.extend_from_slice(face.inner_wires());

            for wire_id in wire_ids {
                let wire = topo.wire(wire_id)?;
                for oriented_edge in wire.edges() {
                    edge_faces
                        .entry(oriented_edge.edge())
                        .or_default()
                        .push(face_id);
                }
            }
        }

        // Classify edges and build face neighbors.
        let mut non_manifold_edges = Vec::new();
        let mut boundary_edges = Vec::new();
        let mut face_neighbors: HashMap<FaceId, SmallVec<[FaceId; 6]>> = HashMap::new();

        // Pre-populate face_neighbors with empty vecs for all faces.
        for &face_id in faces {
            face_neighbors.entry(face_id).or_default();
        }

        for (edge_id, adj_faces) in &edge_faces {
            match adj_faces.len() {
                0 => {} // Shouldn't happen if we built from wires, but harmless.
                1 => boundary_edges.push(*edge_id),
                2 => {
                    // Manifold edge: the two faces are neighbors.
                    let f0 = adj_faces[0];
                    let f1 = adj_faces[1];
                    face_neighbors.entry(f0).or_default().push(f1);
                    face_neighbors.entry(f1).or_default().push(f0);
                }
                _ => non_manifold_edges.push(*edge_id),
            }
        }

        // Deduplicate neighbor lists (a face pair may share multiple edges).
        for neighbors in face_neighbors.values_mut() {
            neighbors.sort_unstable_by_key(|id| id.index());
            neighbors.dedup();
        }

        Ok(Self {
            edge_faces,
            face_neighbors,
            non_manifold_edges,
            boundary_edges,
        })
    }

    /// Returns the faces adjacent to the given edge.
    ///
    /// Returns an empty slice if the edge is not in this index.
    #[must_use]
    pub fn faces_for_edge(&self, edge: EdgeId) -> &[FaceId] {
        self.edge_faces.get(&edge).map_or(&[], SmallVec::as_slice)
    }

    /// Returns the neighbor faces of the given face.
    ///
    /// Returns an empty slice if the face is not in this index.
    #[must_use]
    pub fn neighbors_of_face(&self, face: FaceId) -> &[FaceId] {
        self.face_neighbors
            .get(&face)
            .map_or(&[], SmallVec::as_slice)
    }

    /// Returns `true` if all edges are shared by exactly 2 faces (manifold)
    /// and there are no boundary edges.
    #[must_use]
    pub fn is_manifold(&self) -> bool {
        self.non_manifold_edges.is_empty() && self.boundary_edges.is_empty()
    }

    /// Returns edges shared by more than 2 faces.
    #[must_use]
    pub fn non_manifold_edges(&self) -> &[EdgeId] {
        &self.non_manifold_edges
    }

    /// Returns edges referenced by exactly 1 face.
    #[must_use]
    pub fn boundary_edges(&self) -> &[EdgeId] {
        &self.boundary_edges
    }

    /// Returns the full edge-to-faces map.
    ///
    /// Provided as a compatibility shim for callers that need direct map access.
    #[must_use]
    pub fn edge_faces_map(&self) -> &HashMap<EdgeId, SmallVec<[FaceId; 2]>> {
        &self.edge_faces
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[cfg(feature = "test-utils")]
    #[test]
    fn manifold_cube_adjacency() {
        let mut topo = Topology::new();
        let solid = crate::test_utils::make_unit_cube_manifold(&mut topo);
        let adj = AdjacencyIndex::build(&topo, solid).unwrap();

        assert!(adj.is_manifold());
        assert!(adj.non_manifold_edges().is_empty());
        assert!(adj.boundary_edges().is_empty());

        // A cube has 12 edges, each shared by exactly 2 faces.
        assert_eq!(adj.edge_faces_map().len(), 12);
        for faces in adj.edge_faces_map().values() {
            assert_eq!(faces.len(), 2);
        }

        // Each of the 6 faces has 4 neighbors (cube: every face touches 4 others).
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let face_ids = topo.shell(shell_id).unwrap().faces();
        assert_eq!(face_ids.len(), 6);
        for &fid in face_ids {
            assert_eq!(adj.neighbors_of_face(fid).len(), 4);
        }
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn non_manifold_detection() {
        use crate::edge::{Edge, EdgeCurve};
        use crate::face::{Face, FaceSurface};
        use crate::shell::Shell;
        use crate::solid::Solid;
        use crate::vertex::Vertex;
        use crate::wire::{OrientedEdge, Wire};
        use brepkit_math::vec::{Point3, Vec3};

        let mut topo = Topology::new();

        // Create 5 vertices forming 3 triangular faces that share one edge (e01).
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(0.5, 1.0, 0.0), 1e-7));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.5, -1.0, 0.0), 1e-7));
        let v4 = topo.add_vertex(Vertex::new(Point3::new(0.5, 0.0, 1.0), 1e-7));

        // Shared edge
        let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        // Face 1: v0-v1-v2
        let e12 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e20 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));
        let w1 = topo.add_wire(
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
        let f1 = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Face 2: v0-v1-v3
        let e13 = topo.add_edge(Edge::new(v1, v3, EdgeCurve::Line));
        let e30 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
        let w2 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e01, true),
                    OrientedEdge::new(e13, true),
                    OrientedEdge::new(e30, true),
                ],
                true,
            )
            .unwrap(),
        );
        let f2 = topo.add_face(Face::new(
            w2,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));

        // Face 3: v0-v1-v4
        let e14 = topo.add_edge(Edge::new(v1, v4, EdgeCurve::Line));
        let e40 = topo.add_edge(Edge::new(v4, v0, EdgeCurve::Line));
        let w3 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e01, true),
                    OrientedEdge::new(e14, true),
                    OrientedEdge::new(e40, true),
                ],
                true,
            )
            .unwrap(),
        );
        let f3 = topo.add_face(Face::new(
            w3,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));

        let shell = Shell::new(vec![f1, f2, f3]).unwrap();
        let shell_id = topo.add_shell(shell);
        let solid = topo.add_solid(Solid::new(shell_id, vec![]));

        let adj = AdjacencyIndex::build(&topo, solid).unwrap();

        assert!(!adj.is_manifold());
        assert_eq!(adj.non_manifold_edges().len(), 1);
        assert_eq!(adj.non_manifold_edges()[0], e01);

        // e01 is shared by 3 faces.
        assert_eq!(adj.faces_for_edge(e01).len(), 3);

        // Boundary edges: each of the 6 non-shared edges appears in only 1 face.
        assert_eq!(adj.boundary_edges().len(), 6);
    }

    #[test]
    fn build_from_faces_empty() {
        let topo = Topology::new();
        let adj = AdjacencyIndex::build_from_faces(&topo, &[]).unwrap();
        assert!(adj.is_manifold());
        assert!(adj.edge_faces_map().is_empty());
    }

    #[test]
    fn faces_for_unknown_edge_returns_empty() {
        use crate::arena::Arena;
        use crate::edge::Edge;

        let topo = Topology::new();
        let adj = AdjacencyIndex::build_from_faces(&topo, &[]).unwrap();

        // Create a dummy edge ID that isn't in the index.
        let mut dummy: Arena<Edge> = Arena::new();
        let fake_eid = dummy.alloc(Edge::new(
            {
                let mut va: Arena<crate::vertex::Vertex> = Arena::new();
                va.alloc(crate::vertex::Vertex::new(
                    brepkit_math::vec::Point3::new(0.0, 0.0, 0.0),
                    0.0,
                ))
            },
            {
                let mut va: Arena<crate::vertex::Vertex> = Arena::new();
                va.alloc(crate::vertex::Vertex::new(
                    brepkit_math::vec::Point3::new(1.0, 0.0, 0.0),
                    0.0,
                ))
            },
            crate::edge::EdgeCurve::Line,
        ));

        assert!(adj.faces_for_edge(fake_eid).is_empty());
    }
}
