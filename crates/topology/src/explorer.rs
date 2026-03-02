//! Topology exploration and query utilities.
//!
//! Equivalent to `TopExp_Explorer` in `OpenCascade`. Provides functions
//! for traversing the B-Rep topology graph and querying relationships
//! between entities.

use std::collections::{HashMap, HashSet};

use crate::Topology;
use crate::TopologyError;
use crate::edge::EdgeId;
use crate::face::FaceId;
use crate::solid::SolidId;
use crate::vertex::VertexId;
use crate::wire::WireId;

// ── Solid queries ──────────────────────────────────────────────────

/// Get all unique face IDs from a solid (outer + inner shells).
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn solid_faces(topo: &Topology, solid: SolidId) -> Result<Vec<FaceId>, TopologyError> {
    let solid_data = topo.solid(solid)?;
    let mut faces = Vec::new();

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        faces.extend_from_slice(shell.faces());
    }

    Ok(faces)
}

/// Get all unique edge IDs from a solid.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn solid_edges(topo: &Topology, solid: SolidId) -> Result<Vec<EdgeId>, TopologyError> {
    let mut seen = HashSet::new();
    let mut edges = Vec::new();

    for face_id in solid_faces(topo, solid)? {
        for eid in face_edges(topo, face_id)? {
            if seen.insert(eid.index()) {
                edges.push(eid);
            }
        }
    }

    Ok(edges)
}

/// Get all unique vertex IDs from a solid.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn solid_vertices(topo: &Topology, solid: SolidId) -> Result<Vec<VertexId>, TopologyError> {
    let mut seen = HashSet::new();
    let mut vertices = Vec::new();

    for eid in solid_edges(topo, solid)? {
        let edge = topo.edge(eid)?;
        if seen.insert(edge.start().index()) {
            vertices.push(edge.start());
        }
        if seen.insert(edge.end().index()) {
            vertices.push(edge.end());
        }
    }

    Ok(vertices)
}

// ── Face queries ───────────────────────────────────────────────────

/// Get all unique edge IDs from a face (outer wire + inner wires).
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn face_edges(topo: &Topology, face: FaceId) -> Result<Vec<EdgeId>, TopologyError> {
    let face_data = topo.face(face)?;
    let mut seen = HashSet::new();
    let mut edges = Vec::new();

    for wire_id in
        std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
    {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            if seen.insert(oe.edge().index()) {
                edges.push(oe.edge());
            }
        }
    }

    Ok(edges)
}

/// Get all unique vertex IDs from a face.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn face_vertices(topo: &Topology, face: FaceId) -> Result<Vec<VertexId>, TopologyError> {
    let mut seen = HashSet::new();
    let mut vertices = Vec::new();

    for eid in face_edges(topo, face)? {
        let edge = topo.edge(eid)?;
        if seen.insert(edge.start().index()) {
            vertices.push(edge.start());
        }
        if seen.insert(edge.end().index()) {
            vertices.push(edge.end());
        }
    }

    Ok(vertices)
}

// ── Edge queries ───────────────────────────────────────────────────

/// Build a map from edge index to the faces that reference it.
///
/// This is useful for finding shared edges (manifold edges appear in
/// exactly 2 faces) and boundary edges (appear in only 1 face).
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn edge_to_face_map(
    topo: &Topology,
    solid: SolidId,
) -> Result<HashMap<usize, Vec<FaceId>>, TopologyError> {
    let mut map: HashMap<usize, Vec<FaceId>> = HashMap::new();

    for face_id in solid_faces(topo, solid)? {
        for eid in face_edges(topo, face_id)? {
            map.entry(eid.index()).or_default().push(face_id);
        }
    }

    Ok(map)
}

/// Find all edges shared between two faces of a solid.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn shared_edges(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
) -> Result<Vec<EdgeId>, TopologyError> {
    let edges_a: HashSet<usize> = face_edges(topo, face_a)?
        .iter()
        .map(|e| e.index())
        .collect();
    let edges_b = face_edges(topo, face_b)?;

    Ok(edges_b
        .into_iter()
        .filter(|e| edges_a.contains(&e.index()))
        .collect())
}

/// Find all faces adjacent to a given face (sharing at least one edge).
///
/// Requires a precomputed edge-to-face map for efficiency.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn adjacent_faces(
    topo: &Topology,
    face: FaceId,
    edge_face_map: &HashMap<usize, Vec<FaceId>, impl std::hash::BuildHasher>,
) -> Result<Vec<FaceId>, TopologyError> {
    let mut seen = HashSet::new();
    let mut neighbors = Vec::new();

    for eid in face_edges(topo, face)? {
        if let Some(faces) = edge_face_map.get(&eid.index()) {
            for &fid in faces {
                if fid.index() != face.index() && seen.insert(fid.index()) {
                    neighbors.push(fid);
                }
            }
        }
    }

    Ok(neighbors)
}

// ── Wire queries ───────────────────────────────────────────────────

/// Get all wires from a face.
///
/// Returns the outer wire followed by any inner wires.
///
/// # Errors
///
/// Returns an error if the face lookup fails.
pub fn face_wires(topo: &Topology, face: FaceId) -> Result<Vec<WireId>, TopologyError> {
    let face_data = topo.face(face)?;
    let mut wires = vec![face_data.outer_wire()];
    wires.extend_from_slice(face_data.inner_wires());
    Ok(wires)
}

// ── Counting ───────────────────────────────────────────────────────

/// Count entities in a solid.
///
/// Returns `(faces, edges, vertices)` — the Euler characteristic
/// components for topology validation.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn solid_entity_counts(
    topo: &Topology,
    solid: SolidId,
) -> Result<(usize, usize, usize), TopologyError> {
    let faces = solid_faces(topo, solid)?.len();
    let edges = solid_edges(topo, solid)?.len();
    let vertices = solid_vertices(topo, solid)?.len();
    Ok((faces, edges, vertices))
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    #![allow(clippy::unwrap_used)]

    use crate::Topology;
    use crate::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn cube_entity_counts() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let (f, e, v) = solid_entity_counts(&topo, cube).unwrap();
        assert_eq!(f, 6, "cube should have 6 faces");
        assert_eq!(e, 12, "cube should have 12 edges");
        assert_eq!(v, 8, "cube should have 8 vertices");
    }

    #[test]
    fn cube_euler_characteristic() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let (f, e, v) = solid_entity_counts(&topo, cube).unwrap();
        // Euler characteristic for a convex polyhedron: V - E + F = 2
        #[allow(clippy::cast_possible_wrap)]
        let euler = (v as i64) - (e as i64) + (f as i64);
        assert_eq!(euler, 2, "V-E+F should be 2 for a cube");
    }

    #[test]
    fn cube_edge_to_face_map() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let map = edge_to_face_map(&topo, cube).unwrap();

        // Every edge of a manifold cube is shared by exactly 2 faces.
        for (edge_idx, faces) in &map {
            assert_eq!(
                faces.len(),
                2,
                "edge {edge_idx} should be shared by 2 faces, got {}",
                faces.len()
            );
        }
    }

    #[test]
    fn cube_face_has_4_edges() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let faces = solid_faces(&topo, cube).unwrap();
        for fid in faces {
            let edges = face_edges(&topo, fid).unwrap();
            assert_eq!(edges.len(), 4, "each cube face should have 4 edges");
        }
    }

    #[test]
    fn cube_face_has_4_vertices() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let faces = solid_faces(&topo, cube).unwrap();
        for fid in faces {
            let verts = face_vertices(&topo, fid).unwrap();
            assert_eq!(verts.len(), 4, "each cube face should have 4 vertices");
        }
    }

    #[test]
    fn shared_edges_adjacent_cube_faces() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let faces = solid_faces(&topo, cube).unwrap();
        let map = edge_to_face_map(&topo, cube).unwrap();

        // Find adjacent faces of the first face.
        let neighbors = adjacent_faces(&topo, faces[0], &map).unwrap();

        // Each face of a cube is adjacent to 4 other faces.
        assert_eq!(neighbors.len(), 4, "cube face should have 4 adjacent faces");

        // Check that each neighbor shares exactly 1 edge with face[0].
        for &neighbor in &neighbors {
            let shared = shared_edges(&topo, faces[0], neighbor).unwrap();
            assert_eq!(
                shared.len(),
                1,
                "adjacent cube faces should share exactly 1 edge"
            );
        }
    }

    #[test]
    fn face_wires_returns_outer() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let faces = solid_faces(&topo, cube).unwrap();
        let wires = face_wires(&topo, faces[0]).unwrap();

        // No inner wires on a cube face.
        assert_eq!(wires.len(), 1, "cube face should have 1 wire (outer only)");
    }
}
