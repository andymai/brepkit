//! Topological adjacency graph for face connectivity queries.

use std::collections::HashMap;

use crate::face::FaceId;
use crate::shell::Shell;

/// A precomputed adjacency graph over the faces of a shell.
///
/// Two faces are adjacent if they share at least one edge.
#[derive(Debug, Clone)]
pub struct TopologyGraph {
    /// Map from each face to its list of adjacent faces.
    adjacency: HashMap<usize, Vec<FaceId>>,
}

impl TopologyGraph {
    /// Builds an adjacency graph from a shell.
    ///
    /// # Panics
    ///
    /// This method is not yet implemented.
    #[must_use]
    pub fn from_shell(_shell: &Shell) -> Self {
        // TODO: walk the shell's faces, collect shared edges, and build
        // the adjacency map.
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Returns the faces adjacent to the given face, or an empty slice
    /// if the face is not present in the graph.
    #[must_use]
    pub fn adjacent_faces(&self, face: FaceId) -> &[FaceId] {
        self.adjacency.get(&face.index()).map_or(&[], Vec::as_slice)
    }
}
