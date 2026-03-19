//! Per-face state accumulated during PaveFiller.

use std::collections::HashSet;

use brepkit_topology::vertex::VertexId;

use super::pave::PaveBlockId;

/// Classification of edges and vertices relative to a face.
///
/// Populated incrementally by PaveFiller phases (VE, EF, FF).
/// Consumed by the Builder to split faces.
#[derive(Debug, Clone, Default)]
pub struct FaceInfo {
    /// Pave blocks lying ON the face boundary (original boundary edges, split).
    pub pave_blocks_on: HashSet<PaveBlockId>,
    /// Pave blocks lying IN the face interior (from other faces' edges crossing).
    pub pave_blocks_in: HashSet<PaveBlockId>,
    /// Section pave blocks from face-face intersection curves.
    pub pave_blocks_sc: HashSet<PaveBlockId>,
    /// Vertices on the face boundary.
    pub vertices_on: HashSet<VertexId>,
    /// Vertices in the face interior.
    pub vertices_in: HashSet<VertexId>,
    /// Vertices from section curves.
    pub vertices_sc: HashSet<VertexId>,
}

impl FaceInfo {
    /// Creates empty face info.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if this face has any intersection data.
    #[must_use]
    pub fn has_intersections(&self) -> bool {
        !self.pave_blocks_in.is_empty()
            || !self.pave_blocks_sc.is_empty()
            || !self.vertices_in.is_empty()
            || !self.vertices_sc.is_empty()
    }
}
