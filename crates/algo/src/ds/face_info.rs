//! Per-face state accumulated during PaveFiller.

use std::collections::BTreeSet;

use brepkit_topology::vertex::VertexId;

use super::pave::PaveBlockId;

/// Classification of edges and vertices relative to a face.
///
/// Populated incrementally by PaveFiller phases (VE, EF, FF).
/// Consumed by the Builder to split faces.
///
/// `BTreeSet` (not `HashSet`) so iteration is deterministic by ID — downstream
/// face splitting in `fill_images_faces` iterates `pave_blocks_in` to build
/// `SectionSource` lists, and HashSet's random iteration order produces
/// different split topologies across runs, which cascades into 100–500×
/// per-iter variance in compound boolean benchmarks. See PR #689 for the
/// other two sites that drove the same nondeterminism.
#[derive(Debug, Clone, Default)]
pub struct FaceInfo {
    /// Pave blocks lying ON the face boundary (original boundary edges, split).
    pub pave_blocks_on: BTreeSet<PaveBlockId>,
    /// Pave blocks lying IN the face interior (from other faces' edges crossing).
    pub pave_blocks_in: BTreeSet<PaveBlockId>,
    /// Section pave blocks from face-face intersection curves.
    pub pave_blocks_sc: BTreeSet<PaveBlockId>,
    /// Vertices on the face boundary.
    pub vertices_on: BTreeSet<VertexId>,
    /// Vertices in the face interior.
    pub vertices_in: BTreeSet<VertexId>,
    /// Vertices from section curves.
    pub vertices_sc: BTreeSet<VertexId>,
}
