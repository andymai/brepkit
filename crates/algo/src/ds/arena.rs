//! GFA arena — owns all transient state for a boolean operation.

use std::collections::HashMap;

use brepkit_topology::arena::Arena;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

use super::curve::IntersectionCurveDS;
use super::face_info::FaceInfo;
use super::interference::InterferenceTable;
use super::pave::{CommonBlock, Pave, PaveBlock, PaveBlockId};

/// Owns all transient GFA state for a single boolean operation.
///
/// The PaveFiller reads from `&Topology` and writes to `&mut GfaArena`.
/// Only the Builder's `make_split_edges` phase commits new entities
/// into `&mut Topology`.
#[derive(Debug, Clone)]
pub struct GfaArena {
    /// Arena for pave block allocation.
    pub pave_blocks: Arena<PaveBlock>,
    /// Common blocks (geometrically coincident pave blocks).
    pub common_blocks: Arena<CommonBlock>,
    /// Intersection curves from face-face intersection.
    pub curves: Vec<IntersectionCurveDS>,
    /// Per-face intersection state.
    pub face_info: HashMap<FaceId, FaceInfo>,
    /// All interference records.
    pub interference: InterferenceTable,
    /// Same-domain vertex mapping (original to canonical).
    /// When two vertices are coincident, they map to the same canonical vertex.
    pub same_domain_vertices: HashMap<VertexId, VertexId>,
    /// Per-edge pave blocks (original edge to its pave block IDs).
    pub edge_pave_blocks: HashMap<EdgeId, Vec<PaveBlockId>>,
}

impl GfaArena {
    /// Creates a new empty GFA arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pave_blocks: Arena::new(),
            common_blocks: Arena::new(),
            curves: Vec::new(),
            face_info: HashMap::new(),
            interference: InterferenceTable::default(),
            same_domain_vertices: HashMap::new(),
            edge_pave_blocks: HashMap::new(),
        }
    }

    /// Resolves a vertex to its same-domain canonical vertex.
    /// Returns the input vertex if no SD mapping exists.
    /// Follows chains transitively (e.g. vb→va→vc returns vc).
    #[must_use]
    pub fn resolve_vertex(&self, v: VertexId) -> VertexId {
        let mut current = v;
        loop {
            match self.same_domain_vertices.get(&current).copied() {
                Some(parent) if parent != current => current = parent,
                _ => return current,
            }
        }
    }

    /// Registers two vertices as same-domain (coincident).
    /// Both map to the one with the lower index.
    pub fn merge_vertices(&mut self, v1: VertexId, v2: VertexId) {
        let canonical = if v1.index() <= v2.index() { v1 } else { v2 };
        let other = if canonical == v1 { v2 } else { v1 };
        self.same_domain_vertices.insert(other, canonical);
    }

    /// Gets or creates the `FaceInfo` for the given face.
    pub fn face_info_mut(&mut self, face: FaceId) -> &mut FaceInfo {
        self.face_info.entry(face).or_default()
    }

    /// Gets the `FaceInfo` for the given face, if it exists.
    #[must_use]
    pub fn face_info(&self, face: FaceId) -> Option<&FaceInfo> {
        self.face_info.get(&face)
    }

    /// Initializes a pave block for an edge from its start/end vertices.
    pub fn init_edge_pave_block(
        &mut self,
        edge: EdgeId,
        start_vertex: VertexId,
        start_param: f64,
        end_vertex: VertexId,
        end_param: f64,
    ) -> PaveBlockId {
        let start = Pave::new(start_vertex, start_param);
        let end = Pave::new(end_vertex, end_param);
        let pb = PaveBlock::new(edge, start, end);
        let pb_id = self.pave_blocks.alloc(pb);
        self.edge_pave_blocks.entry(edge).or_default().push(pb_id);
        pb_id
    }
    /// Collect leaf pave blocks (blocks with no children).
    ///
    /// If a block has children, recursively returns their leaves instead.
    pub fn collect_leaf_pave_blocks(&self, pb_ids: &[PaveBlockId]) -> Vec<PaveBlockId> {
        let mut leaves = Vec::new();
        for &pb_id in pb_ids {
            if let Some(pb) = self.pave_blocks.get(pb_id) {
                if pb.children.is_empty() {
                    leaves.push(pb_id);
                } else {
                    leaves.extend(self.collect_leaf_pave_blocks(&pb.children));
                }
            }
        }
        leaves
    }
}

impl Default for GfaArena {
    fn default() -> Self {
        Self::new()
    }
}
