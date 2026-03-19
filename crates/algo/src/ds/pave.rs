//! Pave, `PaveBlock`, and `CommonBlock` — the core GFA edge-splitting types.

use brepkit_topology::arena::Id;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

/// A point on an edge, identified by its vertex and curve parameter.
#[derive(Debug, Clone, Copy)]
pub struct Pave {
    /// The vertex at this point.
    pub vertex: VertexId,
    /// The parameter on the edge's curve at this point.
    pub parameter: f64,
}

impl Pave {
    /// Creates a new pave.
    #[must_use]
    pub const fn new(vertex: VertexId, parameter: f64) -> Self {
        Self { vertex, parameter }
    }
}

/// Typed handle for a [`PaveBlock`] in the GFA arena.
pub type PaveBlockId = Id<PaveBlock>;

/// An edge segment between two pave points.
///
/// During PaveFiller phases, intersection points are accumulated as
/// `extra_paves`. The `update()` method splits this block into children
/// at those points.
#[derive(Debug, Clone)]
pub struct PaveBlock {
    /// The original (pre-split) edge this block belongs to.
    pub original_edge: EdgeId,
    /// The pave at the start of this segment.
    pub start: Pave,
    /// The pave at the end of this segment.
    pub end: Pave,
    /// Intersection points accumulated during PaveFiller phases.
    /// Sorted by parameter before splitting.
    pub extra_paves: Vec<Pave>,
    /// The topology edge created from this block (populated in `MakeSplitEdges`).
    pub split_edge: Option<EdgeId>,
    /// If this block is part of a common block (geometrically coincident
    /// with blocks from other edges). Populated by future EE overlap detection.
    #[allow(dead_code)]
    pub common_block: Option<CommonBlockId>,
    /// Child pave blocks created by `update()`. Empty until split.
    pub children: Vec<PaveBlockId>,
    /// Pre-computed shrunk range (valid parameter interval for intersection).
    /// Used by future EF/FF phases for containment checks.
    #[allow(dead_code)]
    pub shrunk_range: Option<(f64, f64)>,
}

impl PaveBlock {
    /// Creates a new pave block spanning the full original edge.
    #[must_use]
    pub fn new(original_edge: EdgeId, start: Pave, end: Pave) -> Self {
        Self {
            original_edge,
            start,
            end,
            extra_paves: Vec::new(),
            split_edge: None,
            common_block: None,
            children: Vec::new(),
            shrunk_range: None,
        }
    }

    /// Adds an intersection point to be split on later.
    pub fn add_extra_pave(&mut self, pave: Pave) {
        self.extra_paves.push(pave);
    }

    /// Returns true if this block has been split into children.
    /// Used by future Builder edge-image resolution.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_split(&self) -> bool {
        !self.children.is_empty()
    }

    /// Returns the parameter range of this block.
    #[must_use]
    pub fn parameter_range(&self) -> (f64, f64) {
        (self.start.parameter, self.end.parameter)
    }
}

/// Typed handle for a [`CommonBlock`].
pub type CommonBlockId = Id<CommonBlock>;

/// A group of pave blocks from different edges that geometrically coincide.
///
/// When edges from two different solids overlap, their pave blocks are
/// grouped into a common block. The Builder uses this to create a single
/// shared edge in the result, eliminating non-manifold topology.
///
/// Populated by future EE overlap detection in PaveFiller.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CommonBlock {
    /// Pave blocks that share the same geometry (from different edges).
    pub pave_blocks: Vec<PaveBlockId>,
    /// Faces that these pave blocks lie on.
    pub faces: Vec<FaceId>,
}

#[allow(dead_code)]
impl CommonBlock {
    /// Creates a new common block.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pave_blocks: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Adds a pave block to this common block.
    pub fn add_pave_block(&mut self, pb: PaveBlockId) {
        self.pave_blocks.push(pb);
    }

    /// Adds a face to this common block.
    pub fn add_face(&mut self, face: FaceId) {
        if !self.faces.contains(&face) {
            self.faces.push(face);
        }
    }
}

impl Default for CommonBlock {
    fn default() -> Self {
        Self::new()
    }
}
