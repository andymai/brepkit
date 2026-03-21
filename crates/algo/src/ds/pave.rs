//! Pave and `PaveBlock` — the core GFA edge-splitting types.

use brepkit_topology::arena::Id;
use brepkit_topology::edge::EdgeId;
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
    /// Child pave blocks created by `update()`. Empty until split.
    pub children: Vec<PaveBlockId>,
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
            children: Vec::new(),
        }
    }

    /// Adds an intersection point to be split on later.
    pub fn add_extra_pave(&mut self, pave: Pave) {
        self.extra_paves.push(pave);
    }

    /// Returns the parameter range of this block.
    #[must_use]
    pub fn parameter_range(&self) -> (f64, f64) {
        (self.start.parameter, self.end.parameter)
    }
}
