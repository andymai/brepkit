//! Pave, `PaveBlock`, and `CommonBlock` — the core GFA edge-splitting types.

use brepkit_topology::arena::{Arena, Id};
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
    /// with blocks from other edges).
    pub common_block: Option<CommonBlockId>,
    /// Child pave blocks created by `update()`. Empty until split.
    pub children: Vec<PaveBlockId>,
    /// Pre-computed shrunk range (valid parameter interval for intersection).
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
    #[must_use]
    pub fn is_split(&self) -> bool {
        !self.children.is_empty()
    }

    /// Returns the parameter range of this block.
    #[must_use]
    pub fn parameter_range(&self) -> (f64, f64) {
        (self.start.parameter, self.end.parameter)
    }

    /// Split this pave block at all extra paves, creating child blocks
    /// in the arena. Returns the IDs of the created children.
    ///
    /// Extra paves are sorted by parameter, deduplicated, and used to
    /// create contiguous child blocks covering the original range.
    pub fn update(&mut self, arena: &mut Arena<Self>) -> Vec<PaveBlockId> {
        if self.extra_paves.is_empty() {
            return Vec::new();
        }

        // Sort extra paves by parameter
        self.extra_paves.sort_by(|a, b| {
            a.parameter
                .partial_cmp(&b.parameter)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate paves that are too close (within tolerance)
        self.extra_paves
            .dedup_by(|a, b| (a.parameter - b.parameter).abs() < 1e-10);

        // Build child blocks: start -> p1 -> p2 -> ... -> end
        let mut children = Vec::new();
        let mut prev_pave = self.start;

        for &pave in &self.extra_paves {
            // Skip paves at the boundaries
            if (pave.parameter - self.start.parameter).abs() < 1e-10
                || (pave.parameter - self.end.parameter).abs() < 1e-10
            {
                continue;
            }

            let child = Self::new(self.original_edge, prev_pave, pave);
            let child_id = arena.alloc(child);
            children.push(child_id);
            prev_pave = pave;
        }

        // Final segment: last pave -> end
        let last_child = Self::new(self.original_edge, prev_pave, self.end);
        let last_id = arena.alloc(last_child);
        children.push(last_id);

        self.children.clone_from(&children);
        children
    }
}

/// Typed handle for a [`CommonBlock`].
pub type CommonBlockId = Id<CommonBlock>;

/// A group of pave blocks from different edges that geometrically coincide.
///
/// When edges from two different solids overlap, their pave blocks are
/// grouped into a common block. The Builder uses this to create a single
/// shared edge in the result, eliminating non-manifold topology.
#[derive(Debug, Clone)]
pub struct CommonBlock {
    /// Pave blocks that share the same geometry (from different edges).
    pub pave_blocks: Vec<PaveBlockId>,
    /// Faces that these pave blocks lie on.
    pub faces: Vec<FaceId>,
}

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
