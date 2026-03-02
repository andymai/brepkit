//! Edge — a curve bounded by two vertices.

use brepkit_math::nurbs::curve::NurbsCurve;

use crate::arena;
use crate::vertex::VertexId;

/// Typed handle for an [`Edge`] stored in an [`Arena`](crate::Arena).
pub type EdgeId = arena::Id<Edge>;

/// The geometric curve associated with an edge.
#[derive(Debug, Clone)]
pub enum EdgeCurve {
    /// A straight line segment (geometry is fully determined by the vertices).
    Line,
    /// A NURBS curve defining the edge geometry.
    NurbsCurve(NurbsCurve),
}

/// A topological edge: a curve bounded by a start and end vertex.
///
/// An edge where `start == end` is a closed (degenerate) edge such as
/// a full circle.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The vertex at the start of the edge.
    start: VertexId,
    /// The vertex at the end of the edge.
    end: VertexId,
    /// The geometric curve of the edge.
    curve: EdgeCurve,
}

impl Edge {
    /// Creates a new edge between two vertices with the given curve.
    #[must_use]
    pub const fn new(start: VertexId, end: VertexId, curve: EdgeCurve) -> Self {
        Self { start, end, curve }
    }

    /// Returns the start vertex of this edge.
    #[must_use]
    pub const fn start(&self) -> VertexId {
        self.start
    }

    /// Returns the end vertex of this edge.
    #[must_use]
    pub const fn end(&self) -> VertexId {
        self.end
    }

    /// Returns a reference to the curve geometry of this edge.
    #[must_use]
    pub const fn curve(&self) -> &EdgeCurve {
        &self.curve
    }

    /// Returns `true` if the edge is closed (start equals end).
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.start == self.end
    }

    /// Sets the curve geometry of this edge.
    pub fn set_curve(&mut self, curve: EdgeCurve) {
        self.curve = curve;
    }
}
