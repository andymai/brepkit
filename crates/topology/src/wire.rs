//! Wire — an ordered sequence of oriented edges forming a path or loop.

use crate::TopologyError;
use crate::arena;
use crate::edge::{Edge, EdgeId};
use crate::vertex::VertexId;

/// Typed handle for a [`Wire`] stored in an [`Arena`](crate::Arena).
pub type WireId = arena::Id<Wire>;

/// An edge reference with an orientation flag.
///
/// When `forward` is `true` the edge is traversed from its start to its end.
/// When `false` the traversal is reversed (end to start).
#[derive(Debug, Clone, Copy)]
pub struct OrientedEdge {
    /// The referenced edge.
    edge: EdgeId,
    /// `true` if the edge is traversed in its natural direction.
    forward: bool,
}

impl OrientedEdge {
    /// Creates a new oriented edge reference.
    #[must_use]
    pub const fn new(edge: EdgeId, forward: bool) -> Self {
        Self { edge, forward }
    }

    /// Returns the referenced edge id.
    #[must_use]
    pub const fn edge(&self) -> EdgeId {
        self.edge
    }

    /// Returns `true` if this edge is traversed in its natural direction.
    #[must_use]
    pub const fn is_forward(&self) -> bool {
        self.forward
    }

    /// Returns the vertex at the start of traversal for this oriented edge.
    ///
    /// When traversed forward the start vertex is `edge.start()`; when reversed
    /// it is `edge.end()`.
    #[must_use]
    pub const fn oriented_start(&self, edge: &Edge) -> VertexId {
        if self.forward {
            edge.start()
        } else {
            edge.end()
        }
    }

    /// Returns the vertex at the end of traversal for this oriented edge.
    ///
    /// When traversed forward the end vertex is `edge.end()`; when reversed
    /// it is `edge.start()`.
    #[must_use]
    pub const fn oriented_end(&self, edge: &Edge) -> VertexId {
        if self.forward {
            edge.end()
        } else {
            edge.start()
        }
    }
}

/// A topological wire: an ordered chain of oriented edges.
///
/// A wire must contain at least one edge. It may be open (a path) or
/// closed (a loop).
#[derive(Debug, Clone)]
pub struct Wire {
    /// The ordered sequence of oriented edges.
    edges: Vec<OrientedEdge>,
    /// Whether this wire forms a closed loop.
    closed: bool,
}

impl Wire {
    /// Creates a new wire from a non-empty list of oriented edges.
    ///
    /// The `closed` flag indicates whether the wire forms a closed loop.
    /// Topological validation (e.g. checking that the last edge connects
    /// back to the first) is performed separately via
    /// [`validation::validate_wire_closed`](crate::validation::validate_wire_closed).
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::Empty`] if `edges` is empty.
    pub fn new(edges: Vec<OrientedEdge>, closed: bool) -> Result<Self, TopologyError> {
        if edges.is_empty() {
            return Err(TopologyError::Empty { entity: "wire" });
        }
        Ok(Self { edges, closed })
    }

    /// Returns the ordered edges of this wire.
    #[must_use]
    pub fn edges(&self) -> &[OrientedEdge] {
        &self.edges
    }

    /// Returns mutable access to the ordered edges of this wire.
    ///
    /// Allows in-place mutation (reorder, replace) but not removal.
    /// The wire must always contain at least one edge.
    pub fn edges_mut(&mut self) -> &mut [OrientedEdge] {
        &mut self.edges
    }

    /// Returns `true` if this wire forms a closed loop.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }
}
