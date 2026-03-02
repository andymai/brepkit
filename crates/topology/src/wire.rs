//! Wire — an ordered sequence of oriented edges forming a path or loop.

use crate::TopologyError;
use crate::arena;
use crate::edge::EdgeId;

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

    /// Returns `true` if this wire forms a closed loop.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }
}
