//! Shape reference types for the GFA.

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

/// Which boolean argument a shape belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    /// Shape from the first argument (solid A).
    A,
    /// Shape from the second argument (solid B).
    B,
}

/// Reference to a topological shape with its argument rank.
///
/// Used by future interference indexing to look up shapes by rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum ShapeRef {
    /// A vertex from argument A or B.
    Vertex(VertexId, Rank),
    /// An edge from argument A or B.
    Edge(EdgeId, Rank),
    /// A face from argument A or B.
    Face(FaceId, Rank),
}
