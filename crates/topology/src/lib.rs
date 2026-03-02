//! # brepkit-topology
//!
//! Boundary representation (B-Rep) topological data structures.
//!
//! This is layer L1, depending only on `brepkit-math`.
//!
//! # Architecture
//!
//! All topological entities are stored in a central [`Arena`] and referenced
//! via typed index handles ([`VertexId`], [`EdgeId`], etc.). This avoids
//! reference counting overhead and enables efficient traversal.

pub mod arena;
pub mod compound;
pub mod edge;
pub mod face;
pub mod graph;
pub mod shell;
pub mod solid;
#[cfg(feature = "test-utils")]
pub mod test_utils;
pub mod topology;
pub mod validation;
pub mod vertex;
pub mod wire;

pub use arena::Arena;
pub use topology::Topology;

/// Errors from topology operations.
#[derive(Debug, thiserror::Error)]
pub enum TopologyError {
    /// A referenced vertex ID does not exist in the arena.
    #[error("vertex {0:?} not found")]
    VertexNotFound(vertex::VertexId),

    /// A referenced edge ID does not exist in the arena.
    #[error("edge {0:?} not found")]
    EdgeNotFound(edge::EdgeId),

    /// A referenced wire ID does not exist in the arena.
    #[error("wire {0:?} not found")]
    WireNotFound(wire::WireId),

    /// A referenced face ID does not exist in the arena.
    #[error("face {0:?} not found")]
    FaceNotFound(face::FaceId),

    /// A referenced shell ID does not exist in the arena.
    #[error("shell {0:?} not found")]
    ShellNotFound(shell::ShellId),

    /// A referenced solid ID does not exist in the arena.
    #[error("solid {0:?} not found")]
    SolidNotFound(solid::SolidId),

    /// A referenced compound ID does not exist in the arena.
    #[error("compound {0:?} not found")]
    CompoundNotFound(compound::CompoundId),

    /// A wire does not form a closed loop.
    #[error("wire is not closed")]
    WireNotClosed,

    /// The topology is not manifold.
    #[error("non-manifold topology: {reason}")]
    NonManifold {
        /// Description of the manifold violation.
        reason: String,
    },

    /// An empty collection was provided where at least one element is required.
    #[error("empty {entity} — at least one element is required")]
    Empty {
        /// The kind of entity that was empty.
        entity: &'static str,
    },
}
