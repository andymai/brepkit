//! # brepkit-operations
//!
//! CAD modeling operations: booleans, fillets, chamfers, extrusions,
//! sweeps, and tessellation.
//!
//! This is layer L2, depending on `brepkit-math` and `brepkit-topology`.

pub mod boolean;
pub mod chamfer;
pub mod extrude;
pub mod fillet;
pub mod revolve;
pub mod sweep;
pub mod tessellate;
pub mod transform;

/// Errors from modeling operations.
#[derive(Debug, thiserror::Error)]
pub enum OperationsError {
    /// The input shape is invalid for this operation.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Description of what is wrong.
        reason: String,
    },

    /// The operation produced a non-manifold result.
    #[error("non-manifold result")]
    NonManifoldResult,

    /// A referenced topology entity was not found.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math error occurred during the operation.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),
}
