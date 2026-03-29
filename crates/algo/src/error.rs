//! Error types for the algo crate.

/// Errors from GFA algorithm operations.
#[derive(Debug, thiserror::Error)]
pub enum AlgoError {
    /// A topology entity was not found in the arena.
    #[error("topology error: {0}")]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math operation failed.
    #[error("math error: {0}")]
    Math(#[from] brepkit_math::MathError),

    /// Intersection computation failed.
    #[error("intersection failed: {0}")]
    IntersectionFailed(String),

    /// Face splitting produced invalid topology.
    #[error("face splitting failed: {0}")]
    FaceSplitFailed(String),

    /// Shell assembly produced non-manifold result.
    #[error("assembly failed: {0}")]
    AssemblyFailed(String),

    /// Classification could not determine inside/outside state.
    #[error("classification failed: {0}")]
    ClassificationFailed(String),
}
