//! Error types for the check crate.

/// Errors from topology algorithm operations.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    /// A referenced topology entity was not found.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math error occurred.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),

    /// Classification could not determine a result.
    #[error("classification failed: {0}")]
    ClassificationFailed(String),

    /// A validation check encountered an internal error.
    #[error("validation error: {0}")]
    ValidationFailed(String),

    /// Numerical integration did not converge.
    #[error("integration did not converge: {0}")]
    IntegrationFailed(String),

    /// Distance computation could not find a result.
    #[error("distance computation failed: {0}")]
    DistanceFailed(String),
}
