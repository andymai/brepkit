//! Error types for the heal crate.

/// Errors that can occur during shape healing operations.
#[derive(Debug, thiserror::Error)]
pub enum HealError {
    /// A topology lookup or mutation failed.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math operation failed.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),

    /// Analysis detected an unrecoverable problem.
    #[error("analysis failed: {0}")]
    AnalysisFailed(String),

    /// A fix operation could not be applied.
    #[error("fix failed: {0}")]
    FixFailed(String),

    /// An upgrade operation could not be applied.
    #[error("upgrade failed: {0}")]
    UpgradeFailed(String),

    /// Invalid configuration or parameters.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
