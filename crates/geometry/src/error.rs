//! Error types for brepkit-geometry.

/// Errors produced by geometry algorithms.
#[derive(Debug, thiserror::Error)]
pub enum GeomError {
    /// Propagated error from brepkit-math.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),
    /// Input geometry is degenerate (e.g. zero-length curve, collapsed surface).
    #[error("degenerate input: {0}")]
    DegenerateInput(String),
    /// Iterative solver did not converge within the allotted iterations.
    #[error("convergence failure after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations attempted before giving up.
        iterations: usize,
    },
    /// A collection required to be non-empty was empty.
    #[error("empty input")]
    EmptyInput,
}
