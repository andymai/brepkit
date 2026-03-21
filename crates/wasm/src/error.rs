//! WASM-boundary error types.
//!
//! [`WasmError`] aggregates errors from all lower layers. Because
//! `wasm-bindgen` provides a blanket `impl<E: Error> From<E> for JsError`,
//! any `WasmError` can be converted to `JsError` automatically via `?`.

/// Errors that can occur in WASM-exposed operations.
#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    /// A JS-provided handle index does not correspond to a valid entity.
    #[error("invalid {entity} handle: index {index} is out of bounds")]
    InvalidHandle {
        /// The kind of entity (e.g. "face", "solid").
        entity: &'static str,
        /// The raw index that was provided.
        index: usize,
    },

    /// An input value is invalid (NaN, infinite, out of range, etc.).
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Description of what is wrong.
        reason: String,
    },

    /// An error from a modeling operation.
    #[error(transparent)]
    Operations(#[from] brepkit_operations::OperationsError),

    /// An error from topology lookup.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math error (e.g. singular matrix).
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),
}

/// Validate that a `f64` value is finite (not NaN or infinite).
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is NaN or infinite.
pub fn validate_finite(value: f64, name: &str) -> Result<(), WasmError> {
    value
        .is_finite()
        .then_some(())
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("{name} must be finite, got {value}"),
        })
}

/// Validate that a `f64` value is finite and strictly positive.
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is NaN, infinite, zero,
/// or negative.
pub fn validate_positive(value: f64, name: &str) -> Result<(), WasmError> {
    validate_finite(value, name)?;
    (value > 0.0)
        .then_some(())
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("{name} must be positive, got {value}"),
        })
}
