//! WASM-boundary error types.
//!
//! [`WasmError`] aggregates errors from all lower layers. Because
//! `wasm-bindgen` provides a blanket `impl<E: Error> From<E> for JsError`,
//! any `WasmError` can be converted to `JsError` automatically via `?`.

use brepkit_math::curves::SIZE_LIMIT;

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

/// Validate that a `f64` value is a size the load checks accept.
///
/// The value must be finite and at most [`SIZE_LIMIT`] in magnitude. A
/// curve or surface built with a larger radius, distance or centre
/// coordinate fails `is_well_formed`.
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is NaN, infinite, or
/// larger than [`SIZE_LIMIT`] in magnitude.
pub fn validate_size(value: f64, name: &str) -> Result<(), WasmError> {
    validate_finite(value, name)?;
    (value.abs() <= SIZE_LIMIT)
        .then_some(())
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("{name} must be at most {SIZE_LIMIT:e} in magnitude, got {value:e}"),
        })
}

/// Validate that a `f64` value is strictly positive and a size the load
/// checks accept (see [`validate_size`]).
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is NaN, infinite, zero,
/// negative, or larger than [`SIZE_LIMIT`].
pub fn validate_positive(value: f64, name: &str) -> Result<(), WasmError> {
    validate_size(value, name)?;
    (value > 0.0)
        .then_some(())
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("{name} must be positive, got {value}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound is the load checks' own: a size at the limit passes, one
    /// past it is refused either way round, and so is zero or less where a
    /// positive size is asked for.
    #[test]
    fn a_size_past_the_load_checks_limit_is_refused() {
        assert!(validate_positive(SIZE_LIMIT, "radius").is_ok());
        assert!(validate_size(-SIZE_LIMIT, "distance").is_ok());
        assert!(validate_size(0.0, "distance").is_ok());
        for past in [SIZE_LIMIT * 10.0, f64::MAX] {
            assert!(validate_positive(past, "radius").is_err());
            assert!(validate_size(past, "distance").is_err());
            assert!(validate_size(-past, "distance").is_err());
        }
        assert!(validate_positive(0.0, "radius").is_err());
    }
}
