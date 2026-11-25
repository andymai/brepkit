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

/// Validate a tolerance from JS: a size (see [`validate_size`]) that is not
/// below zero.
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is NaN, infinite, below
/// zero, or larger than [`SIZE_LIMIT`].
pub fn validate_tolerance(value: f64, name: &str) -> Result<(), WasmError> {
    validate_size(value, name)?;
    (value >= 0.0)
        .then_some(())
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("{name} must not be below zero, got {value}"),
        })
}

/// Validate each coordinate of a point from JS as a size (see
/// [`validate_size`]).
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if a coordinate is NaN, infinite, or
/// larger than [`SIZE_LIMIT`] in magnitude.
pub fn validate_point(point: brepkit_math::vec::Point3, name: &str) -> Result<(), WasmError> {
    validate_coordinates([point.x(), point.y(), point.z()], name)
}

/// Validate each component of a direction or offset from JS as a size (see
/// [`validate_size`]).
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if a component is NaN, infinite, or
/// larger than [`SIZE_LIMIT`] in magnitude.
pub fn validate_vector(vector: brepkit_math::vec::Vec3, name: &str) -> Result<(), WasmError> {
    validate_coordinates([vector.x(), vector.y(), vector.z()], name)
}

fn validate_coordinates(coordinates: [f64; 3], name: &str) -> Result<(), WasmError> {
    for (axis, c) in ["x", "y", "z"].into_iter().zip(coordinates) {
        validate_size(c, &format!("{name} {axis}"))?;
    }
    Ok(())
}

/// The most iterations a solver or fit takes from JS.
///
/// Each iteration is a pass over the whole system, so a count of
/// `u32::MAX` holds the instance for hours. The counts in the tests and in
/// brepjs are at most 200; this limit is fifty times that.
pub const MAX_ITERATIONS: u32 = 10_000;

/// The largest count a binding takes from JS for the points, segments or
/// sides along one curve, polygon or sweep, or the copies of a pattern.
///
/// A count sizes an array before anything is built, so a count of
/// `u32::MAX` traps the instance when the allocation fails. The counts in
/// the tests and in brepjs are at most 32; this limit is two thousand times
/// that.
pub const MAX_COUNT: u32 = 1 << 16;

/// The largest side of a square grid of samples a binding takes from JS.
///
/// Such a grid holds [`MAX_COUNT`] samples. The sides in the tests and in
/// brepjs are at most 16.
pub const MAX_GRID_SIDE: u32 = 1 << 8;

/// Validate a count from JS against its limit `max` before it sizes an
/// allocation, and give it as a `usize`.
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if `value` is larger than `max`.
pub fn validate_count(value: u32, name: &str, max: u32) -> Result<usize, WasmError> {
    if value > max {
        return Err(WasmError::InvalidInput {
            reason: format!("{name} must be at most {max}, got {value}"),
        });
    }
    usize::try_from(value).map_err(|_| WasmError::InvalidInput {
        reason: format!("{name} {value} does not fit this target"),
    })
}

/// Validate the counts of a pattern from JS.
///
/// The copies it makes, the product of its counts, are at most
/// [`MAX_COUNT`]. Each copy is a whole solid, and the copies' handles are
/// gathered in one array first.
///
/// # Errors
///
/// Returns [`WasmError::InvalidInput`] if the product of `counts` is larger
/// than [`MAX_COUNT`].
pub fn validate_pattern_counts(counts: &[u32]) -> Result<(), WasmError> {
    let copies = counts
        .iter()
        .try_fold(1_u64, |acc, &c| acc.checked_mul(u64::from(c)));
    match copies {
        Some(n) if n <= u64::from(MAX_COUNT) => Ok(()),
        _ => Err(WasmError::InvalidInput {
            reason: format!("a pattern makes at most {MAX_COUNT} copies, got counts {counts:?}"),
        }),
    }
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
