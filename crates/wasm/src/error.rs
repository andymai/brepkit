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

// ── Validation newtypes ──────────────────────────────────────────

/// A finite, strictly positive `f64` value.
///
/// Constructed via [`Positive::new`], which validates at creation time.
#[derive(Debug, Clone, Copy)]
pub struct Positive(f64);

impl Positive {
    /// Create a new `Positive` value, or return an error if the value
    /// is NaN, infinite, zero, or negative.
    ///
    /// # Errors
    ///
    /// Returns [`WasmError::InvalidInput`] if `value` is not finite and positive.
    pub fn new(value: f64, name: &str) -> Result<Self, WasmError> {
        validate_positive(value, name)?;
        Ok(Self(value))
    }

    /// Get the inner `f64` value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite `f64` value (not NaN or infinite).
///
/// Constructed via [`Finite::new`], which validates at creation time.
#[derive(Debug, Clone, Copy)]
pub struct Finite(f64);

impl Finite {
    /// Create a new `Finite` value, or return an error if the value
    /// is NaN or infinite.
    ///
    /// # Errors
    ///
    /// Returns [`WasmError::InvalidInput`] if `value` is not finite.
    pub fn new(value: f64, name: &str) -> Result<Self, WasmError> {
        validate_finite(value, name)?;
        Ok(Self(value))
    }

    /// Get the inner `f64` value.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A flat `[x, y, z, ...]` coordinate array parsed into `Vec<Point3>`.
///
/// Constructed via [`CoordArray3::new`], which validates length divisibility.
pub struct CoordArray3(Vec<brepkit_math::vec::Point3>);

impl CoordArray3 {
    /// Parse a flat coordinate array into `Vec<Point3>`.
    ///
    /// # Errors
    ///
    /// Returns [`WasmError::InvalidInput`] if `coords.len()` is not a multiple of 3.
    pub fn new(coords: &[f64]) -> Result<Self, WasmError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            });
        }
        Ok(Self(
            coords
                .chunks_exact(3)
                .map(|c| brepkit_math::vec::Point3::new(c[0], c[1], c[2]))
                .collect(),
        ))
    }

    /// Get the parsed points.
    #[must_use]
    pub fn points(&self) -> &[brepkit_math::vec::Point3] {
        &self.0
    }

    /// Consume and return the inner `Vec<Point3>`.
    #[must_use]
    pub fn into_points(self) -> Vec<brepkit_math::vec::Point3> {
        self.0
    }
}
