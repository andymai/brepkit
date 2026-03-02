//! Tolerance model for geometric comparisons.
//!
//! CAD kernels need well-defined tolerances for classifying geometric
//! relationships. [`Tolerance`] bundles linear and angular thresholds.

/// Tolerance thresholds for geometric comparisons.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tolerance {
    /// Absolute tolerance for linear (distance) comparisons.
    pub linear: f64,
    /// Absolute tolerance for angular (radian) comparisons.
    pub angular: f64,
}

impl Tolerance {
    /// Sensible defaults for CAD geometry: 1e-7 linear, 1e-12 angular.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            linear: 1e-7,
            angular: 1e-12,
        }
    }

    /// A looser tolerance for visualization or rough checks.
    ///
    /// Linear: 1e-4, angular: 1e-8.
    #[must_use]
    pub const fn loose() -> Self {
        Self {
            linear: 1e-4,
            angular: 1e-8,
        }
    }

    /// A tighter tolerance for high-precision operations.
    ///
    /// Linear: 1e-10, angular: 1e-15.
    #[must_use]
    pub const fn tight() -> Self {
        Self {
            linear: 1e-10,
            angular: 1e-15,
        }
    }

    /// Check whether two `f64` values are approximately equal within the
    /// linear tolerance.
    #[must_use]
    pub fn approx_eq(self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.linear
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::new()
    }
}
