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

    /// Convert the linear tolerance to parameter space given the magnitude
    /// of a surface derivative (or curve tangent).
    ///
    /// The parametric tolerance is `linear / derivative_magnitude`, so a
    /// surface with `||∂S/∂u|| = 1000` and linear tolerance 1e-7 gives
    /// parametric tolerance 1e-10.
    ///
    /// Clamps to `[1e-15, 0.1]` to prevent degeneracy.
    #[must_use]
    pub fn parametric(self, derivative_mag: f64) -> f64 {
        if derivative_mag < 1e-30 {
            return self.linear;
        }
        (self.linear / derivative_mag).clamp(1e-15, 0.1)
    }

    /// A squared linear tolerance, useful for distance² comparisons
    /// that avoid the `sqrt` call.
    #[must_use]
    pub fn linear_sq(self) -> f64 {
        self.linear * self.linear
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tolerance() {
        let tol = Tolerance::new();
        assert!((tol.linear - 1e-7).abs() < 1e-20);
        assert!((tol.angular - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn loose_tolerance() {
        let tol = Tolerance::loose();
        assert!(tol.linear > Tolerance::new().linear);
    }

    #[test]
    fn tight_tolerance() {
        let tol = Tolerance::tight();
        assert!(tol.linear < Tolerance::new().linear);
    }

    #[test]
    fn approx_eq_within_tolerance() {
        let tol = Tolerance::new();
        assert!(tol.approx_eq(1.0, 1.0 + 1e-8));
        assert!(tol.approx_eq(0.0, 1e-8));
    }

    #[test]
    fn approx_eq_outside_tolerance() {
        let tol = Tolerance::new();
        assert!(!tol.approx_eq(1.0, 1.001));
        assert!(!tol.approx_eq(0.0, 0.001));
    }

    #[test]
    fn approx_eq_exact() {
        let tol = Tolerance::new();
        assert!(tol.approx_eq(42.0, 42.0));
        assert!(tol.approx_eq(0.0, 0.0));
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Tolerance::default(), Tolerance::new());
    }
}
