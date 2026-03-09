//! Tolerance model for geometric comparisons.
//!
//! CAD kernels need well-defined tolerances for classifying geometric
//! relationships. [`Tolerance`] bundles linear, angular, and relative
//! thresholds.
//!
//! The [`approx_eq`](Tolerance::approx_eq) comparison is *scale-aware*:
//! two values are equal when `|a - b| <= max(linear, relative * max(|a|, |b|))`.
//! This prevents false negatives when comparing large coordinates.

/// Tolerance thresholds for geometric comparisons.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tolerance {
    /// Absolute tolerance for linear (distance) comparisons.
    pub linear: f64,
    /// Absolute tolerance for angular (radian) comparisons.
    pub angular: f64,
    /// Relative tolerance as a fraction of the larger operand.
    ///
    /// Used by [`approx_eq`](Self::approx_eq) to scale comparisons:
    /// `|a - b| <= max(linear, relative * max(|a|, |b|))`.
    pub relative: f64,
}

impl Tolerance {
    /// Sensible defaults for CAD geometry.
    ///
    /// Linear: 1e-7, angular: 1e-12, relative: 1e-10.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            linear: 1e-7,
            angular: 1e-12,
            relative: 1e-10,
        }
    }

    /// A looser tolerance for visualization or rough checks.
    ///
    /// Linear: 1e-4, angular: 1e-8, relative: 1e-6.
    #[must_use]
    pub const fn loose() -> Self {
        Self {
            linear: 1e-4,
            angular: 1e-8,
            relative: 1e-6,
        }
    }

    /// A tighter tolerance for high-precision operations.
    ///
    /// Linear: 1e-10, angular: 1e-15, relative: 1e-14.
    #[must_use]
    pub const fn tight() -> Self {
        Self {
            linear: 1e-10,
            angular: 1e-15,
            relative: 1e-14,
        }
    }

    /// Scale-aware approximate equality.
    ///
    /// Returns `true` when `|a - b| <= max(linear, relative * max(|a|, |b|))`.
    /// This ensures comparisons remain meaningful at any coordinate magnitude.
    #[must_use]
    pub fn approx_eq(self, a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        diff <= self.linear.max(self.relative * scale)
    }

    /// Purely absolute approximate equality (ignores `relative`).
    ///
    /// Use this when comparing values that are *not* coordinates, e.g.
    /// parameter-space values in `[0, 1]` where relative scaling is wrong.
    #[must_use]
    pub fn approx_eq_abs(self, a: f64, b: f64) -> bool {
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn default_tolerance() {
        let tol = Tolerance::new();
        assert!((tol.linear - 1e-7).abs() < 1e-20);
        assert!((tol.angular - 1e-12).abs() < 1e-20);
        assert!((tol.relative - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn loose_tolerance() {
        let tol = Tolerance::loose();
        assert!(tol.linear > Tolerance::new().linear);
        assert!(tol.relative > Tolerance::new().relative);
    }

    #[test]
    fn tight_tolerance() {
        let tol = Tolerance::tight();
        assert!(tol.linear < Tolerance::new().linear);
        assert!(tol.relative < Tolerance::new().relative);
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
    fn approx_eq_scales_with_magnitude() {
        let tol = Tolerance {
            linear: 1e-7,
            angular: 1e-12,
            relative: 1e-4, // 0.01%
        };
        // Near zero: absolute tolerance dominates
        assert!(tol.approx_eq(0.0, 1e-8));
        assert!(!tol.approx_eq(0.0, 1e-3));

        // Large values: relative tolerance dominates
        // 1e6 * 1e-4 = 100 → differences up to 100 are within tolerance
        assert!(tol.approx_eq(1e6, 1e6 + 50.0));
        assert!(!tol.approx_eq(1e6, 1e6 + 200.0));
    }

    #[test]
    fn approx_eq_abs_ignores_relative() {
        let tol = Tolerance {
            linear: 1e-7,
            angular: 1e-12,
            relative: 1e-4,
        };
        // Even at large scale, abs only uses linear
        assert!(!tol.approx_eq_abs(1e6, 1e6 + 1.0));
        assert!(tol.approx_eq_abs(1e6, 1e6 + 1e-8));
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Tolerance::default(), Tolerance::new());
    }
}
