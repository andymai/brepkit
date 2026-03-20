//! Uniform parameter-space curve sampling.

use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::Point3;

/// Sample `n` evenly-spaced points in parameter space over `[t_start, t_end]`.
///
/// - `n == 0` returns an empty `Vec`.
/// - `n == 1` returns a single point at `t_start`.
/// - `n >= 2` returns points including both endpoints.
#[must_use]
pub fn sample_uniform<C: ParametricCurve>(
    curve: &C,
    t_start: f64,
    t_end: f64,
    n: usize,
) -> Vec<Point3> {
    sample_uniform_with_params(curve, t_start, t_end, n)
        .into_iter()
        .map(|(_, p)| p)
        .collect()
}

/// Sample `n` evenly-spaced `(t, Point3)` pairs over `[t_start, t_end]`.
///
/// - `n == 0` returns an empty `Vec`.
/// - `n == 1` returns `vec![(t_start, curve(t_start))]`.
/// - `n >= 2` returns pairs including both endpoints.
#[must_use]
pub fn sample_uniform_with_params<C: ParametricCurve>(
    curve: &C,
    t_start: f64,
    t_end: f64,
    n: usize,
) -> Vec<(f64, Point3)> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(t_start, curve.evaluate(t_start))];
    }
    let step = (t_end - t_start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = if i == n - 1 {
                t_end // avoid floating-point overshoot on last point
            } else {
                t_start + i as f64 * step
            };
            (t, curve.evaluate(t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use std::f64::consts::TAU;

    fn unit_circle() -> Circle3D {
        Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap()
    }

    #[test]
    fn zero_samples_returns_empty() {
        let c = unit_circle();
        let pts = sample_uniform(&c, 0.0, TAU, 0);
        assert!(pts.is_empty());
    }

    #[test]
    fn one_sample_returns_start() {
        let c = unit_circle();
        let pts = sample_uniform(&c, 0.0, TAU, 1);
        assert_eq!(pts.len(), 1);
        // t=0 must lie on the unit circle (radius == 1).
        let r =
            (pts[0].x() * pts[0].x() + pts[0].y() * pts[0].y() + pts[0].z() * pts[0].z()).sqrt();
        assert!((r - 1.0).abs() < 1e-12, "point not on unit circle: r={r}");
    }

    #[test]
    fn four_samples_on_unit_circle() {
        let c = unit_circle();
        // Per task spec: sample_uniform on Circle3D(r=1) with n=4 → 4 points.
        // Use the full domain [0, TAU].
        let pairs = sample_uniform_with_params(&c, 0.0, TAU, 4);
        assert_eq!(pairs.len(), 4);

        // All points should lie on the unit circle.
        for (_, p) in &pairs {
            let r = (p.x() * p.x() + p.y() * p.y() + p.z() * p.z()).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "point not on unit circle: r={r}");
        }

        // First and last parameter values must be endpoints.
        assert!((pairs[0].0 - 0.0).abs() < 1e-12);
        assert!((pairs[3].0 - TAU).abs() < 1e-12);

        // First point (t=0) and last point (t=TAU) must coincide (full circle).
        let p0 = pairs[0].1;
        let p3 = pairs[3].1;
        let dist =
            ((p0.x() - p3.x()).powi(2) + (p0.y() - p3.y()).powi(2) + (p0.z() - p3.z()).powi(2))
                .sqrt();
        assert!(
            dist < 1e-12,
            "endpoints should coincide on full circle: dist={dist}"
        );
    }

    #[test]
    fn params_cover_full_range() {
        let c = unit_circle();
        let pairs = sample_uniform_with_params(&c, 0.0, TAU, 5);
        assert_eq!(pairs.len(), 5);
        assert!((pairs[0].0 - 0.0).abs() < 1e-12);
        assert!((pairs[4].0 - TAU).abs() < 1e-12);
    }
}
