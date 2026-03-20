//! Adaptive deflection-based curve sampling via recursive midpoint subdivision.

use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};

/// Maximum recursion depth to guard against degenerate curves.
const MAX_DEPTH: u32 = 20;

/// Compute the perpendicular distance from point `p` to the chord `a → b`.
///
/// Uses the formula `|(p - a) × (b - a)| / |b - a|`.
/// Returns `0.0` when `a` and `b` coincide (degenerate chord).
fn chord_deviation(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab: Vec3 = b - a;
    let ab_len = ab.length();
    if ab_len < f64::EPSILON {
        return 0.0;
    }
    let ap: Vec3 = p - a;
    ap.cross(ab).length() / ab_len
}

/// Recursively subdivide the interval `[t_a, t_b]` until the chord deviation
/// at the midpoint is below `max_deflection`.
///
/// `p_a` and `p_b` are the already-evaluated curve points at `t_a` and `t_b`.
/// New points are appended to `out` (excluding `p_a`; `p_b` is added by the
/// outermost caller after all recursion completes).
#[allow(clippy::too_many_arguments)]
fn subdivide<C: ParametricCurve>(
    curve: &C,
    t_a: f64,
    p_a: Point3,
    t_b: f64,
    p_b: Point3,
    max_deflection: f64,
    depth: u32,
    out: &mut Vec<(f64, Point3)>,
) {
    if depth >= MAX_DEPTH {
        return;
    }
    let t_m = 0.5 * (t_a + t_b);
    let p_m = curve.evaluate(t_m);

    if chord_deviation(p_m, p_a, p_b) <= max_deflection {
        // Chord is within tolerance — no need to subdivide further.
        return;
    }

    // Left half
    subdivide(curve, t_a, p_a, t_m, p_m, max_deflection, depth + 1, out);
    out.push((t_m, p_m));
    // Right half
    subdivide(curve, t_m, p_m, t_b, p_b, max_deflection, depth + 1, out);
}

/// Adaptively sample a curve so that every chord's midpoint deviation is
/// below `max_deflection`.
///
/// Returns `(t, Point3)` pairs sorted by increasing `t`, always including
/// the endpoints `t_start` and `t_end`.
///
/// If `max_deflection` is non-positive, the function returns only the two
/// endpoints (no subdivision).
#[must_use]
pub fn sample_deflection<C: ParametricCurve>(
    curve: &C,
    t_start: f64,
    t_end: f64,
    max_deflection: f64,
) -> Vec<(f64, Point3)> {
    let p_start = curve.evaluate(t_start);
    let p_end = curve.evaluate(t_end);

    let mut out = Vec::new();
    out.push((t_start, p_start));

    if max_deflection > 0.0 {
        subdivide(
            curve,
            t_start,
            p_start,
            t_end,
            p_end,
            max_deflection,
            0,
            &mut out,
        );
    }

    out.push((t_end, p_end));
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use std::f64::consts::TAU;

    fn circle(radius: f64) -> Circle3D {
        Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap()
    }

    #[test]
    fn deflection_circle_r10_many_points() {
        let c = circle(10.0);
        let max_dev = 0.01;
        let pairs = sample_deflection(&c, 0.0, TAU, max_dev);

        // Should produce many points to satisfy the tight tolerance on a large circle.
        assert!(
            pairs.len() > 10,
            "expected many points, got {}",
            pairs.len()
        );
    }

    #[test]
    fn deflection_every_midpoint_within_tolerance() {
        let c = circle(10.0);
        let max_dev = 0.01;
        let pairs = sample_deflection(&c, 0.0, TAU, max_dev);

        // For every consecutive pair, verify the midpoint chord deviation is ≤ max_dev.
        for window in pairs.windows(2) {
            let (t_a, p_a) = window[0];
            let (t_b, p_b) = window[1];
            let t_m = 0.5 * (t_a + t_b);
            let p_m = c.evaluate(t_m);
            let dev = chord_deviation(p_m, p_a, p_b);
            assert!(
                dev <= max_dev + 1e-12,
                "chord deviation {dev} exceeds max {max_dev} between t={t_a} and t={t_b}"
            );
        }
    }

    #[test]
    fn deflection_endpoints_always_included() {
        let c = circle(1.0);
        let pairs = sample_deflection(&c, 0.0, TAU, 0.1);
        assert!(!pairs.is_empty());
        assert!((pairs.first().unwrap().0 - 0.0).abs() < 1e-12);
        assert!((pairs.last().unwrap().0 - TAU).abs() < 1e-12);
    }

    #[test]
    fn non_positive_deflection_returns_two_endpoints() {
        let c = circle(1.0);
        let pairs = sample_deflection(&c, 0.0, TAU, 0.0);
        assert_eq!(pairs.len(), 2);
        let pairs_neg = sample_deflection(&c, 0.0, TAU, -1.0);
        assert_eq!(pairs_neg.len(), 2);
    }

    #[test]
    fn points_sorted_by_parameter() {
        let c = circle(5.0);
        let pairs = sample_deflection(&c, 0.0, TAU, 0.05);
        for w in pairs.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "parameters not sorted: {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }
}
