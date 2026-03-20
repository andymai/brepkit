//! Curvature-adaptive curve sampling for NURBS curves.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;

/// Maximum recursion depth to prevent infinite subdivision on degenerate curves.
const MAX_DEPTH: u32 = 20;

/// Compute the curvature κ at parameter `t` on a NURBS curve.
///
/// κ = |C' × C''| / |C'|³
///
/// Returns `0.0` when the first derivative is near-zero (degenerate).
fn curvature_at(curve: &NurbsCurve, t: f64) -> f64 {
    let ders = curve.derivatives(t, 2);
    if ders.len() < 3 {
        return 0.0;
    }
    let cp = ders[1]; // C'(t)
    let cpp = ders[2]; // C''(t)
    let cp_len = cp.length();
    if cp_len < f64::EPSILON {
        return 0.0;
    }
    let cross = cp.cross(cpp);
    cross.length() / (cp_len * cp_len * cp_len)
}

/// Chord length between two `Point3` values.
fn chord(a: Point3, b: Point3) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let dz = b.z() - a.z();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Recursively subdivide `[t_a, t_b]` if the curvature×interval estimate
/// exceeds `tolerance`.
///
/// `p_a` and `p_b` are pre-evaluated curve points at the interval endpoints.
/// New interior points (excluding `p_a`) are appended to `out`.
#[allow(clippy::too_many_arguments)]
fn subdivide(
    curve: &NurbsCurve,
    t_a: f64,
    p_a: Point3,
    t_b: f64,
    p_b: Point3,
    tolerance: f64,
    depth: u32,
    out: &mut Vec<(f64, Point3)>,
) {
    if depth >= MAX_DEPTH {
        return;
    }

    // Approximate arc-length of this interval by the chord.
    let interval_len = chord(p_a, p_b);

    // Average curvature at the two endpoints.
    let kappa_a = curvature_at(curve, t_a);
    let kappa_b = curvature_at(curve, t_b);
    let kappa_avg = 0.5 * (kappa_a + kappa_b);

    if kappa_avg * interval_len <= tolerance {
        // Angular change is within tolerance — no further subdivision needed.
        return;
    }

    let t_m = 0.5 * (t_a + t_b);
    let p_m = curve.evaluate(t_m);

    // Left half
    subdivide(curve, t_a, p_a, t_m, p_m, tolerance, depth + 1, out);
    out.push((t_m, p_m));
    // Right half
    subdivide(curve, t_m, p_m, t_b, p_b, tolerance, depth + 1, out);
}

/// Curvature-adaptive sampling for NURBS curves.
///
/// Subdivides intervals where the product of curvature and interval arc-length
/// exceeds `tolerance` (roughly: angular change per segment ≤ tolerance).
///
/// Always returns at least the two endpoints. If `tolerance` is non-positive,
/// only the two endpoints are returned.
#[must_use]
pub fn sample_curvature(
    curve: &NurbsCurve,
    t_start: f64,
    t_end: f64,
    tolerance: f64,
) -> Vec<(f64, Point3)> {
    let p_start = curve.evaluate(t_start);
    let p_end = curve.evaluate(t_end);

    let mut out = Vec::new();
    out.push((t_start, p_start));

    if tolerance > 0.0 {
        subdivide(
            curve, t_start, p_start, t_end, p_end, tolerance, 0, &mut out,
        );
    }

    out.push((t_end, p_end));
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Point3;

    /// Cubic Bezier with varying curvature: tightly curved near t=0, flatter near t=1.
    /// Control polygon: (0,0,0) → (0.1, 1, 0) → (0.9, 1, 0) → (4, 0, 0)
    fn varying_curvature_bezier() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.1, 1.0, 0.0),
                Point3::new(0.9, 1.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid bezier")
    }

    /// Quarter circle as rational NURBS degree 2.
    fn quarter_circle_nurbs() -> NurbsCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, w, 1.0],
        )
        .expect("valid quarter circle")
    }

    #[test]
    fn endpoints_always_included() {
        let c = varying_curvature_bezier();
        let pts = sample_curvature(&c, 0.0, 1.0, 0.1);
        assert!(!pts.is_empty());
        assert!((pts.first().unwrap().0 - 0.0).abs() < 1e-12);
        assert!((pts.last().unwrap().0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn non_positive_tolerance_returns_two_endpoints() {
        let c = varying_curvature_bezier();
        let pts_zero = sample_curvature(&c, 0.0, 1.0, 0.0);
        assert_eq!(pts_zero.len(), 2);
        let pts_neg = sample_curvature(&c, 0.0, 1.0, -1.0);
        assert_eq!(pts_neg.len(), 2);
    }

    #[test]
    fn parameters_sorted() {
        let c = varying_curvature_bezier();
        let pts = sample_curvature(&c, 0.0, 1.0, 0.05);
        for w in pts.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "parameters not sorted: {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    #[test]
    fn high_curvature_produces_more_points_than_low() {
        // Tight bezier (control points near each other) → high curvature at interior.
        let tight = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.1, 1.0, 0.0),
                Point3::new(0.1, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid");

        // Flat bezier (nearly linear).
        let flat = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.01, 0.0),
                Point3::new(2.0, 0.01, 0.0),
                Point3::new(3.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid");

        let tol = 0.1;
        let pts_tight = sample_curvature(&tight, 0.0, 1.0, tol);
        let pts_flat = sample_curvature(&flat, 0.0, 1.0, tol);

        assert!(
            pts_tight.len() > pts_flat.len(),
            "expected more points for tight curve ({}) than flat curve ({})",
            pts_tight.len(),
            pts_flat.len()
        );
    }

    #[test]
    fn quarter_circle_sample_on_unit_circle() {
        // Verify all sampled points lie on the unit circle.
        let c = quarter_circle_nurbs();
        let pts = sample_curvature(&c, 0.0, 1.0, 0.05);
        assert!(pts.len() >= 2);
        for (_, p) in &pts {
            let r = (p.x() * p.x() + p.y() * p.y() + p.z() * p.z()).sqrt();
            assert!((r - 1.0).abs() < 1e-6, "point not on unit circle: r={r:.8}");
        }
    }
}
