//! Curvature-adaptive curve sampling for NURBS curves.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};

/// Maximum recursion depth to prevent infinite subdivision on degenerate curves.
const MAX_DEPTH: u32 = 20;

/// A curve point with its curvature and unit tangent (`None` where the first
/// derivative vanishes).
#[derive(Clone, Copy)]
struct Probe {
    t: f64,
    point: Point3,
    kappa: f64,
    tangent: Option<Vec3>,
}

/// Probe the curve at `t`: κ = |C' × C''| / |C'|³, and `0.0` where the first
/// derivative is near-zero (degenerate).
fn probe(curve: &NurbsCurve, t: f64) -> Probe {
    let ders = curve.derivatives(t, 2);
    let point = Point3::new(ders[0].x(), ders[0].y(), ders[0].z());
    let (cp, cpp) = (ders[1], ders[2]);
    let cp_len = cp.length();
    if cp_len < f64::EPSILON {
        return Probe {
            t,
            point,
            kappa: 0.0,
            tangent: None,
        };
    }
    Probe {
        t,
        point,
        kappa: cp.cross(cpp).length() / (cp_len * cp_len * cp_len),
        tangent: Some(cp * (1.0 / cp_len)),
    }
}

/// Probe the piece of the curve below `t`, which at a knot differs from the
/// piece the evaluator reads there (the one above it).
fn probe_below(curve: &NurbsCurve, t: f64) -> Probe {
    Probe {
        t,
        point: curve.evaluate(t),
        ..probe(curve, t.next_down())
    }
}

/// The angle between two probes' tangents, `0.0` if either has none.
fn turn(a: &Probe, b: &Probe) -> f64 {
    match (a.tangent, b.tangent) {
        (Some(ta), Some(tb)) => ta.dot(tb).clamp(-1.0, 1.0).acos(),
        _ => 0.0,
    }
}

/// The turning of the curve through `probes` (in order), estimated two
/// ways: its sharpest curvature among them times the polyline through them
/// (an arc that doubles back has a short chord and can be gentle at both
/// ends), and the turn between their tangents (an S has no curvature at its
/// inflection and can have none at its ends).
fn turning(probes: &[&Probe]) -> f64 {
    let kappa = probes.iter().map(|q| q.kappa).fold(0.0, f64::max);
    let length: f64 = probes
        .windows(2)
        .map(|w| (w[1].point - w[0].point).length())
        .sum();
    let turns: f64 = probes.windows(2).map(|w| turn(w[0], w[1])).sum();
    (kappa * length).max(turns)
}

/// Recursively subdivide the interval between probes `a` and `b` while its
/// turning exceeds `tolerance`, reading its midpoint `m` (probed here unless
/// given). New interior points are appended to `out`.
fn subdivide(
    curve: &NurbsCurve,
    a: &Probe,
    b: &Probe,
    m: Option<Probe>,
    tolerance: f64,
    depth: u32,
    out: &mut Vec<(f64, Point3)>,
) {
    if depth >= MAX_DEPTH {
        return;
    }
    let m = m.unwrap_or_else(|| probe(curve, 0.5 * (a.t + b.t)));
    if turning(&[a, &m, b]) <= tolerance {
        return;
    }
    subdivide(curve, a, &m, None, tolerance, depth + 1, out);
    out.push((m.t, m.point));
    subdivide(curve, &m, b, None, tolerance, depth + 1, out);
}

/// Sample one knot span (one polynomial piece) between probes `a` and `b`,
/// reading it at its midpoint and quarter points before keeping it as one
/// segment.
fn sample_span(
    curve: &NurbsCurve,
    a: &Probe,
    b: &Probe,
    tolerance: f64,
    out: &mut Vec<(f64, Point3)>,
) {
    let m = probe(curve, 0.5 * (a.t + b.t));
    let q1 = probe(curve, 0.5 * (a.t + m.t));
    let q3 = probe(curve, 0.5 * (m.t + b.t));
    if turning(&[a, &q1, &m, &q3, b]) <= tolerance {
        return;
    }
    subdivide(curve, a, &m, Some(q1), tolerance, 1, out);
    out.push((m.t, m.point));
    subdivide(curve, &m, b, Some(q3), tolerance, 1, out);
}

/// Curvature-adaptive sampling for NURBS curves.
///
/// Splits the range at the curve's knots, where its tangent or curvature may
/// jump, and subdivides every interval whose estimated turning exceeds
/// `tolerance` (roughly: angular change per segment ≤ tolerance). Each span
/// is one polynomial piece, read at its ends, midpoint and quarter points at
/// least: a non-rational cubic piece with no curvature at all five is
/// straight.
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
    let mut out = vec![(t_start, curve.evaluate(t_start))];
    let (lo, hi) = (t_start.min(t_end), t_start.max(t_end));
    if tolerance <= 0.0 || hi <= lo {
        out.push((t_end, curve.evaluate(t_end)));
        return out;
    }

    let mut cuts = vec![lo];
    for &k in curve.knots() {
        if k - cuts[cuts.len() - 1] >= f64::EPSILON && hi - k >= f64::EPSILON {
            cuts.push(k);
        }
    }
    cuts.push(hi);
    let mut spans: Vec<(Probe, Probe)> = cuts
        .windows(2)
        .map(|w| (probe(curve, w[0]), probe_below(curve, w[1])))
        .collect();
    if t_start > t_end {
        spans = spans.into_iter().rev().map(|(a, b)| (b, a)).collect();
    }
    for (a, b) in &spans {
        sample_span(curve, a, b, tolerance, &mut out);
        out.push((b.t, b.point));
    }
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

    fn planar(degree: usize, knots: Vec<f64>, pts: &[(f64, f64)]) -> NurbsCurve {
        let n = pts.len();
        NurbsCurve::new(
            degree,
            knots,
            pts.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect(),
            vec![1.0; n],
        )
        .expect("valid curve")
    }

    /// The largest angle the tangent turns within any sampled segment, read
    /// at 64 points per segment short of its far end (at a knot the
    /// evaluator reads the next piece there).
    fn max_turn_per_segment(c: &NurbsCurve, pts: &[(f64, Point3)]) -> f64 {
        let mut worst = 0.0_f64;
        for w in pts.windows(2) {
            let (a, b) = (w[0].0, w[1].0);
            let ta = c.tangent(a).unwrap();
            for i in 1..64 {
                let t = a + (b - a) * f64::from(i) / 64.0;
                let turn = ta.dot(c.tangent(t).unwrap()).clamp(-1.0, 1.0).acos();
                worst = worst.max(turn);
            }
        }
        worst
    }

    /// Curves whose curvature vanishes at their ends and midpoint (an ogee
    /// over four spans, a symmetric quintic S, a wave over eight spans), a
    /// closed circle, and a V with a corner at a double knot turn by at most
    /// the tolerance within each segment. The V's two straight pieces are a
    /// segment each, however sharp its corner.
    #[test]
    fn segments_turn_within_tolerance() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let circle = NurbsCurve::new(
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            [
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 0.0),
                (-1.0, -1.0),
                (0.0, -1.0),
                (1.0, -1.0),
                (1.0, 0.0),
            ]
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect(),
            vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0],
        )
        .expect("valid circle");
        let cubic = vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let ogee = planar(
            3,
            cubic,
            &[
                (-3.0, -1.0),
                (-2.0, -1.0),
                (-1.0, -1.0),
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (3.0, 1.0),
            ],
        );
        let quintic = planar(
            5,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[
                (-3.0, -1.0),
                (-2.0, -1.0),
                (-1.0, -1.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (3.0, 1.0),
            ],
        );
        let mut knots = vec![0.0; 4];
        knots.extend((1..8).map(|k| f64::from(k) / 8.0));
        knots.extend([1.0; 4]);
        let wave = planar(
            3,
            knots,
            &[
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 1.0),
                (4.0, 0.0),
                (5.0, 0.0),
                (6.0, 0.0),
                (7.0, -1.0),
                (8.0, 0.0),
                (9.0, 0.0),
                (10.0, 0.0),
            ],
        );
        let vee = planar(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            &[(0.0, 1.0), (0.8, 0.2), (1.0, 0.0), (1.2, 0.2), (2.0, 1.0)],
        );
        for tol in [0.1, 0.02] {
            for (name, c) in [
                ("ogee", &ogee),
                ("quintic", &quintic),
                ("wave", &wave),
                ("circle", &circle),
                ("vee", &vee),
            ] {
                let pts = sample_curvature(c, 0.0, 1.0, tol);
                let turn = max_turn_per_segment(c, &pts);
                assert!(
                    turn <= tol,
                    "{name} at {tol}: {turn} over {} points",
                    pts.len()
                );
            }
        }
        assert_eq!(sample_curvature(&vee, 0.0, 1.0, 0.02).len(), 3);

        // A range walked backwards gives the same points in reverse, and one
        // bounded by knots starts and ends on them.
        let forward = sample_curvature(&wave, 0.0, 1.0, 0.05);
        let backward = sample_curvature(&wave, 1.0, 0.0, 0.05);
        assert_eq!(forward.len(), backward.len());
        for (f, b) in forward.iter().zip(backward.iter().rev()) {
            assert!((f.0 - b.0).abs() < 1e-15, "{} vs {}", f.0, b.0);
        }
        let inner = sample_curvature(&wave, 0.25, 0.75, 0.05);
        assert!(
            (inner[0].0 - 0.25).abs() < 1e-15 && (inner[inner.len() - 1].0 - 0.75).abs() < 1e-15
        );
        assert!(inner.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(max_turn_per_segment(&wave, &inner) <= 0.05);
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
