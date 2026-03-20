//! Segment-to-segment minimum distance.
//!
//! Provides [`segment_segment_distance`] using the standard closed-form
//! algorithm (Shoemake / Ericson "Real-Time Collision Detection" §5.1).

use brepkit_math::vec::Point3;

/// Compute the minimum distance between two line segments `(a0, a1)` and
/// `(b0, b1)`.
///
/// Returns `(distance, closest_point_on_a, closest_point_on_b)`.
///
/// The algorithm handles all degenerate cases (one or both segments collapse
/// to a point, parallel segments) without `panic` or division by zero.
///
/// # Examples
///
/// ```
/// use brepkit_math::vec::Point3;
/// use brepkit_geometry::extrema::segment_segment_distance;
///
/// // Perpendicular segments in 3-D; closest approach is at (0,0,0) on both.
/// let (dist, pa, pb) = segment_segment_distance(
///     Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0),
///     Point3::new(0.0, -1.0, 1.0), Point3::new(0.0, 1.0, 1.0),
/// );
/// assert!((dist - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn segment_segment_distance(
    a0: Point3,
    a1: Point3,
    b0: Point3,
    b1: Point3,
) -> (f64, Point3, Point3) {
    let d1 = a1 - a0; // Direction of segment A
    let d2 = b1 - b0; // Direction of segment B
    let r = a0 - b0;

    let a = d1.dot(d1); // Squared length of A
    let e = d2.dot(d2); // Squared length of B
    let f = d2.dot(r);

    // Both segments degenerate to points.
    if a <= 1e-30 && e <= 1e-30 {
        let diff = a0 - b0;
        let dist = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();
        return (dist, a0, b0);
    }

    let (s, t);
    if a <= 1e-30 {
        // Segment A degenerates to a point; project onto B.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-30 {
            // Segment B degenerates to a point; project onto A.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General non-degenerate case.
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            // If segments are parallel (`denom ≈ 0`), pick `s = 0` arbitrarily.
            s = if denom.abs() > 1e-30 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Compute t for the closest point on segment B given s.
            let t_raw = (b * s + f) / e;
            t = t_raw.clamp(0.0, 1.0);
        }
    }

    // Recompute s if t was clamped (to handle the endpoint boundary cases).
    let s = if a > 1e-30 {
        let b = d1.dot(d2);
        let c = d1.dot(r);
        ((b * t - c) / a).clamp(0.0, 1.0)
    } else {
        s
    };

    let closest_a = a0 + d1 * s;
    let closest_b = b0 + d2 * t;
    let diff = closest_a - closest_b;
    let dist = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();
    (dist, closest_a, closest_b)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // Helper: assert two f64 values are close.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // Helper: 3-D Euclidean distance between two points.
    fn dist3(p: Point3, q: Point3) -> f64 {
        let d = p - q;
        (d.x() * d.x() + d.y() * d.y() + d.z() * d.z()).sqrt()
    }

    #[test]
    fn parallel_segments_same_line() {
        // Co-linear segments with a gap: A=[0,1], B=[3,4] on X-axis.
        let (dist, pa, pb) = segment_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
        );
        assert!(approx_eq(dist, 2.0, 1e-12), "dist={dist}");
        // Closest points: a1=(1,0,0) and b0=(3,0,0).
        assert!(approx_eq(pa.x(), 1.0, 1e-12), "pa.x={}", pa.x());
        assert!(approx_eq(pb.x(), 3.0, 1e-12), "pb.x={}", pb.x());
    }

    #[test]
    fn skew_segments_closest_approach() {
        // Two skew segments: A along X, B along Y at z=1.
        // Closest approach: (0,0,0) on A and (0,0,1) on B → distance = 1.
        let (dist, pa, pb) = segment_segment_distance(
            Point3::new(-2.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, -2.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        );
        assert!(approx_eq(dist, 1.0, 1e-12), "dist={dist}");
        assert!(
            approx_eq(pa.x(), 0.0, 1e-12)
                && approx_eq(pa.y(), 0.0, 1e-12)
                && approx_eq(pa.z(), 0.0, 1e-12),
            "pa={:?}",
            pa
        );
        assert!(
            approx_eq(pb.x(), 0.0, 1e-12)
                && approx_eq(pb.y(), 0.0, 1e-12)
                && approx_eq(pb.z(), 1.0, 1e-12),
            "pb={:?}",
            pb
        );
    }

    #[test]
    fn intersecting_segments_zero_distance() {
        // Two segments that cross at (1,1,0).
        let (dist, pa, pb) = segment_segment_distance(
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        );
        assert!(approx_eq(dist, 0.0, 1e-12), "dist={dist}");
        assert!(dist3(pa, pb) < 1e-12, "pa≠pb: pa={:?} pb={:?}", pa, pb);
    }

    #[test]
    fn degenerate_both_points() {
        // Both segments collapse to points: A=(1,0,0), B=(4,0,0) → distance=3.
        let (dist, pa, pb) = segment_segment_distance(
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
        );
        assert!(approx_eq(dist, 3.0, 1e-12), "dist={dist}");
        assert!(dist3(pa, Point3::new(1.0, 0.0, 0.0)) < 1e-12, "pa={:?}", pa);
        assert!(dist3(pb, Point3::new(4.0, 0.0, 0.0)) < 1e-12, "pb={:?}", pb);
    }

    #[test]
    fn degenerate_a_is_point() {
        // A=(0,0,0), B from (0,2,0) to (0,4,0).
        // Closest point on B to A is (0,2,0), distance=2.
        let (dist, _pa, pb) = segment_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.0, 4.0, 0.0),
        );
        assert!(approx_eq(dist, 2.0, 1e-12), "dist={dist}");
        assert!(dist3(pb, Point3::new(0.0, 2.0, 0.0)) < 1e-12, "pb={:?}", pb);
    }

    #[test]
    fn degenerate_b_is_point() {
        // B=(0,0,3), A from (0,0,0) to (0,0,1).
        // Closest on A is endpoint (0,0,1), distance=2.
        let (dist, pa, _pb) = segment_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(0.0, 0.0, 3.0),
        );
        assert!(approx_eq(dist, 2.0, 1e-12), "dist={dist}");
        assert!(dist3(pa, Point3::new(0.0, 0.0, 1.0)) < 1e-12, "pa={:?}", pa);
    }

    #[test]
    fn closest_points_lie_on_respective_segments() {
        // Property: returned closest points must lie on their respective segments.
        let a0 = Point3::new(1.0, 0.0, 0.0);
        let a1 = Point3::new(5.0, 0.0, 0.0);
        let b0 = Point3::new(3.0, 1.0, 2.0);
        let b1 = Point3::new(3.0, 4.0, 5.0);

        let (dist, pa, pb) = segment_segment_distance(a0, a1, b0, b1);

        // pa must lie on segment A: pa = a0 + s*(a1-a0) for s in [0,1].
        let da = a1 - a0;
        let len_a = (da.x() * da.x() + da.y() * da.y() + da.z() * da.z()).sqrt();
        let pa_a0 = pa - a0;
        let s = (pa_a0.x() * da.x() + pa_a0.y() * da.y() + pa_a0.z() * da.z()) / (len_a * len_a);
        assert!((-1e-12..=1.0 + 1e-12).contains(&s), "s out of [0,1]: s={s}");

        // pb must lie on segment B.
        let db = b1 - b0;
        let len_b = (db.x() * db.x() + db.y() * db.y() + db.z() * db.z()).sqrt();
        let pb_b0 = pb - b0;
        let t = (pb_b0.x() * db.x() + pb_b0.y() * db.y() + pb_b0.z() * db.z()) / (len_b * len_b);
        assert!((-1e-12..=1.0 + 1e-12).contains(&t), "t out of [0,1]: t={t}");

        // dist must equal the distance between pa and pb.
        assert!(
            approx_eq(dist, dist3(pa, pb), 1e-12),
            "dist mismatch: dist={dist} actual={}",
            dist3(pa, pb)
        );
    }
}
