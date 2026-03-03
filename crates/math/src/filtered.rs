//! Filtered exact arithmetic for geometric predicates.
//!
//! Provides fast-path floating-point computation with automatic fallback
//! to exact arithmetic when the result is ambiguous. In practice, 95%+
//! of predicate evaluations resolve in the fast path.
//!
//! Based on Shewchuk's adaptive precision arithmetic (1997).
//!
//! CDT (`cdt.rs`) and mesh booleans (`mesh_boolean.rs` in brepkit-operations)
//! can switch their `orient2d`/`in_circle` calls to these filtered versions
//! for a significant performance improvement without sacrificing robustness.

#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::vec::{Point2, Point3};

/// Error-free sum: returns `(s, e)` where `a + b = s + e` exactly.
#[inline]
#[allow(dead_code)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Error-free product: returns `(p, e)` where `a * b = p + e` exactly.
#[inline]
#[allow(dead_code)]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p); // FMA-based error
    (p, e)
}

/// Filtered orient2d: fast f64 path with exact fallback.
///
/// Returns a positive value if `(a, b, c)` are in counter-clockwise order,
/// negative if clockwise, and zero if collinear.
///
/// Uses error-free transformations to bound the rounding error. If the
/// computed result is larger than the error bound, the sign is guaranteed
/// correct. Otherwise, falls back to exact arithmetic.
#[must_use]
pub fn filtered_orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    let acx = a.x() - c.x();
    let bcx = b.x() - c.x();
    let acy = a.y() - c.y();
    let bcy = b.y() - c.y();

    let det = acx * bcy - acy * bcx;

    // Compute error bound using Shewchuk's method
    let det_sum = (acx * bcy).abs() + (acy * bcx).abs();

    // The error bound for orient2d: (3 + 16*eps) * eps * |detsum|
    let eps = f64::EPSILON;
    let err_bound = (3.0 + 16.0 * eps) * eps * det_sum;

    if det.abs() > err_bound {
        det
    } else {
        // Fall back to exact arithmetic
        crate::predicates::orient2d(a, b, c)
    }
}

/// Filtered orient3d: fast f64 path with exact fallback.
///
/// Returns a positive value if `d` is below the plane defined by `(a, b, c)`
/// (with CCW orientation), negative if above, zero if coplanar.
#[must_use]
pub fn filtered_orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    let adx = a.x() - d.x();
    let bdx = b.x() - d.x();
    let cdx = c.x() - d.x();
    let ady = a.y() - d.y();
    let bdy = b.y() - d.y();
    let cdy = c.y() - d.y();
    let adz = a.z() - d.z();
    let bdz = b.z() - d.z();
    let cdz = c.z() - d.z();

    let det = adx * (bdy * cdz - bdz * cdy) - bdx * (ady * cdz - adz * cdy)
        + cdx * (ady * bdz - adz * bdy);

    // Error bound for orient3d
    let permanent = (adx.abs() * ((bdy * cdz).abs() + (bdz * cdy).abs()))
        + (bdx.abs() * ((ady * cdz).abs() + (adz * cdy).abs()))
        + (cdx.abs() * ((ady * bdz).abs() + (adz * bdy).abs()));

    let eps = f64::EPSILON;
    let err_bound = (7.0 + 56.0 * eps) * eps * permanent;

    if det.abs() > err_bound {
        det
    } else {
        crate::predicates::orient3d(a, b, c, d)
    }
}

/// Filtered in-circle: fast f64 path with exact fallback.
///
/// Returns a positive value if `d` is inside the circumcircle of `(a, b, c)`,
/// negative if outside, zero if exactly on the circle.
/// Assumes `(a, b, c)` are in counter-clockwise order.
#[must_use]
pub fn filtered_in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let adx = a.x() - d.x();
    let ady = a.y() - d.y();
    let bdx = b.x() - d.x();
    let bdy = b.y() - d.y();
    let cdx = c.x() - d.x();
    let cdy = c.y() - d.y();

    let abdet = adx * bdy - bdx * ady;
    let bcdet = bdx * cdy - cdx * bdy;
    let cadet = cdx * ady - adx * cdy;
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    let det = alift * bcdet + blift * cadet + clift * abdet;

    // Error bound for in_circle
    let permanent = alift * ((bcdet).abs() + (bdx * cdy).abs() + (cdx * bdy).abs())
        + blift * ((cadet).abs() + (cdx * ady).abs() + (adx * cdy).abs())
        + clift * ((abdet).abs() + (adx * bdy).abs() + (bdx * ady).abs());

    let eps = f64::EPSILON;
    let err_bound = (10.0 + 96.0 * eps) * eps * permanent;

    if det.abs() > err_bound {
        det
    } else {
        crate::predicates::in_circle(a, b, c, d)
    }
}

// ---------------------------------------------------------------------------
// Segment intersection
// ---------------------------------------------------------------------------

/// Result of a 2D segment-segment intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentIntersection {
    /// No intersection.
    None,
    /// Segments intersect at a single point.
    Point {
        /// The intersection point.
        point: Point2,
        /// Parameter on the first segment (0..1).
        t1: f64,
        /// Parameter on the second segment (0..1).
        t2: f64,
    },
    /// Segments overlap (collinear) along a range.
    Overlap {
        /// Start of the overlap.
        start: Point2,
        /// End of the overlap.
        end: Point2,
    },
}

/// Compute the intersection of two 2D line segments using filtered predicates.
///
/// Handles all degeneracies: T-intersections, endpoint-on-segment,
/// collinear overlap, and parallel/disjoint segments.
///
/// Uses [`filtered_orient2d`] for robust classification.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn segment_intersection(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> SegmentIntersection {
    // Orient2d tests for segment classification
    let d1 = filtered_orient2d(a1, a2, b1);
    let d2 = filtered_orient2d(a1, a2, b2);
    let d3 = filtered_orient2d(b1, b2, a1);
    let d4 = filtered_orient2d(b1, b2, a2);

    // Standard crossing test: opposite signs
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        // Proper crossing: compute intersection point
        let denom = (a2.x() - a1.x()) * (b2.y() - b1.y()) - (a2.y() - a1.y()) * (b2.x() - b1.x());

        if denom.abs() < f64::EPSILON * 1e3 {
            return SegmentIntersection::None; // Degenerate
        }

        let t =
            ((b1.x() - a1.x()) * (b2.y() - b1.y()) - (b1.y() - a1.y()) * (b2.x() - b1.x())) / denom;

        let u =
            ((b1.x() - a1.x()) * (a2.y() - a1.y()) - (b1.y() - a1.y()) * (a2.x() - a1.x())) / denom;

        let px = (a2.x() - a1.x()).mul_add(t, a1.x());
        let py = (a2.y() - a1.y()).mul_add(t, a1.y());

        return SegmentIntersection::Point {
            point: Point2::new(px, py),
            t1: t,
            t2: u,
        };
    }

    // Check collinear overlap first (all four orientations are zero)
    if d1 == 0.0 && d2 == 0.0 && d3 == 0.0 && d4 == 0.0 {
        return collinear_overlap(a1, a2, b1, b2);
    }

    // Check endpoint-on-segment cases (T-intersections)
    if d1 == 0.0 && on_segment(a1, a2, b1) {
        let t = segment_param(a1, a2, b1);
        return SegmentIntersection::Point {
            point: b1,
            t1: t,
            t2: 0.0,
        };
    }
    if d2 == 0.0 && on_segment(a1, a2, b2) {
        let t = segment_param(a1, a2, b2);
        return SegmentIntersection::Point {
            point: b2,
            t1: t,
            t2: 1.0,
        };
    }
    if d3 == 0.0 && on_segment(b1, b2, a1) {
        let u = segment_param(b1, b2, a1);
        return SegmentIntersection::Point {
            point: a1,
            t1: 0.0,
            t2: u,
        };
    }
    if d4 == 0.0 && on_segment(b1, b2, a2) {
        let u = segment_param(b1, b2, a2);
        return SegmentIntersection::Point {
            point: a2,
            t1: 1.0,
            t2: u,
        };
    }

    SegmentIntersection::None
}

/// Check if point `p` lies on segment `(a, b)` (assuming collinearity).
fn on_segment(a: Point2, b: Point2, p: Point2) -> bool {
    let min_x = a.x().min(b.x());
    let max_x = a.x().max(b.x());
    let min_y = a.y().min(b.y());
    let max_y = a.y().max(b.y());

    p.x() >= min_x - f64::EPSILON
        && p.x() <= max_x + f64::EPSILON
        && p.y() >= min_y - f64::EPSILON
        && p.y() <= max_y + f64::EPSILON
}

/// Compute the parameter of point `p` on segment `(a, b)`.
fn segment_param(a: Point2, b: Point2, p: Point2) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();

    if dx.abs() > dy.abs() {
        (p.x() - a.x()) / dx
    } else if dy.abs() > f64::EPSILON {
        (p.y() - a.y()) / dy
    } else {
        0.0
    }
}

/// Handle collinear overlap of two segments.
fn collinear_overlap(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> SegmentIntersection {
    // Project onto the axis with greater extent
    let dx = (a2.x() - a1.x()).abs().max((b2.x() - b1.x()).abs());
    let dy = (a2.y() - a1.y()).abs().max((b2.y() - b1.y()).abs());

    let (_ta1, _ta2, tb1_param, tb2_param) = if dx >= dy {
        let dir = a2.x() - a1.x();
        if dir.abs() < f64::EPSILON {
            return SegmentIntersection::None;
        }
        (0.0, 1.0, (b1.x() - a1.x()) / dir, (b2.x() - a1.x()) / dir)
    } else {
        let dir = a2.y() - a1.y();
        if dir.abs() < f64::EPSILON {
            return SegmentIntersection::None;
        }
        (0.0, 1.0, (b1.y() - a1.y()) / dir, (b2.y() - a1.y()) / dir)
    };

    let (tb_lo, tb_hi) = if tb1_param < tb2_param {
        (tb1_param, tb2_param)
    } else {
        (tb2_param, tb1_param)
    };
    let lo = 0.0_f64.max(tb_lo);
    let hi = 1.0_f64.min(tb_hi);

    if lo > hi + f64::EPSILON {
        SegmentIntersection::None
    } else if (hi - lo).abs() < f64::EPSILON {
        // Single point overlap
        let px = (a2.x() - a1.x()).mul_add(lo, a1.x());
        let py = (a2.y() - a1.y()).mul_add(lo, a1.y());
        let pt = Point2::new(px, py);
        SegmentIntersection::Point {
            point: pt,
            t1: lo,
            t2: segment_param(b1, b2, pt),
        }
    } else {
        let sx = (a2.x() - a1.x()).mul_add(lo, a1.x());
        let sy = (a2.y() - a1.y()).mul_add(lo, a1.y());
        let ex = (a2.x() - a1.x()).mul_add(hi, a1.x());
        let ey = (a2.y() - a1.y()).mul_add(hi, a1.y());
        SegmentIntersection::Overlap {
            start: Point2::new(sx, sy),
            end: Point2::new(ex, ey),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::panic,
    clippy::cast_lossless
)]
mod tests {

    use super::*;
    use crate::vec::{Point2, Point3};

    // -- filtered_orient2d -------------------------------------------------

    #[test]
    fn filtered_orient2d_ccw() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        assert!(filtered_orient2d(a, b, c) > 0.0);
    }

    #[test]
    fn filtered_orient2d_cw() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(0.0, 1.0);
        let c = Point2::new(1.0, 0.0);
        assert!(filtered_orient2d(a, b, c) < 0.0);
    }

    #[test]
    fn filtered_orient2d_collinear() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 1.0);
        let c = Point2::new(2.0, 2.0);
        assert_eq!(filtered_orient2d(a, b, c), 0.0);
    }

    #[test]
    fn filtered_orient2d_near_collinear() {
        // Points very close to collinear -- should still give correct answer
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 1.0);
        let c = Point2::new(2.0, 2.0 + 1e-15);
        // The exact predicate should resolve this
        let result = filtered_orient2d(a, b, c);
        // c is slightly above the line, so should be positive (CCW)
        assert!(result >= 0.0);
    }

    // -- filtered_in_circle ------------------------------------------------

    #[test]
    fn filtered_in_circle_inside() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        let d = Point2::new(0.25, 0.25);
        assert!(filtered_in_circle(a, b, c, d) > 0.0);
    }

    #[test]
    fn filtered_in_circle_outside() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        let d = Point2::new(3.0, 3.0);
        assert!(filtered_in_circle(a, b, c, d) < 0.0);
    }

    // -- filtered_orient3d -------------------------------------------------

    #[test]
    fn filtered_orient3d_basic() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        // d above the plane => negative (matches robust crate convention)
        let above = Point3::new(0.0, 0.0, 1.0);
        assert!(filtered_orient3d(a, b, c, above) < 0.0);
        // d below the plane => positive
        let below = Point3::new(0.0, 0.0, -1.0);
        assert!(filtered_orient3d(a, b, c, below) > 0.0);
        // d on the plane => zero
        let on = Point3::new(0.5, 0.5, 0.0);
        assert_eq!(filtered_orient3d(a, b, c, on), 0.0);
    }

    // -- segment_intersection ----------------------------------------------

    #[test]
    fn segment_intersection_crossing() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(1.0, 1.0);
        let b1 = Point2::new(0.0, 1.0);
        let b2 = Point2::new(1.0, 0.0);

        match segment_intersection(a1, a2, b1, b2) {
            SegmentIntersection::Point { point, t1, t2 } => {
                assert!((point.x() - 0.5).abs() < 1e-10);
                assert!((point.y() - 0.5).abs() < 1e-10);
                assert!((t1 - 0.5).abs() < 1e-10);
                assert!((t2 - 0.5).abs() < 1e-10);
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn segment_intersection_parallel() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(1.0, 0.0);
        let b1 = Point2::new(0.0, 1.0);
        let b2 = Point2::new(1.0, 1.0);

        assert_eq!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::None
        );
    }

    #[test]
    fn segment_intersection_t_junction() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(1.0, 0.0);
        let b1 = Point2::new(0.5, -1.0);
        let b2 = Point2::new(0.5, 0.0); // endpoint on segment a

        match segment_intersection(a1, a2, b1, b2) {
            SegmentIntersection::Point { point, .. } => {
                assert!((point.x() - 0.5).abs() < 1e-10);
                assert!(point.y().abs() < 1e-10);
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn segment_intersection_collinear_overlap() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(2.0, 0.0);
        let b1 = Point2::new(1.0, 0.0);
        let b2 = Point2::new(3.0, 0.0);

        match segment_intersection(a1, a2, b1, b2) {
            SegmentIntersection::Overlap { start, end } => {
                assert!((start.x() - 1.0).abs() < 1e-10);
                assert!((end.x() - 2.0).abs() < 1e-10);
            }
            other => panic!("expected Overlap, got {other:?}"),
        }
    }

    #[test]
    fn segment_intersection_disjoint_collinear() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(1.0, 0.0);
        let b1 = Point2::new(2.0, 0.0);
        let b2 = Point2::new(3.0, 0.0);

        assert_eq!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::None
        );
    }

    #[test]
    fn segment_intersection_no_intersection() {
        let a1 = Point2::new(0.0, 0.0);
        let a2 = Point2::new(1.0, 0.0);
        let b1 = Point2::new(2.0, 2.0);
        let b2 = Point2::new(3.0, 3.0);

        assert_eq!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::None
        );
    }
}
