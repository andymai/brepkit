//! Whether a point lies in a plane region bounded by segments and conic arcs.
//!
//! The curves themselves are read, not a chord polygon, which misreads every
//! point between an arc and its chords.

use std::f64::consts::TAU;

use crate::vec::{Point2, Vec2};

/// A piece of a region's boundary in the plane's 2D frame.
#[derive(Debug, Clone, Copy)]
pub enum Boundary2 {
    /// A straight segment between two points.
    Segment(Point2, Point2),
    /// The arc of the ellipse `center + a cos t u + b sin t v` for `t` from
    /// `t0` to `t1` (`t0 < t1 <= t0 + 2 pi`); a circle when `a == b`. `u` and
    /// `v` are orthonormal.
    Arc {
        /// The ellipse's center.
        center: Point2,
        /// The direction of the `a` axis.
        u: Vec2,
        /// The direction of the `b` axis.
        v: Vec2,
        /// The semi-axis along `u`.
        a: f64,
        /// The semi-axis along `v`.
        b: f64,
        /// The arc's first angle.
        t0: f64,
        /// The arc's last angle.
        t1: f64,
    },
}

/// Ray directions tried in turn, none along an axis or a common diagonal.
const DIRECTIONS: [f64; 6] = [0.613, 1.931, 2.871, 4.127, 5.369, 0.229];

/// Whether `p` lies inside the region `pieces` enclose, holes included.
///
/// Read by the parity of a ray's crossings of every loop's pieces together.
/// `None` when `p` lies within `tol` of the boundary, or every trial ray
/// passes within `tol` of a piece's end or touches an arc without crossing.
#[must_use]
pub fn point_in_region(pieces: &[Boundary2], p: Point2, tol: f64) -> Option<bool> {
    'direction: for angle in DIRECTIONS {
        let d = Vec2::new(angle.cos(), angle.sin());
        let mut odd = false;
        for piece in pieces {
            match crossings(piece, p, d, tol) {
                Crossing::Count(n) => odd ^= n % 2 == 1,
                Crossing::Grazes => continue 'direction,
                Crossing::OnBoundary => return None,
            }
        }
        return Some(odd);
    }
    None
}

/// Whether `p` lies within `tol` of one of the region's boundary pieces.
#[must_use]
pub fn on_boundary(pieces: &[Boundary2], p: Point2, tol: f64) -> bool {
    let along = Vec2::new(1.0, 0.0);
    pieces
        .iter()
        .any(|piece| matches!(crossings(piece, p, along, tol), Crossing::OnBoundary))
}

/// The distance from `p` to the nearest of `pieces`; infinite with none.
#[must_use]
pub fn distance_to_boundary(pieces: &[Boundary2], p: Point2) -> f64 {
    pieces
        .iter()
        .map(|piece| distance_to_piece(piece, p))
        .fold(f64::INFINITY, f64::min)
}

/// The distance from `p` to `piece`.
///
/// Exact for a segment and a circular arc. For an elliptical arc, a branch
/// and bound over its angles that never reads more than the true distance
/// and falls short of it by a millionth or so.
#[must_use]
pub fn distance_to_piece(piece: &Boundary2, p: Point2) -> f64 {
    match *piece {
        Boundary2::Segment(a, b) => {
            let (ab, ap) = (b - a, p - a);
            let len2 = ab.dot(ab);
            let along = if len2 > 0.0 {
                (ap.dot(ab) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (ap - ab * along).length()
        }
        Boundary2::Arc {
            center,
            u,
            v,
            a,
            b,
            t0,
            t1,
        } => {
            let q = p - center;
            let (x0, y0) = (q.dot(u), q.dot(v));
            #[allow(clippy::float_cmp)]
            let circle = a == b;
            if !circle {
                return ellipse_arc_distance(x0, y0, a, b, t0, t1);
            }
            if t0 + (y0.atan2(x0) - t0).rem_euclid(TAU) <= t1 {
                return (x0.hypot(y0) - a).abs();
            }
            let end = |t: f64| (a.mul_add(-t.cos(), x0)).hypot(b.mul_add(-t.sin(), y0));
            end(t0).min(end(t1))
        }
    }
}

/// The most halvings of an elliptical arc's angles a distance reads.
const ELLIPSE_DEPTH: u32 = 60;

/// The distance from `(x0, y0)` to the arc `(a cos t, b sin t)` for `t` from
/// `t0` to `t1`, by branch and bound on the squared distance `F`.
///
/// On an angle interval of half-width `r` about `m`, `F` is at least
/// `F(m) - |F'(m)| r - K r^2 / 2`, with `K = 2 (|a^2 - b^2| + a |x0| + b |y0|)`
/// bounding `|F''|`. An interval whose bound cannot undercut the least `F`
/// seen, less a millionth of it, is dropped; the rest are halved.
fn ellipse_arc_distance(x0: f64, y0: f64, a: f64, b: f64, t0: f64, t1: f64) -> f64 {
    // Bounds that cannot be compared prune nothing, and the halving would
    // run to its full depth everywhere: such an arc reads as touching.
    const MAX_INTERVALS: usize = 1 << 20;
    if ![x0, y0, a, b, t0, t1].iter().all(|v| v.is_finite()) {
        return 0.0;
    }
    let squared = |t: f64| {
        let (s, c) = t.sin_cos();
        a.mul_add(c, -x0).powi(2) + b.mul_add(s, -y0).powi(2)
    };
    let slope = |t: f64| {
        let (s, c) = t.sin_cos();
        2.0 * (a.mul_add(c, -x0) * (-a * s) + b.mul_add(s, -y0) * (b * c))
    };
    let k = 2.0 * (a.mul_add(a, -b * b).abs() + a * x0.abs() + b * y0.abs());
    let noise = (1e-15 * a.max(b).max(x0.hypot(y0))).powi(2);
    let slack = |best: f64| 1e-6_f64.mul_add(best, noise);
    let mut best = squared(t0).min(squared(t1));
    let mut unsettled = f64::INFINITY;
    let mut pending = vec![(t0, t1, 0_u32)];
    let mut visited = 0_usize;
    while let Some((lo, hi, depth)) = pending.pop() {
        visited += 1;
        let (m, r) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let at_m = squared(m);
        best = best.min(at_m);
        let low = at_m - slope(m).abs() * r - 0.5 * k * r * r;
        if low.is_nan() || visited > MAX_INTERVALS {
            return 0.0;
        }
        if low >= best - slack(best) {
            continue;
        }
        if depth == ELLIPSE_DEPTH {
            unsettled = unsettled.min(low);
            continue;
        }
        pending.push((lo, m, depth + 1));
        pending.push((m, hi, depth + 1));
    }
    (best - slack(best)).min(unsettled).max(0.0).sqrt()
}

/// How a ray from a point meets one boundary piece.
enum Crossing {
    /// The point itself lies on the piece.
    OnBoundary,
    /// The ray passes a piece's end or touches it without crossing: another
    /// ray is needed.
    Grazes,
    /// The ray crosses the piece this many times.
    Count(u32),
}

/// How the ray from `p` along `d` meets `piece`.
fn crossings(piece: &Boundary2, p: Point2, d: Vec2, tol: f64) -> Crossing {
    let cross = |x: Vec2, y: Vec2| x.x().mul_add(y.y(), -(x.y() * y.x()));
    match *piece {
        Boundary2::Segment(a, b) => {
            let (ab, ap) = (b - a, p - a);
            let len = ab.length();
            if len <= tol {
                return Crossing::Count(0);
            }
            // Within `tol` of the closed segment: a point at a vertex rounds
            // a hair past the end of both segments meeting there.
            let along = (ap.dot(ab) / (len * len)).clamp(0.0, 1.0);
            if (ap - ab * along).length() <= tol {
                return Crossing::OnBoundary;
            }
            let det = cross(d, ab);
            if det.abs() <= 1e-12 * len {
                // Parallel: a collinear segment ahead grazes the ray.
                return if cross(d, ap).abs() <= tol && ap.dot(d) < 0.0 {
                    Crossing::Grazes
                } else {
                    Crossing::Count(0)
                };
            }
            let s = cross(ap * -1.0, ab) / det;
            let w = cross(ap * -1.0, d) / det;
            if s <= 0.0 || w < -tol / len || w > 1.0 + tol / len {
                return Crossing::Count(0);
            }
            if w * len <= tol || (1.0 - w) * len <= tol {
                return Crossing::Grazes;
            }
            Crossing::Count(1)
        }
        Boundary2::Arc {
            center,
            u,
            v,
            a,
            b,
            t0,
            t1,
        } => {
            let q = p - center;
            let (x0, y0) = (q.dot(u) / a, q.dot(v) / b);
            let (dx, dy) = (d.dot(u) / a, d.dot(v) / b);
            let qa = dx.mul_add(dx, dy * dy);
            let qb = 2.0 * x0.mul_add(dx, y0 * dy);
            let qc = x0.mul_add(x0, y0 * y0) - 1.0;
            let size = a.max(b);
            let on_arc = |t: f64| t0 + (t - t0).rem_euclid(TAU) <= t1;
            // On the ellipse (within tol) and on the arc: on the boundary. The
            // offset is read along the ray from the center, where the ellipse
            // lies at `1 / rho` of the point's distance.
            let rho = x0.hypot(y0);
            let radial = if rho > 0.0 {
                q.length() * (rho - 1.0).abs() / rho
            } else {
                a.min(b)
            };
            let end = |t: f64| center + u * (a * t.cos()) + v * (b * t.sin());
            // Along a slender ellipse the ray from its center meets the curve
            // nearly tangent, so the radial offset can be many times the true
            // distance; it is never less than the square of the axes' ratio
            // of it. Within that margin the arc's own distance decides.
            let squash = (size / a.min(b)).powi(2);
            if (radial <= tol && on_arc(y0.atan2(x0)))
                || (p - end(t0)).length() <= tol
                || (p - end(t1)).length() <= tol
                || (squash > 1.0 + 1e-12
                    && squash.is_finite()
                    && radial <= tol * squash
                    && distance_to_piece(piece, p) <= tol)
            {
                return Crossing::OnBoundary;
            }
            let disc = qb.mul_add(qb, -4.0 * qa * qc);
            if disc < 0.0 {
                return Crossing::Count(0);
            }
            let half = disc.sqrt() / (2.0 * qa);
            if half <= tol {
                return Crossing::Grazes;
            }
            let mut n = 0;
            for s in [-qb / (2.0 * qa) - half, -qb / (2.0 * qa) + half] {
                if s <= 0.0 {
                    continue;
                }
                let t = (y0 + s * dy).atan2(x0 + s * dx);
                let t = t0 + (t - t0).rem_euclid(TAU);
                let (from_start, to_end) = (t - t0, t1 - t);
                if (from_start * size <= tol || to_end.abs() * size <= tol)
                    || (t > t1 && (t - TAU - t0).abs() * size <= tol)
                {
                    return Crossing::Grazes;
                }
                if t <= t1 {
                    n += 1;
                }
            }
            Crossing::Count(n)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use std::f64::consts::PI;

    use super::*;

    /// An arc given a non-finite value reads as touching the point at once
    /// rather than halving without end.
    #[test]
    fn a_non_finite_ellipse_arc_reads_as_touching() {
        for bad in [f64::NAN, f64::INFINITY] {
            let off_centre = super::ellipse_arc_distance(bad, 0.0, 2.0, 1.0, 0.0, 1.0);
            let off_range = super::ellipse_arc_distance(0.5, 0.0, 2.0, 1.0, 0.0, bad);
            assert!(off_centre.abs() < f64::EPSILON && off_range.abs() < f64::EPSILON);
        }
    }

    fn circle(center: (f64, f64), r: f64, t0: f64, t1: f64) -> Boundary2 {
        Boundary2::Arc {
            center: Point2::new(center.0, center.1),
            u: Vec2::new(1.0, 0.0),
            v: Vec2::new(0.0, 1.0),
            a: r,
            b: r,
            t0,
            t1,
        }
    }

    /// A unit disc bounded by one whole circle: a point a sagitta inside its
    /// rim is inside, however few chords a polygon would give it.
    #[test]
    fn a_disc_holds_points_up_to_its_rim() {
        let disc = [circle((0.0, 0.0), 1.0, 0.0, TAU)];
        for r in [0.0, 0.5, 0.999, 0.9999] {
            for k in 0..12 {
                let t = f64::from(k) * PI / 6.0 + 0.1;
                let p = Point2::new(r * t.cos(), r * t.sin());
                assert_eq!(point_in_region(&disc, p, 1e-9), Some(true), "r {r} t {t}");
            }
        }
        assert_eq!(
            point_in_region(&disc, Point2::new(1.001, 0.0), 1e-9),
            Some(false)
        );
        assert_eq!(point_in_region(&disc, Point2::new(1.0, 0.0), 1e-9), None);
    }

    /// A triangle's corners, nudged by rounding past the ends of both edges
    /// meeting there, and an arc's ends, read on the boundary.
    #[test]
    fn points_at_corners_read_on_the_boundary() {
        let (a, b, c) = (
            Point2::new(-18.75, -16.75),
            Point2::new(-20.673_141_121_612_92, -17.530_361_288_064_512),
            Point2::new(-20.75, 4.0),
        );
        let triangle = [
            Boundary2::Segment(a, b),
            Boundary2::Segment(b, c),
            Boundary2::Segment(c, a),
        ];
        for corner in [a, b, c] {
            for (dx, dy) in [(0.0, 0.0), (3e-15, -2e-15), (-4e-15, 1e-15)] {
                let p = Point2::new(corner.x() + dx, corner.y() + dy);
                assert_eq!(point_in_region(&triangle, p, 1e-9), None, "{p:?}");
            }
        }
        let half = [
            circle((0.0, 0.0), 2.0, 0.3, 2.9),
            Boundary2::Segment(
                Point2::new(2.0 * 2.9_f64.cos(), 2.0 * 2.9_f64.sin()),
                Point2::new(2.0 * 0.3_f64.cos(), 2.0 * 0.3_f64.sin()),
            ),
        ];
        for t in [0.3_f64, 2.9] {
            let p = Point2::new(2.0 * t.cos() + 1e-15, 2.0 * t.sin());
            assert_eq!(point_in_region(&half, p, 1e-9), None, "t {t}");
        }
    }

    /// Points on a side or an arc read on the boundary, and points a ray
    /// from which grazes a corner do not.
    #[test]
    fn on_boundary_reads_distance_not_ray_parity() {
        let half = [
            circle((0.0, 0.0), 2.0, 0.0, PI),
            Boundary2::Segment(Point2::new(-2.0, 0.0), Point2::new(2.0, 0.0)),
        ];
        assert!(on_boundary(&half, Point2::new(0.5, 0.0), 1e-9));
        assert!(on_boundary(&half, Point2::new(0.0, 2.0), 1e-9));
        assert!(on_boundary(&half, Point2::new(2.0, 1e-15), 1e-9));
        assert!(!on_boundary(&half, Point2::new(0.5, 0.5), 1e-9));
        assert!(!on_boundary(&half, Point2::new(0.0, 2.1), 1e-9));
    }

    /// A half disc: its diameter and a half circle. Points just inside the
    /// arc read inside, points past the diameter outside.
    #[test]
    fn a_half_disc_of_a_segment_and_an_arc() {
        let half = [
            circle((0.0, 0.0), 2.0, 0.0, PI),
            Boundary2::Segment(Point2::new(-2.0, 0.0), Point2::new(2.0, 0.0)),
        ];
        assert_eq!(
            point_in_region(&half, Point2::new(0.0, 1.999), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(1.4, 1.4), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(0.0, -0.001), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(1.42, 1.42), 1e-9),
            Some(false)
        );
    }

    /// A square with a round hole: the hole's inside reads outside, the ring
    /// inside.
    #[test]
    fn a_square_with_a_round_hole() {
        let (lo, hi) = (Point2::new(-3.0, -3.0), Point2::new(3.0, 3.0));
        let region = [
            Boundary2::Segment(lo, Point2::new(hi.x(), lo.y())),
            Boundary2::Segment(Point2::new(hi.x(), lo.y()), hi),
            Boundary2::Segment(hi, Point2::new(lo.x(), hi.y())),
            Boundary2::Segment(Point2::new(lo.x(), hi.y()), lo),
            circle((0.0, 0.0), 1.0, 0.0, TAU),
        ];
        assert_eq!(
            point_in_region(&region, Point2::new(0.0, 0.0), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&region, Point2::new(0.0, 1.0005), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&region, Point2::new(0.0, 0.9995), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&region, Point2::new(2.9, -2.9), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&region, Point2::new(3.1, 0.0), 1e-9),
            Some(false)
        );
    }

    /// An ellipse 3 by 1, turned: a point just inside its end reads inside.
    #[test]
    fn a_turned_ellipse() {
        let (c, s) = (0.5_f64.cos(), 0.5_f64.sin());
        let ellipse = [Boundary2::Arc {
            center: Point2::new(1.0, 2.0),
            u: Vec2::new(c, s),
            v: Vec2::new(-s, c),
            a: 3.0,
            b: 1.0,
            t0: 0.3,
            t1: 0.3 + TAU,
        }];
        let at = |x: f64, y: f64| Point2::new(1.0 + x * c - y * s, 2.0 + x * s + y * c);
        assert_eq!(point_in_region(&ellipse, at(2.999, 0.0), 1e-9), Some(true));
        assert_eq!(point_in_region(&ellipse, at(3.001, 0.0), 1e-9), Some(false));
        assert_eq!(point_in_region(&ellipse, at(0.0, 0.999), 1e-9), Some(true));
    }

    /// The first trial ray from a point passes through a triangle's vertex:
    /// it grazes, the next ray decides, inside and outside alike.
    #[test]
    fn a_ray_through_a_vertex_tries_another() {
        let d = Vec2::new(DIRECTIONS[0].cos(), DIRECTIONS[0].sin());
        let apex = Point2::new(0.0, 0.0) + d * 2.0;
        let (b, c) = (Point2::new(-2.0, 0.5), Point2::new(0.5, -2.0));
        let triangle = [
            Boundary2::Segment(apex, b),
            Boundary2::Segment(b, c),
            Boundary2::Segment(c, apex),
        ];
        assert_eq!(
            point_in_region(&triangle, Point2::new(0.0, 0.0), 1e-9),
            Some(true)
        );
        let behind = Point2::new(0.0, 0.0) + d * -3.0;
        assert_eq!(point_in_region(&triangle, behind, 1e-9), Some(false));
    }

    /// A side lying along the first trial ray: the ray grazes it, and the
    /// next ray reads the point outside.
    #[test]
    fn a_side_along_the_ray_tries_another() {
        let d = Vec2::new(DIRECTIONS[0].cos(), DIRECTIONS[0].sin());
        let n = Vec2::new(-d.y(), d.x());
        let p = Point2::new(0.0, 0.0);
        let (q1, q2) = (p + d, p + d * 2.0);
        let q3 = p + d * 1.5 + n;
        let triangle = [
            Boundary2::Segment(q1, q2),
            Boundary2::Segment(q2, q3),
            Boundary2::Segment(q3, q1),
        ];
        assert_eq!(point_in_region(&triangle, p, 1e-9), Some(false));
    }

    /// The segment of a unit circle between angles 2.5 and 4.0, across the
    /// angle's wrap at pi, closed by its chord.
    #[test]
    fn an_arc_across_the_wrap() {
        let (t0, t1) = (2.5_f64, 4.0_f64);
        let segment = [
            circle((0.0, 0.0), 1.0, t0, t1),
            Boundary2::Segment(
                Point2::new(t1.cos(), t1.sin()),
                Point2::new(t0.cos(), t0.sin()),
            ),
        ];
        assert_eq!(
            point_in_region(&segment, Point2::new(-0.9, 0.0), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&segment, Point2::new(-0.7, 0.0), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&segment, Point2::new(-1.01, 0.0), 1e-9),
            Some(false)
        );
    }

    /// The upper half of an ellipse 3 by 1, closed by its major axis.
    #[test]
    fn half_an_ellipse() {
        let half = [
            Boundary2::Arc {
                center: Point2::new(0.0, 0.0),
                u: Vec2::new(1.0, 0.0),
                v: Vec2::new(0.0, 1.0),
                a: 3.0,
                b: 1.0,
                t0: 0.0,
                t1: PI,
            },
            Boundary2::Segment(Point2::new(-3.0, 0.0), Point2::new(3.0, 0.0)),
        ];
        assert_eq!(
            point_in_region(&half, Point2::new(0.0, 0.99), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(0.0, 1.01), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(2.9, 0.1), 1e-9),
            Some(true)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(2.9, 0.3), 1e-9),
            Some(false)
        );
        assert_eq!(
            point_in_region(&half, Point2::new(0.0, -0.1), 1e-9),
            Some(false)
        );
    }

    /// A long ellipse's end: a point 5e-7 past it is off the boundary.
    #[test]
    fn a_long_ellipse_reads_its_end_by_distance() {
        let ellipse = [Boundary2::Arc {
            center: Point2::new(0.0, 0.0),
            u: Vec2::new(1.0, 0.0),
            v: Vec2::new(0.0, 1.0),
            a: 100.0,
            b: 1.0,
            t0: 0.0,
            t1: TAU,
        }];
        let past = Point2::new(100.000_000_5, 0.0);
        assert_eq!(point_in_region(&ellipse, past, 1e-7), Some(false));
    }

    /// A segment's distance runs to its nearest point, its ends included,
    /// and a zero-length one is a point.
    #[test]
    fn distance_to_a_segment() {
        let seg = Boundary2::Segment(Point2::new(0.0, 0.0), Point2::new(2.0, 0.0));
        let at = |x: f64, y: f64| distance_to_piece(&seg, Point2::new(x, y));
        assert!((at(1.0, 0.5) - 0.5).abs() < 1e-15);
        assert!((at(3.0, 0.0) - 1.0).abs() < 1e-15);
        assert!((at(-3.0, 4.0) - 5.0).abs() < 1e-15);
        let dot = Boundary2::Segment(Point2::new(1.0, 1.0), Point2::new(1.0, 1.0));
        assert!((distance_to_piece(&dot, Point2::new(4.0, 5.0)) - 5.0).abs() < 1e-15);
    }

    /// A circular arc's distance is radial where the point's angle lies on
    /// the arc, and to the nearer end where it does not.
    #[test]
    fn distance_to_a_circular_arc() {
        let arc = circle((1.0, 1.0), 2.0, 0.0, PI / 2.0);
        let at = |x: f64, y: f64| distance_to_piece(&arc, Point2::new(x, y));
        assert!((at(1.0 + 3.0 * 0.6, 1.0 + 3.0 * 0.8) - 1.0).abs() < 1e-14);
        assert!((at(1.0 + 0.6, 1.0 + 0.8) - 1.0).abs() < 1e-14);
        assert!((at(1.0, 1.0) - 2.0).abs() < 1e-14);
        assert!((at(1.0, -1.0) - 2.0_f64.hypot(2.0)).abs() < 1e-14);
        assert!((at(-2.0, 1.0) - 3.0_f64.hypot(2.0)).abs() < 1e-14);
    }

    /// The least distance from `p` to the ellipse arc, by dense sampling
    /// refined by golden section: a value the curve reaches.
    fn sampled_ellipse_distance(arc: &Boundary2, p: Point2) -> f64 {
        let Boundary2::Arc {
            center,
            u,
            v,
            a,
            b,
            t0,
            t1,
        } = *arc
        else {
            return f64::NAN;
        };
        let at = |t: f64| (center + u * (a * t.cos()) + v * (b * t.sin()) - p).length();
        let n = 20_000;
        let step = (t1 - t0) / f64::from(n);
        let mut best = (at(t0), t0);
        for k in 1..=n {
            let t = step.mul_add(f64::from(k), t0);
            if at(t) < best.0 {
                best = (at(t), t);
            }
        }
        let (mut lo, mut hi) = ((best.1 - step).max(t0), (best.1 + step).min(t1));
        for _ in 0..100 {
            let (m1, m2) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
            if at(m1) < at(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        best.0.min(at(0.5 * (lo + hi)))
    }

    /// An ellipse's distance never exceeds what the curve reaches and falls
    /// short of it by no more than a millionth: from outside, from near the
    /// centre where two angles are nearest, from on and beside the curve,
    /// and from past an arc's end.
    #[test]
    fn a_point_beside_a_slender_ellipse_is_on_it() {
        // An ellipse 10 by 0.1, and a point 5e-8 off it along its normal at
        // the angle 0.1, where the ray from the centre runs 84 degrees off
        // that normal and meets the curve some ten times farther away.
        let piece = Boundary2::Arc {
            center: Point2::new(0.0, 0.0),
            u: Vec2::new(1.0, 0.0),
            v: Vec2::new(0.0, 1.0),
            a: 10.0,
            b: 0.1,
            t0: 0.0,
            t1: TAU,
        };
        let t: f64 = 0.1;
        let on = Point2::new(10.0 * t.cos(), 0.1 * t.sin());
        let normal = Vec2::new(0.1 * t.cos(), 10.0 * t.sin());
        let normal = normal * (1.0 / normal.length());
        for side in [1.0, -1.0] {
            let p = on + normal * (side * 5e-8);
            assert!(distance_to_piece(&piece, p) < 1e-7);
            assert_eq!(point_in_region(&[piece], p, 1e-7), None, "side {side}");
        }
    }

    #[test]
    fn distance_to_an_elliptical_arc() {
        let (c, s) = (0.5_f64.cos(), 0.5_f64.sin());
        let turned = |t0: f64, t1: f64| Boundary2::Arc {
            center: Point2::new(1.0, 2.0),
            u: Vec2::new(c, s),
            v: Vec2::new(-s, c),
            a: 3.0,
            b: 1.0,
            t0,
            t1,
        };
        let local = |x: f64, y: f64| Point2::new(1.0 + x * c - y * s, 2.0 + x * s + y * c);
        let whole = turned(0.3, 0.3 + TAU);
        let half = turned(0.0, PI);
        let cases = [
            (&whole, local(5.0, 4.0)),
            (&whole, local(0.1, 0.05)),
            (&whole, local(2.0, 0.0)),
            (&whole, local(3.0 * 0.8_f64.cos(), 0.8_f64.sin())),
            (
                &whole,
                local(3.0 * 0.8_f64.cos(), 1.000_001 * 0.8_f64.sin()),
            ),
            (&half, local(0.5, -2.0)),
            (&half, local(-3.5, -0.5)),
        ];
        for (arc, p) in cases {
            let read = distance_to_piece(arc, p);
            let reached = sampled_ellipse_distance(arc, p);
            assert!(read <= reached + 1e-12, "{p:?}: {read} above {reached}");
            assert!(
                read >= reached * (1.0 - 1e-6) - 1e-12,
                "{p:?}: {read} below {reached}"
            );
        }
    }

    /// The boundary's distance is its nearest piece's, and infinite with
    /// none.
    #[test]
    fn distance_to_the_nearest_piece() {
        let pieces = [
            Boundary2::Segment(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            circle((2.0, 0.0), 1.0, 0.0, PI),
        ];
        let p = Point2::new(2.0, 0.5);
        assert!((distance_to_boundary(&pieces, p) - 0.5).abs() < 1e-15);
        assert!(distance_to_boundary(&[], p).is_infinite());
    }
}
