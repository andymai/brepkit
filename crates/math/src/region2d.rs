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
            let along = ap.dot(ab) / (len * len);
            if (0.0..=1.0).contains(&along) && cross(ab, ap).abs() / len <= tol {
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
            // On the ellipse (within tol) and on the arc: on the boundary.
            let radial = (x0.hypot(y0) - 1.0).abs() * a.min(b);
            if radial <= tol && on_arc(y0.atan2(x0)) {
                return Crossing::OnBoundary;
            }
            let disc = qb.mul_add(qb, -4.0 * qa * qc);
            if disc < 0.0 {
                return Crossing::Count(0);
            }
            let half = disc.sqrt() / (2.0 * qa);
            if half <= tol / size {
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
}
