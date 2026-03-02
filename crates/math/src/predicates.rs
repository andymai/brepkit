//! Exact geometric predicates backed by the [`robust`] crate.
//!
//! These wrappers accept [`Point2`] values and return either raw `f64` results
//! or a classified [`Orientation`] enum.

use crate::vec::Point2;

/// Classification of the orientation of three points in the plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// The triple (a, b, c) is counter-clockwise (positive orientation).
    CounterClockwise,
    /// The triple (a, b, c) is clockwise (negative orientation).
    Clockwise,
    /// The three points are collinear.
    Collinear,
}

/// Convert a [`Point2`] to a [`robust::Coord`].
const fn to_coord(p: Point2) -> robust::Coord<f64> {
    robust::Coord { x: p.x(), y: p.y() }
}

/// Compute the exact orientation determinant of the triangle (a, b, c).
///
/// Returns a positive value if the points are counter-clockwise,
/// negative if clockwise, and zero if collinear.
#[must_use]
pub fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    robust::orient2d(to_coord(a), to_coord(b), to_coord(c))
}

/// Classify the orientation of three points in the plane.
#[must_use]
pub fn orientation2d(a: Point2, b: Point2, c: Point2) -> Orientation {
    let det = orient2d(a, b, c);
    if det > 0.0 {
        Orientation::CounterClockwise
    } else if det < 0.0 {
        Orientation::Clockwise
    } else {
        Orientation::Collinear
    }
}

/// Exact in-circle test for four 2D points.
///
/// Returns a positive value if `d` lies inside the circumcircle of (a, b, c)
/// (when a, b, c are in counter-clockwise order), negative if outside, and
/// zero if on the circle.
#[must_use]
pub fn in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    robust::incircle(to_coord(a), to_coord(b), to_coord(c), to_coord(d))
}
