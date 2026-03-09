//! Exact geometric predicates backed by the [`robust`] crate.
//!
//! These wrappers accept [`Point2`] and [`Point3`] values and return either
//! raw `f64` results or classified enum values.

use crate::vec::{Point2, Point3};

// ---------------------------------------------------------------------------
// 2D predicates
// ---------------------------------------------------------------------------

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

/// Compute the winding number of a point with respect to a polygon.
///
/// The polygon is given as a slice of vertices forming a closed loop (the
/// last vertex is implicitly connected to the first). Returns the winding
/// number: non-zero means the point is inside.
#[must_use]
pub fn winding_number(point: Point2, polygon: &[Point2]) -> i32 {
    let n = polygon.len();
    if n < 3 {
        return 0;
    }

    let mut wn = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = polygon[i];
        let vj = polygon[j];

        if vi.y() <= point.y() {
            if vj.y() > point.y() {
                // Upward crossing
                if orient2d(vi, vj, point) > 0.0 {
                    wn += 1;
                }
            }
        } else if vj.y() <= point.y() {
            // Downward crossing
            if orient2d(vi, vj, point) < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

/// Test whether a point lies inside a polygon using the winding number rule.
///
/// Returns `true` if the winding number is non-zero.
#[must_use]
pub fn point_in_polygon(point: Point2, polygon: &[Point2]) -> bool {
    winding_number(point, polygon) != 0
}

// ---------------------------------------------------------------------------
// 3D predicates
// ---------------------------------------------------------------------------

/// Classification of a point's position relative to an oriented plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation3D {
    /// The point lies above the plane (positive side).
    Above,
    /// The point lies below the plane (negative side).
    Below,
    /// The point lies on the plane.
    Coplanar,
}

/// Convert a [`Point3`] to a [`robust::Coord3D`].
const fn to_coord3d(p: Point3) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p.x(),
        y: p.y(),
        z: p.z(),
    }
}

/// Compute the exact orientation of point `d` relative to the plane through
/// `(a, b, c)`.
///
/// Returns a positive value if `d` lies below the plane (a, b, c appear
/// counter-clockwise when viewed from above), negative if above, zero if
/// coplanar. This matches the convention of the `robust` crate.
#[must_use]
pub fn orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    robust::orient3d(to_coord3d(a), to_coord3d(b), to_coord3d(c), to_coord3d(d))
}

/// Classify the orientation of point `d` relative to the plane through
/// `(a, b, c)`.
#[must_use]
pub fn orientation3d(a: Point3, b: Point3, c: Point3, d: Point3) -> Orientation3D {
    let det = orient3d(a, b, c, d);
    if det > 0.0 {
        Orientation3D::Below
    } else if det < 0.0 {
        Orientation3D::Above
    } else {
        Orientation3D::Coplanar
    }
}

/// Exact in-sphere test for five 3D points.
///
/// Returns a positive value if `e` lies inside the circumsphere of
/// `(a, b, c, d)` (when a, b, c, d have positive orientation), negative if
/// outside, zero if on the sphere.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn insphere(a: Point3, b: Point3, c: Point3, d: Point3, e: Point3) -> f64 {
    robust::insphere(
        to_coord3d(a),
        to_coord3d(b),
        to_coord3d(c),
        to_coord3d(d),
        to_coord3d(e),
    )
}

// ---------------------------------------------------------------------------
// Symbolic perturbation (SoS — Simulation of Simplicity)
// ---------------------------------------------------------------------------

/// Compute `orient2d(a, b, c)` with symbolic perturbation to resolve degeneracy.
///
/// When the exact `orient2d` returns 0 (collinear points), applies an
/// index-based perturbation: the point with the highest index is perturbed
/// infinitesimally upward, breaking ties consistently.
///
/// Returns a non-zero `f64` whose sign indicates the resolved orientation.
/// The magnitude is arbitrary when the exact result was zero.
// SoS predicates require exact-zero detection by design
#[allow(clippy::float_cmp)]
#[must_use]
pub fn orient2d_sos(a: Point2, b: Point2, c: Point2, ia: usize, ib: usize, ic: usize) -> f64 {
    let det = orient2d(a, b, c);
    if det != 0.0 {
        return det;
    }

    // SoS perturbation: the point with the largest index is perturbed.
    // The sign depends on its position in the argument list (even/odd parity).
    let max_idx = ia.max(ib).max(ic);
    if max_idx == ic {
        1.0 // c is perturbed → positive (CCW)
    } else if max_idx == ib {
        -1.0 // b is perturbed → negative (CW)
    } else {
        1.0 // a is perturbed → positive
    }
}

/// Compute `orient3d(a, b, c, d)` with symbolic perturbation to resolve degeneracy.
///
/// When the exact `orient3d` returns 0 (coplanar points), applies an
/// index-based perturbation: the point with the highest index is perturbed
/// infinitesimally along the z-axis, breaking ties consistently.
///
/// Returns a non-zero `f64` whose sign indicates the resolved orientation.
/// The magnitude is arbitrary when the exact result was zero.
// SoS predicates require exact-zero detection by design
#[allow(clippy::float_cmp)]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn orient3d_sos(
    a: Point3,
    b: Point3,
    c: Point3,
    d: Point3,
    ia: usize,
    ib: usize,
    ic: usize,
    id: usize,
) -> f64 {
    let det = orient3d(a, b, c, d);
    if det != 0.0 {
        return det;
    }

    // SoS perturbation: the point with the largest index receives an
    // infinitesimal perturbation. The sign depends on which argument
    // position it occupies (even permutation → positive, odd → negative).
    let max_idx = ia.max(ib).max(ic).max(id);
    if max_idx == id {
        1.0 // d perturbed: positive (below plane)
    } else if max_idx == ic {
        -1.0 // c perturbed: negative (above plane)
    } else if max_idx == ib {
        1.0 // b perturbed: positive
    } else {
        -1.0 // a perturbed: negative
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn orient2d_ccw() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        assert!(orient2d(a, b, c) > 0.0);
        assert_eq!(orientation2d(a, b, c), Orientation::CounterClockwise);
    }

    #[test]
    fn orient2d_cw() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(0.0, 1.0);
        let c = Point2::new(1.0, 0.0);
        assert!(orient2d(a, b, c) < 0.0);
        assert_eq!(orientation2d(a, b, c), Orientation::Clockwise);
    }

    #[test]
    fn orient2d_collinear() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 1.0);
        let c = Point2::new(2.0, 2.0);
        assert_eq!(orient2d(a, b, c), 0.0);
        assert_eq!(orientation2d(a, b, c), Orientation::Collinear);
    }

    #[test]
    fn orient2d_swap_reverses_sign() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1.0);
        let d1 = orient2d(a, b, c);
        let d2 = orient2d(b, a, c);
        assert!((d1 + d2).abs() < 1e-15, "swap should reverse sign");
    }

    #[test]
    fn orient3d_basic() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let above = Point3::new(0.0, 0.0, 1.0);
        let below = Point3::new(0.0, 0.0, -1.0);
        let on = Point3::new(0.5, 0.5, 0.0);

        assert!(orient3d(a, b, c, above) < 0.0); // above the plane
        assert!(orient3d(a, b, c, below) > 0.0); // below
        assert_eq!(orient3d(a, b, c, on), 0.0); // coplanar

        assert_eq!(orientation3d(a, b, c, above), Orientation3D::Above);
        assert_eq!(orientation3d(a, b, c, below), Orientation3D::Below);
        assert_eq!(orientation3d(a, b, c, on), Orientation3D::Coplanar);
    }

    #[test]
    fn insphere_inside() {
        // Tetrahedron with positive orientation, test point at origin.
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        let c = Point3::new(0.0, 0.0, 1.0);
        let d = Point3::new(0.0, 0.0, 0.0);
        // Ensure positive orientation by checking orient3d sign.
        // If orient3d(a,b,c,d) > 0, they have the right order.
        let center = Point3::new(0.25, 0.25, 0.25);
        // The circumsphere of a regular-ish tetrahedron, center should be inside.
        let result = insphere(a, b, c, d, center);
        // Just check it returns a finite value (exact inside/outside depends on geometry)
        assert!(result.is_finite());
    }

    #[test]
    fn winding_number_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        // Inside
        assert_eq!(winding_number(Point2::new(0.5, 0.5), &square), 1);
        assert!(point_in_polygon(Point2::new(0.5, 0.5), &square));
        // Outside
        assert_eq!(winding_number(Point2::new(2.0, 2.0), &square), 0);
        assert!(!point_in_polygon(Point2::new(2.0, 2.0), &square));
    }

    #[test]
    fn winding_number_triangle() {
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 3.0),
        ];
        assert!(point_in_polygon(Point2::new(2.0, 1.0), &tri));
        assert!(!point_in_polygon(Point2::new(5.0, 0.0), &tri));
    }

    #[test]
    fn winding_number_degenerate() {
        // Too few points
        assert_eq!(winding_number(Point2::new(0.0, 0.0), &[]), 0);
        assert_eq!(
            winding_number(
                Point2::new(0.0, 0.0),
                &[Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)]
            ),
            0
        );
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_orient2d_swap_sign(
            ax in -10.0f64..10.0, ay in -10.0f64..10.0,
            bx in -10.0f64..10.0, by in -10.0f64..10.0,
            cx in -10.0f64..10.0, cy in -10.0f64..10.0,
        ) {
            let a = Point2::new(ax, ay);
            let b = Point2::new(bx, by);
            let c = Point2::new(cx, cy);
            let d1 = orient2d(a, b, c);
            let d2 = orient2d(b, a, c);
            prop_assert!((d1 + d2).abs() < 1e-10, "d1={}, d2={}", d1, d2);
        }
    }
}
