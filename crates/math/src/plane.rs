//! Plane intersection utilities.

use crate::vec::{Point3, Vec3};

/// Compute the intersection line of two planes.
///
/// Each plane is defined by a normal `n` and signed distance `d` such that
/// `n · p = d` for all points `p` on the plane.
///
/// Returns `Some((point, direction))` where `point` lies on the intersection
/// line and `direction` is a unit vector along it. Returns `None` if the
/// planes are parallel (or coincident) within `tolerance`.
#[must_use]
pub fn plane_plane_intersection(
    n1: Vec3,
    d1: f64,
    n2: Vec3,
    d2: f64,
    tolerance: f64,
) -> Option<(Point3, Vec3)> {
    let dir = n1.cross(n2);
    let det = dir.length_squared();

    if det < tolerance * tolerance {
        return None;
    }

    // Solve for a point on the line: p = (a * n1 + b * n2)
    // where n1 · p = d1 and n2 · p = d2.
    //
    // n1·n1 * a + n1·n2 * b = d1
    // n1·n2 * a + n2·n2 * b = d2
    //
    // det = (n1·n1)(n2·n2) - (n1·n2)^2 = |n1 × n2|^2
    let n1n1 = n1.dot(n1);
    let n2n2 = n2.dot(n2);
    let n1n2 = n1.dot(n2);

    let a = d1.mul_add(n2n2, -(d2 * n1n2)) / det;
    let b = d2.mul_add(n1n1, -(d1 * n1n2)) / det;

    let point = Point3::new(
        n1.x().mul_add(a, n2.x() * b),
        n1.y().mul_add(a, n2.y() * b),
        n1.z().mul_add(a, n2.z() * b),
    );

    // Normalize direction (det > 0 so length > 0).
    let unit_dir = dir * (1.0 / det.sqrt());

    Some((point, unit_dir))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn xy_xz_intersection() {
        // XY plane (z=0) and XZ plane (y=0) intersect along the X axis.
        let (pt, dir) = plane_plane_intersection(
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
            1e-10,
        )
        .expect("planes should intersect");

        // Point should lie on both planes (z ≈ 0, y ≈ 0).
        assert!(pt.z().abs() < 1e-10);
        assert!(pt.y().abs() < 1e-10);

        // Direction should be along ±X.
        assert!(dir.x().abs() > 0.99);
        assert!(dir.y().abs() < 1e-10);
        assert!(dir.z().abs() < 1e-10);
    }

    #[test]
    fn offset_planes() {
        // z=1 and y=2 → line at y=2, z=1 along X.
        let (pt, dir) = plane_plane_intersection(
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(0.0, 1.0, 0.0),
            2.0,
            1e-10,
        )
        .expect("planes should intersect");

        assert!((pt.z() - 1.0).abs() < 1e-10);
        assert!((pt.y() - 2.0).abs() < 1e-10);
        assert!(dir.x().abs() > 0.99);
    }

    #[test]
    fn parallel_planes_return_none() {
        // Two parallel XY planes at z=0 and z=1.
        let result = plane_plane_intersection(
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1e-10,
        );
        assert!(result.is_none());
    }
}
