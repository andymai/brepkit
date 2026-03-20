//! Regular grid sampling of parametric surfaces.

use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::Point3;

/// Sample a regular N×M grid of surface points.
///
/// Returns `nu` rows, each with `nv` points.
///
/// - Row `i` corresponds to `u = u_range.0 + i*(u_range.1 - u_range.0) / (nu - 1)`.
/// - Column `j` corresponds to `v = v_range.0 + j*(v_range.1 - v_range.0) / (nv - 1)`.
///
/// Edge cases:
/// - If `nu == 0` or `nv == 0`, an empty `Vec` is returned.
/// - If `nu == 1`, the single row is evaluated at `u_range.0`.
/// - If `nv == 1`, each row contains one point evaluated at `v_range.0`.
#[must_use]
pub fn surface_grid<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
) -> Vec<Vec<Point3>> {
    if nu == 0 || nv == 0 {
        return Vec::new();
    }

    (0..nu)
        .map(|i| {
            let u = if nu == 1 {
                u_range.0
            } else if i == nu - 1 {
                u_range.1
            } else {
                u_range.0 + i as f64 * (u_range.1 - u_range.0) / (nu - 1) as f64
            };

            (0..nv)
                .map(|j| {
                    let v = if nv == 1 {
                        v_range.0
                    } else if j == nv - 1 {
                        v_range.1
                    } else {
                        v_range.0 + j as f64 * (v_range.1 - v_range.0) / (nv - 1) as f64
                    };
                    surface.evaluate(u, v)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use std::f64::consts::TAU;

    fn unit_cylinder() -> CylindricalSurface {
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap()
    }

    #[test]
    fn zero_nu_returns_empty() {
        let s = unit_cylinder();
        let grid = surface_grid(&s, (0.0, TAU), (0.0, 1.0), 0, 4);
        assert!(grid.is_empty());
    }

    #[test]
    fn zero_nv_returns_empty() {
        let s = unit_cylinder();
        let grid = surface_grid(&s, (0.0, TAU), (0.0, 1.0), 4, 0);
        assert!(grid.is_empty());
    }

    #[test]
    fn grid_dimensions_correct() {
        let s = unit_cylinder();
        let grid = surface_grid(&s, (0.0, TAU), (0.0, 1.0), 5, 3);
        assert_eq!(grid.len(), 5, "expected 5 rows");
        for (i, row) in grid.iter().enumerate() {
            assert_eq!(row.len(), 3, "row {i} should have 3 columns");
        }
    }

    #[test]
    fn all_points_on_cylinder_surface() {
        // Cylinder P(u,v) = (cos u, sin u, v).  All points must satisfy x²+y²=1.
        let s = unit_cylinder();
        let grid = surface_grid(&s, (0.0, TAU), (0.0, 2.0), 8, 4);
        for row in &grid {
            for p in row {
                let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
                assert!((r - 1.0).abs() < 1e-10, "point not on cylinder: r={r}");
            }
        }
    }

    #[test]
    fn first_and_last_u_are_endpoints() {
        let s = unit_cylinder();
        let u0 = 0.0_f64;
        let u1 = TAU;
        let grid = surface_grid(&s, (u0, u1), (0.0, 1.0), 4, 2);
        assert_eq!(grid.len(), 4);

        // First row: u = u0 → point should equal surface.evaluate(u0, 0.0)
        let expected_first = s.evaluate(u0, 0.0);
        let actual_first = grid[0][0];
        let d0 = {
            let dx = actual_first.x() - expected_first.x();
            let dy = actual_first.y() - expected_first.y();
            let dz = actual_first.z() - expected_first.z();
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        assert!(d0 < 1e-12, "first row u mismatch: dist={d0}");

        // Last row: u = u1.
        let expected_last = s.evaluate(u1, 0.0);
        let actual_last = grid[3][0];
        let d1 = {
            let dx = actual_last.x() - expected_last.x();
            let dy = actual_last.y() - expected_last.y();
            let dz = actual_last.z() - expected_last.z();
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        assert!(d1 < 1e-12, "last row u mismatch: dist={d1}");
    }

    #[test]
    fn single_row_and_column() {
        let s = unit_cylinder();
        let grid = surface_grid(&s, (0.0, TAU), (0.5, 1.5), 1, 1);
        assert_eq!(grid.len(), 1);
        assert_eq!(grid[0].len(), 1);
        let expected = s.evaluate(0.0, 0.5);
        let p = grid[0][0];
        let d = {
            let dx = p.x() - expected.x();
            let dy = p.y() - expected.y();
            let dz = p.z() - expected.z();
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        assert!(d < 1e-12);
    }

    #[test]
    fn v_endpoint_included_in_last_column() {
        let s = unit_cylinder();
        let v0 = 0.0_f64;
        let v1 = 3.0_f64;
        let grid = surface_grid(&s, (0.0, 1.0), (v0, v1), 2, 4);
        // Last column (j=3) should have v = v1.
        for (i, row) in grid.iter().enumerate() {
            let expected = s.evaluate(if i == 0 { 0.0 } else { 1.0 }, v1);
            let actual = row[3];
            let d = {
                let dx = actual.x() - expected.x();
                let dy = actual.y() - expected.y();
                let dz = actual.z() - expected.z();
                (dx * dx + dy * dy + dz * dz).sqrt()
            };
            assert!(d < 1e-12, "v endpoint mismatch at row {i}: dist={d}");
        }
    }
}
