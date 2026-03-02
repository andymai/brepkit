//! NURBS surface fitting from a grid of data points.
//!
//! Equivalent to `GeomAPI_PointsToBSplineSurface` in `OpenCascade`.

use crate::MathError;
use crate::nurbs::fitting::interpolate;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

/// Interpolate a NURBS surface through a grid of data points.
///
/// The input is a row-major grid where `points[i][j]` is the point
/// at row `i`, column `j`. All rows must have the same number of
/// columns. The resulting surface passes through every input point.
///
/// # Parameters
///
/// - `points` — 2D grid of data points (rows × cols)
/// - `degree_u` — polynomial degree in the u (row) direction
/// - `degree_v` — polynomial degree in the v (column) direction
///
/// # Algorithm
///
/// Uses the tensor-product approach:
/// 1. For each row, fit a NURBS curve through the columns
/// 2. Extract control points from those curves (they form a grid)
/// 3. For each column of control points, fit another NURBS curve
/// 4. The final control point grid and combined knot vectors form
///    the interpolated surface
///
/// # Errors
///
/// Returns an error if the grid is too small or rows have inconsistent
/// lengths.
pub fn interpolate_surface(
    points: &[Vec<Point3>],
    degree_u: usize,
    degree_v: usize,
) -> Result<NurbsSurface, MathError> {
    let num_rows = points.len();
    if num_rows < 2 {
        return Err(MathError::EmptyInput);
    }

    let num_cols = points[0].len();
    if num_cols < 2 {
        return Err(MathError::EmptyInput);
    }

    // Validate grid consistency.
    for row in points {
        if row.len() != num_cols {
            return Err(MathError::InvalidControlPointGrid {
                expected_rows: num_rows,
                expected_cols: num_cols,
            });
        }
    }

    // Step 1: Fit curves through each row (u-direction).
    let mut row_curves = Vec::with_capacity(num_rows);
    for row in points {
        let curve = interpolate(row, degree_v)?;
        row_curves.push(curve);
    }

    // All row curves should have the same number of control points
    // (= num_cols, since we're interpolating exactly).
    let n_cp_v = row_curves[0].control_points().len();

    // Step 2: Extract column-wise control points from row curves.
    // cp_grid[col][row] = row_curves[row].control_point[col]
    let mut col_points: Vec<Vec<Point3>> = Vec::with_capacity(n_cp_v);
    for j in 0..n_cp_v {
        let col: Vec<Point3> = row_curves.iter().map(|c| c.control_points()[j]).collect();
        col_points.push(col);
    }

    // Step 3: Fit curves through each column (v-direction).
    let mut col_curves = Vec::with_capacity(n_cp_v);
    for col in &col_points {
        let curve = interpolate(col, degree_u)?;
        col_curves.push(curve);
    }

    // Step 4: Extract the final control point grid and knot vectors.
    let n_cp_u = col_curves[0].control_points().len();
    let knots_u = col_curves[0].knots().to_vec();
    let knots_v = row_curves[0].knots().to_vec();

    let mut control_points: Vec<Vec<Point3>> = Vec::with_capacity(n_cp_u);
    for i in 0..n_cp_u {
        let row: Vec<Point3> = col_curves.iter().map(|c| c.control_points()[i]).collect();
        control_points.push(row);
    }

    let weights: Vec<Vec<f64>> = control_points
        .iter()
        .map(|row| vec![1.0; row.len()])
        .collect();

    NurbsSurface::new(
        degree_u,
        degree_v,
        knots_u,
        knots_v,
        control_points,
        weights,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use crate::tolerance::Tolerance;
    use crate::vec::Point3;

    use super::*;

    #[test]
    fn interpolate_flat_grid() {
        // A 3×3 flat grid on the XY plane.
        let points = vec![
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            ],
            vec![
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, 1.0, 0.0),
            ],
            vec![
                Point3::new(0.0, 2.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, 2.0, 0.0),
            ],
        ];

        let surface = interpolate_surface(&points, 2, 2).unwrap();

        let tol = Tolerance::new();

        // Check corners.
        let p00 = surface.evaluate(0.0, 0.0);
        assert!(tol.approx_eq(p00.x(), 0.0), "corner (0,0) x: {}", p00.x());
        assert!(tol.approx_eq(p00.y(), 0.0), "corner (0,0) y: {}", p00.y());
        assert!(tol.approx_eq(p00.z(), 0.0), "corner (0,0) z: {}", p00.z());

        let p11 = surface.evaluate(1.0, 1.0);
        assert!(tol.approx_eq(p11.x(), 2.0), "corner (1,1) x: {}", p11.x());
        assert!(tol.approx_eq(p11.y(), 2.0), "corner (1,1) y: {}", p11.y());

        // All z should be 0 (flat grid).
        let p_mid = surface.evaluate(0.5, 0.5);
        assert!(
            p_mid.z().abs() < 0.01,
            "flat grid mid z should be ~0, got {}",
            p_mid.z()
        );
    }

    #[test]
    fn interpolate_curved_grid() {
        // A 3×3 grid with z-height forming a paraboloid.
        let points = vec![
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 1.0),
                Point3::new(2.0, 0.0, 4.0),
            ],
            vec![
                Point3::new(0.0, 1.0, 1.0),
                Point3::new(1.0, 1.0, 2.0),
                Point3::new(2.0, 1.0, 5.0),
            ],
            vec![
                Point3::new(0.0, 2.0, 4.0),
                Point3::new(1.0, 2.0, 5.0),
                Point3::new(2.0, 2.0, 8.0),
            ],
        ];

        let surface = interpolate_surface(&points, 2, 2).unwrap();

        let tol = Tolerance::new();
        // Check corners pass through.
        let p00 = surface.evaluate(0.0, 0.0);
        assert!(tol.approx_eq(p00.z(), 0.0), "corner z: {}", p00.z());

        let p11 = surface.evaluate(1.0, 1.0);
        assert!(tol.approx_eq(p11.z(), 8.0), "far corner z: {}", p11.z());
    }

    #[test]
    fn interpolate_surface_too_small() {
        let points = vec![vec![Point3::new(0.0, 0.0, 0.0)]];
        assert!(interpolate_surface(&points, 1, 1).is_err());
    }

    #[test]
    fn interpolate_surface_inconsistent_rows() {
        let points = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.0)], // different length
        ];
        assert!(interpolate_surface(&points, 1, 1).is_err());
    }
}
