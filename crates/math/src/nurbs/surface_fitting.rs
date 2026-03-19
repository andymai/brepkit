//! NURBS surface fitting from a grid of data points via least-squares
//! progressive-iterative approximation (LSPIA).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use crate::MathError;
use crate::nurbs::basis::basis_funs;
use crate::nurbs::fitting;
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

/// Approximate a NURBS surface through a grid of data points using LSPIA.
///
/// Uses tensor-product basis `N_i(u) * N_j(v)` for iterative refinement.
/// This is more efficient than direct least-squares for large grids, achieving
/// O(rows * cols) per iteration.
///
/// # Parameters
///
/// - `points` -- 2D grid of data points (rows x cols)
/// - `degree_u`, `degree_v` -- polynomial degrees in each direction
/// - `num_cps_u`, `num_cps_v` -- number of control points in each direction
/// - `tolerance` -- convergence threshold for max point deviation
/// - `max_iterations` -- maximum number of PIA iterations
///
/// # Errors
///
/// Returns [`MathError::EmptyInput`] if the grid is empty.
/// Returns [`MathError::InvalidControlPointGrid`] if rows have inconsistent lengths.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]
pub fn approximate_surface_lspia(
    points: &[Vec<Point3>],
    degree_u: usize,
    degree_v: usize,
    num_cps_u: usize,
    num_cps_v: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<NurbsSurface, MathError> {
    // Validate input grid.
    if points.is_empty() || points[0].is_empty() {
        return Err(MathError::EmptyInput);
    }
    let rows = points.len();
    let cols = points[0].len();
    for row in points {
        if row.len() != cols {
            return Err(MathError::InvalidControlPointGrid {
                expected_rows: rows,
                expected_cols: cols,
            });
        }
    }

    let pu = degree_u.min(rows - 1);
    let pv = degree_v.min(cols - 1);
    let mu = num_cps_u.min(rows).max(pu + 1);
    let mv = num_cps_v.min(cols).max(pv + 1);

    // Compute parameters for rows (u) and columns (v).
    let params_u = compute_grid_params_u(points, rows, cols);
    let params_v = compute_grid_params_v(points, rows, cols);

    // Build knot vectors.
    let knots_u = fitting::build_approximation_knots(&params_u, pu, mu, rows);
    let knots_v = fitting::build_approximation_knots(&params_v, pv, mv, cols);

    // Initialize control points grid (mu x mv).
    let mut cps: Vec<Vec<Point3>> = Vec::with_capacity(mu);
    for i in 0..mu {
        let mut row = Vec::with_capacity(mv);
        for j in 0..mv {
            let ui = if mu > 1 { i * (rows - 1) / (mu - 1) } else { 0 };
            let vj = if mv > 1 { j * (cols - 1) / (mv - 1) } else { 0 };
            row.push(points[ui.min(rows - 1)][vj.min(cols - 1)]);
        }
        cps.push(row);
    }

    let weights = vec![vec![1.0; mv]; mu];

    // Precompute basis values.
    let basis_u_data: Vec<(usize, Vec<f64>)> = params_u
        .iter()
        .map(|&u| {
            let span = fitting::find_span(u, pu, &knots_u, mu);
            let n = basis_funs(span, u, pu, &knots_u);
            (span, n)
        })
        .collect();
    let basis_v_data: Vec<(usize, Vec<f64>)> = params_v
        .iter()
        .map(|&v| {
            let span = fitting::find_span(v, pv, &knots_v, mv);
            let n = basis_funs(span, v, pv, &knots_v);
            (span, n)
        })
        .collect();

    // Compute step size for LSPIA convergence.
    // Estimate lambda_max per direction via max column sum of squared basis values.
    let mu_u = col_sum_lambda(&basis_u_data, pu, mu);
    let mu_v = col_sum_lambda(&basis_v_data, pv, mv);
    let lambda_max = mu_u * mu_v;
    let step: f64 = if lambda_max < 1e-30 {
        1.0
    } else {
        1.0 / lambda_max
    };

    // Iterate.
    for iter in 0..max_iterations {
        let surface = NurbsSurface::new(
            pu,
            pv,
            knots_u.clone(),
            knots_v.clone(),
            cps.clone(),
            weights.clone(),
        )?;

        let mut max_err = 0.0f64;
        let mut deltas = vec![vec![(0.0f64, 0.0f64, 0.0f64); mv]; mu];

        for (i, (su, nu)) in basis_u_data.iter().enumerate() {
            for (j, (sv, nv)) in basis_v_data.iter().enumerate() {
                let q = surface.evaluate(params_u[i], params_v[j]);
                let err_x = points[i][j].x() - q.x();
                let err_y = points[i][j].y() - q.y();
                let err_z = points[i][j].z() - q.z();
                let err_mag = (err_x * err_x + err_y * err_y + err_z * err_z).sqrt();
                max_err = max_err.max(err_mag);

                for (ku, &bu) in nu.iter().enumerate() {
                    for (kv, &bv) in nv.iter().enumerate() {
                        let ci = su - pu + ku;
                        let cj = sv - pv + kv;
                        if ci < mu && cj < mv {
                            let w = bu * bv;
                            deltas[ci][cj].0 += w * err_x;
                            deltas[ci][cj].1 += w * err_y;
                            deltas[ci][cj].2 += w * err_z;
                        }
                    }
                }
            }
        }

        if max_err < tolerance {
            return NurbsSurface::new(pu, pv, knots_u, knots_v, cps, weights);
        }

        // Update control points: CP += step * delta.
        for i in 0..mu {
            for j in 0..mv {
                cps[i][j] = Point3::new(
                    step.mul_add(deltas[i][j].0, cps[i][j].x()),
                    step.mul_add(deltas[i][j].1, cps[i][j].y()),
                    step.mul_add(deltas[i][j].2, cps[i][j].z()),
                );
            }
        }

        if iter == max_iterations - 1 {
            return NurbsSurface::new(pu, pv, knots_u, knots_v, cps, weights);
        }
    }

    NurbsSurface::new(pu, pv, knots_u, knots_v, cps, weights)
}

/// Compute average chord-length parameters in the u direction (across rows).
#[allow(clippy::cast_precision_loss)]
fn compute_grid_params_u(points: &[Vec<Point3>], rows: usize, cols: usize) -> Vec<f64> {
    let mut params = vec![0.0f64; rows];
    for j in 0..cols {
        let col: Vec<Point3> = (0..rows).map(|i| points[i][j]).collect();
        let col_params = fitting::chord_length_params(&col);
        for (i, &p) in col_params.iter().enumerate() {
            params[i] += p;
        }
    }
    let cols_f = cols as f64;
    for p in &mut params {
        *p /= cols_f;
    }
    params
}

/// Compute average chord-length parameters in the v direction (across columns).
#[allow(clippy::cast_precision_loss)]
fn compute_grid_params_v(points: &[Vec<Point3>], rows: usize, cols: usize) -> Vec<f64> {
    let mut params = vec![0.0f64; cols];
    for i in 0..rows {
        let row_params = fitting::chord_length_params(&points[i]);
        for (j, &p) in row_params.iter().enumerate() {
            params[j] += p;
        }
    }
    let rows_f = rows as f64;
    for p in &mut params {
        *p /= rows_f;
    }
    params
}

/// Estimate the spectral radius of `N^T N` via max column sum of squared basis values.
fn col_sum_lambda(basis_data: &[(usize, Vec<f64>)], p: usize, m: usize) -> f64 {
    let mut col_sums = vec![0.0f64; m];
    for (span, n_vals) in basis_data {
        for (k, &nv) in n_vals.iter().enumerate() {
            let j = span - p + k;
            if j < m {
                col_sums[j] += nv * nv;
            }
        }
    }
    col_sums.iter().copied().fold(0.0f64, f64::max).max(1e-30)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_lossless, clippy::suboptimal_flops)]

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

    // ── LSPIA surface tests ────────────────────────────────────────────

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_surface_fits_plane() {
        let grid: Vec<Vec<Point3>> = (0..10)
            .map(|i| {
                (0..10)
                    .map(|j| {
                        let u = i as f64 / 9.0;
                        let v = j as f64 / 9.0;
                        Point3::new(u, v, 0.5)
                    })
                    .collect()
            })
            .collect();
        let surface = approximate_surface_lspia(&grid, 3, 3, 6, 6, 1e-6, 50).unwrap();
        let p = surface.evaluate(0.5, 0.5);
        assert!(
            (p.z() - 0.5).abs() < 0.01,
            "plane z at center: expected ~0.5, got {}",
            p.z()
        );
    }

    #[test]
    fn lspia_surface_empty_returns_error() {
        let grid: Vec<Vec<Point3>> = Vec::new();
        assert!(approximate_surface_lspia(&grid, 3, 3, 4, 4, 1e-6, 50).is_err());
    }

    #[test]
    fn lspia_surface_inconsistent_rows_returns_error() {
        let grid = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.0)],
        ];
        assert!(approximate_surface_lspia(&grid, 1, 1, 2, 2, 1e-6, 50).is_err());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_surface_fits_paraboloid() {
        let grid: Vec<Vec<Point3>> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| {
                        let u = i as f64 / 7.0;
                        let v = j as f64 / 7.0;
                        Point3::new(u, v, u * u + v * v)
                    })
                    .collect()
            })
            .collect();
        let surface = approximate_surface_lspia(&grid, 3, 3, 6, 6, 1e-4, 100).unwrap();
        let p = surface.evaluate(0.5, 0.5);
        // Expected z = 0.25 + 0.25 = 0.5
        assert!(
            (p.z() - 0.5).abs() < 0.15,
            "paraboloid z at center: expected ~0.5, got {}",
            p.z()
        );
    }
}
