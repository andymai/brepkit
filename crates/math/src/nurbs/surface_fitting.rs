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
use crate::madd::Madd;
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
/// lengths, and [`MathError::TooMuchWork`] if the fits would take more than
/// [`fitting::MAX_FIT_WORK`] multiply-adds.
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

    for row in points {
        if row.len() != num_cols {
            return Err(MathError::InvalidControlPointGrid {
                expected_rows: num_rows,
                expected_cols: num_cols,
            });
        }
    }
    let (pu, pv) = (degree_u.min(num_rows - 1), degree_v.min(num_cols - 1));
    fitting::check_fit_work(
        "a surface interpolation",
        (num_rows as f64).madd(
            fitting::interpolation_work(num_cols, pv),
            num_cols as f64 * fitting::interpolation_work(num_rows, pu),
        ),
    )?;
    // Each row is interpolated by its chord lengths, so a point repeated
    // along one, or one that is not a number, is refused before any is
    // fitted. The second stage interpolates the columns of the row curves'
    // control points, not the data's, and `interpolate` reads each of those.
    for row in points {
        fitting::check_rising(&fitting::chord_length_params(row))?;
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
    #[cfg(test)]
    {
        let curves = |curves: &[crate::nurbs::curve::NurbsCurve]| {
            cells::<crate::nurbs::curve::NurbsCurve>(curves.len())
                + curves
                    .iter()
                    .map(|c| c.knots().len() + 3 * c.control_points().len() + c.weights().len())
                    .sum::<usize>()
        };
        // The most one curve's interpolation held, noted by it.
        let inner = fitting::HELD_CELLS.with(std::cell::Cell::get);
        fitting::note_held(
            inner
                + rows_cells(points)
                + curves(&row_curves)
                + rows_cells(&col_points)
                + curves(&col_curves)
                + rows_cells(&control_points)
                + rows_cells(&weights)
                + knots_u.len()
                + knots_v.len(),
        );
    }

    // Each curve takes its degree at most one under its points, so the
    // surface takes the degrees its curves took.
    NurbsSurface::new(pu, pv, knots_u, knots_v, control_points, weights)
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
/// Returns [`MathError::TooMuchWork`] if `max_iterations` passes would take
/// more than [`fitting::MAX_FIT_WORK`] multiply-adds.
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
    fitting::check_fit_work(
        "an LSPIA surface fit",
        lspia_surface_work([rows, cols], [mu, mv], [pu, pv], max_iterations),
    )?;

    let params_u = compute_grid_params_u(points, rows, cols);
    let params_v = compute_grid_params_v(points, rows, cols);
    fitting::check_finite(&params_u)?;
    fitting::check_finite(&params_v)?;

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
        #[cfg(test)]
        HELD_CELLS.with(|most| {
            let held = rows_cells(points)
                + cells::<f64>(params_u.capacity() + params_v.capacity())
                + cells::<f64>(knots_u.capacity() + knots_v.capacity())
                + rows_cells(&cps)
                + rows_cells(&weights)
                + basis_cells(&basis_u_data)
                + basis_cells(&basis_v_data)
                + cells::<f64>(surface.knots_u().len() + surface.knots_v().len())
                + rows_cells(surface.control_points())
                + rows_cells(surface.weights())
                + rows_cells(&deltas);
            most.set(most.get().max(held));
        });

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
                    step.madd(deltas[i][j].0, cps[i][j].x()),
                    step.madd(deltas[i][j].1, cps[i][j].y()),
                    step.madd(deltas[i][j].2, cps[i][j].z()),
                );
            }
        }

        if iter == max_iterations - 1 {
            return NurbsSurface::new(pu, pv, knots_u, knots_v, cps, weights);
        }
    }

    NurbsSurface::new(pu, pv, knots_u, knots_v, cps, weights)
}

/// The multiply-adds of an LSPIA fit of a grid of `rows` by `cols` points
/// by `mu` by `mv` control points at degrees `pu` and `pv` over `iterations`
/// passes: the start, which reads the grid for its parameters, then on each
/// pass the surface at every point and the corrections it spreads.
fn lspia_surface_work(
    [rows, cols]: [usize; 2],
    [mu, mv]: [usize; 2],
    [pu, pv]: [usize; 2],
    iterations: usize,
) -> f64 {
    let (rows, cols, mu, mv) = (rows as f64, cols as f64, mu as f64, mv as f64);
    let (bu, bv) = ((pu + 1) as f64, (pv + 1) as f64);
    let start = 2.0 * rows * cols + rows * bu * bu + cols * bv * bv + mu * mv;
    let pass = (rows * cols).madd(bu * bu + bv * bv + 7.0 * bu * bv, 5.0 * mu * mv);
    (iterations as f64).madd(pass, start)
}

/// The most cells of eight bytes an interpolation of a grid of `rows` by
/// `cols` points holds at once ([`interpolate_surface`]), the grid
/// included; `None` past `usize`.
///
/// They are the grid, three cells a point and three for each row's header;
/// each row's curve and each column's, with their knots, points and
/// weights; the columns of the row curves' points; the net and its
/// weights; and the most one curve's interpolation holds
/// ([`fitting::interpolation_cells`]).
#[must_use]
pub fn interpolate_surface_cells(
    [rows, cols]: [usize; 2],
    [degree_u, degree_v]: [usize; 2],
) -> Option<usize> {
    if rows < 2 || cols < 2 {
        return Some(0);
    }
    let (pu, pv) = (degree_u.min(rows - 1), degree_v.min(cols - 1));
    let curve = std::mem::size_of::<crate::nurbs::curve::NurbsCurve>().div_ceil(8);
    let points = rows.checked_mul(cols)?;
    // Each curve's points and weights hold four cells a point.
    let row_curves = rows.checked_mul(cols.checked_mul(5)?.checked_add(pv + 1 + curve)?)?;
    let col_curves = cols.checked_mul(rows.checked_mul(5)?.checked_add(pu + 1 + curve)?)?;
    let inner =
        fitting::interpolation_cells(cols, pv)?.max(fitting::interpolation_cells(rows, pu)?);
    points
        .checked_mul(10)?
        .checked_add(rows.checked_mul(9)?)?
        .checked_add(cols.checked_mul(3)?)?
        .checked_add(row_curves)?
        .checked_add(col_curves)?
        .checked_add(rows + pu + 1 + cols + pv + 1)?
        .checked_add(inner)
}

/// The most cells of eight bytes an LSPIA fit of a grid of `rows` by `cols`
/// points ([`approximate_surface_lspia`]) holds at once, the grid included,
/// counted from its sizes; `None` past `usize`.
///
/// The grid holds three cells a point and three for each row's header. The
/// parameters, read first, hold each direction's count and one line's
/// chords at a time, at most `2 rows + 2 cols`. The passes then hold more:
/// the parameters, each one's span and basis values with a header of four,
/// the knots, the control net and its weights with their rows' headers, a
/// copy of those for the surface each pass reads, the corrections, and the
/// basis each point's evaluation reads, with three cells for each of their
/// headers. With the outer headers and the surface's caches, that is at
/// most:
///
/// `rows (pu + 6) + cols (pv + 6) + 11 mu mv + 17 mu + 2 mv + 3 (pu + pv) + 72`.
#[must_use]
pub fn lspia_surface_cells(
    [rows, cols]: [usize; 2],
    [degree_u, degree_v]: [usize; 2],
    [num_cps_u, num_cps_v]: [usize; 2],
) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let (pu, pv) = (degree_u.min(rows - 1), degree_v.min(cols - 1));
    let mu = num_cps_u.min(rows).max(pu + 1);
    let mv = num_cps_v.min(cols).max(pv + 1);
    let grid = cols.checked_mul(3)?.checked_add(3)?.checked_mul(rows)?;
    rows.checked_mul(pu.checked_add(6)?)?
        .checked_add(cols.checked_mul(pv.checked_add(6)?)?)?
        .checked_add(mu.checked_mul(mv)?.checked_mul(11)?)?
        .checked_add(mu.checked_mul(17)?)?
        .checked_add(mv.checked_mul(2)?)?
        .checked_add(pu.checked_add(pv)?.checked_mul(3)?.checked_add(72)?)?
        .checked_add(grid)
}

#[cfg(test)]
thread_local! {
    /// The most cells an LSPIA surface fit held at once on this thread, read
    /// from its buffers' capacities.
    static HELD_CELLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The cells of eight bytes `count` items of `T` take.
#[cfg(test)]
const fn cells<T>(count: usize) -> usize {
    (count * std::mem::size_of::<T>()).div_ceil(8)
}

/// The cells a grid of rows takes, its rows' headers included.
#[cfg(test)]
fn rows_cells<T>(rows: &[Vec<T>]) -> usize {
    cells::<Vec<T>>(rows.len())
        + rows
            .iter()
            .map(|row| cells::<T>(row.capacity()))
            .sum::<usize>()
}

/// The cells each parameter's span and basis values take.
#[cfg(test)]
fn basis_cells(data: &[(usize, Vec<f64>)]) -> usize {
    cells::<(usize, Vec<f64>)>(data.len())
        + data
            .iter()
            .map(|(_, values)| cells::<f64>(values.capacity()))
            .sum::<usize>()
}

/// Compute average chord-length parameters in the u direction (across rows).
#[allow(clippy::cast_precision_loss)]
fn compute_grid_params_u(points: &[Vec<Point3>], rows: usize, cols: usize) -> Vec<f64> {
    let mut params = vec![0.0f64; rows];
    for j in 0..cols {
        let col_params = fitting::chord_length_params_by(rows, |i| points[i][j]);
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

    fn swell(rows: usize, cols: usize) -> Vec<Vec<Point3>> {
        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| {
                        let (u, v) = (i as f64 / (rows - 1) as f64, j as f64 / (cols - 1) as f64);
                        Point3::new(u * 10.0, v * 10.0, (u * 5.0).sin() * (v * 4.0).cos())
                    })
                    .collect()
            })
            .collect()
    }

    /// The largest surface fits within the wasm bindings' memory limits
    /// (2^22 cells, degree 25, 256 control points a side, 10,000
    /// iterations) are refused before they start: an interpolation of
    /// 41,120 by 41,120 points at degree 25 counts 7.3e12 multiply-adds, one
    /// of 600 by 600 points 1.5e9, and an LSPIA fit at degree 25 of even 26
    /// by 26 points over 10,000 iterations 4e10.
    #[test]
    fn the_largest_surface_fits_within_the_memory_limits_are_refused() {
        let side = (1 << 22) / (4 * 25 + 2);
        let largest = 2.0 * side as f64 * fitting::interpolation_work(side, 25);
        assert!(largest > fitting::MAX_FIT_WORK, "{largest:e}");
        let refused = |found: &Result<NurbsSurface, MathError>| {
            matches!(found, Err(MathError::TooMuchWork { .. }))
        };
        let started = std::time::Instant::now();
        let found = interpolate_surface(&swell(600, 600), 25, 25);
        assert!(refused(&found), "{found:?}");
        let found = approximate_surface_lspia(&swell(26, 26), 25, 25, 256, 256, 1e-6, 10_000);
        assert!(refused(&found), "{found:?}");
        assert!(started.elapsed().as_secs() < 1, "{:?}", started.elapsed());
    }

    /// A grid whose fifth row repeats its fourth is refused, as its columns
    /// of control points repeat too, and one with a point that is not a
    /// number is refused before any row is fitted.
    #[test]
    fn a_grid_with_a_repeated_row_or_an_unreadable_point_is_refused() {
        let refused = |found: &Result<NurbsSurface, MathError>| {
            matches!(found, Err(MathError::ParameterOutOfRange { .. }))
        };
        let mut repeated = swell(40, 40);
        repeated[4] = repeated[3].clone();
        let found = interpolate_surface(&repeated, 3, 3);
        assert!(refused(&found), "{found:?}");
        let mut unread = swell(40, 40);
        unread[7][9] = Point3::new(0.0, f64::NAN, 0.0);
        for found in [
            interpolate_surface(&unread, 3, 3),
            approximate_surface_lspia(&unread, 3, 3, 10, 10, 1e-6, 50),
        ] {
            assert!(refused(&found), "{found:?}");
        }
    }

    /// The flat 3 by 3 grid `(j, i, 0)` with its centre moved to (1, 0, 0):
    /// its middle column of data repeats a point, but the column of the row
    /// curves' control points that the second stage interpolates does not,
    /// so the fit stands, the same as its two stages by hand. Rows that
    /// repeat make a column of control points repeat, which is refused.
    #[test]
    fn a_grid_is_read_by_the_columns_its_second_stage_interpolates() {
        let mut grid: Vec<Vec<Point3>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| Point3::new(f64::from(j), f64::from(i), 0.0))
                    .collect()
            })
            .collect();
        grid[1][1] = Point3::new(1.0, 0.0, 0.0);
        let surface = interpolate_surface(&grid, 2, 2).unwrap();
        let rows: Vec<crate::nurbs::curve::NurbsCurve> = grid
            .iter()
            .map(|row| interpolate(row, 2).unwrap())
            .collect();
        let columns: Vec<crate::nurbs::curve::NurbsCurve> = (0..3)
            .map(|j| {
                let column: Vec<Point3> = rows.iter().map(|c| c.control_points()[j]).collect();
                interpolate(&column, 2).unwrap()
            })
            .collect();
        for (i, row) in surface.control_points().iter().enumerate() {
            for (j, &point) in row.iter().enumerate() {
                let by_hand = columns[j].control_points()[i];
                let bits = |p: Point3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
                assert_eq!(bits(point), bits(by_hand), "{i} {j}");
            }
        }
        let mut repeated = swell(6, 6);
        repeated[3] = repeated[2].clone();
        let found = interpolate_surface(&repeated, 2, 2);
        assert!(
            matches!(found, Err(MathError::ParameterOutOfRange { .. })),
            "{found:?}"
        );
    }

    /// A grid two points wide and one 100 wide, each with as many rows as
    /// its count holds within 2^22 cells at degree 3 by 8 by 8 control
    /// points: the fit's buffers at their most, read from their
    /// capacities, stay within the count, and the count within 128 cells of
    /// them. The count the binding charged, three cells a point, each
    /// side's basis values and eleven cells a control point, held the grid
    /// two points wide to 10 cells a row against the 18 it holds.
    #[test]
    fn an_lspia_surface_fit_holds_at_most_its_count() {
        let limit = 1_usize << 22;
        for cols in [2, 100] {
            let count = |rows: usize| lspia_surface_cells([rows, cols], [3, 3], [8, 8]).unwrap();
            let (mut low, mut high) = (1, limit);
            while high - low > 1 {
                let mid = low.midpoint(high);
                if count(mid) <= limit {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            let rows = low;
            let grid: Vec<Vec<Point3>> = (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let z = ((i * 7 + j * 3) % 11) as f64 * 0.1;
                            Point3::new(i as f64, j as f64, z)
                        })
                        .collect()
                })
                .collect();
            HELD_CELLS.with(|held| held.set(0));
            approximate_surface_lspia(&grid, 3, 3, 8, 8, 1e-12, 1).unwrap();
            let held = HELD_CELLS.with(std::cell::Cell::get);
            let counted = count(rows);
            assert!(
                held <= counted && counted <= limit && counted - held <= 128,
                "{rows} by {cols}: held {held}, counted {counted}"
            );
        }
    }
    /// Surface interpolations of grids of 40 by 30 and 3 by 200 points at
    /// degrees 1, 3 and 25 by 3: the most each holds at once, read from its
    /// buffers' capacities and with the most one curve's interpolation held,
    /// is its count.
    #[test]
    fn a_surface_interpolation_holds_at_most_its_count() {
        for (rows, cols) in [(40, 30), (3, 200)] {
            for (degree_u, degree_v) in [(1, 1), (3, 3), (25, 3)] {
                let grid: Vec<Vec<Point3>> = (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| {
                                let z = ((i * 7 + j * 3) % 11) as f64 * 0.1;
                                Point3::new(i as f64, j as f64, z)
                            })
                            .collect()
                    })
                    .collect();
                fitting::HELD_CELLS.with(|most| most.set(0));
                drop(interpolate_surface(&grid, degree_u, degree_v));
                let held = fitting::HELD_CELLS.with(std::cell::Cell::get);
                let count = interpolate_surface_cells([rows, cols], [degree_u, degree_v]).unwrap();
                assert_eq!(held, count, "{rows} by {cols} at {degree_u}, {degree_v}");
            }
        }
    }
    /// A grid of 3 rows of 5 points, interpolated at degree 3 each way, in
    /// three poses. The columns of 3 points take degree 2, the most 3
    /// points allow, as their interpolation does, and the surface was given
    /// degree 3 against their knots, which it refused. It now takes degrees
    /// 2 and 3, its net is the columns' fits of the rows' fits bit for bit,
    /// and its corners are the grid's.
    #[test]
    fn a_grid_with_fewer_rows_than_its_degree_takes_the_degree_its_columns_allow() {
        let poses: [&dyn Fn(Point3) -> Point3; 3] = [
            &|p| p,
            &|p| Point3::new(-p.y() + 3.0, p.x() - 2.0, p.z() + 1.0),
            &|p| Point3::new(p.x() + 1e3, p.z() - 5.0, -p.y()),
        ];
        for pose in poses {
            let grid: Vec<Vec<Point3>> = (0..3)
                .map(|i| {
                    (0..5)
                        .map(|j| {
                            let (u, v) = (f64::from(i), f64::from(j));
                            pose(Point3::new(u, v, (u * 0.7).sin() * (v * 0.5).cos()))
                        })
                        .collect()
                })
                .collect();
            let surface = interpolate_surface(&grid, 3, 3).unwrap();
            assert_eq!((surface.degree_u(), surface.degree_v()), (2, 3));
            let rows: Vec<crate::nurbs::curve::NurbsCurve> = grid
                .iter()
                .map(|row| interpolate(row, 3).unwrap())
                .collect();
            let bits = |p: Point3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
            for j in 0..5 {
                let column: Vec<Point3> = rows.iter().map(|c| c.control_points()[j]).collect();
                let fitted = interpolate(&column, 3).unwrap();
                assert_eq!(fitted.degree(), 2);
                for i in 0..3 {
                    assert_eq!(
                        bits(surface.control_points()[i][j]),
                        bits(fitted.control_points()[i])
                    );
                }
            }
            for (u, v, i, j) in [
                (0.0, 0.0, 0, 0),
                (1.0, 0.0, 2, 0),
                (0.0, 1.0, 0, 4),
                (1.0, 1.0, 2, 4),
            ] {
                assert!((surface.evaluate(u, v) - grid[i][j]).length() < 1e-9);
            }
        }
    }
}
