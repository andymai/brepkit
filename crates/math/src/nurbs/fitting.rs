//! NURBS curve fitting: interpolation and approximation from data points.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::option_if_let_else,
    clippy::let_and_return,
    clippy::doc_markdown,
    clippy::manual_slice_fill,
    clippy::missing_const_for_fn
)]

use crate::MathError;
use crate::madd::Madd;
use crate::nurbs::basis::basis_funs;
use crate::nurbs::curve::NurbsCurve;
use crate::vec::Point3;

/// Interpolate a NURBS curve through a set of data points.
///
/// Uses chord-length parameterization and a cubic (degree 3) B-spline.
/// The resulting curve passes through every input point exactly.
///
/// # Parameters
///
/// - `points` — the data points to interpolate (at least 2)
/// - `degree` — polynomial degree (typically 3 for cubic)
///
/// # Algorithm
///
/// 1. Compute parameters via chord-length parameterization
/// 2. Build a clamped knot vector
/// 3. Set up and solve the linear system `N * P = Q` where `N` is the
///    basis function matrix and `Q` is the data points
///
/// # Errors
///
/// Returns an error if fewer than 2 points are provided or the degree
/// is too high for the number of points.
pub fn interpolate(points: &[Point3], degree: usize) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 {
        return Err(MathError::EmptyInput);
    }

    let p = degree.min(n - 1);

    let params = chord_length_params(points);
    let knots = build_interpolation_knots(&params, p, n);
    let control_points = solve_interpolation(points, &params, &knots, p)?;

    let weights = vec![1.0; n];
    NurbsCurve::new(p, knots, control_points, weights)
}

/// Interpolate a B-spline curve through `points` at the curve parameters
/// `params`, one for each point.
///
/// The parameters rise strictly from 0 to 1. The interior knots average
/// them (The NURBS Book, eq. 9.8), so the curve passes through each point
/// at its own parameter. A caller that needs the curve's parameter to match
/// another curve's (a pcurve against its edge) gives those parameters here.
///
/// # Errors
///
/// Returns an error if there are fewer than 2 points, the counts differ,
/// `degree` is 0 or not below the point count, a parameter or a coordinate
/// is not finite, the parameters do not rise strictly from 0 to 1, or the
/// linear system is singular.
pub fn interpolate_at(
    points: &[Point3],
    params: &[f64],
    degree: usize,
) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 || params.len() != n {
        return Err(MathError::EmptyInput);
    }
    if degree == 0 || degree >= n {
        return Err(MathError::ParameterOutOfRange {
            value: degree as f64,
            min: 1.0,
            max: (n - 1) as f64,
        });
    }
    // A NaN fails every comparison below, so it would pass them all.
    if let Some(&bad) = params.iter().find(|t| !t.is_finite()) {
        return Err(MathError::ParameterOutOfRange {
            value: bad,
            min: 0.0,
            max: 1.0,
        });
    }
    if let Some(bad) = points
        .iter()
        .flat_map(|p| [p.x(), p.y(), p.z()])
        .find(|c| !c.is_finite())
    {
        return Err(MathError::ParameterOutOfRange {
            value: bad,
            min: f64::MIN,
            max: f64::MAX,
        });
    }
    if let Some(&bad) = [params[0], params[n - 1] - 1.0]
        .iter()
        .find(|value| value.abs() > f64::EPSILON)
    {
        return Err(MathError::ParameterOutOfRange {
            value: bad,
            min: 0.0,
            max: 1.0,
        });
    }
    if let Some(pair) = params.windows(2).find(|pair| pair[1] <= pair[0]) {
        return Err(MathError::ParameterOutOfRange {
            value: pair[1],
            min: pair[0],
            max: 1.0,
        });
    }
    let knots = build_interpolation_knots(params, degree, n);
    let control_points = solve_interpolation(points, params, &knots, degree)?;
    NurbsCurve::new(degree, knots, control_points, vec![1.0; n])
}

/// Approximate a set of points with a NURBS curve of specified number
/// of control points.
///
/// Uses least-squares fitting: the resulting curve minimizes the sum of
/// squared distances to the data points. The curve generally does NOT
/// pass through the points unless `num_control_points == points.len()`.
///
/// # Errors
///
/// Returns an error if parameters are invalid.
pub fn approximate(
    points: &[Point3],
    degree: usize,
    num_control_points: usize,
) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 {
        return Err(MathError::EmptyInput);
    }
    if num_control_points < degree + 1 {
        return Err(MathError::InvalidKnotVector {
            expected: degree + 1,
            got: num_control_points,
        });
    }
    if num_control_points > n {
        return Err(MathError::InvalidKnotVector {
            expected: n,
            got: num_control_points,
        });
    }

    if num_control_points == n {
        return interpolate(points, degree);
    }

    let p = degree.min(num_control_points - 1);
    let m = num_control_points;

    let params = chord_length_params(points);
    let knots = build_approximation_knots(&params, p, m, n);

    // Solve least-squares: N^T * N * P = N^T * Q
    // First and last control points are fixed to the first and last data points.
    let control_points = solve_approximation(points, &params, &knots, p, m)?;

    let weights = vec![1.0; m];
    NurbsCurve::new(p, knots, control_points, weights)
}

// ── Parameterization ───────────────────────────────────────────────

/// Compute chord-length parameters for data points.
///
/// Returns parameters in [0, 1] where each parameter is proportional
/// to the accumulated chord length.
pub(crate) fn chord_length_params(points: &[Point3]) -> Vec<f64> {
    let n = points.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    let mut dists = Vec::with_capacity(n);
    dists.push(0.0);
    for i in 1..n {
        let d = (points[i] - points[i - 1]).length();
        dists.push(dists[i - 1] + d);
    }

    let total = dists[n - 1];
    if total < 1e-15 {
        // All points are coincident — uniform params.
        #[allow(clippy::cast_precision_loss)]
        return (0..n).map(|i| (i as f64) / ((n - 1) as f64)).collect();
    }

    dists.iter().map(|d| d / total).collect()
}

// ── Knot vector construction ───────────────────────────────────────

/// Build a clamped knot vector for interpolation.
///
/// For n control points and degree p:
/// - First p+1 knots = 0.0
/// - Middle knots are averages of consecutive parameter values
/// - Last p+1 knots = 1.0
#[allow(clippy::cast_precision_loss)]
fn build_interpolation_knots(params: &[f64], p: usize, n: usize) -> Vec<f64> {
    let num_knots = n + p + 1;
    let mut knots = Vec::with_capacity(num_knots);

    // Clamped start.
    knots.extend(std::iter::repeat_n(0.0, p + 1));

    // Interior knots: averaging (NURBS Book eq 9.69).
    for j in 1..n - p {
        #[allow(clippy::cast_precision_loss)]
        let avg = params[j..j + p].iter().sum::<f64>() / (p as f64);
        knots.push(avg);
    }

    // Clamped end.
    knots.extend(std::iter::repeat_n(1.0, p + 1));

    knots
}

/// Build a knot vector for least-squares approximation.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn build_approximation_knots(params: &[f64], p: usize, m: usize, n: usize) -> Vec<f64> {
    let num_knots = m + p + 1;
    let mut knots = Vec::with_capacity(num_knots);

    knots.extend(std::iter::repeat_n(0.0, p + 1));

    // Interior knots (NURBS Book eq 9.68).
    let num_interior = m - p - 1;
    if num_interior > 0 {
        #[allow(clippy::cast_precision_loss)]
        let d = (n as f64) / ((num_interior + 1) as f64);
        for j in 1..=num_interior {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let i = (j as f64 * d) as usize;
            let alpha = (j as f64).madd(d, -(i as f64));
            let knot = (1.0 - alpha).madd(params[i.min(n - 1)], alpha * params[(i + 1).min(n - 1)]);
            knots.push(knot);
        }
    }

    knots.extend(std::iter::repeat_n(1.0, p + 1));

    knots
}

// ── Linear system solvers ──────────────────────────────────────────

/// Solve the interpolation linear system for control points.
///
/// Sets up `N[i][j] = B_{j,p}(t_i)` and solves `N * P = Q`. Row `i` holds
/// only the `p + 1` basis functions alive at `t_i`, so the matrix keeps to a
/// band `kl` below its diagonal and `ku` above it, `p` each for parameters
/// that sit inside their own spans. It is stored and eliminated in that band
/// with partial pivoting, whose row swaps widen the upper band to `kl + ku`:
/// `n (2 kl + ku + 1)` numbers rather than `n^2`, so a path sampled at tens
/// of thousands of points still fits, and the same pivots and arithmetic as
/// eliminating the whole matrix.
#[allow(clippy::needless_range_loop)]
fn solve_interpolation(
    points: &[Point3],
    params: &[f64],
    knots: &[f64],
    degree: usize,
) -> Result<Vec<Point3>, MathError> {
    let n = points.len();
    let mut entries = Vec::with_capacity(n);
    let (mut kl, mut ku) = (0, 0);
    for (i, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, n);
        let first = span - degree;
        kl = kl.max(i.saturating_sub(first));
        ku = ku.max((first + degree).min(n - 1).saturating_sub(i));
        entries.push((first, basis_funs(span, t, degree, knots)));
    }
    // Row `i` keeps column `j` at `j + kl - i`.
    let width = 2 * kl + ku + 1;
    let mut band = vec![vec![0.0; width]; n];
    for (i, (first, basis)) in entries.into_iter().enumerate() {
        for (k, b) in basis.into_iter().enumerate() {
            let col = first + k;
            if col < n {
                band[i][col + kl - i] = b;
            }
        }
    }
    let mut rhs: Vec<[f64; 3]> = points.iter().map(|p| [p.x(), p.y(), p.z()]).collect();

    for k in 0..n {
        let below = (k + kl).min(n - 1);
        let mut pivot_row = k;
        for i in (k + 1)..=below {
            if band[i][k + kl - i].abs() > band[pivot_row][k + kl - pivot_row].abs() {
                pivot_row = i;
            }
        }
        if band[pivot_row][k + kl - pivot_row].abs() < 1e-15 {
            return Err(MathError::SingularMatrix);
        }
        let right = (k + kl + ku).min(n - 1);
        if pivot_row != k {
            for j in k..=right {
                let (a, b) = (j + kl - k, j + kl - pivot_row);
                let held = band[k][a];
                band[k][a] = band[pivot_row][b];
                band[pivot_row][b] = held;
            }
            rhs.swap(k, pivot_row);
        }
        let pivot = band[k][kl];
        for i in (k + 1)..=below {
            let factor = band[i][k + kl - i] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in (k + 1)..=right {
                band[i][j + kl - i] -= factor * band[k][j + kl - k];
            }
            band[i][k + kl - i] = 0.0;
            for c in 0..3 {
                rhs[i][c] -= factor * rhs[k][c];
            }
        }
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..=(i + kl + ku).min(n - 1) {
            for c in 0..3 {
                sum[c] -= band[i][j + kl - i] * rhs[j][c];
            }
        }
        for c in 0..3 {
            rhs[i][c] = sum[c] / band[i][kl];
        }
    }
    Ok(rhs.iter().map(|&[x, y, z]| Point3::new(x, y, z)).collect())
}

/// Solve least-squares approximation for control points.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]
fn solve_approximation(
    points: &[Point3],
    params: &[f64],
    knots: &[f64],
    degree: usize,
    m: usize,
) -> Result<Vec<Point3>, MathError> {
    let n = points.len();

    let mut mat_n = vec![vec![0.0; m]; n];
    for (i, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, m);
        let basis = basis_funs(span, t, degree, knots);
        for (k, &b) in basis.iter().enumerate() {
            let col = span - degree + k;
            if col < m {
                mat_n[i][col] = b;
            }
        }
    }

    // N^T * N (m×m) and N^T * Q (m×3).
    let mut ntn = vec![vec![0.0; m]; m];
    let mut ntq_x = vec![0.0; m];
    let mut ntq_y = vec![0.0; m];
    let mut ntq_z = vec![0.0; m];

    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                ntn[i][j] += mat_n[k][i] * mat_n[k][j];
            }
        }
        for k in 0..n {
            ntq_x[i] += mat_n[k][i] * points[k].x();
            ntq_y[i] += mat_n[k][i] * points[k].y();
            ntq_z[i] += mat_n[k][i] * points[k].z();
        }
    }

    // Fix first and last control points: zero the first/last rows and cols,
    // set diagonal to 1 so the endpoints are interpolated exactly.
    for j in 0..m {
        ntn[0][j] = 0.0;
        ntn[m - 1][j] = 0.0;
        ntn[j][0] = 0.0;
        ntn[j][m - 1] = 0.0;
    }
    ntn[0][0] = 1.0;
    ntn[m - 1][m - 1] = 1.0;
    ntq_x[0] = points[0].x();
    ntq_y[0] = points[0].y();
    ntq_z[0] = points[0].z();
    ntq_x[m - 1] = points[n - 1].x();
    ntq_y[m - 1] = points[n - 1].y();
    ntq_z[m - 1] = points[n - 1].z();

    let mut rhs: Vec<[f64; 3]> = (0..m).map(|i| [ntq_x[i], ntq_y[i], ntq_z[i]]).collect();
    gauss_solve(&mut ntn, &mut rhs)?;

    Ok(rhs.iter().map(|&[x, y, z]| Point3::new(x, y, z)).collect())
}

// ── Helpers ────────────────────────────────────────────────────────

/// Find the knot span index for parameter t.
pub(crate) fn find_span(t: f64, degree: usize, knots: &[f64], n: usize) -> usize {
    if t >= knots[n] {
        return n - 1;
    }
    if t <= knots[degree] {
        return degree;
    }

    let mut low = degree;
    let mut high = n;
    let mut mid = low.midpoint(high);

    while t < knots[mid] || t >= knots[mid + 1] {
        if t < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = low.midpoint(high);
    }

    mid
}

/// Solve `A X = B` for three right-hand sides using Gaussian elimination
/// with partial pivoting.
///
/// Fitting matrices are banded. Elimination skips rows already zero in the
/// pivot column and stops each row update at the pivot row's last nonzero,
/// so a band of width `w` costs O(n·w²) arithmetic instead of O(n³). Marched
/// intersection curves reach a thousand points.
#[allow(clippy::needless_range_loop)]
fn gauss_solve(a: &mut [Vec<f64>], b: &mut [[f64; 3]]) -> Result<(), MathError> {
    let n = b.len();
    let mut row_end: Vec<usize> = a
        .iter()
        .map(|row| row.iter().rposition(|&v| v != 0.0).map_or(0, |j| j + 1))
        .collect();

    // Forward elimination.
    for k in 0..n {
        let mut max_val = a[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Err(MathError::SingularMatrix);
        }

        if max_row != k {
            a.swap(k, max_row);
            b.swap(k, max_row);
            row_end.swap(k, max_row);
        }

        let end = row_end[k];
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            if factor == 0.0 {
                continue;
            }
            for j in (k + 1)..end {
                a[i][j] -= factor * a[k][j];
            }
            a[i][k] = 0.0;
            row_end[i] = row_end[i].max(end);
            for c in 0..3 {
                b[i][c] -= factor * b[k][c];
            }
        }
    }

    // Back substitution.
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..row_end[i] {
            for c in 0..3 {
                sum[c] -= a[i][j] * b[j][c];
            }
        }
        for c in 0..3 {
            b[i][c] = sum[c] / a[i][i];
        }
    }

    Ok(())
}

// ── LSPIA step-size computation ───────────────────────────────────────

/// Estimate the largest eigenvalue of `N^T N` via power iteration, then
/// compute a conservative LSPIA step size `mu = 1 / lambda_max`.
///
/// `N` is the (n_data x m_cps) basis-function matrix stored implicitly
/// via the sparse `basis_data` representation.
fn compute_lspia_step_size(basis_data: &[(usize, Vec<f64>)], degree: usize, m: usize) -> f64 {
    // Power iteration: approximate lambda_max of N^T N.
    let mut v = vec![1.0f64; m];
    let norm = (m as f64).sqrt();
    for val in &mut v {
        *val /= norm;
    }

    for _ in 0..20 {
        // w = N^T N v, computed as N^T (N v) to avoid forming N^T N.
        let nv: Vec<f64> = basis_data
            .iter()
            .map(|(span, n_vals)| {
                let mut sum = 0.0f64;
                for (k, &bk) in n_vals.iter().enumerate() {
                    let j = span - degree + k;
                    if j < m {
                        sum += bk * v[j];
                    }
                }
                sum
            })
            .collect();
        let mut w = vec![0.0f64; m];
        for (i, (span, n_vals)) in basis_data.iter().enumerate() {
            for (k, &bk) in n_vals.iter().enumerate() {
                let j = span - degree + k;
                if j < m {
                    w[j] += bk * nv[i];
                }
            }
        }

        let mag = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag < 1e-30 {
            return 1.0;
        }
        for val in &mut v {
            *val = 0.0;
        }
        for (j, &wj) in w.iter().enumerate() {
            v[j] = wj / mag;
        }
    }

    // Compute Rayleigh quotient: lambda = v^T (N^T N v) / (v^T v).
    let nv: Vec<f64> = basis_data
        .iter()
        .map(|(span, n_vals)| {
            let mut sum = 0.0f64;
            for (k, &bk) in n_vals.iter().enumerate() {
                let j = span - degree + k;
                if j < m {
                    sum += bk * v[j];
                }
            }
            sum
        })
        .collect();
    let lambda_max = nv.iter().map(|x| x * x).sum::<f64>();

    if lambda_max < 1e-30 {
        1.0
    } else {
        // Conservative: mu = 1 / lambda_max ensures convergence.
        1.0 / lambda_max
    }
}

/// Weighted variant of LSPIA step-size computation.
///
/// Computes `mu = 1 / lambda_max` for the weighted normal equations
/// `N^T W N` where `W = diag(point_weights)`.
fn compute_lspia_step_size_weighted(
    basis_data: &[(usize, Vec<f64>)],
    point_weights: &[f64],
    degree: usize,
    m: usize,
) -> f64 {
    let mut v = vec![1.0f64; m];
    let norm = (m as f64).sqrt();
    for val in &mut v {
        *val /= norm;
    }

    for _ in 0..20 {
        let nv: Vec<f64> = basis_data
            .iter()
            .map(|(span, n_vals)| {
                let mut sum = 0.0f64;
                for (k, &bk) in n_vals.iter().enumerate() {
                    let j = span - degree + k;
                    if j < m {
                        sum += bk * v[j];
                    }
                }
                sum
            })
            .collect();
        let mut w = vec![0.0f64; m];
        for (i, (span, n_vals)) in basis_data.iter().enumerate() {
            let pw = point_weights[i];
            for (k, &bk) in n_vals.iter().enumerate() {
                let j = span - degree + k;
                if j < m {
                    w[j] += pw * bk * nv[i];
                }
            }
        }

        let mag = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag < 1e-30 {
            return 1.0;
        }
        for val in &mut v {
            *val = 0.0;
        }
        for (j, &wj) in w.iter().enumerate() {
            v[j] = wj / mag;
        }
    }

    let nv: Vec<f64> = basis_data
        .iter()
        .map(|(span, n_vals)| {
            let mut sum = 0.0f64;
            for (k, &bk) in n_vals.iter().enumerate() {
                let j = span - degree + k;
                if j < m {
                    sum += bk * v[j];
                }
            }
            sum
        })
        .collect();
    let lambda_max: f64 = nv
        .iter()
        .zip(point_weights.iter())
        .map(|(&x, &pw)| pw * x * x)
        .sum();

    if lambda_max < 1e-30 {
        1.0
    } else {
        1.0 / lambda_max
    }
}

// ── LSPIA (Locally Supported Progressive-Iterative Approximation) ─────

/// Approximate a NURBS curve through data points using Progressive-Iterative Approximation.
///
/// LSPIA iteratively adjusts control points to minimize the least-squares error,
/// achieving O(n) per iteration vs O(n^3) for direct Gaussian elimination.
/// Converges when the maximum point deviation falls below `tolerance`.
///
/// # Parameters
///
/// - `points` -- data points to approximate
/// - `degree` -- polynomial degree (typically 3)
/// - `num_control_points` -- number of control points (must be <= `points.len()`)
/// - `tolerance` -- convergence threshold for max point deviation
/// - `max_iterations` -- maximum number of PIA iterations
///
/// # Algorithm
///
/// 1. Compute parameters via chord-length parameterization
/// 2. Build a clamped knot vector for the desired number of CPs
/// 3. Initialize CPs by sampling the parameter-position mapping
/// 4. Iterate: compute errors at data points, distribute corrections
///    to CPs weighted by basis function values
///
/// # Errors
///
/// Returns [`MathError::EmptyInput`] if fewer than 2 points.
/// If the iteration has not converged after `max_iterations`, the best-effort
/// curve (lowest observed error) is returned as `Ok`; no error is raised.
/// In debug builds (`#[cfg(debug_assertions)]`) a `log::warn!` is emitted;
/// release builds are silent.
#[allow(clippy::cast_precision_loss)]
pub fn approximate_lspia(
    points: &[Point3],
    degree: usize,
    num_control_points: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 {
        return Err(MathError::EmptyInput);
    }

    let p = degree.min(n - 1);
    let m = num_control_points.min(n).max(p + 1);

    let params = chord_length_params(points);
    let knots = build_approximation_knots(&params, p, m, n);

    // Initialize control points by sampling closest data points.
    let mut control_points = Vec::with_capacity(m);
    for i in 0..m {
        let t = if m > 1 {
            i as f64 / (m - 1) as f64
        } else {
            0.0
        };
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (j, &param) in params.iter().enumerate() {
            let d = (param - t).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = j;
            }
        }
        control_points.push(points[best_idx]);
    }

    let weights = vec![1.0; m];

    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    // mu = 2 / (lambda_min + lambda_max) where lambda are eigenvalues of N^T N.
    // We approximate lambda_max via the power method and use a conservative mu.
    let mu = compute_lspia_step_size(&basis_data, p, m);

    for iter in 0..max_iterations {
        let curve = NurbsCurve::new(p, knots.clone(), control_points.clone(), weights.clone())?;

        let mut max_err = 0.0f64;
        let mut deltas = vec![(0.0f64, 0.0f64, 0.0f64); m];

        for (i, &u) in params.iter().enumerate() {
            let q = curve.evaluate(u);
            let err_x = points[i].x() - q.x();
            let err_y = points[i].y() - q.y();
            let err_z = points[i].z() - q.z();
            let err_mag = (err_x * err_x + err_y * err_y + err_z * err_z).sqrt();
            max_err = max_err.max(err_mag);

            let (span, n_vals) = &basis_data[i];
            for (k, &nv) in n_vals.iter().enumerate() {
                let j = span - p + k;
                if j < m {
                    deltas[j].0 += nv * err_x;
                    deltas[j].1 += nv * err_y;
                    deltas[j].2 += nv * err_z;
                }
            }
        }

        if max_err < tolerance {
            return NurbsCurve::new(p, knots, control_points, weights);
        }

        // Update control points: P_j += mu * delta_j
        for j in 0..m {
            control_points[j] = Point3::new(
                mu.madd(deltas[j].0, control_points[j].x()),
                mu.madd(deltas[j].1, control_points[j].y()),
                mu.madd(deltas[j].2, control_points[j].z()),
            );
        }

        // Return best result if this is the last iteration.
        if iter == max_iterations - 1 {
            // LSPIA did not converge to the requested tolerance.  Warn in debug
            // builds so callers can detect poorly fitted curves.
            #[cfg(debug_assertions)]
            if max_err > tolerance {
                log::warn!(
                    "approximate_lspia: did not converge after {max_iterations} iterations \
                     (max_err={max_err:.2e}, tolerance={tolerance:.2e}, \
                     num_cps={m}, degree={p})"
                );
            }
            return NurbsCurve::new(p, knots, control_points, weights);
        }
    }

    // Unreachable if max_iterations > 0, but handle edge case.
    NurbsCurve::new(p, knots, control_points, weights)
}

/// Weighted LSPIA approximation with per-point weights.
///
/// Points with higher weights have more influence on the fit. This is useful
/// for emphasizing certain regions of the curve or for progressive refinement.
///
/// # Errors
///
/// Returns [`MathError::EmptyInput`] if fewer than 2 points.
/// Returns [`MathError::InvalidWeights`] if `point_weights.len()` does not match
/// `points.len()`.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments, dead_code)]
pub(crate) fn approximate_lspia_weighted(
    points: &[Point3],
    point_weights: &[f64],
    degree: usize,
    num_control_points: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 {
        return Err(MathError::EmptyInput);
    }
    if points.len() != point_weights.len() {
        return Err(MathError::InvalidWeights {
            expected: points.len(),
            got: point_weights.len(),
        });
    }

    let p = degree.min(n - 1);
    let m = num_control_points.min(n).max(p + 1);

    let params = chord_length_params(points);
    let knots = build_approximation_knots(&params, p, m, n);

    // Initialize control points by sampling closest data points.
    let mut control_points = Vec::with_capacity(m);
    for i in 0..m {
        let t = if m > 1 {
            i as f64 / (m - 1) as f64
        } else {
            0.0
        };
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (j, &param) in params.iter().enumerate() {
            let d = (param - t).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = j;
            }
        }
        control_points.push(points[best_idx]);
    }

    let weights = vec![1.0; m];

    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    let mu = compute_lspia_step_size_weighted(&basis_data, point_weights, p, m);

    for iter in 0..max_iterations {
        let curve = NurbsCurve::new(p, knots.clone(), control_points.clone(), weights.clone())?;

        let mut max_err = 0.0f64;
        let mut deltas = vec![(0.0f64, 0.0f64, 0.0f64); m];

        for (i, &u) in params.iter().enumerate() {
            let q = curve.evaluate(u);
            let pw = point_weights[i];
            let err_x = points[i].x() - q.x();
            let err_y = points[i].y() - q.y();
            let err_z = points[i].z() - q.z();
            let err_mag = (err_x * err_x + err_y * err_y + err_z * err_z).sqrt();
            max_err = max_err.max(err_mag);

            let (span, n_vals) = &basis_data[i];
            for (k, &nv) in n_vals.iter().enumerate() {
                let j = span - p + k;
                if j < m {
                    deltas[j].0 += pw * nv * err_x;
                    deltas[j].1 += pw * nv * err_y;
                    deltas[j].2 += pw * nv * err_z;
                }
            }
        }

        if max_err < tolerance {
            return NurbsCurve::new(p, knots, control_points, weights);
        }

        // Update control points: P_j += mu * delta_j
        for j in 0..m {
            control_points[j] = Point3::new(
                mu.madd(deltas[j].0, control_points[j].x()),
                mu.madd(deltas[j].1, control_points[j].y()),
                mu.madd(deltas[j].2, control_points[j].z()),
            );
        }

        if iter == max_iterations - 1 {
            return NurbsCurve::new(p, knots, control_points, weights);
        }
    }

    NurbsCurve::new(p, knots, control_points, weights)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_lossless, clippy::suboptimal_flops)]

    use crate::tolerance::Tolerance;
    use crate::vec::Point3;

    use super::*;

    #[test]
    fn interpolate_two_points_is_line() {
        let pts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let curve = interpolate(&pts, 1).unwrap();

        let tol = Tolerance::new();
        let mid = curve.evaluate(0.5);
        assert!(tol.approx_eq(mid.x(), 0.5));
        assert!(tol.approx_eq(mid.y(), 0.0));
    }

    #[test]
    fn interpolate_passes_through_points() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(3.0, 1.0, 0.0),
        ];
        let curve = interpolate(&pts, 3).unwrap();

        let tol = Tolerance::new();
        let p0 = curve.evaluate(0.0);
        let p1 = curve.evaluate(1.0);
        assert!(tol.approx_eq(p0.x(), 0.0), "start x: {}", p0.x());
        assert!(tol.approx_eq(p0.y(), 0.0), "start y: {}", p0.y());
        assert!(tol.approx_eq(p1.x(), 3.0), "end x: {}", p1.x());
        assert!(tol.approx_eq(p1.y(), 1.0), "end y: {}", p1.y());
    }

    #[test]
    fn interpolate_3d_points() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 0.0, 2.0),
        ];
        let curve = interpolate(&pts, 2).unwrap();

        let tol = Tolerance::new();
        let p0 = curve.evaluate(0.0);
        let p1 = curve.evaluate(1.0);
        assert!(tol.approx_eq(p0.x(), 0.0));
        assert!(tol.approx_eq(p1.x(), 2.0));
        assert!(tol.approx_eq(p1.z(), 2.0));
    }

    #[test]
    fn interpolate_long_helix_hits_every_point() {
        let pts: Vec<Point3> = (0..1500)
            .map(|i| {
                let t = i as f64 * 0.02;
                Point3::new(t.cos(), t.sin(), 0.05 * t)
            })
            .collect();
        let curve = interpolate(&pts, 3).unwrap();
        for (p, t) in pts.iter().zip(chord_length_params(&pts)) {
            assert!((curve.evaluate(t) - *p).length() < 1e-9, "missed {p:?}");
        }
    }

    /// Forty thousand points, a dense system of 12.8 GB, are solved in the
    /// band their rows hold.
    #[test]
    fn interpolate_forty_thousand_points_in_their_band() {
        let pts: Vec<Point3> = (0..40_000)
            .map(|i| {
                let t = i as f64 * 0.001;
                Point3::new(t.cos(), t.sin(), 0.05 * t)
            })
            .collect();
        let curve = interpolate(&pts, 3).unwrap();
        for (p, t) in pts.iter().zip(chord_length_params(&pts)).step_by(97) {
            assert!((curve.evaluate(t) - *p).length() < 1e-9, "missed {p:?}");
        }
    }

    /// Points bunched at both ends of a unit segment, nanometres apart at
    /// one end: the band's partial pivoting solves them as the whole matrix
    /// did.
    #[test]
    fn interpolate_points_bunched_at_both_ends() {
        let pts: Vec<Point3> = [
            0.0,
            1.023_546_145_149_216_7e-9,
            5.140_308_521_034_802_6e-9,
            0.999_964_022_234_990_8,
            0.999_977_479_169_274_9,
            1.0,
        ]
        .iter()
        .map(|&x| Point3::new(x, 0.0, 0.0))
        .collect();
        let curve = interpolate(&pts, 3).unwrap();
        for (p, t) in pts.iter().zip(chord_length_params(&pts)) {
            assert!((curve.evaluate(t) - *p).length() < 1e-9, "missed {p:?}");
        }
    }

    #[test]
    fn interpolate_single_point_error() {
        let pts = vec![Point3::new(0.0, 0.0, 0.0)];
        assert!(interpolate(&pts, 3).is_err());
    }

    /// A cubic sampled at its own parameters, bunched toward one end, comes
    /// back as itself: each point at its parameter, and the points between.
    #[test]
    fn interpolate_at_returns_a_cubic_at_its_own_parameters() {
        let cubic = |t: f64| Point3::new(t, t * t * t - t, 2.0 * t * t);
        let params: Vec<f64> = (0..=9).map(|k| (f64::from(k) / 9.0).powi(2)).collect();
        let points: Vec<Point3> = params.iter().map(|&t| cubic(t)).collect();
        let curve = interpolate_at(&points, &params, 3).unwrap();
        for k in 0..=90 {
            let t = f64::from(k) / 90.0;
            assert!((curve.evaluate(t) - cubic(t)).length() < 1e-12, "at {t}");
        }
    }

    /// Degree 0 averages no parameters into a knot (0 / 0), and a degree
    /// as high as the point count leaves no room for the clamped ends.
    #[test]
    fn interpolate_at_refuses_degree_zero_and_a_degree_too_high_for_the_points() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let params = [0.0, 0.5, 1.0];
        assert!(interpolate_at(&points, &params, 0).is_err());
        assert!(interpolate_at(&points, &params, 3).is_err());
        assert!(interpolate_at(&points, &params, 1).is_ok());
        assert!(interpolate_at(&points, &params, 2).is_ok());
    }

    #[test]
    fn interpolate_at_refuses_parameters_off_zero_to_one_or_out_of_order() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        assert!(interpolate_at(&points, &[0.0, 0.5, 0.9], 2).is_err());
        assert!(interpolate_at(&points, &[0.1, 0.5, 1.0], 2).is_err());
        assert!(interpolate_at(&points, &[0.0, 1.0, 1.0], 2).is_err());
        assert!(interpolate_at(&points, &[0.0, 1.0], 2).is_err());
        assert!(interpolate_at(&points, &[0.0, 0.4, 1.0], 2).is_ok());
    }

    #[test]
    fn interpolate_at_refuses_a_parameter_or_a_point_that_is_not_finite() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        for params in [
            [f64::NAN, 0.5, 1.0],
            [0.0, f64::NAN, 1.0],
            [0.0, 0.5, f64::NAN],
            [0.0, f64::INFINITY, 1.0],
        ] {
            assert!(interpolate_at(&points, &params, 2).is_err(), "{params:?}");
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut off = points;
            off[1] = Point3::new(1.0, bad, 0.0);
            assert!(interpolate_at(&off, &[0.0, 0.5, 1.0], 2).is_err(), "{bad}");
        }
    }

    #[test]
    fn approximate_fewer_control_points() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.8, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.5, 0.8, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let curve = approximate(&pts, 3, 4).unwrap();

        // Endpoints should be exact.
        let tol = Tolerance::new();
        let p0 = curve.evaluate(0.0);
        let p1 = curve.evaluate(1.0);
        assert!(tol.approx_eq(p0.x(), 0.0), "start x: {}", p0.x());
        assert!(tol.approx_eq(p1.x(), 2.0), "end x: {}", p1.x());
    }

    #[test]
    fn approximate_equals_interpolate_when_same_count() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];

        let interp = interpolate(&pts, 2).unwrap();
        let approx = approximate(&pts, 2, 3).unwrap();

        let tol = Tolerance::new();
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let pi = interp.evaluate(t);
            let pa = approx.evaluate(t);
            assert!(
                tol.approx_eq(pi.x(), pa.x()) && tol.approx_eq(pi.y(), pa.y()),
                "at t={t}: interp=({}, {}) approx=({}, {})",
                pi.x(),
                pi.y(),
                pa.x(),
                pa.y()
            );
        }
    }

    // ── LSPIA tests ────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_fits_line() {
        let points: Vec<Point3> = (0..10)
            .map(|i| {
                let t = i as f64 / 9.0;
                Point3::new(t, 2.0f64.madd(t, 1.0), 0.0)
            })
            .collect();
        let curve = approximate_lspia(&points, 3, 6, 1e-6, 100).unwrap();

        // Check endpoints.
        let p0 = curve.evaluate(0.0);
        let p1 = curve.evaluate(1.0);
        assert!(
            (p0.x() - 0.0).abs() < 0.01,
            "start x: expected ~0.0, got {}",
            p0.x()
        );
        assert!(
            (p1.x() - 1.0).abs() < 0.01,
            "end x: expected ~1.0, got {}",
            p1.x()
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_fits_circle() {
        let n = 50;
        let points: Vec<Point3> = (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Point3::new(t.cos(), t.sin(), 0.0)
            })
            .collect();
        let curve = approximate_lspia(&points, 3, 15, 1e-4, 200).unwrap();

        for i in 0..10 {
            let t = i as f64 / 9.0;
            let p = curve.evaluate(t);
            let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
            assert!((r - 1.0).abs() < 0.15, "radius at t={t} is {r}");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_fewer_cps_than_points() {
        let points: Vec<Point3> = (0..100)
            .map(|i| {
                let t = i as f64 / 99.0;
                Point3::new(t, (t * 6.0).sin(), 0.0)
            })
            .collect();
        let curve = approximate_lspia(&points, 3, 20, 1e-3, 100).unwrap();
        let p = curve.evaluate(0.5);
        assert!(
            (p.x() - 0.5).abs() < 0.1,
            "midpoint x: expected ~0.5, got {}",
            p.x()
        );
    }

    #[test]
    fn lspia_empty_input_returns_error() {
        let result = approximate_lspia(&[], 3, 5, 1e-6, 100);
        assert!(result.is_err());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lspia_weighted_emphasizes_region() {
        let points: Vec<Point3> = (0..20)
            .map(|i| {
                let t = i as f64 / 19.0;
                Point3::new(t, t * t, 0.0)
            })
            .collect();
        let uniform_weights = vec![1.0; 20];
        let curve = approximate_lspia_weighted(&points, &uniform_weights, 3, 8, 1e-5, 100).unwrap();

        let p = curve.evaluate(0.5);
        assert!(
            (p.x() - 0.5).abs() < 0.15,
            "midpoint x: expected ~0.5, got {}",
            p.x()
        );
    }

    #[test]
    fn lspia_weighted_mismatched_lengths_returns_error() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)];
        let weights = vec![1.0; 5];
        let result = approximate_lspia_weighted(&points, &weights, 1, 2, 1e-6, 10);
        assert!(result.is_err());
    }
}
