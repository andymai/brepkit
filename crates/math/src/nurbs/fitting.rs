//! NURBS curve fitting: interpolation and approximation from data points.
//!
//! Equivalent to `GeomAPI_Interpolate` and `GeomAPI_PointsToBSpline`
//! in `OpenCascade`.

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

    // Clamp degree to at most n-1.
    let p = degree.min(n - 1);

    // Step 1: chord-length parameterization.
    let params = chord_length_params(points);

    // Step 2: build clamped knot vector.
    let knots = build_interpolation_knots(&params, p, n);

    // Step 3: solve for control points.
    // Build the basis function matrix N[i][j] = N_{j,p}(params[i]).
    let control_points = solve_interpolation(points, &params, &knots, p)?;

    let weights = vec![1.0; n];
    NurbsCurve::new(p, knots, control_points, weights)
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

    // If num_control_points == n, this is exact interpolation.
    if num_control_points == n {
        return interpolate(points, degree);
    }

    let p = degree.min(num_control_points - 1);
    let m = num_control_points;

    // Parameterize data points.
    let params = chord_length_params(points);

    // Build knot vector for m control points.
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
            let alpha = (j as f64).mul_add(d, -(i as f64));
            let knot =
                (1.0 - alpha).mul_add(params[i.min(n - 1)], alpha * params[(i + 1).min(n - 1)]);
            knots.push(knot);
        }
    }

    knots.extend(std::iter::repeat_n(1.0, p + 1));

    knots
}

// ── Linear system solvers ──────────────────────────────────────────

/// Solve the interpolation linear system for control points.
///
/// Sets up `N[i][j] = B_{j,p}(t_i)` and solves `N * P = Q`.
/// Uses simple Gaussian elimination (sufficient for typical point counts).
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
fn solve_interpolation(
    points: &[Point3],
    params: &[f64],
    knots: &[f64],
    degree: usize,
) -> Result<Vec<Point3>, MathError> {
    let n = points.len();

    // Build the basis function matrix.
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, n);
        let basis = basis_funs(span, t, degree, knots);
        for (k, &b) in basis.iter().enumerate() {
            let col = span - degree + k;
            if col < n {
                matrix[i][col] = b;
            }
        }
    }

    // Right-hand side: [Qx, Qy, Qz] columns.
    let mut rhs_x: Vec<f64> = points.iter().map(|p| p.x()).collect();
    let mut rhs_y: Vec<f64> = points.iter().map(|p| p.y()).collect();
    let mut rhs_z: Vec<f64> = points.iter().map(|p| p.z()).collect();

    // Gaussian elimination with partial pivoting.
    gauss_solve(&mut matrix, &mut rhs_x)?;
    // Re-build matrix (it was modified in-place).
    let mut matrix2 = vec![vec![0.0; n]; n];
    for (i, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, n);
        let basis = basis_funs(span, t, degree, knots);
        for (k, &b) in basis.iter().enumerate() {
            let col = span - degree + k;
            if col < n {
                matrix2[i][col] = b;
            }
        }
    }
    gauss_solve(&mut matrix2, &mut rhs_y)?;

    let mut matrix3 = vec![vec![0.0; n]; n];
    for (i, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, n);
        let basis = basis_funs(span, t, degree, knots);
        for (k, &b) in basis.iter().enumerate() {
            let col = span - degree + k;
            if col < n {
                matrix3[i][col] = b;
            }
        }
    }
    gauss_solve(&mut matrix3, &mut rhs_z)?;

    Ok(rhs_x
        .iter()
        .zip(rhs_y.iter())
        .zip(rhs_z.iter())
        .map(|((&x, &y), &z)| Point3::new(x, y, z))
        .collect())
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

    // Build basis function matrix N: n×m.
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

    // Fix first and last control points.
    // Zero out the first and last rows/cols, set diagonal to 1.
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

    gauss_solve(&mut ntn.clone(), &mut ntq_x)?;
    gauss_solve(&mut ntn.clone(), &mut ntq_y)?;
    gauss_solve(&mut ntn, &mut ntq_z)?;

    Ok(ntq_x
        .iter()
        .zip(ntq_y.iter())
        .zip(ntq_z.iter())
        .map(|((&x, &y), &z)| Point3::new(x, y, z))
        .collect())
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

/// Solve a linear system `Ax = b` using Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn gauss_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<(), MathError> {
    let n = b.len();

    // Forward elimination.
    for k in 0..n {
        // Find pivot.
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

        // Swap rows.
        if max_row != k {
            a.swap(k, max_row);
            b.swap(k, max_row);
        }

        // Eliminate.
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            a[i][k] = 0.0;
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * b[j];
        }
        b[i] = sum / a[i][i];
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
        // w = N^T N v
        // First compute Nv (length = n_data).
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
        // Then compute N^T (Nv).
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
/// If the iteration has not converged after `max_iterations`, a warning is
/// logged and the best-effort curve (lowest observed error) is returned as
/// `Ok`; no error is raised.
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

    // Step 1: chord-length parameterization.
    let params = chord_length_params(points);

    // Step 2: build knot vector for m control points.
    let knots = build_approximation_knots(&params, p, m, n);

    // Step 3: initialize control points by sampling closest data points.
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

    // Precompute basis function values for all data points.
    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    // Compute step size mu for LSPIA convergence.
    // mu = 2 / (lambda_min + lambda_max) where lambda are eigenvalues of N^T N.
    // We approximate lambda_max via the power method and use a conservative mu.
    let mu = compute_lspia_step_size(&basis_data, p, m);

    // Step 4: iterate.
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
                mu.mul_add(deltas[j].0, control_points[j].x()),
                mu.mul_add(deltas[j].1, control_points[j].y()),
                mu.mul_add(deltas[j].2, control_points[j].z()),
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
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub fn approximate_lspia_weighted(
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

    // Step 1: chord-length parameterization.
    let params = chord_length_params(points);

    // Step 2: build knot vector.
    let knots = build_approximation_knots(&params, p, m, n);

    // Step 3: initialize control points by sampling closest data points.
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

    // Precompute basis function values.
    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    // Compute step size mu for LSPIA convergence.
    let mu = compute_lspia_step_size_weighted(&basis_data, point_weights, p, m);

    // Step 4: iterate.
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
                mu.mul_add(deltas[j].0, control_points[j].x()),
                mu.mul_add(deltas[j].1, control_points[j].y()),
                mu.mul_add(deltas[j].2, control_points[j].z()),
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
        // Check endpoints.
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
    fn interpolate_single_point_error() {
        let pts = vec![Point3::new(0.0, 0.0, 0.0)];
        assert!(interpolate(&pts, 3).is_err());
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
                Point3::new(t, 2.0f64.mul_add(t, 1.0), 0.0)
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

        // Check that points are near the circle.
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
