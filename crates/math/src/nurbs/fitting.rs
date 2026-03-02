//! NURBS curve fitting: interpolation and approximation from data points.
//!
//! Equivalent to `GeomAPI_Interpolate` and `GeomAPI_PointsToBSpline`
//! in `OpenCascade`.

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
fn chord_length_params(points: &[Point3]) -> Vec<f64> {
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
fn build_approximation_knots(params: &[f64], p: usize, m: usize, n: usize) -> Vec<f64> {
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
fn find_span(t: f64, degree: usize, knots: &[f64], n: usize) -> usize {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

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
}
