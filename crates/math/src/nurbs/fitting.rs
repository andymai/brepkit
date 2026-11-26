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
use crate::nurbs::basis::{basis_funs, basis_funs_into};
use crate::nurbs::curve::NurbsCurve;
use crate::vec::Point3;

/// The most multiply-adds a fit takes, counted from its sizes before it
/// allocates or starts.
///
/// The slowest fit per multiply-add counted, a cubic interpolation through
/// many points, takes about 2.4 ns for each natively and 3.4 ns in wasm, so
/// a fit at this limit runs for about 3 s in wasm. The largest fit the other tests make counts 2.4e6.
pub const MAX_FIT_WORK: f64 = 1e9;

/// Refuses `what`, which would take `work` multiply-adds, past
/// [`MAX_FIT_WORK`].
pub(crate) fn check_fit_work(what: &'static str, work: f64) -> Result<(), MathError> {
    if work <= MAX_FIT_WORK {
        Ok(())
    } else {
        Err(MathError::TooMuchWork {
            what,
            work,
            limit: MAX_FIT_WORK,
        })
    }
}

/// The multiply-adds of an interpolation through `n` points at degree `p`:
/// each point's basis, and its row of the banded solve, at most `p` cells
/// below the diagonal and `2p` above, for the three coordinates. Its
/// parameters rise strictly and its knots average them, so each lies inside
/// the support of its own basis function, which keeps the band to `p` each
/// side before the row swaps.
pub(crate) fn interpolation_work(n: usize, p: usize) -> f64 {
    band_work(n, p, p, p)
}

/// The multiply-adds of a banded interpolation solve of `n` rows at degree
/// `p`, `kl` cells below the diagonal and `ku` above before its row swaps.
fn band_work(n: usize, p: usize, kl: usize, ku: usize) -> f64 {
    let (n, p, kl, ku) = (n as f64, p as f64, kl as f64, ku as f64);
    n * ((p + 1.0).powi(2) + kl * (kl + ku + 3.0) + 3.0 * (kl + ku))
}

/// Refuses parameters that do not rise strictly, or one that is not a
/// number: points repeated in a row have repeated chord-length parameters,
/// which make an interpolation's system singular.
pub(crate) fn check_rising(params: &[f64]) -> Result<(), MathError> {
    if let Some(pair) = params
        .windows(2)
        .find(|pair| pair[0].is_nan() || pair[1].is_nan() || pair[1] <= pair[0])
    {
        return Err(MathError::ParameterOutOfRange {
            value: pair[1],
            min: pair[0],
            max: 1.0,
        });
    }
    Ok(())
}

/// Refuses parameters with one that is not finite, as points with one that
/// is not finite give.
pub(crate) fn check_finite(params: &[f64]) -> Result<(), MathError> {
    if let Some(&bad) = params.iter().find(|t| !t.is_finite()) {
        return Err(MathError::ParameterOutOfRange {
            value: bad,
            min: 0.0,
            max: 1.0,
        });
    }
    Ok(())
}

/// The multiply-adds of a least-squares fit of `n` points by `m` control
/// points at degree `p`: each point's basis and its terms of the normal
/// matrix and the right-hand side, and the banded solve, `p` cells below the
/// diagonal and `p` above before its row swaps.
pub(crate) fn least_squares_work(n: usize, m: usize, p: usize) -> f64 {
    let (n, p1) = (n as f64, (p + 1) as f64);
    n * (2.0 * p1 * p1 + 3.0 * p1) + band_work(m, 0, p, p)
}

/// The multiply-adds of an LSPIA fit of `n` points by `m` control points at
/// degree `p` over `iterations` passes: the start, which searches the
/// sorted parameters twice for each control point's nearest, at most `log2
/// n + 2` steps each, and reads the basis and the step size, then on each
/// pass the curve at every point from its basis, and the corrections it
/// spreads.
pub(crate) fn lspia_work(n: usize, m: usize, p: usize, iterations: usize) -> f64 {
    let (n, m, p1, iterations) = (n as f64, m as f64, (p + 1) as f64, iterations as f64);
    let start = m * 2.0f64.madd(n.log2(), 4.0) + n * p1 * p1 + 42.0 * n * p1;
    let pass = n * (7.0 * p1 + 4.0) + 5.0 * m;
    iterations.madd(pass, start)
}

/// The most cells of eight bytes an interpolation of `n` points at
/// `degree` holds at once; `None` past `usize`.
///
/// They are its parameters and knots, each row's first column and basis
/// values, its band of at most `3 p + 1` cells a row, and its right-hand
/// sides and solved points, three cells each.
#[must_use]
pub fn interpolation_cells(n: usize, degree: usize) -> Option<usize> {
    let p = degree.min(n.saturating_sub(1));
    n.checked_mul(p.checked_mul(4)?.checked_add(11)?)?
        .checked_add(p.checked_add(1)?)
}

/// The most cells of eight bytes a least-squares fit of `n` points by `m`
/// control points at `degree` holds at once ([`approximate`]); `None` past
/// `usize`.
///
/// They are its parameters and knots, the band of its normal matrix,
/// `3 p + 1` cells a control point, its right-hand sides and solved
/// points, three cells each, and one point's basis values; an
/// interpolation's where `m` is not below `n`.
#[must_use]
pub fn least_squares_cells(n: usize, m: usize, degree: usize) -> Option<usize> {
    if m >= n {
        return interpolation_cells(n, degree);
    }
    let p = degree.min(m.saturating_sub(1));
    n.checked_add(m.checked_mul(p.checked_mul(3)?.checked_add(8)?)?)?
        .checked_add(p.checked_mul(2)?.checked_add(2)?)
}

/// The most cells of eight bytes an LSPIA fit of `n` points by `m` control
/// points at `degree` holds at once ([`approximate_lspia`]); `None` past
/// `usize`.
///
/// They are its parameters and knots, its control points and weights, each
/// point's span and basis values with a header of four, and the largest of
/// the step size's power iteration (two vectors a control point and one a
/// point), the curve it checks once (its knots, points and weights again)
/// and each pass's corrections.
#[must_use]
pub fn lspia_cells(n: usize, m: usize, degree: usize) -> Option<usize> {
    let p = degree.min(n.saturating_sub(1));
    let m = m.min(n).max(p + 1);
    let held = n
        .checked_mul(p.checked_add(6)?)?
        .checked_add(m.checked_mul(5)?)?
        .checked_add(p.checked_add(1)?)?;
    let most = m
        .checked_mul(2)?
        .checked_add(n)?
        .max(m.checked_mul(5)?.checked_add(p.checked_add(1)?)?);
    held.checked_add(most)
}

#[cfg(test)]
thread_local! {
    /// The most cells a fit held at once on this thread, read from its
    /// buffers' capacities.
    pub(crate) static HELD_CELLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Notes that a fit holds `cells` at once.
#[cfg(test)]
pub(crate) fn note_held(cells: usize) {
    HELD_CELLS.with(|most| most.set(most.get().max(cells)));
}

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
/// is too high for the number of points, and [`MathError::TooMuchWork`] if
/// the fit would take more than [`MAX_FIT_WORK`] multiply-adds.
pub fn interpolate(points: &[Point3], degree: usize) -> Result<NurbsCurve, MathError> {
    let n = points.len();
    if n < 2 {
        return Err(MathError::EmptyInput);
    }

    let p = degree.min(n - 1);
    check_fit_work("an interpolation", interpolation_work(n, p))?;

    let params = chord_length_params(points);
    check_rising(&params)?;
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
/// linear system is singular, and [`MathError::TooMuchWork`] if the fit
/// would take more than [`MAX_FIT_WORK`] multiply-adds.
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
    check_rising(params)?;
    check_fit_work("an interpolation", interpolation_work(n, degree))?;
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
/// Returns an error if parameters are invalid, and
/// [`MathError::TooMuchWork`] if the fit would take more than
/// [`MAX_FIT_WORK`] multiply-adds.
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
    check_fit_work("a least-squares fit", least_squares_work(n, m, p))?;

    let params = chord_length_params(points);
    check_finite(&params)?;
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
    chord_length_params_by(points.len(), |i| points[i])
}

/// [`chord_length_params`] for the `n` points `point(0)` to `point(n - 1)`,
/// read where they lie, in one buffer of `n` parameters.
pub(crate) fn chord_length_params_by(n: usize, point: impl Fn(usize) -> Point3) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n];
    }

    let mut dists = Vec::with_capacity(n);
    dists.push(0.0);
    let mut last = point(0);
    for i in 1..n {
        let next = point(i);
        let d = (next - last).length();
        dists.push(dists[i - 1] + d);
        last = next;
    }

    let total = dists[n - 1];
    if total < 1e-15 {
        // All points are coincident — uniform params.
        #[allow(clippy::cast_precision_loss)]
        for (i, d) in dists.iter_mut().enumerate() {
            *d = (i as f64) / ((n - 1) as f64);
        }
        return dists;
    }

    for d in &mut dists {
        *d /= total;
    }
    dists
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
    let stride = degree + 1;
    let mut firsts = Vec::with_capacity(n);
    let (mut kl, mut ku) = (0, 0);
    for (i, &t) in params.iter().enumerate() {
        let first = find_span(t, degree, knots, n) - degree;
        kl = kl.max(i.saturating_sub(first));
        ku = ku.max((first + degree).min(n - 1).saturating_sub(i));
        firsts.push(first);
    }
    check_fit_work("an interpolation", band_work(n, degree, kl, ku))?;
    let mut basis = vec![0.0; n * stride];
    for (i, (&t, &first)) in params.iter().zip(&firsts).enumerate() {
        basis_funs_into(
            first + degree,
            t,
            degree,
            knots,
            &mut basis[i * stride..(i + 1) * stride],
        );
    }
    // Row `i` keeps column `j` at `i * width + j + kl - i`.
    let width = 2 * kl + ku + 1;
    let at = |i: usize, j: usize| i * width + j + kl - i;
    let mut band = vec![0.0; n * width];
    for (i, &first) in firsts.iter().enumerate() {
        for (k, &b) in basis[i * stride..(i + 1) * stride].iter().enumerate() {
            let col = first + k;
            if col < n {
                band[at(i, col)] = b;
            }
        }
    }
    let rhs: Vec<[f64; 3]> = points.iter().map(|p| [p.x(), p.y(), p.z()]).collect();
    // The solved points join these at the end of the solve.
    #[cfg(test)]
    note_held(
        params.len()
            + knots.len()
            + firsts.capacity()
            + basis.capacity()
            + band.capacity()
            + 3 * rhs.capacity()
            + 3 * n,
    );
    solve_band(band, (kl, ku), rhs)
}

/// Solve a banded system for three right-hand sides by Gaussian elimination
/// with partial pivoting: `band` keeps row `i`'s column `j` at `i * (2 kl +
/// ku + 1) + j + kl - i`, `kl` cells below the diagonal and `ku` above, with
/// room for the row swaps to widen the upper band to `kl + ku`. Its pivots
/// and arithmetic are those of eliminating the whole matrix.
fn solve_band(
    mut band: Vec<f64>,
    (kl, ku): (usize, usize),
    mut rhs: Vec<[f64; 3]>,
) -> Result<Vec<Point3>, MathError> {
    let n = rhs.len();
    let width = 2 * kl + ku + 1;
    let at = |i: usize, j: usize| i * width + j + kl - i;
    for k in 0..n {
        let below = (k + kl).min(n - 1);
        let mut pivot_row = k;
        for i in (k + 1)..=below {
            if band[at(i, k)].abs() > band[at(pivot_row, k)].abs() {
                pivot_row = i;
            }
        }
        let pivot = band[at(pivot_row, k)].abs();
        if pivot.is_nan() || pivot < 1e-15 {
            return Err(MathError::SingularMatrix);
        }
        let right = (k + kl + ku).min(n - 1);
        if pivot_row != k {
            for j in k..=right {
                band.swap(at(k, j), at(pivot_row, j));
            }
            rhs.swap(k, pivot_row);
        }
        let pivot = band[at(k, k)];
        for i in (k + 1)..=below {
            let factor = band[at(i, k)] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in (k + 1)..=right {
                band[at(i, j)] -= factor * band[at(k, j)];
            }
            band[at(i, k)] = 0.0;
            for c in 0..3 {
                rhs[i][c] -= factor * rhs[k][c];
            }
        }
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..=(i + kl + ku).min(n - 1) {
            for c in 0..3 {
                sum[c] -= band[at(i, j)] * rhs[j][c];
            }
        }
        for c in 0..3 {
            rhs[i][c] = sum[c] / band[at(i, i)];
        }
    }
    Ok(rhs.iter().map(|&[x, y, z]| Point3::new(x, y, z)).collect())
}

/// Solve least-squares approximation for control points.
///
/// Each point has `degree + 1` basis values that are not zero, so its terms
/// of the normal matrix `NᵀN` lie in a band `degree` wide each side of the
/// diagonal, and are summed point by point in the order of the points: the
/// same sums as over the whole matrix, without its zero terms.
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
    let (kl, ku) = (degree, degree);
    let width = 2 * kl + ku + 1;
    let at = |i: usize, j: usize| i * width + j + kl - i;
    let mut band = vec![0.0; m * width];
    let mut rhs = vec![[0.0; 3]; m];
    let mut basis = vec![0.0; degree + 1];
    for (k, &t) in params.iter().enumerate() {
        let span = find_span(t, degree, knots, m);
        basis_funs_into(span, t, degree, knots, &mut basis);
        let first = span - degree;
        for (a, &ba) in basis.iter().enumerate() {
            let i = first + a;
            if i >= m {
                continue;
            }
            for (b, &bb) in basis.iter().enumerate() {
                let j = first + b;
                if j < m {
                    band[at(i, j)] += ba * bb;
                }
            }
            rhs[i][0] += ba * points[k].x();
            rhs[i][1] += ba * points[k].y();
            rhs[i][2] += ba * points[k].z();
        }
    }

    // Fix first and last control points: zero the first/last rows and cols,
    // set diagonal to 1 so the endpoints are interpolated exactly.
    for j in 0..=degree.min(m - 1) {
        band[at(0, j)] = 0.0;
        band[at(j, 0)] = 0.0;
        band[at(m - 1, m - 1 - j)] = 0.0;
        band[at(m - 1 - j, m - 1)] = 0.0;
    }
    band[at(0, 0)] = 1.0;
    band[at(m - 1, m - 1)] = 1.0;
    rhs[0] = [points[0].x(), points[0].y(), points[0].z()];
    rhs[m - 1] = [points[n - 1].x(), points[n - 1].y(), points[n - 1].z()];
    // The solved points join these at the end of the solve.
    #[cfg(test)]
    note_held(
        params.len()
            + knots.len()
            + band.capacity()
            + 3 * rhs.capacity()
            + basis.capacity()
            + 3 * m,
    );
    solve_band(band, (kl, ku), rhs)
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

/// The index of the parameter nearest `t` in the sorted `params`, the first
/// of those as near: the one a scan of every parameter in turn finds.
///
/// Each distance `|param - t|` falls as the parameters rise to `t` and grows
/// past it, with rounding at most leaving ties, so the nearest is one of the
/// two either side of `t`, and those as near lie just below it: a second
/// search of the parameters below finds the first of them, however many
/// repeat.
fn nearest_param(params: &[f64], t: f64) -> usize {
    let above = params.partition_point(|&param| {
        nearest_step();
        param < t
    });
    let distance = |j: usize| (params[j] - t).abs();
    let best = match (
        above.checked_sub(1),
        (above < params.len()).then_some(above),
    ) {
        (Some(below), Some(above)) if distance(above) < distance(below) => return above,
        (Some(below), _) => below,
        (None, Some(above)) => above,
        (None, None) => return 0,
    };
    let near = distance(best);
    params[..best].partition_point(|&param| {
        nearest_step();
        (param - t).abs() > near
    })
}

#[cfg(test)]
thread_local! {
    /// The parameters [`nearest_param`] compared, on this thread.
    static NEAREST_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Counts one comparison of [`nearest_param`]'s searches.
#[inline]
fn nearest_step() {
    #[cfg(test)]
    NEAREST_STEPS.with(|steps| steps.set(steps.get() + 1));
}

/// The curve of equal weights on `control_points` at the parameter whose
/// span and basis are `basis`, summed as [`NurbsCurve::evaluate`] sums it.
fn basis_point((span, values): &(usize, Vec<f64>), p: usize, control_points: &[Point3]) -> Point3 {
    let (mut wx, mut wy, mut wz, mut ww) = (0.0, 0.0, 0.0, 0.0);
    for (j, &value) in values.iter().enumerate().take(p + 1) {
        let point = control_points[span - p + j];
        wx += value * point.x();
        wy += value * point.y();
        wz += value * point.z();
        ww += value;
    }
    if ww == 0.0 {
        Point3::new(wx, wy, wz)
    } else {
        Point3::new(wx / ww, wy / ww, wz / ww)
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
/// Returns [`MathError::EmptyInput`] if fewer than 2 points, and
/// [`MathError::TooMuchWork`] if `max_iterations` passes would take more
/// than [`MAX_FIT_WORK`] multiply-adds.
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
    check_fit_work("an LSPIA fit", lspia_work(n, m, p, max_iterations))?;

    let params = chord_length_params(points);
    check_finite(&params)?;
    let knots = build_approximation_knots(&params, p, m, n);

    // Initialize control points by sampling closest data points.
    let mut control_points: Vec<Point3> = (0..m)
        .map(|i| {
            let t = if m > 1 {
                i as f64 / (m - 1) as f64
            } else {
                0.0
            };
            points[nearest_param(&params, t)]
        })
        .collect();

    let weights = vec![1.0; m];

    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    // mu = 2 / (lambda_min + lambda_max) where lambda are eigenvalues of N^T N.
    // We approximate lambda_max via the power method and use a conservative mu.
    #[cfg(test)]
    let held = params.capacity()
        + knots.capacity()
        + 3 * control_points.capacity()
        + weights.capacity()
        + 4 * basis_data.capacity()
        + basis_data
            .iter()
            .map(|(_, values)| values.capacity())
            .sum::<usize>();
    // The step's power iteration holds two vectors a control point and one
    // a point, and the check copies the curve's knots, points and weights.
    #[cfg(test)]
    note_held(held + (2 * m + n).max(knots.len() + 4 * m));
    let mu = compute_lspia_step_size(&basis_data, p, m);
    // The curve is checked once: each pass reads it from the basis.
    NurbsCurve::new(p, knots.clone(), control_points.clone(), weights.clone())?;

    for iter in 0..max_iterations {
        let mut max_err = 0.0f64;
        let mut deltas = vec![(0.0f64, 0.0f64, 0.0f64); m];
        #[cfg(test)]
        note_held(held + 3 * deltas.capacity());

        for i in 0..n {
            let q = basis_point(&basis_data[i], p, &control_points);
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
/// `points.len()`, and [`MathError::TooMuchWork`] if `max_iterations` passes
/// would take more than [`MAX_FIT_WORK`] multiply-adds.
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
    check_fit_work("an LSPIA fit", lspia_work(n, m, p, max_iterations))?;

    let params = chord_length_params(points);
    check_finite(&params)?;
    let knots = build_approximation_knots(&params, p, m, n);

    // Initialize control points by sampling closest data points.
    let mut control_points: Vec<Point3> = (0..m)
        .map(|i| {
            let t = if m > 1 {
                i as f64 / (m - 1) as f64
            } else {
                0.0
            };
            points[nearest_param(&params, t)]
        })
        .collect();

    let weights = vec![1.0; m];

    let mut basis_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
    for &u in &params {
        let span = find_span(u, p, &knots, m);
        let n_vals = basis_funs(span, u, p, &knots);
        basis_data.push((span, n_vals));
    }

    let mu = compute_lspia_step_size_weighted(&basis_data, point_weights, p, m);
    NurbsCurve::new(p, knots.clone(), control_points.clone(), weights.clone())?;

    for iter in 0..max_iterations {
        let mut max_err = 0.0f64;
        let mut deltas = vec![(0.0f64, 0.0f64, 0.0f64); m];

        for i in 0..n {
            let q = basis_point(&basis_data[i], p, &control_points);
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

    /// The wasm bindings' memory limits on a fit: 2^22 cells of working
    /// arrays, degree 25, 2^16 control points and 10,000 iterations.
    const CELLS: usize = 1 << 22;
    const DEGREE: usize = 25;
    const CONTROLS: usize = 1 << 16;
    const ITERATIONS: usize = 10_000;

    fn wave(n: usize) -> Vec<Point3> {
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                Point3::new(t * 10.0, (t * 7.0).sin(), (t * 3.0).cos())
            })
            .collect()
    }

    fn too_much(found: &Result<NurbsCurve, MathError>) -> bool {
        matches!(found, Err(MathError::TooMuchWork { .. }))
    }

    /// The largest fit of each kind within the memory limits: an
    /// interpolation keeps `4p + 2` cells a point, so 41,120 points at
    /// degree 25, and a least-squares fit `3p + 4` cells for each point and
    /// control point, so 53,092 of them, and the work of both is bounded.
    /// The fit of 2,362 points by 1,183 control points at degree 25, which
    /// summed its normal matrix whole for 3.3e9 multiply-adds, runs at
    /// once. LSPIA keeps `p + 1` cells a point, so 161,319 points at degree
    /// 25 by 2^16 control points over 10,000 iterations, and is refused
    /// before it starts.
    #[test]
    fn the_largest_curve_fits_within_the_memory_limits_are_refused_or_bounded() {
        let n = CELLS / (4 * DEGREE + 2);
        assert!(interpolation_work(n, DEGREE) <= MAX_FIT_WORK);
        let most = CELLS / (3 * DEGREE + 4);
        assert!(least_squares_work(most, most, DEGREE) <= MAX_FIT_WORK);
        let started = std::time::Instant::now();
        let found = approximate(&wave(2362), DEGREE, 1183);
        assert!(!too_much(&found), "{found:?}");
        let n = CELLS / (DEGREE + 1);
        let found = approximate_lspia(&wave(n), DEGREE, CONTROLS, 1e-6, ITERATIONS);
        assert!(too_much(&found), "{found:?}");
        assert!(started.elapsed().as_secs() < 1, "{:?}", started.elapsed());
    }

    /// An LSPIA fit of 1,000 points by 100 control points at degree 3 takes
    /// every pass it may within the limit, and is refused one pass past it.
    /// It settles on its first pass, so the fit within the limit is quick.
    #[test]
    fn a_fit_just_within_the_work_limit_runs() {
        let (n, m, p) = (1000, 100, 3);
        let within = |passes: usize| lspia_work(n, m, p, passes) <= MAX_FIT_WORK;
        let mut passes = 1;
        while within(passes * 2) {
            passes *= 2;
        }
        let mut step = passes / 2;
        while step > 0 {
            if within(passes + step) {
                passes += step;
            }
            step /= 2;
        }
        let points = wave(n);
        assert!(approximate_lspia(&points, p, m, f64::INFINITY, passes).is_ok());
        assert!(too_much(&approximate_lspia(
            &points,
            p,
            m,
            f64::INFINITY,
            passes + 1
        )));
    }

    /// A thousand copies of the origin before the points (1, 0, 0) to
    /// (1000, 0, 0): their chords are zero, so their parameters repeat and
    /// the interpolation's system is singular. They are refused before the
    /// solve, as are points with one that is not a number, by every fit.
    #[test]
    fn repeated_or_unreadable_points_are_refused_before_the_solve() {
        let refused = |found: &Result<NurbsCurve, MathError>| {
            matches!(found, Err(MathError::ParameterOutOfRange { .. }))
        };
        let started = std::time::Instant::now();
        let mut repeated = vec![Point3::new(0.0, 0.0, 0.0); 1000];
        repeated.extend((1..=1000).map(|i| Point3::new(f64::from(i), 0.0, 0.0)));
        let found = interpolate(&repeated, 3);
        assert!(refused(&found), "{found:?}");
        let mut unread = wave(100);
        unread[40] = Point3::new(f64::NAN, 0.0, 0.0);
        for found in [
            interpolate(&unread, 3),
            approximate(&unread, 3, 20),
            approximate_lspia(&unread, 3, 20, 1e-6, 100),
        ] {
            assert!(refused(&found), "{found:?}");
        }
        assert!(started.elapsed().as_secs() < 1, "{:?}", started.elapsed());
    }

    /// Points along a wavering curve, from a fixed seed, some of them
    /// repeated.
    fn scattered(n: usize, seed: u64) -> Vec<Point3> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut x = 0.0;
        let mut last = Point3::new(0.0, 0.0, 0.0);
        (0..n)
            .map(|i| {
                if i % 17 != 5 {
                    x += 0.1 + next();
                    last = Point3::new(x, next() * 3.0 - 1.5, (x * 0.7).sin() + next() * 0.2);
                }
                last
            })
            .collect()
    }

    /// The least-squares fit summed over the whole of its normal matrix and
    /// solved by dense elimination with partial pivoting, the terms in the
    /// same order as the band takes them.
    fn dense_least_squares(
        points: &[Point3],
        degree: usize,
        m: usize,
    ) -> Result<Vec<Point3>, MathError> {
        let n = points.len();
        let p = degree.min(m - 1);
        let params = chord_length_params(points);
        let knots = build_approximation_knots(&params, p, m, n);
        let mut basis = vec![vec![0.0; m]; n];
        for (i, &t) in params.iter().enumerate() {
            let span = find_span(t, p, &knots, m);
            for (k, &b) in basis_funs(span, t, p, &knots).iter().enumerate() {
                if span - p + k < m {
                    basis[i][span - p + k] = b;
                }
            }
        }
        let mut a = vec![vec![0.0; m]; m];
        let mut rhs = vec![[0.0; 3]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    a[i][j] += basis[k][i] * basis[k][j];
                }
            }
            for k in 0..n {
                rhs[i][0] += basis[k][i] * points[k].x();
                rhs[i][1] += basis[k][i] * points[k].y();
                rhs[i][2] += basis[k][i] * points[k].z();
            }
        }
        for j in 0..m {
            (a[0][j], a[m - 1][j], a[j][0], a[j][m - 1]) = (0.0, 0.0, 0.0, 0.0);
        }
        (a[0][0], a[m - 1][m - 1]) = (1.0, 1.0);
        rhs[0] = [points[0].x(), points[0].y(), points[0].z()];
        rhs[m - 1] = [points[n - 1].x(), points[n - 1].y(), points[n - 1].z()];
        for k in 0..m {
            let mut row = k;
            for i in (k + 1)..m {
                if a[i][k].abs() > a[row][k].abs() {
                    row = i;
                }
            }
            let pivot = a[row][k].abs();
            if pivot.is_nan() || pivot < 1e-15 {
                return Err(MathError::SingularMatrix);
            }
            a.swap(k, row);
            rhs.swap(k, row);
            for i in (k + 1)..m {
                let factor = a[i][k] / a[k][k];
                if factor == 0.0 {
                    continue;
                }
                for j in (k + 1)..m {
                    a[i][j] -= factor * a[k][j];
                }
                a[i][k] = 0.0;
                for c in 0..3 {
                    rhs[i][c] -= factor * rhs[k][c];
                }
            }
        }
        for i in (0..m).rev() {
            let mut sum = rhs[i];
            for j in (i + 1)..m {
                for c in 0..3 {
                    sum[c] -= a[i][j] * rhs[j][c];
                }
            }
            for c in 0..3 {
                rhs[i][c] = sum[c] / a[i][i];
            }
        }
        Ok(rhs.iter().map(|&[x, y, z]| Point3::new(x, y, z)).collect())
    }

    /// LSPIA as it ran with a scan of every parameter for each control
    /// point and the curve rebuilt on each pass.
    fn scanned_lspia(
        points: &[Point3],
        degree: usize,
        num_control_points: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Vec<Point3> {
        let n = points.len();
        let p = degree.min(n - 1);
        let m = num_control_points.min(n).max(p + 1);
        let params = chord_length_params(points);
        let knots = build_approximation_knots(&params, p, m, n);
        let mut control_points: Vec<Point3> = (0..m)
            .map(|i| {
                let t = if m > 1 {
                    i as f64 / (m - 1) as f64
                } else {
                    0.0
                };
                let mut best = (0, f64::INFINITY);
                for (j, &param) in params.iter().enumerate() {
                    if (param - t).abs() < best.1 {
                        best = (j, (param - t).abs());
                    }
                }
                points[best.0]
            })
            .collect();
        let weights = vec![1.0; m];
        let basis: Vec<(usize, Vec<f64>)> = params
            .iter()
            .map(|&u| {
                let span = find_span(u, p, &knots, m);
                (span, basis_funs(span, u, p, &knots))
            })
            .collect();
        let mu = compute_lspia_step_size(&basis, p, m);
        for _ in 0..max_iterations {
            let curve =
                NurbsCurve::new(p, knots.clone(), control_points.clone(), weights.clone()).unwrap();
            let mut max_err = 0.0_f64;
            let mut deltas = vec![(0.0, 0.0, 0.0); m];
            for (i, &u) in params.iter().enumerate() {
                let q = curve.evaluate(u);
                let e = (
                    points[i].x() - q.x(),
                    points[i].y() - q.y(),
                    points[i].z() - q.z(),
                );
                max_err = max_err.max((e.0 * e.0 + e.1 * e.1 + e.2 * e.2).sqrt());
                let (span, values) = &basis[i];
                for (k, &v) in values.iter().enumerate() {
                    let j = span - p + k;
                    if j < m {
                        deltas[j].0 += v * e.0;
                        deltas[j].1 += v * e.1;
                        deltas[j].2 += v * e.2;
                    }
                }
            }
            if max_err < tolerance {
                break;
            }
            for j in 0..m {
                control_points[j] = Point3::new(
                    mu.madd(deltas[j].0, control_points[j].x()),
                    mu.madd(deltas[j].1, control_points[j].y()),
                    mu.madd(deltas[j].2, control_points[j].z()),
                );
            }
        }
        control_points
    }

    fn same_points(a: &[Point3], b: &[Point3]) -> bool {
        let bits = |p: &Point3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| bits(x) == bits(y))
    }

    /// The banded least-squares fit takes the same terms, pivots and
    /// arithmetic as the dense sum and elimination, so its control points
    /// are the same bit for bit, on the fits of the tests above and on
    /// scattered points at degrees 1 to 5 and 25.
    #[test]
    fn the_banded_least_squares_fit_matches_the_dense_one() {
        let mut cases: Vec<(Vec<Point3>, usize, usize)> = vec![
            (
                (0..20)
                    .map(|i| Point3::new(f64::from(i), (f64::from(i) * 0.3).sin(), 0.0))
                    .collect(),
                3,
                4,
            ),
            (
                (0..20)
                    .map(|i| Point3::new(f64::from(i), (f64::from(i) * 0.3).sin(), 0.0))
                    .collect(),
                2,
                3,
            ),
        ];
        for (seed, n) in [(1_u64, 40), (7, 120), (99, 300)] {
            for (degree, m) in [(1, 9), (2, 17), (3, 25), (4, 33), (5, 40), (25, 39)] {
                cases.push((scattered(n, seed), degree, m.min(n - 1)));
            }
        }
        let mut solved = 0;
        for (points, degree, m) in &cases {
            let banded = approximate(points, *degree, *m).map(|c| c.control_points().to_vec());
            let dense = dense_least_squares(points, *degree, *m);
            let agree = match (&banded, &dense) {
                (Ok(a), Ok(b)) => {
                    solved += 1;
                    same_points(a, b)
                }
                (Err(MathError::SingularMatrix), Err(MathError::SingularMatrix)) => true,
                _ => false,
            };
            assert!(
                agree,
                "{} points at {degree} by {m}: {banded:?} {dense:?}",
                points.len()
            );
        }
        assert!(solved >= 15, "{solved} of {} solved", cases.len());
    }

    /// LSPIA's start picks each control point's nearest parameter by a
    /// search of the sorted parameters, the first of those as near, as a
    /// scan of every parameter did, and each pass reads the curve from the
    /// basis as its evaluation sums it: the control points are the same bit
    /// for bit, repeated points and all.
    #[test]
    fn lspia_matches_its_scan_and_rebuilt_curve() {
        for (seed, n) in [(3_u64, 60), (11, 200), (5, 500)] {
            let points = scattered(n, seed);
            for (degree, m, passes) in [(2, 12, 40), (3, 30, 25), (3, n / 3, 10), (5, n / 2, 5)] {
                let fitted = approximate_lspia(&points, degree, m, 1e-9, passes).unwrap();
                let scanned = scanned_lspia(&points, degree, m, 1e-9, passes);
                assert!(
                    same_points(fitted.control_points(), &scanned),
                    "{n} at {degree} by {m}"
                );
            }
        }
    }

    /// The first parameter a scan of every parameter in turn finds nearest
    /// `t`.
    fn scanned_nearest(params: &[f64], t: f64) -> usize {
        let mut best = (0, f64::INFINITY);
        for (j, &param) in params.iter().enumerate() {
            if (param - t).abs() < best.1 {
                best = (j, (param - t).abs());
            }
        }
        best.0
    }

    /// A long prefix of parameters closer together than a parameter near
    /// the middle rounds to, or all alike, before one at 1: each control
    /// point below the middle is as near the whole prefix, and the scan
    /// takes its first. A walk back from the prefix's last took 3.5e5 steps
    /// for 500 control points by 5,001 clustered parameters, past a count of
    /// 1.4e4, and 2.0e7 for 2,000 by 20,001 alike ones: about `m n / 2`, or
    /// 5.3e9 for 2^16 by 161,319. The two searches take at most `2 log2 n +
    /// 4` steps for each control point, as the work counts them, and find
    /// the parameter the scan finds.
    #[test]
    fn a_clustered_prefix_finds_each_nearest_within_its_count() {
        let count = |n: usize, m: usize| m as f64 * 2.0f64.madd((n as f64).log2(), 4.0);
        let spread = |m: usize| (0..m).map(move |i| i as f64 / (m - 1) as f64);
        let steps = || NEAREST_STEPS.with(std::cell::Cell::get) as f64;
        let n = 5_001;
        let mut params: Vec<f64> = (0..n - 1).map(|k| k as f64 * 1e-20).collect();
        params.push(1.0);
        NEAREST_STEPS.with(|steps| steps.set(0));
        for t in spread(500) {
            assert_eq!(
                nearest_param(&params, t),
                scanned_nearest(&params, t),
                "{t}"
            );
        }
        assert!(
            steps() <= count(n, 500),
            "{} past {}",
            steps(),
            count(n, 500)
        );
        let n = CELLS / (DEGREE + 1);
        let mut params = vec![0.0; n - 1];
        params.push(1.0);
        NEAREST_STEPS.with(|steps| steps.set(0));
        for t in spread(CONTROLS) {
            let scanned = if (1.0 - t).abs() < t.abs() { n - 1 } else { 0 };
            assert_eq!(nearest_param(&params, t), scanned, "{t}");
        }
        assert!(
            steps() <= count(n, CONTROLS),
            "{} past {}",
            steps(),
            count(n, CONTROLS)
        );
    }

    /// Each curve fit of 2,000 points, by 300 control points where it
    /// approximates, at degrees 0, 1, 3 and 25: the most it holds at once,
    /// read from its buffers' capacities, is its count. The interpolation at
    /// degree 0, whose knots average no parameters, is refused for its work
    /// before its solve holds its band.
    #[test]
    fn each_curve_fit_holds_at_most_its_count() {
        let points = wave(2000);
        let held = |fit: &dyn Fn() -> Result<NurbsCurve, MathError>| {
            HELD_CELLS.with(|most| most.set(0));
            drop(fit());
            HELD_CELLS.with(std::cell::Cell::get)
        };
        for degree in [0, 1, 3, 25] {
            let cases = [
                (
                    held(&|| interpolate(&points, degree)),
                    interpolation_cells(2000, degree).unwrap(),
                ),
                (
                    held(&|| approximate(&points, degree, 300)),
                    least_squares_cells(2000, 300, degree).unwrap(),
                ),
                (
                    held(&|| approximate_lspia(&points, degree, 300, 1e-9, 2)),
                    lspia_cells(2000, 300, degree).unwrap(),
                ),
            ];
            for (k, &(held, count)) in cases.iter().enumerate() {
                let refused = degree == 0 && k == 0 && held == 0;
                assert!(
                    held == count || refused,
                    "degree {degree}, fit {k}: {held} against {count}"
                );
            }
        }
    }
}
