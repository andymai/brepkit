//! Standalone NURBS basis function evaluation.
//!
//! Free functions extracted from curve/surface types so both can share them.
//! Algorithm numbers refer to Piegl & Tiller, *The NURBS Book*.

/// Maximum degree that uses stack-allocated temporaries in basis functions.
///
/// CAD practice uses at most degree 7 (cubic and quartic are by far the most
/// common). This gives a generous buffer while keeping the stack arrays small.
const MAX_STACK_DEGREE: usize = 10;

/// Find the knot span index for parameter `u` (A2.1).
///
/// Returns the index `i` such that `knots[i] <= u < knots[i+1]`,
/// clamped to the valid range `[degree, n-1]` where `n` is the number
/// of control points.
#[must_use]
pub fn find_span(n: usize, degree: usize, u: f64, knots: &[f64]) -> usize {
    // Clamp to the upper end of the parameter domain.
    if u >= knots[n] {
        return n - 1;
    }
    // Clamp to the lower end.
    if u <= knots[degree] {
        return degree;
    }

    // Binary search for the span.
    let mut low = degree;
    let mut high = n;
    let mut mid = usize::midpoint(low, high);
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = usize::midpoint(low, high);
    }
    mid
}

/// O(1) span lookup for uniform knot vectors.
///
/// `step` is the constant spacing between internal knots (as returned by
/// [`uniform_knot_step`]). The caller must verify the knot vector is uniform
/// before using this — passing an incorrect `step` gives wrong results.
#[must_use]
pub fn find_span_uniform(n: usize, degree: usize, u: f64, knots: &[f64], step: f64) -> usize {
    if u >= knots[n] {
        return n - 1;
    }
    if u <= knots[degree] {
        return degree;
    }
    let span = degree + ((u - knots[degree]) / step) as usize;
    span.min(n - 1)
}

/// Check if internal knots are uniformly spaced.
///
/// Returns the step size if the internal knots `knots[degree..=n]` are
/// equidistant (within 1e-12), or `None` otherwise.
#[must_use]
pub fn uniform_knot_step(knots: &[f64], degree: usize) -> Option<f64> {
    let n = knots.len() - degree - 1; // number of control points
    if n <= degree + 1 {
        return None; // Bezier or too few knots to be meaningfully uniform
    }
    let first_internal = degree;
    let last_internal = n; // knots[degree..=n] are internal
    if last_internal <= first_internal + 1 {
        return None;
    }
    let step = knots[first_internal + 1] - knots[first_internal];
    if step <= 0.0 {
        return None;
    }
    for i in (first_internal + 1)..last_internal {
        let actual_step = knots[i + 1] - knots[i];
        if (actual_step - step).abs() > 1e-12 {
            return None;
        }
    }
    Some(step)
}

/// Maximum output degree for stack-allocated caller buffers.
///
/// Callers can use `[f64; MAX_STACK_OUTPUT + 1]` for degrees up to this value.
/// Covers all practical CAD usage (cubic through septic).
pub const MAX_STACK_OUTPUT: usize = 8;

/// Write non-zero basis function values into `out` (A2.2, zero-allocation variant).
///
/// `out` must have length >= `degree + 1`. Writes `N_{span-degree,degree}(u)`
/// through `N_{span,degree}(u)` into `out[0..=degree]`.
pub fn basis_funs_into(span: usize, u: f64, degree: usize, knots: &[f64], out: &mut [f64]) {
    // Stack-allocate left/right temporaries for typical degrees.
    let mut left_buf = [0.0_f64; MAX_STACK_DEGREE + 1];
    let mut right_buf = [0.0_f64; MAX_STACK_DEGREE + 1];
    let mut left_vec;
    let mut right_vec;
    let (left, right): (&mut [f64], &mut [f64]) = if degree <= MAX_STACK_DEGREE {
        (&mut left_buf[..=degree], &mut right_buf[..=degree])
    } else {
        left_vec = vec![0.0; degree + 1];
        right_vec = vec![0.0; degree + 1];
        (&mut left_vec, &mut right_vec)
    };

    out[0] = 1.0;

    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = right[r + 1].mul_add(temp, saved);
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/// Compute the non-zero basis functions at parameter `u` (A2.2).
///
/// Returns a vector of length `degree + 1` containing `N_{span-degree,degree}(u)`
/// through `N_{span,degree}(u)`.
///
/// The `left` and `right` temporaries are stack-allocated for degrees up to
/// [`MAX_STACK_DEGREE`] (covers all practical CAD usage), falling back to heap
/// allocation for higher degrees.
#[must_use]
pub fn basis_funs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0; degree + 1];
    basis_funs_into(span, u, degree, knots, &mut n);
    n
}

/// Compute basis function derivatives at parameter `u` (A2.3).
///
/// Returns a 2D vector `ders[k][j]` where `ders[k][j]` is the `k`-th derivative
/// of the basis function `N_{span-degree+j, degree}` evaluated at `u`.
/// `k` ranges from `0` to `n_derivs`, `j` from `0` to `degree`.
///
/// The `left` and `right` temporaries are stack-allocated for degrees up to
/// [`MAX_STACK_DEGREE`].
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
#[must_use]
pub fn ders_basis_funs(
    span: usize,
    u: f64,
    degree: usize,
    n_derivs: usize,
    knots: &[f64],
) -> Vec<Vec<f64>> {
    let p = degree;
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    // Stack-allocate left/right temporaries for typical degrees.
    let mut left_buf = [0.0_f64; MAX_STACK_DEGREE + 1];
    let mut right_buf = [0.0_f64; MAX_STACK_DEGREE + 1];
    let mut left_vec;
    let mut right_vec;
    let (left, right): (&mut [f64], &mut [f64]) = if p <= MAX_STACK_DEGREE {
        (&mut left_buf[..=p], &mut right_buf[..=p])
    } else {
        left_vec = vec![0.0; p + 1];
        right_vec = vec![0.0; p + 1];
        (&mut left_vec, &mut right_vec)
    };

    ndu[0][0] = 1.0;

    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            // Lower triangle
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            // Upper triangle
            ndu[r][j] = right[r + 1].mul_add(temp, saved);
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    // Load the basis functions
    let mut ders = vec![vec![0.0; p + 1]; n_derivs + 1];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }

    // Compute derivatives (Eq. [2.9])
    let mut a = vec![vec![0.0; p + 1]; 2];
    for r in 0..=p {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;

        // Compute k-th derivative
        for k in 1..=n_derivs {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = (p as isize) - (k as isize);

            if rk >= 0 {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }

            let j1 = if rk >= -1 { 1usize } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk {
                k - 1
            } else {
                (p as isize - rk) as usize
            };

            for j in j1..=j2 {
                a[s2][j] =
                    (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                d += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }

            if (r as isize) <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                d += a[s2][k] * ndu[r][pk as usize];
            }

            ders[k][r] = d;

            // Switch rows
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Multiply through by the correct factors (Eq. [2.9])
    // Degree values are always small, so usize→f64 is lossless in practice.
    #[allow(clippy::cast_precision_loss)]
    let mut r = p as f64;
    for (idx, row) in ders.iter_mut().enumerate().skip(1).take(n_derivs) {
        for val in &mut row[..=p] {
            *val *= r;
        }
        #[allow(clippy::cast_precision_loss)]
        let factor = (p as isize - idx as isize) as f64;
        r *= factor;
    }

    ders
}

/// Binomial coefficient C(n, k) via iterative multiplication.
pub(crate) fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_lossless, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    // Uniform cubic knot vector: [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    // 6 control points, degree 3
    fn cubic_knots() -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    }

    #[test]
    fn find_span_interior() {
        let knots = cubic_knots();
        assert_eq!(find_span(6, 3, 0.5, &knots), 3);
        assert_eq!(find_span(6, 3, 1.5, &knots), 4);
        assert_eq!(find_span(6, 3, 2.5, &knots), 5);
    }

    #[test]
    fn find_span_endpoints() {
        let knots = cubic_knots();
        // At the start
        assert_eq!(find_span(6, 3, 0.0, &knots), 3);
        // At the end
        assert_eq!(find_span(6, 3, 3.0, &knots), 5);
    }

    #[test]
    fn find_span_at_knot() {
        let knots = cubic_knots();
        assert_eq!(find_span(6, 3, 1.0, &knots), 4);
        assert_eq!(find_span(6, 3, 2.0, &knots), 5);
    }

    #[test]
    fn basis_funs_partition_of_unity() {
        let knots = cubic_knots();
        let span = find_span(6, 3, 1.5, &knots);
        let n = basis_funs(span, 1.5, 3, &knots);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "partition of unity: sum = {sum}");
    }

    #[test]
    fn basis_funs_non_negative() {
        let knots = cubic_knots();
        for u_int in 0..=30 {
            let u = u_int as f64 / 10.0;
            let span = find_span(6, 3, u, &knots);
            let n = basis_funs(span, u, 3, &knots);
            for &val in &n {
                assert!(val >= -1e-15, "non-negative: got {val} at u={u}");
            }
        }
    }

    #[test]
    fn ders_basis_funs_zeroth_matches_basis_funs() {
        let knots = cubic_knots();
        let u = 1.5;
        let span = find_span(6, 3, u, &knots);
        let n = basis_funs(span, u, 3, &knots);
        let ders = ders_basis_funs(span, u, 3, 2, &knots);

        for j in 0..=3 {
            assert!(
                (ders[0][j] - n[j]).abs() < 1e-14,
                "zeroth deriv mismatch at j={j}"
            );
        }
    }

    #[test]
    fn first_derivatives_sum_to_zero() {
        // Sum of first derivatives of basis functions = d/du(1) = 0
        let knots = cubic_knots();
        let u = 1.5;
        let span = find_span(6, 3, u, &knots);
        let ders = ders_basis_funs(span, u, 3, 1, &knots);
        let sum: f64 = ders[1].iter().sum();
        assert!(
            sum.abs() < 1e-12,
            "sum of first derivatives should be 0, got {sum}"
        );
    }

    #[test]
    fn basis_funs_into_matches_basis_funs() {
        let knots = cubic_knots();
        let span = find_span(6, 3, 1.5, &knots);
        let expected = basis_funs(span, 1.5, 3, &knots);
        let mut out = [0.0f64; 4];
        basis_funs_into(span, 1.5, 3, &knots, &mut out);
        for (a, b) in expected.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15, "mismatch: {a} vs {b}");
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_basis_funs_into_matches(u in 0.0f64..=3.0) {
            let knots = cubic_knots();
            let span = find_span(6, 3, u, &knots);
            let expected = basis_funs(span, u, 3, &knots);
            let mut out = [0.0f64; 4];
            basis_funs_into(span, u, 3, &knots, &mut out);
            for (a, b) in expected.iter().zip(out.iter()) {
                prop_assert!((a - b).abs() < 1e-14);
            }
        }

        #[test]
        fn prop_partition_of_unity(u in 0.0f64..=3.0) {
            let knots = cubic_knots();
            let span = find_span(6, 3, u, &knots);
            let n = basis_funs(span, u, 3, &knots);
            let sum: f64 = n.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-12, "sum = {}", sum);
        }

        #[test]
        fn prop_non_negative(u in 0.0f64..=3.0) {
            let knots = cubic_knots();
            let span = find_span(6, 3, u, &knots);
            let n = basis_funs(span, u, 3, &knots);
            for &val in &n {
                prop_assert!(val >= -1e-15, "negative basis: {}", val);
            }
        }

        #[test]
        fn prop_first_deriv_sum_zero(u in 0.0f64..=3.0) {
            let knots = cubic_knots();
            let span = find_span(6, 3, u, &knots);
            let ders = ders_basis_funs(span, u, 3, 1, &knots);
            let sum: f64 = ders[1].iter().sum();
            prop_assert!(sum.abs() < 1e-10, "first deriv sum = {}", sum);
        }

        #[test]
        fn prop_find_span_uniform_matches_binary(u in 0.0f64..=3.0) {
            let knots = cubic_knots();
            let step = uniform_knot_step(&knots, 3).expect("cubic_knots is uniform");
            let expected = find_span(6, 3, u, &knots);
            let got = find_span_uniform(6, 3, u, &knots, step);
            prop_assert_eq!(got, expected, "u={}", u);
        }
    }

    // ── Uniform knot helpers ──────────────────────────────────────────

    #[test]
    fn uniform_knot_step_detects_uniform() {
        let knots = cubic_knots(); // [0,0,0,0, 1,2,3, 3,3,3,3] — step=1.0
        let step = uniform_knot_step(&knots, 3);
        assert_eq!(step, Some(1.0));
    }

    #[test]
    fn uniform_knot_step_rejects_non_uniform() {
        // Knots with non-uniform internal spacing
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 4.0, 4.0, 4.0, 4.0];
        assert_eq!(uniform_knot_step(&knots, 3), None);
    }

    #[test]
    fn uniform_knot_step_rejects_bezier() {
        // Single-span Bezier: no internal knots to be "uniform" over
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(uniform_knot_step(&knots, 3), None);
    }

    #[test]
    fn find_span_uniform_interior() {
        let knots = cubic_knots();
        let step = uniform_knot_step(&knots, 3).expect("uniform");
        assert_eq!(find_span_uniform(6, 3, 0.5, &knots, step), 3);
        assert_eq!(find_span_uniform(6, 3, 1.5, &knots, step), 4);
        assert_eq!(find_span_uniform(6, 3, 2.5, &knots, step), 5);
    }

    #[test]
    fn find_span_uniform_endpoints() {
        let knots = cubic_knots();
        let step = uniform_knot_step(&knots, 3).expect("uniform");
        assert_eq!(find_span_uniform(6, 3, 0.0, &knots, step), 3);
        assert_eq!(find_span_uniform(6, 3, 3.0, &knots, step), 5);
    }

    #[test]
    fn find_span_uniform_at_knot() {
        let knots = cubic_knots();
        let step = uniform_knot_step(&knots, 3).expect("uniform");
        assert_eq!(find_span_uniform(6, 3, 1.0, &knots, step), 4);
        assert_eq!(find_span_uniform(6, 3, 2.0, &knots, step), 5);
    }

    #[test]
    fn find_span_uniform_quadratic() {
        // Quadratic with more internal knots: [0,0,0, 0.25,0.5,0.75, 1,1,1]
        // 6 control points, degree 2
        let knots = vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0];
        let step = uniform_knot_step(&knots, 2).expect("uniform");
        assert!((step - 0.25).abs() < 1e-15);
        for i in 0..=100 {
            let u = i as f64 / 100.0;
            let expected = find_span(6, 2, u, &knots);
            let got = find_span_uniform(6, 2, u, &knots, step);
            assert_eq!(got, expected, "mismatch at u={u}");
        }
    }
}
