//! Precomputed polynomial (power-basis) form for fast NURBS basis evaluation.
//!
//! [`PowerBasis1D`] converts B-spline basis functions from the recursive
//! Cox-de Boor form into explicit polynomials in a shifted local coordinate
//! `t = u - knots[span]`. Evaluation then uses Horner's method — O(p) per
//! basis function instead of the O(p^2) recurrence.
//!
//! This is beneficial when the same knot vector is evaluated many times
//! (e.g. tessellation grids), since the O(p^2) conversion cost is amortised.

use super::basis::basis_funs_into;

/// Precomputed polynomial coefficients for a 1-D B-spline basis.
///
/// For each knot span `[knots[j], knots[j+1]]`, stores `(degree+1)^2`
/// coefficients representing the `(degree+1)` non-zero basis functions as
/// polynomials in `t = u - knots[j]` (shifted local coordinate).
///
/// Storage layout:
/// `coeffs[span_index * (p+1)^2 + basis_fn * (p+1) + poly_coeff]`
/// where `span_index = span - degree`.
pub struct PowerBasis1D {
    degree: usize,
    n_spans: usize,
    coeffs: Vec<f64>,
    /// `knots[span]` for each active span, used to compute the local shift.
    span_starts: Vec<f64>,
}

impl PowerBasis1D {
    /// Build a power-basis representation from a knot vector and degree.
    ///
    /// Samples the Cox-de Boor basis at `(p+1)` points per span and solves a
    /// Vandermonde system (via Newton divided differences) to recover the
    /// polynomial coefficients.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_knots(knots: &[f64], degree: usize) -> Self {
        let p = degree;
        let n_ctrl = knots.len() - degree - 1; // number of control points
        let n_spans = n_ctrl - degree; // number of active spans
        let block = (p + 1) * (p + 1);
        let mut coeffs = vec![0.0; n_spans * block];
        let mut span_starts = Vec::with_capacity(n_spans);

        // Temporaries for sampling.
        let mut basis_vals = vec![0.0; p + 1];
        // Per-basis-function sample values for interpolation.
        let mut samples = vec![0.0; p + 1];
        // Local-coordinate sample points.
        let mut t_pts = vec![0.0; p + 1];

        for si in 0..n_spans {
            let span = si + p; // actual span index
            let t0 = knots[span];
            let t1 = knots[span + 1];
            span_starts.push(t0);
            let h = t1 - t0;

            // Skip zero-length spans (coefficients remain zero).
            if h <= 0.0 {
                continue;
            }

            // Build sample points in local coordinates.
            for k in 0..=p {
                t_pts[k] = k as f64 * h / p as f64;
            }

            // For each basis function j, collect samples and interpolate.
            // We sample all basis functions at each u, then pick out function j.
            //
            // First, gather all samples at each point.
            let mut all_samples = vec![0.0; (p + 1) * (p + 1)];
            for k in 0..=p {
                let u = t0 + t_pts[k];
                basis_funs_into(span, u, degree, knots, &mut basis_vals);
                for j in 0..=p {
                    all_samples[j * (p + 1) + k] = basis_vals[j];
                }
            }

            // For each basis function, solve Newton interpolation then convert
            // to monomial form.
            for j in 0..=p {
                // Copy samples for this basis function.
                for k in 0..=p {
                    samples[k] = all_samples[j * (p + 1) + k];
                }

                // Newton divided differences (in-place on `samples`).
                // After this, samples[k] = f[t_0, t_1, ..., t_k].
                for k in 1..=p {
                    for i in (k..=p).rev() {
                        samples[i] = (samples[i] - samples[i - 1]) / (t_pts[i] - t_pts[i - k]);
                    }
                }

                // Convert Newton form to monomial form.
                // Newton form: f(t) = d[0] + d[1]*(t - t_0) + d[2]*(t - t_0)*(t - t_1) + ...
                // We expand right-to-left using Horner-like nesting.
                //
                // Start with the highest-degree coefficient and fold down:
                //   poly = d[p]
                //   poly = poly * (t - t_{p-1}) + d[p-1]
                //   ...
                //   poly = poly * (t - t_0) + d[0]
                //
                // We track the monomial coefficients explicitly.
                let base = si * block + j * (p + 1);
                let c = &mut coeffs[base..=base + p];

                // Initialize with just the highest Newton coefficient.
                c[0] = samples[p];

                // Fold in each lower Newton coefficient.
                for m in (0..p).rev() {
                    // Multiply current polynomial by (t - t_pts[m]):
                    //   c[k+1] += c[k], c[k] = c[k] * (-t_pts[m]) + d[m] if k==0
                    // Process from high to low to avoid overwriting.
                    for k in (1..=p - m).rev() {
                        // c[k] = c[k-1] + c[k] * (-t_pts[m])
                        c[k] = (-t_pts[m]).mul_add(c[k], c[k - 1]);
                    }
                    c[0] = (-t_pts[m]).mul_add(c[0], samples[m]);
                }
            }
        }

        Self {
            degree,
            n_spans,
            coeffs,
            span_starts,
        }
    }

    /// Evaluate all `(degree+1)` non-zero basis functions at `u` via Horner's method.
    ///
    /// Writes results into `out[0..=degree]`. The caller must provide the span
    /// index (from `find_span`).
    pub fn horner(&self, span: usize, u: f64, out: &mut [f64]) {
        let p = self.degree;
        let span_idx = span - p;
        let t = u - self.span_starts[span_idx];
        let base = span_idx * (p + 1) * (p + 1);

        for j in 0..=p {
            let coeff_base = base + j * (p + 1);
            let mut val = self.coeffs[coeff_base + p];
            for k in (0..p).rev() {
                val = val.mul_add(t, self.coeffs[coeff_base + k]);
            }
            out[j] = val;
        }
    }

    /// Evaluate basis functions and their first derivatives at `u`.
    ///
    /// Writes values into `vals[0..=degree]` and first derivatives into
    /// `derivs[0..=degree]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn horner_with_derivs(&self, span: usize, u: f64, vals: &mut [f64], derivs: &mut [f64]) {
        let p = self.degree;
        let span_idx = span - p;
        let t = u - self.span_starts[span_idx];
        let base = span_idx * (p + 1) * (p + 1);

        for j in 0..=p {
            let coeff_base = base + j * (p + 1);

            if p == 0 {
                vals[j] = self.coeffs[coeff_base];
                derivs[j] = 0.0;
                continue;
            }

            // Horner for value and derivative simultaneously.
            // f(t)  = c[p]*t^p + c[p-1]*t^{p-1} + ... + c[0]
            // f'(t) = p*c[p]*t^{p-1} + (p-1)*c[p-1]*t^{p-2} + ... + c[1]
            let mut val = self.coeffs[coeff_base + p];
            let mut dval = self.coeffs[coeff_base + p] * p as f64;

            for k in (1..p).rev() {
                val = val.mul_add(t, self.coeffs[coeff_base + k]);
                dval = dval.mul_add(t, self.coeffs[coeff_base + k] * k as f64);
            }
            val = val.mul_add(t, self.coeffs[coeff_base]);

            vals[j] = val;
            derivs[j] = dval;
        }
    }

    /// Number of active knot spans.
    #[must_use]
    pub const fn n_spans(&self) -> usize {
        self.n_spans
    }

    /// Polynomial degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::basis::{basis_funs, ders_basis_funs, find_span};
    use super::*;

    fn cubic_knots() -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    }

    #[test]
    fn power_basis_matches_cox_de_boor() {
        let knots = cubic_knots();
        let degree = 3;
        let pb = PowerBasis1D::from_knots(&knots, degree);
        for &u in &[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let span = find_span(6, degree, u, &knots);
            let expected = basis_funs(span, u, degree, &knots);
            let mut got = [0.0_f64; 4];
            pb.horner(span, u, &mut got[..=degree]);
            for j in 0..=degree {
                assert!(
                    (expected[j] - got[j]).abs() < 1e-12,
                    "u={u}, span={span}, j={j}: {:.15} vs {:.15}",
                    expected[j],
                    got[j]
                );
            }
        }
    }

    #[test]
    fn horner_derivs_match_ders_basis_funs() {
        let knots = cubic_knots();
        let degree = 3;
        let pb = PowerBasis1D::from_knots(&knots, degree);
        for &u in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let span = find_span(6, degree, u, &knots);
            let expected = ders_basis_funs(span, u, degree, 1, &knots);
            let mut vals = [0.0_f64; 4];
            let mut derivs = [0.0_f64; 4];
            pb.horner_with_derivs(span, u, &mut vals[..=degree], &mut derivs[..=degree]);
            for j in 0..=degree {
                assert!(
                    (expected[0][j] - vals[j]).abs() < 1e-12,
                    "u={u}, j={j}: val {:.15} vs {:.15}",
                    expected[0][j],
                    vals[j]
                );
                assert!(
                    (expected[1][j] - derivs[j]).abs() < 1e-10,
                    "u={u}, j={j}: deriv {:.15} vs {:.15}",
                    expected[1][j],
                    derivs[j]
                );
            }
        }
    }

    #[test]
    fn power_basis_partition_of_unity() {
        let knots = cubic_knots();
        let degree = 3;
        let pb = PowerBasis1D::from_knots(&knots, degree);
        for i in 0..=30 {
            #[allow(clippy::cast_precision_loss)]
            let u = i as f64 / 10.0;
            let span = find_span(6, degree, u, &knots);
            let mut vals = [0.0_f64; 4];
            pb.horner(span, u, &mut vals[..=degree]);
            let sum: f64 = vals.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "u={u}: partition of unity sum = {sum}"
            );
        }
    }

    #[test]
    fn quadratic_knots() {
        // Quadratic with non-uniform internal spacing.
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.5, 3.0, 3.0, 3.0];
        let degree = 2;
        let n = knots.len() - degree - 1; // 5 control points
        let pb = PowerBasis1D::from_knots(&knots, degree);

        for i in 0..=60 {
            #[allow(clippy::cast_precision_loss)]
            let u = i as f64 / 20.0;
            let span = find_span(n, degree, u, &knots);
            let expected = basis_funs(span, u, degree, &knots);
            let mut got = [0.0_f64; 3];
            pb.horner(span, u, &mut got[..=degree]);
            for j in 0..=degree {
                assert!(
                    (expected[j] - got[j]).abs() < 1e-12,
                    "u={u}, j={j}: {:.15} vs {:.15}",
                    expected[j],
                    got[j]
                );
            }
        }
    }

    #[test]
    fn linear_basis() {
        // Degree 1: [0,0,1,2,3,3] — 4 control points, 3 spans
        let knots = vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        let degree = 1;
        let n = knots.len() - degree - 1; // 4
        let pb = PowerBasis1D::from_knots(&knots, degree);

        for i in 0..=30 {
            #[allow(clippy::cast_precision_loss)]
            let u = i as f64 / 10.0;
            let span = find_span(n, degree, u, &knots);
            let expected = basis_funs(span, u, degree, &knots);
            let mut got = [0.0_f64; 2];
            pb.horner(span, u, &mut got[..=degree]);
            for j in 0..=degree {
                assert!(
                    (expected[j] - got[j]).abs() < 1e-14,
                    "u={u}, j={j}: {:.15} vs {:.15}",
                    expected[j],
                    got[j]
                );
            }
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_power_basis_matches_cox_de_boor(u in 0.0_f64..=3.0) {
            let knots = cubic_knots();
            let degree = 3;
            let pb = PowerBasis1D::from_knots(&knots, degree);
            let span = find_span(6, degree, u, &knots);
            let expected = basis_funs(span, u, degree, &knots);
            let mut got = [0.0_f64; 4];
            pb.horner(span, u, &mut got[..=degree]);
            for j in 0..=degree {
                prop_assert!(
                    (expected[j] - got[j]).abs() < 1e-10,
                    "u={}, j={}: {} vs {}",
                    u,
                    j,
                    expected[j],
                    got[j]
                );
            }
        }

        #[test]
        fn prop_derivs_sum_to_zero(u in 0.0_f64..=3.0) {
            let knots = cubic_knots();
            let degree = 3;
            let pb = PowerBasis1D::from_knots(&knots, degree);
            let span = find_span(6, degree, u, &knots);
            let mut vals = [0.0_f64; 4];
            let mut derivs = [0.0_f64; 4];
            pb.horner_with_derivs(span, u, &mut vals[..=degree], &mut derivs[..=degree]);
            let sum: f64 = derivs.iter().sum();
            prop_assert!(
                sum.abs() < 1e-8,
                "u={}: derivative sum = {}",
                u,
                sum
            );
        }
    }
}
