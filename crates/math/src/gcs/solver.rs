//! DogLeg trust-region solver.
//!
//! Blends Gauss-Newton and steepest-descent steps inside a trust region
//! for globally convergent nonlinear least-squares solving. This is the
//! default solver in FreeCAD's PlaneGCS for the same reason: it converges
//! reliably even when the initial guess is far from the solution.

use super::qr::QrResult;

/// Result of a solve attempt.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations used.
    pub iterations: usize,
    /// Maximum absolute residual after solving.
    pub max_residual: f64,
}

/// DogLeg trust-region solver for the system `r(params) = 0`.
///
/// - `params`: initial parameter values (modified in-place on success)
/// - `residual_fn`: compute residuals given current params
/// - `jacobian_fn`: compute row-major Jacobian given current params
/// - `num_residuals`: number of residual equations
/// - `max_iter`: maximum iterations
/// - `tol`: convergence tolerance on max |residual|
///
/// Returns the solve result. On convergence, `params` holds the solution.
#[allow(clippy::too_many_lines)]
pub fn solve_dogleg<F, G>(
    params: &mut [f64],
    residual_fn: &F,
    jacobian_fn: &G,
    num_residuals: usize,
    max_iter: usize,
    tol: f64,
) -> SolveResult
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = params.len();
    if n == 0 || num_residuals == 0 {
        let r = residual_fn(params);
        let max_r = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        return SolveResult {
            converged: max_r < tol,
            iterations: 0,
            max_residual: max_r,
        };
    }

    // Scale-aware initial trust radius
    let param_norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
    let delta_max = 1e4;
    let delta_min = 1e-15;
    let mut delta = (1.0_f64).max(0.1 * param_norm).min(delta_max);

    for iteration in 0..max_iter {
        let r = residual_fn(params);
        let max_r = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

        if max_r < tol {
            return SolveResult {
                converged: true,
                iterations: iteration,
                max_residual: max_r,
            };
        }

        // Build Jacobian. Save a copy before factorization, since QR modifies
        // the matrix in-place and we need the original values for the gradient
        // and predicted-reduction computations below.
        let jac_orig = jacobian_fn(params);
        let mut jac = jac_orig.clone();
        let qr = QrResult::factorize(&mut jac, num_residuals, n);

        // Gauss-Newton step: solve J * h_gn = -r
        let neg_r: Vec<f64> = r.iter().map(|&v| -v).collect();
        let h_gn = qr.solve_least_squares(&neg_r);

        // Gradient: g = J^T * r  (use the saved pre-factorization Jacobian)
        let mut g = vec![0.0; n];
        for i in 0..num_residuals {
            for j in 0..n {
                g[j] += jac_orig[i * n + j] * r[i];
            }
        }

        // Steepest descent step: h_sd = -alpha * g
        // alpha = ||g||² / ||J*g||²
        let g_norm_sq: f64 = g.iter().map(|&v| v * v).sum();
        if g_norm_sq < 1e-300 {
            // Zero gradient — we're at a stationary point
            return SolveResult {
                converged: max_r < tol,
                iterations: iteration,
                max_residual: max_r,
            };
        }

        // Compute J*g
        let mut jg = vec![0.0; num_residuals];
        for i in 0..num_residuals {
            for j in 0..n {
                jg[i] += jac_orig[i * n + j] * g[j];
            }
        }
        let jg_norm_sq: f64 = jg.iter().map(|&v| v * v).sum();
        let alpha = if jg_norm_sq > 1e-300 {
            g_norm_sq / jg_norm_sq
        } else {
            1.0
        };

        let h_sd: Vec<f64> = g.iter().map(|&v| -alpha * v).collect();

        // DogLeg blending
        let h = dogleg_step(&h_gn, &h_sd, delta);
        let h_norm = h.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // Trial point
        let trial: Vec<f64> = params.iter().zip(h.iter()).map(|(&p, &d)| p + d).collect();
        let r_trial = residual_fn(&trial);

        // Compute reduction ratio
        let cost_current: f64 = r.iter().map(|&v| v * v).sum::<f64>() * 0.5;
        let cost_trial: f64 = r_trial.iter().map(|&v| v * v).sum::<f64>() * 0.5;
        let actual_reduction = cost_current - cost_trial;

        // Predicted reduction from linear model
        let mut jh = vec![0.0; num_residuals];
        for i in 0..num_residuals {
            for j in 0..n {
                jh[i] += jac_orig[i * n + j] * h[j];
            }
        }
        let predicted: f64 = {
            let mut pred = 0.0;
            for i in 0..num_residuals {
                pred += r[i] * jh[i];
                pred += 0.5 * jh[i] * jh[i];
            }
            -pred
        };

        let rho = if predicted.abs() < 1e-300 {
            if actual_reduction > 0.0 { 1.0 } else { 0.0 }
        } else {
            actual_reduction / predicted
        };

        // Update trust region
        if rho > 0.75 {
            delta = (2.0 * delta).min(delta_max);
        } else if rho < 0.25 {
            delta = (delta / 4.0).max(delta_min);
        }

        // Accept or reject step
        if rho > 0.0 {
            params.copy_from_slice(&trial);
        }

        // Check if step is too small to make progress
        if h_norm < 1e-15 * (1.0 + param_norm) {
            let final_r = residual_fn(params);
            let final_max = final_r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            return SolveResult {
                converged: final_max < tol,
                iterations: iteration + 1,
                max_residual: final_max,
            };
        }
    }

    let r = residual_fn(params);
    let max_r = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    SolveResult {
        converged: max_r < tol,
        iterations: max_iter,
        max_residual: max_r,
    }
}

/// Compute the DogLeg step: choose GN, SD, or a blend based on trust radius.
fn dogleg_step(h_gn: &[f64], h_sd: &[f64], delta: f64) -> Vec<f64> {
    let gn_norm = h_gn.iter().map(|&v| v * v).sum::<f64>().sqrt();

    // If GN step is within trust region, use it
    if gn_norm <= delta {
        return h_gn.to_vec();
    }

    let sd_norm = h_sd.iter().map(|&v| v * v).sum::<f64>().sqrt();

    // If even SD step exceeds trust region, scale it down
    if sd_norm >= delta {
        let scale = delta / sd_norm;
        return h_sd.iter().map(|&v| v * scale).collect();
    }

    // Interpolate between SD and GN: h = h_sd + t * (h_gn - h_sd)
    // Find t such that ||h_sd + t * (h_gn - h_sd)|| = delta
    let diff: Vec<f64> = h_gn.iter().zip(h_sd.iter()).map(|(&g, &s)| g - s).collect();

    let a: f64 = diff.iter().map(|&v| v * v).sum();
    let b: f64 = h_sd
        .iter()
        .zip(diff.iter())
        .map(|(&s, &d)| s * d)
        .sum::<f64>()
        * 2.0;
    let c: f64 = sd_norm * sd_norm - delta * delta;

    // Solve a*t² + b*t + c = 0 for the positive root
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let t = (-b + discriminant.sqrt()) / (2.0 * a);
    let t = t.clamp(0.0, 1.0);

    h_sd.iter()
        .zip(diff.iter())
        .map(|(&s, &d)| s + t * d)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fix_x_converges_in_one_iter() {
        // Single param, single constraint: x = 5.0, starting at x = 3.0
        let mut params = vec![3.0];
        let result = solve_dogleg(
            &mut params,
            &|p: &[f64]| vec![p[0] - 5.0],
            &|_p: &[f64]| vec![1.0],
            1,
            100,
            1e-12,
        );
        assert!(result.converged);
        assert!(result.iterations <= 2, "iters = {}", result.iterations);
        assert!((params[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn quadratic_residual() {
        // x² - 4 = 0 → x = 2
        let mut params = vec![3.0];
        let result = solve_dogleg(
            &mut params,
            &|p: &[f64]| vec![p[0] * p[0] - 4.0],
            &|p: &[f64]| vec![2.0 * p[0]],
            1,
            100,
            1e-12,
        );
        assert!(result.converged);
        assert!((params[0] - 2.0).abs() < 1e-10, "x = {}", params[0]);
    }

    #[test]
    fn two_variable_system() {
        // x + y = 3, x - y = 1 → x = 2, y = 1
        let mut params = vec![0.0, 0.0];
        let result = solve_dogleg(
            &mut params,
            &|p: &[f64]| vec![p[0] + p[1] - 3.0, p[0] - p[1] - 1.0],
            &|_p: &[f64]| vec![1.0, 1.0, 1.0, -1.0],
            2,
            100,
            1e-12,
        );
        assert!(result.converged);
        assert!((params[0] - 2.0).abs() < 1e-10);
        assert!((params[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_system() {
        let mut params = vec![];
        let result = solve_dogleg(
            &mut params,
            &|_p: &[f64]| vec![],
            &|_p: &[f64]| vec![],
            0,
            100,
            1e-12,
        );
        assert!(result.converged);
    }

    #[test]
    fn trust_region_shrinks_on_bad_step() {
        // A system where the Gauss-Newton step overshoots.
        // f(x) = x^3 - 8, starting far from solution
        let mut params = vec![10.0];
        let result = solve_dogleg(
            &mut params,
            &|p: &[f64]| vec![p[0] * p[0] * p[0] - 8.0],
            &|p: &[f64]| vec![3.0 * p[0] * p[0]],
            1,
            200,
            1e-10,
        );
        assert!(result.converged, "max_r = {}", result.max_residual);
        assert!((params[0] - 2.0).abs() < 1e-8, "x = {}", params[0]);
    }
}
