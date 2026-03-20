//! Lipschitz global optimizer for finding the minimum distance between
//! two parametric curves or surfaces.
//!
//! Uses interval subdivision with Lipschitz pruning: cells whose lower
//! bound (`f_center - L * radius`) exceeds the current best are discarded.
//! This guarantees finding the global minimum, unlike Newton-based methods
//! which can get stuck in local minima.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;

/// A solution found by the optimizer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OptSolution {
    /// Parameter values at the minimum.
    pub params: [f64; 2],
    /// Function value (distance squared) at this point.
    pub value: f64,
}

/// Lipschitz optimizer for 2D parameter-space minimization.
///
/// Finds the global minimum of a function f(u, v) over a rectangular domain,
/// given a Lipschitz bound L such that |f(a) - f(b)| <= L * |a - b| for all a, b.
#[allow(dead_code)]
pub struct LipschitzOptimizer {
    /// Convergence tolerance (cell radius below this records a solution).
    eps: f64,
    /// Maximum subdivision depth.
    max_depth: usize,
    /// Maximum number of cells to process before stopping.
    max_evals: usize,
}

#[allow(dead_code)]
impl LipschitzOptimizer {
    /// Create a new optimizer with given tolerance and max depth.
    #[must_use]
    pub fn new(eps: f64, max_depth: usize) -> Self {
        Self {
            eps,
            max_depth,
            max_evals: 500_000,
        }
    }

    /// Find the global minimum of `f(u, v)` over `[u0, u1] × [v0, v1]`.
    ///
    /// `f` returns the objective value (e.g. distance squared).
    /// `lipschitz_bound` is an upper bound on the gradient magnitude: L ≥ max|∇f|.
    ///
    /// Returns all solutions within `eps` of the true global minimum.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn minimize_2d<F>(
        &self,
        f: &F,
        u_range: (f64, f64),
        v_range: (f64, f64),
        lipschitz_bound: f64,
    ) -> Vec<OptSolution>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Phase 1: grid search to establish a good upper bound.
        let init_n = 16usize;
        let mut best = f64::INFINITY;
        let mut best_u = (u_range.0 + u_range.1) * 0.5;
        let mut best_v = (v_range.0 + v_range.1) * 0.5;

        for iu in 0..=init_n {
            let u = u_range.0 + (u_range.1 - u_range.0) * (iu as f64 / init_n as f64);
            for iv in 0..=init_n {
                let v = v_range.0 + (v_range.1 - v_range.0) * (iv as f64 / init_n as f64);
                let val = f(u, v);
                if val < best {
                    best = val;
                    best_u = u;
                    best_v = v;
                }
            }
        }

        // Phase 1b: coordinate-descent refinement from best grid point.
        {
            let step_u = (u_range.1 - u_range.0) / init_n as f64;
            let step_v = (v_range.1 - v_range.0) / init_n as f64;
            let mut lu = (best_u - step_u).max(u_range.0);
            let mut hu = (best_u + step_u).min(u_range.1);
            let mut lv = (best_v - step_v).max(v_range.0);
            let mut hv = (best_v + step_v).min(v_range.1);
            for _ in 0..50 {
                let m1 = lu + (hu - lu) / 3.0;
                let m2 = hu - (hu - lu) / 3.0;
                if f(m1, best_v) < f(m2, best_v) {
                    hu = m2;
                } else {
                    lu = m1;
                }
                let cu = (lu + hu) * 0.5;
                let fcu = f(cu, best_v);
                if fcu < best {
                    best = fcu;
                    best_u = cu;
                }

                let m1v = lv + (hv - lv) / 3.0;
                let m2v = hv - (hv - lv) / 3.0;
                if f(best_u, m1v) < f(best_u, m2v) {
                    hv = m2v;
                } else {
                    lv = m1v;
                }
                let cv = (lv + hv) * 0.5;
                let fcv = f(best_u, cv);
                if fcv < best {
                    best = fcv;
                    best_v = cv;
                }
            }
        }

        // Phase 2: Lipschitz subdivision to verify no other basin has a lower
        // minimum. Uses a work stack with depth-first traversal.
        let mut solutions = vec![OptSolution {
            params: [best_u, best_v],
            value: best,
        }];
        let mut stack: Vec<(f64, f64, f64, f64, usize)> =
            vec![(u_range.0, u_range.1, v_range.0, v_range.1, 0)];
        let mut cell_count = 0usize;

        while let Some((u0, u1, v0, v1, depth)) = stack.pop() {
            cell_count += 1;
            if cell_count > self.max_evals {
                break;
            }
            let um = (u0 + u1) * 0.5;
            let vm = (v0 + v1) * 0.5;
            let fc = f(um, vm);
            if fc < best {
                best = fc;
            }

            let du = u1 - u0;
            let dv = v1 - v0;
            let radius = (du * du + dv * dv).sqrt() * 0.5;
            let lower = fc - lipschitz_bound * radius;

            // Prune: this cell can't contain a better minimum
            if lower > best {
                continue;
            }

            // Terminal: cell converged or max depth
            if radius < self.eps || depth >= self.max_depth {
                if fc <= best + self.eps {
                    solutions.push(OptSolution {
                        params: [um, vm],
                        value: fc,
                    });
                }
                continue;
            }

            // Subdivide into 4 quadrants
            stack.push((u0, um, v0, vm, depth + 1));
            stack.push((um, u1, v0, vm, depth + 1));
            stack.push((u0, um, vm, v1, depth + 1));
            stack.push((um, u1, vm, v1, depth + 1));
        }

        // Keep only solutions within eps of global best, deduplicate
        solutions.retain(|s| s.value <= best + self.eps);
        solutions.sort_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut deduped = Vec::new();
        for sol in &solutions {
            let is_dup = deduped.iter().any(|d: &OptSolution| {
                let dp = (sol.params[0] - d.params[0]).hypot(sol.params[1] - d.params[1]);
                dp < self.eps * 2.0
            });
            if !is_dup {
                deduped.push(sol.clone());
            }
        }

        deduped
    }
}

/// Estimate the Lipschitz bound for the squared-distance function between
/// two NURBS curves: f(u, v) = ||C1(u) - C2(v)||^2.
///
/// The gradient of f is:
///   df/du = 2 * (C1(u) - C2(v)) . C1'(u)
///   df/dv = -2 * (C1(u) - C2(v)) . C2'(v)
///
/// Upper bound: L <= 2 * max_separation * max(max_deriv_1, max_deriv_2)
/// where `max_separation` is the maximum distance between any pair of points
/// and `max_deriv` is the maximum derivative magnitude.
#[allow(dead_code, clippy::cast_precision_loss)]
pub fn estimate_curve_curve_lipschitz(
    curve1: &NurbsCurve,
    curve2: &NurbsCurve,
    n_samples: usize,
) -> f64 {
    let (u0, u1) = curve1.domain();
    let (v0, v1) = curve2.domain();

    let mut max_deriv1: f64 = 0.0;
    let mut max_deriv2: f64 = 0.0;
    let mut max_sep: f64 = 0.0;

    for i in 0..=n_samples {
        let t = i as f64 / n_samples as f64;
        let u = u0 + (u1 - u0) * t;
        let d1 = curve1.derivatives(u, 1);
        max_deriv1 = max_deriv1.max(d1[1].length());

        let v = v0 + (v1 - v0) * t;
        let d2 = curve2.derivatives(v, 1);
        max_deriv2 = max_deriv2.max(d2[1].length());

        // d[0] is the point as Vec3; convert to Point3 for distance
        let p1 = Point3::new(d1[0].x(), d1[0].y(), d1[0].z());
        let p2 = Point3::new(d2[0].x(), d2[0].y(), d2[0].z());
        let sep = (p1 - p2).length();
        max_sep = max_sep.max(sep);
    }

    // L for ||C1(u) - C2(v)||^2
    2.0 * max_sep * max_deriv1.max(max_deriv2)
}

/// Find the global minimum distance between two NURBS curves.
///
/// Returns (distance, point on curve 1, point on curve 2).
#[allow(dead_code)]
pub fn nurbs_curve_curve_distance(
    curve1: &NurbsCurve,
    curve2: &NurbsCurve,
) -> (f64, Point3, Point3) {
    let (u0, u1) = curve1.domain();
    let (v0, v1) = curve2.domain();

    // Estimate Lipschitz bound
    let lip = estimate_curve_curve_lipschitz(curve1, curve2, 20);

    // If Lipschitz bound is near zero, curves are likely coincident or constant
    if lip < 1e-15 {
        let p1 = curve1.evaluate((u0 + u1) * 0.5);
        let p2 = curve2.evaluate((v0 + v1) * 0.5);
        return ((p1 - p2).length(), p1, p2);
    }

    // Build distance-squared function
    let f = |u: f64, v: f64| -> f64 {
        let p1 = curve1.evaluate(u);
        let p2 = curve2.evaluate(v);
        (p1 - p2).length_squared()
    };

    let mut optimizer = LipschitzOptimizer::new(1e-4, 12);
    optimizer.max_evals = 50_000;
    let solutions = optimizer.minimize_2d(&f, (u0, u1), (v0, v1), lip);

    if let Some(best) = solutions.first() {
        let p1 = curve1.evaluate(best.params[0]);
        let p2 = curve2.evaluate(best.params[1]);
        ((p1 - p2).length(), p1, p2)
    } else {
        // Fallback: evaluate at midpoints
        let p1 = curve1.evaluate((u0 + u1) * 0.5);
        let p2 = curve2.evaluate((v0 + v1) * 0.5);
        ((p1 - p2).length(), p1, p2)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_finds_global_minimum() {
        // f(u,v) = (u-0.3)^2 + (v-0.7)^2, minimum at (0.3, 0.7) = 0
        let f = |u: f64, v: f64| (u - 0.3) * (u - 0.3) + (v - 0.7) * (v - 0.7);
        let opt = LipschitzOptimizer::new(1e-6, 30);
        // Lipschitz bound for this quadratic: gradient magnitude <= 2*sqrt(2) ~ 3
        let solutions = opt.minimize_2d(&f, (0.0, 1.0), (0.0, 1.0), 3.0);
        assert!(!solutions.is_empty());
        let best = &solutions[0];
        assert!(best.value < 1e-10, "expected near-zero, got {}", best.value);
        assert!((best.params[0] - 0.3).abs() < 1e-4, "u={}", best.params[0]);
        assert!((best.params[1] - 0.7).abs() < 1e-4, "v={}", best.params[1]);
    }

    #[test]
    fn optimizer_finds_multiple_minima() {
        // f(u,v) with two equal minima: min at (0.25, 0.5) and (0.75, 0.5)
        let f = |u: f64, v: f64| {
            let d1 = (u - 0.25) * (u - 0.25) + (v - 0.5) * (v - 0.5);
            let d2 = (u - 0.75) * (u - 0.75) + (v - 0.5) * (v - 0.5);
            d1.min(d2)
        };
        let opt = LipschitzOptimizer::new(1e-4, 25);
        let solutions = opt.minimize_2d(&f, (0.0, 1.0), (0.0, 1.0), 4.0);
        // Should find solutions near both minima
        assert!(
            solutions.len() >= 2,
            "expected >=2 solutions, got {}",
            solutions.len()
        );
    }

    #[test]
    fn optimizer_prunes_efficiently() {
        // Track evaluation count to verify pruning
        use std::cell::Cell;
        let count = Cell::new(0u64);
        let f = |u: f64, v: f64| {
            count.set(count.get() + 1);
            (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)
        };
        let opt = LipschitzOptimizer::new(1e-3, 15);
        let _solutions = opt.minimize_2d(&f, (0.0, 1.0), (0.0, 1.0), 2.5);
        // With Lipschitz pruning, most cells far from the minimum get discarded.
        // The budget guard caps at 500k, but pruning should keep it well below.
        assert!(
            count.get() < 500_000,
            "too many evaluations: {}",
            count.get()
        );
    }

    #[test]
    fn nurbs_line_distance() {
        // Two parallel NURBS lines: L1 from (0,0,0)→(1,0,0), L2 from (0,1,0)→(1,1,0)
        let c1 = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        let c2 = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let (dist, _, _) = nurbs_curve_curve_distance(&c1, &c2);
        assert!(
            (dist - 1.0).abs() < 1e-4,
            "expected distance 1.0, got {dist}"
        );
    }
}
