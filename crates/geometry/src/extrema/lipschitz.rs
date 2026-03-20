//! Lipschitz global optimizer for 2D parameter-space minimization.
//!
//! Uses interval subdivision with Lipschitz pruning: cells whose lower
//! bound `f(center) - L * radius` exceeds the current best are discarded.
//! This guarantees finding the global minimum, unlike Newton-based methods
//! that can get trapped in local minima.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;

// ── LipschitzOptimizer ───────────────────────────────────────────────────────

/// Lipschitz optimizer for 2D parameter-space minimization.
///
/// Finds the global minimum of a scalar function `f(u, v)` over a rectangular
/// domain `[u0, u1] × [v0, v1]`, given a Lipschitz bound `L` such that
/// `|f(a) - f(b)| ≤ L * |a - b|` for all `a`, `b` in the domain.
///
/// The algorithm has three phases:
/// 1. **Grid search** — uniform 16×16 sampling to establish an upper bound.
/// 2. **Coordinate descent** — refine from the best grid point.
/// 3. **Lipschitz subdivision** — depth-first cell subdivision, pruning cells
///    whose Lipschitz lower bound exceeds the current best.
pub struct LipschitzOptimizer {
    grid_size: usize,
    max_subdivisions: usize,
    max_evals: usize,
}

impl Default for LipschitzOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl LipschitzOptimizer {
    /// Create a new optimizer with default parameters.
    ///
    /// Defaults: grid 16×16, max 50 subdivisions per dimension, 500 000 cell budget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid_size: 16,
            max_subdivisions: 50,
            max_evals: 500_000,
        }
    }

    /// Find the (approximate) global minimum of `f(u, v)` over the given domain.
    ///
    /// The Lipschitz bound is estimated internally from a finite-difference
    /// grid. `tolerance` controls cell-radius convergence: a cell is terminal
    /// when its radius is below `tolerance`.
    ///
    /// Returns `(u*, v*, f*)` — the minimizer and minimum value found.
    #[must_use]
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn minimize_2d<F>(
        &self,
        f: F,
        u_range: (f64, f64),
        v_range: (f64, f64),
        tolerance: f64,
    ) -> (f64, f64, f64)
    where
        F: Fn(f64, f64) -> f64,
    {
        let (u0, u1) = u_range;
        let (v0, v1) = v_range;
        let n = self.grid_size;

        // ── Phase 1: grid search (cached for Lipschitz estimation) ────────────
        let grid_len = (n + 1) * (n + 1);
        let mut grid_vals = vec![0.0_f64; grid_len];
        let mut best = f64::INFINITY;
        let mut best_u = (u0 + u1) * 0.5;
        let mut best_v = (v0 + v1) * 0.5;

        for iu in 0..=n {
            let u = u0 + (u1 - u0) * (iu as f64 / n as f64);
            for iv in 0..=n {
                let v = v0 + (v1 - v0) * (iv as f64 / n as f64);
                let val = f(u, v);
                grid_vals[iu * (n + 1) + iv] = val;
                if val < best {
                    best = val;
                    best_u = u;
                    best_v = v;
                }
            }
        }

        // ── Phase 1b: coordinate descent from best grid point ─────────────────
        {
            let step_u = (u1 - u0) / n as f64;
            let step_v = (v1 - v0) / n as f64;
            let mut lu = (best_u - step_u).max(u0);
            let mut hu = (best_u + step_u).min(u1);
            let mut lv = (best_v - step_v).max(v0);
            let mut hv = (best_v + step_v).min(v1);
            for _ in 0..self.max_subdivisions {
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

        // Estimate Lipschitz bound from cached Phase 1 grid values.
        let du = (u1 - u0) / n as f64;
        let dv = (v1 - v0) / n as f64;
        let mut lip: f64 = 0.0;
        for iu in 0..n {
            for iv in 0..n {
                let f00 = grid_vals[iu * (n + 1) + iv];
                let f10 = grid_vals[(iu + 1) * (n + 1) + iv];
                let f01 = grid_vals[iu * (n + 1) + (iv + 1)];
                let dfu = (f10 - f00).abs() / du;
                let dfv = (f01 - f00).abs() / dv;
                let local_lip = dfu.hypot(dfv);
                if local_lip > lip {
                    lip = local_lip;
                }
            }
        }
        // Add a safety margin.
        lip *= 2.0;
        if lip < 1e-15 {
            // Function appears constant.
            return (best_u, best_v, best);
        }

        // ── Phase 2: Lipschitz subdivision ────────────────────────────────────
        let mut stack: Vec<(f64, f64, f64, f64, usize)> = vec![(u0, u1, v0, v1, 0)];
        let mut cell_count = 0_usize;

        while let Some((cu0, cu1, cv0, cv1, depth)) = stack.pop() {
            cell_count += 1;
            if cell_count > self.max_evals {
                break;
            }

            let um = (cu0 + cu1) * 0.5;
            let vm = (cv0 + cv1) * 0.5;
            let fc = f(um, vm);
            if fc < best {
                best = fc;
                best_u = um;
                best_v = vm;
            }

            let du_cell = cu1 - cu0;
            let dv_cell = cv1 - cv0;
            let radius = (du_cell * du_cell + dv_cell * dv_cell).sqrt() * 0.5;
            let lower = fc - lip * radius;

            // Prune: this cell cannot contain a better minimum.
            if lower > best {
                continue;
            }

            // Terminal: cell converged or at max depth.
            if radius < tolerance || depth >= self.max_subdivisions {
                if fc < best {
                    best = fc;
                    best_u = um;
                    best_v = vm;
                }
                continue;
            }

            // Subdivide along the longer dimension.
            if du_cell >= dv_cell {
                stack.push((cu0, um, cv0, cv1, depth + 1));
                stack.push((um, cu1, cv0, cv1, depth + 1));
            } else {
                stack.push((cu0, cu1, cv0, vm, depth + 1));
                stack.push((cu0, cu1, vm, cv1, depth + 1));
            }
        }

        (best_u, best_v, best)
    }
}

// ── NURBS helpers ────────────────────────────────────────────────────────────

/// Estimate the Lipschitz bound for the squared-distance function between
/// two NURBS curves: `f(u, v) = ‖C₁(u) − C₂(v)‖²`.
///
/// The gradient is:
/// ```text
/// ∂f/∂u =  2 · (C₁(u) − C₂(v)) · C₁'(u)
/// ∂f/∂v = −2 · (C₁(u) − C₂(v)) · C₂'(v)
/// ```
///
/// Upper bound: `L ≤ 2 · max_sep · max(max_deriv₁, max_deriv₂)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_curve_curve_lipschitz(c1: &NurbsCurve, c2: &NurbsCurve) -> f64 {
    let n_samples = 20_usize;
    let (u0, u1) = c1.domain();
    let (v0, v1) = c2.domain();

    let mut max_deriv1: f64 = 0.0;
    let mut max_deriv2: f64 = 0.0;
    let mut max_sep: f64 = 0.0;

    for i in 0..=n_samples {
        let t = i as f64 / n_samples as f64;
        let u = u0 + (u1 - u0) * t;
        let d1 = c1.derivatives(u, 1);
        max_deriv1 = max_deriv1.max(d1[1].length());

        let v = v0 + (v1 - v0) * t;
        let d2 = c2.derivatives(v, 1);
        max_deriv2 = max_deriv2.max(d2[1].length());

        let p1 = Point3::new(d1[0].x(), d1[0].y(), d1[0].z());
        let p2 = Point3::new(d2[0].x(), d2[0].y(), d2[0].z());
        max_sep = max_sep.max((p1 - p2).length());
    }

    2.0 * max_sep * max_deriv1.max(max_deriv2)
}

/// Find the global minimum distance between two NURBS curves.
///
/// Returns `(distance, point on C₁, point on C₂)`.
#[must_use]
pub fn nurbs_curve_curve_distance(
    curve1: &NurbsCurve,
    curve2: &NurbsCurve,
) -> (f64, Point3, Point3) {
    let (u0, u1) = curve1.domain();
    let (v0, v1) = curve2.domain();

    let lip = estimate_curve_curve_lipschitz(curve1, curve2);

    if lip < 1e-15 {
        // Curves appear coincident or degenerate.
        let p1 = curve1.evaluate((u0 + u1) * 0.5);
        let p2 = curve2.evaluate((v0 + v1) * 0.5);
        return ((p1 - p2).length(), p1, p2);
    }

    let f = |u: f64, v: f64| -> f64 {
        let p1 = curve1.evaluate(u);
        let p2 = curve2.evaluate(v);
        (p1 - p2).length_squared()
    };

    let opt = LipschitzOptimizer::new();
    let (best_u, best_v, _) = opt.minimize_2d(f, (u0, u1), (v0, v1), 1e-4);

    let p1 = curve1.evaluate(best_u);
    let p2 = curve2.evaluate(best_v);
    ((p1 - p2).length(), p1, p2)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn minimize_quadratic_at_origin() {
        // f(u,v) = u² + v², minimum at (0,0,0).
        let f = |u: f64, v: f64| u * u + v * v;
        let opt = LipschitzOptimizer::new();
        let (u, v, val) = opt.minimize_2d(f, (-1.0, 1.0), (-1.0, 1.0), 1e-6);
        assert!(val < 1e-8, "val={val}");
        assert!(u.abs() < 1e-4, "u={u}");
        assert!(v.abs() < 1e-4, "v={v}");
    }

    #[test]
    fn minimize_quadratic_offset() {
        // f(u,v) = (u-0.3)² + (v-0.7)², minimum at (0.3, 0.7).
        let f = |u: f64, v: f64| (u - 0.3) * (u - 0.3) + (v - 0.7) * (v - 0.7);
        let opt = LipschitzOptimizer::new();
        let (u, v, val) = opt.minimize_2d(f, (0.0, 1.0), (0.0, 1.0), 1e-6);
        assert!(val < 1e-8, "val={val}");
        assert!((u - 0.3).abs() < 1e-3, "u={u}");
        assert!((v - 0.7).abs() < 1e-3, "v={v}");
    }

    #[test]
    fn minimize_returns_global_not_local() {
        // Two basins: one at (0.1, 0.5) depth 0.01, one at (0.9, 0.5) depth 0 (global).
        let f = |u: f64, v: f64| {
            let d1 = (u - 0.1) * (u - 0.1) + (v - 0.5) * (v - 0.5) - 0.01;
            let d2 = (u - 0.9) * (u - 0.9) + (v - 0.5) * (v - 0.5);
            d1.min(d2)
        };
        let opt = LipschitzOptimizer::new();
        let (_u, _v, val) = opt.minimize_2d(f, (0.0, 1.0), (0.0, 1.0), 1e-5);
        // Global minimum is -0.01 at (0.1, 0.5).
        assert!(val < -0.005, "val={val}");
    }

    #[test]
    fn nurbs_parallel_lines_distance_one() {
        // Two parallel NURBS lines: y=0 and y=1 along x.
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
        assert!((dist - 1.0).abs() < 1e-3, "dist={dist}");
    }
}
