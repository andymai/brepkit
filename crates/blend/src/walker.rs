//! Newton-Raphson walking engine.
//!
//! Traces the blend surface along a spine by solving a 4×4 constraint system
//! at each step using Newton-Raphson iteration with adaptive step control.
//!
//! The walker produces a sequence of [`CircSection`]s that can be assembled
//! into a NURBS surface via [`approximate_blend_surface`].

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use crate::BlendError;
use crate::blend_func::{BlendContext, BlendFunction, BlendParams};
use crate::section::CircSection;
use crate::spine::Spine;

// ──────────────────────────── 4×4 linear solver ────────────────────────────

/// Solve a 4×4 linear system `Ax = b` using Gaussian elimination with partial
/// pivoting.
///
/// Returns `None` if the matrix is singular (pivot below `1e-30`).
#[must_use]
pub fn solve_4x4(a: &[[f64; 4]; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    // Copy into augmented matrix.
    let mut m = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j];
        }
        m[i][4] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..4 {
        // Find pivot.
        let mut max_abs = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..4 {
            let abs_val = m[row][col].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_row = row;
            }
        }

        if max_abs < 1e-30 {
            return None;
        }

        // Swap rows.
        if max_row != col {
            m.swap(col, max_row);
        }

        // Eliminate below.
        let pivot = m[col][col];
        for row in (col + 1)..4 {
            let factor = m[row][col] / pivot;
            for j in col..5 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution.
    let mut x = [0.0_f64; 4];
    for i in (0..4).rev() {
        let mut sum = m[i][4];
        for j in (i + 1)..4 {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }

    Some(x)
}

// ──────────────────────────── WalkerConfig ────────────────────────────

/// Configuration for the walking engine.
#[derive(Debug, Clone, Copy)]
pub struct WalkerConfig {
    /// Convergence tolerance in 3D space.
    pub tol_3d: f64,
    /// Maximum Newton iterations per step.
    pub max_newton_iters: usize,
    /// Maximum step size as a fraction of total spine length.
    pub max_step_fraction: f64,
    /// Minimum step size before declaring failure.
    pub min_step: f64,
    /// Maximum number of walking steps.
    pub max_steps: usize,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            tol_3d: 1e-7,
            max_newton_iters: 20,
            max_step_fraction: 0.05,
            min_step: 1e-10,
            max_steps: 1000,
        }
    }
}

// ──────────────────────────── WalkResult ────────────────────────────

/// Result of a successful walk along the spine.
#[derive(Debug)]
pub struct WalkResult {
    /// Cross-sections collected during the walk.
    pub sections: Vec<CircSection>,
    /// Surface parameters at the walk endpoint.
    pub end_params: BlendParams,
}

// ──────────────────────────── Walker ────────────────────────────

/// Newton-Raphson walking engine that traces a blend surface along a spine.
///
/// At each step the walker solves a 4×4 nonlinear system to find surface
/// parameters `(u1, v1, u2, v2)` satisfying the blend constraints, then
/// advances along the spine with adaptive step control.
pub struct Walker<'a, F: BlendFunction> {
    /// The blend constraint function to solve.
    func: &'a F,
    /// First support surface.
    surf1: &'a dyn ParametricSurface,
    /// Second support surface.
    surf2: &'a dyn ParametricSurface,
    /// Spine (edge chain) to walk along.
    spine: &'a Spine,
    /// Topology arena for spine evaluation.
    topo: &'a Topology,
    /// Walker configuration.
    config: WalkerConfig,
}

impl<'a, F: BlendFunction> Walker<'a, F> {
    /// Create a new walker.
    #[must_use]
    pub fn new(
        func: &'a F,
        surf1: &'a dyn ParametricSurface,
        surf2: &'a dyn ParametricSurface,
        spine: &'a Spine,
        topo: &'a Topology,
        config: WalkerConfig,
    ) -> Self {
        Self {
            func,
            surf1,
            surf2,
            spine,
            topo,
            config,
        }
    }

    /// Build a [`BlendContext`] for the given spine parameter.
    fn make_context(&self, s: f64) -> Result<BlendContext, BlendError> {
        let guide_point = self.spine.evaluate(self.topo, s)?;
        let nplan = self.spine.tangent(self.topo, s)?;
        let t = if self.spine.length() > f64::EPSILON {
            s / self.spine.length()
        } else {
            0.0
        };
        Ok(BlendContext {
            guide_point,
            nplan,
            t,
        })
    }

    /// Compute the L2 norm of a 4-element residual vector.
    fn residual_norm(f: &[f64; 4]) -> f64 {
        (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt()
    }

    /// Run Newton-Raphson iteration to convergence from an initial guess.
    ///
    /// Returns the converged parameters, or `None` if convergence fails.
    fn newton_solve(&self, initial: BlendParams, ctx: &BlendContext) -> Option<BlendParams> {
        let mut params = initial;

        for _iter in 0..self.config.max_newton_iters {
            let f = self.func.value(self.surf1, self.surf2, &params, ctx);
            if Self::residual_norm(&f) < self.config.tol_3d {
                return Some(params);
            }

            let j = self.func.jacobian(self.surf1, self.surf2, &params, ctx);
            let neg_f = [-f[0], -f[1], -f[2], -f[3]];
            let delta = solve_4x4(&j, &neg_f)?;

            params.u1 += delta[0];
            params.v1 += delta[1];
            params.u2 += delta[2];
            params.v2 += delta[3];
        }

        // Check final residual.
        let f = self.func.value(self.surf1, self.surf2, &params, ctx);
        if Self::residual_norm(&f) < self.config.tol_3d {
            Some(params)
        } else {
            None
        }
    }

    /// Find initial blend parameters at spine parameter `s0`.
    ///
    /// Projects the guide point onto both surfaces to form an initial guess,
    /// then refines with Newton-Raphson.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError::StartSolutionFailure`] if Newton iteration fails
    /// to converge.
    pub fn find_start(&self, s0: f64) -> Result<BlendParams, BlendError> {
        let ctx = self.make_context(s0)?;

        // Initial guess: project guide point onto both surfaces.
        let (u1, v1) = self.surf1.project_point(ctx.guide_point);
        let (u2, v2) = self.surf2.project_point(ctx.guide_point);

        let initial = BlendParams { u1, v1, u2, v2 };
        self.newton_solve(initial, &ctx)
            .ok_or_else(|| BlendError::StartSolutionFailure {
                edge: self.spine.edges()[0],
                t: ctx.t,
            })
    }

    /// Walk the blend along the spine from `s_start` to `s_end`.
    ///
    /// Uses adaptive step control: starts with a large step, halves on Newton
    /// failure, increases by 1.5× on success.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError::WalkingFailure`] if the step size shrinks below
    /// `min_step` or the step count exceeds `max_steps`.
    #[allow(clippy::too_many_lines)]
    pub fn walk(
        &self,
        start_params: BlendParams,
        s_start: f64,
        s_end: f64,
    ) -> Result<WalkResult, BlendError> {
        let span = (s_end - s_start).abs();
        let direction = if s_end >= s_start { 1.0 } else { -1.0 };
        let mut step = self.config.max_step_fraction * span;
        let mut s = s_start;
        let mut params = start_params;
        let mut sections = Vec::new();
        let mut step_count = 0_usize;

        // Collect the starting section.
        let ctx0 = self.make_context(s)?;
        let sec0 = self.func.section(self.surf1, self.surf2, &params, &ctx0);
        sections.push(sec0);

        // Previous params for linear extrapolation (predictor).
        let mut prev_params: Option<BlendParams> = None;
        let mut prev_s = s;

        #[allow(clippy::while_float)]
        while (s - s_end).abs() > self.config.min_step {
            step_count += 1;
            if step_count > self.config.max_steps {
                let ctx = self.make_context(s)?;
                let f = self.func.value(self.surf1, self.surf2, &params, &ctx);
                return Err(BlendError::WalkingFailure {
                    edge: self.spine.edges()[0],
                    t: ctx.t,
                    residual: Self::residual_norm(&f),
                });
            }

            // Clamp step to not overshoot the end.
            let clamped_step = step.min((s_end - s).abs());
            let s_next = s + direction * clamped_step;

            // Predictor: linear extrapolation from last two solutions.
            let predicted = if let Some(prev) = prev_params {
                let ds_old = s - prev_s;
                if ds_old.abs() > f64::EPSILON {
                    let ds_new = s_next - s;
                    let ratio = ds_new / ds_old;
                    BlendParams {
                        u1: params.u1 + ratio * (params.u1 - prev.u1),
                        v1: params.v1 + ratio * (params.v1 - prev.v1),
                        u2: params.u2 + ratio * (params.u2 - prev.u2),
                        v2: params.v2 + ratio * (params.v2 - prev.v2),
                    }
                } else {
                    params
                }
            } else {
                params
            };

            // Corrector: Newton at the new spine station.
            let ctx_next = self.make_context(s_next)?;
            if let Some(converged) = self.newton_solve(predicted, &ctx_next) {
                // Accept step.
                prev_params = Some(params);
                prev_s = s;
                params = converged;
                s = s_next;

                let sec = self
                    .func
                    .section(self.surf1, self.surf2, &params, &ctx_next);
                sections.push(sec);

                // Increase step (but don't exceed max).
                step = (step * 1.5).min(self.config.max_step_fraction * span);
            } else {
                // Halve step and retry.
                step *= 0.5;
                if step < self.config.min_step {
                    let f = self.func.value(self.surf1, self.surf2, &params, &ctx_next);
                    return Err(BlendError::WalkingFailure {
                        edge: self.spine.edges()[0],
                        t: ctx_next.t,
                        residual: Self::residual_norm(&f),
                    });
                }
            }
        }

        Ok(WalkResult {
            sections,
            end_params: params,
        })
    }
}

// ──────────────────────── NURBS surface approximation ──────────────────────

/// Build a NURBS surface from a sequence of circular-arc cross-sections.
///
/// Each section becomes a rational quadratic arc in the U direction (degree 2,
/// 3 control points). The V direction interpolates through the sections with
/// degree `min(n-1, 3)` using a uniform knot vector.
///
/// # Errors
///
/// Returns [`BlendError::Math`] if the NURBS surface construction fails
/// (e.g., too few sections).
pub fn approximate_blend_surface(sections: &[CircSection]) -> Result<NurbsSurface, BlendError> {
    let n = sections.len();
    if n < 2 {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }

    // U direction: rational quadratic arc (degree 2, 3 control points).
    let degree_u = 2;
    let knots_u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // V direction: degree = min(n-1, 3).
    let degree_v = (n - 1).min(3);
    let knots_v = build_uniform_knots(n, degree_v);

    // Build control point grid: n rows (V) × 3 columns (U).
    let mut control_points = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for sec in sections {
        let half_angle = sec.half_angle();
        let w_mid = half_angle.cos();

        // cp0 = p1, cp2 = p2.
        let cp0 = sec.p1;
        let cp2 = sec.p2;

        // cp1 = apex of the rational quadratic on the arc.
        // The weighted midpoint lies at center + radius * mid_direction,
        // but for a rational Bezier the control point is the intersection
        // of tangent lines at cp0 and cp2. For a circular arc this is:
        // cp1 = (cp0 + cp2) / 2 + (1/cos(half_angle) - 1) * (center - midpoint)
        // Equivalently: cp1 = center + (mid_dir / cos(half_angle)) * radius
        // where mid_dir bisects the two contact directions.
        let midpoint = Point3::new(
            (cp0.x() + cp2.x()) * 0.5,
            (cp0.y() + cp2.y()) * 0.5,
            (cp0.z() + cp2.z()) * 0.5,
        );

        let cp1 = if w_mid.abs() > 1e-15 {
            // Tangent intersection point for the rational quadratic.
            let center_to_mid = Vec3::new(
                midpoint.x() - sec.center.x(),
                midpoint.y() - sec.center.y(),
                midpoint.z() - sec.center.z(),
            );
            let scale = 1.0 / w_mid;
            Point3::new(
                sec.center.x() + center_to_mid.x() * scale,
                sec.center.y() + center_to_mid.y() * scale,
                sec.center.z() + center_to_mid.z() * scale,
            )
        } else {
            midpoint
        };

        control_points.push(vec![cp0, cp1, cp2]);
        weights.push(vec![1.0, w_mid, 1.0]);
    }

    let surf = NurbsSurface::new(
        degree_u,
        degree_v,
        knots_u,
        knots_v,
        control_points,
        weights,
    )?;

    Ok(surf)
}

/// Build a uniform clamped knot vector for `n` control points and `degree`.
fn build_uniform_knots(n: usize, degree: usize) -> Vec<f64> {
    // Clamped knot vector: degree+1 zeros, internal knots, degree+1 ones.
    let num_knots = n + degree + 1;
    let mut knots = Vec::with_capacity(num_knots);

    knots.extend(std::iter::repeat_n(0.0, degree + 1));

    let num_internal = num_knots.saturating_sub(2 * (degree + 1));
    for i in 1..=num_internal {
        #[allow(clippy::cast_precision_loss)]
        let val = i as f64 / (num_internal + 1) as f64;
        knots.push(val);
    }

    knots.extend(std::iter::repeat_n(1.0, degree + 1));

    knots
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::blend_func::ConstRadBlend;
    use brepkit_math::traits::ParametricSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;

    // ── Test plane surface ──

    struct TestPlane {
        origin: Point3,
        u_dir: Vec3,
        v_dir: Vec3,
        norm: Vec3,
    }

    impl ParametricSurface for TestPlane {
        fn evaluate(&self, u: f64, v: f64) -> Point3 {
            Point3::new(
                self.origin.x() + u * self.u_dir.x() + v * self.v_dir.x(),
                self.origin.y() + u * self.u_dir.y() + v * self.v_dir.y(),
                self.origin.z() + u * self.u_dir.z() + v * self.v_dir.z(),
            )
        }

        fn normal(&self, _u: f64, _v: f64) -> Vec3 {
            self.norm
        }

        fn project_point(&self, point: Point3) -> (f64, f64) {
            let d = point - self.origin;
            let dv = Vec3::new(d.x(), d.y(), d.z());
            (dv.dot(self.u_dir), dv.dot(self.v_dir))
        }

        fn partial_u(&self, _u: f64, _v: f64) -> Vec3 {
            self.u_dir
        }

        fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
            self.v_dir
        }
    }

    fn make_perpendicular_planes() -> (TestPlane, TestPlane) {
        // XY plane (z=0, normal +Z) and XZ plane (y=0, normal +Y)
        let plane1 = TestPlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            u_dir: Vec3::new(1.0, 0.0, 0.0),
            v_dir: Vec3::new(0.0, 1.0, 0.0),
            norm: Vec3::new(0.0, 0.0, 1.0),
        };
        let plane2 = TestPlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            u_dir: Vec3::new(0.0, 1.0, 0.0),
            v_dir: Vec3::new(0.0, 0.0, 1.0),
            norm: Vec3::new(1.0, 0.0, 0.0),
        };
        (plane1, plane2)
    }

    fn make_line_edge(topo: &mut Topology, a: Point3, b: Point3) -> brepkit_topology::edge::EdgeId {
        let v0 = topo.add_vertex(Vertex::new(a, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(b, 1e-7));
        topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line))
    }

    // ── solve_4x4 tests ──

    #[test]
    fn solve_4x4_identity() {
        let a = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = solve_4x4(&a, &b).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - b[i]).abs() < 1e-12,
                "x[{i}] = {}, expected {}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn solve_4x4_known_system() {
        // A * [1, 2, 3, 4]^T = b
        let a = [
            [2.0, 1.0, -1.0, 3.0],
            [1.0, 3.0, 2.0, -1.0],
            [-1.0, 2.0, 4.0, 1.0],
            [3.0, -1.0, 1.0, 2.0],
        ];
        let x_exact = [1.0, 2.0, 3.0, 4.0];
        let mut b = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                b[i] += a[i][j] * x_exact[j];
            }
        }
        let x = solve_4x4(&a, &b).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-10,
                "x[{i}] = {}, expected {}",
                x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn solve_4x4_singular_returns_none() {
        // Row 2 = Row 0, so matrix is singular.
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let b = [1.0, 1.0, 1.0, 1.0];
        assert!(solve_4x4(&a, &b).is_none());
    }

    #[test]
    fn find_start_converges_for_perpendicular_planes() {
        let (p1, p2) = make_perpendicular_planes();
        let mut topo = Topology::new();
        // Edge along Y axis (the intersection line of the two planes).
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let blend = ConstRadBlend { radius: 1.0 };
        let config = WalkerConfig::default();

        let walker = Walker::new(&blend, &p1, &p2, &spine, &topo, config);
        let params = walker.find_start(5.0).unwrap();

        // Verify: the solution points should lie on the two planes.
        let pt1 = p1.evaluate(params.u1, params.v1);
        let pt2 = p2.evaluate(params.u2, params.v2);

        // Both points should be at distance `radius` from the ball center.
        let ctx = walker.make_context(5.0).unwrap();
        let sec = blend.section(&p1, &p2, &params, &ctx);
        let d1 = (pt1 - sec.center).length();
        let d2 = (pt2 - sec.center).length();
        assert!((d1 - 1.0).abs() < 1e-5, "d1 = {d1}, expected 1.0");
        assert!((d2 - 1.0).abs() < 1e-5, "d2 = {d2}, expected 1.0");
    }

    #[test]
    fn walk_straight_edge_uniform_sections() {
        let (p1, p2) = make_perpendicular_planes();
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let blend = ConstRadBlend { radius: 1.0 };
        let config = WalkerConfig::default();

        let walker = Walker::new(&blend, &p1, &p2, &spine, &topo, config);
        let start = walker.find_start(0.0).unwrap();
        let result = walker.walk(start, 0.0, 10.0).unwrap();

        // Should have at least a few sections.
        assert!(
            result.sections.len() >= 3,
            "expected >=3 sections, got {}",
            result.sections.len()
        );

        // All sections should have the same radius (constant-radius blend).
        for sec in &result.sections {
            assert!(
                (sec.radius - 1.0).abs() < 0.1,
                "section radius = {}, expected ~1.0",
                sec.radius
            );
        }

        // Sections should span the spine (first near y=0, last near y=10).
        let first_y = result.sections[0].p1.y();
        let last_y = result.sections.last().unwrap().p1.y();
        assert!(first_y < 1.0, "first section y = {first_y}");
        assert!(last_y > 9.0, "last section y = {last_y}");
    }

    #[test]
    fn walk_respects_max_steps() {
        let (p1, p2) = make_perpendicular_planes();
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let blend = ConstRadBlend { radius: 1.0 };

        // Tiny max_step_fraction to force many steps, but cap at 5.
        let config = WalkerConfig {
            max_steps: 5,
            max_step_fraction: 0.001,
            ..WalkerConfig::default()
        };

        let walker = Walker::new(&blend, &p1, &p2, &spine, &topo, config);
        let start = walker.find_start(0.0).unwrap();
        let result = walker.walk(start, 0.0, 10.0);

        // Should fail with WalkingFailure due to max_steps exceeded.
        assert!(result.is_err(), "expected WalkingFailure");
        match result.unwrap_err() {
            BlendError::WalkingFailure { .. } => {} // expected
            other => panic!("expected WalkingFailure, got {other:?}"),
        }
    }
}
