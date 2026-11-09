// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Newton-Raphson walking engine.
//!
//! Traces the blend surface along a spine by solving a 4×4 constraint system
//! at each step using Newton-Raphson iteration with adaptive step control.
//!
//! The walker produces a sequence of [`CircSection`]s.
//! [`interpolate_blend_surface`] makes them a NURBS band that passes through
//! each section; [`approximate_blend_surface`] makes one that takes them as
//! its control points.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::Point3;
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
    let mut m = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j];
        }
        m[i][4] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..4 {
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

        if max_row != col {
            m.swap(col, max_row);
        }

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
    /// Largest turn of the spine's direction across one step, in radians:
    /// the band is fitted through the sections, so round a tight bend a
    /// step of a fixed share of the spine left it far off the true round.
    pub max_turn: f64,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            tol_3d: 1e-7,
            max_newton_iters: 20,
            max_step_fraction: 0.05,
            min_step: 1e-10,
            max_steps: 1000,
            max_turn: 0.025,
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

    /// `params` with each surface's parameters moved by whole turns into
    /// the range of a surface that closes on itself (a B-spline wall round
    /// a closed outline). Off that range a B-spline surface is extrapolated,
    /// so a ball rolling across the seam walked off the wall.
    fn wrapped(&self, params: BlendParams) -> BlendParams {
        let (u1, v1) = wrap_params(self.surf1, params.u1, params.v1);
        let (u2, v2) = wrap_params(self.surf2, params.u2, params.v2);
        BlendParams { u1, v1, u2, v2 }
    }

    /// `to - from` on each surface, the short way round a closed one.
    fn step_between(&self, from: BlendParams, to: BlendParams) -> BlendParams {
        let (u1, v1) = (
            short_way(self.surf1.closed_u(), to.u1 - from.u1),
            short_way(self.surf1.closed_v(), to.v1 - from.v1),
        );
        let (u2, v2) = (
            short_way(self.surf2.closed_u(), to.u2 - from.u2),
            short_way(self.surf2.closed_v(), to.v2 - from.v2),
        );
        BlendParams { u1, v1, u2, v2 }
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
            params = self.wrapped(params);
        }

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

    /// Solve the blend at spine station `s` and return both the converged
    /// parameters and the resulting cross-section.
    ///
    /// `guess` seeds Newton from a previous station's solution (continuation),
    /// falling back to a fresh guide-point projection when `None`. Returns
    /// `None` if Newton fails to converge.
    pub(crate) fn solve_section(
        &self,
        s: f64,
        guess: Option<BlendParams>,
    ) -> Option<(BlendParams, CircSection)> {
        let ctx = self.make_context(s).ok()?;
        let initial = guess.unwrap_or_else(|| {
            let (u1, v1) = self.surf1.project_point(ctx.guide_point);
            let (u2, v2) = self.surf2.project_point(ctx.guide_point);
            BlendParams { u1, v1, u2, v2 }
        });
        let params = self.newton_solve(initial, &ctx)?;
        let sec = self.func.section(self.surf1, self.surf2, &params, &ctx);
        Some((params, sec))
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

        let ctx0 = self.make_context(s)?;
        let sec0 = self.func.section(self.surf1, self.surf2, &params, &ctx0);
        sections.push(sec0);
        let mut nplan = ctx0.nplan;

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
                    let d = self.step_between(prev, params);
                    self.wrapped(BlendParams {
                        u1: params.u1 + ratio * d.u1,
                        v1: params.v1 + ratio * d.v1,
                        u2: params.u2 + ratio * d.u2,
                        v2: params.v2 + ratio * d.v2,
                    })
                } else {
                    params
                }
            } else {
                params
            };

            // Corrector: Newton at the new spine station.
            let ctx_next = self.make_context(s_next)?;
            let turns_too_far = |ctx: &BlendContext| {
                clamped_step > MIN_TURN_STEP * span
                    && nplan.dot(ctx.nplan) < self.config.max_turn.cos()
            };
            if turns_too_far(&ctx_next) {
                // Not a step: the halving stops at its floor, so these
                // retries are bounded without drawing on `max_steps`, which
                // a long curved spine would otherwise run out of.
                step = 0.5 * clamped_step;
                step_count -= 1;
                continue;
            }
            if let Some(converged) = self.newton_solve(predicted, &ctx_next) {
                nplan = ctx_next.nplan;
                prev_params = Some(params);
                prev_s = s;
                params = converged;
                s = s_next;

                let sec = self
                    .func
                    .section(self.surf1, self.surf2, &params, &ctx_next);
                sections.push(sec);

                step = (step * 1.5).min(self.config.max_step_fraction * span);
            } else {
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

/// The shortest step, as a share of the spine, that a turn of the spine's
/// direction can refuse (a kink in a G1 chain read from a file must not
/// halve the step for ever).
const MIN_TURN_STEP: f64 = 1e-3;

/// `(u, v)` moved by whole turns into the range of each direction in which
/// `surface` closes on itself.
fn wrap_params(surface: &dyn ParametricSurface, u: f64, v: f64) -> (f64, f64) {
    let wrap = |x: f64, range: Option<(f64, f64)>| match range {
        Some((a, b)) if b - a > f64::EPSILON => a + (x - a).rem_euclid(b - a),
        _ => x,
    };
    (wrap(u, surface.closed_u()), wrap(v, surface.closed_v()))
}

/// A parameter step `d` the short way round a direction that closes over
/// `range`, unchanged in an open one.
fn short_way(range: Option<(f64, f64)>, d: f64) -> f64 {
    match range {
        Some((a, b)) if b - a > f64::EPSILON => d - (d / (b - a)).round() * (b - a),
        _ => d,
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

    // Build the control-point grid indexed `[row_u][col_v]`: U is the rational
    // quadratic arc (3 rows — start / apex / end), V interpolates through the
    // `n` sections (n columns). Each section contributes one column entry to
    // each arc row. Laying the grid out section-major would transpose U and V,
    // making `knots_u` (length 6) be validated against the section count — the
    // source of `InvalidKnotVector { expected: n+3, got: 6 }` for n != 3.
    let mut arc_start = Vec::with_capacity(n); // cp0 per section
    let mut arc_apex = Vec::with_capacity(n); // cp1 per section
    let mut arc_end = Vec::with_capacity(n); // cp2 per section
    let mut w_start = Vec::with_capacity(n);
    let mut w_apex = Vec::with_capacity(n);
    let mut w_end = Vec::with_capacity(n);

    for sec in sections {
        let ([cp0, cp1, cp2], w_mid) = section_arc(sec);
        arc_start.push(cp0);
        arc_apex.push(cp1);
        arc_end.push(cp2);
        w_start.push(1.0);
        w_apex.push(w_mid);
        w_end.push(1.0);
    }

    let control_points = vec![arc_start, arc_apex, arc_end];
    let weights = vec![w_start, w_apex, w_end];

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

/// A section's arc as a rational quadratic: its three control points and
/// the apex's weight.
fn section_arc(sec: &CircSection) -> ([Point3; 3], f64) {
    let half_angle = sec.half_angle();
    let w_mid = half_angle.cos();
    let cp0 = sec.p1;
    let cp2 = sec.p2;
    // cp1 = apex (middle control point) of the rational quadratic arc.
    // For the standard circular-arc representation (apex weight = cos θ),
    // the apex is the tangent intersection at distance r/cos θ from the
    // center along the chord-midpoint bisector. The chord midpoint M sits
    // at distance r·cos θ from the center, so
    //   apex = center + (M − center) / cos²θ = center + (M − center) / w_mid²
    // Using 1/w_mid (instead of 1/w_mid²) places the apex on the arc and
    // yields a non-circular conic.
    let midpoint = Point3::new(
        (cp0.x() + cp2.x()) * 0.5,
        (cp0.y() + cp2.y()) * 0.5,
        (cp0.z() + cp2.z()) * 0.5,
    );
    let cp1 = if w_mid.abs() > 1e-15 {
        sec.center + (midpoint - sec.center) * (1.0 / (w_mid * w_mid))
    } else {
        midpoint
    };
    ([cp0, cp1, cp2], w_mid)
}

/// Build a blend band that passes through every one of its sections.
///
/// As [`approximate_blend_surface`], each section is a rational quadratic
/// arc across the band, but along it the band interpolates the sections'
/// arcs (in homogeneous coordinates, each parameter step the farthest the
/// centre or a contact moves, with averaged knots) rather than taking them
/// as control points, so a section inside the band lies on it, contact
/// points included.
///
/// # Errors
///
/// Returns [`BlendError::Math`] with fewer than two sections, when the
/// sections all lie at one place, when the collocation system is singular
/// or not finite (a run of sections at one place), or when a solved weight
/// is not positive.
pub fn interpolate_blend_surface(sections: &[CircSection]) -> Result<NurbsSurface, BlendError> {
    let n = sections.len();
    if n < 2 {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }
    let degree_v = (n - 1).min(3);
    // Chord-length parameters, each step the farthest the centre or a
    // contact moves (a ball pivoting on its centre, rounding a round's rim
    // at the round's own radius, keeps its centre still).
    let mut params = Vec::with_capacity(n);
    params.push(0.0);
    for pair in sections.windows(2) {
        let step = (pair[1].center - pair[0].center)
            .length()
            .max((pair[1].p1 - pair[0].p1).length())
            .max((pair[1].p2 - pair[0].p2).length());
        params.push(params.last().copied().unwrap_or(0.0) + step);
    }
    let total = params.last().copied().unwrap_or(0.0);
    if total <= 0.0 {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }
    for param in &mut params {
        *param /= total;
    }
    if let Some(last) = params.last_mut() {
        *last = 1.0;
    }
    // Averaged knots (The NURBS Book, eq. 9.8).
    let mut knots_v = vec![0.0; degree_v + 1];
    for j in 1..n - degree_v {
        #[allow(clippy::cast_precision_loss)]
        let knot = params[j..j + degree_v].iter().sum::<f64>() / degree_v as f64;
        knots_v.push(knot);
    }
    knots_v.extend(std::iter::repeat_n(1.0, degree_v + 1));
    // The collocation matrix, banded: Gaussian elimination without pivoting
    // is stable on it (it is totally positive).
    let mut matrix = vec![vec![0.0; n]; n];
    for (row, &param) in matrix.iter_mut().zip(&params) {
        let span = brepkit_math::nurbs::basis::find_span(n, degree_v, param, &knots_v);
        let basis = brepkit_math::nurbs::basis::basis_funs(span, param, degree_v, &knots_v);
        for (k, value) in basis.into_iter().enumerate() {
            if let Some(entry) = row.get_mut(span - degree_v + k) {
                *entry = value;
            }
        }
    }
    // Twelve right-hand sides: three arc rows of homogeneous points.
    let mut rhs: Vec<[f64; 12]> = sections
        .iter()
        .map(|sec| {
            let (points, w_mid) = section_arc(sec);
            let weights = [1.0, w_mid, 1.0];
            let mut out = [0.0; 12];
            for (k, (point, weight)) in points.iter().zip(weights).enumerate() {
                out[4 * k] = point.x() * weight;
                out[4 * k + 1] = point.y() * weight;
                out[4 * k + 2] = point.z() * weight;
                out[4 * k + 3] = weight;
            }
            out
        })
        .collect();
    let band = degree_v + 1;
    for k in 0..n {
        let pivot = matrix[k][k];
        if !pivot.is_finite() || pivot.abs() < 1e-14 {
            return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
        }
        for i in k + 1..n.min(k + band) {
            let factor = matrix[i][k] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in k..n.min(k + band) {
                matrix[i][j] -= factor * matrix[k][j];
            }
            let source = rhs[k];
            for (value, &from) in rhs[i].iter_mut().zip(&source) {
                *value -= factor * from;
            }
        }
    }
    for k in (0..n).rev() {
        let mut value = rhs[k];
        for j in k + 1..n.min(k + band) {
            let factor = matrix[k][j];
            for (out, &from) in value.iter_mut().zip(&rhs[j]) {
                *out -= factor * from;
            }
        }
        let pivot = matrix[k][k];
        for out in &mut value {
            *out /= pivot;
        }
        rhs[k] = value;
    }
    if rhs.iter().flatten().any(|value| !value.is_finite()) {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }
    let mut control_points = (0..3).map(|_| Vec::with_capacity(n)).collect::<Vec<_>>();
    let mut weights = (0..3).map(|_| Vec::with_capacity(n)).collect::<Vec<_>>();
    for solved in &rhs {
        for k in 0..3 {
            let weight = solved[4 * k + 3];
            if weight <= 1e-9 {
                return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
            }
            control_points[k].push(Point3::new(
                solved[4 * k] / weight,
                solved[4 * k + 1] / weight,
                solved[4 * k + 2] / weight,
            ));
            weights[k].push(weight);
        }
    }
    Ok(NurbsSurface::new(
        2,
        degree_v,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        knots_v,
        control_points,
        weights,
    )?)
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

    /// Build `n` identical quarter-circle sections of a radius-1 fillet between
    /// the planes `z = 0` and `y = 0`, swept along +X.
    fn quarter_circle_sections(n: usize) -> Vec<CircSection> {
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f64;
                CircSection {
                    p1: Point3::new(x, 1.0, 0.0),
                    p2: Point3::new(x, 0.0, 1.0),
                    center: Point3::new(x, 1.0, 1.0),
                    radius: 1.0,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: x,
                }
            })
            .collect()
    }

    /// Assert that the arc cross-section (U direction) of `surf` at parameter
    /// `v` is an exact radius-1 circular arc from `p1` to `p2` about `center`.
    fn assert_circular_arc(surf: &NurbsSurface, v: f64, sec: &CircSection) {
        let start = surf.evaluate(0.0, v);
        let end = surf.evaluate(1.0, v);
        assert!(
            (start - sec.p1).length() < 1e-9,
            "arc start {start:?} != p1 {:?}",
            sec.p1
        );
        assert!(
            (end - sec.p2).length() < 1e-9,
            "arc end {end:?} != p2 {:?}",
            sec.p2
        );
        // Every point along U must lie on the circle of radius 1 about center.
        for k in 0..=8 {
            let u = f64::from(k) / 8.0;
            let p = surf.evaluate(u, v);
            let r = (p - sec.center).length();
            assert!(
                (r - 1.0).abs() < 1e-9,
                "U={u} V={v}: radius {r} != 1 (point {p:?} not on the fillet arc)"
            );
        }
    }

    /// Assert circularity across the *whole* (U, V) grid, including interior
    /// sections. For `quarter_circle_sections` every cross-section is a unit
    /// circle centred on the line `(·, 1, 1)` parallel to X — only X shifts
    /// between sections — so every surface sample must sit at distance 1 from
    /// that axis: `sqrt((y-1)^2 + (z-1)^2) == 1`. This catches V-interpolation
    /// regressions (knot vector / evaluator) that endpoint-only sampling misses.
    fn assert_circular_over_grid(surf: &NurbsSurface) {
        for ku in 0..=8 {
            for kv in 0..=8 {
                let u = f64::from(ku) / 8.0;
                let v = f64::from(kv) / 8.0;
                let p = surf.evaluate(u, v);
                let r = ((p.y() - 1.0).powi(2) + (p.z() - 1.0).powi(2)).sqrt();
                assert!(
                    (r - 1.0).abs() < 1e-9,
                    "U={u} V={v}: off-axis radius {r} != 1 (point {p:?})"
                );
            }
        }
    }

    /// Regression for the transposed control-point grid: the surface must build
    /// for any section count, not only `n == 3`. With the U/V axes swapped, the
    /// arc's 6-knot vector was validated against the section count, so every
    /// `n != 3` failed with `InvalidKnotVector { expected: n + 3, got: 6 }`.
    #[test]
    fn approximate_blend_surface_handles_many_sections() {
        for n in [2usize, 3, 4, 5, 8, 21] {
            let sections = quarter_circle_sections(n);
            let surf = approximate_blend_surface(&sections)
                .unwrap_or_else(|e| panic!("n={n} sections should build a surface, got {e:?}"));

            // U is the rational quadratic arc; V interpolates the sections.
            assert_eq!(surf.degree_u(), 2, "U (arc) must be degree 2 for n={n}");
            assert_eq!(
                surf.degree_v(),
                (n - 1).min(3),
                "V (sections) degree for n={n}"
            );

            // The arc shape must be preserved at both ends of the section span
            // (clamped knots interpolate the first and last section exactly).
            assert_circular_arc(&surf, 0.0, &sections[0]);
            assert_circular_arc(&surf, 1.0, &sections[n - 1]);

            // ...and at every interior section, not just the endpoints.
            assert_circular_over_grid(&surf);
        }
    }

    /// The sections of a radius-1 ball rolling round the foot of a boss of
    /// radius 3 standing on `z = 0`, at each of `angles` about `+z`.
    fn boss_foot_sections(angles: &[f64]) -> Vec<CircSection> {
        let (wall, radius) = (3.0, 1.0);
        angles
            .iter()
            .map(|&t| {
                let at = |reach: f64, z: f64| Point3::new(reach * t.cos(), reach * t.sin(), z);
                CircSection {
                    p1: at(wall + radius, 0.0),
                    p2: at(wall, radius),
                    center: at(wall + radius, radius),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t,
                }
            })
            .collect()
    }

    /// Along a curved spine, at each section's own parameter (each step the
    /// farthest the centre or a contact moves), the band's row is that
    /// section's arc: it starts and ends on the contacts and stays on the
    /// ball, in the section's plane.
    #[test]
    fn interpolate_blend_surface_passes_through_its_sections() {
        let angles = [0.0, 0.1, 0.35, 0.5, 0.9, 1.2, 1.5];
        for n in 2..=angles.len() {
            let sections = boss_foot_sections(&angles[..n]);
            let surf = interpolate_blend_surface(&sections).unwrap();
            let mut params = vec![0.0];
            for pair in sections.windows(2) {
                let step = (pair[1].center - pair[0].center)
                    .length()
                    .max((pair[1].p1 - pair[0].p1).length())
                    .max((pair[1].p2 - pair[0].p2).length());
                params.push(params[params.len() - 1] + step);
            }
            let total = params[n - 1];
            for (sec, param) in sections.iter().zip(&params) {
                let v = param / total;
                let case = format!("n = {n}, section at {}", sec.t);
                assert!((surf.evaluate(0.0, v) - sec.p1).length() < 1e-12, "{case}");
                assert!((surf.evaluate(1.0, v) - sec.p2).length() < 1e-12, "{case}");
                let plane = (sec.p1 - sec.center)
                    .cross(sec.p2 - sec.center)
                    .normalize()
                    .unwrap();
                for k in 0..=8 {
                    let p = surf.evaluate(f64::from(k) / 8.0, v);
                    let off_ball = ((p - sec.center).length() - sec.radius).abs();
                    let off_plane = (p - sec.center).dot(plane).abs();
                    assert!(off_ball < 1e-12, "{case}: {off_ball:e} off the ball");
                    assert!(off_plane < 1e-12, "{case}: {off_plane:e} off its plane");
                }
            }
        }
    }

    /// Two sections at one place give the band no length to run along: the
    /// interpolation refuses them, and the band built on them as control
    /// points still stands.
    #[test]
    fn interpolate_blend_surface_refuses_coincident_sections() {
        let sections = boss_foot_sections(&[0.4, 0.4]);
        assert!(interpolate_blend_surface(&sections).is_err());
        assert!(approximate_blend_surface(&sections).is_ok());
    }

    /// Four sections at one place and one beyond put four parameters at
    /// zero, and the collocation basis there is not finite: the
    /// interpolation refuses them, and the band built on them as control
    /// points still stands.
    #[test]
    fn interpolate_blend_surface_refuses_a_run_of_coincident_sections() {
        let sections = boss_foot_sections(&[0.4, 0.4, 0.4, 0.4, 0.5]);
        assert!(interpolate_blend_surface(&sections).is_err());
        assert!(approximate_blend_surface(&sections).is_ok());
    }
}
