//! Point projection onto NURBS curves and surfaces.
//!
//! Finds the closest point on a curve or surface to a given point in space.
//! Used for Boolean classification, snapping, distance queries, and
//! tessellation refinement.
//!
//! Algorithms follow NURBS Book A6.1–A6.6: subdivision for initial guess
//! followed by Newton–Raphson refinement.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// Maximum Newton iterations before declaring convergence failure.
const MAX_ITERATIONS: usize = 50;

/// Halvings of a damped descent step before it counts as no step: enough to
/// shrink any step on a model of sane scale below its tolerance, and a bound
/// where no step can meet the tolerance at all.
const MAX_HALVINGS: usize = 64;

/// Number of grid subdivisions per direction for surface coarse search.
const SURFACE_GRID_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of projecting a point onto a curve.
#[derive(Debug, Clone, Copy)]
pub struct CurveProjection {
    /// Parameter value at the closest point.
    pub parameter: f64,
    /// The closest point on the curve.
    pub point: Point3,
    /// Distance from the input point to the closest point.
    pub distance: f64,
}

/// Result of projecting a point onto a surface.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceProjection {
    /// Parameter value u at the closest point.
    pub u: f64,
    /// Parameter value v at the closest point.
    pub v: f64,
    /// The closest point on the surface.
    pub point: Point3,
    /// Distance from the input point to the closest point.
    pub distance: f64,
}

// ---------------------------------------------------------------------------
// Curve projection
// ---------------------------------------------------------------------------

/// Find the closest point on a NURBS curve to the given point.
///
/// Samples each Bezier segment (knot span) for initial guesses, then
/// Newton–Raphson refinement (NURBS Book A6.1 + A6.3–A6.4).
///
/// # Errors
///
/// None for a curve that constructed: the `Result` stays in the public
/// signature, which callers across the workspace match on.
#[allow(clippy::unnecessary_wraps)]
pub fn project_point_to_curve(
    curve: &NurbsCurve,
    point: Point3,
    tolerance: f64,
) -> Result<CurveProjection, MathError> {
    let u_min = curve.knots()[curve.degree()];

    let candidates = curve_coarse_search(curve, point);

    // Run Newton from each candidate and keep the globally closest result.
    let mut best_u = u_min;
    let mut best_pt = curve.evaluate(u_min);
    let mut best_dist = (best_pt - point).length();

    for (u_guess, lo, hi) in candidates {
        let (u_refined, pt_refined) = curve_newton_refine(curve, point, u_guess, lo, hi, tolerance);
        let dist = (pt_refined - point).length();
        if dist < best_dist {
            best_dist = dist;
            best_u = u_refined;
            best_pt = pt_refined;
        }
    }

    Ok(CurveProjection {
        parameter: best_u,
        point: best_pt,
        distance: best_dist,
    })
}

/// Coarse search: sample each knot span (each of the curve's Bezier
/// segments) to find multiple candidate parameter values for Newton
/// refinement. The spans are sampled on the curve itself: decomposing it
/// into segments first costs knot insertions, on every projection, for the
/// same points.
///
/// Returns the candidates with the span each came from, as
/// `(u, span_start, span_end)`. Newton runs inside its seed's span, where
/// the curve is one polynomial piece: at a corner (a knot of multiplicity
/// `p`) the other piece's derivatives would carry it away from the corner.
/// So the seeds are the closest sample of each of the closest spans (a
/// short span's crowded samples take one seed, not all of them, and a knot
/// ends one span and starts the next, so it can seed both) and the closest
/// local minima of the sampled distance along the curve (several basins in
/// one span). An interior span ends one step below its knot, since the
/// curve is read there by the next piece.
#[allow(clippy::cast_precision_loss)]
fn curve_coarse_search(curve: &NurbsCurve, point: Point3) -> Vec<(f64, f64, f64)> {
    type Span = (f64, f64, Vec<(f64, f64)>);
    const SEEDS: usize = 5;
    let knots = curve.knots();
    let p = curve.degree();
    let (lo, hi) = (knots[p], knots[knots.len() - p - 1]);

    // Each span's (start, end) and its (parameter, distance_sq) samples, in
    // order along the curve.
    let n_samples = (p + 1).max(5) * 2;
    let mut spans: Vec<Span> = Vec::new();
    for span in knots.windows(2) {
        let (u_start, u_end) = (span[0], span[1]);
        if u_end <= u_start || u_start < lo || u_end > hi {
            continue;
        }
        let top = if u_end < hi { u_end.next_down() } else { u_end };
        let samples = (0..=n_samples)
            .map(|i| {
                let t = i as f64 / n_samples as f64;
                let u = t.mul_add(u_end - u_start, u_start).min(top);
                (u, (curve.evaluate(u) - point).length_squared())
            })
            .collect();
        spans.push((u_start, top, samples));
    }

    let closest = |samples: &[(f64, f64)]| {
        samples
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0.0, f64::INFINITY))
    };
    let mut nearest: Vec<(f64, f64, f64, f64)> = spans
        .iter()
        .map(|(u_start, top, samples)| {
            let (u, d) = closest(samples);
            (d, u, *u_start, *top)
        })
        .collect();
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
    nearest.truncate(SEEDS);

    // A sample's neighbours along the curve; a span's end sample is the
    // next span's start, so it looks past it. A run of equal minima counts
    // once.
    let mut minima: Vec<(f64, f64, f64, f64)> = Vec::new();
    for (s, (u_start, top, samples)) in spans.iter().enumerate() {
        let last = samples.len() - 1;
        let mut in_run = false;
        for (i, &(u, d)) in samples.iter().enumerate() {
            let prev = if i > 0 {
                Some(samples[i - 1].1)
            } else {
                s.checked_sub(1).map(|r| {
                    let before = &spans[r].2;
                    before[before.len() - 2].1
                })
            };
            let next = if i < last {
                Some(samples[i + 1].1)
            } else {
                spans.get(s + 1).map(|after| after.2[1].1)
            };
            let minimum = prev.is_none_or(|q| d <= q) && next.is_none_or(|q| d <= q);
            if minimum && !in_run {
                minima.push((d, u, *u_start, *top));
            }
            in_run = minimum;
        }
    }
    minima.sort_by(|a, b| a.0.total_cmp(&b.0));
    minima.truncate(SEEDS);

    let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
    for (_, u, u_start, top) in nearest.into_iter().chain(minima) {
        if !seeds
            .iter()
            .any(|s| s.0.to_bits() == u.to_bits() && s.1.to_bits() == u_start.to_bits())
        {
            seeds.push((u, u_start, top));
        }
    }
    seeds
}

/// Newton–Raphson refinement for curve point projection.
///
/// Finds parameter u in `[u_min, u_max]` that minimizes ||C(u) - P||
/// starting from `u_init`, and returns the closest point it evaluated.
/// Each step starts from the closest iterate so far, and a step that lands
/// farther is halved back toward it (a backtracking line search): a plain
/// Newton step can overshoot the minimum, clamp to the interval's end and
/// cycle there.
#[allow(clippy::suspicious_operation_groupings)]
fn curve_newton_refine(
    curve: &NurbsCurve,
    point: Point3,
    u_init: f64,
    u_min: f64,
    u_max: f64,
    tolerance: f64,
) -> (f64, Point3) {
    let tol_sq = tolerance * tolerance;
    let mut u = u_init;
    let mut best: Option<(f64, Point3, f64)> = None;

    for _ in 0..MAX_ITERATIONS {
        let ders = curve.derivatives(u, 2);
        let c_pt = Point3::new(ders[0].x(), ders[0].y(), ders[0].z());
        let c_prime = ders[1]; // C'(u)
        let c_double_prime = ders[2]; // C''(u)
        let diff = c_pt - point; // C(u) - P

        let dist_sq = diff.length_squared();

        if let Some((best_u, _, best_dist_sq)) = best
            && dist_sq >= best_dist_sq
        {
            if (u - best_u).abs() < tolerance * (1.0 + best_u.abs()) {
                break;
            }
            u = 0.5 * (u + best_u);
            continue;
        }
        best = Some((u, c_pt, dist_sq));

        // Convergence check 1: point coincidence.
        if dist_sq < tol_sq {
            break;
        }

        // f(u) = C'(u) · (C(u) - P)
        let f_val = c_prime.dot(diff);

        // Convergence check 2: zero cosine (perpendicularity).
        // cos²(angle) = (C'·diff)² / (|C'|² · |diff|²) < tol²
        let c_prime_len_sq = c_prime.length_squared();
        if c_prime_len_sq > 1e-30 && dist_sq > tol_sq {
            let cos_sq = (f_val * f_val) / (c_prime_len_sq * dist_sq);
            if cos_sq < tol_sq {
                break;
            }
        }

        // f'(u) = C''(u) · (C(u) - P) + |C'(u)|². Where it is not positive
        // the Newton step climbs toward a farthest point, and the
        // Gauss-Newton step (|C'(u)|² alone) descends instead.
        let mut f_prime = c_double_prime.dot(diff) + c_prime_len_sq;
        if f_prime <= 0.0 {
            f_prime = c_prime_len_sq;
        }

        // Guard against zero denominator.
        if f_prime.abs() < 1e-30 {
            break;
        }

        let delta_u = f_val / f_prime;
        let u_new = (u - delta_u).clamp(u_min, u_max);

        // Guard NaN.
        if u_new.is_nan() {
            break;
        }

        // Convergence check 3: parameter step negligible.
        if (u_new - u).abs() < tolerance * (1.0 + u.abs()) {
            let pt = curve.evaluate(u_new);
            let d_sq = (pt - point).length_squared();
            if d_sq < dist_sq {
                best = Some((u_new, pt, d_sq));
            }
            break;
        }

        u = u_new;
    }

    best.map_or_else(|| (u_init, curve.evaluate(u_init)), |b| (b.0, b.1))
}

// ---------------------------------------------------------------------------
// Surface projection
// ---------------------------------------------------------------------------

/// Find the closest point on a NURBS surface to the given point.
///
/// Uses grid evaluation for initial guess, then 2D Newton–Raphson
/// refinement (NURBS Book A6.2 + A6.5–A6.6). A result off the surface is
/// then checked against every knot-span patch: each lies within the box of
/// its own control points, so only a patch whose box comes nearer than the
/// result can hold a nearer point, and those are searched nearest box first.
///
/// # Errors
///
/// Returns [`MathError::ConvergenceFailure`] if Newton iteration does not
/// converge within the maximum number of iterations and no patch search
/// reaches a point either.
pub fn project_point_to_surface(
    surface: &NurbsSurface,
    point: Point3,
    tolerance: f64,
) -> Result<SurfaceProjection, MathError> {
    let seeded = project_from_coarse_grid(surface, point, tolerance);
    if tolerance.is_nan()
        || tolerance <= 0.0
        || seeded.as_ref().is_ok_and(|p| p.distance <= tolerance)
    {
        return seeded;
    }
    match search_span_patches(surface, point, tolerance, seeded.as_ref().ok().copied()) {
        Some(found) => Ok(found),
        None => seeded,
    }
}

fn project_from_coarse_grid(
    surface: &NurbsSurface,
    point: Point3,
    tolerance: f64,
) -> Result<SurfaceProjection, MathError> {
    let (u_guess, v_guess) = surface_coarse_search(surface, point);

    let knots_u = surface.knots_u();
    let knots_v = surface.knots_v();
    let pu = surface.degree_u();
    let pv = surface.degree_v();
    let u_min = knots_u[pu];
    let u_max = knots_u[knots_u.len() - pu - 1];
    let v_min = knots_v[pv];
    let v_max = knots_v[knots_v.len() - pv - 1];

    // Wrapping lets Newton cross a closed direction's seam from a seed on
    // the far copy of it; where the seam has a kink it can bounce across
    // without converging, so a failed wrapped solve retries clamped.
    let wraps = surface.is_periodic_u() || surface.is_periodic_v();
    let (u_final, v_final, pt_final) = surface_newton_refine(
        surface, point, u_guess, v_guess, u_min, u_max, v_min, v_max, tolerance, wraps,
    )
    .or_else(|err| {
        if wraps {
            surface_newton_refine(
                surface, point, u_guess, v_guess, u_min, u_max, v_min, v_max, tolerance, false,
            )
        } else {
            Err(err)
        }
    })?;
    let dist = (pt_final - point).length();

    Ok(SurfaceProjection {
        u: u_final,
        v: v_final,
        point: pt_final,
        distance: dist,
    })
}

/// Coarse search: evaluate surface on a uniform grid and find the closest
/// grid point.
#[allow(clippy::cast_precision_loss)]
fn surface_coarse_search(surface: &NurbsSurface, point: Point3) -> (f64, f64) {
    let knots_u = surface.knots_u();
    let knots_v = surface.knots_v();
    let pu = surface.degree_u();
    let pv = surface.degree_v();
    let u_min = knots_u[pu];
    let u_max = knots_u[knots_u.len() - pu - 1];
    let v_min = knots_v[pv];
    let v_max = knots_v[knots_v.len() - pv - 1];

    let mut best_u = u_min;
    let mut best_v = v_min;
    let mut best_dist_sq = f64::INFINITY;

    let n = SURFACE_GRID_SIZE;
    for i in 0..=n {
        let u = (i as f64 / n as f64).mul_add(u_max - u_min, u_min);
        for j in 0..=n {
            let v = (j as f64 / n as f64).mul_add(v_max - v_min, v_min);
            let pt = surface.evaluate(u, v);
            let d_sq = (pt - point).length_squared();
            if d_sq < best_dist_sq {
                best_dist_sq = d_sq;
                best_u = u;
                best_v = v;
            }
        }
    }

    (best_u, best_v)
}

/// 2D Newton–Raphson refinement for surface point projection.
///
/// Solves the 2×2 system at each step to find the (u, v) that minimizes
/// ||S(u,v) - P||.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
#[allow(clippy::suspicious_operation_groupings)]
fn surface_newton_refine(
    surface: &NurbsSurface,
    point: Point3,
    u_init: f64,
    v_init: f64,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    tolerance: f64,
    wrap_closed: bool,
) -> Result<(f64, f64, Point3), MathError> {
    let mut u = u_init;
    let mut v = v_init;
    let (closed_u, closed_v) = (
        wrap_closed && surface.is_periodic_u(),
        wrap_closed && surface.is_periodic_v(),
    );

    for _ in 0..MAX_ITERATIONS {
        let ders = surface.derivatives(u, v, 1);
        let s_pt = Point3::new(ders[0][0].x(), ders[0][0].y(), ders[0][0].z());
        let deriv_u = ders[1][0]; // ∂S/∂u
        let deriv_v = ders[0][1]; // ∂S/∂v
        let r = s_pt - point; // S(u,v) - P

        // Convergence check 1: point coincidence.
        let dist = r.length();
        if dist < tolerance {
            return Ok((u, v, s_pt));
        }

        // Convergence check 2: zero cosine in both directions.
        let du_len = deriv_u.length();
        let dv_len = deriv_v.length();
        let dot_du_r = deriv_u.dot(r);
        let dot_dv_r = deriv_v.dot(r);
        if du_len > 0.0 && dv_len > 0.0 {
            let cos_u = dot_du_r.abs() / (du_len * dist);
            let cos_v = dot_dv_r.abs() / (dv_len * dist);
            if cos_u < tolerance && cos_v < tolerance {
                return Ok((u, v, s_pt));
            }
        }

        let Some((delta_u, delta_v)) = gauss_newton_step(deriv_u, deriv_v, dot_du_r, dot_dv_r)
        else {
            return Ok((u, v, s_pt));
        };

        let (u_new, step_u) = advance_param(u, delta_u, u_min, u_max, closed_u);
        let (v_new, step_v) = advance_param(v, delta_v, v_min, v_max, closed_v);

        // Convergence check 3: parameter step negligible.
        let step = (deriv_u * step_u + deriv_v * step_v).length();
        if step < tolerance {
            let pt = surface.evaluate(u_new, v_new);
            return Ok((u_new, v_new, pt));
        }

        u = u_new;
        v = v_new;
    }

    Err(MathError::ConvergenceFailure {
        iterations: MAX_ITERATIONS,
    })
}

/// Along a closed direction the step wraps across the seam instead of
/// stopping at the domain end: a seed on the far copy of the seam (the
/// coarse grid samples both ends) must still reach a point just short of
/// it. Returns the new parameter and the step taken to it.
fn advance_param(x: f64, delta: f64, lo: f64, hi: f64, closed: bool) -> (f64, f64) {
    if closed && hi > lo {
        (lo + (x + delta - lo).rem_euclid(hi - lo), delta)
    } else {
        let next = (x + delta).clamp(lo, hi);
        (next, next - x)
    }
}

/// The Gauss–Newton step toward the foot of the normal, from the partials
/// and their dot products with `S - P`, or `None` where both partials
/// vanish and there is no step to take.
#[allow(clippy::similar_names)]
fn gauss_newton_step(
    deriv_u: Vec3,
    deriv_v: Vec3,
    dot_du_r: f64,
    dot_dv_r: f64,
) -> Option<(f64, f64)> {
    // J = [S_u · S_u,  S_u · S_v]
    //     [S_v · S_u,  S_v · S_v]
    let j00 = deriv_u.dot(deriv_u);
    let j01 = deriv_u.dot(deriv_v);
    let j11 = deriv_v.dot(deriv_v);
    let rhs0 = -dot_du_r;
    let rhs1 = -dot_dv_r;

    // Use a relative threshold so the singularity test stays meaningful
    // near surface poles / cone apex where both derivatives shrink to zero.
    let det = j00.mul_add(j11, -(j01 * j01));
    if det.abs() < (j00 + j11).max(1e-30) * 1e-12 {
        // Near-singular: Tikhonov (Levenberg–Marquardt) regularisation
        // biases the step toward zero rather than blowing up near poles
        // and apices.
        let lambda = (j00 + j11).max(1e-10) * 1e-4;
        let j00r = j00 + lambda;
        let j11r = j11 + lambda;
        let det_r = j00r.mul_add(j11r, -(j01 * j01));
        if det_r.abs() < 1e-30 {
            if j00 > j11 {
                Some((rhs0 / j00.max(1e-30), 0.0))
            } else if j11 > 1e-30 {
                Some((0.0, rhs1 / j11.max(1e-30)))
            } else {
                None
            }
        } else {
            Some((
                rhs0.mul_add(j11r, -(rhs1 * j01)) / det_r,
                j00r.mul_add(rhs1, -(j01 * rhs0)) / det_r,
            ))
        }
    } else {
        Some((
            rhs0.mul_add(j11, -(rhs1 * j01)) / det,
            j00.mul_add(rhs1, -(j01 * rhs0)) / det,
        ))
    }
}

/// One knot span of each direction: its parameter box, and the distance
/// to the box of the control points that shape it, which bounds how near
/// any point of the patch can come.
struct SpanPatch {
    u: (f64, f64),
    v: (f64, f64),
    reach: f64,
}

impl SpanPatch {
    fn holds(&self, (u, v): (f64, f64)) -> bool {
        (self.u.0..=self.u.1).contains(&u) && (self.v.0..=self.v.1).contains(&v)
    }

    /// The nearest of a 3 x 3 grid of samples over the patch.
    fn nearest_sample(&self, surface: &NurbsSurface, point: Point3) -> (f64, f64) {
        let mut best = (f64::INFINITY, self.u.0, self.v.0);
        for fu in [0.0_f64, 0.5, 1.0] {
            let u = fu.mul_add(self.u.1 - self.u.0, self.u.0);
            for fv in [0.0_f64, 0.5, 1.0] {
                let v = fv.mul_add(self.v.1 - self.v.0, self.v.0);
                let d = (surface.evaluate(u, v) - point).length_squared();
                if d < best.0 {
                    best = (d, u, v);
                }
            }
        }
        (best.1, best.2)
    }
}

/// The knot-span patches whose control box comes nearer than `bound`,
/// nearest first, leaving out any that holds `settled`. `None` for a weight
/// that is not positive, which can carry the surface outside its control
/// points' hull.
fn span_patches(
    surface: &NurbsSurface,
    point: Point3,
    bound: f64,
    settled: Option<(f64, f64)>,
) -> Option<Vec<SpanPatch>> {
    if surface
        .weights()
        .iter()
        .flatten()
        .any(|&w| w.is_nan() || w <= 0.0)
    {
        return None;
    }
    let spans = |knots: &[f64], p: usize| -> Vec<usize> {
        (p..knots.len() - p - 1)
            .filter(|&k| knots[k] < knots[k + 1])
            .collect()
    };
    let (knots_u, knots_v) = (surface.knots_u(), surface.knots_v());
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let rows = surface.control_points();
    let spans_v = spans(knots_v, pv);
    let mut patches = Vec::new();
    for i in spans(knots_u, pu) {
        for &j in &spans_v {
            let hull = Aabb3::from_points(
                rows[i - pu..=i]
                    .iter()
                    .flat_map(|row| row[j - pv..=j].iter().copied()),
            );
            let reach = hull.distance_squared_to_point(point).sqrt();
            let patch = SpanPatch {
                u: (knots_u[i], knots_u[i + 1]),
                v: (knots_v[j], knots_v[j + 1]),
                reach,
            };
            if reach < bound && !settled.is_some_and(|uv| patch.holds(uv)) {
                patches.push(patch);
            }
        }
    }
    patches.sort_by(|a, b| a.reach.total_cmp(&b.reach));
    Some(patches)
}

/// Search the patches that could hold a point nearer than `start` by more
/// than `tolerance`, skipping those that hold `start` itself, whose valley
/// its refinement already settled.
fn search_span_patches(
    surface: &NurbsSurface,
    point: Point3,
    tolerance: f64,
    start: Option<SurfaceProjection>,
) -> Option<SurfaceProjection> {
    let bound = start.map_or(f64::INFINITY, |b| b.distance - tolerance);
    let Some(patches) = span_patches(surface, point, bound, start.map(|b| (b.u, b.v))) else {
        return start;
    };
    let (knots_u, knots_v) = (surface.knots_u(), surface.knots_v());
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let domain = [
        knots_u[pu],
        knots_u[knots_u.len() - pu - 1],
        knots_v[pv],
        knots_v[knots_v.len() - pv - 1],
    ];
    let closed = (surface.is_periodic_u(), surface.is_periodic_v());
    let mut best = start;
    for patch in &patches {
        if best.is_some_and(|b| patch.reach >= b.distance - tolerance) {
            break;
        }
        let (u, v) = patch.nearest_sample(surface, point);
        let found = surface_descend(surface, point, (u, v), domain, closed, tolerance);
        if found.distance.is_finite()
            && best.is_none_or(|b| found.distance < b.distance - tolerance)
        {
            best = Some(found);
        }
    }
    best
}

/// Gauss–Newton from `seed`, halving any step that does not bring the
/// point nearer: it settles where the undamped iteration can circle a
/// valley, and never ends farther than it began.
#[allow(clippy::similar_names)]
fn surface_descend(
    surface: &NurbsSurface,
    point: Point3,
    seed: (f64, f64),
    [u_min, u_max, v_min, v_max]: [f64; 4],
    (closed_u, closed_v): (bool, bool),
    tolerance: f64,
) -> SurfaceProjection {
    let (mut u, mut v) = seed;
    let mut at = surface.evaluate(u, v);
    let mut dist = (at - point).length();
    for _ in 0..MAX_ITERATIONS {
        if dist < tolerance {
            break;
        }
        let ders = surface.derivatives(u, v, 1);
        let (deriv_u, deriv_v) = (ders[1][0], ders[0][1]);
        let r = at - point;
        let (dot_du_r, dot_dv_r) = (deriv_u.dot(r), deriv_v.dot(r));
        let (du_len, dv_len) = (deriv_u.length(), deriv_v.length());
        if du_len > 0.0
            && dv_len > 0.0
            && dot_du_r.abs() < tolerance * du_len * dist
            && dot_dv_r.abs() < tolerance * dv_len * dist
        {
            break;
        }
        let Some((delta_u, delta_v)) = gauss_newton_step(deriv_u, deriv_v, dot_du_r, dot_dv_r)
        else {
            break;
        };
        let mut moved = false;
        let mut scale = 1.0;
        for _ in 0..MAX_HALVINGS {
            let (u_new, step_u) = advance_param(u, scale * delta_u, u_min, u_max, closed_u);
            let (v_new, step_v) = advance_param(v, scale * delta_v, v_min, v_max, closed_v);
            if (deriv_u * step_u + deriv_v * step_v).length() < tolerance {
                break;
            }
            let next = surface.evaluate(u_new, v_new);
            let d = (next - point).length();
            if d < dist {
                (u, v, at, dist) = (u_new, v_new, next, d);
                moved = true;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
    }
    SurfaceProjection {
        u,
        v,
        point: at,
        distance: dist,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vec::Vec3;

    const TOL: f64 = 1e-8;

    /// A simple line from (0,0,0) to (10,0,0) as a degree-1 NURBS.
    fn line_curve() -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .expect("valid line")
    }

    /// Quarter circle arc as a rational NURBS (degree 2).
    fn quarter_circle() -> NurbsCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, w, 1.0],
        )
        .expect("valid quarter circle")
    }

    /// Cubic Bezier curve.
    fn cubic_bezier() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid cubic")
    }

    /// Bilinear flat patch (z=0 plane, from (0,0) to (1,1)).
    fn flat_patch() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid flat patch")
    }

    // -- Curve tests -------------------------------------------------------

    #[test]
    fn project_to_line() {
        let c = line_curve();
        // Point (5, 3, 0) — closest point should be (5, 0, 0) at u=0.5.
        let res =
            project_point_to_curve(&c, Point3::new(5.0, 3.0, 0.0), TOL).expect("should converge");
        assert!((res.parameter - 0.5).abs() < TOL, "u={}", res.parameter);
        assert!((res.point.x() - 5.0).abs() < TOL);
        assert!((res.point.y()).abs() < TOL);
        assert!((res.distance - 3.0).abs() < TOL, "dist={}", res.distance);
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn project_to_circle() {
        let c = quarter_circle();
        // Point (2, 2, 0) — closest point should be on the unit circle at 45°.
        let res =
            project_point_to_curve(&c, Point3::new(2.0, 2.0, 0.0), TOL).expect("should converge");
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (res.point.x() - expected).abs() < 1e-6,
            "x={} expected={}",
            res.point.x(),
            expected
        );
        assert!(
            (res.point.y() - expected).abs() < 1e-6,
            "y={} expected={}",
            res.point.y(),
            expected
        );
        // Distance from (2,2) to unit circle at 45° = sqrt(8) - 1.
        let expected_dist = 2.0_f64.hypot(2.0) - 1.0;
        assert!(
            (res.distance - expected_dist).abs() < 1e-6,
            "dist={} expected={}",
            res.distance,
            expected_dist
        );
    }

    #[test]
    fn project_endpoint() {
        let c = cubic_bezier();
        // Project a point very close to the start endpoint.
        let res =
            project_point_to_curve(&c, Point3::new(0.0, 0.01, 0.0), TOL).expect("should converge");
        assert!(res.distance < 0.02, "dist={}", res.distance);
        assert!(res.parameter < 0.1, "u={}", res.parameter);
    }

    #[test]
    fn project_far_point() {
        let c = cubic_bezier();
        // A point far away should still converge.
        let res =
            project_point_to_curve(&c, Point3::new(2.0, 100.0, 0.0), TOL).expect("should converge");
        // The closest point should be roughly at the top of the curve (y ≈ 1.5).
        assert!(res.point.y() > 0.0);
        assert!(res.distance < 100.0);
    }

    #[test]
    fn project_on_curve() {
        let c = cubic_bezier();
        // Evaluate a point on the curve, then project it back.
        let u_orig = 0.3;
        let pt_on = c.evaluate(u_orig);
        let res = project_point_to_curve(&c, pt_on, TOL).expect("should converge");
        assert!(res.distance < TOL, "dist={}", res.distance);
        assert!(
            (res.parameter - u_orig).abs() < 1e-4,
            "u={} expected={}",
            res.parameter,
            u_orig
        );
    }

    /// A cubic B-spline over four spans: points on it at and beside its
    /// interior knots, and points off it, project to the closest of 100,001
    /// samples or closer.
    #[test]
    fn project_across_knot_spans() {
        let c = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, -1.0, 0.5),
                Point3::new(3.0, 2.5, 0.0),
                Point3::new(4.0, 0.0, -0.5),
                Point3::new(5.0, 1.5, 0.0),
                Point3::new(6.0, 0.0, 0.0),
            ],
            vec![1.0; 7],
        )
        .expect("valid cubic");
        let dense = dense_samples(&c);
        for u in [0.25, 0.25 + 1e-3, 0.5 - 1e-3, 0.5, 0.75, 0.75 + 1e-3] {
            let on = c.evaluate(u);
            let res = project_point_to_curve(&c, on, TOL).expect("should converge");
            assert!(res.distance < TOL, "u={u}: dist {}", res.distance);
            for off in [Vec3::new(0.0, 0.3, 0.2), Vec3::new(0.1, -0.4, 0.0)] {
                let p = on + off;
                let res = project_point_to_curve(&c, p, TOL).expect("should converge");
                let brute = closest_sample(&dense, p);
                assert!(
                    res.distance <= brute + 1e-9,
                    "u={u}: {} > {brute}",
                    res.distance
                );
            }
        }
    }

    /// 100,001 points evenly spaced in the parameter of a curve on `[0, 1]`.
    fn dense_samples(c: &NurbsCurve) -> Vec<Point3> {
        (0..=100_000)
            .map(|k| c.evaluate(f64::from(k) / 100_000.0))
            .collect()
    }

    fn closest_sample(dense: &[Point3], p: Point3) -> f64 {
        dense
            .iter()
            .map(|q| (*q - p).length())
            .fold(f64::INFINITY, f64::min)
    }

    /// A rational quadratic with a corner (a double knot) at `u = 0.5`:
    /// points on it beside the corner project to themselves, and points
    /// below the corner to the closest of 100,001 samples or closer. A run
    /// seeded at the corner reads the far piece's derivatives there unless
    /// it stays in its own span.
    #[test]
    fn project_beside_a_rational_corner() {
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.8, 0.2, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.2, 0.2, 0.0),
                Point3::new(2.0, 1.0, 0.0),
            ],
            vec![1.0, 2.0, 1.0, 0.5, 1.0],
        )
        .expect("valid quadratic");
        for u in [0.48, 0.495, 0.4999, 0.5, 0.5001, 0.505, 0.52] {
            let res = project_point_to_curve(&c, c.evaluate(u), TOL).expect("should converge");
            assert!(res.distance < TOL, "u={u}: dist {}", res.distance);
        }
        let corner = project_point_to_curve(&c, Point3::new(1.038, -0.282, 0.0), TOL)
            .expect("should converge");
        assert!(
            (corner.parameter - 0.5).abs() < 1e-9,
            "u={}",
            corner.parameter
        );
        let dense = dense_samples(&c);
        for i in 0..6 {
            for j in 0..6 {
                let p = Point3::new(0.5 + 0.2 * f64::from(i), -1.0 + 0.2 * f64::from(j), 0.0);
                let res = project_point_to_curve(&c, p, TOL).expect("should converge");
                let brute = closest_sample(&dense, p);
                assert!(
                    res.distance <= brute + 1e-9,
                    "{p:?}: {} > {brute}",
                    res.distance
                );
            }
        }
    }

    /// A quadratic with a span 0.0008 long beside a long one: the short
    /// span's samples crowd together, and points on the long span beside it
    /// project to themselves.
    #[test]
    fn project_beside_a_short_span() {
        let pts = [
            (0.369, 2.254, 0.523),
            (1.49, 0.935, 1.239),
            (2.611, 2.353, 0.691),
            (3.864, 0.113, 0.841),
            (4.381, 1.241, 0.004),
            (5.108, 0.051, 1.203),
            (6.235, 2.743, 1.368),
        ];
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.404, 0.404, 0.5225, 0.5233, 1.0, 1.0, 1.0],
            pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect(),
            vec![1.0; 7],
        )
        .expect("valid quadratic");
        for u in [0.526, 0.527, 0.528, 0.53, 0.535, 0.54] {
            let res = project_point_to_curve(&c, c.evaluate(u), TOL).expect("should converge");
            assert!(
                res.distance < TOL,
                "u={u}: at {} dist {}",
                res.parameter,
                res.distance
            );
        }
    }

    /// A rational quartic whose long span ends at a double knot: from the
    /// seed at that knot a Newton step overshoots the point on the curve at
    /// `u = 0.847` and the next one clamps back to the knot, a cycle that
    /// halving the steps back toward the closer iterate breaks.
    #[test]
    fn project_where_newton_overshoots() {
        let pts = [
            (0.214, 2.374, 0.023),
            (1.728, 0.844, 0.444),
            (2.048, 2.959, 0.461),
            (3.318, 2.28, 1.146),
            (4.482, 0.889, 1.641),
            (5.768, 1.669, 1.668),
            (6.109, 0.189, 0.556),
            (7.604, 0.684, 1.081),
            (8.086, 1.522, 1.079),
            (9.709, 2.279, 1.852),
            (10.577, 2.794, 1.997),
            (11.147, 0.728, 1.253),
            (12.239, 2.713, 1.874),
        ];
        let c = NurbsCurve::new(
            4,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.247, 0.247, 0.247, 0.247, 0.877, 0.877, 0.913, 0.913,
                1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect(),
            vec![
                1.138, 1.092, 0.938, 1.464, 0.77, 1.44, 1.286, 0.536, 1.319, 0.675, 1.002, 0.793,
                1.279,
            ],
        )
        .expect("valid quartic");
        let res = project_point_to_curve(&c, c.evaluate(0.847), TOL).expect("should converge");
        assert!(
            res.distance < TOL,
            "u={}: dist {}",
            res.parameter,
            res.distance
        );
    }

    // -- Surface tests -----------------------------------------------------

    #[test]
    fn project_to_flat_quad() {
        let s = flat_patch();
        // Point (0.5, 0.5, 3.0) — should project to (0.5, 0.5, 0.0).
        let res =
            project_point_to_surface(&s, Point3::new(0.5, 0.5, 3.0), TOL).expect("should converge");
        assert!((res.point.x() - 0.5).abs() < TOL, "x={}", res.point.x());
        assert!((res.point.y() - 0.5).abs() < TOL, "y={}", res.point.y());
        assert!((res.point.z()).abs() < TOL, "z={}", res.point.z());
        assert!((res.distance - 3.0).abs() < TOL, "dist={}", res.distance);
    }

    #[test]
    fn project_on_surface() {
        let s = flat_patch();
        // Point directly on the surface.
        let res =
            project_point_to_surface(&s, Point3::new(0.3, 0.7, 0.0), TOL).expect("should converge");
        assert!(res.distance < TOL, "dist={}", res.distance);
    }

    #[test]
    fn project_above_surface() {
        let s = flat_patch();
        // Point at height 1 above the center.
        let res =
            project_point_to_surface(&s, Point3::new(0.5, 0.5, 1.0), TOL).expect("should converge");
        assert!(
            (res.distance - 1.0).abs() < TOL,
            "dist={} expected=1.0",
            res.distance
        );
        assert!((res.u - 0.5).abs() < TOL, "u={}", res.u);
        assert!((res.v - 0.5).abs() < TOL, "v={}", res.v);
    }

    /// Bilinear degenerate "cone apex" patch.
    ///
    /// Control grid:
    ///   v=0 row: apex=(0,0,0)  apex=(0,0,0)   ← S_u = 0 everywhere on this row
    ///   v=1 row: (-1,0,1)      (1,0,1)
    ///
    /// Parametric formula: S(u,v) = (v·(2u-1), 0, v)
    ///
    /// The Jacobian is rank-1 at v=0 (both S_u and S_v are degenerate there),
    /// which triggers the LM-regularisation branch in `project_point_to_surface`.
    fn apex_patch() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)], // v=0: apex
                vec![Point3::new(-1.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)], // v=1: base
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid apex patch")
    }

    /// Project a point whose nearest surface location is the degenerate apex.
    ///
    /// The surface S(u,v)=(v(2u−1), 0, v) lies in the xz-plane.  The query
    /// point (0, 1, 0) is displaced only in y, so its nearest surface point is
    /// the apex (0,0,0) — the only point that minimises the xz-distance.
    /// Without LM regularisation the Newton step blows up at v→0; with it the
    /// solver should converge and return (u≈0.5, v≈0, dist≈1).
    #[test]
    fn project_to_apex_singularity() {
        let s = apex_patch();
        let res = project_point_to_surface(&s, Point3::new(0.0, 1.0, 0.0), 1e-6)
            .expect("should converge at cone apex singularity");
        // Nearest point must be the apex.
        assert!(
            res.point.x().abs() < 1e-6 && res.point.y().abs() < 1e-6 && res.point.z().abs() < 1e-6,
            "nearest point should be apex, got ({:.4},{:.4},{:.4})",
            res.point.x(),
            res.point.y(),
            res.point.z()
        );
        assert!(
            (res.distance - 1.0).abs() < 1e-6,
            "distance to apex should be 1.0, got {:.8}",
            res.distance
        );
    }

    /// Project a point off-axis but close to the apex.  The solver must still
    /// converge despite starting near the singularity.
    #[test]
    fn project_near_apex_off_axis() {
        let s = apex_patch();
        // S(0.7, 0.05) = (0.05*(2*0.7-1), 0, 0.05) = (0.05*0.4, 0, 0.05) = (0.02, 0, 0.05)
        // Query close to that surface point but displaced in y.
        let res = project_point_to_surface(&s, Point3::new(0.02, 0.3, 0.05), 1e-6)
            .expect("should converge near apex");
        assert!(
            (res.distance - 0.3).abs() < 0.02,
            "expected distance ≈ 0.3, got {:.6}",
            res.distance
        );
        // Nearest surface point should be close to S(0.7, 0.05) = (0.02, 0, 0.05).
        assert!(
            (res.point.z() - 0.05).abs() < 0.02,
            "nearest point z should be ≈ 0.05, got z={:.4}",
            res.point.z()
        );
    }

    /// A closed cubic wall about 940 mm round, its control net wiggling in
    /// and out across 117 knot spans.
    #[allow(clippy::cast_precision_loss)]
    fn wavy_ring() -> NurbsSurface {
        let columns = 120;
        let mut knots_u = vec![0.0; 4];
        knots_u.extend((1..columns - 3).map(|k| k as f64));
        knots_u.extend([(columns - 3) as f64; 4]);
        let ring = |i: usize, z: f64| {
            let a = std::f64::consts::TAU * (i % (columns - 1)) as f64 / (columns - 1) as f64;
            let r = if i.is_multiple_of(2) { 150.0 } else { 140.0 } + 0.2 * z;
            Point3::new(r * a.cos(), r * a.sin(), z)
        };
        NurbsSurface::new(
            3,
            3,
            knots_u,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            (0..columns)
                .map(|i| (0..4).map(|j| ring(i, 15.0 * j as f64 - 10.0)).collect())
                .collect(),
            vec![vec![1.0; 4]; columns],
        )
        .expect("valid wall")
    }

    /// The coarse grid samples the wavy ring about every fifteenth span,
    /// 120 mm apart, and Newton from there settles in a valley of the
    /// wiggle or fails to settle at all. Every point on the wall must
    /// project back onto it.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn points_on_a_wall_of_many_spans_project_onto_it() {
        let wall = wavy_ring();
        for k in 0..400 {
            let u = 117.0 * (k as f64 + 0.5) / 400.0;
            let v = 0.2 + 0.6 * f64::from(k % 7) / 6.0;
            let on = wall.evaluate(u, v);
            let res = project_point_to_surface(&wall, on, 1e-6).expect("projects");
            assert!(
                res.distance < 1e-5,
                "S({u}, {v}) projects {} away, to ({}, {})",
                res.distance,
                res.u,
                res.v
            );
        }
    }

    /// A point with no position fails, and a tolerance nothing can meet
    /// still returns, rather than searching the spans without end.
    #[test]
    fn a_nan_point_fails_and_a_zero_tolerance_returns() {
        let wall = wavy_ring();
        let nan = Point3::new(f64::NAN, 0.0, 0.0);
        assert!(project_point_to_surface(&wall, nan, 1e-6).is_err());
        let off = Point3::new(70.0, 0.0, 5.0);
        let _ = project_point_to_surface(&wall, off, 0.0);
    }
}
