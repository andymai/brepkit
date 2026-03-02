//! Point projection onto NURBS curves and surfaces.
//!
//! Finds the closest point on a curve or surface to a given point in space.
//! Used for Boolean classification, snapping, distance queries, and
//! tessellation refinement.
//!
//! Algorithms follow NURBS Book A6.1–A6.6: subdivision for initial guess
//! followed by Newton–Raphson refinement.

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::curve_to_bezier_segments;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

/// Maximum Newton iterations before declaring convergence failure.
const MAX_ITERATIONS: usize = 50;

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
/// Uses Bezier decomposition for initial guess, then Newton–Raphson
/// refinement (NURBS Book A6.1 + A6.3–A6.4).
///
/// # Errors
///
/// Returns an error if Bezier decomposition fails (invalid curve data).
pub fn project_point_to_curve(
    curve: &NurbsCurve,
    point: Point3,
    tolerance: f64,
) -> Result<CurveProjection, MathError> {
    let knots = curve.knots();
    let p = curve.degree();
    let u_min = knots[p];
    let u_max = knots[knots.len() - p - 1];

    let candidates = curve_coarse_search(curve, point)?;

    // Run Newton from each candidate and keep the globally closest result.
    let mut best_u = u_min;
    let mut best_pt = curve.evaluate(u_min);
    let mut best_dist = (best_pt - point).length();

    for u_guess in candidates {
        let (u_refined, pt_refined) =
            curve_newton_refine(curve, point, u_guess, u_min, u_max, tolerance);
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

/// Coarse search: decompose into Bezier segments and sample points to find
/// multiple candidate parameter values for Newton refinement.
///
/// Returns a sorted list of candidate parameters (best first) to use as
/// Newton seeds. Using multiple seeds avoids converging to a local minimum.
#[allow(clippy::cast_precision_loss)]
fn curve_coarse_search(curve: &NurbsCurve, point: Point3) -> Result<Vec<f64>, MathError> {
    let segments = curve_to_bezier_segments(curve)?;

    // Collect all (distance_sq, parameter) samples.
    let mut samples: Vec<(f64, f64)> = Vec::new();

    for seg in &segments {
        let knots = seg.knots();
        let p = seg.degree();
        let u_start = knots[p];
        let u_end = knots[knots.len() - p - 1];

        // Sample points along the segment.
        let n_samples = (p + 1).max(5) * 2;
        for i in 0..=n_samples {
            let t = i as f64 / n_samples as f64;
            let u = t.mul_add(u_end - u_start, u_start);
            let pt = seg.evaluate(u);
            let d_sq = (pt - point).length_squared();
            samples.push((d_sq, u));
        }
    }

    // Sort by distance and return the best candidates.
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take the top few unique candidates (spatially separated).
    let mut candidates = Vec::new();
    let max_candidates = 5;
    for &(_, u) in &samples {
        if candidates.len() >= max_candidates {
            break;
        }
        // Skip candidates too close to one we already have.
        let dominated = candidates.iter().any(|&c: &f64| (c - u).abs() < 1e-10);
        if !dominated {
            candidates.push(u);
        }
    }

    Ok(candidates)
}

/// Newton–Raphson refinement for curve point projection.
///
/// Finds parameter u that minimizes ||C(u) - P|| starting from `u_init`.
/// Always returns a result — falls back to the best iterate if formal
/// convergence criteria are not met within [`MAX_ITERATIONS`].
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
    let mut best_u = u;
    let mut best_dist_sq = f64::INFINITY;

    for _ in 0..MAX_ITERATIONS {
        let ders = curve.derivatives(u, 2);
        let c_pt = Point3::new(ders[0].x(), ders[0].y(), ders[0].z());
        let c_prime = ders[1]; // C'(u)
        let c_double_prime = ders[2]; // C''(u)
        let diff = c_pt - point; // C(u) - P

        let dist_sq = diff.length_squared();

        // Track best solution.
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_u = u;
        }

        // Convergence check 1: point coincidence.
        if dist_sq < tol_sq {
            return (u, c_pt);
        }

        // f(u) = C'(u) · (C(u) - P)
        let f_val = c_prime.dot(diff);

        // Convergence check 2: zero cosine (perpendicularity).
        // cos²(angle) = (C'·diff)² / (|C'|² · |diff|²) < tol²
        let c_prime_len_sq = c_prime.length_squared();
        if c_prime_len_sq > 1e-30 && dist_sq > tol_sq {
            let cos_sq = (f_val * f_val) / (c_prime_len_sq * dist_sq);
            if cos_sq < tol_sq {
                return (u, c_pt);
            }
        }

        // f'(u) = C''(u) · (C(u) - P) + |C'(u)|²
        let f_prime = c_double_prime.dot(diff) + c_prime_len_sq;

        // Guard against zero denominator.
        if f_prime.abs() < 1e-30 {
            return (u, c_pt);
        }

        let delta_u = f_val / f_prime;
        let u_new = (u - delta_u).clamp(u_min, u_max);

        // Guard NaN.
        if u_new.is_nan() {
            break;
        }

        // Convergence check 3: parameter step negligible.
        let du = (u_new - u).abs();
        if du < tolerance * (1.0 + u.abs()) {
            let pt = curve.evaluate(u_new);
            return (u_new, pt);
        }

        u = u_new;
    }

    // Return the best point found during iteration.
    let pt = curve.evaluate(best_u);
    (best_u, pt)
}

// ---------------------------------------------------------------------------
// Surface projection
// ---------------------------------------------------------------------------

/// Find the closest point on a NURBS surface to the given point.
///
/// Uses grid evaluation for initial guess, then 2D Newton–Raphson
/// refinement (NURBS Book A6.2 + A6.5–A6.6).
///
/// # Errors
///
/// Returns [`MathError::ConvergenceFailure`] if Newton iteration does not
/// converge within the maximum number of iterations.
pub fn project_point_to_surface(
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

    let (u_final, v_final, pt_final) = surface_newton_refine(
        surface, point, u_guess, v_guess, u_min, u_max, v_min, v_max, tolerance,
    )?;
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
) -> Result<(f64, f64, Point3), MathError> {
    let mut u = u_init;
    let mut v = v_init;

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

        // Build the 2×2 Jacobian and right-hand side.
        // J = [S_u · S_u,  S_u · S_v]
        //     [S_v · S_u,  S_v · S_v]
        let j00 = deriv_u.dot(deriv_u);
        let j01 = deriv_u.dot(deriv_v);
        let j11 = deriv_v.dot(deriv_v);
        // rhs = [-S_u · r, -S_v · r]
        let rhs0 = -dot_du_r;
        let rhs1 = -dot_dv_r;

        // Solve 2×2 system via Cramer's rule: det = j00*j11 - j01²
        let det = j00.mul_add(j11, -(j01 * j01));
        if det.abs() < 1e-30 {
            // Singular Jacobian — can't improve further.
            return Ok((u, v, s_pt));
        }

        let delta_u = rhs0.mul_add(j11, -(rhs1 * j01)) / det;
        let delta_v = j00.mul_add(rhs1, -(j01 * rhs0)) / det;

        let u_new = (u + delta_u).clamp(u_min, u_max);
        let v_new = (v + delta_v).clamp(v_min, v_max);

        // Convergence check 3: parameter step negligible.
        let step = (deriv_u * (u_new - u) + deriv_v * (v_new - v)).length();
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

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
}
