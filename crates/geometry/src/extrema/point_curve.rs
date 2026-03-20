//! Point-to-curve projection algorithms.
//!
//! Provides analytic fast paths for common curve types ([`Line3D`], [`Circle3D`])
//! and a generic Newton-Raphson solver for any [`ParametricCurve`].

use brepkit_math::curves::{Circle3D, Line3D};
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::Point3;

use super::CurveProjection;

/// Maximum iterations for Newton-Raphson refinement.
const MAX_ITER: usize = 50;

/// Convergence tolerance on the parameter step.
const PARAM_TOL: f64 = 1e-10;

/// Number of uniform samples used in the global search phase.
const N_SAMPLES: usize = 64;

// ── Analytic fast paths ──────────────────────────────────────────────────────

/// Project a point onto a bounded line segment, clamped to `[t_start, t_end]`.
///
/// The line is parameterized as `P(t) = origin + t * direction`, where
/// `direction` is a unit vector. The unconstrained closest parameter is the
/// orthogonal projection; clamping handles the finite-segment case.
///
/// # Examples
///
/// ```
/// use brepkit_math::curves::Line3D;
/// use brepkit_math::vec::{Point3, Vec3};
/// use brepkit_geometry::extrema::point_to_line;
///
/// let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
/// let proj = point_to_line(Point3::new(3.0, 2.0, 0.0), &line, 0.0, 10.0);
/// assert!((proj.parameter - 3.0).abs() < 1e-12);
/// assert!((proj.distance - 2.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn point_to_line(point: Point3, line: &Line3D, t_start: f64, t_end: f64) -> CurveProjection {
    // Unconstrained projection: t = dot(point - origin, direction)
    let t_unclamped = line.project(point);
    let t = t_unclamped.clamp(t_start, t_end);
    let closest = line.evaluate(t);
    let diff = closest - point;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();
    CurveProjection {
        distance,
        point: closest,
        parameter: t,
    }
}

/// Project a point onto a full circle (closed, periodic).
///
/// The circle is parameterized as `P(t) = center + r*(cos(t)*u + sin(t)*v)`.
/// This uses the analytic projection: project the point into the circle's plane,
/// compute `atan2`, then evaluate.
///
/// # Examples
///
/// ```
/// use brepkit_math::curves::Circle3D;
/// use brepkit_math::vec::{Point3, Vec3};
/// use brepkit_geometry::extrema::point_to_circle;
///
/// let circle = Circle3D::new(
///     Point3::new(0.0, 0.0, 0.0),
///     Vec3::new(0.0, 0.0, 1.0),
///     1.0,
/// ).unwrap();
/// let proj = point_to_circle(Point3::new(0.0, 5.0, 0.0), &circle);
/// assert!((proj.distance - 4.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn point_to_circle(point: Point3, circle: &Circle3D) -> CurveProjection {
    let t = circle.project(point);
    let closest = circle.evaluate(t);
    let diff = closest - point;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();
    CurveProjection {
        distance,
        point: closest,
        parameter: t,
    }
}

// ── Generic Newton-Raphson solver ────────────────────────────────────────────

/// Project a point onto a parametric curve over `[t_start, t_end]`.
///
/// **Algorithm:**
/// 1. Sample the curve at [`N_SAMPLES`] uniformly-spaced parameters to find the
///    global closest sample (avoids local-minimum traps on non-convex curves).
/// 2. Refine the best sample using Newton-Raphson on the stationarity condition
///    `dot(C(t) - P, C'(t)) = 0`, clamping `t` to `[t_start, t_end]` after
///    each step.
///
/// Convergence is declared when the parameter update `|Δt|` drops below
/// `1e-10`, or after 50 iterations (whichever comes first).
///
/// # Examples
///
/// ```
/// use brepkit_math::curves::Circle3D;
/// use brepkit_math::vec::{Point3, Vec3};
/// use brepkit_geometry::extrema::point_to_curve;
/// use std::f64::consts::TAU;
///
/// let circle = Circle3D::new(
///     Point3::new(0.0, 0.0, 0.0),
///     Vec3::new(0.0, 0.0, 1.0),
///     2.0,
/// ).unwrap();
/// let proj = point_to_curve(Point3::new(0.0, 10.0, 0.0), &circle, 0.0, TAU);
/// assert!(proj.distance < 9.0); // closest point on circle is ≤ 8.0 from query
/// ```
#[must_use]
pub fn point_to_curve<C: ParametricCurve>(
    point: Point3,
    curve: &C,
    t_start: f64,
    t_end: f64,
) -> CurveProjection {
    // Degenerate range: evaluate the single point.
    if t_end <= t_start {
        let p = curve.evaluate(t_start);
        return CurveProjection {
            distance: (p - point).length(),
            point: p,
            parameter: t_start,
        };
    }

    // ── Phase 1: global search ───────────────────────────────────────────────
    let step = (t_end - t_start) / (N_SAMPLES - 1) as f64;
    let mut best_t = t_start;
    let mut best_dist_sq = f64::INFINITY;

    for i in 0..N_SAMPLES {
        let t = if i == N_SAMPLES - 1 {
            t_end
        } else {
            t_start + i as f64 * step
        };
        let p = curve.evaluate(t);
        let diff = p - point;
        let d2 = diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z();
        if d2 < best_dist_sq {
            best_dist_sq = d2;
            best_t = t;
        }
    }

    // ── Phase 2: Newton-Raphson refinement ───────────────────────────────────
    // We solve f(t) = dot(C(t) - P, C'(t)) = 0 where C'(t) is the actual
    // velocity (not necessarily unit-length).
    //
    // The `ParametricCurve::tangent` method returns a *unit* tangent, so we
    // recover the velocity magnitude via finite differences:
    //   C'(t) ≈ (C(t+h) - C(t-h)) / (2h)
    //
    // Newton step: Δt = f / f', where
    //   f  = dot(C(t)-P, velocity(t))
    //   f' ≈ dot(velocity(t), velocity(t))   [Gauss-Newton, drops curvature term]
    let h = (t_end - t_start) * 1e-6;
    let h = h.max(1e-9);
    let mut t = best_t;
    for _ in 0..MAX_ITER {
        let p = curve.evaluate(t);
        let diff = p - point;

        // Finite-difference velocity (actual C'(t), not unit-normalised).
        let t_fwd = (t + h).min(t_end);
        let t_bwd = (t - h).max(t_start);
        let p_fwd = curve.evaluate(t_fwd);
        let p_bwd = curve.evaluate(t_bwd);
        let inv2h = 1.0 / (t_fwd - t_bwd);
        let vel_x = (p_fwd.x() - p_bwd.x()) * inv2h;
        let vel_y = (p_fwd.y() - p_bwd.y()) * inv2h;
        let vel_z = (p_fwd.z() - p_bwd.z()) * inv2h;

        // f(t) = dot(C(t) - P, C'(t))
        let f = diff.x() * vel_x + diff.y() * vel_y + diff.z() * vel_z;

        // f'(t) ≈ |C'(t)|^2 (Gauss-Newton approximation).
        let vel_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
        if vel_sq < f64::EPSILON {
            break;
        }

        let delta = f / vel_sq;
        let t_new = (t - delta).clamp(t_start, t_end);

        if (t_new - t).abs() < PARAM_TOL {
            t = t_new;
            break;
        }
        t = t_new;
    }

    let closest = curve.evaluate(t);
    let diff = closest - point;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();
    CurveProjection {
        distance,
        point: closest,
        parameter: t,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Vec3;
    use std::f64::consts::{PI, TAU};

    // ── point_to_line ────────────────────────────────────────────────────────

    #[test]
    fn line_point_above_midpoint() {
        // Line along X-axis from 0 to 10; query point directly above x=5.
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let proj = point_to_line(Point3::new(5.0, 3.0, 0.0), &line, 0.0, 10.0);
        assert!(
            (proj.parameter - 5.0).abs() < 1e-12,
            "param={}",
            proj.parameter
        );
        assert!(
            (proj.distance - 3.0).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
        assert!((proj.point.x() - 5.0).abs() < 1e-12);
        assert!((proj.point.y()).abs() < 1e-12);
    }

    #[test]
    fn line_point_before_start_clamps() {
        // Query point is "behind" t=0; should clamp to t_start.
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let proj = point_to_line(Point3::new(-5.0, 0.0, 0.0), &line, 0.0, 10.0);
        assert!(
            (proj.parameter - 0.0).abs() < 1e-12,
            "param={}",
            proj.parameter
        );
        assert!(
            (proj.distance - 5.0).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
    }

    #[test]
    fn line_point_past_end_clamps() {
        // Query point is past t_end; should clamp to t_end.
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let proj = point_to_line(Point3::new(15.0, 0.0, 0.0), &line, 0.0, 10.0);
        assert!(
            (proj.parameter - 10.0).abs() < 1e-12,
            "param={}",
            proj.parameter
        );
        assert!(
            (proj.distance - 5.0).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
    }

    #[test]
    fn line_point_on_line_zero_distance() {
        let line = Line3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0)).unwrap();
        let proj = point_to_line(Point3::new(1.0, 2.0, 5.0), &line, 0.0, 20.0);
        assert!(proj.distance < 1e-12, "dist={}", proj.distance);
        assert!(
            (proj.parameter - 2.0).abs() < 1e-12,
            "param={}",
            proj.parameter
        );
    }

    // ── point_to_circle ──────────────────────────────────────────────────────

    #[test]
    fn circle_point_on_positive_y_axis() {
        // Unit circle in XY plane; point at (0, 5, 0).
        // The analytic projection must produce the closest point on the circle,
        // which lies in the direction of (0,5,0) from the center.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let q = Point3::new(0.0, 5.0, 0.0);
        let proj = point_to_circle(q, &circle);
        // Distance must be 5 - 1 = 4.
        assert!(
            (proj.distance - 4.0).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
        // Closest point must lie on the circle (distance from center = 1).
        let r = (proj.point.x() * proj.point.x()
            + proj.point.y() * proj.point.y()
            + proj.point.z() * proj.point.z())
        .sqrt();
        assert!((r - 1.0).abs() < 1e-12, "closest not on circle: r={r}");
        // Closest point must be collinear with center and query point in-plane.
        // Because q is in the XY plane and the circle is in XY, closest must
        // point in the direction of q: x≈0, y≈1, z≈0.
        assert!(proj.point.x().abs() < 1e-12, "x={}", proj.point.x());
        assert!((proj.point.y() - 1.0).abs() < 1e-12, "y={}", proj.point.y());
    }

    #[test]
    fn circle_point_on_axis_distance_equals_radius() {
        // Point on the circle's axis is equidistant from all points;
        // distance should be the radius.
        let r = 3.0_f64;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        // Point directly on the axis (center), offset in Z.
        let proj = point_to_circle(Point3::new(0.0, 0.0, 5.0), &circle);
        // The closest point is anywhere on the circle; distance is sqrt(r^2 + 25).
        let expected = (r * r + 25.0_f64).sqrt();
        assert!(
            (proj.distance - expected).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
    }

    #[test]
    fn circle_point_in_plane() {
        // Circle of radius 2 in XY; query point at (3, 0, 0).
        // Closest circle point must be on the circle in the direction of (3,0,0),
        // i.e. at (2,0,0), with distance = 1.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let proj = point_to_circle(Point3::new(3.0, 0.0, 0.0), &circle);
        assert!(
            (proj.distance - 1.0).abs() < 1e-12,
            "dist={}",
            proj.distance
        );
        // Closest point must lie on the circle.
        let r = (proj.point.x() * proj.point.x()
            + proj.point.y() * proj.point.y()
            + proj.point.z() * proj.point.z())
        .sqrt();
        assert!((r - 2.0).abs() < 1e-12, "closest not on circle: r={r}");
        // Closest point must be at (2, 0, 0).
        assert!((proj.point.x() - 2.0).abs() < 1e-12, "x={}", proj.point.x());
        assert!(proj.point.y().abs() < 1e-12, "y={}", proj.point.y());
    }

    // ── point_to_curve (generic) ─────────────────────────────────────────────

    #[test]
    fn generic_circle_matches_analytic() {
        // Using the generic solver on a circle should match the analytic result.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let q = Point3::new(0.0, 5.0, 0.0);

        let analytic = point_to_circle(q, &circle);
        let generic = point_to_curve(q, &circle, 0.0, TAU);

        assert!(
            (analytic.distance - generic.distance).abs() < 1e-6,
            "analytic={} generic={}",
            analytic.distance,
            generic.distance
        );
        assert!(
            (analytic.parameter - generic.parameter).abs() < 1e-6,
            "analytic_t={} generic_t={}",
            analytic.parameter,
            generic.parameter
        );
    }

    #[test]
    fn generic_stationarity_condition_satisfied() {
        // After projection, dot(C(t)-P, C'(t)) must be near zero
        // (Karush-Kuhn-Tucker stationarity for interior minimizers).
        // We check using the analytic circle tangent scaled by radius.
        let r = 5.0_f64;
        let circle =
            Circle3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let q = Point3::new(3.0, 7.0, 3.0);
        let proj = point_to_curve(q, &circle, 0.0, TAU);

        let p = circle.evaluate(proj.parameter);
        // The actual velocity C'(t) for a circle has magnitude r; the unit
        // tangent returned by ParametricCurve::tangent is C'(t)/r.
        // We check dot(C(t)-P, unit_tangent) ≈ 0 (equivalent, just scaled by r).
        let tan = circle.tangent(proj.parameter);
        let diff = p - q;
        let dot = diff.x() * tan.x() + diff.y() * tan.y() + diff.z() * tan.z();
        assert!(dot.abs() < 1e-6, "stationarity violated: dot={dot}");
    }

    #[test]
    fn generic_bounded_domain_clamping() {
        // Half-circle domain [0, π]; query point at t = 3π/2 (outside domain).
        // The closest point on [0, π] should be an endpoint.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        // t = 3π/2 evaluates to (0, -1, 0).
        let q = circle.evaluate(3.0 * PI / 2.0);
        let proj = point_to_curve(q, &circle, 0.0, PI);
        // The closest endpoint should be at t = 0 or t = π (symmetric, both at distance √2).
        assert!(
            proj.parameter < 1e-10 || (proj.parameter - PI).abs() < 1e-10,
            "expected endpoint, got t={}",
            proj.parameter
        );
        assert!(
            (proj.distance - 2.0_f64.sqrt()).abs() < 1e-6,
            "dist={}",
            proj.distance
        );
    }
}
