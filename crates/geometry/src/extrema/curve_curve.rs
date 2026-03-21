//! Curve-to-curve minimum distance.
//!
//! Provides analytic fast paths for common curve combinations and a generic
//! fallback that samples both curves and refines with Newton-Raphson.

use brepkit_math::curves::Line3D;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::Point3;

use super::ExtremaSolution;

/// Number of samples per curve in the global search phase.
const N_SAMPLES: usize = 32;

/// Maximum Newton iterations for refinement.
const MAX_ITER: usize = 50;

/// Convergence tolerance on the parameter step magnitude.
const PARAM_TOL: f64 = 1e-10;

// ── Analytic fast path: line-to-line ────────────────────────────────────────

/// Minimum distance between two bounded line segments.
///
/// Uses the closed-form formula for the closest approach of two infinite lines,
/// then clamps parameters to `[t1_range]` and `[t2_range]` respectively.
///
/// Handles parallel and degenerate (point) cases without division by zero.
///
/// # Examples
///
/// ```
/// use brepkit_math::curves::Line3D;
/// use brepkit_math::vec::{Point3, Vec3};
/// use brepkit_geometry::extrema::line_to_line;
///
/// // Two parallel lines offset by 1 in Y.
/// let l1 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
/// let l2 = Line3D::new(Point3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
/// let sol = line_to_line(&l1, (0.0, 10.0), &l2, (0.0, 10.0));
/// assert!((sol.distance - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn line_to_line(
    l1: &Line3D,
    t1_range: (f64, f64),
    l2: &Line3D,
    t2_range: (f64, f64),
) -> ExtremaSolution {
    let d1 = l1.direction();
    let d2 = l2.direction();

    // Vector between origins.
    let r = l1.origin() - l2.origin();

    let a = d1.dot(d1); // |d1|² (= 1 for unit directions)
    let e = d2.dot(d2); // |d2|²
    let f = d2.dot(r);

    let (s, t);
    if a < 1e-30 && e < 1e-30 {
        // Both degenerate to points.
        s = t1_range.0;
        t = t2_range.0;
    } else if a < 1e-30 {
        s = t1_range.0;
        t = (f / e).clamp(t2_range.0, t2_range.1);
    } else {
        let c = d1.dot(r);
        if e < 1e-30 {
            t = t2_range.0;
            s = (-c / a).clamp(t1_range.0, t1_range.1);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            // If lines are parallel (denom ≈ 0), pick s = t1_start.
            let s_raw = if denom.abs() > 1e-30 {
                (b * f - c * e) / denom
            } else {
                t1_range.0
            };
            s = s_raw.clamp(t1_range.0, t1_range.1);

            // Compute t for s, then clamp.
            let t_raw = (b * s + f) / e;
            t = t_raw.clamp(t2_range.0, t2_range.1);
        }
    }

    // Recompute s if t was clamped.
    let s = if a > 1e-30 {
        let b = d1.dot(d2);
        let c = d1.dot(r);
        ((b * t - c) / a).clamp(t1_range.0, t1_range.1)
    } else {
        s
    };

    let pa = l1.evaluate(s);
    let pb = l2.evaluate(t);
    let diff = pa - pb;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();

    ExtremaSolution {
        distance,
        point_a: pa,
        point_b: pb,
        param_a: s,
        param_b: t,
    }
}

// ── Generic curve-to-curve solver ────────────────────────────────────────────

/// Minimum distance between two parametric curves.
///
/// **Algorithm:**
/// 1. Sample both curves at `N_SAMPLES` points and find the closest pair.
/// 2. Refine with Newton-Raphson on the two-variable stationarity system:
///    - `∂f/∂t₁ = dot(C₁(t₁) − C₂(t₂), C₁'(t₁)) = 0`
///    - `∂f/∂t₂ = dot(C₁(t₁) − C₂(t₂), −C₂'(t₂)) = 0`
///
/// Parameters are clamped to their respective domains after each Newton step.
///
/// # Examples
///
/// ```
/// use brepkit_math::curves::Circle3D;
/// use brepkit_math::vec::{Point3, Vec3};
/// use brepkit_geometry::extrema::curve_to_curve;
/// use std::f64::consts::TAU;
///
/// // Two concentric coplanar circles of radius 1 and 2.
/// let c1 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
/// let c2 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
/// let sol = curve_to_curve(&c1, (0.0, TAU), &c2, (0.0, TAU));
/// assert!((sol.distance - 1.0).abs() < 1e-4);
/// ```
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn curve_to_curve<C1: ParametricCurve, C2: ParametricCurve>(
    c1: &C1,
    t1_range: (f64, f64),
    c2: &C2,
    t2_range: (f64, f64),
) -> ExtremaSolution {
    let (t1_start, t1_end) = t1_range;
    let (t2_start, t2_end) = t2_range;

    // Degenerate ranges: evaluate endpoints directly.
    if t1_end <= t1_start || t2_end <= t2_start {
        let p1 = c1.evaluate(t1_start);
        let p2 = c2.evaluate(t2_start);
        return ExtremaSolution {
            distance: (p1 - p2).length(),
            point_a: p1,
            point_b: p2,
            param_a: t1_start,
            param_b: t2_start,
        };
    }

    // ── Phase 1: grid search ──────────────────────────────────────────────────
    let step1 = (t1_end - t1_start) / (N_SAMPLES - 1) as f64;
    let step2 = (t2_end - t2_start) / (N_SAMPLES - 1) as f64;

    let mut best_t1 = t1_start;
    let mut best_t2 = t2_start;
    let mut best_dist_sq = f64::INFINITY;

    // Collect sample points to avoid re-evaluation.
    let samples1: Vec<(f64, Point3)> = (0..N_SAMPLES)
        .map(|i| {
            let t = if i == N_SAMPLES - 1 {
                t1_end
            } else {
                t1_start + i as f64 * step1
            };
            (t, c1.evaluate(t))
        })
        .collect();

    let samples2: Vec<(f64, Point3)> = (0..N_SAMPLES)
        .map(|i| {
            let t = if i == N_SAMPLES - 1 {
                t2_end
            } else {
                t2_start + i as f64 * step2
            };
            (t, c2.evaluate(t))
        })
        .collect();

    for (t1, p1) in &samples1 {
        for (t2, p2) in &samples2 {
            let diff = *p1 - *p2;
            let d2 = diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z();
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_t1 = *t1;
                best_t2 = *t2;
            }
        }
    }

    // ── Phase 2: Newton-Raphson refinement ────────────────────────────────────
    // Variables: x = [t1, t2]
    // f1(t1,t2) = dot(C1(t1) - C2(t2),  C1'(t1)) = 0
    // f2(t1,t2) = dot(C1(t1) - C2(t2), -C2'(t2)) = 0
    //
    // Jacobian (Gauss-Newton approximation):
    //   J11 ≈  |C1'(t1)|²,  J12 = -dot(C1'(t1), C2'(t2))
    //   J21 = -dot(C1'(t1), C2'(t2)),  J22 ≈ |C2'(t2)|²
    let h1 = ((t1_end - t1_start) * 1e-6).max(1e-9);
    let h2 = ((t2_end - t2_start) * 1e-6).max(1e-9);

    let mut t1 = best_t1;
    let mut t2 = best_t2;

    for _ in 0..MAX_ITER {
        let p1 = c1.evaluate(t1);
        let p2 = c2.evaluate(t2);
        let diff = p1 - p2;

        // Finite-difference velocities.
        let t1_fwd = (t1 + h1).min(t1_end);
        let t1_bwd = (t1 - h1).max(t1_start);
        let inv2h1 = 1.0 / (t1_fwd - t1_bwd);
        let p1f = c1.evaluate(t1_fwd);
        let p1b = c1.evaluate(t1_bwd);
        let vel1 = brepkit_math::vec::Vec3::new(
            (p1f.x() - p1b.x()) * inv2h1,
            (p1f.y() - p1b.y()) * inv2h1,
            (p1f.z() - p1b.z()) * inv2h1,
        );

        let t2_fwd = (t2 + h2).min(t2_end);
        let t2_bwd = (t2 - h2).max(t2_start);
        let inv2h2 = 1.0 / (t2_fwd - t2_bwd);
        let p2f = c2.evaluate(t2_fwd);
        let p2b = c2.evaluate(t2_bwd);
        let vel2 = brepkit_math::vec::Vec3::new(
            (p2f.x() - p2b.x()) * inv2h2,
            (p2f.y() - p2b.y()) * inv2h2,
            (p2f.z() - p2b.z()) * inv2h2,
        );

        let f1 = diff.dot(vel1);
        let f2 = -diff.dot(vel2);

        // Jacobian entries.
        let j11 = vel1.dot(vel1);
        let j12 = -vel1.dot(vel2);
        let j22 = vel2.dot(vel2);
        let det = j11 * j22 - j12 * j12;

        if det.abs() < f64::EPSILON {
            break;
        }

        let dt1 = (f1 * j22 - f2 * j12) / det;
        let dt2 = (f2 * j11 - f1 * j12) / det;

        let t1_new = (t1 - dt1).clamp(t1_start, t1_end);
        let t2_new = (t2 - dt2).clamp(t2_start, t2_end);

        if (t1_new - t1).abs() < PARAM_TOL && (t2_new - t2).abs() < PARAM_TOL {
            t1 = t1_new;
            t2 = t2_new;
            break;
        }
        t1 = t1_new;
        t2 = t2_new;
    }

    let pa = c1.evaluate(t1);
    let pb = c2.evaluate(t2);
    let diff = pa - pb;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();

    ExtremaSolution {
        distance,
        point_a: pa,
        point_b: pb,
        param_a: t1,
        param_b: t2,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use std::f64::consts::TAU;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── line_to_line ─────────────────────────────────────────────────────────

    #[test]
    fn parallel_lines_unit_separation() {
        // Two parallel X-axis lines, y=0 and y=1 — distance = 1.
        let l1 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let l2 = Line3D::new(Point3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let sol = line_to_line(&l1, (0.0, 10.0), &l2, (0.0, 10.0));
        assert!(approx(sol.distance, 1.0, 1e-12), "dist={}", sol.distance);
    }

    #[test]
    fn perpendicular_skew_lines_closest_approach() {
        // L1 along X at z=0, L2 along Y at z=1.
        // Closest approach: (0,0,0) on L1 and (0,0,1) on L2 → distance = 1.
        let l1 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let l2 = Line3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        let sol = line_to_line(&l1, (-5.0, 5.0), &l2, (-5.0, 5.0));
        assert!(approx(sol.distance, 1.0, 1e-12), "dist={}", sol.distance);
        assert!(
            approx(sol.point_a.x(), 0.0, 1e-12),
            "pa.x={}",
            sol.point_a.x()
        );
        assert!(
            approx(sol.point_b.y(), 0.0, 1e-12),
            "pb.y={}",
            sol.point_b.y()
        );
    }

    #[test]
    fn intersecting_lines_zero_distance() {
        // Two lines that cross at origin.
        let l1 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let l2 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        let sol = line_to_line(&l1, (-5.0, 5.0), &l2, (-5.0, 5.0));
        assert!(sol.distance < 1e-12, "dist={}", sol.distance);
    }

    #[test]
    fn line_clamped_to_domain_endpoint() {
        // L1 along X, capped at t=[0,2]; L2 along Y from (5,1,0).
        // Closest on L1 is endpoint (2,0,0); L2 foot for (2,0,0):
        //   project (2,0,0) onto L2: dot((2-5, 0-1, 0), (0,1,0)) = -1 → pb=(5,0,0).
        // distance = |(2,0,0) - (5,0,0)| = 3.
        let l1 = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let l2 = Line3D::new(Point3::new(5.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap();
        let sol = line_to_line(&l1, (0.0, 2.0), &l2, (-5.0, 5.0));
        assert!(
            approx(sol.distance, 3.0, 1e-12),
            "dist={} expected=3.0",
            sol.distance
        );
        assert!(approx(sol.param_a, 2.0, 1e-10), "t1={}", sol.param_a);
    }

    // ── curve_to_curve (generic) ─────────────────────────────────────────────

    #[test]
    fn generic_circle_to_circle_concentric_coplanar() {
        // Concentric circles of radius 1 and 2 in the XY plane.
        // Minimum distance = 2 - 1 = 1.
        let c1 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let c2 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let sol = curve_to_curve(&c1, (0.0, TAU), &c2, (0.0, TAU));
        assert!(approx(sol.distance, 1.0, 1e-4), "dist={}", sol.distance);
    }

    #[test]
    fn generic_circles_axially_offset() {
        // Circle radius 1 at z=0 and circle radius 1 at z=3 (same axis).
        // Minimum distance = 3 (straight across).
        let c1 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let c2 = Circle3D::new(Point3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let sol = curve_to_curve(&c1, (0.0, TAU), &c2, (0.0, TAU));
        assert!(approx(sol.distance, 3.0, 1e-4), "dist={}", sol.distance);
    }

    #[test]
    fn closest_points_stationarity_condition() {
        // For two circles at different heights, the closest points should satisfy:
        // dot(C1(t1) - C2(t2), C1'(t1)) ≈ 0.
        let c1 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let c2 = Circle3D::new(Point3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let sol = curve_to_curve(&c1, (0.0, TAU), &c2, (0.0, TAU));

        let diff = sol.point_a - sol.point_b;
        let tan1 = c1.tangent(sol.param_a);
        let dot = diff.dot(tan1);
        assert!(dot.abs() < 1e-4, "stationarity: dot={dot}");
    }
}
