//! Curve type conversion utilities.
//!
//! Provides geometrically exact conversions from analytic curve
//! types (Line3D, Circle3D, Parabola3D, Hyperbola3D, plus
//! Ellipse3D once #623 lands) to their NURBS representations.
//! These are used by healing operations that need a uniform NURBS
//! representation for fitting or comparison.
//!
//! - `line_to_nurbs`: degree 1, 2 CPs (non-rational).
//! - `circle_to_nurbs`: degree 2, 9 CPs (rational, four quarter-arcs).
//! - `parabola_to_nurbs`: degree 2, 3 CPs (non-rational Bézier).
//! - `hyperbola_to_nurbs`: degree 2, 3 CPs (rational, conic-arc form).

use std::f64::consts::FRAC_PI_4;

use brepkit_math::curves::{Circle3D, Hyperbola3D, Parabola3D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;

use crate::HealError;

/// Convert a line segment to a degree-1 NURBS curve.
///
/// Returns a NURBS with 2 control points, knots `[0, 0, 1, 1]`, and
/// uniform weights. The parameter domain is `[0, 1]`.
///
/// # Errors
///
/// Returns [`HealError`] if the NURBS construction fails (degenerate input).
pub fn line_to_nurbs(start: Point3, end: Point3) -> Result<NurbsCurve, HealError> {
    let curve = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![start, end],
        vec![1.0, 1.0],
    )?;
    Ok(curve)
}

/// Convert a full circle to a degree-2 rational NURBS curve.
///
/// Uses the standard 9-control-point representation with knots at
/// multiples of pi/2. The parameter domain is `[0, 2*pi]`.
///
/// # Algorithm
///
/// The circle is decomposed into four quarter arcs, each represented by
/// 3 control points (on-curve, off-curve, on-curve). The off-curve
/// control points have weight `w = cos(pi/4) = sqrt(2)/2`.
///
/// # Errors
///
/// Returns [`HealError`] if the NURBS construction fails.
#[allow(clippy::too_many_lines)]
pub fn circle_to_nurbs(circle: &Circle3D) -> Result<NurbsCurve, HealError> {
    let center = circle.center();
    let u = circle.u_axis();
    let v = circle.v_axis();
    let r = circle.radius();

    // Weight for off-curve control points: cos(pi/4) = sqrt(2)/2.
    let w = FRAC_PI_4.cos();

    // 9 control points around the circle.
    // On-curve points at angles 0, pi/2, pi, 3pi/2, 2pi.
    // Off-curve points at tangent intersections between consecutive on-curve points.
    let cp = [
        // 0: angle 0
        center + u * r,
        // 1: tangent intersection between angle 0 and pi/2
        center + u * r + v * r,
        // 2: angle pi/2
        center + v * r,
        // 3: tangent intersection between pi/2 and pi
        center + u * (-r) + v * r,
        // 4: angle pi
        center + u * (-r),
        // 5: tangent intersection between pi and 3pi/2
        center + u * (-r) + v * (-r),
        // 6: angle 3pi/2
        center + v * (-r),
        // 7: tangent intersection between 3pi/2 and 2pi
        center + u * r + v * (-r),
        // 8: angle 2pi (same point as angle 0)
        center + u * r,
    ];

    let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];

    // Clamped knot vector with uniform integer spacing.
    // 4 Bezier segments, each spanning one unit.
    // Domain: [0, 4]. To evaluate at angle t, use t' = t * 4 / (2*pi) = 2t/pi.
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];

    let curve = NurbsCurve::new(2, knots, cp.to_vec(), weights)?;
    Ok(curve)
}

/// Convert a parabolic arc (parameter range `[t_min, t_max]`) to a
/// degree-2 NURBS curve.
///
/// A parabola is geometrically exact as a degree-2 *non-rational*
/// Bézier (single segment, 3 CPs, weights all 1). This is the simplest
/// of the analytic-curve conversions:
///
/// - **CP 0** = `parabola.evaluate(t_min)` (start of arc)
/// - **CP 2** = `parabola.evaluate(t_max)` (end of arc)
/// - **CP 1** = tangent intersection of the parabola at the endpoints
///
/// The tangent intersection is at `vertex + axis_dir * (t_min·t_max
/// / 4f) + u_axis * (t_min + t_max)/2`, which lies on both endpoint
/// tangent lines.
///
/// Parameter range must be non-empty (`t_max > t_min`); the parabola
/// is otherwise degenerate.
///
/// # Errors
///
/// Returns [`HealError`] if `t_max <= t_min` or NURBS construction
/// fails.
pub fn parabola_to_nurbs(
    parabola: &Parabola3D,
    t_min: f64,
    t_max: f64,
) -> Result<NurbsCurve, HealError> {
    if t_max <= t_min {
        return Err(brepkit_math::MathError::ParameterOutOfRange {
            value: t_max,
            min: t_min,
            max: f64::INFINITY,
        }
        .into());
    }

    let p0 = parabola.evaluate(t_min);
    let p2 = parabola.evaluate(t_max);
    // Tangent intersection (Bézier middle CP): the tangent lines at
    // t_min and t_max meet at axial = t_min·t_max / (4f), tangential
    // = (t_min + t_max)/2 in the parabola's local (axis_dir, u_axis)
    // frame.
    let f = parabola.focal_length();
    let p1 = parabola.vertex()
        + parabola.axis_dir() * (t_min * t_max / (4.0 * f))
        + parabola.u_axis() * f64::midpoint(t_min, t_max);

    Ok(NurbsCurve::new(
        2,
        vec![t_min, t_min, t_min, t_max, t_max, t_max],
        vec![p0, p1, p2],
        vec![1.0, 1.0, 1.0],
    )?)
}

/// Convert a hyperbolic arc (parameter range `[t_min, t_max]`) to a
/// degree-2 *rational* NURBS curve.
///
/// A hyperbola is geometrically exact as a 3-CP rational Bézier:
/// the standard conic-arc form (Piegl-Tiller §7.4) where the middle
/// CP is the tangent intersection at the endpoints and the middle
/// weight is determined by the half-arc-angle.
///
/// # Algorithm
///
/// In the hyperbola's local `(u_axis, v_axis)` frame, the arc traces
/// `(a·cosh(t), b·sinh(t))` for `t ∈ [t_min, t_max]`. The conic-arc
/// rational form has:
///
/// - **CP 0** = `H(t_min)` (start of arc)
/// - **CP 2** = `H(t_max)` (end of arc)
/// - **CP 1** = tangent intersection at the two endpoints, located at
///   `H(t_min) + tanh(B) · T(t_min)` (in scaled coords), where
///   `B = (t_max - t_min) / 2`. The tangent at parameter `t` is
///   `T(t) = (a·sinh(t), b·cosh(t))`.
/// - **Weights**: `(1, cosh(B), 1)`. The middle weight `> 1` is
///   what distinguishes a hyperbolic arc from a parabolic arc
///   (`w₁ = 1`) or an elliptic arc (`w₁ < 1`).
///
/// # Errors
///
/// Returns [`HealError`] if `t_max <= t_min` or NURBS construction
/// fails.
pub fn hyperbola_to_nurbs(
    hyperbola: &Hyperbola3D,
    t_min: f64,
    t_max: f64,
) -> Result<NurbsCurve, HealError> {
    if t_max <= t_min {
        return Err(brepkit_math::MathError::ParameterOutOfRange {
            value: t_max,
            min: t_min,
            max: f64::INFINITY,
        }
        .into());
    }

    let center = hyperbola.center();
    let u = hyperbola.u_axis();
    let v = hyperbola.v_axis();
    let a = hyperbola.semi_major();
    let b = hyperbola.semi_minor();

    let p0 = hyperbola.evaluate(t_min);
    let p2 = hyperbola.evaluate(t_max);

    // Tangent intersection (Piegl-Tiller conic-arc form). In scaled
    // coords (X = x/a, Y = y/b) on the standard hyperbola
    // X² − Y² = 1, the middle CP is at:
    //     P1 = P0 + tanh(B) · T0
    // where B = (t_max − t_min)/2 and T0 = (sinh(t_min), cosh(t_min)).
    // In unscaled coords (multiplying X by a, Y by b):
    let half = 0.5 * (t_max - t_min);
    let tanh_b = half.tanh();
    let p1_x = a * (t_min.cosh() + tanh_b * t_min.sinh());
    let p1_y = b * (t_min.sinh() + tanh_b * t_min.cosh());
    let p1 = center + u * p1_x + v * p1_y;

    let w1 = half.cosh();

    Ok(NurbsCurve::new(
        2,
        vec![t_min, t_min, t_min, t_max, t_max, t_max],
        vec![p0, p1, p2],
        vec![1.0, w1, 1.0],
    )?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use brepkit_math::vec::Vec3;

    use super::*;

    #[test]
    fn line_to_nurbs_roundtrip() {
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(3.0, 4.0, 0.0);
        let nurbs = line_to_nurbs(start, end).unwrap();

        // Evaluate at t=0 and t=1.
        let p0 = nurbs.evaluate(0.0);
        let p1 = nurbs.evaluate(1.0);
        assert!((p0 - start).length() < 1e-10);
        assert!((p1 - end).length() < 1e-10);

        // Midpoint.
        let mid = nurbs.evaluate(0.5);
        let expected_mid = Point3::new(1.5, 2.0, 0.0);
        assert!((mid - expected_mid).length() < 1e-10);
    }

    #[test]
    fn circle_to_nurbs_roundtrip() {
        let r = 2.0;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let nurbs = circle_to_nurbs(&circle).unwrap();

        // NURBS domain is [0, 4]. At knot values 0, 1, 2, 3, 4 the curve
        // passes through the on-curve control points (angles 0, pi/2, pi,
        // 3pi/2, 2pi). Between knots, the parameterization is nonlinear
        // (rational), but ALL points must lie on the circle.
        for i in 0..=32 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / 32.0 * 4.0;
            let p = nurbs.evaluate(t);
            let dist_from_center = Vec3::new(p.x(), p.y(), p.z()).length();
            assert!(
                (dist_from_center - r).abs() < 1e-6,
                "at t={t:.3}: point={p:?}, dist_from_center={dist_from_center:.6}, expected {r}"
            );
        }

        // Check quarter points are exact.
        let p0 = nurbs.evaluate(0.0);
        let p1 = nurbs.evaluate(1.0);
        let p2 = nurbs.evaluate(2.0);
        let p3 = nurbs.evaluate(3.0);
        assert!((p0 - circle.evaluate(0.0)).length() < 1e-10);
        assert!((p1 - circle.evaluate(std::f64::consts::FRAC_PI_2)).length() < 1e-10);
        assert!((p2 - circle.evaluate(std::f64::consts::PI)).length() < 1e-10);
        assert!((p3 - circle.evaluate(3.0 * std::f64::consts::FRAC_PI_2)).length() < 1e-10);
    }

    #[test]
    fn circle_to_nurbs_has_correct_radius() {
        let center = Point3::new(1.0, 2.0, 3.0);
        let radius = 5.0;
        let circle = Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        let nurbs = circle_to_nurbs(&circle).unwrap();

        // The NURBS domain is [0, 4]. All evaluated points should be exactly
        // `radius` from `center`.
        for i in 0..=20 {
            #[allow(clippy::cast_precision_loss)]
            let t_nurbs = i as f64 / 20.0 * 4.0;
            let p = nurbs.evaluate(t_nurbs);
            let dist_from_center = (p - center).length();
            assert!(
                (dist_from_center - radius).abs() < 1e-6,
                "at t={t_nurbs:.3}: distance from center = {dist_from_center:.6}, expected {radius}"
            );
        }
    }

    #[test]
    fn parabola_to_nurbs_evaluates_exactly_on_parabola() {
        // Every NURBS evaluation must land on the analytic parabola
        // within fp tolerance — degree-2 polynomial Bézier is exact
        // for parabolas.
        let parabola = Parabola3D::new(
            Point3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0_f64,
        )
        .unwrap();
        let nurbs = parabola_to_nurbs(&parabola, -3.0, 5.0).unwrap();

        // Sample the NURBS over its domain [t_min, t_max] and check
        // each point against the analytic parabola at the same
        // parameter.
        let mut max_err = 0.0_f64;
        for k in 0..=32 {
            #[allow(clippy::cast_precision_loss)]
            let t = -3.0 + 8.0 * (k as f64 / 32.0);
            let p_nurbs = nurbs.evaluate(t);
            let p_analytic = parabola.evaluate(t);
            max_err = max_err.max((p_nurbs - p_analytic).length());
        }
        assert!(
            max_err < 1e-9,
            "exact parabola residual {max_err} exceeds 1e-9"
        );
    }

    #[test]
    fn parabola_to_nurbs_endpoints_match() {
        // The first and last CPs must equal P(t_min) and P(t_max).
        let parabola =
            Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let nurbs = parabola_to_nurbs(&parabola, -2.0, 3.0).unwrap();

        let p_start = nurbs.evaluate(-2.0);
        let p_end = nurbs.evaluate(3.0);
        let expected_start = parabola.evaluate(-2.0);
        let expected_end = parabola.evaluate(3.0);

        assert!((p_start - expected_start).length() < 1e-12);
        assert!((p_end - expected_end).length() < 1e-12);
    }

    #[test]
    fn parabola_to_nurbs_rejects_inverted_range() {
        let parabola =
            Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let err = parabola_to_nurbs(&parabola, 5.0, 1.0).unwrap_err();
        match err {
            HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value, min, ..
            }) => {
                assert!((value - 1.0).abs() < 1e-12);
                assert!((min - 5.0).abs() < 1e-12);
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn hyperbola_to_nurbs_evaluates_exactly_on_hyperbola() {
        // Every NURBS evaluation must satisfy the implicit equation
        // (lx/a)² - (ly/b)² = 1 in the hyperbola's local frame.
        let center = Point3::new(2.0, -1.0, 3.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let a = 2.0_f64;
        let b = 1.5_f64;
        let hyp = Hyperbola3D::new(center, normal, a, b).unwrap();
        let t_min = -1.5_f64;
        let t_max = 1.5_f64;
        let nurbs = hyperbola_to_nurbs(&hyp, t_min, t_max).unwrap();
        let u = hyp.u_axis();
        let v = hyp.v_axis();

        let mut max_err = 0.0_f64;
        for k in 0..=64 {
            #[allow(clippy::cast_precision_loss)]
            let t = t_min + (t_max - t_min) * (k as f64 / 64.0);
            let p = nurbs.evaluate(t);
            let off = p - center;
            let lx = off.dot(u) / a;
            let ly = off.dot(v) / b;
            let resid = (lx * lx - ly * ly - 1.0).abs();
            max_err = max_err.max(resid);
        }
        assert!(
            max_err < 1e-9,
            "exact rational hyperbola residual {max_err} exceeds 1e-9"
        );
    }

    #[test]
    fn hyperbola_to_nurbs_endpoints_match() {
        // First and last NURBS evaluations must equal H(t_min) and
        // H(t_max) exactly.
        let hyp = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
        )
        .unwrap();
        let nurbs = hyperbola_to_nurbs(&hyp, -1.0, 1.0).unwrap();
        let p_start = nurbs.evaluate(-1.0);
        let p_end = nurbs.evaluate(1.0);
        let expected_start = hyp.evaluate(-1.0);
        let expected_end = hyp.evaluate(1.0);
        assert!((p_start - expected_start).length() < 1e-12);
        assert!((p_end - expected_end).length() < 1e-12);
    }

    #[test]
    fn hyperbola_to_nurbs_rejects_inverted_range() {
        let hyp = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
        )
        .unwrap();
        let err = hyperbola_to_nurbs(&hyp, 5.0, 1.0).unwrap_err();
        match err {
            HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value, min, ..
            }) => {
                assert!((value - 1.0).abs() < 1e-12);
                assert!((min - 5.0).abs() < 1e-12);
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }
}
