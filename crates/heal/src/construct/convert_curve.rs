//! Curve type conversion utilities.
//!
//! Provides conversions between analytic curve types (Line3D, Circle3D) and
//! their NURBS representations. These are used by healing operations that
//! need a uniform NURBS representation for fitting or comparison.

use std::f64::consts::FRAC_PI_4;

use brepkit_math::curves::Circle3D;
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
}
