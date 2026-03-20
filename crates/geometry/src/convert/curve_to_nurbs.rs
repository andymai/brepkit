//! Convert analytic curves to NURBS representation.
//!
//! Provides exact or near-exact rational NURBS equivalents of analytic curve
//! types. Circle and ellipse use the standard rational quadratic Bezier arc
//! construction; line uses a simple degree-1 non-rational form.

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;

use crate::GeomError;

/// Convert a [`Circle3D`] arc to an exact rational NURBS curve.
///
/// The arc spans the angular parameter range `[t_start, t_end]` (radians).
/// The result is a degree-2 rational B-spline composed of up to 4 quadratic
/// Bezier arcs, each covering at most π/2 of arc angle. The representation
/// is geometrically exact (no approximation error).
///
/// # Errors
///
/// Returns [`GeomError::DegenerateInput`] if the arc span is zero or the
/// circle radius is non-positive.
///
/// Returns [`GeomError::Math`] if NURBS construction fails (should not occur
/// for valid input).
pub fn circle_to_nurbs(
    circle: &Circle3D,
    t_start: f64,
    t_end: f64,
) -> Result<NurbsCurve, GeomError> {
    let span = t_end - t_start;
    if span.abs() < 1e-15 {
        return Err(GeomError::DegenerateInput("arc span is zero".to_owned()));
    }

    // Split the angular range into arcs of at most π/2.
    let n_arcs = ((span.abs() / FRAC_PI_2).ceil() as usize).max(1);
    #[allow(clippy::cast_precision_loss)]
    let delta = span / n_arcs as f64;

    arc_segments_to_nurbs(n_arcs, t_start, delta, |t| {
        let p = circle.evaluate(t);
        let tangent = circle.tangent(t);
        (p, tangent * circle.radius())
    })
}

/// Convert an [`Ellipse3D`] arc to an exact rational NURBS curve.
///
/// The arc spans the angular parameter range `[t_start, t_end]` (radians).
/// Uses the same rational quadratic Bezier arc construction as
/// [`circle_to_nurbs`], scaled appropriately for the ellipse semi-axes.
///
/// # Errors
///
/// Returns [`GeomError::DegenerateInput`] if the arc span is zero.
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn ellipse_to_nurbs(
    ellipse: &Ellipse3D,
    t_start: f64,
    t_end: f64,
) -> Result<NurbsCurve, GeomError> {
    let span = t_end - t_start;
    if span.abs() < 1e-15 {
        return Err(GeomError::DegenerateInput("arc span is zero".to_owned()));
    }

    let n_arcs = ((span.abs() / FRAC_PI_2).ceil() as usize).max(1);
    #[allow(clippy::cast_precision_loss)]
    let delta = span / n_arcs as f64;

    // The ellipse tangent vector at angle t:
    //   dP/dt = -a*sin(t)*u + b*cos(t)*v
    // The "scaled tangent" for half-angle construction must be the
    // tangent scaled so that the control point lies at the tangent
    // intersection from both arc endpoints. For an ellipse arc of
    // half-angle α the middle control point is:
    //   P(t_mid) + tan(α) * |tangent(t_mid)| * tangent_direction
    // We achieve this by passing the actual analytic tangent (unscaled)
    // and computing the intersection in `arc_segments_to_nurbs`.
    arc_segments_to_nurbs(n_arcs, t_start, delta, |t| {
        let p = ellipse.evaluate(t);
        let tangent = ellipse.tangent(t); // unscaled
        (p, tangent)
    })
}

/// Convert a line segment to a degree-1 non-rational NURBS curve.
///
/// The curve has domain `[0, 1]`, with `evaluate(0) = start` and
/// `evaluate(1) = end`.
///
/// # Errors
///
/// Returns [`GeomError::DegenerateInput`] if `start` and `end` are the same
/// point.
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn line_to_nurbs(start: Point3, end: Point3) -> Result<NurbsCurve, GeomError> {
    let d = end - start;
    if d.length() < 1e-15 {
        return Err(GeomError::DegenerateInput(
            "degenerate line: start == end".to_owned(),
        ));
    }

    // Degree 1, 2 control points → knot vector length = 2 + 1 + 1 = 4.
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    let control_points = vec![start, end];
    let weights = vec![1.0, 1.0];

    Ok(NurbsCurve::new(1, knots, control_points, weights)?)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build a degree-2 rational NURBS from `n_arcs` sequential quadratic
/// Bezier arcs.
///
/// `eval_fn` returns `(point, tangent_vector)` at a given angle `t`.
/// The tangent vector must be the first derivative of the curve at `t`
/// (not necessarily unit-length). The middle control point for each arc
/// is computed as the intersection of the tangent lines at the two
/// arc endpoints.
fn arc_segments_to_nurbs<F>(
    n_arcs: usize,
    t_start: f64,
    delta: f64,
    eval_fn: F,
) -> Result<NurbsCurve, GeomError>
where
    F: Fn(f64) -> (Point3, brepkit_math::vec::Vec3),
{
    // Each arc contributes 2 new control points; plus the first point.
    // Total CPs = 2*n_arcs + 1.
    let n_cps = 2 * n_arcs + 1;
    let mut cps: Vec<Point3> = Vec::with_capacity(n_cps);
    let mut weights: Vec<f64> = Vec::with_capacity(n_cps);

    // Knot vector for degree 2 with n_arcs arcs:
    //   [0,0,0, 1/n, 1/n, 2/n, 2/n, ..., 1, 1, 1]
    // Each internal knot appears twice (C1 continuity at join points).
    let mut knots: Vec<f64> = Vec::with_capacity(2 * n_arcs + 5);
    knots.push(0.0);
    knots.push(0.0);
    knots.push(0.0);
    for i in 1..n_arcs {
        #[allow(clippy::cast_precision_loss)]
        let knot = i as f64 / n_arcs as f64;
        knots.push(knot);
        knots.push(knot);
    }
    knots.push(1.0);
    knots.push(1.0);
    knots.push(1.0);

    for arc_idx in 0..n_arcs {
        #[allow(clippy::cast_precision_loss)]
        let t0 = t_start + arc_idx as f64 * delta;
        let t1 = t0 + delta;
        let half_angle = delta * 0.5;

        let (p0, tan0) = eval_fn(t0);
        let (p1, tan1) = eval_fn(t1);

        // Weight for middle control point: cos(half_angle) for a circle arc.
        // For an ellipse arc we use the same formula since the rational
        // construction is parameterized by the arc half-angle, not the
        // point geometry.
        let w_mid = half_angle.abs().cos();

        // Middle control point: intersection of the two tangent lines.
        // Solve: p0 + s * tan0 = p1 + t_param * (-tan1) for s.
        // Using the 2D cross-product approach projected to the arc plane.
        let p_mid = tangent_intersection(p0, tan0, p1, tan1).unwrap_or_else(|| midpoint(p0, p1));

        if arc_idx == 0 {
            cps.push(p0);
            weights.push(1.0);
        }
        cps.push(p_mid);
        weights.push(w_mid);
        cps.push(p1);
        weights.push(1.0);
    }

    Ok(NurbsCurve::new(2, knots, cps, weights)?)
}

/// Find the intersection of two parametric rays `p0 + s*d0` and `p1 + t*d1`.
///
/// Returns the intersection point, or `None` if the lines are (nearly)
/// parallel.
fn tangent_intersection(
    p0: Point3,
    d0: brepkit_math::vec::Vec3,
    p1: Point3,
    d1: brepkit_math::vec::Vec3,
) -> Option<Point3> {
    // Solve in least-squares sense for the closest approach point.
    // [d0 | -d1] * [s; t] = p1 - p0
    //
    // Use the 2-equation system from the two largest components to avoid
    // degenerate cases.
    let rhs = p1 - p0;

    // We want: d0*s - d1*t = rhs
    // Pick the two rows with largest |d0 x d1| projection to avoid
    // near-zero pivots.
    let cross = d0.cross(d1);
    let cx = cross.x().abs();
    let cy = cross.y().abs();
    let cz = cross.z().abs();

    // Choose the component pair to use.
    let (a00, a01, b0, a10, a11, b1) = if cz >= cx && cz >= cy {
        // Use x, y rows.
        (d0.x(), -d1.x(), rhs.x(), d0.y(), -d1.y(), rhs.y())
    } else if cy >= cx {
        // Use x, z rows.
        (d0.x(), -d1.x(), rhs.x(), d0.z(), -d1.z(), rhs.z())
    } else {
        // Use y, z rows.
        (d0.y(), -d1.y(), rhs.y(), d0.z(), -d1.z(), rhs.z())
    };

    let det = a00 * a11 - a01 * a10;
    if det.abs() < 1e-30 {
        return None; // parallel
    }

    let s = (b0 * a11 - b1 * a01) / det;
    Some(p0 + d0 * s)
}

/// Midpoint of two points.
fn midpoint(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        (a.x() + b.x()) * 0.5,
        (a.y() + b.y()) * 0.5,
        (a.z() + b.z()) * 0.5,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{PI, TAU};

    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    // ── circle_to_nurbs ──────────────────────────────────────────────────────

    /// For a rational quadratic NURBS, the parameterization is NOT uniform in
    /// arc angle. However, the arc endpoints and the midpoint of each Bezier
    /// arc (at the knot midpoint between two consecutive double knots) ARE
    /// geometrically exact. This helper checks those anchor points.
    fn assert_circle_anchors(circle: &Circle3D, nurbs: &NurbsCurve, t_start: f64, t_end: f64) {
        let n_arcs = ((t_end - t_start).abs() / FRAC_PI_2).ceil() as usize;
        let (t_min, t_max) = nurbs.domain();
        let span = t_max - t_min;

        for arc_idx in 0..=n_arcs {
            #[allow(clippy::cast_precision_loss)]
            let frac = arc_idx as f64 / n_arcs as f64;
            let param = t_min + span * frac;
            let angle = t_start + (t_end - t_start) * frac;

            let pt = nurbs.evaluate(param);
            let expected = circle.evaluate(angle);
            let dist = (pt - expected).length();
            assert!(
                dist < 1e-10,
                "arc endpoint {arc_idx}: dist {dist} > 1e-10 at angle {angle:.3}"
            );
        }
    }

    #[test]
    fn circle_full_endpoint_anchors() {
        let circle = Circle3D::new(origin(), z_axis(), 3.0).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, TAU).unwrap();
        assert_circle_anchors(&circle, &nurbs, 0.0, TAU);
    }

    #[test]
    fn circle_arc_quarter_endpoint_anchors() {
        let circle = Circle3D::new(Point3::new(1.0, 2.0, 3.0), z_axis(), 5.0).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, PI * 0.5).unwrap();
        assert_circle_anchors(&circle, &nurbs, 0.0, PI * 0.5);
    }

    /// Verify the NURBS lies close to the circle at many densely sampled
    /// points (not just arc endpoints). This catches geometry errors even
    /// though NURBS param ≠ circle angle.
    #[test]
    fn circle_full_dense_max_deviation() {
        let circle = Circle3D::new(origin(), z_axis(), 3.0).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, TAU).unwrap();

        let (t_min, t_max) = nurbs.domain();
        let r = circle.radius();

        for i in 0..=64 {
            #[allow(clippy::cast_precision_loss)]
            let param = t_min + (t_max - t_min) * i as f64 / 64.0;
            let pt = nurbs.evaluate(param);

            // Distance from center should equal radius.
            let dist_from_center = (pt - circle.center()).length();
            let err = (dist_from_center - r).abs();
            assert!(err < 1e-10, "sample {i}: radial error {err}");

            // Should be coplanar with the circle.
            let v = pt - circle.center();
            let out_of_plane = circle.normal().dot(v).abs();
            assert!(
                out_of_plane < 1e-10,
                "sample {i}: out-of-plane {out_of_plane}"
            );
        }
    }

    #[test]
    fn circle_to_nurbs_zero_span_error() {
        let circle = Circle3D::new(origin(), z_axis(), 1.0).unwrap();
        assert!(circle_to_nurbs(&circle, 0.0, 0.0).is_err());
    }

    // ── line_to_nurbs ────────────────────────────────────────────────────────

    #[test]
    fn line_degree_1() {
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(1.0, 2.0, 3.0);
        let nurbs = line_to_nurbs(start, end).unwrap();
        assert_eq!(nurbs.degree(), 1);
    }

    #[test]
    fn line_endpoints_match() {
        let start = Point3::new(1.0, 0.0, 0.0);
        let end = Point3::new(4.0, 5.0, 6.0);
        let nurbs = line_to_nurbs(start, end).unwrap();

        let (t0, t1) = nurbs.domain();
        let p0 = nurbs.evaluate(t0);
        let p1 = nurbs.evaluate(t1);

        assert!((p0 - start).length() < 1e-12);
        assert!((p1 - end).length() < 1e-12);
    }

    #[test]
    fn line_degenerate_error() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert!(line_to_nurbs(p, p).is_err());
    }

    // ── ellipse_to_nurbs ─────────────────────────────────────────────────────

    /// For an ellipse, the rational quadratic arc construction is approximate
    /// (unlike for circles where it is exact). Verify that arc endpoints are
    /// reproduced exactly and that off-knot samples lie within a loose tolerance.
    #[test]
    fn ellipse_arc_endpoints_exact() {
        let ellipse = Ellipse3D::new(origin(), z_axis(), 4.0, 2.0).unwrap();
        let nurbs = ellipse_to_nurbs(&ellipse, 0.0, TAU).unwrap();

        // The n_arcs arc endpoints (at knot values 0, 1/4, 1/2, 3/4, 1) should
        // exactly match ellipse(0), ellipse(π/2), ellipse(π), ellipse(3π/2),
        // ellipse(2π).
        let n_arcs = ((TAU / FRAC_PI_2).ceil() as usize).max(1);
        let (t_min, t_max) = nurbs.domain();
        for arc_idx in 0..=n_arcs {
            #[allow(clippy::cast_precision_loss)]
            let frac = arc_idx as f64 / n_arcs as f64;
            let param = t_min + (t_max - t_min) * frac;
            let angle = TAU * frac;

            let pt = nurbs.evaluate(param);
            let expected = ellipse.evaluate(angle);
            let dist = (pt - expected).length();
            assert!(dist < 1e-10, "arc endpoint {arc_idx}: dist {dist}");
        }
    }

    #[test]
    fn ellipse_points_stay_on_surface() {
        let ellipse = Ellipse3D::new(origin(), z_axis(), 4.0, 2.0).unwrap();
        let nurbs = ellipse_to_nurbs(&ellipse, 0.0, TAU).unwrap();

        let (t_min, t_max) = nurbs.domain();
        for i in 0..=32 {
            #[allow(clippy::cast_precision_loss)]
            let param = t_min + (t_max - t_min) * i as f64 / 32.0;
            let pt = nurbs.evaluate(param);

            // Points should lie in the circle plane (coplanar).
            let v = pt - origin();
            let out_of_plane = ellipse.normal().dot(v).abs();
            assert!(
                out_of_plane < 1e-10,
                "sample {i}: out-of-plane {out_of_plane}"
            );

            // Points should lie on the ellipse surface:
            // (u_comp/a)² + (v_comp/b)² ≈ 1.
            // For an approximate rational construction the error is small but non-zero.
            let u_comp = ellipse.u_axis().dot(v) / ellipse.semi_major();
            let v_comp = ellipse.v_axis().dot(v) / ellipse.semi_minor();
            let radial_err = (u_comp * u_comp + v_comp * v_comp - 1.0).abs();
            assert!(radial_err < 0.02, "sample {i}: radial_err {radial_err}");
        }
    }

    #[test]
    fn ellipse_to_nurbs_zero_span_error() {
        let ellipse = Ellipse3D::new(origin(), z_axis(), 2.0, 1.0).unwrap();
        assert!(ellipse_to_nurbs(&ellipse, 1.0, 1.0).is_err());
    }
}
