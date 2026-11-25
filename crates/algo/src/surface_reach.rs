//! Proved bounds on how far a B-spline curve lies from a face's surface,
//! along its whole length: a plane by the hull of each knot span's Bezier
//! points, a cylinder, cone, sphere or torus by its implicit form
//! ([`brepkit_math::analytic_intersection::spline_off_surface`]), and a
//! B-spline surface by a pcurve's pieces
//! ([`crate::builder::nurbs_pcurve::proved_distance`]).

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Vec3;
use brepkit_topology::face::FaceSurface;

/// An upper bound of the distance from `surface` of every point of `curve`
/// over its parameters `[t0, t1]`, or `None` where no bound is proved. A
/// bound above `aim` is read no tighter than about twice the curve's own
/// distance from the surface.
pub fn spline_off_face(
    curve: &NurbsCurve,
    span: (f64, f64),
    surface: &FaceSurface,
    aim: f64,
) -> Option<f64> {
    match surface {
        FaceSurface::Plane { normal, d } => spline_off_plane(curve, span, *normal, *d),
        FaceSurface::Nurbs(s) => crate::builder::nurbs_pcurve::proved_distance(curve, span, s, aim),
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => brepkit_math::analytic_intersection::spline_off_surface(
            curve,
            span,
            &surface.as_analytic()?,
            aim,
        ),
    }
}

/// The greatest distance from the plane `normal . p = d` of the Bezier
/// points of each knot span of `curve` over `[t0, t1]`, with their
/// rounding: with positive weights each span lies in the hull of its own.
fn spline_off_plane(curve: &NurbsCurve, (t0, t1): (f64, f64), normal: Vec3, d: f64) -> Option<f64> {
    if t0.is_nan()
        || t1.is_nan()
        || t0 >= t1
        || !curve.weights().iter().all(|&w| w > 0.0 && w.is_normal())
    {
        return None;
    }
    let size = curve
        .control_points()
        .iter()
        .map(|p| p.x().abs().max(p.y().abs()).max(p.z().abs()))
        .fold(0.0_f64, f64::max);
    let mut breaks = vec![t0];
    for &k in curve.knots() {
        if k > t0 && k < t1 && breaks.last().is_some_and(|&last| k > last) {
            breaks.push(k);
        }
    }
    breaks.push(t1);
    let mut worst = 0.0_f64;
    for span in breaks.windows(2) {
        for q in
            brepkit_math::nurbs::decompose::curve_bezier_between(curve, span[0], span[1]).ok()?
        {
            if q[3].is_nan() || q[3] <= 0.0 {
                return None;
            }
            let off = (normal.x() * q[0] + normal.y() * q[1] + normal.z() * q[2]) / q[3] - d;
            worst = worst.max(off.abs());
        }
    }
    let bound = worst + 64.0 * f64::EPSILON * (size + d.abs());
    bound.is_finite().then_some(bound)
}
