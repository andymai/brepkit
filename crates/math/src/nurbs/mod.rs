//! NURBS curve and surface representations.
//!
//! Non-Uniform Rational B-Spline (NURBS) geometry is the standard
//! representation for free-form curves and surfaces in CAD.

pub mod basis;
pub mod bezier_clip;
pub mod curve;
pub mod decompose;
pub mod evaluator;
pub mod fitting;
pub mod intersection;
pub mod knot_ops;
pub mod power_basis;
pub mod projection;
pub mod self_intersection;
pub mod span_index;
pub mod surface;
pub mod surface_fitting;

pub use bezier_clip::{
    CurveCurveHit, CurveCurveOverlap, CurveCurveResult, curve_curve_intersect,
    curve_curve_intersect_full,
};
pub use curve::NurbsCurve;
pub use decompose::{
    BezierPatch, curve_degree_elevate, curve_to_bezier_segments, surface_to_bezier_patches,
};
pub use evaluator::SurfaceEvaluator;
pub use fitting::{approximate, interpolate};
pub use knot_ops::{
    curve_knot_insert, curve_knot_refine, curve_knot_remove, curve_split, surface_knot_insert_u,
    surface_knot_insert_v,
};
pub use power_basis::PowerBasis1D;
pub use projection::{
    CurveProjection, SurfaceProjection, project_point_to_curve, project_point_to_surface,
};
pub use surface::NurbsSurface;
pub use surface_fitting::interpolate_surface;

/// A homogeneous point `[w x, w y, w z, w]` as its point, the origin for a
/// weight of zero over a numerator of zero (it adds nothing to the curve).
/// `None` for a point at infinity: a weight of zero over a numerator that is
/// not, or a quotient that overflows. Evaluation scales a point by its
/// weight, so it would drop that numerator.
pub(crate) fn cartesian(q: [f64; 4]) -> Option<crate::vec::Point3> {
    let point = if q[3] == 0.0 {
        if q[..3].iter().any(|&c| c != 0.0) {
            return None;
        }
        crate::vec::Point3::new(0.0, 0.0, 0.0)
    } else {
        crate::vec::Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])
    };
    [point.x(), point.y(), point.z()]
        .iter()
        .all(|c| c.is_finite())
        .then_some(point)
}

/// The control points and weights of the homogeneous net `net`, or
/// [`crate::MathError::PoleAtInfinity`] over `(start, end)` when one of
/// its points is at infinity.
pub(crate) fn cartesian_net(
    net: &[[f64; 4]],
    (start, end): (f64, f64),
) -> Result<(Vec<crate::vec::Point3>, Vec<f64>), crate::MathError> {
    let points = net
        .iter()
        .map(|&q| cartesian(q).ok_or(crate::MathError::PoleAtInfinity { start, end }))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((points, net.iter().map(|q| q[3]).collect()))
}

/// The knots on either side of `u` in `knots`: the span a knot inserted at
/// `u`, or removed there, reshapes.
pub(crate) fn span_around(knots: &[f64], u: f64) -> (f64, f64) {
    let below = knots.iter().copied().filter(|&k| k < u).reduce(f64::max);
    let above = knots.iter().copied().filter(|&k| k > u).reduce(f64::min);
    (below.unwrap_or(u), above.unwrap_or(u))
}
