//! PCurve computation: project 3D edge curves into a face's (u,v) parameter space.
//!
//! For plane faces, uses [`PlaneFrame`] to project 3D lines to 2D lines.
//! For analytic surfaces (cylinder, cone, sphere, torus), samples points along
//! the 3D curve, projects via `surface.project_point()`, unwraps periodicity,
//! and fits a [`NurbsCurve2D`] (or a `Line2D` if collinear in UV).

use std::f64::consts::TAU;

use brepkit_math::curves2d::{Curve2D, Line2D, NurbsCurve2D};
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

use super::plane_frame::PlaneFrame;

/// Number of sample points for pcurve fitting on non-plane surfaces.
const PCURVE_SAMPLES: usize = 16;

/// Compute the 2D pcurve for a 3D edge on a given surface.
///
/// For plane faces, uses `PlaneFrame` to project the 3D line endpoints
/// into (u,v) space and constructs a `Line2D`. For analytic surfaces,
/// samples points along the 3D edge, projects to UV, unwraps periodicity,
/// and fits a `NurbsCurve2D` (or `Line2D` if the UV curve is straight).
///
/// `wire_pts` is needed for plane faces to establish the `PlaneFrame` origin.
///
/// Returns a `Curve2D` parameterized on \[0, 1\] from start to end.
#[must_use]
pub fn compute_pcurve_on_surface(
    curve_3d: &EdgeCurve,
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
) -> Curve2D {
    if let FaceSurface::Plane { normal, .. } = surface {
        // For straight edges on planes, the pcurve is a Line2D.
        // For curved edges (Circle, Ellipse, NurbsCurve), fall through to the
        // sampling-based path to produce a proper curved pcurve.
        if matches!(curve_3d, EdgeCurve::Line) {
            let owned;
            let frame = if let Some(f) = frame {
                f
            } else {
                owned = PlaneFrame::from_plane_face(*normal, wire_pts);
                &owned
            };
            let p0 = frame.project(start);
            let p1 = frame.project(end);
            let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
            return Curve2D::Line(make_line2d_safe(p0, dir));
        }
        // Curved edge on plane: sample and project via PlaneFrame below.
    }

    // For plane surfaces with curved edges, use PlaneFrame for projection.
    let uv_pts = if let FaceSurface::Plane { normal, .. } = surface {
        let owned;
        let f = if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        };
        sample_edge_to_uv_via_frame(curve_3d, start, end, f)
    } else {
        sample_edge_to_uv(curve_3d, start, end, surface)
    };
    pcurve_through_uv_samples(uv_pts, start, end, surface)
}

/// The pcurve of a face's own boundary edge. An open circle or ellipse
/// arc runs its native span (counter-clockwise in its own parameter from
/// its stored start to its stored end), so a half or major arc traces
/// itself; the shorter-arc convention of [`compute_pcurve_on_surface`]
/// would trace the complement or, at exactly half a turn, either side.
/// `forward = false` traverses the edge from its stored end.
#[must_use]
pub fn compute_boundary_pcurve_on_surface(
    curve_3d: &EdgeCurve,
    stored_start: Point3,
    stored_end: Point3,
    forward: bool,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
) -> Curve2D {
    let (start, end) = if forward {
        (stored_start, stored_end)
    } else {
        (stored_end, stored_start)
    };
    if !matches!(curve_3d, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
        || (stored_start - stored_end).length() <= 1e-12
    {
        return compute_pcurve_on_surface(curve_3d, start, end, surface, wire_pts, frame);
    }
    let mut pts_3d = Vec::with_capacity(PCURVE_SAMPLES + 1);
    sample_edge_uniform_native(
        curve_3d,
        stored_start,
        stored_end,
        PCURVE_SAMPLES,
        forward,
        &mut pts_3d,
    );
    pts_3d.push(end);
    let uv_pts = if let FaceSurface::Plane { normal, .. } = surface {
        let owned;
        let f = if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        };
        pts_3d.iter().map(|&p| f.project(p)).collect()
    } else {
        let mut uv: Vec<Point2> = pts_3d
            .iter()
            .map(|&p| {
                let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
                Point2::new(u, v)
            })
            .collect();
        let (u_period, v_period) = surface_periods(surface);
        unwrap_periodic_params(&mut uv, u_period, v_period);
        uv
    };
    pcurve_through_uv_samples(uv_pts, start, end, surface)
}

/// A pcurve through UV samples running from `start` to `end`: a `Line2D`
/// when they are collinear and distinct, else a fitted NURBS.
fn pcurve_through_uv_samples(
    uv_pts: Vec<Point2>,
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
) -> Curve2D {
    if uv_pts.len() < 2 {
        // Degenerate: just project endpoints.
        let (u0, v0) = surface.project_point(start).unwrap_or((0.0, 0.0));
        let (u1, v1) = surface.project_point(end).unwrap_or((1.0, 0.0));
        let p0 = Point2::new(u0, v0);
        let p1 = Point2::new(u1, v1);
        let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
        return Curve2D::Line(make_line2d_safe(p0, dir));
    }

    // Check collinearity -- if all points are (nearly) on a line in UV,
    // use a Line2D instead of a NURBS fit.
    // EXCEPTION: Line2D uses arc-length parameterization (evaluate(t) =
    // origin + unit_dir * t), not [0,1] mapping. For curves that need
    // evaluate(0)→start and evaluate(1)→end, use NURBS interpolation
    // instead, which naturally maps [0,1] to the full extent.
    if is_collinear_2d(&uv_pts, 1e-6) {
        let p0 = uv_pts[0];
        let pn = uv_pts[uv_pts.len() - 1];
        let dx = pn.x() - p0.x();
        let dy = pn.y() - p0.y();
        let len_sq = dx * dx + dy * dy;
        // For non-degenerate lines (p0 ≠ pn), use Line2D.
        // For closed collinear curves (p0 ≈ pn), fall through to NURBS
        // fit to preserve [0,1] parameterization.
        if len_sq >= 1e-12 {
            let dir = Vec2::new(dx, dy);
            return Curve2D::Line(make_line2d_safe(p0, dir));
        }
        // p0 ≈ pn: fall through to NURBS interpolation.
    }

    // Fit a NURBS curve through the UV sample points.
    fit_nurbs2d_through_points(&uv_pts)
}

/// Project a 3D point onto a surface's parameter space.
///
/// For planes, uses `PlaneFrame`. For analytic/NURBS, uses
/// `ParametricSurface::project_point()`.
pub fn project_point_on_surface(
    p: Point3,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
) -> Point2 {
    if let FaceSurface::Plane { normal, .. } = surface {
        let owned;
        let frame = if let Some(f) = frame {
            f
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        };
        return frame.project(p);
    }
    let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
    Point2::new(u, v)
}

/// Build a `PlaneFrame` for a plane face.
#[allow(dead_code)]
pub fn plane_frame_for_face(normal: Vec3, wire_pts: &[Point3]) -> PlaneFrame {
    PlaneFrame::from_plane_face(normal, wire_pts)
}

/// Create a `Line2D` safely, handling degenerate (zero-length) directions.
pub(super) fn make_line2d_safe(origin: Point2, dir: Vec2) -> Line2D {
    Line2D::new(origin, dir).unwrap_or_else(|_| {
        // Degenerate edge -- fallback to x-axis direction.
        // Safety: (1, 0) is non-zero, so Line2D::new cannot fail.
        #[allow(clippy::unwrap_used)]
        Line2D::new(origin, Vec2::new(1.0, 0.0)).unwrap()
    })
}

/// Unwrap periodic UV parameters (public wrapper for use by other modules).
pub(super) fn unwrap_periodic_params_pub(
    pts: &mut [Point2],
    u_period: Option<f64>,
    v_period: Option<f64>,
) {
    unwrap_periodic_params(pts, u_period, v_period);
}

/// Returns `(u_period, v_period)` for a surface -- `Some(TAU)` if periodic.
pub(super) fn surface_periods(surface: &FaceSurface) -> (Option<f64>, Option<f64>) {
    match surface {
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => (None, None),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => (Some(TAU), None),
        FaceSurface::Sphere(_) => (Some(TAU), None),
        FaceSurface::Torus(_) => (Some(TAU), Some(TAU)),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sample points along a 3D edge curve and project each to surface UV.
///
/// Returns UV points with periodicity unwrapped.
pub(super) fn sample_edge_to_uv(
    curve_3d: &EdgeCurve,
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
) -> Vec<Point2> {
    let n = PCURVE_SAMPLES;
    let mut pts_3d = Vec::with_capacity(n + 1);
    // A cone's closed rim runs a full turn from its vertex, wherever the
    // circle's own frame puts parameter 0, so its UV samples start where the
    // seam lines leaving that vertex do.
    let closed_cone_rim = matches!(surface, FaceSurface::Cone(_)) && (start - end).length() < 1e-10;
    let at = edge_point_at(curve_3d, start, end);
    for i in 0..=n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f64 / n as f64;
        let p = match curve_3d {
            EdgeCurve::Circle(c) if closed_cone_rim => {
                ParametricCurve::evaluate(c, c.project(start) + TAU * t)
            }
            EdgeCurve::Ellipse(e) if closed_cone_rim => {
                ParametricCurve::evaluate(e, e.project(start) + TAU * t)
            }
            _ => at(t),
        };
        pts_3d.push(p);
    }

    let mut uv_pts: Vec<Point2> = pts_3d
        .iter()
        .map(|&p| {
            let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
            Point2::new(u, v)
        })
        .collect();

    // Unwrap periodicity.
    let (u_period, v_period) = surface_periods(surface);
    unwrap_periodic_params(&mut uv_pts, u_period, v_period);

    uv_pts
}

/// Evaluate a 3D edge curve at parameter t in [0, 1].
///
/// For `Line`, linearly interpolates between start and end.
/// For open `Circle`/`Ellipse` edges, traces the shorter arc between the
/// endpoints. For closed edges and `NurbsCurve`, uses the curve domain.
pub(super) fn evaluate_edge_at_t(curve: &EdgeCurve, start: Point3, end: Point3, t: f64) -> Point3 {
    edge_point_at(curve, start, end)(t)
}

/// [`evaluate_edge_at_t`] for many `t` along one edge: the span a closed or
/// NURBS curve covers is read once, since a NURBS edge on part of its curve
/// finds it by projecting both end vertices.
pub(super) fn edge_point_at(
    curve: &EdgeCurve,
    start: Point3,
    end: Point3,
) -> impl Fn(f64) -> Point3 + '_ {
    let span = match curve {
        EdgeCurve::Line => None,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) if (start - end).length() > 1e-12 => None,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
            Some(curve.domain_with_endpoints(start, end))
        }
    };
    move |t| match span {
        Some((t0, t1)) => curve.evaluate_with_endpoints(t0 + (t1 - t0) * t, start, end),
        None => evaluate_open_edge_at_t(curve, start, end, t),
    }
}

/// An edge at fraction `t`, an open circle or ellipse along its shorter arc
/// and any other curve over its span.
fn evaluate_open_edge_at_t(curve: &EdgeCurve, start: Point3, end: Point3, t: f64) -> Point3 {
    match curve {
        EdgeCurve::Line => Point3::new(
            start.x() + (end.x() - start.x()) * t,
            start.y() + (end.y() - start.y()) * t,
            start.z() + (end.z() - start.z()) * t,
        ),
        // Open circle/ellipse edges follow the shorter-arc convention shared
        // with tessellation and topology wire sampling: `domain_with_endpoints`
        // returns the full [0, 2pi] regardless of endpoints, which is only
        // right for closed edges — sampling an open arc over it would trace
        // the whole circle.
        EdgeCurve::Circle(c) if (start - end).length() > 1e-12 => {
            let t0 = c.project(start);
            let d = shorter_arc_delta(c.project(end) - t0);
            ParametricCurve::evaluate(c, t0 + d * t)
        }
        EdgeCurve::Ellipse(e) if (start - end).length() > 1e-12 => {
            let t0 = e.project(start);
            let d = shorter_arc_delta(e.project(end) - t0);
            ParametricCurve::evaluate(e, t0 + d * t)
        }
        _ => {
            // Closed curves and NURBS: parametric evaluation over the domain.
            let (t0, t1) = curve.domain_with_endpoints(start, end);
            let param = t0 + (t1 - t0) * t;
            curve.evaluate_with_endpoints(param, start, end)
        }
    }
}

/// Uniformly sample an edge along its NATIVE span: a circle or ellipse edge
/// runs counter-clockwise in its curve's own parameter from its stored
/// start to its stored end (`domain_with_endpoints`), so a major arc, such
/// as the 276 degree remainder of a rim split at two paves, traces itself
/// and not its short complement. `forward = false` reverses the samples.
///
/// Section arcs (below a half turn by construction) go through
/// [`evaluate_edge_at_t`]'s shorter-arc convention instead; boundary edges
/// must come here.
pub(super) fn sample_edge_uniform_native(
    curve: &EdgeCurve,
    start: Point3,
    end: Point3,
    n: usize,
    forward: bool,
    out: &mut Vec<Point3>,
) {
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    let span = t1 - t0;
    #[allow(clippy::cast_precision_loss)]
    for k in 0..n {
        let f = k as f64 / n as f64;
        let f = if forward { f } else { 1.0 - f };
        out.push(curve.evaluate_with_endpoints(f.mul_add(span, t0), start, end));
    }
}

/// Wrap an angular difference into (-pi, pi] — the shorter way around.
pub(super) fn shorter_arc_delta(d: f64) -> f64 {
    let w = d.rem_euclid(TAU);
    if w > std::f64::consts::PI { w - TAU } else { w }
}

/// Sample points along a 3D edge curve and project each to UV via `PlaneFrame`.
///
/// Used for curved edges (Circle, Ellipse) on plane surfaces where
/// `surface.project_point()` returns `None`.
fn sample_edge_to_uv_via_frame(
    curve_3d: &EdgeCurve,
    start: Point3,
    end: Point3,
    frame: &PlaneFrame,
) -> Vec<Point2> {
    let n = PCURVE_SAMPLES;
    let mut uv_pts = Vec::with_capacity(n + 1);
    let at = edge_point_at(curve_3d, start, end);
    for i in 0..=n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f64 / n as f64;
        uv_pts.push(frame.project(at(t)));
    }
    uv_pts
}

/// Unwrap periodic UV parameters to remove seam jumps.
///
/// Shifts each point by the whole number of periods that brings it to the
/// copy nearest its predecessor, maintaining continuity across any copy
/// distance (a mixed-copy wire can jump multiple periods at once).
fn unwrap_periodic_params(pts: &mut [Point2], u_period: Option<f64>, v_period: Option<f64>) {
    if pts.len() < 2 {
        return;
    }

    // Shift each point to the period copy NEAREST its predecessor. A single
    // ±period step (the old form) cannot recover a multi-period jump, which
    // arises when a wire mixes period copies (an edge stored with a negative-u
    // endpoint following edges unwrapped upward) — the walk then folds and a
    // valid band loop measures zero area. Rounding the jump handles any copy
    // distance and is identical to the single step for ordinary seam jumps.
    for i in 1..pts.len() {
        if let Some(period) = u_period {
            let du = pts[i].x() - pts[i - 1].x();
            let k = (du / period).round();
            if k != 0.0 {
                pts[i] = Point2::new(period.mul_add(-k, pts[i].x()), pts[i].y());
            }
        }
        if let Some(period) = v_period {
            let dv = pts[i].y() - pts[i - 1].y();
            let k = (dv / period).round();
            if k != 0.0 {
                pts[i] = Point2::new(pts[i].x(), period.mul_add(-k, pts[i].y()));
            }
        }
    }
}

/// Check if a sequence of 2D points is approximately collinear.
fn is_collinear_2d(pts: &[Point2], tol: f64) -> bool {
    if pts.len() < 3 {
        return true;
    }
    let p0 = pts[0];
    let pn = pts[pts.len() - 1];
    let dx = pn.x() - p0.x();
    let dy = pn.y() - p0.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < tol * tol {
        // p0 ≈ pn — either all points are clustered (degenerate) or this
        // is a closed curve (circle). Check if intermediate points lie on
        // a LINE (collinear in UV, e.g. circle on cylinder at constant v)
        // or spread in 2D (actual closed loop, e.g. circle on plane).
        //
        // Use the line from p0 to the farthest intermediate point as the
        // collinearity reference. If all points are near that line, collinear.
        let mut farthest_idx = 1;
        let mut farthest_dist_sq = 0.0_f64;
        for (i, p) in pts[1..pts.len() - 1].iter().enumerate() {
            let ex = p.x() - p0.x();
            let ey = p.y() - p0.y();
            let d2 = ex * ex + ey * ey;
            if d2 > farthest_dist_sq {
                farthest_dist_sq = d2;
                farthest_idx = i + 1;
            }
        }
        if farthest_dist_sq < tol * tol {
            return true; // All points clustered — degenerate.
        }
        // Check collinearity against line p0→farthest.
        let pf = pts[farthest_idx];
        let fdx = pf.x() - p0.x();
        let fdy = pf.y() - p0.y();
        let flen = farthest_dist_sq.sqrt();
        let inv_flen = 1.0 / flen;
        for (i, p) in pts.iter().enumerate() {
            if i == 0 || i == farthest_idx {
                continue;
            }
            let ex = p.x() - p0.x();
            let ey = p.y() - p0.y();
            let dist = (ex * fdy - ey * fdx).abs() * inv_flen;
            if dist > tol {
                return false; // 2D spread — closed loop.
            }
        }
        return true; // All on a line — collinear (e.g. cylinder UV).
    }
    let inv_len = 1.0 / len_sq.sqrt();
    for p in &pts[1..pts.len() - 1] {
        let ex = p.x() - p0.x();
        let ey = p.y() - p0.y();
        // Perpendicular distance from p to line(p0, pn).
        let dist = (ex * dy - ey * dx).abs() * inv_len;
        if dist > tol {
            return false;
        }
    }
    true
}

/// Fit a `NurbsCurve2D` through 2D sample points via NURBS interpolation.
///
/// Lifts 2D points to 3D (z=0), uses the math crate's `interpolate`, then
/// extracts 2D control points from the result.
fn fit_nurbs2d_through_points(pts: &[Point2]) -> Curve2D {
    let fallback = || -> Curve2D {
        let p0 = pts[0];
        let p1 = pts[pts.len() - 1];
        let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
        Curve2D::Line(make_line2d_safe(p0, dir))
    };

    let pts_3d: Vec<Point3> = pts.iter().map(|p| Point3::new(p.x(), p.y(), 0.0)).collect();
    let degree = 3.min(pts_3d.len() - 1);
    let Ok(nurbs_3d) = brepkit_math::nurbs::fitting::interpolate(&pts_3d, degree) else {
        return fallback();
    };

    let cp_2d: Vec<Point2> = nurbs_3d
        .control_points()
        .iter()
        .map(|p| Point2::new(p.x(), p.y()))
        .collect();
    let weights = nurbs_3d.weights().to_vec();
    let knots = nurbs_3d.knots().to_vec();
    NurbsCurve2D::new(nurbs_3d.degree(), knots, cp_2d, weights)
        .map_or_else(|_| fallback(), Curve2D::Nurbs)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::vec::Vec3;

    #[test]
    fn line_on_xy_plane_produces_line2d_with_roundtrip() {
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(3.0, 4.0, 0.0);
        let wire_pts = vec![
            start,
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];

        let pcurve =
            compute_pcurve_on_surface(&EdgeCurve::Line, start, end, &surface, &wire_pts, None);

        let frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &wire_pts);
        let expected_start = frame.project(start);
        let expected_end = frame.project(end);
        let len = ((expected_end.x() - expected_start.x()).powi(2)
            + (expected_end.y() - expected_start.y()).powi(2))
        .sqrt();

        let p_start = pcurve.evaluate(0.0);
        let p_end = pcurve.evaluate(len);

        assert!((p_start.x() - expected_start.x()).abs() < 1e-10);
        assert!((p_start.y() - expected_start.y()).abs() < 1e-10);
        assert!((p_end.x() - expected_end.x()).abs() < 1e-10);
        assert!((p_end.y() - expected_end.y()).abs() < 1e-10);
    }

    #[test]
    fn line_on_tilted_plane_roundtrips() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let surface = FaceSurface::Plane { normal, d: 5.0 };
        let start = Point3::new(1.0, 2.0, 5.0);
        let end = Point3::new(4.0, 6.0, 5.0);
        let wire_pts = vec![
            start,
            Point3::new(10.0, 0.0, 5.0),
            Point3::new(0.0, 10.0, 5.0),
        ];

        let pcurve =
            compute_pcurve_on_surface(&EdgeCurve::Line, start, end, &surface, &wire_pts, None);

        let frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        let expected_start = frame.project(start);
        let expected_end = frame.project(end);
        let len = ((expected_end.x() - expected_start.x()).powi(2)
            + (expected_end.y() - expected_start.y()).powi(2))
        .sqrt();

        let p0 = pcurve.evaluate(0.0);
        let p1 = pcurve.evaluate(len);

        let start_back = frame.evaluate(p0.x(), p0.y());
        let end_back = frame.evaluate(p1.x(), p1.y());
        assert!((start_back - start).length() < 1e-10);
        assert!((end_back - end).length() < 1e-10);
    }

    #[test]
    fn project_point_on_xy_plane() {
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let wire_pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let p = Point3::new(5.0, 3.0, 0.0);
        let uv = project_point_on_surface(p, &surface, &wire_pts, None);

        let frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &wire_pts);
        let back = frame.evaluate(uv.x(), uv.y());
        assert!((back - p).length() < 1e-10);
    }

    #[test]
    fn unwrap_periodic_removes_seam_jump() {
        let mut pts = vec![
            Point2::new(6.0, 0.0),
            Point2::new(6.2, 0.0),
            Point2::new(0.1, 0.0), // Jump from ~6.2 to ~0.1 (crossed 2pi)
            Point2::new(0.3, 0.0),
        ];
        unwrap_periodic_params(&mut pts, Some(TAU), None);
        assert!(
            (pts[2].x() - (0.1 + TAU)).abs() < 0.01,
            "got {}",
            pts[2].x()
        );
    }

    #[test]
    fn collinearity_detection() {
        let line_pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 3.0),
        ];
        assert!(is_collinear_2d(&line_pts, 1e-6));

        let curve_pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 3.0),
        ];
        assert!(!is_collinear_2d(&curve_pts, 1e-6));
    }

    #[test]
    fn pcurve_line_on_cylinder_is_vertical() {
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let start = Point3::new(1.0, 0.0, 0.0);
        let end = Point3::new(1.0, 0.0, 5.0);
        let pcurve = compute_pcurve_on_surface(&EdgeCurve::Line, start, end, &surface, &[], None);

        assert!(
            matches!(pcurve, Curve2D::Line(_)),
            "expected Line, got {:?}",
            pcurve
        );
    }

    #[test]
    fn pcurve_circle_on_cylinder_is_horizontal() {
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let start = Point3::new(1.0, 0.0, 3.0);
        let end = Point3::new(-1.0, 0.0, 3.0);
        let curve_3d = EdgeCurve::Circle(circle);
        let pcurve = compute_pcurve_on_surface(&curve_3d, start, end, &surface, &[], None);

        assert!(
            matches!(pcurve, Curve2D::Line(_)),
            "expected Line for equatorial circle, got {:?}",
            pcurve
        );
    }

    #[test]
    fn surface_periods_correct() {
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        assert_eq!(surface_periods(&plane), (None, None));

        let cyl = FaceSurface::Cylinder(
            brepkit_math::surfaces::CylindricalSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
            )
            .unwrap(),
        );
        assert_eq!(surface_periods(&cyl), (Some(TAU), None));

        let sphere = FaceSurface::Sphere(
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap(),
        );
        assert_eq!(surface_periods(&sphere), (Some(TAU), None));

        let torus = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0)
                .unwrap(),
        );
        assert_eq!(surface_periods(&torus), (Some(TAU), Some(TAU)));
    }

    /// The batch sampler must agree with per-sample `evaluate_edge_at_t` on
    /// every curve arm and both traversal directions — it exists only to
    /// hoist the per-arm setup, never to change the sampled points.
    /// A NURBS edge on part of its curve, either way along it: each sample
    /// is the curve at the matching fraction of the edge's own span.
    #[test]
    fn edge_point_at_walks_a_nurbs_sub_span() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_topology::edge::EdgeCurve;

        let nurbs = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, -1.0, 1.0),
                Point3::new(3.0, 1.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 0.8, 1.2, 1.0, 1.0],
        )
        .unwrap();
        let curve = EdgeCurve::NurbsCurve(nurbs.clone());
        for (u0, u1) in [(0.2, 0.8), (0.8, 0.2), (0.35, 0.6)] {
            let (start, end) = (nurbs.evaluate(u0), nurbs.evaluate(u1));
            let at = edge_point_at(&curve, start, end);
            for k in 0..=8 {
                let t = f64::from(k) / 8.0;
                let want = nurbs.evaluate(u0 + (u1 - u0) * t);
                assert!(
                    (at(t) - want).length() < 1e-7,
                    "({u0}, {u1}) t={t}: {:?} vs {want:?}",
                    at(t)
                );
                assert!((evaluate_edge_at_t(&curve, start, end, t) - want).length() < 1e-7);
            }
        }
    }

    #[test]
    fn native_sampler_traces_the_major_arc() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::EdgeCurve;

        let circle = Circle3D::new_with_ref(
            Point3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.5,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        // A rim remainder: stored start at 0 and stored end at 276 degrees,
        // so the native arc is the major one and the chord midpoint sits on
        // the minor side.
        let span = 276.0_f64.to_radians();
        let (start, end) = (circle.evaluate(0.0), circle.evaluate(span));
        let curve = EdgeCurve::Circle(circle.clone());
        let n = 12usize;
        let mut fwd = Vec::new();
        sample_edge_uniform_native(&curve, start, end, n, true, &mut fwd);
        assert_eq!(fwd.len(), n);
        for (k, p) in fwd.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let want = circle.evaluate(span * k as f64 / n as f64);
            assert!((*p - want).length() < 1e-12, "k={k}: {p:?} vs {want:?}");
        }
        let chord_mid = start + (end - start) * 0.5;
        let far = fwd[n / 2];
        assert!((far - circle.center()).dot(chord_mid - circle.center()) < 0.0);

        let mut rev = Vec::new();
        sample_edge_uniform_native(&curve, start, end, n, false, &mut rev);
        for (k, p) in rev.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let want = circle.evaluate(span * (1.0 - k as f64 / n as f64));
            assert!((*p - want).length() < 1e-12, "rev k={k}: {p:?} vs {want:?}");
        }
    }
}
