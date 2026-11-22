//! PCurve computation: project 3D edge curves into a face's (u,v) parameter space.
//!
//! For plane faces, uses [`PlaneFrame`] to project 3D lines to 2D lines.
//! For analytic surfaces (cylinder, cone, sphere, torus), samples points along
//! the 3D curve, projects via `surface.project_point()`, unwraps periodicity,
//! and fits a [`NurbsCurve2D`] (or a `Line2D` if collinear in UV).

use brepkit_math::madd::Madd;
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
    let (pts_3d, uv_pts) = if let FaceSurface::Plane { normal, .. } = surface {
        let owned;
        let f = if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        };
        (
            Vec::new(),
            sample_edge_to_uv_via_frame(curve_3d, start, end, f),
        )
    } else {
        sample_edge_on_surface(curve_3d, start, end, surface)
    };
    if let EdgeCurve::Ellipse(ellipse) = curve_3d
        && !is_collinear_2d(&uv_pts, 1e-6)
        && let Some(pcurve) = wall_ellipse_pcurve(
            ellipse,
            surface,
            ellipse_sample_span(ellipse, start, end, surface),
        )
    {
        return pcurve;
    }
    pcurve_through_uv_samples(uv_pts, &pts_3d, start, end, surface)
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
    if let FaceSurface::Plane { normal, .. } = surface {
        let owned;
        let f = if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        };
        if let Some(arc) = plane_conic_arc(curve_3d, stored_start, stored_end, forward, f) {
            return arc;
        }
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
        lift_ellipse_samples(curve_3d, surface, &pts_3d, &mut uv);
        uv
    };
    if let EdgeCurve::Ellipse(ellipse) = curve_3d
        && !is_collinear_2d(&uv_pts, 1e-6)
    {
        let (t0, t1) = curve_3d.domain_with_endpoints(stored_start, stored_end);
        let span = if forward { (t0, t1) } else { (t1, t0) };
        if let Some(pcurve) = wall_ellipse_pcurve(ellipse, surface, span) {
            return pcurve;
        }
    }
    pcurve_through_uv_samples(uv_pts, &pts_3d, start, end, surface)
}

/// A circle or ellipse arc in a plane's frame, exactly: the frame maps the
/// conic `c + cos t a + sin t b` affinely, so its image is the conic through
/// the projected centre and axes, built as rational quadratic pieces of at
/// most a quarter turn, from the traversal's start to its end on [0, 1].
///
/// A fit through a fixed number of samples drifts at the ends of a long
/// arc (7e-3 rad off the tangent of a hole's rim piece running most of the
/// way round), and the walk then reads the piece as turning off the arc it
/// continues. `None` for other curves, or an arc of no sweep.
fn plane_conic_arc(
    curve_3d: &EdgeCurve,
    stored_start: Point3,
    stored_end: Point3,
    forward: bool,
    frame: &PlaneFrame,
) -> Option<Curve2D> {
    let centre = match curve_3d {
        EdgeCurve::Circle(c) => c.center(),
        EdgeCurve::Ellipse(e) => e.center(),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
    };
    let at = |t: f64| curve_3d.evaluate_with_endpoints(t, stored_start, stored_end);
    let c = frame.project(centre);
    let a = frame.project(at(0.0)) - c;
    let b = frame.project(at(std::f64::consts::FRAC_PI_2)) - c;
    let (t0, t1) = curve_3d.domain_with_endpoints(stored_start, stored_end);
    let (ta, tb) = if forward { (t0, t1) } else { (t1, t0) };
    let sweep = tb - ta;
    if !sweep.is_finite() || sweep.abs() < 1e-12 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pieces = (sweep.abs() / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
    #[allow(clippy::cast_precision_loss)]
    let step = sweep / pieces as f64;
    let on = |t: f64| c + a * t.cos() + b * t.sin();
    let w = (step * 0.5).cos();
    let mut points = vec![on(ta)];
    let mut weights = vec![1.0];
    let mut knots = vec![0.0; 3];
    for k in 0..pieces {
        #[allow(clippy::cast_precision_loss)]
        let lo = (k as f64).madd(step, ta);
        let mid = 0.5f64.madd(step, lo);
        points.push(c + (a * mid.cos() + b * mid.sin()) * (1.0 / w));
        weights.push(w);
        points.push(on(lo + step));
        weights.push(1.0);
        #[allow(clippy::cast_precision_loss)]
        let knot = (k + 1) as f64 / pieces as f64;
        if k + 1 < pieces {
            knots.extend([knot, knot]);
        }
    }
    knots.extend([1.0; 3]);
    NurbsCurve2D::new(2, knots, points, weights)
        .ok()
        .map(Curve2D::Nurbs)
}

/// A pcurve through UV samples running from `start` to `end`: a `Line2D`
/// when they are collinear and distinct, else a fitted NURBS. `pts_3d` are
/// the samples on the edge, one for each UV sample, or none.
fn pcurve_through_uv_samples(
    uv_pts: Vec<Point2>,
    pts_3d: &[Point3],
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

    if matches!(surface, FaceSurface::Plane { .. }) {
        return fit_nurbs2d_through_points(&uv_pts);
    }
    fit_nurbs2d_at_chords_3d(&uv_pts, pts_3d)
}

/// A NURBS through UV samples, each at the length along the edge's chords
/// in 3D to its 3D sample.
///
/// A curved surface's parameters are not lengths alike: a cylinder's `u` is
/// an angle and its `v` a length. Chords read in UV then give samples
/// uneven parameters where the curve runs mostly along one of them, and the
/// interpolant swings past its samples. A closed curve round a thin strut's
/// wall left its first sample back across the wall's seam, so the face's
/// trace read that boundary edge as leaving the other way. Samples with no
/// 3D length between them (at a cone's apex) keep the chords in UV.
fn fit_nurbs2d_at_chords_3d(uv_pts: &[Point2], pts_3d: &[Point3]) -> Curve2D {
    let n = uv_pts.len();
    if n < 2 || pts_3d.len() != n {
        return fit_nurbs2d_through_points(uv_pts);
    }
    let mut params = Vec::with_capacity(n);
    let mut run = 0.0;
    params.push(0.0);
    for w in pts_3d.windows(2) {
        run += (w[1] - w[0]).length();
        params.push(run);
    }
    if !run.is_finite() || run <= 0.0 {
        return fit_nurbs2d_through_points(uv_pts);
    }
    for t in &mut params {
        *t /= run;
    }
    params[n - 1] = 1.0;
    if params.windows(2).any(|w| w[1] - w[0] <= 1e-9) {
        return fit_nurbs2d_through_points(uv_pts);
    }
    let lifted: Vec<Point3> = uv_pts
        .iter()
        .map(|p| Point3::new(p.x(), p.y(), 0.0))
        .collect();
    let Ok(fit) = brepkit_math::nurbs::fitting::interpolate_at(&lifted, &params, 3.min(n - 1))
    else {
        return fit_nurbs2d_through_points(uv_pts);
    };
    let cp_2d: Vec<Point2> = fit
        .control_points()
        .iter()
        .map(|p| Point2::new(p.x(), p.y()))
        .collect();
    NurbsCurve2D::new(
        fit.degree(),
        fit.knots().to_vec(),
        cp_2d,
        fit.weights().to_vec(),
    )
    .map_or_else(|_| fit_nurbs2d_through_points(uv_pts), Curve2D::Nurbs)
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
    sample_edge_on_surface(curve_3d, start, end, surface).1
}

/// [`sample_edge_to_uv`] with the 3D samples the UV samples project.
fn sample_edge_on_surface(
    curve_3d: &EdgeCurve,
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
) -> (Vec<Point3>, Vec<Point2>) {
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
            EdgeCurve::Ellipse(e) => {
                let (ta, tb) = ellipse_sample_span(e, start, end, surface);
                ParametricCurve::evaluate(e, (tb - ta).madd(t, ta))
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
    lift_ellipse_samples(curve_3d, surface, &pts_3d, &mut uv_pts);

    (pts_3d, uv_pts)
}

/// The way the `u` of a cylinder or a cone runs along an ellipse on it as
/// the ellipse's parameter grows: `1.0` or `-1.0`. `None` on another
/// surface, and when the axis does not pass inside the ellipse's shadow
/// along the axis.
///
/// `u` is the angle about the axis of the point's shadow. The shadow of the
/// ellipse is `q0 + M (cos t, sin t)`, an ellipse traced the way the sign
/// of `det M` says. A plane section of the wall encloses the axis, so the
/// axis's point lies inside the shadow, at `M^-1 (-q0)` within the unit
/// circle. The angle about a point inside a convex loop runs strictly one
/// way round the loop, one turn each turn.
pub(super) fn ellipse_u_sense(
    ellipse: &brepkit_math::curves::Ellipse3D,
    surface: &FaceSurface,
) -> Option<f64> {
    let (origin, x, y) = match surface {
        FaceSurface::Cylinder(c) => (c.origin(), c.x_axis(), c.y_axis()),
        FaceSurface::Cone(c) => (c.apex(), c.x_axis(), c.y_axis()),
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => return None,
    };
    let centre = ellipse.center() - origin;
    let (a, b) = (
        ellipse.u_axis() * ellipse.semi_major(),
        ellipse.v_axis() * ellipse.semi_minor(),
    );
    let (q0x, q0y) = (x.dot(centre), y.dot(centre));
    let (m11, m12, m21, m22) = (x.dot(a), x.dot(b), y.dot(a), y.dot(b));
    let det = m11.madd(m22, -(m12 * m21));
    if !det.is_finite() || det.abs() <= f64::EPSILON * ellipse.semi_major() * ellipse.semi_minor() {
        return None;
    }
    let (wx, wy) = (
        m12.madd(q0y, -(m22 * q0x)) / det,
        m21.madd(q0x, -(m11 * q0y)) / det,
    );
    (wx.hypot(wy) < 1.0).then_some(det.signum())
}

/// Lift the `u` of UV samples taken in order along an ellipse on a
/// cylinder or a cone, `uv[k]` the projection of `pts_3d[k]`. Unwrapped to
/// the nearest copy, a step over which `u` runs most of a turn (round a
/// steep cut's near vertex) reads as a short step back. `u` runs one way
/// along the ellipse ([`ellipse_u_sense`]), so each step advances it the
/// way the step runs round the ellipse, by its change read that way round.
/// The samples lie under half a turn of the ellipse's parameter apart.
fn lift_ellipse_samples(
    curve: &EdgeCurve,
    surface: &FaceSurface,
    pts_3d: &[Point3],
    uv: &mut [Point2],
) {
    let EdgeCurve::Ellipse(ellipse) = curve else {
        return;
    };
    let Some(sense) = ellipse_u_sense(ellipse, surface) else {
        return;
    };
    if pts_3d.len() != uv.len() {
        return;
    }
    for k in 1..uv.len() {
        let step = shorter_arc_delta(ellipse.project(pts_3d[k]) - ellipse.project(pts_3d[k - 1]));
        let way = sense * step.signum();
        let prev = uv[k - 1].x();
        let ahead = (way * (uv[k].x() - prev)).rem_euclid(TAU);
        // A step of no length, read with rounding a hair back.
        let ahead = if ahead > TAU - 1e-9 {
            ahead - TAU
        } else {
            ahead
        };
        uv[k] = Point2::new(way.madd(ahead, prev), uv[k].y());
    }
}

/// The span of its parameter that an ellipse from `start` to `end` runs
/// over when sampled on `surface` ([`sample_edge_on_surface`]): a whole
/// turn from its vertex when closed, else the shorter arc between its ends.
fn ellipse_sample_span(
    ellipse: &brepkit_math::curves::Ellipse3D,
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
) -> (f64, f64) {
    let a0 = ellipse.project(start);
    let gap = (start - end).length();
    if matches!(surface, FaceSurface::Cone(_)) && gap < 1e-10 {
        (a0, a0 + TAU)
    } else if gap > 1e-12 {
        (a0, a0 + shorter_arc_delta(ellipse.project(end) - a0))
    } else {
        EdgeCurve::Ellipse(ellipse.clone()).domain_with_endpoints(start, end)
    }
}

/// The most a wall ellipse's pcurve moves in `u` between two samples.
const WALL_ELLIPSE_STEP: f64 = std::f64::consts::FRAC_PI_8;
/// How far a wall ellipse's pcurve may stand off the ellipse's own `(u, v)`
/// at the points each piece is tested at: in `u`, and in `v` as a fraction
/// of `max(1, |v|)`.
const WALL_ELLIPSE_DEVIATION: f64 = 1e-6;
/// The most pieces a wall ellipse's pcurve takes before the fit gives up.
const WALL_ELLIPSE_MAX_PIECES: usize = 4096;

/// A sample of an ellipse on a wall: its parameter `s` on `[0, 1]`, its
/// `(u, v)` with `u` lifted along the ellipse, and their rates in `s`.
#[derive(Clone, Copy)]
struct WallSample {
    s: f64,
    uv: Point2,
    rate: Vec2,
}

/// The pcurve of an ellipse on a cylinder or a cone as its parameter runs
/// evenly from `ta` to `tb` over `[0, 1]`: cubic pieces through samples
/// that each match the ellipse's `(u, v)` and their rates exactly. `None`
/// when [`ellipse_u_sense`] is, or when the pieces cannot meet the tests.
///
/// A fit through even samples cannot follow a steep cut, whose `u` runs
/// most of a turn round the near vertex within a hundredth of the
/// parameter: it swings back across the turn. Here a piece splits in two
/// until its run in `u` is at most [`WALL_ELLIPSE_STEP`], its cubic in `u`
/// lies in the region where a cubic between two increasing ends with those
/// end rates is monotone (`alpha^2 + beta^2 <= 9`, Fritsch and Carlson),
/// and at a quarter, half and three quarters of the way it stands within
/// [`WALL_ELLIPSE_DEVIATION`] of the ellipse's `(u, v)`.
#[allow(clippy::too_many_lines)]
fn wall_ellipse_pcurve(
    ellipse: &brepkit_math::curves::Ellipse3D,
    surface: &FaceSurface,
    (ta, tb): (f64, f64),
) -> Option<Curve2D> {
    let sense = ellipse_u_sense(ellipse, surface)?;
    let (origin, x, y, axis, v_per_height) = match surface {
        FaceSurface::Cylinder(c) => (c.origin(), c.x_axis(), c.y_axis(), c.axis(), 1.0),
        FaceSurface::Cone(c) => (
            c.apex(),
            c.x_axis(),
            c.y_axis(),
            c.axis(),
            1.0 / c.half_angle().sin(),
        ),
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => return None,
    };
    let span = tb - ta;
    if !span.is_finite() || span.abs() < 1e-12 || !v_per_height.is_finite() {
        return None;
    }
    let way = sense * span.signum();
    // The angle about the axis, its value in `v`, and both rates in `s`.
    let exact = |s: f64| {
        let t = span.madd(s, ta);
        let p = ellipse.evaluate(t) - origin;
        let d = ellipse.tangent(t) * span;
        let (px, py, dx, dy) = (x.dot(p), y.dot(p), x.dot(d), y.dot(d));
        let r2 = px.madd(px, py * py);
        (
            py.atan2(px),
            axis.dot(p) * v_per_height,
            px.madd(dy, -(py * dx)) / r2,
            axis.dot(d) * v_per_height,
        )
    };
    // `u` runs one way along the ellipse: on from `prev` by the advance read
    // that way round, a step of no length read with rounding a hair back.
    let lift = |prev: f64, raw: f64| {
        let ahead = (way * (raw - prev)).rem_euclid(TAU);
        let ahead = if ahead > TAU - 1e-9 {
            ahead - TAU
        } else {
            ahead
        };
        way.madd(ahead, prev)
    };
    let sample = |s: f64, prev_u: f64| {
        let (raw, v, du, dv) = exact(s);
        WallSample {
            s,
            uv: Point2::new(lift(prev_u, raw), v),
            rate: Vec2::new(du, dv),
        }
    };
    let hermite = |a: &WallSample, b: &WallSample, f: f64| {
        let h = b.s - a.s;
        let (f2, f3) = (f * f, f * f * f);
        let (h00, h10, h01, h11) = (
            2.0f64.madd(f3, (-3.0f64).madd(f2, 1.0)),
            (-2.0f64).madd(f2, f3) + f,
            (-2.0f64).madd(f3, 3.0 * f2),
            f3 - f2,
        );
        let at = |p0: f64, m0: f64, p1: f64, m1: f64| {
            h00.madd(p0, h10.madd(h * m0, h01.madd(p1, h11 * h * m1)))
        };
        Point2::new(
            at(a.uv.x(), a.rate.x(), b.uv.x(), b.rate.x()),
            at(a.uv.y(), a.rate.y(), b.uv.y(), b.rate.y()),
        )
    };
    let fits = |a: &WallSample, b: &WallSample| {
        let h = b.s - a.s;
        let run = b.uv.x() - a.uv.x();
        if run.abs() > WALL_ELLIPSE_STEP || run * way < 0.0 {
            return false;
        }
        if run.abs() > 0.0 {
            let (alpha, beta) = (a.rate.x() * h / run, b.rate.x() * h / run);
            if alpha < 0.0 || beta < 0.0 || alpha.madd(alpha, beta * beta) > 9.0 {
                return false;
            }
        }
        [0.25, 0.5, 0.75].into_iter().all(|f| {
            let truth = sample(h.madd(f, a.s), a.uv.x());
            let fit = hermite(a, b, f);
            (fit.x() - truth.uv.x()).abs() <= WALL_ELLIPSE_DEVIATION
                && (fit.y() - truth.uv.y()).abs()
                    <= WALL_ELLIPSE_DEVIATION * truth.uv.y().abs().max(1.0)
        })
    };

    let (u0, v0) = surface.project_point(ellipse.evaluate(ta))?;
    let (_, _, du0, dv0) = exact(0.0);
    let first = WallSample {
        s: 0.0,
        uv: Point2::new(u0, v0),
        rate: Vec2::new(du0, dv0),
    };
    // Even samples to start from, each lifted from the one before.
    let mut even = vec![first];
    for k in 1..=PCURVE_SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let s = k as f64 / PCURVE_SAMPLES as f64;
        let prev = even[even.len() - 1].uv.x();
        even.push(sample(s, prev));
    }
    let mut pending: Vec<WallSample> = even.drain(1..).rev().collect();
    let mut done = even;
    while let Some(b) = pending.pop() {
        let a = done[done.len() - 1];
        if fits(&a, &b) {
            done.push(b);
            continue;
        }
        if b.s - a.s < 1e-12 || done.len() + pending.len() >= WALL_ELLIPSE_MAX_PIECES {
            return None;
        }
        pending.push(b);
        pending.push(sample(f64::midpoint(a.s, b.s), a.uv.x()));
    }

    // Each piece as a Bezier cubic, joined at triple knots.
    let pieces = done.len() - 1;
    let mut points = Vec::with_capacity(3 * pieces + 1);
    let mut knots = vec![0.0; 4];
    points.push(done[0].uv);
    for (k, w) in done.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let third = (b.s - a.s) / 3.0;
        points.push(a.uv + a.rate * third);
        points.push(b.uv - b.rate * third);
        points.push(b.uv);
        if k + 1 < pieces {
            knots.extend([b.s; 3]);
        }
    }
    knots.extend([1.0; 4]);
    let weights = vec![1.0; points.len()];
    NurbsCurve2D::new(3, knots, points, weights)
        .ok()
        .map(Curve2D::Nurbs)
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
        out.push(curve.evaluate_with_endpoints(f.madd(span, t0), start, end));
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
                pts[i] = Point2::new(period.madd(-k, pts[i].x()), pts[i].y());
            }
        }
        if let Some(period) = v_period {
            let dv = pts[i].y() - pts[i - 1].y();
            let k = (dv / period).round();
            if k != 0.0 {
                pts[i] = Point2::new(pts[i].x(), period.madd(-k, pts[i].y()));
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
pub(super) fn fit_nurbs2d_through_points(pts: &[Point2]) -> Curve2D {
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

/// A cone cut by a plane at a slope near its own, and an exact lift of `u`
/// along the ellipse, for tests here and in the face splitter.
#[cfg(test)]
pub(super) mod cone_ellipse_fixture {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use brepkit_math::madd::Madd;
    use brepkit_math::vec::Point3;
    use brepkit_topology::face::FaceSurface;

    /// The cone `x^2 + y^2 = z^2` with its seam turned 1.1 off the plane
    /// `y = 0`, and the closed ellipse where the plane `z = 1 + m x` cuts it,
    /// for a slope `m` under 1. The ellipse's near vertex lies about 0.5 from
    /// the axis and its far one `1 / (1 - m)` out (25 at `m = 0.96`, 1e5 at
    /// `m = 0.99999`), so its `u` runs most of the turn round the near vertex,
    /// over a span of its parameter that narrows as `m` nears 1.
    pub(in crate::builder) fn oblique_cone(
        m: f64,
    ) -> (FaceSurface, brepkit_math::curves::Ellipse3D) {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::{FRAC_PI_4, TAU};

        let cone = ConicalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            FRAC_PI_4,
            Vec3::new(1.1f64.cos(), 1.1f64.sin(), 0.0),
        )
        .unwrap();
        let surface = FaceSurface::Cone(cone);
        let flat = 1.0 - m * m;
        let x0 = m / flat;
        let ellipse = Ellipse3D::new_with_ref(
            Point3::new(x0, 0.0, m.madd(x0, 1.0)),
            Vec3::new(-m, 0.0, 1.0),
            m.hypot(1.0) / flat,
            1.0 / flat.sqrt(),
            Vec3::new(1.0, 0.0, m),
        )
        .unwrap();
        for k in 0..16 {
            let p = ellipse.evaluate(f64::from(k) * TAU / 16.0);
            assert!(
                (p.x().hypot(p.y()) - p.z()).abs() < 1e-9 * p.z(),
                "the ellipse lies off the cone at {p:?}"
            );
        }
        (surface, ellipse)
    }

    /// The `u` of the point of `ellipse` at its parameter `from` on the cone
    /// `surface`, and the `u` the point reaches moving along the ellipse to
    /// its parameter `to`.
    ///
    /// The lift takes no steps, so it cannot skip a turn. Across the axis the
    /// point lies at `(X(t), Y(t))` in the cone's frame, each a sinusoid in
    /// the parameter, and its `u` is the angle of that point. The angle's
    /// principal value jumps by a turn only where the point crosses the half
    /// line `X < 0, Y = 0`, at the roots of `Y(t) = 0`, found in closed form.
    pub(in crate::builder) fn lifted_u(
        surface: &FaceSurface,
        ellipse: &brepkit_math::curves::Ellipse3D,
        from: f64,
        to: f64,
    ) -> (f64, f64) {
        use std::f64::consts::TAU;
        let cone = match surface {
            FaceSurface::Cone(cone) => Some(cone),
            _ => None,
        }
        .expect("a cone");
        let centre = ellipse.center() - cone.apex();
        let (a, b) = (
            ellipse.u_axis() * ellipse.semi_major(),
            ellipse.v_axis() * ellipse.semi_minor(),
        );
        let (x, y) = (cone.x_axis(), cone.y_axis());
        let across = |t: f64| {
            (
                x.dot(centre) + x.dot(a) * t.cos() + x.dot(b) * t.sin(),
                y.dot(centre) + y.dot(a) * t.cos() + y.dot(b) * t.sin(),
            )
        };
        let angle = |t: f64| {
            let (px, py) = across(t);
            py.atan2(px)
        };
        // Y(t) = y0 + r cos(t - phi).
        let (y0, yc, ys) = (y.dot(centre), y.dot(a), y.dot(b));
        let (r, phi) = (yc.hypot(ys), ys.atan2(yc));
        let (lo, hi) = (from.min(to), from.max(to));
        let mut turns = 0.0;
        if y0.abs() < r {
            let alpha = (-y0 / r).acos();
            for base in [phi + alpha, phi - alpha] {
                let mut k = ((lo - base) / TAU).floor();
                loop {
                    let root = TAU.madd(k, base);
                    if root >= hi {
                        break;
                    }
                    if root > lo && across(root).0 < 0.0 {
                        // Y rising through the half line along the motion:
                        // the principal angle jumps a turn up, the true one
                        // runs on down.
                        let rising = (ys * root.cos() - yc * root.sin() > 0.0) == (to > from);
                        turns += if rising { -1.0 } else { 1.0 };
                    }
                    k += 1.0;
                }
            }
        }
        let (u0, _) = surface.project_point(ellipse.evaluate(from)).unwrap();
        (u0, TAU.madd(turns, u0 + angle(to) - angle(from)))
    }

    /// Check that `pcurve` follows `ellipse` on the cone `surface` as the
    /// ellipse's parameter runs evenly from `ta` to `tb` over `[0, 1]`, on
    /// some copy of the period: at 4001 even points its `(u, v)` stands
    /// within 2e-6 of the exact lift ([`lifted_u`]) in `u`, and within
    /// 2e-6 of `max(1, |v|)` in `v`; its `u` never steps back against the
    /// way the ellipse's `u` runs; and it leaves each end that way. Returns
    /// the largest deviation in `u` and in relative `v`.
    pub(in crate::builder) fn assert_pcurve_follows(
        pcurve: &brepkit_math::curves2d::Curve2D,
        surface: &FaceSurface,
        ellipse: &brepkit_math::curves::Ellipse3D,
        (ta, tb): (f64, f64),
        what: &str,
    ) -> (f64, f64) {
        use std::f64::consts::TAU;
        const N: u32 = 4000;
        let (u_start, u_end) = lifted_u(surface, ellipse, ta, tb);
        let way = (u_end - u_start).signum();
        let first = pcurve.evaluate(0.0);
        let last = pcurve.evaluate(1.0);
        let copy = ((first.x() - u_start) / TAU).round() * TAU;
        let mut worst = (0.0_f64, 0.0_f64);
        let mut prev: Option<f64> = None;
        for k in 0..=N {
            let s = f64::from(k) / f64::from(N);
            let t = (tb - ta).madd(s, ta);
            let (_, u) = lifted_u(surface, ellipse, ta, t);
            let (_, v) = surface.project_point(ellipse.evaluate(t)).unwrap();
            let p = pcurve.evaluate(s);
            let (du, dv) = (
                (p.x() - copy - u).abs(),
                (p.y() - v).abs() / v.abs().max(1.0),
            );
            worst = (worst.0.max(du), worst.1.max(dv));
            assert!(
                du <= 2e-6 && dv <= 2e-6,
                "{what}: at {s} the pcurve stands at {p:?}, the ellipse at ({}, {v})",
                u + copy
            );
            if let Some(q) = prev {
                assert!(
                    (p.x() - q) * way >= -1e-12,
                    "{what}: at {s} the pcurve's u steps back from {q} to {}",
                    p.x()
                );
            }
            prev = Some(p.x());
        }
        for d in [1e-9, 1e-6, 1e-2] {
            assert!(
                (pcurve.evaluate(d).x() - first.x()) * way > 0.0,
                "{what}: the pcurve leaves its start against u, {d} along"
            );
            assert!(
                (last.x() - pcurve.evaluate(1.0 - d).x()) * way > 0.0,
                "{what}: the pcurve reaches its end against u, {d} short"
            );
        }
        worst
    }
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

    /// A closed curve round a thin cylinder, rising and falling a few times
    /// the radius over the turn, as where a strut leaves a chord. Its pcurve
    /// runs round the axis one way all the way: read with chords in UV, the
    /// fit ran back across the seam from its first sample.
    #[test]
    fn a_closed_curve_round_a_thin_wall_runs_one_way_in_its_pcurve() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::EdgeCurve;

        let radius = 8.0;
        let cylinder =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), radius)
                .unwrap();
        let x = cylinder.x_axis();
        let y = cylinder.y_axis();
        let surface = FaceSurface::Cylinder(cylinder);
        // The curve starts 0.6 of a radian before its highest point.
        let start_u = 1.0_f64;
        let at = |u: f64| {
            let h = 30.0 + 6.0 * (u - start_u - 0.6).cos();
            Point3::new(0.0, 0.0, h) + x * (radius * u.cos()) + y * (radius * u.sin())
        };
        let points: Vec<Point3> = (0..=64)
            .map(|k| at(start_u + TAU * f64::from(k) / 64.0))
            .collect();
        let nurbs = brepkit_math::nurbs::fitting::interpolate(&points, 3).unwrap();
        let curve = EdgeCurve::NurbsCurve(nurbs);
        let vertex = points[0];
        let pcurve = compute_pcurve_on_surface(&curve, vertex, vertex, &surface, &[], None);
        let knots = match &pcurve {
            Curve2D::Nurbs(n) => n.knots().to_vec(),
            other => vec![0.0, other.evaluate(0.0).x()],
        };
        assert!(matches!(pcurve, Curve2D::Nurbs(_)));
        let (t0, t1) = (knots[0], knots[knots.len() - 1]);
        let us: Vec<f64> = (0..=400)
            .map(|k| pcurve.evaluate(t0 + (t1 - t0) * f64::from(k) / 400.0).x())
            .collect();
        let way = (us[400] - us[0]).signum();
        assert!(((us[400] - us[0]).abs() - TAU).abs() < 1e-6);
        for w in us.windows(2) {
            assert!(
                (w[1] - w[0]) * way > 0.0,
                "the pcurve turns back at u = {:.4} on its way from {:.4}",
                w[0],
                us[0]
            );
        }
    }

    /// The ellipse of [`cone_ellipse_fixture::oblique_cone`] at slopes 0.96,
    /// 0.99 and 0.99999, as a closed rim from its parameter 2, and an open
    /// arc of it from parameter 3 to 3.4 across its near vertex, both ways
    /// round. At slope 0.99999 nearly the whole turn of `u` lies within a
    /// hundredth of the parameter round the near vertex, between two even
    /// samples. The rim's samples run one way through one turn. The rim's
    /// pcurve and the arc's follow the ellipse's `(u, v)` along their whole
    /// length, with `u` running one way, and leave each end that way
    /// ([`cone_ellipse_fixture::assert_pcurve_follows`]).
    #[test]
    fn samples_along_a_steep_oblique_cone_ellipse_keep_its_turn() {
        use super::cone_ellipse_fixture::{assert_pcurve_follows, lifted_u, oblique_cone};
        use brepkit_topology::edge::EdgeCurve;
        use std::f64::consts::TAU;

        for m in [0.96, 0.99, 0.99999] {
            let (surface, ellipse) = oblique_cone(m);
            let curve = EdgeCurve::Ellipse(ellipse.clone());
            let start = ellipse.evaluate(2.0);
            let us: Vec<f64> = sample_edge_to_uv(&curve, start, start, &surface)
                .iter()
                .map(|p| p.x())
                .collect();
            let (_, want) = lifted_u(&surface, &ellipse, 2.0, 2.0 + TAU);
            let turn = us[us.len() - 1] - us[0];
            assert!(
                (turn - (want - us[0])).abs() < 1e-9,
                "slope {m}: the rim's samples run {turn}, the rim {}",
                want - us[0]
            );
            let steps: Vec<f64> = us.windows(2).map(|w| w[1] - w[0]).collect();
            assert!(
                steps.iter().all(|&d| d * turn > 0.0),
                "slope {m}: the rim's u turns back: {us:?}"
            );
            let rim = compute_pcurve_on_surface(&curve, start, start, &surface, &[], None);
            assert!(
                (rim.evaluate(0.0).x() - us[0]).abs() < 1e-12,
                "slope {m}: the rim's pcurve leaves its samples' start"
            );
            assert_pcurve_follows(
                &rim,
                &surface,
                &ellipse,
                (2.0, 2.0 + TAU),
                &format!("slope {m}, rim"),
            );
            let (a, b) = (ellipse.evaluate(3.0), ellipse.evaluate(3.4));
            for forward in [true, false] {
                let pcurve =
                    compute_boundary_pcurve_on_surface(&curve, a, b, forward, &surface, &[], None);
                let span = if forward { (3.0, 3.4) } else { (3.4, 3.0) };
                let (u_from, u_to) = lifted_u(&surface, &ellipse, span.0, span.1);
                let (first, last) = (pcurve.evaluate(0.0).x(), pcurve.evaluate(1.0).x());
                assert!(
                    (first - u_from).abs() < 1e-9 && (last - u_to).abs() < 1e-9,
                    "slope {m}, forward {forward}: the pcurve runs {first} to {last}, \
                     the arc {u_from} to {u_to}"
                );
                assert_pcurve_follows(
                    &pcurve,
                    &surface,
                    &ellipse,
                    span,
                    &format!("slope {m}, arc forward {forward}"),
                );
            }
        }
    }
}
