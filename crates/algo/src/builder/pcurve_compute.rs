//! PCurve computation: project 3D edge curves into a face's (u,v) parameter space.
//!
//! For plane faces, uses [`PlaneFrame`] to project 3D lines to 2D lines.
//! For analytic surfaces (cylinder, cone, sphere, torus), samples points along
//! the 3D curve, projects via `surface.project_point()`, unwraps periodicity,
//! and fits a [`NurbsCurve2D`], or gives a `Line2D` where a bound shows that
//! the straight run in UV lies on the edge in 3D.

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
    let owned;
    let plane_frame = if let FaceSurface::Plane { normal, .. } = surface {
        Some(if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        })
    } else {
        None
    };
    let (pts_3d, uv_pts) = if let Some(f) = plane_frame {
        sample_edge_via_frame(curve_3d, start, end, f)
    } else {
        sample_edge_on_surface(curve_3d, start, end, surface)
    };
    let straight = straight_pcurve(
        &uv_pts,
        &pts_3d,
        curve_3d,
        (start, end),
        surface,
        plane_frame,
    );
    if straight.is_none()
        && let Some(pcurve) = proved_conic_pcurve(curve_3d, &pts_3d, surface, plane_frame)
    {
        return pcurve;
    }
    if straight.is_none()
        && let EdgeCurve::Ellipse(ellipse) = curve_3d
        && let Some(pcurve) = wall_ellipse_pcurve(
            ellipse,
            surface,
            ellipse_sample_span(ellipse, start, end, surface),
        )
    {
        return unproved(pcurve, "wall ellipse");
    }
    pcurve_through_uv_samples(uv_pts, &pts_3d, start, end, surface, straight)
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
    let owned;
    let plane_frame = if let FaceSurface::Plane { normal, .. } = surface {
        Some(if let Some(fr) = frame {
            fr
        } else {
            owned = PlaneFrame::from_plane_face(*normal, wire_pts);
            &owned
        })
    } else {
        None
    };
    let uv_pts = if let Some(f) = plane_frame {
        pts_3d.iter().map(|&p| f.project(p)).collect()
    } else {
        let mut uv: Vec<Point2> = pts_3d
            .iter()
            .map(|&p| {
                let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
                Point2::new(u, v)
            })
            .collect();
        lift_edge_samples(curve_3d, surface, &pts_3d, &mut uv);
        uv
    };
    let straight = straight_pcurve(
        &uv_pts,
        &pts_3d,
        curve_3d,
        (start, end),
        surface,
        plane_frame,
    );
    if straight.is_none()
        && let Some(pcurve) = proved_conic_pcurve(curve_3d, &pts_3d, surface, plane_frame)
    {
        return pcurve;
    }
    if straight.is_none()
        && let EdgeCurve::Ellipse(ellipse) = curve_3d
    {
        let (t0, t1) = curve_3d.domain_with_endpoints(stored_start, stored_end);
        let span = if forward { (t0, t1) } else { (t1, t0) };
        if let Some(pcurve) = wall_ellipse_pcurve(ellipse, surface, span) {
            return unproved(pcurve, "wall ellipse");
        }
    }
    pcurve_through_uv_samples(uv_pts, &pts_3d, start, end, surface, straight)
}

/// A circle or ellipse arc in a plane's frame: the frame maps the conic
/// `c + cos t a + sin t b` affinely, so its image is the conic through the
/// projected centre and axes, built as rational quadratic pieces of at most
/// a quarter turn, from the traversal's start to its end on [0, 1]. The
/// span comes from the stored ends, read against the conic where they lie,
/// so its bound in [`plane_conic_span`] takes no reserve.
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
    let (t0, t1) = curve_3d.domain_with_endpoints(stored_start, stored_end);
    plane_conic_span(
        curve_3d,
        if forward { (t0, t1) } else { (t1, t0) },
        frame,
        0.0,
    )
}

/// [`plane_conic_arc`] over the span of the conic's own parameter from `ta`
/// to `tb`, where its bound holds.
///
/// The axes go into the frame as vectors, so their rounding goes with their
/// sizes, not with where the conic lies. The image then lies within
/// `reserve + 64 ε size` of the conic's shadow on the frame's plane, `size`
/// the centre's distance from the frame's origin plus the lengths of the two
/// axes: the frame's axes are square to each other within a few ε, and the
/// projections, the angles, the weights and the control points round within
/// that. `reserve` is what the caller's reading of the span and of the
/// conic adds. `None` where the bound passes the linear tolerance.
fn plane_conic_span(
    curve_3d: &EdgeCurve,
    (ta, tb): (f64, f64),
    frame: &PlaneFrame,
    reserve: f64,
) -> Option<Curve2D> {
    let (centre, (big_a, big_b)) = match curve_3d {
        EdgeCurve::Circle(c) => (
            c.center(),
            (c.u_axis() * c.radius(), c.v_axis() * c.radius()),
        ),
        EdgeCurve::Ellipse(e) => (
            e.center(),
            (e.u_axis() * e.semi_major(), e.v_axis() * e.semi_minor()),
        ),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
    };
    let flat = |w: Vec3| Vec2::new(w.dot(frame.u_axis()), w.dot(frame.v_axis()));
    let c = frame.project(centre);
    let (a, b) = (flat(big_a), flat(big_b));
    let sweep = tb - ta;
    if !sweep.is_finite() || sweep.abs() < 1e-12 {
        return None;
    }
    let size = (centre - frame.origin()).length() + big_a.length() + big_b.length();
    let bound = (64.0 * f64::EPSILON).madd(size, reserve);
    if bound.is_nan() || bound > brepkit_math::tolerance::Tolerance::new().linear {
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

/// A pcurve through UV samples running from `start` to `end`: `straight`
/// when there is one (see [`straight_pcurve`]), else a fitted NURBS.
/// `pts_3d` are the samples on the edge, one for each UV sample, or none.
fn pcurve_through_uv_samples(
    uv_pts: Vec<Point2>,
    pts_3d: &[Point3],
    start: Point3,
    end: Point3,
    surface: &FaceSurface,
    straight: Option<Curve2D>,
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
    if let Some(line) = straight {
        return line;
    }

    if matches!(surface, FaceSurface::Plane { .. }) {
        return unproved(fit_nurbs2d_through_points(&uv_pts), "fit on a plane");
    }
    unproved(fit_nurbs2d_at_chords_3d(&uv_pts, pts_3d), "fit")
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

    lift_edge_samples(curve_3d, surface, &pts_3d, &mut uv_pts);

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

/// The `u` of a point of `ellipse` on a cylinder or a cone `surface`
/// reached by a run `run` of the ellipse's parameter on from a point whose
/// `u` is `u_from`; `u_raw` is the point's own projected `u`. `None` when
/// [`ellipse_u_sense`] is.
///
/// `u` runs strictly one way along the ellipse, so a run of the parameter
/// under a turn advances it by less than a turn, the way the run goes: the
/// point lies the projected value's advance on, read that way round. Within
/// rounding of the start's ruling, the run says which end: none, or a turn.
pub(super) fn ellipse_u_after(
    ellipse: &brepkit_math::curves::Ellipse3D,
    surface: &FaceSurface,
    (u_from, u_raw): (f64, f64),
    run: f64,
) -> Option<f64> {
    use std::f64::consts::PI;
    let sense = ellipse_u_sense(ellipse, surface)?;
    let way = sense * run.signum();
    let ahead = (way * (u_raw - u_from)).rem_euclid(TAU);
    let ahead = if ahead < 1e-9 && run.abs() > PI {
        ahead + TAU
    } else if ahead > TAU - 1e-9 && run.abs() < PI {
        ahead - TAU
    } else {
        ahead
    };
    Some(way.madd(ahead, u_from))
}

/// Lift UV samples `uv`, the projections of points `pts_3d` taken in order
/// along one edge whose curve is `curve`, onto the copies of the periods the
/// edge reaches them on, from the first sample's copy.
///
/// Along an ellipse on a cylinder or a cone, `u` runs one way, so each step
/// advances it by [`ellipse_u_after`], however far: round a steep cut's near
/// vertex `u` runs most of a turn between two samples, which a step to the
/// nearest copy reads as a short step back. Along any other curve each
/// sample goes to the copy nearest the one before. The samples lie under
/// half a turn of the curve's parameter apart.
pub(super) fn lift_edge_samples(
    curve: &EdgeCurve,
    surface: &FaceSurface,
    pts_3d: &[Point3],
    uv: &mut [Point2],
) {
    let (u_period, v_period) = surface_periods(surface);
    unwrap_periodic_params(uv, u_period, v_period);
    let EdgeCurve::Ellipse(ellipse) = curve else {
        return;
    };
    if pts_3d.len() != uv.len() || ellipse_u_sense(ellipse, surface).is_none() {
        return;
    }
    for k in 1..uv.len() {
        let step = shorter_arc_delta(ellipse.project(pts_3d[k]) - ellipse.project(pts_3d[k - 1]));
        if let Some(u) = ellipse_u_after(ellipse, surface, (uv[k - 1].x(), uv[k].x()), step) {
            uv[k] = Point2::new(u, uv[k].y());
        }
    }
}

/// The UV samples of points `pts_3d` taken in order along one edge whose
/// curve is `curve`, lifted as [`lift_edge_samples`] lifts them from the
/// first point's own projection. `None` when a point does not project.
pub(super) fn edge_uv_samples(
    curve: &EdgeCurve,
    surface: &FaceSurface,
    pts_3d: &[Point3],
) -> Option<Vec<Point2>> {
    let mut uv: Vec<Point2> = pts_3d
        .iter()
        .map(|&p| surface.project_point(p).map(|(u, v)| Point2::new(u, v)))
        .collect::<Option<_>>()?;
    lift_edge_samples(curve, surface, pts_3d, &mut uv);
    Some(uv)
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
/// of `max(1, r)`, `r` the point's distance from the axis. `v` is a length
/// along the wall's ruling, on a cylinder and on a cone, so its error is a
/// distance in model units; `r`, unlike `v`, does not move with a
/// cylinder's parameter origin along its axis.
const WALL_ELLIPSE_DEVIATION: f64 = 1e-6;
/// The most pieces a wall ellipse's pcurve takes before the fit gives up.
const WALL_ELLIPSE_MAX_PIECES: usize = 4096;

/// A sample of an ellipse on a wall: its parameter `s` on `[0, 1]`, its
/// `(u, v)` with `u` lifted along the ellipse, their rates in `s`, and its
/// distance from the axis.
#[derive(Clone, Copy)]
struct WallSample {
    s: f64,
    uv: Point2,
    rate: Vec2,
    radius: f64,
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
    // The angle about the axis, its value in `v`, both rates in `s`, and
    // the distance from the axis.
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
            r2.sqrt(),
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
        let (raw, v, du, dv, radius) = exact(s);
        WallSample {
            s,
            uv: Point2::new(lift(prev_u, raw), v),
            rate: Vec2::new(du, dv),
            radius,
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
                && (fit.y() - truth.uv.y()).abs() <= WALL_ELLIPSE_DEVIATION * truth.radius.max(1.0)
        })
    };

    let (u0, v0) = surface.project_point(ellipse.evaluate(ta))?;
    let (_, _, du0, dv0, radius0) = exact(0.0);
    let first = WallSample {
        s: 0.0,
        uv: Point2::new(u0, v0),
        rate: Vec2::new(du0, dv0),
        radius: radius0,
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

/// The most pieces a proved pcurve takes before it gives up.
const PROVED_PCURVE_MAX_PIECES: usize = 4096;

/// A pcurve for a line, circle or ellipse edge through the samples
/// `pts_3d`, whose image is proved within the linear tolerance of the edge
/// at every point, or `None`. On a plane it is the conic's image in the
/// frame ([`plane_conic_span`]), within the tolerance of the edge's shadow
/// on the plane; on a cylinder, cone, ball or ring it is a
/// [`proved_hermite_pcurve`].
fn proved_conic_pcurve(
    curve: &EdgeCurve,
    pts_3d: &[Point3],
    surface: &FaceSurface,
    frame: Option<&PlaneFrame>,
) -> Option<Curve2D> {
    let (curve, gap) = unit_conic(curve)?;
    let curve = &curve;
    // The pcurve follows the unit form's arc between the points nearest the
    // first and last samples, within the arc gap of the stored arc, which
    // may take at most a quarter of the tolerance.
    let off_form = unit_conic_arc_gap(gap);
    if off_form > 0.25 * brepkit_math::tolerance::Tolerance::new().linear {
        return None;
    }
    if let Some(f) = frame {
        // The span comes from the samples, each rounded where it lies.
        let (theta0, sweep) = conic_sweep(curve, pts_3d)?;
        let reserve = off_form + sample_rounding(pts_3d);
        return plane_conic_span(curve, (theta0, theta0 + sweep), f, reserve);
    }
    proved_hermite_pcurve(curve, pts_3d, surface, off_form)
}

/// A line, circle or ellipse edge along its samples, by the fraction `s` of
/// their run from first to last.
enum EdgeRun<'a> {
    Line(Point3, Point3),
    Circle(&'a brepkit_math::curves::Circle3D, f64, f64),
    Ellipse(&'a brepkit_math::curves::Ellipse3D, f64, f64),
}

impl EdgeRun<'_> {
    /// The edge's point at `s` and its rate in `s`.
    fn at(&self, s: f64) -> (Point3, Vec3) {
        match self {
            Self::Line(a, b) => (*a + (*b - *a) * s, *b - *a),
            Self::Circle(c, t0, sweep) => {
                let t = sweep.madd(s, *t0);
                (c.evaluate(t), c.tangent(t) * (c.radius() * sweep))
            }
            Self::Ellipse(e, t0, sweep) => {
                let t = sweep.madd(s, *t0);
                (e.evaluate(t), e.tangent(t) * *sweep)
            }
        }
    }

    /// The parameter the edge's own curve runs over its samples.
    const fn sweep(&self) -> f64 {
        match self {
            Self::Line(..) => 0.0,
            Self::Circle(_, _, w) | Self::Ellipse(_, _, w) => *w,
        }
    }

    /// A bound on the norm of the edge's fourth derivative in `s`: the
    /// radius, or the longer semi-axis, times the sweep to the fourth.
    fn fourth(&self) -> f64 {
        match self {
            Self::Line(..) => 0.0,
            Self::Circle(c, _, sweep) => c.radius() * sweep.powi(4),
            Self::Ellipse(e, _, sweep) => e.semi_major() * sweep.powi(4),
        }
    }
}

/// An analytic surface's point at `(u, v)` and its two partial
/// derivatives there. `None` on a plane or a B-spline surface.
fn surface_point_and_rates(surface: &FaceSurface, u: f64, v: f64) -> Option<(Point3, Vec3, Vec3)> {
    use brepkit_math::traits::ParametricSurface;
    fn jet(s: &impl ParametricSurface, u: f64, v: f64) -> (Point3, Vec3, Vec3) {
        (s.evaluate(u, v), s.partial_u(u, v), s.partial_v(u, v))
    }
    match surface {
        FaceSurface::Cylinder(c) => Some(jet(c, u, v)),
        FaceSurface::Cone(c) => Some(jet(c, u, v)),
        FaceSurface::Sphere(s) => Some(jet(s, u, v)),
        FaceSurface::Torus(t) => Some(jet(t, u, v)),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
    }
}

/// A bound on the norm of an analytic surface's partial derivative taken
/// `i` times in `u` and `j` times in `v`, `i + j >= 2`, over the band of `v`
/// whose largest `|v|` is `v_most`. A rod's are its radius when `j = 0` and
/// nothing else; a cone's `|v| cos a` when `j = 0`, `cos a` when `j = 1` and
/// nothing else; a ball's its radius; a ring's `R + r` when `j = 0` and `r`
/// else. `None` on a plane or a B-spline surface.
fn partial_bound(surface: &FaceSurface, i: usize, j: usize, v_most: f64) -> Option<f64> {
    match surface {
        FaceSurface::Cylinder(c) => Some(if j == 0 { c.radius() } else { 0.0 }),
        FaceSurface::Cone(c) => Some(match j {
            0 => c.half_angle().cos() * v_most,
            1 if i > 0 => c.half_angle().cos(),
            _ => 0.0,
        }),
        FaceSurface::Sphere(s) => Some(s.radius()),
        FaceSurface::Torus(t) => Some(if j == 0 {
            t.major_radius() + t.minor_radius()
        } else {
            t.minor_radius()
        }),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
    }
}

/// A bound on `|D^k S [w_1, ..., w_k]|`, `k = slots.len()` from two to four,
/// each `w` with `|u|` and `|v|` parts at most the slot's pair: the sum over
/// each choice of the slots read along `v` of the bound on that partial
/// ([`partial_bound`]) times the parts chosen. `None` where
/// [`partial_bound`] is.
fn multilinear_bound(surface: &FaceSurface, slots: &[(f64, f64)], v_most: f64) -> Option<f64> {
    let k = slots.len();
    let mut total = 0.0;
    for chosen in 0..(1_usize << k) {
        let j = chosen.count_ones() as usize;
        let mut term = partial_bound(surface, k - j, j, v_most)?;
        for (slot, &(along_u, along_v)) in slots.iter().enumerate() {
            term *= if chosen >> slot & 1 == 1 {
                along_v
            } else {
                along_u
            };
        }
        total += term;
    }
    Some(total)
}

/// The rounding of an edge's 3D samples where they lie: `2 ε` times their
/// largest coordinate. Each coordinate of a sample rounds within `ε / 2` of
/// its size, so a sample stands within `sqrt(3) ε / 2` times the largest
/// coordinate of its exact place. A bound reads that shift at most twice:
/// once at the end of a chord and once at the end of the arc.
fn sample_rounding(pts_3d: &[Point3]) -> f64 {
    let far = pts_3d.iter().fold(0.0_f64, |m, p| {
        [p.x(), p.y(), p.z()]
            .into_iter()
            .fold(m, |m, c| max_or_nan(m, c.abs()))
    });
    2.0 * f64::EPSILON * far
}

/// `surface`, `curve` and `points` moved by one offset so the surface's own
/// origin lies at nothing, with that offset. The surface keeps its frame
/// and its (u, v), and the curve its axes and sizes. Each moved point
/// rounds within `ε / 2` of its new size. `None` on a plane or a B-spline
/// surface, and for a B-spline edge.
fn moved_to_surface_origin(
    surface: &FaceSurface,
    curve: &EdgeCurve,
    points: &[Point3],
) -> Option<(FaceSurface, EdgeCurve, Vec<Point3>, Vec3)> {
    let offset = Point3::new(0.0, 0.0, 0.0) - surface_origin(surface)?;
    let surface = surface_moved(surface, offset)?;
    let curve = conic_moved(curve, offset)?;
    let points = points.iter().map(|&p| p + offset).collect();
    Some((surface, curve, points, offset))
}

/// `pcurve`, with no proof that its image lies within the tolerance of its
/// edge. The perf counters count it and the log records it at debug level,
/// so a run can find each one.
fn unproved(pcurve: Curve2D, how: &str) -> Curve2D {
    crate::perf::bump_unproved_pcurve();
    log::debug!("unproved pcurve: {how}");
    pcurve
}

/// An analytic surface's own origin: a rod's axis point, a cone's apex, a
/// ball's or a ring's centre. `None` on a plane or a B-spline surface.
fn surface_origin(surface: &FaceSurface) -> Option<Point3> {
    match surface {
        FaceSurface::Cylinder(c) => Some(c.origin()),
        FaceSurface::Cone(c) => Some(c.apex()),
        FaceSurface::Sphere(s) => Some(s.center()),
        FaceSurface::Torus(t) => Some(t.center()),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
    }
}

/// An analytic surface moved by `offset`, its frame and its (u, v) kept.
/// `None` on a plane or a B-spline surface.
fn surface_moved(surface: &FaceSurface, offset: Vec3) -> Option<FaceSurface> {
    match surface {
        FaceSurface::Cylinder(c) => Some(FaceSurface::Cylinder(c.translated(offset))),
        FaceSurface::Cone(c) => Some(FaceSurface::Cone(c.translated(offset))),
        FaceSurface::Sphere(s) => Some(FaceSurface::Sphere(s.translated(offset))),
        FaceSurface::Torus(t) => Some(FaceSurface::Torus(t.translated(offset))),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
    }
}

/// A line, circle or ellipse edge moved by `offset`, its axes and sizes
/// kept (a line holds no points of its own). `None` for a B-spline.
fn conic_moved(curve: &EdgeCurve, offset: Vec3) -> Option<EdgeCurve> {
    match curve {
        EdgeCurve::Line => Some(EdgeCurve::Line),
        EdgeCurve::Circle(c) => brepkit_math::curves::Circle3D::with_axes(
            c.center() + offset,
            c.normal(),
            c.radius(),
            c.u_axis(),
            c.v_axis(),
        )
        .ok()
        .map(EdgeCurve::Circle),
        EdgeCurve::Ellipse(e) => brepkit_math::curves::Ellipse3D::with_axes(
            e.center() + offset,
            e.normal(),
            e.semi_major(),
            e.semi_minor(),
            e.u_axis(),
            e.v_axis(),
        )
        .ok()
        .map(EdgeCurve::Ellipse),
        EdgeCurve::NurbsCurve(_) => None,
    }
}

/// A node of a [`proved_hermite_pcurve`]: the fraction `s` along the edge,
/// the edge's (u, v) there and its rate in `s`, how far the surface at that
/// (u, v) stands off the edge, and how far the surface's rate along the
/// node's rate stands off the edge's rate.
#[derive(Clone, Copy)]
struct HermiteNode {
    s: f64,
    uv: Point2,
    rate: Vec2,
    gap: f64,
    gap_rate: f64,
}

/// A pcurve of cubic pieces for a line, circle or ellipse edge on a
/// cylinder, cone, ball or ring, each piece matching the edge's (u, v) and
/// their rates at its ends, split until a bound proves its image within the
/// linear tolerance of the edge. `None` past [`PROVED_PCURVE_MAX_PIECES`],
/// for another curve or surface, or where a node does not evaluate.
///
/// On a piece of length `h` let `f` be its cubic and `g(s) = S(f(s)) - C(s)`,
/// `S` the surface and `C` the edge at the same fraction. `g` stands within
/// `h^4 / 384` times a bound on its fourth derivative of its cubic Hermite
/// interpolant from its ends (the error's Peano kernel does not change
/// sign), and that interpolant stays within the larger end gap plus
/// `4 h / 27` times the two end rate gaps. A cubic's fourth derivative is
/// nothing, so by Faa di Bruno the image's fourth derivative is
/// `D^4 S [f', f', f', f'] + 6 D^3 S [f', f', f''] + 4 D^2 S [f', f''']
/// + 3 D^2 S [f'', f'']`, each bounded by [`multilinear_bound`] from the
/// most `|u|` and `|v|` parts of the cubic's derivatives over the piece,
/// read from their Bezier points. The edge's fourth derivative is at most
/// [`EdgeRun::fourth`].
fn proved_hermite_pcurve(
    curve: &EdgeCurve,
    pts_3d: &[Point3],
    surface: &FaceSurface,
    off_form: f64,
) -> Option<Curve2D> {
    // The proof runs with the surface's own origin at nothing, so its
    // rounding goes with the surface's size and the edge's reach from it.
    // The arc read from the first and last samples can end up to
    // [`sample_rounding`] off the stored edge's ends: the one term that goes
    // with where they lie, counted once.
    let reserve = sample_rounding(pts_3d);
    let (surface, curve, pts_3d, _) = moved_to_surface_origin(surface, curve, pts_3d)?;
    let (surface, curve, pts_3d) = (&surface, &curve, &pts_3d[..]);
    let run = match curve {
        EdgeCurve::Line => EdgeRun::Line(*pts_3d.first()?, *pts_3d.last()?),
        EdgeCurve::Circle(c) => {
            let (t0, sweep) = conic_sweep(curve, pts_3d)?;
            EdgeRun::Circle(c, t0, sweep)
        }
        EdgeCurve::Ellipse(e) => {
            let (t0, sweep) = conic_sweep(curve, pts_3d)?;
            EdgeRun::Ellipse(e, t0, sweep)
        }
        EdgeCurve::NurbsCurve(_) => return None,
    };
    partial_bound(surface, 2, 0, 0.0)?;
    let (u_period, v_period) = surface_periods(surface);
    let size = pts_3d.iter().fold(1.0_f64, |m, p| {
        [p.x(), p.y(), p.z()]
            .into_iter()
            .fold(m, |m, c| max_or_nan(m, c.abs()))
    });
    let allowed = 64.0f64.madd(
        -f64::EPSILON * size,
        brepkit_math::tolerance::Tolerance::new().linear - reserve,
    ) - off_form;
    if allowed.is_nan() || allowed <= 0.0 {
        return None;
    }
    let nearest = |raw: f64, want: f64, period: Option<f64>| match period {
        Some(p) => p.madd(((want - raw) / p).round(), raw),
        None => raw,
    };
    // The node at `s`, its (u, v) lifted onto the copies of the periods it
    // reaches running on from `from` at that node's rate.
    let node = |s: f64, from: Option<&HermiteNode>| -> Option<HermiteNode> {
        let (p, dp) = run.at(s);
        let (u_raw, v_raw) = surface.project_point(p)?;
        let (u, v) = match from {
            None => (u_raw, v_raw),
            Some(a) => {
                let guess = a.uv + a.rate * (s - a.s);
                let u = match curve {
                    EdgeCurve::Ellipse(e) => {
                        ellipse_u_after(e, surface, (a.uv.x(), u_raw), run.sweep() * (s - a.s))
                    }
                    EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::NurbsCurve(_) => None,
                }
                .unwrap_or_else(|| nearest(u_raw, guess.x(), u_period));
                (u, nearest(v_raw, guess.y(), v_period))
            }
        };
        let (q, su, sv) = surface_point_and_rates(surface, u, v)?;
        let (nu, nv) = (su.length_squared(), sv.length_squared());
        if nu.is_nan() || nv.is_nan() || nu <= 0.0 || nv <= 0.0 {
            return None;
        }
        let rate = Vec2::new(su.dot(dp) / nu, sv.dot(dp) / nv);
        let node = HermiteNode {
            s,
            uv: Point2::new(u, v),
            rate,
            gap: (q - p).length(),
            gap_rate: (su * rate.x() + sv * rate.y() - dp).length(),
        };
        [u, v, rate.x(), rate.y(), node.gap, node.gap_rate]
            .iter()
            .all(|x| x.is_finite())
            .then_some(node)
    };
    // The most `|u|` and `|v|` parts among vectors.
    let parts = |ws: &[Vec2]| {
        ws.iter().fold((0.0_f64, 0.0_f64), |(u, v), w| {
            (max_or_nan(u, w.x().abs()), max_or_nan(v, w.y().abs()))
        })
    };
    let bound = |a: &HermiteNode, b: &HermiteNode| -> f64 {
        let h = b.s - a.s;
        // The piece's Bezier points less its first: `e1`, `e2`, `e3`.
        let e1 = a.rate * (h / 3.0);
        let e3 = b.uv - a.uv;
        let e2 = e3 - b.rate * (h / 3.0);
        let first = parts(&[a.rate, b.rate, (e2 - e1) * (3.0 / h)]);
        let second = parts(&[
            (e2 - e1 * 2.0) * (6.0 / (h * h)),
            (e3 - e2 * 2.0 + e1) * (6.0 / (h * h)),
        ]);
        let third = parts(&[(e3 - e2 * 3.0 + e1 * 3.0) * (6.0 / (h * h * h))]);
        let v_most = [a.uv.y(), a.uv.y() + e1.y(), a.uv.y() + e2.y(), b.uv.y()]
            .iter()
            .fold(0.0_f64, |m, &v| max_or_nan(m, v.abs()));
        let terms = [
            (1.0, vec![first; 4]),
            (6.0, vec![first, first, second]),
            (4.0, vec![first, third]),
            (3.0, vec![second, second]),
        ];
        let mut image = 0.0;
        for (count, slots) in &terms {
            let Some(term) = multilinear_bound(surface, slots, v_most) else {
                return f64::NAN;
            };
            image = count.madd(term, image);
        }
        let ends = (4.0 / 27.0 * h).madd(a.gap_rate + b.gap_rate, max_or_nan(a.gap, b.gap));
        (h.powi(4) / 384.0).madd(image + run.fourth(), ends)
    };

    // Even nodes to start from, each lifted from the one before.
    let mut even = vec![node(0.0, None)?];
    for k in 1..=PCURVE_SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let s = k as f64 / PCURVE_SAMPLES as f64;
        let before = even[even.len() - 1];
        even.push(node(s, Some(&before))?);
    }
    let mut pending: Vec<HermiteNode> = even.drain(1..).rev().collect();
    let mut done = even;
    while let Some(b) = pending.pop() {
        let a = done[done.len() - 1];
        let err = bound(&a, &b);
        if err.is_finite() && err <= allowed {
            done.push(b);
            continue;
        }
        if b.s - a.s < 1e-9 || done.len() + pending.len() >= PROVED_PCURVE_MAX_PIECES {
            return None;
        }
        pending.push(b);
        pending.push(node(f64::midpoint(a.s, b.s), Some(&a))?);
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
fn sample_edge_via_frame(
    curve_3d: &EdgeCurve,
    start: Point3,
    end: Point3,
    frame: &PlaneFrame,
) -> (Vec<Point3>, Vec<Point2>) {
    let n = PCURVE_SAMPLES;
    let at = edge_point_at(curve_3d, start, end);
    #[allow(clippy::cast_precision_loss)]
    let pts_3d: Vec<Point3> = (0..=n).map(|i| at(i as f64 / n as f64)).collect();
    let uv_pts = pts_3d.iter().map(|&p| frame.project(p)).collect();
    (pts_3d, uv_pts)
}

/// The straight pcurve from the first UV sample to the last, where a bound
/// ([`straight_image_bound`]) holds its image on the edge. `None` where the
/// samples do not lie within 1e-6 of a line in (u, v), where the first and
/// last coincide, or where no bound holds.
///
/// On a curved surface (u, v) are not lengths: an angle runs its radius in
/// space for each unit, so a line in (u, v) through samples 1e-6 off it can
/// stand 1e-6 times the radius off the edge between them. No pcurve comes
/// nearer the edge than the edge's samples stand off the surface, `d`; the
/// straight one stands where the bound passes `d` by at most the linear
/// tolerance or, for an edge `d` off the surface (whose pcurve is its
/// shadow), by at most `d`. On a B-spline surface there is no bound on the
/// surface's bend here, so samples within 1e-6 of a line still give it.
fn straight_pcurve(
    uv_pts: &[Point2],
    pts_3d: &[Point3],
    curve: &EdgeCurve,
    ends: (Point3, Point3),
    surface: &FaceSurface,
    frame: Option<&PlaneFrame>,
) -> Option<Curve2D> {
    if uv_pts.len() < 2 || !is_collinear_2d(uv_pts, 1e-6) {
        return None;
    }
    let (first, last) = (uv_pts[0], uv_pts[uv_pts.len() - 1]);
    let run = last - first;
    // Line2D runs by length from its origin, not over [0, 1], so a closed
    // curve whose samples lie on a line keeps a fit that runs round it.
    if run.length_squared() < 1e-12 {
        return None;
    }
    let line = Curve2D::Line(make_line2d_safe(first, run));
    if matches!(surface, FaceSurface::Nurbs(_)) {
        return Some(unproved(line, "straight on a B-spline surface"));
    }
    // On a rod, cone, ball or ring the bound runs with the surface's own
    // origin at nothing, as in [`proved_hermite_pcurve`].
    let reserve = sample_rounding(pts_3d);
    let local = moved_to_surface_origin(surface, curve, pts_3d);
    let (surface, curve, pts_3d, ends, reserve) = match &local {
        Some((surface, curve, points, offset)) => (
            surface,
            curve,
            &points[..],
            (ends.0 + *offset, ends.1 + *offset),
            reserve,
        ),
        None => (surface, curve, pts_3d, ends, 0.0),
    };
    let lift = |q: Point2| match frame {
        Some(f) => Some(f.evaluate(q.x(), q.y())),
        None => surface.evaluate(q.x(), q.y()),
    };
    let mut off_surface = 0.0_f64;
    let mut size = 1.0_f64;
    for (&q, &p) in uv_pts.iter().zip(pts_3d) {
        off_surface = max_or_nan(off_surface, (lift(q)? - p).length());
        size = [p.x(), p.y(), p.z()]
            .into_iter()
            .fold(size, |m, c| max_or_nan(m, c.abs()));
    }
    let bound = straight_image_bound(uv_pts, pts_3d, curve, ends, surface, lift)?;
    let rounding = (64.0 * f64::EPSILON).madd(size, reserve);
    let tol = brepkit_math::tolerance::Tolerance::new().linear;
    let excess = bound + rounding - off_surface;
    (excess.is_finite() && excess <= max_or_nan(tol, off_surface)).then_some(line)
}

/// A bound on how far the image on `surface` of the straight run in (u, v)
/// from the first UV sample to the last strays from the edge, both ways:
/// each point of either lies within the bound of the other. `lift` maps
/// (u, v) onto the surface. `pts_3d` are the edge's samples, evenly spaced
/// in its own parameter, each projecting to its UV sample. `None` where no
/// bound holds.
///
/// The least of three bounds. In each, `f(t)` is the image at the fraction
/// `t` of the run and `c(t)` the edge at the same fraction of its parameter.
///
/// Between two samples, `f` stands off its chord by at most an eighth of
/// the square of the step times a bound on `|f''|`, and the edge strays off
/// its own chord by its sagitta or its Bezier hull ([`piece_stray`]). So
/// within each step the two lie within the larger of their gaps at the step's
/// ends plus those two.
///
/// That bound grows with the bend of each, and so cannot hold a circle that
/// a run along a parallel or a meridian traces exactly. On a surface turned
/// about an axis, the run with `v` held at its start traces a parallel at an
/// even pace in `u`, and with `u` held, on a ball or a ring, a meridian at an
/// even pace in `v`. Holding it moves the image by at most the run in the
/// held parameter times the surface's speed along it. Then
/// [`arcs_apart`] bounds the gap between that arc and a circular edge.
fn straight_image_bound(
    uv_pts: &[Point2],
    pts_3d: &[Point3],
    curve: &EdgeCurve,
    (start, end): (Point3, Point3),
    surface: &FaceSurface,
    lift: impl Fn(Point2) -> Option<Point3>,
) -> Option<f64> {
    let raw = curve;
    let (curve, gap) = unit_conic(curve)?;
    let curve = &curve;
    let n = uv_pts.len() - 1;
    if pts_3d.len() != n + 1 || n == 0 {
        return None;
    }
    let (first, last) = (uv_pts[0], uv_pts[n]);
    let run = last - first;
    #[allow(clippy::cast_precision_loss)]
    let step = 1.0 / n as f64;
    let nurbs_span =
        matches!(curve, EdgeCurve::NurbsCurve(_)).then(|| curve.domain_with_endpoints(start, end));
    let bend = image_bend_bound(surface, first, last)?;
    let mut gaps = Vec::with_capacity(n + 1);
    for (k, &p) in pts_3d.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = k as f64 * step;
        gaps.push((lift(first + run * t)? - p).length());
    }
    let mut pieces = 0.0_f64;
    for k in 0..n {
        let span = nurbs_span.map(|(t0, t1)| {
            #[allow(clippy::cast_precision_loss)]
            let at = |j: usize| (t1 - t0).madd(j as f64 * step, t0);
            (at(k), at(k + 1))
        });
        let stray = piece_stray(raw, pts_3d[k], pts_3d[k + 1], span)?;
        let piece = max_or_nan(gaps[k], gaps[k + 1]) + (step * step / 8.0).madd(bend, stray);
        pieces = max_or_nan(pieces, piece);
    }

    let EdgeCurve::Circle(circle) = curve else {
        return Some(pieces);
    };
    let (theta0, sweep) = conic_sweep(curve, pts_3d)?;
    let edge = |t: f64| circle.evaluate(sweep.madd(t, theta0));
    let (du, dv) = (run.x(), run.y());
    let (u0, v0) = (first.x(), first.y());
    let parallel = match surface {
        FaceSurface::Cylinder(c) => Some((c.origin() + c.axis() * v0, 1.0)),
        FaceSurface::Cone(c) => Some((c.apex() + c.axis() * (v0 * c.half_angle().sin()), 1.0)),
        FaceSurface::Sphere(s) => Some((
            s.center() + s.z_axis() * (s.radius() * v0.sin()),
            s.radius(),
        )),
        FaceSurface::Torus(t) => Some((
            t.center() + t.z_axis() * (t.minor_radius() * v0.sin()),
            t.minor_radius(),
        )),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
    }
    .and_then(|(centre, speed_v)| {
        let arc = |t: f64| surface.evaluate(du.madd(t, u0), v0);
        let apart = arcs_apart(arc, centre, du, edge, circle, sweep)?;
        Some(speed_v.madd(dv.abs(), apart))
    });
    let meridian = match surface {
        FaceSurface::Sphere(s) => Some((s.center(), s.radius())),
        FaceSurface::Torus(t) => {
            let (sin_u, cos_u) = u0.sin_cos();
            Some((
                t.center() + (t.x_axis() * cos_u + t.y_axis() * sin_u) * t.major_radius(),
                t.major_radius() + t.minor_radius(),
            ))
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_) => None,
    }
    .and_then(|(centre, speed_u)| {
        let arc = |t: f64| surface.evaluate(u0, dv.madd(t, v0));
        let apart = arcs_apart(arc, centre, dv, edge, circle, sweep)?;
        Some(speed_u.madd(du.abs(), apart))
    });
    // A NaN bound proves nothing, and `f64::min` passes over it to the
    // least of the others. The two arc bounds compare the unit form's arc,
    // which the stored one lies within the arc gap of.
    Some(
        [parallel, meridian]
            .into_iter()
            .flatten()
            .map(|bound| bound + unit_conic_arc_gap(gap))
            .fold(pieces, f64::min),
    )
}

/// A bound on `|f''|` along the straight run in (u, v) from `a` to `b`,
/// `f(t)` the surface at the fraction `t` of the run:
/// `du^2 |S_uu| + 2 |du dv| |S_uv| + dv^2 |S_vv|`, with each second
/// derivative bounded over the run. `None` on a B-spline surface.
fn image_bend_bound(surface: &FaceSurface, a: Point2, b: Point2) -> Option<f64> {
    let (du, dv) = ((b.x() - a.x()).abs(), (b.y() - a.y()).abs());
    let (uu, uv, vv) = match surface {
        FaceSurface::Plane { .. } => (0.0, 0.0, 0.0),
        FaceSurface::Cylinder(c) => (c.radius(), 0.0, 0.0),
        FaceSurface::Cone(c) => {
            let cos_a = c.half_angle().cos();
            (cos_a * max_or_nan(a.y().abs(), b.y().abs()), cos_a, 0.0)
        }
        FaceSurface::Sphere(s) => (s.radius(), s.radius(), s.radius()),
        FaceSurface::Torus(t) => (
            t.major_radius() + t.minor_radius(),
            t.minor_radius(),
            t.minor_radius(),
        ),
        FaceSurface::Nurbs(_) => return None,
    };
    Some((du * du).madd(uu, (2.0 * du * dv).madd(uv, dv * dv * vv)))
}

/// How far an edge strays from the chord between its points `a` and `b`, at
/// most: nothing for a line; for a circle or an ellipse its longer semi-axis
/// times `1 - cos(phi / 2)`, `phi` its sweep from `a` to `b` (the arc is an
/// affine image of a unit arc, which lies within `1 - cos(phi / 2)` of its
/// chord along the middle radius); for a B-spline over its parameters
/// `span`, the farthest point of the hulls of its Bezier pieces. `None` for
/// a sweep over a quarter turn or a weight not above nothing.
fn piece_stray(curve: &EdgeCurve, a: Point3, b: Point3, span: Option<(f64, f64)>) -> Option<f64> {
    let (curve, gap) = unit_conic(curve)?;
    let curve = &curve;
    // The stored piece lies within the arc gap of the unit form's piece
    // between the points nearest `a` and `b`, which lie within the gap of
    // `a` and `b`, so its chord within the gap of theirs.
    let extra = gap + unit_conic_arc_gap(gap);
    let sagitta = |phi: f64, semi: f64| {
        let phi = shorter_arc_delta(phi).abs();
        (phi <= std::f64::consts::FRAC_PI_2).then(|| 2.0 * semi * (phi / 4.0).sin().powi(2))
    };
    let stray = match curve {
        EdgeCurve::Line => Some(0.0),
        EdgeCurve::Circle(c) => sagitta(c.project(b) - c.project(a), c.radius()),
        EdgeCurve::Ellipse(e) => sagitta(e.project(b) - e.project(a), e.semi_major()),
        EdgeCurve::NurbsCurve(nurbs) => {
            let (t0, t1) = span?;
            if !(t0.is_finite() && t1.is_finite()) {
                return None;
            }
            let (lo, hi) = (t0.min(t1), t0.max(t1));
            let mut breaks = vec![lo];
            breaks.extend(nurbs.knots().iter().copied().filter(|&k| k > lo && k < hi));
            breaks.push(hi);
            let mut most = 0.0_f64;
            for w in breaks.windows(2) {
                if w[1] <= w[0] {
                    continue;
                }
                let hull =
                    brepkit_math::nurbs::decompose::curve_bezier_between(nurbs, w[0], w[1]).ok()?;
                for [x, y, z, weight] in hull {
                    if weight <= 0.0 || !weight.is_finite() {
                        return None;
                    }
                    let p = Point3::new(x / weight, y / weight, z / weight);
                    most = max_or_nan(most, segment_distance_3d(p, a, b));
                }
            }
            Some(most)
        }
    };
    stray.map(|stray| stray + extra)
}

/// The larger of `a` and `b`, or NaN where either is NaN. `f64::max` drops
/// a NaN, so a bound built with it could pass a term it never read.
pub(super) fn max_or_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The distance from `p` to the segment from `a` to `b`.
fn segment_distance_3d(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    let t = if len_sq > 0.0 {
        ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
}

/// The circle or ellipse `curve` with the sizes its bounds read, on unit
/// axes square to each other with its normal their cross product, and a
/// bound on how far its point at each parameter lies from the stored
/// conic's. The first axis keeps its direction and the second is squared to
/// it; a circle's radius is the stored one times the first axis's length,
/// an ellipse's semi-axes the stored ones times their axes' lengths. A
/// circle's axes may have any length: its parameter, read from the angle of
/// a point's shadow on its axes, does not change with their common length.
/// An ellipse's axes must be unit within 64 units of rounding, as its
/// projection reads its semi-axes against unit axes. `None` where a length
/// or a size is not finite or not positive, or an ellipse's major semi-axis
/// comes out shorter than its minor. A line or a B-spline comes back as it
/// is, with no gap.
///
/// At one parameter the stored point is `c + cos t A + sin t B` and the
/// unit form's `c + cos t A' + sin t B'`, so the two lie within
/// `|A - A'| + |B - B'|` of each other: that covers a difference in the
/// axes' lengths and their skew. Each difference is read as it is computed,
/// plus its rounding, and is nothing where the two are the same to the bit.
pub(super) fn unit_conic(curve: &EdgeCurve) -> Option<(EdgeCurve, f64)> {
    let slack = 64.0 * f64::EPSILON;
    // The unit first axis, the second squared to it, and both lengths.
    let frame = |u: Vec3, v: Vec3| {
        let (lu, lv) = (u.length(), v.length());
        if !(lu.is_finite() && lv.is_finite() && lu > 0.0 && lv > 0.0) {
            return None;
        }
        let first = u * (1.0 / lu);
        let rest = v - first * first.dot(v);
        let second = rest.normalize().ok()?;
        Some((first, second, lu, lv))
    };
    // `|a x - b y|` for sizes `a`, `b` and axes `x`, `y`, with its rounding.
    let apart = |a: f64, x: Vec3, b: f64, y: Vec3| {
        let bits = |v: Vec3| [v.x().to_bits(), v.y().to_bits(), v.z().to_bits()];
        if a.to_bits() == b.to_bits() && bits(x) == bits(y) {
            return 0.0;
        }
        let gap = (x * a - y * b).length();
        (2.0 * f64::EPSILON).madd(a * x.length() + b * y.length(), gap) * (1.0 + f64::EPSILON)
    };
    let gap_ok = |gap: f64| (gap.is_finite() && gap >= 0.0).then_some(gap);
    match curve {
        EdgeCurve::Circle(c) => {
            let (u, w, lu, _) = frame(c.u_axis(), c.v_axis())?;
            let radius = c.radius() * lu;
            if !(radius.is_finite() && radius > 0.0) {
                return None;
            }
            let gap =
                apart(c.radius(), c.u_axis(), radius, u) + apart(c.radius(), c.v_axis(), radius, w);
            let unit =
                brepkit_math::curves::Circle3D::with_axes(c.center(), u.cross(w), radius, u, w)
                    .ok()?;
            Some((EdgeCurve::Circle(unit), gap_ok(gap)?))
        }
        EdgeCurve::Ellipse(e) => {
            let (u, w, lu, lv) = frame(e.u_axis(), e.v_axis())?;
            let near_one = |l: f64| (l - 1.0).abs() <= slack;
            let (major, minor) = (e.semi_major() * lu, e.semi_minor() * lv);
            if !near_one(lu) || !near_one(lv) || major < minor || minor.is_nan() || minor <= 0.0 {
                return None;
            }
            let gap = apart(e.semi_major(), e.u_axis(), major, u)
                + apart(e.semi_minor(), e.v_axis(), minor, w);
            let unit = brepkit_math::curves::Ellipse3D::with_axes(
                e.center(),
                u.cross(w),
                major,
                minor,
                u,
                w,
            )
            .ok()?;
            Some((EdgeCurve::Ellipse(unit), gap_ok(gap)?))
        }
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => Some((curve.clone(), 0.0)),
    }
}

/// How far an arc of a stored conic and the arc of its [`unit_conic`]
/// form between the nearest points to its ends can stand apart, `gap` the
/// forms' gap at one parameter: the gap, plus the shift of each end, at most
/// twice the gap as each end's nearest point lies no farther than its point
/// at the same parameter.
pub(super) const fn unit_conic_arc_gap(gap: f64) -> f64 {
    3.0 * gap
}

/// The parameter of a circular or elliptic edge at its first sample and
/// its signed sweep over the samples `pts_3d`, which lie evenly along it
/// under a quarter turn apart. `None` for another curve, or when a step
/// reads a quarter turn or more.
fn conic_sweep(curve: &EdgeCurve, pts_3d: &[Point3]) -> Option<(f64, f64)> {
    let curve = &unit_conic(curve)?.0;
    let project = |p: Point3| match curve {
        EdgeCurve::Circle(c) => Some(c.project(p)),
        EdgeCurve::Ellipse(e) => Some(e.project(p)),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    };
    let theta0 = project(*pts_3d.first()?)?;
    if theta0.is_nan() {
        return None;
    }
    let mut sweep = 0.0;
    for w in pts_3d.windows(2) {
        let step = shorter_arc_delta(project(w[1])? - project(w[0])?);
        if step.is_nan() || step.abs() >= std::f64::consts::FRAC_PI_2 {
            return None;
        }
        sweep += step;
    }
    Some((theta0, sweep))
}

/// A bound on `|p(t) - q(t)|` over `t` in [0, 1], `p` an arc round `centre`
/// at an even angular speed `omega` (`p'' = -omega^2 (p - centre)`), `q`
/// the arc of `circle` from its parameter `theta0` on by `sweep`, with
/// `q(t) = circle(theta0 + sweep t)`. `None` where `p` does not evaluate.
///
/// With `d = p - q`, `d'' = -omega^2 d - omega^2 (c - centre)
/// + (sweep^2 - omega^2) (q - c)`, `c` the circle's centre and `r` its
/// radius, so `|d''| <= omega^2 |d| + k` with
/// `k = omega^2 |c - centre| + |sweep^2 - omega^2| r`. Over a piece of
/// length `h`, `|d|` stands within `h^2 / 8` of `|d''|` above the larger of
/// its ends, so it is at most that larger end plus `h^2 k / 8`, over
/// `1 - h^2 omega^2 / 8`. The pieces turn at most a quarter turn each.
fn arcs_apart(
    p: impl Fn(f64) -> Option<Point3>,
    centre: Point3,
    omega: f64,
    q: impl Fn(f64) -> Point3,
    circle: &brepkit_math::curves::Circle3D,
    sweep: f64,
) -> Option<f64> {
    let turn = max_or_nan(omega.abs(), sweep.abs()) / std::f64::consts::FRAC_PI_2;
    if !turn.is_finite() || turn > 64.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pieces = turn.ceil().max(1.0) as usize;
    #[allow(clippy::cast_precision_loss)]
    let h = 1.0 / pieces as f64;
    let k = (omega * omega).madd(
        (circle.center() - centre).length(),
        (sweep * sweep - omega * omega).abs() * circle.radius(),
    );
    let shrink = 1.0 - h * h * omega * omega / 8.0;
    let mut before = (p(0.0)? - q(0.0)).length();
    let mut most = 0.0_f64;
    for j in 1..=pieces {
        #[allow(clippy::cast_precision_loss)]
        let t = if j == pieces { 1.0 } else { j as f64 * h };
        let gap = (p(t)? - q(t)).length();
        most = max_or_nan(
            most,
            (h * h / 8.0).madd(k, max_or_nan(before, gap)) / shrink,
        );
        before = gap;
    }
    Some(most)
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
    /// 2e-6 of `max(1, r)` in `v`, `r` the point's distance from the axis;
    /// its `u` never steps back against the way the ellipse's `u` runs; and
    /// it leaves each end that way. Returns the largest deviation in `u` and
    /// in relative `v`.
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
        let radius = |p: Point3| match surface {
            FaceSurface::Cone(c) => {
                let w = p - c.apex();
                (w - c.axis() * c.axis().dot(w)).length()
            }
            _ => 0.0,
        };
        let first = pcurve.evaluate(0.0);
        let last = pcurve.evaluate(1.0);
        let copy = ((first.x() - u_start) / TAU).round() * TAU;
        let mut worst = (0.0_f64, 0.0_f64);
        let mut prev: Option<f64> = None;
        for k in 0..=N {
            let s = f64::from(k) / f64::from(N);
            let t = (tb - ta).madd(s, ta);
            let (_, u) = lifted_u(surface, ellipse, ta, t);
            let at = ellipse.evaluate(t);
            let (_, v) = surface.project_point(at).unwrap();
            let p = pcurve.evaluate(s);
            let (du, dv) = (
                (p.x() - copy - u).abs(),
                (p.y() - v).abs() / radius(at).max(1.0),
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

    /// A rigid motion of a scene: upright, turned about a slanted axis, or
    /// mirrored through a slanted plane.
    struct Pose {
        rows: [Vec3; 3],
        shift: Vec3,
    }

    impl Pose {
        fn new(k: usize, scale: f64) -> Self {
            let (x, y, z) = (
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            );
            match k {
                0 => Self {
                    rows: [x, y, z],
                    shift: Vec3::new(0.0, 0.0, 0.0),
                },
                1 => {
                    let a = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
                    let (s, c) = 0.7f64.sin_cos();
                    let turn = |e: Vec3| e * c + a.cross(e) * s + a * (a.dot(e) * (1.0 - c));
                    let (cx, cy, cz) = (turn(x), turn(y), turn(z));
                    Self {
                        rows: [
                            Vec3::new(cx.x(), cy.x(), cz.x()),
                            Vec3::new(cx.y(), cy.y(), cz.y()),
                            Vec3::new(cx.z(), cy.z(), cz.z()),
                        ],
                        shift: Vec3::new(0.3, -0.2, 0.1) * scale,
                    }
                }
                _ => {
                    let n = Vec3::new(1.0, -2.0, 0.5).normalize().unwrap();
                    let flip = |e: Vec3| e - n * (2.0 * n.dot(e));
                    let (cx, cy, cz) = (flip(x), flip(y), flip(z));
                    Self {
                        rows: [
                            Vec3::new(cx.x(), cy.x(), cz.x()),
                            Vec3::new(cx.y(), cy.y(), cz.y()),
                            Vec3::new(cx.z(), cy.z(), cz.z()),
                        ],
                        shift: Vec3::new(-0.2, 0.4, 0.3) * scale,
                    }
                }
            }
        }

        fn vec(&self, v: Vec3) -> Vec3 {
            Vec3::new(
                self.rows[0].dot(v),
                self.rows[1].dot(v),
                self.rows[2].dot(v),
            )
        }

        fn point(&self, p: Point3) -> Point3 {
            Point3::new(0.0, 0.0, 0.0) + self.vec(p - Point3::new(0.0, 0.0, 0.0)) + self.shift
        }
    }

    /// A short arc of a curve on a ball, a rod or a ring of size `size`, in
    /// pose `pose`: a circle tilted to the ball's axis, a steep plane section
    /// of the rod, and a circle that runs round the ring and through its hole
    /// (a section by a plane that touches the ring twice). Each arc is a
    /// quarter of a thousandth of a turn of its curve.
    fn short_arcs(pose: &Pose, size: f64) -> Vec<(&'static str, FaceSurface, EdgeCurve, f64, f64)> {
        arcs_of_span(pose, size, 2.5e-3)
    }

    /// [`short_arcs`] with each arc `span` of its curve's parameter long.
    fn arcs_of_span(
        pose: &Pose,
        size: f64,
        span: f64,
    ) -> Vec<(&'static str, FaceSurface, EdgeCurve, f64, f64)> {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface, ToroidalSurface};
        let origin = Point3::new(0.0, 0.0, 0.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let ball = FaceSurface::Sphere(
            SphericalSurface::with_axis_and_ref_dir(
                pose.point(origin),
                size,
                pose.vec(z),
                pose.vec(x),
            )
            .unwrap(),
        );
        let tilt = Vec3::new(0.3, -0.4, 0.866).normalize().unwrap();
        let ball_circle = Circle3D::new(
            pose.point(origin + tilt * (0.25 * size)),
            pose.vec(tilt),
            size * (1.0f64 - 0.0625).sqrt(),
        )
        .unwrap();
        let rod = FaceSurface::Cylinder(
            CylindricalSurface::new(pose.point(origin), pose.vec(z), size).unwrap(),
        );
        let steep = 1.4f64;
        let rod_ellipse = Ellipse3D::new_with_ref(
            pose.point(origin),
            pose.vec(Vec3::new(-steep.sin(), 0.0, steep.cos())),
            size / steep.cos(),
            size,
            pose.vec(Vec3::new(steep.cos(), 0.0, steep.sin())),
        )
        .unwrap();
        let tube = 0.4 * size;
        let ring = FaceSurface::Torus(
            ToroidalSurface::with_axis_and_ref_dir(
                pose.point(origin),
                size,
                tube,
                pose.vec(z),
                pose.vec(x),
            )
            .unwrap(),
        );
        let (sb, cb) = (tube / size, (1.0 - (tube / size).powi(2)).sqrt());
        let ring_circle = Circle3D::new_with_ref(
            pose.point(origin + y * tube),
            pose.vec(Vec3::new(-sb, 0.0, cb)),
            size,
            pose.vec(Vec3::new(cb, 0.0, sb)),
        )
        .unwrap();
        vec![
            ("ball", ball, EdgeCurve::Circle(ball_circle), 1.0, span),
            ("rod", rod, EdgeCurve::Ellipse(rod_ellipse), 1.0, span),
            ("ring", ring, EdgeCurve::Circle(ring_circle), 0.7, span),
        ]
    }

    /// How far a pcurve's image strays from its edge in 3D, read at 2001
    /// points along it, and the pcurve's kind. `start` and `end` are the
    /// edge's ends, the way the pcurve runs.
    fn pcurve_stray(
        pcurve: &Curve2D,
        surface: &FaceSurface,
        curve: &EdgeCurve,
        (start, end): (Point3, Point3),
    ) -> (f64, &'static str) {
        let (lo, hi, kind) = match pcurve {
            Curve2D::Line(line) => {
                let (u, v) = surface.project_point(end).unwrap();
                let o = line.evaluate(0.0);
                let near = |a: f64, b: f64| b + ((a - b) / TAU).round() * TAU;
                let (u, v) = match surface {
                    FaceSurface::Torus(_) => (near(o.x(), u), near(o.y(), v)),
                    _ => (near(o.x(), u), v),
                };
                (0.0, line.project(Point2::new(u, v)), "line")
            }
            Curve2D::Nurbs(n) => {
                let (a, b) = n.domain();
                (a, b, "nurbs")
            }
            Curve2D::Circle(_) | Curve2D::Ellipse(_) => (0.0, 1.0, "conic"),
        };
        let off = |p: Point3| match curve {
            EdgeCurve::Circle(c) => {
                let w = p - c.center();
                let h = w.dot(c.normal());
                let flat = (w - c.normal() * h).length() - c.radius();
                h.hypot(flat)
            }
            EdgeCurve::Ellipse(e) => (p - e.evaluate(e.project(p))).length(),
            EdgeCurve::Line => segment_distance_3d(p, start, end),
            EdgeCurve::NurbsCurve(_) => f64::NAN,
        };
        let worst = (0..=2000)
            .map(|k| {
                let uv = pcurve.evaluate((hi - lo).madd(f64::from(k) / 2000.0, lo));
                off(surface.evaluate(uv.x(), uv.y()).unwrap())
            })
            .fold(0.0, max_or_nan);
        (worst, kind)
    }

    /// The pcurves of the arc of `curve` from `a` to `b` on `surface`, as a
    /// section edge and as a face's boundary edge both ways, each with how
    /// far its image strays from the arc in 3D ([`pcurve_stray`]) and its
    /// kind.
    fn arc_pcurves(
        surface: &FaceSurface,
        curve: &EdgeCurve,
        a: Point3,
        b: Point3,
    ) -> Vec<(&'static str, f64, &'static str)> {
        let section = compute_pcurve_on_surface(curve, a, b, surface, &[], None);
        let (off, kind) = pcurve_stray(&section, surface, curve, (a, b));
        let mut out = vec![("section", off, kind)];
        if matches!(curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
            for (forward, way) in [(true, "boundary forward"), (false, "boundary back")] {
                let rim =
                    compute_boundary_pcurve_on_surface(curve, a, b, forward, surface, &[], None);
                let ends = if forward { (a, b) } else { (b, a) };
                let (off, kind) = pcurve_stray(&rim, surface, curve, ends);
                out.push((way, off, kind));
            }
        }
        out
    }

    /// A short arc on a ball, a rod and a ring of size 1e4 and 1e-3, in three
    /// poses, as a section edge and as a face's boundary edge both ways: its
    /// pcurve's image lies within the linear tolerance of the arc in 3D. Its
    /// samples lie within 1e-6 of a line in (u, v). On a size of 1e4 that
    /// line strays 7e-4 to 5e-3 off the arc, so the pcurve is the proved
    /// one; on a size of 1e-3 it strays under 1e-9, and the pcurve stays
    /// that line.
    #[test]
    fn a_short_arc_on_a_large_ball_rod_or_ring_gets_a_pcurve_on_its_edge() {
        for size in [1e4, 1e-3] {
            for k in 0..3 {
                let pose = Pose::new(k, size);
                for (what, surface, curve, at, span) in short_arcs(&pose, size) {
                    let point = |t: f64| match &curve {
                        EdgeCurve::Circle(c) => c.evaluate(t),
                        EdgeCurve::Ellipse(e) => e.evaluate(t),
                        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                    };
                    for t in [at, at + span * 0.5, at + span] {
                        let (u, v) = surface.project_point(point(t)).unwrap();
                        let back = surface.evaluate(u, v).unwrap();
                        assert!(
                            (back - point(t)).length() < 1e-9 * size,
                            "{what} off its surface"
                        );
                    }
                    let (samples, uv) =
                        sample_edge_on_surface(&curve, point(at), point(at + span), &surface);
                    assert!(
                        is_collinear_2d(&uv, 1e-6),
                        "{what} {size:e}: samples off a line"
                    );
                    if size > 1.0 {
                        assert!(
                            proved_conic_pcurve(&curve, &samples, &surface, None).is_some(),
                            "{what} pose {k}: no proved pcurve"
                        );
                    }
                    for (way, off, kind) in
                        arc_pcurves(&surface, &curve, point(at), point(at + span))
                    {
                        let name = format!("{what} size {size:e} pose {k} {way}");
                        assert!(off <= 1e-7, "{name}: the {kind} pcurve strays {off:e}");
                        if size < 1.0 {
                            assert_eq!(kind, "line", "{name}");
                        }
                    }
                }
            }
        }
    }

    /// A quarter turn of each curve of [`short_arcs`] on a ball, a rod and a
    /// ring of size 1e4, in three poses, and a quarter of a circle of radius
    /// 1e4 on a plane: each pcurve is the proved one, and its image lies
    /// within the linear tolerance of the arc. A cubic through the sixteen
    /// samples strays 1.3e-3 to 5.1e-2 off these arcs.
    #[test]
    fn a_quarter_turn_on_a_large_surface_gets_a_proved_pcurve() {
        use std::f64::consts::FRAC_PI_2;
        for k in 0..3 {
            let pose = Pose::new(k, 1e4);
            for (what, surface, curve, at, span) in arcs_of_span(&pose, 1e4, FRAC_PI_2) {
                let point = |t: f64| match &curve {
                    EdgeCurve::Circle(c) => c.evaluate(t),
                    EdgeCurve::Ellipse(e) => e.evaluate(t),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                };
                let (a, b) = (point(at), point(at + span));
                let (samples, _) = sample_edge_on_surface(&curve, a, b, &surface);
                let proved = proved_conic_pcurve(&curve, &samples, &surface, None);
                let pieces = match &proved {
                    Some(Curve2D::Nurbs(n)) => n.control_points().len() / 3,
                    _ => 0,
                };
                assert!(pieces > 0, "{what} pose {k}: no proved pcurve: {proved:?}");
                for (way, off, kind) in arc_pcurves(&surface, &curve, a, b) {
                    assert!(
                        off <= 1e-7,
                        "{what} pose {k} {way}: the {kind} pcurve strays {off:e} ({pieces} pieces)"
                    );
                }
            }
        }
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let centre = Point3::new(0.0, 0.0, 0.0);
        let circle =
            brepkit_math::curves::Circle3D::new(centre, Vec3::new(0.0, 0.0, 1.0), 1e4).unwrap();
        let frame = PlaneFrame::from_normal_and_point(Vec3::new(0.0, 0.0, 1.0), centre);
        let (a, b) = (circle.evaluate(0.3), circle.evaluate(0.3 + FRAC_PI_2));
        let pcurve =
            compute_pcurve_on_surface(&EdgeCurve::Circle(circle), a, b, &plane, &[], Some(&frame));
        assert!(matches!(pcurve, Curve2D::Nurbs(_)), "{pcurve:?}");
        let (lo, hi) = match &pcurve {
            Curve2D::Nurbs(n) => n.domain(),
            _ => (0.0, 0.0),
        };
        let off = (0..=4000)
            .map(|k| {
                let q = pcurve.evaluate((hi - lo).madd(f64::from(k) / 4000.0, lo));
                ((frame.evaluate(q.x(), q.y()) - centre).length() - 1e4).abs()
            })
            .fold(0.0, max_or_nan);
        assert!(off <= 1e-7, "the plane's pcurve strays {off:e}");
    }

    /// How far a B-spline pcurve's image, read at 401 points, stands off the
    /// stored circle `c` over its parameters from `t0` to `t1`: each distance
    /// from a search over 801 of the stored circle's own points and a golden
    /// section about the nearest.
    fn stray_from_stored(
        pcurve: &Curve2D,
        surface: &FaceSurface,
        c: &brepkit_math::curves::Circle3D,
        (t0, t1): (f64, f64),
    ) -> f64 {
        let (lo, hi) = match pcurve {
            Curve2D::Nurbs(n) => n.domain(),
            Curve2D::Line(_) | Curve2D::Circle(_) | Curve2D::Ellipse(_) => return f64::NAN,
        };
        let step = (t1 - t0) / 800.0;
        (0..=400)
            .map(|k| {
                let q = pcurve.evaluate((hi - lo).madd(f64::from(k) / 400.0, lo));
                let p = surface.evaluate(q.x(), q.y()).unwrap();
                let gap = |t: f64| (c.evaluate(t) - p).length();
                let nearest = (0..=800)
                    .map(|j| step.madd(f64::from(j), t0))
                    .min_by(|&x, &y| gap(x).total_cmp(&gap(y)))
                    .unwrap();
                let (mut a, mut b) = (nearest - step, nearest + step);
                for _ in 0..200 {
                    let (m1, m2) = ((b - a).madd(0.381_966, a), (a - b).madd(0.381_966, b));
                    if gap(m1) < gap(m2) {
                        b = m2;
                    } else {
                        a = m1;
                    }
                }
                gap(f64::midpoint(a, b)).min(gap(nearest))
            })
            .fold(0.0, max_or_nan)
    }

    /// The quarter turns of [`a_quarter_turn_on_a_large_surface_gets_a_proved_pcurve`]
    /// with each circle stored on axes 1e12 long and its radius divided by
    /// as much: the pcurve is proved, as for the unit form, and lies within
    /// the tolerance of the stored circle, read from its own points. An ellipse on axes 2 and 3 long, whose projection
    /// misreads its points, or on axes slightly askew, gets no proved
    /// pcurve.
    #[test]
    fn a_conic_on_long_axes_gets_the_proof_of_its_unit_form() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use std::f64::consts::FRAC_PI_2;
        for k in 0..3 {
            let pose = Pose::new(k, 1e4);
            for (what, surface, curve, at, span) in arcs_of_span(&pose, 1e4, FRAC_PI_2) {
                let long = match &curve {
                    EdgeCurve::Circle(c) => EdgeCurve::Circle(
                        Circle3D::with_axes(
                            c.center(),
                            c.normal(),
                            c.radius() * 1e-12,
                            c.u_axis() * 1e12,
                            c.v_axis() * 1e12,
                        )
                        .unwrap(),
                    ),
                    EdgeCurve::Ellipse(e) => EdgeCurve::Ellipse(
                        Ellipse3D::with_axes(
                            e.center(),
                            e.normal(),
                            e.semi_major() / 2.0,
                            e.semi_minor() / 3.0,
                            e.u_axis() * 2.0,
                            e.v_axis() * 3.0,
                        )
                        .unwrap(),
                    ),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                };
                let point = |t: f64| match &curve {
                    EdgeCurve::Circle(c) => c.evaluate(t),
                    EdgeCurve::Ellipse(e) => e.evaluate(t),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                };
                let (a, b) = (point(at), point(at + span));
                let (samples, _) = sample_edge_on_surface(&long, a, b, &surface);
                let proved = proved_conic_pcurve(&long, &samples, &surface, None);
                if let EdgeCurve::Ellipse(e) = &curve {
                    assert!(proved.is_none(), "{what} pose {k}: proved on scaled axes");
                    let askew = Ellipse3D::with_axes(
                        e.center(),
                        e.normal(),
                        e.semi_major(),
                        e.semi_minor(),
                        e.u_axis(),
                        (e.v_axis() + e.u_axis() * 1e-6).normalize().unwrap(),
                    )
                    .unwrap();
                    let askew = EdgeCurve::Ellipse(askew);
                    let (samples, _) = sample_edge_on_surface(&askew, a, b, &surface);
                    assert!(proved_conic_pcurve(&askew, &samples, &surface, None).is_none());
                    continue;
                }
                assert!(
                    proved.is_some(),
                    "{what} pose {k}: no proved pcurve on long axes"
                );
                let EdgeCurve::Circle(stored) = &long else {
                    continue;
                };
                let mut pcurves = vec![(
                    "section",
                    compute_pcurve_on_surface(&long, a, b, &surface, &[], None),
                    (a, b),
                )];
                for (forward, way) in [(true, "forward"), (false, "back")] {
                    let ends = if forward { (a, b) } else { (b, a) };
                    pcurves.push((
                        way,
                        compute_boundary_pcurve_on_surface(
                            &long,
                            a,
                            b,
                            forward,
                            &surface,
                            &[],
                            None,
                        ),
                        ends,
                    ));
                }
                for (way, pcurve, _) in pcurves {
                    assert!(matches!(pcurve, Curve2D::Nurbs(_)), "{what} pose {k} {way}");
                    let off = stray_from_stored(&pcurve, &surface, stored, (at, at + span));
                    assert!(
                        off <= 1e-7,
                        "{what} pose {k} {way}: the pcurve on long axes strays {off:e}"
                    );
                }
            }
        }
    }

    /// A piece of a circle of radius 1e4 stored as radius 1e-8 on axes 1e12
    /// long: the bound on how far it strays from its chord covers its true
    /// sagitta, as for its unit form.
    #[test]
    fn a_circle_on_long_axes_strays_from_its_chord_by_its_true_sagitta() {
        use brepkit_math::curves::Circle3D;
        let unit =
            Circle3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 1e4).unwrap();
        let long = Circle3D::with_axes(
            unit.center(),
            unit.normal(),
            1e-8,
            unit.u_axis() * 1e12,
            unit.v_axis() * 1e12,
        )
        .unwrap();
        let (a, b) = (unit.evaluate(0.3), unit.evaluate(0.4));
        assert!((long.evaluate(0.3) - a).length() < 1e-9);
        let middle = unit.evaluate(0.35);
        let sagitta = (middle - (a + (b - a) * 0.5)).length();
        for curve in [EdgeCurve::Circle(unit), EdgeCurve::Circle(long)] {
            let stray = piece_stray(&curve, a, b, None).unwrap();
            assert!(
                stray >= sagitta,
                "stray {stray:e} under the sagitta {sagitta:e}"
            );
        }
    }

    /// The quarter turns of [`a_quarter_turn_on_a_large_surface_gets_a_proved_pcurve`]
    /// on the ball, the rod and the ring of size 1e4, each scene moved
    /// `offset` along x, in three poses: moved 1e6 or 1e8, the pcurve is
    /// proved and lies within the tolerance of the stored circle; moved
    /// 1e10, where one rounding of a sample's coordinates passes the
    /// tolerance, the proof refuses.
    #[test]
    fn a_quarter_turn_far_from_the_origin_is_proved_or_refused() {
        use std::f64::consts::FRAC_PI_2;
        for offset in [1e6, 1e8, 1e10] {
            for k in 0..3 {
                let mut pose = Pose::new(k, 1e4);
                pose.shift += Vec3::new(offset, 0.0, 0.0);
                for (what, surface, curve, at, span) in arcs_of_span(&pose, 1e4, FRAC_PI_2) {
                    let EdgeCurve::Circle(circle) = &curve else {
                        continue;
                    };
                    let (a, b) = (circle.evaluate(at), circle.evaluate(at + span));
                    let (samples, _) = sample_edge_on_surface(&curve, a, b, &surface);
                    let proved = proved_conic_pcurve(&curve, &samples, &surface, None);
                    let name = format!("{what} pose {k} moved {offset:e}");
                    if offset > 1e9 {
                        assert!(proved.is_none(), "{name}: proved past one rounding");
                        continue;
                    }
                    let pcurve = proved.unwrap_or_else(|| {
                        unreachable!("{name}: no proved pcurve");
                    });
                    let off = stray_from_stored(&pcurve, &surface, circle, (at, at + span));
                    assert!(off <= 1e-7, "{name}: the pcurve strays {off:e}");
                }
            }
        }
    }

    /// How far the image of a plane pcurve strays from `edge` over its
    /// parameters from `t0` to `t1`, at its most, read from the frame's
    /// origin: 201 points of the pcurve, each against an 801-point search
    /// and a golden section.
    fn plane_stray(
        pcurve: &Curve2D,
        frame: &PlaneFrame,
        edge: impl Fn(f64) -> Vec3,
        (t0, t1): (f64, f64),
    ) -> f64 {
        let Curve2D::Nurbs(n) = pcurve else {
            return f64::NAN;
        };
        let (lo, hi) = n.domain();
        let step = (t1 - t0) / 800.0;
        (0..=200)
            .map(|k| {
                let q = pcurve.evaluate((hi - lo).madd(f64::from(k) / 200.0, lo));
                let p = frame.u_axis() * q.x() + frame.v_axis() * q.y();
                let gap = |t: f64| (edge(t) - p).length();
                let nearest = (0..=800)
                    .map(|j| step.madd(f64::from(j), t0))
                    .min_by(|&x, &y| gap(x).total_cmp(&gap(y)))
                    .unwrap();
                let (mut a, mut b) = (nearest - step, nearest + step);
                for _ in 0..200 {
                    let (m1, m2) = ((b - a).madd(0.381_966, a), (a - b).madd(0.381_966, b));
                    if gap(m1) < gap(m2) {
                        b = m2;
                    } else {
                        a = m1;
                    }
                }
                gap(f64::midpoint(a, b)).min(gap(nearest))
            })
            .fold(0.0, max_or_nan)
    }

    /// A plane through `centre` with normal `n`, and on it a circle of
    /// radius `r` and an ellipse of semi-axes `1.7 r` and `r`, each with its
    /// axis vectors.
    fn plane_conics(
        n: Vec3,
        centre: Point3,
        r: f64,
    ) -> (FaceSurface, Vec<(&'static str, EdgeCurve, Vec3, Vec3)>) {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        let seed = Vec3::new(0.6, -0.2, 0.7);
        let e1 = (seed - n * n.dot(seed)).normalize().unwrap();
        let e2 = n.cross(e1);
        let surface = FaceSurface::Plane {
            normal: n,
            d: n.dot(centre - Point3::new(0.0, 0.0, 0.0)),
        };
        let conics = vec![
            (
                "circle",
                EdgeCurve::Circle(Circle3D::with_axes(centre, e1.cross(e2), r, e1, e2).unwrap()),
                e1 * r,
                e2 * r,
            ),
            (
                "ellipse",
                EdgeCurve::Ellipse(
                    Ellipse3D::with_axes(centre, e1.cross(e2), 1.7 * r, r, e1, e2).unwrap(),
                ),
                e1 * (1.7 * r),
                e2 * r,
            ),
        ];
        (surface, conics)
    }

    /// The plane normals of [`a_conic_on_a_plane_far_from_the_origin_is_proved_or_refused`].
    fn plane_normals() -> [Vec3; 3] {
        [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0).normalize().unwrap(),
            Vec3::new(-0.3, 0.9, -0.1).normalize().unwrap(),
        ]
    }

    /// A circle and an ellipse of size 1e4 and 1e-6 on a plane, on three
    /// plane normals, an arc of each from stored ends with the frame from a
    /// third point of the conic. Moved 1e8 from the origin, the pcurve from
    /// the samples and the boundary pcurve from the stored ends are proved,
    /// and each lies within the tolerance of the stored arc between its
    /// ends, read from the frame's origin. Moved 1e10, where one rounding
    /// of a sample's coordinates passes the tolerance, the pcurve from the
    /// samples is refused; the boundary pcurve, read from the stored ends
    /// alone, stays proved within the tolerance of the shadow on the frame's
    /// plane of its own arc, from its start round to its end (the frame's
    /// origin and the ends round 1e-6 off the conic).
    #[test]
    fn a_conic_on_a_plane_far_from_the_origin_is_proved_or_refused() {
        for (k, n) in plane_normals().into_iter().enumerate() {
            for far in [1e8, 1e10] {
                for r in [1e4, 1e-6] {
                    let centre = Point3::new(far, -0.7 * far, 0.4 * far);
                    let (surface, conics) = plane_conics(n, centre, r);
                    for (what, curve, big_a, big_b) in conics {
                        let at = |t: f64| match &curve {
                            EdgeCurve::Circle(c) => c.evaluate(t),
                            EdgeCurve::Ellipse(e) => e.evaluate(t),
                            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                        };
                        let read = |p: Point3| match &curve {
                            EdgeCurve::Circle(c) => c.project(p),
                            EdgeCurve::Ellipse(e) => e.project(p),
                            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                        };
                        let (a, b) = (at(0.4), at(1.7));
                        let wire = [at(3.4), a, b];
                        let frame = PlaneFrame::from_plane_face(n, &wire);
                        let ta = read(a);
                        let span = (ta, ta + shorter_arc_delta(read(b) - ta));
                        let native = curve.domain_with_endpoints(a, b);
                        let o = frame.origin();
                        let edge = |t: f64| (centre - o) + big_a * t.cos() + big_b * t.sin();
                        let normal = frame.u_axis().cross(frame.v_axis());
                        let shadow = |t: f64| {
                            let w = edge(t);
                            w - normal * normal.dot(w)
                        };
                        let name = format!("{what} r {r:e} normal {k} moved {far:e}");
                        let (samples, _) = sample_edge_via_frame(&curve, a, b, &frame);
                        let proved = proved_conic_pcurve(&curve, &samples, &surface, Some(&frame));
                        let boundary = plane_conic_arc(&curve, a, b, true, &frame)
                            .unwrap_or_else(|| unreachable!("{name}: no boundary pcurve"));
                        if far > 1e9 {
                            assert!(proved.is_none(), "{name}: proved past one rounding");
                            let off = plane_stray(&boundary, &frame, shadow, native);
                            assert!(off <= 1e-7, "{name}: the boundary pcurve strays {off:e}");
                            continue;
                        }
                        let proved =
                            proved.unwrap_or_else(|| unreachable!("{name}: no proved pcurve"));
                        for (way, pcurve, span) in
                            [("samples", proved, span), ("boundary", boundary, native)]
                        {
                            let off = plane_stray(&pcurve, &frame, edge, span);
                            assert!(off <= 1e-7, "{name} {way}: the pcurve strays {off:e}");
                        }
                    }
                }
            }
        }
    }

    /// A quarter turn on a ball, a rod and a ring of size 1e4 moved 1e10
    /// from the origin, and an arc of a circle and of an ellipse on a plane
    /// moved as far, where the proof refuses, gets its pcurve from the fit,
    /// and the perf counters count it as unproved.
    #[cfg(feature = "perf-counters")]
    #[test]
    fn a_pcurve_with_no_proof_is_counted() {
        let mut pose = Pose::new(1, 1e4);
        pose.shift += Vec3::new(1e10, 0.0, 0.0);
        for (what, surface, curve, at, span) in
            arcs_of_span(&pose, 1e4, std::f64::consts::FRAC_PI_2)
        {
            let EdgeCurve::Circle(circle) = &curve else {
                continue;
            };
            let (a, b) = (circle.evaluate(at), circle.evaluate(at + span));
            let before = crate::perf::snapshot().unproved_pcurves;
            let _ = compute_pcurve_on_surface(&curve, a, b, &surface, &[], None);
            let after = crate::perf::snapshot().unproved_pcurves;
            assert!(after > before, "{what}: the fit is not counted");
        }
        for n in plane_normals() {
            let (surface, conics) = plane_conics(n, Point3::new(1e10, -7e9, 4e9), 1e4);
            for (what, curve, _, _) in conics {
                let at = |t: f64| match &curve {
                    EdgeCurve::Circle(c) => c.evaluate(t),
                    EdgeCurve::Ellipse(e) => e.evaluate(t),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
                };
                let (a, b) = (at(0.4), at(1.7));
                let wire = [at(3.4), a, b];
                let before = crate::perf::snapshot().unproved_pcurves;
                let _ = compute_pcurve_on_surface(&curve, a, b, &surface, &wire, None);
                let after = crate::perf::snapshot().unproved_pcurves;
                assert!(after > before, "plane {what}: the fit is not counted");
            }
        }
    }

    /// Arcs that a straight run in (u, v) traces exactly, on surfaces of
    /// size 1e4 and 1 and of size 1e4 moved 1e8 from the origin, in three
    /// poses: a parallel and a meridian of a ball and of a ring, a parallel
    /// of a rod and of a cone, a ruling of a rod and of a cone. Each keeps a
    /// straight pcurve, within the linear tolerance of the arc in 3D.
    #[test]
    fn runs_along_parallels_meridians_and_rulings_keep_straight_pcurves() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{
            ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
        };
        let origin = Point3::new(0.0, 0.0, 0.0);
        let (x, z) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        for (size, far) in [(1e4, 0.0), (1.0, 0.0), (1e4, 1e8)] {
            for k in 0..3 {
                let mut pose = Pose::new(k, size);
                pose.shift += Vec3::new(far, 0.0, 0.0);
                let (centre, axis) = (pose.point(origin), pose.vec(z));
                let ball_surface =
                    SphericalSurface::with_axis_and_ref_dir(centre, size, axis, pose.vec(x))
                        .unwrap();
                let tube = 0.3 * size;
                let ring_surface =
                    ToroidalSurface::with_axis_and_ref_dir(centre, size, tube, axis, pose.vec(x))
                        .unwrap();
                // The way `v` grows at `u` turns about this axis.
                let ball_meridian = |u: f64| {
                    let (s, c) = u.sin_cos();
                    (ball_surface.x_axis() * c + ball_surface.y_axis() * s).cross(axis)
                };
                let ring_out = |u: f64| {
                    let (s, c) = u.sin_cos();
                    ring_surface.x_axis() * c + ring_surface.y_axis() * s
                };
                let ball = FaceSurface::Sphere(ball_surface.clone());
                let ring = FaceSurface::Torus(ring_surface.clone());
                let rod =
                    FaceSurface::Cylinder(CylindricalSurface::new(centre, axis, size).unwrap());
                let cone = FaceSurface::Cone(ConicalSurface::new(centre, axis, 0.5).unwrap());
                let along = |s: &FaceSurface, u: f64, v: f64| s.evaluate(u, v).unwrap();
                let lat = 0.4f64;
                let mut cases: Vec<(&str, &FaceSurface, EdgeCurve, Point3, Point3)> = Vec::new();
                let parallel = |height: f64, radius: f64| {
                    EdgeCurve::Circle(Circle3D::new(centre + axis * height, axis, radius).unwrap())
                };
                cases.push((
                    "ball parallel",
                    &ball,
                    parallel(size * lat.sin(), size * lat.cos()),
                    along(&ball, 0.3, lat),
                    along(&ball, 2.1, lat),
                ));
                cases.push((
                    "ball meridian",
                    &ball,
                    EdgeCurve::Circle(Circle3D::new(centre, ball_meridian(0.9), size).unwrap()),
                    along(&ball, 0.9, -1.1),
                    along(&ball, 0.9, 1.2),
                ));
                cases.push((
                    "ring parallel",
                    &ring,
                    parallel(tube * 0.8f64.sin(), 0.8f64.cos().madd(tube, size)),
                    along(&ring, 0.3, 0.8),
                    along(&ring, 2.1, 0.8),
                ));
                let u0 = 0.6f64;
                cases.push((
                    "ring meridian",
                    &ring,
                    EdgeCurve::Circle(
                        Circle3D::new(centre + ring_out(u0) * size, ring_out(u0).cross(axis), tube)
                            .unwrap(),
                    ),
                    along(&ring, u0, 0.4),
                    along(&ring, u0, 2.9),
                ));
                cases.push((
                    "rod parallel",
                    &rod,
                    parallel(0.7 * size, size),
                    along(&rod, 0.3, 0.7 * size),
                    along(&rod, 2.1, 0.7 * size),
                ));
                cases.push((
                    "cone parallel",
                    &cone,
                    parallel(2.0 * size * 0.5f64.sin(), 2.0 * size * 0.5f64.cos()),
                    along(&cone, 0.3, 2.0 * size),
                    along(&cone, 2.1, 2.0 * size),
                ));
                cases.push((
                    "rod ruling",
                    &rod,
                    EdgeCurve::Line,
                    along(&rod, 0.7, -0.3 * size),
                    along(&rod, 0.7, 2.0 * size),
                ));
                cases.push((
                    "cone ruling",
                    &cone,
                    EdgeCurve::Line,
                    along(&cone, 0.7, 0.5 * size),
                    along(&cone, 0.7, 3.0 * size),
                ));
                for (what, surface, curve, a, b) in cases {
                    for (way, off, kind) in arc_pcurves(surface, &curve, a, b) {
                        let name = format!("{what} size {size:e} moved {far:e} pose {k} {way}");
                        assert_eq!(kind, "line", "{name}");
                        assert!(off <= 1e-7, "{name}: strays {off:e}");
                    }
                }
            }
        }
    }

    /// Curved edges on a plane: an arc of a circle of radius 1e4 whose
    /// sagitta, 5e-7, is under the samples' 1e-6 but over the tolerance, and
    /// a B-spline with one inner control point 6e-7 off its chord. Each gets
    /// a fit that lies within the tolerance of it. A B-spline along a line,
    /// and a shorter arc of sagitta 5e-9, each keep a straight pcurve. So
    /// does an arc that stands up off the plane, its shadow a segment.
    #[test]
    fn curved_edges_on_a_plane_keep_a_straight_pcurve_only_within_the_tolerance() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::nurbs::curve::NurbsCurve;
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let frame =
            PlaneFrame::from_normal_and_point(Vec3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, 0.0));
        let stray = |pcurve: &Curve2D, on: &dyn Fn(Point3) -> f64, end: Point3| {
            let hi = match pcurve {
                Curve2D::Line(line) => line.project(frame.project(end)),
                Curve2D::Nurbs(n) => n.domain().1,
                Curve2D::Circle(_) | Curve2D::Ellipse(_) => 1.0,
            };
            (0..=2000)
                .map(|k| {
                    let q = pcurve.evaluate(hi * f64::from(k) / 2000.0);
                    on(frame.evaluate(q.x(), q.y()))
                })
                .fold(0.0, max_or_nan)
        };
        let radius = 1e4;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        let to_circle = |p: Point3| (p - Point3::new(0.0, 0.0, 0.0)).length() - radius;
        for (sweep, straight) in [(2.0e-5, false), (2.0e-6, true)] {
            let (a, b) = (circle.evaluate(0.4), circle.evaluate(0.4 + sweep));
            let curve = EdgeCurve::Circle(circle.clone());
            let pcurve = compute_pcurve_on_surface(&curve, a, b, &plane, &[], Some(&frame));
            assert_eq!(
                matches!(pcurve, Curve2D::Line(_)),
                straight,
                "sweep {sweep}"
            );
            let off = stray(&pcurve, &|p| to_circle(p).abs(), b);
            assert!(off <= 1e-7, "sweep {sweep}: strays {off:e}");
        }
        for (bump, straight) in [(6e-7, false), (0.0, true)] {
            let points = vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, bump, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(3.0, 0.0, 0.0),
            ];
            let nurbs = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                points.clone(),
                vec![1.0; 4],
            )
            .unwrap();
            let (a, b) = (points[0], points[3]);
            let curve = EdgeCurve::NurbsCurve(nurbs.clone());
            let pcurve = compute_pcurve_on_surface(&curve, a, b, &plane, &[], Some(&frame));
            assert_eq!(matches!(pcurve, Curve2D::Line(_)), straight, "bump {bump}");
            let to_curve = |p: Point3| {
                brepkit_math::nurbs::projection::project_point_to_curve(&nurbs, p, 1e-14)
                    .unwrap()
                    .distance
            };
            let off = stray(&pcurve, &to_curve, b);
            assert!(off <= 1e-7, "bump {bump}: strays {off:e}");
        }
        let upright =
            Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 1.0).unwrap();
        let (a, b) = (Point3::new(-0.6, 0.0, 0.2), Point3::new(0.6, 0.0, 0.2));
        let pcurve =
            compute_pcurve_on_surface(&EdgeCurve::Circle(upright), a, b, &plane, &[], Some(&frame));
        assert!(matches!(pcurve, Curve2D::Line(_)), "{pcurve:?}");
    }

    /// The section of a cylinder of radius 1000 by a plane at one radian to
    /// its axis, through the axis at the origin, as a closed rim and as an
    /// arc from its parameter 0.5 to 5.5, on the cylinder with its parameter
    /// origin at the section's centre and moved 1e6 along the axis. The
    /// pcurves stand off the ellipse, in 3D at each parameter, by the same
    /// distance on both, within `2e-6 * max(1, r)` = 2e-3 of it.
    #[test]
    fn a_cylinder_sections_pcurve_keeps_its_accuracy_wherever_the_axis_origin_lies() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::EdgeCurve;
        use std::f64::consts::TAU;

        let (radius, tilt) = (1000.0_f64, 1.0_f64);
        let ellipse = Ellipse3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(-tilt.sin(), 0.0, tilt.cos()),
            radius / tilt.cos(),
            radius,
            Vec3::new(tilt.cos(), 0.0, tilt.sin()),
        )
        .unwrap();
        let curve = EdgeCurve::Ellipse(ellipse.clone());
        let worst = |surface: &FaceSurface, pcurve: &Curve2D, (ta, tb): (f64, f64)| {
            (0..=4000)
                .map(|k| {
                    let s = f64::from(k) / 4000.0;
                    let at = pcurve.evaluate(s);
                    let on = surface.evaluate(at.x(), at.y()).unwrap();
                    (on - ellipse.evaluate((tb - ta).madd(s, ta))).length()
                })
                .fold(0.0, f64::max)
        };
        let mut errors: Vec<(f64, f64)> = Vec::new();
        for offset in [0.0, -1e6] {
            let surface = FaceSurface::Cylinder(
                CylindricalSurface::new(
                    Point3::new(0.0, 0.0, offset),
                    Vec3::new(0.0, 0.0, 1.0),
                    radius,
                )
                .unwrap(),
            );
            for k in 0..16 {
                let p = ellipse.evaluate(f64::from(k) * TAU / 16.0);
                assert!((p.x().hypot(p.y()) - radius).abs() < 1e-9 * radius);
            }
            let start = ellipse.evaluate(0.3);
            let rim = compute_pcurve_on_surface(&curve, start, start, &surface, &[], None);
            let arc = compute_boundary_pcurve_on_surface(
                &curve,
                ellipse.evaluate(0.5),
                ellipse.evaluate(5.5),
                true,
                &surface,
                &[],
                None,
            );
            errors.push((
                worst(&surface, &rim, (0.3, 0.3 + TAU)),
                worst(&surface, &arc, (0.5, 5.5)),
            ));
        }
        let (near, far) = (errors[0], errors[1]);
        for (what, a, b) in [("rim", near.0, far.0), ("arc", near.1, far.1)] {
            assert!(
                a <= 2e-3 && b <= 2e-3 && (a - b).abs() <= 1e-6,
                "{what}: the pcurve stands {a} off with the origin at the centre, {b} with it \
                 1e6 away"
            );
        }
    }
}
