//! Shared helper functions for PaveFiller phases.
//!
//! Extracted from phase_ee, phase_ef, and phase_ve to eliminate
//! duplicated vertex-lookup and pave-insertion logic.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Pave};
use crate::error::AlgoError;

/// The box of an edge's span from `start` to `end`: exact for a line, circle
/// or ellipse, and over `samples + 1` points along any other curve. A circle
/// sampled from a vertex off its axes' extremes misses them by up to its
/// chords' sag, past an edge crossing it there.
pub(super) fn edge_span_box(
    curve: &EdgeCurve,
    start: Point3,
    end: Point3,
    samples: usize,
) -> Aabb3 {
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    match curve {
        EdgeCurve::Line => Aabb3::from_points([start, end]),
        EdgeCurve::Circle(c) => c.arc_aabb(t0, t1),
        EdgeCurve::Ellipse(e) => e.arc_aabb(t0, t1),
        EdgeCurve::NurbsCurve(_) => Aabb3::from_points((0..=samples).map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (i as f64 / samples as f64);
            curve.evaluate_with_endpoints(t, start, end)
        })),
    }
}

/// The box of every wire of a face: exact on lines, circles and ellipses,
/// each B-spline edge in the hulls of its Bezier pieces' nets, and each
/// edge's end vertices, which lie within their tolerance of its curve.
/// `None` where a B-spline edge has no such boxes, or the face has no edge.
fn wires_box(topo: &Topology, face_id: FaceId) -> Result<Option<Aabb3>, AlgoError> {
    let face = topo.face(face_id)?;
    let mut held: Option<Aabb3> = None;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let span = match edge.curve() {
                EdgeCurve::NurbsCurve(_) => {
                    let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
                    let Some(span) =
                        super::phase_ef::curve_piece_boxes(edge.curve(), start, end, t0, t1)
                            .and_then(|boxes| boxes.into_iter().reduce(Aabb3::union))
                    else {
                        return Ok(None);
                    };
                    span
                }
                EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                    edge_span_box(edge.curve(), start, end, 0)
                }
            };
            let span = span.union(Aabb3::from_points([start, end]));
            held = Some(held.map_or(span, |b| b.union(span)));
        }
    }
    Ok(held)
}

/// A box that holds the whole of a face on a cylinder, cone, sphere or
/// torus; `None` for a plane or a NURBS face. A face's boundary alone does
/// not bound it: a ball less a small cap is bounded by the cap's circle,
/// and holds nearly all of the ball.
///
/// Along any axis, the highest and lowest points of a face lie on its
/// boundary or at a point inside it where the surface's normal is along
/// the axis. On a cylinder or a cone such points fill whole rulings, at one
/// height along each, and a ruling leaves a bounded face through its
/// boundary, at that height; a cone's apex is the one point of its own. On
/// a sphere they are the poles along each axis, counted where the face's
/// wires hold them. A torus face takes the box of its patch.
///
/// Where a wire has an edge that cannot be boxed, a ball or a ring face
/// takes the box of its whole surface.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails, or if a cylinder or a
/// cone face has a wire edge that cannot be boxed: nothing then bounds the
/// face along the axis.
pub(super) fn analytic_face_box(
    topo: &Topology,
    face_id: FaceId,
    tol: Tolerance,
) -> Result<Option<Aabb3>, AlgoError> {
    use brepkit_topology::face::FaceSurface;
    let surface = topo.face(face_id)?.surface();
    if matches!(surface, FaceSurface::Plane { .. } | FaceSurface::Nurbs(_)) {
        return Ok(None);
    }
    let Some(wires) = wires_box(topo, face_id)? else {
        return match surface {
            FaceSurface::Sphere(ball) => Ok(Some(ball.aabb())),
            FaceSurface::Torus(ring) => Ok(Some(ring.aabb())),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => Err(AlgoError::IntersectionFailed(
                format!("face {face_id:?} has a wire edge whose span cannot be bounded"),
            )),
            FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => Ok(None),
        };
    };
    let near = 10.0 * tol.linear;
    let with = |b: Aabb3, p: Point3| b.union(Aabb3::from_points([p]));
    Ok(Some(match surface {
        FaceSurface::Cylinder(_) => wires,
        FaceSurface::Cone(cone) => {
            let apex = cone.apex();
            let trim = crate::classifier::LateralTrim::new(topo, face_id)?;
            if trim.is_none_or(|trim| trim.holds(apex, near)) {
                with(wires, apex)
            } else {
                wires
            }
        }
        FaceSurface::Sphere(ball) => {
            let trim = crate::classifier::SphereFaceTrim::new(topo, face_id)?;
            let (c, r) = (ball.center(), ball.radius());
            let mut held = wires;
            for (x, y, z) in [
                (r, 0.0, 0.0),
                (-r, 0.0, 0.0),
                (0.0, r, 0.0),
                (0.0, -r, 0.0),
                (0.0, 0.0, r),
                (0.0, 0.0, -r),
            ] {
                let pole = Point3::new(c.x() + x, c.y() + y, c.z() + z);
                if trim.as_ref().is_none_or(|trim| trim.holds(pole, near)) {
                    held = with(held, pole);
                }
            }
            held
        }
        FaceSurface::Torus(ring) => wires.union(
            super::phase_ff::torus_patch_box(topo, face_id, ring).unwrap_or_else(|| ring.aabb()),
        ),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => return Ok(None),
    }))
}

/// A box that holds the whole of a NURBS face: the box of the control
/// points of its surface cut by knot insertion to the parameters its
/// pcurves bound ([`pcurve_parameters`]), or of the whole surface's control
/// points where they bound none. Each point of a knot span blends the
/// control points the span holds, with weights that are positive and sum to
/// one, so it lies in their box, whether or not the knots are clamped. The
/// box is grown by a bound on the rounding of the knot insertion and of the
/// surface's evaluation, as a share of the largest coordinate of the
/// surface's control points ([`rounding_reach`]): an evaluated point sums
/// `(p + 1)(q + 1)` products of basis functions and weighted control
/// points, each read to a few roundings for degrees `p` and `q`, over the
/// sum of the weighted basis functions, and each knot inserted takes `p` or
/// `q` steps of a rounded blend. The weights are positive, so the quotient
/// is rounded as a share of the control points' size, whatever the weights.
/// The terms can cancel, so a face far smaller than its control points is
/// read with their rounding: on a sheet with control points 1e12 from the
/// origin, a face of size 5 stood 0.28 roundings of 1e12 outside its cut
/// box. Over dense evaluation of a sheet a million and ten billion from the
/// origin, points stood outside the cut box by up to 2 roundings of their
/// coordinates, and outside the whole control box by 1. `reach` is how far
/// the face's own edges and vertices may stand off the surface.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails, or if a weight of the
/// surface is not a positive normal float: its control points then bound
/// nothing.
pub(super) fn nurbs_face_box(
    topo: &Topology,
    face_id: FaceId,
    surface: &NurbsSurface,
    tol: Tolerance,
    reach: f64,
) -> Result<Aabb3, AlgoError> {
    if !surface
        .weights()
        .iter()
        .flatten()
        .all(|&w| w > 0.0 && w.is_normal())
    {
        return Err(AlgoError::IntersectionFailed(format!(
            "face {face_id:?} has a weight that is not a positive normal float, so its control \
             points do not bound it"
        )));
    }
    let held = pcurve_parameters(topo, face_id, surface, tol, reach)?
        .and_then(|range| cut_net_box(surface, range))
        .unwrap_or_else(|| surface.aabb());
    #[allow(clippy::cast_precision_loss)]
    let roundings = {
        let (p, q) = (surface.degree_u(), surface.degree_v());
        (4 * (p + 1) * (q + 1) + 16 * (p + q + 1)) as f64
    };
    let size = surface
        .control_points()
        .iter()
        .flatten()
        .fold(0.0_f64, |m, p| m.max(largest_coordinate(*p)));
    Ok(held.expanded(rounding_reach(size, roundings)))
}

/// A box that holds a face of an analytic surface, `held`, grown by a bound
/// on the rounding of the surface's evaluator. A point sums the surface's
/// centre, origin or apex and three rounded terms along its frame, each a
/// sine or cosine times a radius, or a height along the axis no longer than
/// the point's reach from the centre, so it stands within sixteen roundings
/// of the sum of those terms' sizes of where the box's corners put it. The
/// terms can cancel, so the bound scales by their sizes, not by the box's.
pub(super) fn analytic_rounding_box(
    surface: &brepkit_topology::face::FaceSurface,
    held: Aabb3,
) -> Aabb3 {
    use brepkit_topology::face::FaceSurface;
    let (centre, radii) = match surface {
        FaceSurface::Cylinder(c) => (c.origin(), c.radius()),
        FaceSurface::Cone(c) => (c.apex(), 0.0),
        FaceSurface::Sphere(s) => (s.center(), s.radius()),
        FaceSurface::Torus(t) => (t.center(), t.major_radius() + t.minor_radius()),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => (Point3::new(0.0, 0.0, 0.0), 0.0),
    };
    let frame = largest_coordinate(centre);
    let reach = largest_coordinate(held.min).max(largest_coordinate(held.max)) + frame;
    held.expanded(rounding_reach(2.0 * (reach + frame + radii.abs()), 16.0))
}

fn largest_coordinate(p: Point3) -> f64 {
    p.x().abs().max(p.y().abs()).max(p.z().abs())
}

/// `roundings` roundings of `size`, the largest size of the terms that a
/// surface's evaluator sums for a coordinate: how far a point it computes,
/// with each term rounded no more than that many times all told, can stand
/// past a box whose corners hold the exact point.
fn rounding_reach(size: f64, roundings: f64) -> f64 {
    roundings * f64::EPSILON * size.max(f64::MIN_POSITIVE)
}

/// The box `[u0, u1, v0, v1]` in a NURBS surface's parameters of a face's
/// pcurves, which holds the parameters of every point of the face. On a
/// surface whose sides are free ([`sides_are_free`]), a face reaches a side
/// only along edges of its own, so it is the part of the surface that its
/// wires enclose, and each wire's pcurves run round a closed loop in the
/// parameters, which encloses only points of its own box. The box is grown
/// by the widest gap between the ends of two pcurves that meet.
///
/// `None` where a side of the surface is not proved free within `reach`, an
/// edge of a wire has no pcurve on the face or one that cannot be boxed,
/// the ends of a pcurve do not lie on its edge's vertices, or two pcurves
/// that meet in a wire leave a gap of more than a millionth of the
/// parameters' spans.
fn pcurve_parameters(
    topo: &Topology,
    face_id: FaceId,
    surface: &NurbsSurface,
    tol: Tolerance,
    reach: f64,
) -> Result<Option<[f64; 4]>, AlgoError> {
    use brepkit_math::vec::Point2;
    if !sides_are_free(surface, reach) {
        return Ok(None);
    }
    let ((u_min, u_max), (v_min, v_max)) = (surface.domain_u(), surface.domain_v());
    let span = (u_max - u_min) + (v_max - v_min);
    let face = topo.face(face_id)?;
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut gap = 0.0_f64;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let (mut first, mut last): (Option<Point2>, Option<Point2>) = (None, None);
        for oe in topo.wire(wid)?.edges() {
            let Some(pcurve) = topo.pcurves().get(oe.edge(), face_id) else {
                return Ok(None);
            };
            let Some((low, high)) = pcurve_box(pcurve) else {
                return Ok(None);
            };
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                pcurve.evaluate(pcurve.t_start()),
                pcurve.evaluate(pcurve.t_end()),
            );
            for (uv, vertex) in [(a, edge.start()), (b, edge.end())] {
                let vertex = topo.vertex(vertex)?;
                let reach = (10.0 * tol.linear)
                    .max(vertex.tolerance())
                    .max(edge.tolerance().unwrap_or(0.0));
                let off = (surface.evaluate(uv.x(), uv.y()) - vertex.point()).length();
                if off.is_nan() || off > reach {
                    return Ok(None);
                }
            }
            let (from, to) = if oe.is_forward() { (a, b) } else { (b, a) };
            if let Some(end) = last {
                gap = gap.max((from - end).length());
            }
            first.get_or_insert(from);
            last = Some(to);
            lo = [lo[0].min(low.x()), lo[1].min(low.y())];
            hi = [hi[0].max(high.x()), hi[1].max(high.y())];
        }
        if let (Some(start), Some(end)) = (first, last) {
            gap = gap.max((start - end).length());
        }
    }
    let closed = lo[0] <= hi[0] && lo[1] <= hi[1] && gap <= 1e-6 * span;
    if !closed {
        return Ok(None);
    }
    Ok(Some([lo[0] - gap, hi[0] + gap, lo[1] - gap, hi[1] + gap]))
}

/// Whether each side of a B-spline surface's domain keeps farther than
/// `reach` from the side across from it, along `u` and along `v`: the
/// surface then neither closes on itself, to within `reach`, nor shrinks to
/// a point along a side, as each side runs from one of the other two sides
/// to the other. A side is the curve whose control points blend the
/// surface's rows (or columns) by their basis functions at the domain's
/// end, which holds whether or not the knots are clamped, and its points
/// lie in their box. A surface that closes on itself with its first and
/// last control points apart, as one on knots that are not clamped does,
/// is caught too.
fn sides_are_free(surface: &NurbsSurface, reach: f64) -> bool {
    let net = surface.control_points();
    let weights = surface.weights();
    let (rows, columns) = (net.len(), net.first().map_or(0, Vec::len));
    // The control points of the side at `at` along `u`, or along `v`.
    let side = |along_u: bool, at: f64| -> Option<Aabb3> {
        let (knots, p, count, across) = if along_u {
            (surface.knots_u(), surface.degree_u(), rows, columns)
        } else {
            (surface.knots_v(), surface.degree_v(), columns, rows)
        };
        let span = brepkit_math::nurbs::basis::find_span(count, p, at, knots);
        let basis = brepkit_math::nurbs::basis::basis_funs(span, at, p, knots);
        let first = span.checked_sub(p)?;
        let points = (0..across).map(|j| {
            let (mut sum, mut total) = (Vec3::new(0.0, 0.0, 0.0), 0.0);
            for (k, &b) in basis.iter().enumerate() {
                let (i, j) = if along_u {
                    (first + k, j)
                } else {
                    (j, first + k)
                };
                let w = b * weights[i][j];
                sum += Vec3::new(net[i][j].x(), net[i][j].y(), net[i][j].z()) * w;
                total += w;
            }
            (total > 0.0).then(|| Point3::new(sum.x() / total, sum.y() / total, sum.z() / total))
        });
        Aabb3::try_from_points(points.collect::<Option<Vec<_>>>()?)
    };
    let apart = |a: Option<Aabb3>, b: Option<Aabb3>| matches!((a, b), (Some(a), Some(b)) if !a.expanded(reach).intersects(b));
    let ((u_min, u_max), (v_min, v_max)) = (surface.domain_u(), surface.domain_v());
    apart(side(true, u_min), side(true, u_max)) && apart(side(false, v_min), side(false, v_max))
}

/// A box in the parameters that holds a pcurve over its interval: the
/// interval's ends for a line, the whole curve for a circle or an ellipse,
/// and a B-spline's control points, which hold it where its weights are
/// positive normal floats and the interval lies within its knots. Past
/// them its end pieces run on out of that box. `None` otherwise.
///
/// An ellipse `c + a cos t u + b sin t v` reaches `hypot(a u.x, b v.x)` from
/// its centre along `x` and `hypot(a u.y, b v.y)` along `y`, whichever of its
/// semi-axes is the longer: data read from a file can hold the shorter
/// first. Each reach is grown by the rounding of the evaluator.
fn pcurve_box(
    pcurve: &brepkit_topology::pcurve::PCurve,
) -> Option<(brepkit_math::vec::Point2, brepkit_math::vec::Point2)> {
    use brepkit_math::curves2d::Curve2D;
    use brepkit_math::vec::Point2;
    if !(pcurve.t_start().is_finite()
        && pcurve.t_end().is_finite()
        && pcurve.curve().is_well_formed())
    {
        return None;
    }
    let around = |c: Point2, r: f64| {
        (
            Point2::new(c.x() - r, c.y() - r),
            Point2::new(c.x() + r, c.y() + r),
        )
    };
    let points: Vec<Point2> = match pcurve.curve() {
        Curve2D::Line(_) => vec![
            pcurve.evaluate(pcurve.t_start()),
            pcurve.evaluate(pcurve.t_end()),
        ],
        Curve2D::Circle(c) => return Some(around(c.center(), c.radius())),
        Curve2D::Ellipse(e) => {
            let (a, b) = (e.semi_major(), e.semi_minor());
            let (sin, cos) = e.rotation().sin_cos();
            let c = e.center();
            let grow = 8.0 * f64::EPSILON * (a + b + c.x().abs().max(c.y().abs()));
            let (x, y) = ((a * cos).hypot(b * sin), (a * sin).hypot(b * cos));
            return Some((
                Point2::new(c.x() - x - grow, c.y() - y - grow),
                Point2::new(c.x() + x + grow, c.y() + y + grow),
            ));
        }
        Curve2D::Nurbs(n) => {
            let (first, last) = n.domain();
            let (t0, t1) = (pcurve.t_start(), pcurve.t_end());
            if t0.min(t1) < first
                || t0.max(t1) > last
                || !n.weights().iter().all(|&w| w > 0.0 && w.is_normal())
            {
                return None;
            }
            n.control_points().to_vec()
        }
    };
    let low = points
        .iter()
        .fold(Point2::new(f64::INFINITY, f64::INFINITY), |m, p| {
            Point2::new(m.x().min(p.x()), m.y().min(p.y()))
        });
    let high = points
        .iter()
        .fold(Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY), |m, p| {
            Point2::new(m.x().max(p.x()), m.y().max(p.y()))
        });
    (low.x().is_finite() && low.y().is_finite() && high.x().is_finite() && high.y().is_finite())
        .then_some((low, high))
}

/// The box of the control points of `surface` cut to the parameters
/// `[u0, u1, v0, v1]`, taken within its domain: each cut is a knot inserted
/// to the degree, and the control points kept are those whose basis
/// functions are not zero somewhere inside the range. Where a knot cannot
/// be inserted, the uncut net's control points of the spans the range
/// meets, which hold it too. `None` for a range that holds no area of the
/// domain.
fn cut_net_box(surface: &NurbsSurface, [u0, u1, v0, v1]: [f64; 4]) -> Option<Aabb3> {
    use brepkit_math::nurbs::knot_ops::{surface_knot_insert_u, surface_knot_insert_v};
    let ((u_min, u_max), (v_min, v_max)) = (surface.domain_u(), surface.domain_v());
    let (u0, u1, v0, v1) = (u0.max(u_min), u1.min(u_max), v0.max(v_min), v1.min(v_max));
    if !(u0 < u1 && v0 < v1) {
        return None;
    }
    let mut cut = surface.clone();
    for u in [u0, u1] {
        if u > u_min && u < u_max {
            cut = surface_knot_insert_u(&cut, u, cut.degree_u()).unwrap_or(cut);
        }
    }
    for v in [v0, v1] {
        if v > v_min && v < v_max {
            cut = surface_knot_insert_v(&cut, v, cut.degree_v()).unwrap_or(cut);
        }
    }
    // A basis function of degree `p` from knot `i` is not zero inside the
    // range where its support, knots `i` to `i + p + 1`, overlaps it.
    let kept = |knots: &[f64], p: usize, count: usize, (a, b): (f64, f64)| -> Vec<usize> {
        (0..count)
            .filter(|&i| knots[i] < b && knots[i + p + 1] > a)
            .collect()
    };
    let net = cut.control_points();
    let columns = net.first().map_or(0, Vec::len);
    let rows = kept(cut.knots_u(), cut.degree_u(), net.len(), (u0, u1));
    let cols = kept(cut.knots_v(), cut.degree_v(), columns, (v0, v1));
    Aabb3::try_from_points(
        rows.iter()
            .flat_map(|&i| cols.iter().map(move |&j| net[i][j])),
    )
}

/// Find a vertex near the given point among all pave block vertices.
///
/// Returns the resolved (same-domain canonical) vertex first encountered within
/// `tol.linear` of `point`, scanning pave blocks in `edge_pave_blocks` order
/// (ascending `EdgeId`, start-before-end). When the arena's spatial index is
/// available (built after Phase VV) the lookup is O(1) and returns the exact
/// same vertex; otherwise it falls back to the linear scan.
pub(super) fn find_nearby_pave_vertex(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    tol: Tolerance,
) -> Option<VertexId> {
    if let Some(index) = &arena.pave_vertex_index {
        return index.find_within(point, tol.linear);
    }
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    crate::perf::bump_pave_vertex_probe();
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved)
                        && (v.point() - point).length() <= tol.linear
                    {
                        return Some(resolved);
                    }
                }
            }
        }
    }
    None
}

/// The nearest pave vertex whose tolerance ball, or a new vertex's ball of
/// `reach`, holds `p`. A vertex carried over from an inexact operand covers
/// the fit error of every point computed for its corner, so a crossing or a
/// section end found there is that vertex's.
pub(super) fn pave_vertex_within_tolerance(
    topo: &Topology,
    arena: &GfaArena,
    p: Point3,
    reach: f64,
    tol: Tolerance,
) -> Option<VertexId> {
    if let Some(balls) = &arena.pave_ball_index
        && balls.answers(p, reach)
    {
        let found = balls.nearest_within(topo, p, reach, tol.linear);
        // Every tolerance raise must reach the index; the scan below is the
        // reference it stands in for.
        debug_assert_eq!(
            found,
            pave_vertex_within_tolerance_scan(topo, arena, p, reach, tol)
        );
        return found;
    }
    pave_vertex_within_tolerance_scan(topo, arena, p, reach, tol)
}

fn pave_vertex_within_tolerance_scan(
    topo: &Topology,
    arena: &GfaArena,
    p: Point3,
    reach: f64,
    tol: Tolerance,
) -> Option<VertexId> {
    let mut best: Option<(f64, VertexId)> = None;
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            let ends = [pb.start.vertex, pb.end.vertex];
            for vid in ends
                .into_iter()
                .chain(pb.extra_paves.iter().map(|pave| pave.vertex))
            {
                let resolved = arena.resolve_vertex(vid);
                let Ok(v) = topo.vertex(resolved) else {
                    continue;
                };
                let ball = v.tolerance().max(reach);
                let d = (v.point() - p).length();
                if ball > tol.linear && d <= ball && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, resolved));
                }
            }
        }
    }
    best.map(|(_, v)| v)
}

/// The gap below which two computations of one junction read as the same
/// point.
const WELD_BAND: f64 = 1e-5;

/// The vertex where a boundary edge of one face crosses the other face,
/// nearest `p` within ten times its tolerance or the weld band. A section of
/// two faces leaves each one through its boundary, so its end there is that
/// edge's crossing of the other face, however far the section's own end and
/// the crossing drift apart: on operands whose vertices sit off their faces,
/// or where the section's end was clipped against a curved boundary's
/// chords. The vertex's tolerance grows to hold `p`.
pub(super) fn ef_crossing_vertex(
    topo: &mut Topology,
    arena: &GfaArena,
    (fa, fb): (FaceId, FaceId),
    p: Point3,
) -> Option<VertexId> {
    let boundary = |f: FaceId| -> Vec<EdgeId> {
        topo.face(f).ok().map_or_else(Vec::new, |face| {
            std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .filter_map(|w| topo.wire(w).ok())
                .flat_map(|w| {
                    w.edges()
                        .iter()
                        .map(brepkit_topology::wire::OrientedEdge::edge)
                })
                .collect()
        })
    };
    let (edges_a, edges_b) = (boundary(fa), boundary(fb));
    let mut best: Option<(f64, VertexId)> = None;
    for i in &arena.interference.ef {
        let crate::ds::Interference::EF {
            edge,
            face,
            new_vertex: Some(v),
            ..
        } = i
        else {
            continue;
        };
        let across =
            (*face == fb && edges_a.contains(edge)) || (*face == fa && edges_b.contains(edge));
        if !across {
            continue;
        }
        let v = arena.resolve_vertex(*v);
        let Ok(vertex) = topo.vertex(v) else {
            continue;
        };
        let d = (vertex.point() - p).length();
        if d <= (10.0 * vertex.tolerance()).max(WELD_BAND) && best.is_none_or(|(b, _)| d < b) {
            best = Some((d, v));
        }
    }
    let (d, v) = best?;
    if let Ok(vertex) = topo.vertex_mut(v)
        && vertex.tolerance() < d
    {
        vertex.set_tolerance(d);
        if let Some(balls) = &arena.pave_ball_index {
            balls.note_tolerance(v, d);
        }
    }
    Some(v)
}

/// Widened variant of [`find_nearby_pave_vertex`] for tangential contacts.
///
/// A grazing crossing's solved position is only accurate to
/// `sqrt(2 * r * residual)`, so the exact junction vertex can sit microns
/// outside the linear tolerance. This scans every pave-block endpoint within
/// `radius` and returns the nearest candidate that passes `accept` (the
/// caller checks genuine curve/surface incidence, which is what makes the
/// widened radius safe). If the accepted candidates span more than one
/// distinct position (beyond `tol_linear` of each other), the contact is
/// ambiguous — two different junctions inside the window — and `None` is
/// returned so the caller keeps the solved point rather than merging
/// distinct junctions. The spatial index is deliberately not used: its
/// 3x3x3 cell stencil is exhaustive only for radius <= one tolerance cell.
pub(super) fn find_nearby_pave_vertex_widened(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    radius: f64,
    tol_linear: f64,
    accept: impl Fn(Point3) -> bool,
) -> Option<VertexId> {
    let mut best: Option<(f64, Point3, VertexId)> = None;
    let mut ambiguous = false;
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved) {
                        let d = (v.point() - point).length();
                        if d <= radius && accept(v.point()) {
                            match &best {
                                Some((bd, bp, _)) => {
                                    if (v.point() - *bp).length() > tol_linear {
                                        ambiguous = true;
                                    }
                                    if d < *bd {
                                        best = Some((d, v.point(), resolved));
                                    }
                                }
                                None => best = Some((d, v.point(), resolved)),
                            }
                        }
                    }
                }
            }
        }
    }
    if ambiguous {
        return None;
    }
    best.map(|(_, _, v)| v)
}

/// Add a pave to the appropriate pave block of an edge.
///
/// Finds the pave block whose parameter range contains the pave's
/// parameter (with a small guard band) and adds the extra pave to it.
pub(super) fn add_pave_to_edge(arena: &mut GfaArena, topo: &Topology, edge_id: EdgeId, pave: Pave) {
    if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id) {
        let pb_ids_copy: Vec<_> = pb_ids.clone();
        for pb_id in pb_ids_copy {
            if block_spans_interior(topo, arena, pb_id, pave.parameter) {
                arena.add_extra_pave(topo, pb_id, pave);
            }
        }
    }
}

/// The foot of `p` on an edge's curve between the parameters `range`, as
/// its parameter and its distance from `p`.
///
/// `ends` are the edge's vertex positions. `samples` are places on that part
/// of the curve, in order, each with its parameter: the nearest chord
/// between them gives the start.
pub(super) fn foot_on_curve(
    p: Point3,
    curve: &EdgeCurve,
    ends: (Point3, Point3),
    range: (f64, f64),
    samples: &[(f64, Point3)],
) -> (f64, f64) {
    let at = |t: f64| curve.evaluate_with_endpoints(t, ends.0, ends.1);
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    let mut nearest = (f64::INFINITY, lo);
    for chord in samples.windows(2) {
        let dir = chord[1].1 - chord[0].1;
        let len2 = dir.dot(dir);
        let s = if len2 > 0.0 {
            ((p - chord[0].1).dot(dir) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let gap2 = (chord[0].1 + dir * s - p).length_squared();
        if gap2 < nearest.0 {
            nearest = (gap2, s.madd(chord[1].0 - chord[0].0, chord[0].0));
        }
    }
    let mut t = nearest.1.clamp(lo, hi);
    let h = (hi - lo) * 1e-7;
    if h > 0.0 {
        for _ in 0..20 {
            let (below, above) = ((t - h).max(lo), (t + h).min(hi));
            let speed = (at(above) - at(below)) * (1.0 / (above - below));
            let speed2 = speed.dot(speed);
            if speed2 <= 0.0 {
                break;
            }
            let next = (t + (p - at(t)).dot(speed) / speed2).clamp(lo, hi);
            let moved = (next - t).abs();
            t = next;
            if moved <= 1e-11 * (hi - lo) {
                break;
            }
        }
    }
    (t, (at(t) - p).length())
}

/// The angular parameter of a closed circle or ellipse edge's seam vertex.
pub(super) fn closed_seam_parameter(topo: &Topology, edge_id: EdgeId) -> Option<f64> {
    let edge = topo.edge(edge_id).ok()?;
    let sp = topo.vertex(edge.start()).ok()?.point();
    match edge.curve() {
        EdgeCurve::Circle(c) => Some(c.project(sp)),
        EdgeCurve::Ellipse(e) => Some(e.project(sp)),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    }
}

/// Whether `t` lies strictly inside a pave block.
///
/// The whole block of a closed circle or ellipse spans the curve's own
/// period from its angular origin, but the block's two ends are the edge's
/// one vertex, which sits wherever the seam is. The origin is then a point
/// inside the block and the seam's parameter is not, so the test runs from
/// the seam.
pub(super) fn block_spans_interior(
    topo: &Topology,
    arena: &GfaArena,
    pb_id: crate::ds::PaveBlockId,
    t: f64,
) -> bool {
    use std::f64::consts::TAU;
    let Some(pb) = arena.pave_blocks.get(pb_id) else {
        return false;
    };
    let (a, b) = pb.parameter_range();
    let whole_turn = ((b - a).abs() - TAU).abs() < 1e-9
        && arena.resolve_vertex(pb.start.vertex) == arena.resolve_vertex(pb.end.vertex);
    if whole_turn && let Some(seam) = closed_seam_parameter(topo, pb.original_edge) {
        let from_seam = (t - seam).rem_euclid(TAU);
        return from_seam > 1e-10 && from_seam < TAU - 1e-10;
    }
    pb.spans_interior(t)
}
