//! Shared helper functions for PaveFiller phases.
//!
//! Extracted from phase_ee, phase_ef, and phase_ve to eliminate
//! duplicated vertex-lookup and pave-insertion logic.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
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
/// torus; `None` for a plane or a NURBS face, or where a wire cannot be
/// boxed. A face's boundary alone does not bound it: a ball less a small
/// cap is bounded by the cap's circle, and holds nearly all of the ball.
///
/// Along any axis, the highest and lowest points of a face lie on its
/// boundary or at a point inside it where the surface's normal is along
/// the axis. On a cylinder or a cone such points fill whole rulings, at one
/// height along each, and a ruling leaves a bounded face through its
/// boundary, at that height; a cone's apex is the one point of its own. On
/// a sphere they are the poles along each axis, counted where the face's
/// wires hold them. A torus face takes the box of its patch.
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
        return Ok(None);
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

/// A NURBS face's box grown to where its surface rises off the boundary
/// between its edges (a lid flat along its rim), past `bounds`, the box of
/// its boundary. Each knot span lies in the box of the control points it
/// blends, so the spans under the face's parameters hold the face. A face
/// whose control points all lie in `bounds` keeps it. A boundary sample
/// that does not project, or a surface closed on itself, takes every span;
/// a weight that is not positive leaves `bounds`, the control points then
/// bounding nothing.
pub(super) fn nurbs_face_box(
    topo: &Topology,
    face_id: FaceId,
    surface: &NurbsSurface,
    bounds: Aabb3,
    tol: Tolerance,
) -> Result<Aabb3, AlgoError> {
    const SAMPLES: u32 = 8;
    let whole = surface.aabb();
    let held = bounds.expanded(tol.linear);
    if !surface.weights().iter().flatten().all(|&w| w > 0.0)
        || (held.contains_point(whole.min) && held.contains_point(whole.max))
    {
        return Ok(bounds);
    }
    if surface.is_periodic_u() || surface.is_periodic_v() {
        return Ok(bounds.union(whole));
    }
    // The parameter range the boundary samples reach, widened by the largest
    // step between neighbouring samples, which an edge can swing out across.
    let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    let (mut step_u, mut step_v) = (0.0_f64, 0.0_f64);
    for eid in brepkit_topology::explorer::face_edges(topo, face_id)? {
        let edge = topo.edge(eid)?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        let mut last: Option<(f64, f64)> = None;
        for i in 0..=SAMPLES {
            let t = t0 + (t1 - t0) * f64::from(i) / f64::from(SAMPLES);
            let p = edge.curve().evaluate_with_endpoints(t, sp, ep);
            // Newton from the sample before, a step along the edge away; a
            // foot off the sample sends it to a search of the whole surface.
            let near = last.and_then(|from| {
                brepkit_math::nurbs::projection::project_point_to_surface_from(
                    surface, p, from, 1e-12,
                )
                .ok()
                .filter(|foot| foot.distance <= crate::classifier::WARM_FOOT)
            });
            let Some(foot) = near.or_else(|| {
                brepkit_math::nurbs::projection::project_point_to_surface(surface, p, tol.linear)
                    .ok()
            }) else {
                return Ok(bounds.union(whole));
            };
            (u0, u1, v0, v1) = (
                u0.min(foot.u),
                u1.max(foot.u),
                v0.min(foot.v),
                v1.max(foot.v),
            );
            if let Some((u, v)) = last {
                step_u = step_u.max((foot.u - u).abs());
                step_v = step_v.max((foot.v - v).abs());
            }
            last = Some((foot.u, foot.v));
        }
    }
    if u0 > u1 {
        return Ok(bounds);
    }
    let cps = surface.control_points();
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let (nu, nv) = (cps.len(), cps[0].len());
    let span = |n, p, x, knots: &[f64]| brepkit_math::nurbs::basis::find_span(n, p, x, knots);
    let (iu0, iu1) = (
        span(nu, pu, u0 - step_u, surface.knots_u()),
        span(nu, pu, u1 + step_u, surface.knots_u()),
    );
    let (iv0, iv1) = (
        span(nv, pv, v0 - step_v, surface.knots_v()),
        span(nv, pv, v1 + step_v, surface.knots_v()),
    );
    Ok(bounds.union(Aabb3::from_points(
        cps[iu0 - pu..=iu1]
            .iter()
            .flat_map(|row| row[iv0 - pv..=iv1].iter().copied()),
    )))
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
