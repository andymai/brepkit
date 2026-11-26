//! Phase FF-Coplanar: coplanar face splitting.
//!
//! Handles the case where two faces from different solids lie on the same
//! plane and partially overlap. Phase FF skips these because parallel planes
//! have no intersection line. This phase runs after FF and creates section
//! edges by clipping one face's boundary edges (segments and arcs, exactly)
//! to the other face's region.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Interference, IntersectionCurveDS, Pave, PaveBlock, PaveBlockId};
use crate::error::AlgoError;

use super::helpers::find_nearby_pave_vertex;
use super::phase_ff::find_nearby_face_vertex;

/// Detect coplanar face pairs between two solids and create section edges
/// for boundary edges of one face that lie inside the other.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    let planes_a = collect_plane_faces(topo, &faces_a)?;
    let planes_b = collect_plane_faces(topo, &faces_b)?;

    if planes_a.is_empty() || planes_b.is_empty() {
        return Ok(());
    }

    let bboxes_a = compute_face_bboxes(topo, &planes_a)?;
    let bboxes_b = compute_face_bboxes(topo, &planes_b)?;

    log::debug!(
        "FF-coplanar: checking {} × {} plane face pairs",
        planes_a.len(),
        planes_b.len()
    );

    // Boundaries are projected once per face into the plane's canonical
    // frame (the same frame for every pair on that plane, either normal
    // sign), so a face paired many times pays its arc sampling once.
    let mut regions: HashMap<FaceId, Region2> = HashMap::new();
    // Vertices this phase may weld to: the regular FF phase's section
    // endpoints (the pave vertex index predates phase FF and never sees
    // them) plus every vertex minted here, so a crossing computed once along
    // an arc and once along a line, or shared with a regular section, is one
    // vertex.
    let mut minted: Vec<(Point3, VertexId)> = Vec::new();
    for curve in &arena.curves {
        for &pb_id in &curve.pave_blocks {
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            for pave in [pb.start, pb.end] {
                let vid = arena.resolve_vertex(pave.vertex);
                if let Ok(v) = topo.vertex(vid) {
                    minted.push((v.point(), vid));
                }
            }
        }
    }

    for (idx_a, &(fa, na, da)) in planes_a.iter().enumerate() {
        let bbox_a = &bboxes_a[idx_a];

        for (idx_b, &(fb, nb, db)) in planes_b.iter().enumerate() {
            let bbox_b = &bboxes_b[idx_b];

            let dot = na.dot(nb);
            if dot.abs() < 1.0 - tol.angular {
                continue;
            }

            // Coplanar test accounts for normal direction: anti-parallel
            // normals describe the same plane when da == -db.
            let sign = if dot > 0.0 { 1.0 } else { -1.0 };
            if (da - db * sign).abs() > tol.linear {
                continue;
            }

            if !bbox_a
                .expanded(tol.linear)
                .intersects(bbox_b.expanded(tol.linear))
            {
                continue;
            }

            if has_existing_ff_interference(arena, fa, fb) {
                continue;
            }

            let frame = PlaneFrame2D::canonical(na, da);
            for fid in [fa, fb] {
                if let Entry::Vacant(slot) = regions.entry(fid) {
                    let mut region = face_region_2d(topo, fid, &frame, tol.linear)?;
                    attach_paves(&mut region, topo, arena);
                    slot.insert(region);
                }
            }
            process_coplanar_pair(
                topo,
                arena,
                &mut minted,
                (fa, &regions[&fa]),
                (fb, &regions[&fb]),
                &frame,
                tol,
            )?;
        }
    }

    Ok(())
}

/// Collect `(FaceId, normal, d)` for all plane faces in the list.
fn collect_plane_faces(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<(FaceId, Vec3, f64)>, AlgoError> {
    let mut result = Vec::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            result.push((fid, *normal, *d));
        }
    }
    Ok(result)
}

/// Compute AABBs for plane faces by sampling boundary edges.
fn compute_face_bboxes(
    topo: &Topology,
    planes: &[(FaceId, Vec3, f64)],
) -> Result<Vec<Aabb3>, AlgoError> {
    let mut bboxes = Vec::with_capacity(planes.len());
    for &(fid, _, _) in planes {
        bboxes.push(compute_face_bbox(topo, fid)?);
    }
    Ok(bboxes)
}

/// Compute AABB for a face by sampling its boundary edges.
fn compute_face_bbox(topo: &Topology, face_id: FaceId) -> Result<Aabb3, AlgoError> {
    let edges = brepkit_topology::explorer::face_edges(topo, face_id)?;
    let mut points = Vec::new();

    for eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        let n: usize = 8;
        for i in 0..=n {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            points.push(pt);
        }
    }

    Aabb3::try_from_points(points).ok_or_else(|| {
        AlgoError::IntersectionFailed(format!(
            "face {face_id:?} has no boundary point, or one that is not finite"
        ))
    })
}

/// Whether a section on the face the piece must split already runs from
/// `p_start` to `p_end`, either way.
///
/// Searches `arena.curves` for an intersection curve attached to `target`
/// that runs along the piece from one of its ends to the other, so the
/// coplanar phase does not duplicate a section the regular FF phase already
/// gave that face. A longer section on the same line can have the middle of
/// the piece, and a curved one its two ends. A curve attached only to the
/// other face of the pair does not count: it never splits `target`.
fn has_existing_section_at(
    arena: &GfaArena,
    topo: &Topology,
    target: FaceId,
    p_start: Point3,
    p_end: Point3,
    tol: Tolerance,
) -> bool {
    let near = |p: Point3, q: Point3| (p - q).length() < tol.linear * 10.0;
    let mid = lerp3(p_start, p_end, 0.5);
    let piece = Aabb3::from_points([p_start, p_end]);
    arena.curves.iter().any(|curve| {
        if (curve.face_a != target && curve.face_b != target)
            || !curve.bbox.expanded(tol.linear).intersects(piece)
        {
            return false;
        }
        let Some(((s, t_s), (e, t_e))) = section_ends(topo, arena, curve) else {
            return false;
        };
        if !((near(s, p_start) && near(e, p_end)) || (near(s, p_end) && near(e, p_start))) {
            return false;
        }
        // A line section with the piece's two ends is the piece, whether its
        // parameters run over its length or over 0 to 1. An arc of a circle
        // or an ellipse stands off its chord the most at the middle of its
        // parameters, where its tangent runs along the chord. A B-spline
        // lies in the hull of its Bezier poles when their weights are
        // positive, whatever its speed along them.
        match &curve.curve {
            EdgeCurve::Line => true,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => near(
                curve.curve.evaluate_with_endpoints(0.5 * (t_s + t_e), s, e),
                mid,
            ),
            EdgeCurve::NurbsCurve(nurbs) => {
                spline_along_segment(nurbs, (t_s, t_e), (p_start, p_end), tol.linear * 10.0)
            }
        }
    })
}

/// Whether the part of `curve` between the parameters `t_s` and `t_e` lies
/// within `reach` of the segment from `a` to `b`: each Bezier pole of each
/// piece between its knots does, with a positive weight.
fn spline_along_segment(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    (t_s, t_e): (f64, f64),
    (a, b): (Point3, Point3),
    reach: f64,
) -> bool {
    let (lo, hi) = (t_s.min(t_e), t_s.max(t_e));
    let mut cuts = vec![lo];
    cuts.extend(curve.knots().iter().copied().filter(|&k| k > lo && k < hi));
    cuts.push(hi);
    let along = b - a;
    let length2 = along.dot(along);
    cuts.windows(2).all(|piece| {
        brepkit_math::nurbs::decompose::curve_bezier_between(curve, piece[0], piece[1]).is_ok_and(
            |poles| {
                poles.into_iter().all(|[x, y, z, w]| {
                    if w.is_nan() || w <= 0.0 {
                        return false;
                    }
                    let p = Point3::new(x / w, y / w, z / w);
                    let s = if length2 > 0.0 {
                        ((p - a).dot(along) / length2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    (p - (a + along * s)).length() <= reach
                })
            },
        )
    })
}

/// The two ends of a section, each with its parameter: the paves of its
/// pave blocks at the lowest and the highest parameter.
fn section_ends(
    topo: &Topology,
    arena: &GfaArena,
    curve: &IntersectionCurveDS,
) -> Option<((Point3, f64), (Point3, f64))> {
    let paves = curve
        .pave_blocks
        .iter()
        .filter_map(|&pb| arena.pave_blocks.get(pb))
        .flat_map(|pb| [pb.start, pb.end]);
    let first = paves
        .clone()
        .min_by(|a, b| a.parameter.total_cmp(&b.parameter))?;
    let last = paves.max_by(|a, b| a.parameter.total_cmp(&b.parameter))?;
    let at = |p: Pave| {
        topo.vertex(arena.resolve_vertex(p.vertex))
            .ok()
            .map(|v| (v.point(), p.parameter))
    };
    Some((at(first)?, at(last)?))
}

/// Check if an FF interference already exists for this face pair.
fn has_existing_ff_interference(arena: &GfaArena, fa: FaceId, fb: FaceId) -> bool {
    arena.interference.ff.iter().any(|interf| {
        matches!(interf,
            Interference::FF { f1, f2, .. } if (*f1 == fa && *f2 == fb) || (*f1 == fb && *f2 == fa)
        )
    })
}

/// Process a single coplanar face pair: clip each face's boundary edges to
/// the other face's region and create section edges for the pieces inside.
///
/// Boundary arcs are handled as arcs, never as their chords. A chord that
/// deviates from its arc by more than tolerance puts a crossing off the true
/// boundary (a knuckle end's radius truncated where the pin cap's chord
/// crosses it) or invents one (a bore arc's chord crossing a pin diameter the
/// arc never reaches), and the phantom pieces then split the partner face
/// into slivers nothing pairs with.
fn process_coplanar_pair(
    topo: &mut Topology,
    arena: &mut GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    (face_a, region_a): (FaceId, &Region2),
    (face_b, region_b): (FaceId, &Region2),
    frame: &PlaneFrame2D,
    tol: Tolerance,
) -> Result<(), AlgoError> {
    // A boundary edge lying on the partner's boundary (its whole length on
    // one partner edge) is the faces' common boundary, not a dividing
    // section; it gets a common block below instead.
    let mut pieces: Vec<SectionPiece> = Vec::new();
    for e in &region_b.edges {
        if coincident_boundary_edge(e, &region_a.edges, tol.linear).is_none() {
            clip_to_region(e, (face_a, region_a), frame, tol.linear, &mut pieces)?;
        }
    }
    for e in &region_a.edges {
        if coincident_boundary_edge(e, &region_b.edges, tol.linear).is_none() {
            clip_to_region(e, (face_b, region_b), frame, tol.linear, &mut pieces)?;
        }
    }
    let faces = [face_a, face_b];
    for piece in pieces {
        match piece {
            SectionPiece::Line { target, start, end } => {
                if !has_existing_section_at(arena, topo, target, start, end, tol) {
                    create_section_edge(topo, arena, minted, faces, start, end, tol)?;
                }
            }
            SectionPiece::Arc {
                target,
                circle,
                start,
                end,
                mid,
            } => {
                // The wall sharing this arc meets the partner plane in the
                // same circle, so the regular FF phase usually emitted the
                // arc already, split at the same partner crossings; only
                // what it left uncovered ON THE FACE TO SPLIT is emitted
                // here (a section attached to the other face alone leaves
                // the target unsplit).
                let (t_s, t_e, p_s, p_e) = oriented_arc_span(&circle, start, end, mid);
                let ang_tol = tol.linear * 10.0 / circle.radius();
                for (a, b) in uncovered_arc_spans(arena, target, &circle, t_s, t_e, ang_tol) {
                    let p0 = if (a - t_s).abs() <= ang_tol {
                        p_s
                    } else {
                        circle.evaluate(a)
                    };
                    let p1 = if (b - t_e).abs() <= ang_tol {
                        p_e
                    } else {
                        circle.evaluate(b)
                    };
                    create_arc_section_edges(
                        topo,
                        arena,
                        minted,
                        faces,
                        &circle,
                        (p0, p1),
                        (a, b),
                        tol,
                    );
                }
            }
            SectionPiece::Curve {
                target,
                along,
                span,
                start,
                end,
                mid,
            } => {
                if !has_existing_section_along(
                    arena,
                    topo,
                    target,
                    (&along.curve, start, end),
                    mid,
                    tol,
                ) {
                    create_curve_section_edge(
                        topo,
                        arena,
                        minted,
                        faces,
                        &along,
                        (start, end),
                        span,
                        tol,
                    );
                }
            }
        }
    }

    // For each boundary edge of face_b that coincides with a boundary edge
    // of face_a, create a CommonBlock linking their PaveBlocks. This enables
    // edge sharing for flush-face (touching) booleans where the faces share
    // a boundary segment.
    for e in &region_b.edges {
        if let Some(ai) = coincident_boundary_edge(e, &region_a.edges, tol.linear) {
            create_coplanar_common_block(arena, region_a.edges[ai].eid, e.eid, tol.linear);
        }
    }

    Ok(())
}

/// A circular arc in the pair's 2D frame, from angle `a0` sweeping `sweep`
/// radians (signed; positive is counter-clockwise in the frame).
#[derive(Clone)]
struct Arc2 {
    center: Point2,
    radius: f64,
    a0: f64,
    sweep: f64,
    circle: brepkit_math::curves::Circle3D,
}

/// The 2D shape of a boundary edge: a straight segment, an exact arc, or a
/// sampled polyline for the other curve kinds (their region test and
/// crossings use the samples; as a section source they keep their chord).
enum Shape2 {
    Seg,
    Arc(Arc2),
    Poly(Vec<Point2>),
}

/// A boundary edge of a coplanar face, in wire traversal order.
struct BoundaryEdge {
    eid: brepkit_topology::edge::EdgeId,
    p2_start: Point2,
    p2_end: Point2,
    p3_start: Point3,
    p3_end: Point3,
    shape: Shape2,
    /// The ellipse a sampled edge stands for, which its pieces follow.
    along: Option<CurveSpan>,
    /// The paves an ellipse edge has from the earlier phases, each a point
    /// and its parameter: where it crosses or touches another edge.
    paves: Vec<(Point3, f64)>,
    /// The edge's curve and its ends in the edge's own order.
    own: (EdgeCurve, Point3, Point3),
}

/// An edge's curve over the edge, read in its wire's direction: the
/// sampled polyline's parameter `s` in `[0, n]` stands for the curve's at
/// the same fraction of `span`, from its start in the wire's direction.
#[derive(Clone)]
struct CurveSpan {
    curve: EdgeCurve,
    /// The edge's ends in its own order, which the curve reads its
    /// parameters from.
    ends: (Point3, Point3),
    span: (f64, f64),
    forward: bool,
}

impl CurveSpan {
    /// The curve's parameter at the polyline's `s` of `n`.
    fn at(&self, s: f64, n: f64) -> f64 {
        let (t0, t1) = self.span;
        let f = if self.forward { s / n } else { 1.0 - s / n };
        (t1 - t0).madd(f, t0)
    }

    fn point(&self, t: f64) -> Point3 {
        self.curve
            .evaluate_with_endpoints(t, self.ends.0, self.ends.1)
    }

    /// The parameter in the span of the ellipse's point nearest `p`, where
    /// `p` lies within `reach` of it. `None` for any other curve.
    fn through(&self, p: Point3, reach: f64) -> Option<f64> {
        let EdgeCurve::Ellipse(e) = &self.curve else {
            return None;
        };
        let t = e.project(p);
        ((e.evaluate(t) - p).length() <= reach)
            .then(|| self.param_of(p, t))
            .flatten()
    }

    /// The parameter in the span of a pave at `p` whose own parameter is
    /// `t`: an ellipse's read round a turn into the span from the point, a
    /// B-spline's as it is. `None` past the span.
    fn param_of(&self, p: Point3, t: f64) -> Option<f64> {
        let (lo, hi) = (self.span.0.min(self.span.1), self.span.0.max(self.span.1));
        let t = match &self.curve {
            EdgeCurve::Ellipse(e) => lo + (e.project(p) - lo).rem_euclid(std::f64::consts::TAU),
            _ => t,
        };
        let slack = 1e-9 * (hi - lo).abs().max(1.0);
        (t >= lo - slack && t <= hi + slack).then_some(t.clamp(lo, hi))
    }
}

/// A face's boundary in the plane's frame: every wire's edges, and the
/// bounds of their endpoints for the containment ray.
struct Region2 {
    edges: Vec<BoundaryEdge>,
    lo: Point2,
    hi: Point2,
    /// The face's region read on its own curves, holes included: lines,
    /// circles and ellipses as they are, a B-spline by its bounded Bezier
    /// pieces. `None` where it cannot be built.
    exact: Option<brepkit_topology::planar::PlanarRegion>,
}

impl Region2 {
    /// Whether `p` lies inside the region and farther than `tol` from its
    /// boundary: read on the region's own curves where they are known, an
    /// answer they cannot settle counting as not inside, and else on the
    /// sampled boundary.
    fn holds(&self, p: Point2, tol: f64) -> bool {
        match &self.exact {
            Some(region) => region.side(p, tol) == Some(true),
            None => !on_boundary(p, &self.edges, tol) && inside_region(p, self, tol),
        }
    }
}

/// A clipped piece of a boundary edge inside the partner face; `target` is
/// that partner, the face the piece must split.
enum SectionPiece {
    Line {
        target: FaceId,
        start: Point3,
        end: Point3,
    },
    Arc {
        target: FaceId,
        circle: brepkit_math::curves::Circle3D,
        start: Point3,
        end: Point3,
        mid: Point3,
    },
    /// A piece of an ellipse boundary edge, along its curve
    /// from parameter `span.0` (at `start`) to `span.1` (at `end`),
    /// `span.0 < span.1`.
    Curve {
        target: FaceId,
        along: CurveSpan,
        span: (f64, f64),
        start: Point3,
        end: Point3,
        mid: Point3,
    },
}

/// Samples along a curved non-circle boundary edge.
const POLY_SAMPLES: usize = 16;

fn angle_of(center: Point2, p: Point2) -> f64 {
    (p.y() - center.y()).atan2(p.x() - center.x())
}

fn arc_point(arc: &Arc2, s: f64) -> Point2 {
    let phi = if arc.sweep >= 0.0 {
        arc.a0 + s
    } else {
        arc.a0 - s
    };
    Point2::new(
        arc.radius.madd(phi.cos(), arc.center.x()),
        arc.radius.madd(phi.sin(), arc.center.y()),
    )
}

/// Offset along the arc (in its sweep direction) of the point at angle
/// `phi`, when that point lies on the arc within `ang_tol`.
fn arc_offset_of(arc: &Arc2, phi: f64, ang_tol: f64) -> Option<f64> {
    use std::f64::consts::TAU;
    let s = if arc.sweep >= 0.0 {
        (phi - arc.a0).rem_euclid(TAU)
    } else {
        (arc.a0 - phi).rem_euclid(TAU)
    };
    let len = arc.sweep.abs();
    if s <= len + ang_tol {
        Some(s.min(len))
    } else if s >= TAU - ang_tol {
        Some(0.0)
    } else {
        None
    }
}

fn lerp3(a: Point3, b: Point3, t: f64) -> Point3 {
    Point3::new(
        (b.x() - a.x()).madd(t, a.x()),
        (b.y() - a.y()).madd(t, a.y()),
        (b.z() - a.z()).madd(t, a.z()),
    )
}

/// Collect a face's boundary (outer and inner wires) in the frame, its
/// exact region read to the linear tolerance `tol`.
fn face_region_2d(
    topo: &Topology,
    face_id: FaceId,
    frame: &PlaneFrame2D,
    tol: f64,
) -> Result<Region2, AlgoError> {
    use std::f64::consts::TAU;
    let face = topo.face(face_id)?;
    let mut edges = Vec::new();
    let mut lo = Point2::new(f64::INFINITY, f64::INFINITY);
    let mut hi = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut grow = |p: Point2| {
        lo = Point2::new(lo.x().min(p.x()), lo.y().min(p.y()));
        hi = Point2::new(hi.x().max(p.x()), hi.y().max(p.y()));
    };
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (v_start, v_end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (p3_start, p3_end) = if oe.is_forward() {
                (v_start, v_end)
            } else {
                (v_end, v_start)
            };
            let p2_start = frame.project(p3_start);
            let p2_end = frame.project(p3_end);
            grow(p2_start);
            grow(p2_end);
            let shape = match edge.curve() {
                EdgeCurve::Line => Shape2::Seg,
                EdgeCurve::Circle(circle) => {
                    let center = frame.project(circle.center());
                    let radius = circle.radius();
                    let a0 = angle_of(center, p2_start);
                    let a1 = angle_of(center, p2_end);
                    let closed = (v_start - v_end).length() < 1e-9;
                    let sweep = if closed {
                        // A closed rim runs in the circle's own direction,
                        // which the frame sees as CCW iff the normals agree.
                        let ccw = circle.normal().dot(frame.normal) > 0.0;
                        if ccw == oe.is_forward() { TAU } else { -TAU }
                    } else {
                        // The arc runs through its native midpoint; pick the
                        // sweep direction that reaches it from the start.
                        let (t0, t1) = edge.curve().domain_with_endpoints(v_start, v_end);
                        let mid2 = frame.project(circle.evaluate(0.5 * (t0 + t1)));
                        let ccw = (a1 - a0).rem_euclid(TAU);
                        let to_mid = (angle_of(center, mid2) - a0).rem_euclid(TAU);
                        if to_mid <= ccw { ccw } else { ccw - TAU }
                    };
                    grow(Point2::new(center.x() - radius, center.y() - radius));
                    grow(Point2::new(center.x() + radius, center.y() + radius));
                    Shape2::Arc(Arc2 {
                        center,
                        radius,
                        a0,
                        sweep,
                        circle: circle.clone(),
                    })
                }
                EdgeCurve::NurbsCurve(nurbs)
                    if let Some(arc) = nurbs_arc(
                        edge.curve(),
                        nurbs,
                        (v_start, v_end),
                        (p2_start, p2_end),
                        oe.is_forward(),
                        frame,
                    ) =>
                {
                    grow(Point2::new(
                        arc.center.x() - arc.radius,
                        arc.center.y() - arc.radius,
                    ));
                    grow(Point2::new(
                        arc.center.x() + arc.radius,
                        arc.center.y() + arc.radius,
                    ));
                    Shape2::Arc(arc)
                }
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                    let (t0, t1) = edge.curve().domain_with_endpoints(v_start, v_end);
                    let mut samples: Vec<Point2> = (1..POLY_SAMPLES)
                        .map(|k| {
                            let t = (t1 - t0).madd(k as f64 / POLY_SAMPLES as f64, t0);
                            frame.project(edge.curve().evaluate_with_endpoints(t, v_start, v_end))
                        })
                        .collect();
                    if !oe.is_forward() {
                        samples.reverse();
                    }
                    for &p in &samples {
                        grow(p);
                    }
                    let mut poly = Vec::with_capacity(POLY_SAMPLES + 1);
                    poly.push(p2_start);
                    poly.extend(samples);
                    poly.push(p2_end);
                    Shape2::Poly(poly)
                }
            };
            let along = matches!(
                (&shape, edge.curve()),
                (Shape2::Poly(_), EdgeCurve::Ellipse(_))
            )
            .then(|| CurveSpan {
                curve: edge.curve().clone(),
                ends: (v_start, v_end),
                span: edge.curve().domain_with_endpoints(v_start, v_end),
                forward: oe.is_forward(),
            });
            edges.push(BoundaryEdge {
                eid: oe.edge(),
                p2_start,
                p2_end,
                p3_start,
                p3_end,
                shape,
                along,
                paves: Vec::new(),
                own: (edge.curve().clone(), v_start, v_end),
            });
        }
    }
    let exact = brepkit_topology::planar::face_region_2d(
        topo,
        face_id,
        frame.origin,
        frame.u_axis,
        frame.v_axis,
        tol,
    )
    .ok()
    .flatten();
    Ok(Region2 {
        edges,
        lo,
        hi,
        exact,
    })
}

/// Give each ellipse edge of `region` its paves from the earlier phases:
/// every pave of its pave blocks and their children, ends and extra paves
/// alike, as a point and its parameter.
fn attach_paves(region: &mut Region2, topo: &Topology, arena: &GfaArena) {
    for e in &mut region.edges {
        if e.along.is_none() {
            continue;
        }
        let mut pending: Vec<_> = arena
            .edge_pave_blocks
            .get(&e.eid)
            .cloned()
            .unwrap_or_default();
        while let Some(pb_id) = pending.pop() {
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            pending.extend(pb.children.iter().copied());
            for pave in [pb.start, pb.end]
                .into_iter()
                .chain(pb.extra_paves.iter().copied())
            {
                if let Ok(v) = topo.vertex(arena.resolve_vertex(pave.vertex)) {
                    e.paves.push((v.point(), pave.parameter));
                }
            }
        }
    }
}

/// The arc a NURBS boundary edge traces in the pair's frame, when it is a
/// circular arc (a partial revolve's rim). Each knot span of a rational
/// quadratic with positive weights is a conic, and the span is a circular
/// arc exactly when its two control legs are equal and its middle weight,
/// over the root of the product of its end weights, is the sine of half the
/// angle at its middle control point (the cosine of half its sweep). Every
/// span must lie on one circle. `p2_start` and `p2_end` are the edge's ends
/// in its wire's direction, `forward` that direction against the edge's own.
fn nurbs_arc(
    curve: &EdgeCurve,
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    (v_start, v_end): (Point3, Point3),
    (p2_start, p2_end): (Point2, Point2),
    forward: bool,
    frame: &PlaneFrame2D,
) -> Option<Arc2> {
    use std::f64::consts::{PI, TAU};
    if nurbs.degree() != 2 || !nurbs.weights().iter().all(|w| w.is_finite() && *w > 0.0) {
        return None;
    }
    let (t0, t1) = curve.domain_with_endpoints(v_start, v_end);
    if t1 <= t0 {
        return None;
    }
    let at = |t: f64| frame.project(curve.evaluate_with_endpoints(t, v_start, v_end));
    let span = t1 - t0;
    let closed = (v_start - v_end).length() < 1e-9;
    let mut breaks: Vec<f64> = std::iter::once(t0)
        .chain(nurbs.knots().iter().copied().filter(|&k| k > t0 && k < t1))
        .chain(std::iter::once(t1))
        .collect();
    breaks.dedup();
    let mut circle_2d: Option<(Point2, f64)> = None;
    for w in breaks.windows(2) {
        let poles = brepkit_math::nurbs::decompose::curve_bezier_between(nurbs, w[0], w[1]).ok()?;
        let [first, middle, last] = poles.as_slice() else {
            return None;
        };
        let mut points = [Point2::new(0.0, 0.0); 3];
        let mut weights = [0.0; 3];
        for (k, &[x, y, z, h]) in [first, middle, last].into_iter().enumerate() {
            if h.is_nan() || h <= 0.0 {
                return None;
            }
            let pole = Point3::new(x / h, y / h, z / h);
            if (pole - frame.origin).dot(frame.normal).abs()
                > 1e-9 * (1.0 + (pole - frame.origin).length())
            {
                return None;
            }
            points[k] = frame.project(pole);
            weights[k] = h;
        }
        let leg = |a: Point2, b: Point2| (b.x() - a.x()).hypot(b.y() - a.y());
        let (l0, l1) = (leg(points[1], points[0]), leg(points[1], points[2]));
        if l0 <= 0.0 || l1 <= 0.0 || (l0 - l1).abs() > 1e-9 * (l0 + l1) {
            return None;
        }
        let cos_apex = ((points[0].x() - points[1].x()) * (points[2].x() - points[1].x())
            + (points[0].y() - points[1].y()) * (points[2].y() - points[1].y()))
            / (l0 * l1);
        let sin_half_apex = (0.5 * (1.0 - cos_apex)).max(0.0).sqrt();
        let shape = weights[1] / (weights[0] * weights[2]).sqrt();
        if (shape - sin_half_apex).abs() > 1e-9 {
            return None;
        }
        let center = circumcenter(points[0], at(0.5 * (w[0] + w[1])), points[2])?;
        let radius = leg(center, points[0]);
        let eps = 1e-9 * (1.0 + radius);
        match circle_2d {
            None => circle_2d = Some((center, radius)),
            Some((c, r)) => {
                if leg(c, center) > eps || (r - radius).abs() > eps {
                    return None;
                }
            }
        }
    }
    let (center, radius) = circle_2d?;
    let eps = 1e-9 * (1.0 + radius);
    let off = |p: Point2| ((p.x() - center.x()).hypot(p.y() - center.y()) - radius).abs();
    if off(p2_start) > eps || off(p2_end) > eps {
        return None;
    }
    let a0 = angle_of(center, p2_start);
    let sweep = if closed {
        // The edge runs with its parameters; a quarter of the way along
        // shows which way round that is in the frame.
        let quarter =
            (angle_of(center, at(span.madd(0.25, t0))) - angle_of(center, at(t0))).rem_euclid(TAU);
        if (quarter < PI) == forward { TAU } else { -TAU }
    } else {
        let ccw = (angle_of(center, p2_end) - a0).rem_euclid(TAU);
        let to_mid = (angle_of(center, at(span.madd(0.5, t0))) - a0).rem_euclid(TAU);
        if to_mid <= ccw { ccw } else { ccw - TAU }
    };
    let circle =
        brepkit_math::curves::Circle3D::new(frame.unproject(center), frame.normal, radius).ok()?;
    Some(Arc2 {
        center,
        radius,
        a0,
        sweep,
        circle,
    })
}

/// The centre of the circle through three points, `None` when they are
/// collinear.
fn circumcenter(a: Point2, b: Point2, c: Point2) -> Option<Point2> {
    let (bx, by) = (b.x() - a.x(), b.y() - a.y());
    let (cx, cy) = (c.x() - a.x(), c.y() - a.y());
    let d = 2.0 * bx.madd(cy, -(by * cx));
    let scale = bx.hypot(by).max(cx.hypot(cy));
    if d.abs() <= 1e-12 * scale * scale {
        return None;
    }
    let (b2, c2) = (bx.madd(bx, by * by), cx.madd(cx, cy * cy));
    let ux = cy.madd(b2, -(by * c2)) / d;
    let uy = bx.madd(c2, -(cx * b2)) / d;
    Some(Point2::new(a.x() + ux, a.y() + uy))
}

/// Even-odd containment against the exact boundary (segments and arcs;
/// sampled curves by their samples): a ray from `pt` is cast along a
/// direction that passes no boundary vertex and grazes no arc, and its
/// crossings with every wire's edges are counted. No sampled polygon, so
/// no sagitta band along the arcs where a span's midpoint could read the
/// wrong side.
fn inside_region(pt: Point2, region: &Region2, tol: f64) -> bool {
    let reach = (region.hi.x() - region.lo.x()).hypot(region.hi.y() - region.lo.y())
        + (pt.x() - region.lo.x()).hypot(pt.y() - region.lo.y())
        + 1.0;
    // Directions with no rational relation to the frame axes, tried until
    // one is clear of every vertex and tangency.
    for k in 0..16 {
        let ang = 0.3_f64 + 0.71 * f64::from(k);
        let end = Point2::new(reach.madd(ang.cos(), pt.x()), reach.madd(ang.sin(), pt.y()));
        if let Some(n) = ray_crossings(pt, end, region, tol) {
            return n % 2 == 1;
        }
    }
    // Every direction hit a vertex or grazed an arc: fall back to the
    // winding number, whose arcs are walked in chords within `tol` of the
    // arc, so its only error band is the boundary band the callers already
    // exclude.
    winding_inside(pt, region, tol)
}

/// Even-odd containment by winding number; arcs are walked in chords whose
/// sagitta stays within `tol`.
fn winding_inside(pt: Point2, region: &Region2, tol: f64) -> bool {
    let angle = |p: Point2, q: Point2| -> f64 {
        let (ax, ay) = (p.x() - pt.x(), p.y() - pt.y());
        let (bx, by) = (q.x() - pt.x(), q.y() - pt.y());
        (ax * by - ay * bx).atan2(ax * bx + ay * by)
    };
    let mut total = 0.0;
    for e in &region.edges {
        match &e.shape {
            Shape2::Seg => total += angle(e.p2_start, e.p2_end),
            Shape2::Poly(poly) => {
                for w in poly.windows(2) {
                    total += angle(w[0], w[1]);
                }
            }
            Shape2::Arc(arc) => {
                let len = arc.sweep.abs();
                let sag = tol.min(arc.radius * 0.5).max(1e-12);
                let step = 2.0 * (1.0 - sag / arc.radius).acos();
                let n = ((len / step).ceil() as usize).clamp(8, 1 << 20);
                let mut prev = arc_point(arc, 0.0);
                for k in 1..=n {
                    let next = arc_point(arc, len * k as f64 / n as f64);
                    total += angle(prev, next);
                    prev = next;
                }
            }
        }
    }
    let turns = (total / std::f64::consts::TAU).round();
    turns.abs().rem_euclid(2.0) > 0.5
}

/// Crossings of the ray `a..b` with the region's boundary, or `None` when
/// the ray passes within `tol` of an edge end or grazes an arc (an
/// ambiguous count; the caller tries another direction).
fn ray_crossings(a: Point2, b: Point2, region: &Region2, tol: f64) -> Option<usize> {
    let mut n = 0usize;
    let ray_len = (b.x() - a.x()).hypot(b.y() - a.y());
    let (rx, ry) = ((b.x() - a.x()) / ray_len, (b.y() - a.y()) / ray_len);
    // Signed distance of a point from the ray's line.
    let off_line = |p: Point2| (p.x() - a.x()) * ry - (p.y() - a.y()) * rx;
    let clean_seg = |c: Point2, d: Point2| -> Option<usize> {
        let Some((t, u)) = seg_seg_params(a, b, c, d) else {
            // Parallel: a collinear edge along the ray is ambiguous, any
            // other parallel edge is simply not crossed.
            return (off_line(c).abs() > tol).then_some(0);
        };
        let seg_len = (d.x() - c.x()).hypot(d.y() - c.y());
        if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&u) {
            return Some(0);
        }
        if u * seg_len <= tol || (1.0 - u) * seg_len <= tol || t * ray_len <= tol {
            return None;
        }
        Some(1)
    };
    for e in &region.edges {
        match &e.shape {
            Shape2::Seg => n += clean_seg(e.p2_start, e.p2_end)?,
            Shape2::Poly(poly) => {
                for w in poly.windows(2) {
                    n += clean_seg(w[0], w[1])?;
                }
            }
            Shape2::Arc(arc) => {
                // A ray tangent to the circle is ambiguous; one starting
                // inside it legitimately leaves through a single hit.
                if (off_line(arc.center).abs() - arc.radius).abs() <= tol {
                    return None;
                }
                let hits = seg_circle(a, b, arc.center, arc.radius, tol);
                let ang_tol = tol / arc.radius;
                let len = arc.sweep.abs();
                for (t, p) in hits {
                    if t * ray_len <= tol {
                        return None;
                    }
                    if let Some(s) = arc_offset_of(arc, angle_of(arc.center, p), ang_tol) {
                        if s <= ang_tol || s >= len - ang_tol {
                            return None;
                        }
                        n += 1;
                    }
                }
            }
        }
    }
    Some(n)
}

/// Whether `pt` lies on the boundary edge within `tol`.
fn point_on_edge_2d(pt: Point2, e: &BoundaryEdge, tol: f64) -> bool {
    match &e.shape {
        Shape2::Seg => point_on_segment_2d(pt, e.p2_start, e.p2_end, tol),
        Shape2::Arc(arc) => {
            let d = (pt.x() - arc.center.x()).hypot(pt.y() - arc.center.y());
            (d - arc.radius).abs() <= tol
                && arc_offset_of(arc, angle_of(arc.center, pt), tol / arc.radius).is_some()
        }
        Shape2::Poly(poly) => poly
            .windows(2)
            .any(|w| point_on_segment_2d(pt, w[0], w[1], tol)),
    }
}

fn on_boundary(pt: Point2, edges: &[BoundaryEdge], tol: f64) -> bool {
    edges.iter().any(|e| point_on_edge_2d(pt, e, tol))
}

/// Points along a boundary edge that must all lie on a partner edge for the
/// two to share a carrier: the endpoints plus interior samples, so a segment
/// whose endpoints sit on an arc (a chord) or on a sampled curve is never
/// mistaken for that arc, and a circle-like sampled curve on an arc is.
fn carrier_probes(e: &BoundaryEdge) -> Vec<Point2> {
    match &e.shape {
        Shape2::Seg => vec![
            e.p2_start,
            Point2::new(
                0.5 * (e.p2_start.x() + e.p2_end.x()),
                0.5 * (e.p2_start.y() + e.p2_end.y()),
            ),
            e.p2_end,
        ],
        Shape2::Arc(arc) => {
            let len = arc.sweep.abs();
            (0..=8)
                .map(|k| arc_point(arc, len * k as f64 / 8.0))
                .collect()
        }
        Shape2::Poly(poly) => poly.clone(),
    }
}

/// Index of the partner edge that coincides with `e` along both their whole
/// lengths. A sub-span of a longer partner edge is not coincident: it is
/// clipped like any other edge, so the shared interval splits the longer
/// carrier at its ends instead of one common block covering the two
/// unequal pave blocks.
fn coincident_boundary_edge(e: &BoundaryEdge, target: &[BoundaryEdge], tol: f64) -> Option<usize> {
    let probes = carrier_probes(e);
    target.iter().position(|t| {
        probes.iter().all(|&p| point_on_edge_2d(p, t, tol))
            && carrier_probes(t)
                .iter()
                .all(|&p| point_on_edge_2d(p, e, tol))
    })
}

/// Parameters `t` along the segment `a..b` where it crosses the target edge.
/// A collinear partner segment contributes its own endpoints, so a partial
/// overlap splits the source at the overlap's ends (the shared interval is
/// then skipped as boundary and the rest clipped on its own).
fn seg_crossings(a: Point2, b: Point2, t: &BoundaryEdge, tol: f64, out: &mut Vec<f64>) {
    let mut seg = |c: Point2, d: Point2| {
        if let Some(x) = seg_seg_param(a, b, c, d) {
            out.push(x);
        } else if point_on_segment_2d(c, a, b, tol) || point_on_segment_2d(d, a, b, tol) {
            let (dx, dy) = (b.x() - a.x(), b.y() - a.y());
            let len2 = dx * dx + dy * dy;
            for q in [c, d] {
                let x = ((q.x() - a.x()) * dx + (q.y() - a.y()) * dy) / len2;
                if x > 0.0 && x < 1.0 && point_on_segment_2d(q, a, b, tol) {
                    out.push(x);
                }
            }
        }
    };
    match &t.shape {
        Shape2::Seg => seg(t.p2_start, t.p2_end),
        Shape2::Poly(poly) => {
            for w in poly.windows(2) {
                seg(w[0], w[1]);
            }
        }
        Shape2::Arc(arc) => {
            for (x, p) in seg_circle(a, b, arc.center, arc.radius, tol) {
                if arc_offset_of(arc, angle_of(arc.center, p), tol / arc.radius).is_some() {
                    out.push(x);
                }
            }
        }
    }
}

/// Along-arc offsets where the arc crosses the target edge.
fn arc_crossings(arc: &Arc2, t: &BoundaryEdge, tol: f64, out: &mut Vec<f64>) {
    let ang_tol = tol / arc.radius;
    let mut push_point = |p: Point2| {
        if let Some(s) = arc_offset_of(arc, angle_of(arc.center, p), ang_tol) {
            out.push(s);
        }
    };
    match &t.shape {
        Shape2::Seg => {
            for (_, p) in seg_circle(t.p2_start, t.p2_end, arc.center, arc.radius, tol) {
                push_point(p);
            }
        }
        Shape2::Poly(poly) => {
            for w in poly.windows(2) {
                for (_, p) in seg_circle(w[0], w[1], arc.center, arc.radius, tol) {
                    push_point(p);
                }
            }
        }
        Shape2::Arc(other) => {
            let concentric =
                (arc.center.x() - other.center.x()).hypot(arc.center.y() - other.center.y()) <= tol
                    && (arc.radius - other.radius).abs() <= tol;
            if concentric {
                // Coincident carriers: the overlap starts and ends at the
                // partner arc's endpoints.
                push_point(t.p2_start);
                push_point(t.p2_end);
            } else {
                for p in circle_circle(arc.center, arc.radius, other.center, other.radius, tol) {
                    if arc_offset_of(other, angle_of(other.center, p), tol / other.radius).is_some()
                    {
                        push_point(p);
                    }
                }
            }
        }
    }
}

/// Parameters `(t, u)` of the crossing of the lines through `a..b` and
/// `c..d` (`None` when parallel).
fn seg_seg_params(a: Point2, b: Point2, c: Point2, d: Point2) -> Option<(f64, f64)> {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let ex = d.x() - c.x();
    let ey = d.y() - c.y();
    let denom = dx * ey - dy * ex;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((c.x() - a.x()) * ey - (c.y() - a.y()) * ex) / denom;
    let u = ((c.x() - a.x()) * dy - (c.y() - a.y()) * dx) / denom;
    Some((t, u))
}

/// Parameter along `a..b` of its crossing with `c..d`, if the segments cross.
fn seg_seg_param(a: Point2, b: Point2, c: Point2, d: Point2) -> Option<f64> {
    let (t, u) = seg_seg_params(a, b, c, d)?;
    ((-1e-9..=1.0 + 1e-9).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u))
        .then(|| t.clamp(0.0, 1.0))
}

/// Crossings of the segment `a..b` with the full circle: `(t, point)`. A
/// segment within `tol` of grazing the circle counts once, at the foot of
/// the centre's perpendicular.
fn seg_circle(a: Point2, b: Point2, center: Point2, radius: f64, tol: f64) -> Vec<(f64, Point2)> {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let qa = dx * dx + dy * dy;
    if qa < 1e-30 {
        return Vec::new();
    }
    let seg_len = qa.sqrt();
    let fx = a.x() - center.x();
    let fy = a.y() - center.y();
    let t_foot = -(fx * dx + fy * dy) / qa;
    let foot = Point2::new(dx.madd(t_foot, a.x()), dy.madd(t_foot, a.y()));
    let h = (foot.x() - center.x()).hypot(foot.y() - center.y());
    if h > radius + tol {
        return Vec::new();
    }
    let roots = if h >= radius - tol {
        vec![t_foot]
    } else {
        let half = (radius * radius - h * h).sqrt() / seg_len;
        vec![t_foot - half, t_foot + half]
    };
    let eps_t = tol / seg_len;
    roots
        .into_iter()
        .filter(|t| (-eps_t..=1.0 + eps_t).contains(t))
        .map(|t| {
            let t = t.clamp(0.0, 1.0);
            (t, Point2::new(dx.madd(t, a.x()), dy.madd(t, a.y())))
        })
        .collect()
}

/// Crossings of two distinct circles; circles within `tol` of tangency
/// touch at one point.
fn circle_circle(c1: Point2, r1: f64, c2: Point2, r2: f64, tol: f64) -> Vec<Point2> {
    let dx = c2.x() - c1.x();
    let dy = c2.y() - c1.y();
    let d = dx.hypot(dy);
    if d <= tol || d > r1 + r2 + tol || d < (r1 - r2).abs() - tol {
        return Vec::new();
    }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let (ex, ey) = (dx / d, dy / d);
    let mx = ex.madd(a, c1.x());
    let my = ey.madd(a, c1.y());
    if d >= r1 + r2 - tol || d <= (r1 - r2).abs() + tol {
        return vec![Point2::new(mx, my)];
    }
    let h = r1.madd(r1, -(a * a)).max(0.0).sqrt();
    vec![
        Point2::new(mx - h * ey, my + h * ex),
        Point2::new(mx + h * ey, my - h * ex),
    ]
}

/// Clip a boundary edge of one face to the other face's region and collect
/// the pieces inside. Each maximal in-region span between consecutive
/// boundary crossings becomes one piece; a span running along the partner's
/// boundary is not a section.
fn clip_to_region(
    e: &BoundaryEdge,
    (target_face, target): (FaceId, &Region2),
    frame: &PlaneFrame2D,
    tol: f64,
    out: &mut Vec<SectionPiece>,
) -> Result<(), AlgoError> {
    match &e.shape {
        Shape2::Arc(arc) => {
            let len = arc.sweep.abs();
            let mut ss = vec![0.0, len];
            for t in &target.edges {
                arc_crossings(arc, t, tol, &mut ss);
            }
            ss.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            ss.dedup_by(|x, y| (*x - *y).abs() * arc.radius < tol);
            let ang_tol = tol / arc.radius;
            for w in ss.windows(2) {
                let (sa, sb) = (w[0], w[1]);
                if (sb - sa) * arc.radius < tol {
                    continue;
                }
                let mid2 = arc_point(arc, 0.5 * (sa + sb));
                if !target.holds(mid2, tol) {
                    continue;
                }
                let on_circle = |p2: Point2| -> Point3 {
                    let t = arc.circle.project(frame.unproject(p2));
                    arc.circle.evaluate(t)
                };
                let start = if sa <= ang_tol {
                    e.p3_start
                } else {
                    on_circle(arc_point(arc, sa))
                };
                let end = if (len - sb).abs() <= ang_tol {
                    e.p3_end
                } else {
                    on_circle(arc_point(arc, sb))
                };
                out.push(SectionPiece::Arc {
                    target: target_face,
                    circle: arc.circle.clone(),
                    start,
                    end,
                    mid: on_circle(mid2),
                });
            }
        }
        Shape2::Seg => {
            let (a, b) = (e.p2_start, e.p2_end);
            let d = Point2::new(b.x() - a.x(), b.y() - a.y());
            let seg_len = d.x().hypot(d.y());
            if seg_len < tol {
                return Ok(());
            }
            let mut ts = vec![0.0, 1.0];
            for t in &target.edges {
                seg_crossings(a, b, t, tol, &mut ts);
            }
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            ts.dedup_by(|x, y| (*x - *y).abs() * seg_len < tol);
            for w in ts.windows(2) {
                let (ta, tb) = (w[0], w[1]);
                if (tb - ta) * seg_len < tol {
                    continue;
                }
                let tm = 0.5 * (ta + tb);
                let mid = Point2::new(d.x().madd(tm, a.x()), d.y().madd(tm, a.y()));
                if !target.holds(mid, tol) {
                    continue;
                }
                out.push(SectionPiece::Line {
                    target: target_face,
                    start: lerp3(e.p3_start, e.p3_end, ta),
                    end: lerp3(e.p3_start, e.p3_end, tb),
                });
            }
        }
        Shape2::Poly(poly) => {
            let n = poly.len().saturating_sub(1);
            if n == 0 {
                return Ok(());
            }
            let nf = n as f64;
            // Where the samples cross `edges`, as the polyline's parameter:
            // sub-segment k spans [k, k + 1].
            let sampled = |edges: &[&BoundaryEdge]| -> Vec<f64> {
                let mut ss = Vec::new();
                for (k, w) in poly.windows(2).enumerate() {
                    let mut local = Vec::new();
                    for t in edges {
                        seg_crossings(w[0], w[1], t, tol, &mut local);
                    }
                    ss.extend(local.into_iter().map(|x| x + k as f64));
                }
                ss
            };
            let Some(along) = &e.along else {
                // A B-spline keeps its chords: the regular FF phase gives its
                // true pieces, and this only partitions the partner alike.
                let mut ss: Vec<f64> = vec![0.0, nf];
                ss.extend(sampled(&target.edges.iter().collect::<Vec<_>>()));
                ss.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                let at = |s: f64| -> Point2 {
                    let k = (s.floor() as usize).min(n - 1);
                    let f = s - k as f64;
                    let (c, d) = (poly[k], poly[k + 1]);
                    Point2::new(
                        (d.x() - c.x()).madd(f, c.x()),
                        (d.y() - c.y()).madd(f, c.y()),
                    )
                };
                let dist = |s0: f64, s1: f64| {
                    let (p, q) = (at(s0), at(s1));
                    (q.x() - p.x()).hypot(q.y() - p.y())
                };
                ss.dedup_by(|x, y| dist(*x, *y) < tol);
                for w in ss.windows(2) {
                    let (sa, sb) = (w[0], w[1]);
                    if dist(sa, sb) < tol {
                        continue;
                    }
                    let mid = at(0.5 * (sa + sb));
                    if !target.holds(mid, tol) {
                        continue;
                    }
                    let start = if sa <= 0.0 {
                        e.p3_start
                    } else {
                        frame.unproject(at(sa))
                    };
                    let end = if sb >= nf {
                        e.p3_end
                    } else {
                        frame.unproject(at(sb))
                    };
                    out.push(SectionPiece::Line {
                        target: target_face,
                        start,
                        end,
                    });
                }
                return Ok(());
            };
            // An ellipse is cut where it meets the partner's boundary, each
            // place proved: against a line, a circle or an ellipse of the
            // boundary by `section_meetings`, read in pieces of at most a
            // half turn, whose ends are checked on the other curve, and at
            // its own paves, where it crosses or touches the other solid's
            // edges. A B-spline of the boundary is met where the samples
            // cross it. Each run between the cuts follows the ellipse: its
            // chord stands off the ellipse by its sag, and the partner's
            // split along it would meet no edge of the face the ellipse
            // bounds. A meeting no bound settles refuses the pair.
            let mut cuts: Vec<(f64, Point3, u8)> = vec![
                (along.at(0.0, nf), e.p3_start, 2),
                (along.at(nf, nf), e.p3_end, 2),
            ];
            let own = half_turns(&along.curve, along.ends);
            // A closed edge's two ends are one point, which the merge below
            // takes for one cut: its half turns' own ends cut it too, so a
            // loop wholly inside the partner still comes as pieces round its
            // turn.
            if (along.ends.0 - along.ends.1).length() <= tol {
                for &(_, p, _) in own.iter().skip(1) {
                    if let Some(at) = along.through(p, tol) {
                        cuts.push((at, p, 0));
                    }
                }
            }
            let unproved: Vec<&BoundaryEdge> = target
                .edges
                .iter()
                .filter(|t| matches!(t.own.0, EdgeCurve::NurbsCurve(_)))
                .collect();
            for s in sampled(&unproved) {
                let t = along.at(s, nf);
                cuts.push((t, along.point(t), 0));
            }
            for t in &target.edges {
                if matches!(t.own.0, EdgeCurve::NurbsCurve(_)) {
                    continue;
                }
                let theirs = half_turns(&t.own.0, (t.own.1, t.own.2));
                for &(_, p, _) in &theirs {
                    if let Some(at) = along.through(p, tol) {
                        cuts.push((at, p, 1));
                    }
                }
                for &(_, p, _) in &own {
                    if on_edge(p, t, frame, tol)
                        && let Some(at) = along.through(p, tol)
                    {
                        cuts.push((at, p, 1));
                    }
                }
                for mine in &own {
                    for other in &theirs {
                        let met = crate::builder::face_splitter::section_meetings(
                            (&mine.0, mine.1, mine.2),
                            (&other.0, other.1, other.2),
                            tol,
                        )
                        .ok_or_else(|| {
                            AlgoError::IntersectionFailed(
                                "an ellipse edge's meeting with a coplanar face's boundary \
                                 is not settled"
                                    .into(),
                            )
                        })?;
                        for p in met {
                            if let Some(at) = along.through(p, tol) {
                                cuts.push((at, p, 1));
                            }
                        }
                    }
                }
            }
            // The edge's own paves cut it too, at their points: where it
            // touches the partner's boundary no crossing shows, and a section
            // running on through the touch would leave the partner's face
            // pinched there, round no vertex.
            cuts.extend(
                e.paves
                    .iter()
                    .filter_map(|&(p, t)| along.param_of(p, t).map(|t| (t, p, 1))),
            );
            cuts.sort_by(|a, b| {
                if along.forward {
                    a.0.total_cmp(&b.0)
                } else {
                    b.0.total_cmp(&a.0)
                }
            });
            let mut merged: Vec<(f64, Point3, u8)> = Vec::with_capacity(cuts.len());
            for cut in cuts {
                match merged.last_mut() {
                    Some(last) if (last.1 - cut.1).length() <= tol => {
                        if cut.2 > last.2 {
                            *last = cut;
                        }
                    }
                    _ => merged.push(cut),
                }
            }
            for w in merged.windows(2) {
                let ((ta, start, _), (tb, end, _)) = (w[0], w[1]);
                if (end - start).length() < tol {
                    continue;
                }
                let mid3 = along.point(f64::midpoint(ta, tb));
                let mid = frame.project(mid3);
                if !target.holds(mid, tol) {
                    continue;
                }
                let (span, start, end) = if ta <= tb {
                    ((ta, tb), start, end)
                } else {
                    ((tb, ta), end, start)
                };
                out.push(SectionPiece::Curve {
                    target: target_face,
                    along: along.clone(),
                    span,
                    start,
                    end,
                    mid: mid3,
                });
            }
        }
    }
    Ok(())
}

/// `curve` from `start` to `end` in pieces of at most a half turn, each as
/// the curve and its two ends: a line whole, a circle or an ellipse cut at
/// equal steps of its parameter. Nothing for a B-spline.
fn half_turns(
    curve: &EdgeCurve,
    (start, end): (Point3, Point3),
) -> Vec<(EdgeCurve, Point3, Point3)> {
    match curve {
        EdgeCurve::Line => vec![(EdgeCurve::Line, start, end)],
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
            // A whole conic runs a turn from its own seam, wherever that
            // lies on its parameter: its domain starts at the seam's.
            let (t0, t1) = if (start - end).length() < 1e-9 {
                let from = match curve {
                    EdgeCurve::Circle(c) => c.project(start),
                    EdgeCurve::Ellipse(e) => e.project(start),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => 0.0,
                };
                (from, from + std::f64::consts::TAU)
            } else {
                curve.domain_with_endpoints(start, end)
            };
            let pieces = ((t1 - t0).abs() / (0.99 * std::f64::consts::PI))
                .ceil()
                .max(1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let count = pieces as usize;
            let at = |k: usize| {
                if k == 0 {
                    start
                } else if k == count {
                    end
                } else {
                    curve.evaluate_with_endpoints((t1 - t0).madd(k as f64 / pieces, t0), start, end)
                }
            };
            (0..count)
                .map(|k| (curve.clone(), at(k), at(k + 1)))
                .collect()
        }
        EdgeCurve::NurbsCurve(_) => Vec::new(),
    }
}

/// Whether `p` lies on the boundary edge `t`: on its ellipse, read by the
/// ellipse's nearest point, or on its 2D shape.
fn on_edge(p: Point3, t: &BoundaryEdge, frame: &PlaneFrame2D, tol: f64) -> bool {
    match &t.along {
        Some(along) => along.through(p, tol).is_some(),
        None => point_on_edge_2d(frame.project(p), t, tol),
    }
}

/// The piece's span in the circle's own parameter, oriented so its midpoint
/// lies inside `[t_s, t_e]` (`t_e > t_s`), with the endpoints in that order.
fn oriented_arc_span(
    circle: &brepkit_math::curves::Circle3D,
    start: Point3,
    end: Point3,
    mid: Point3,
) -> (f64, f64, Point3, Point3) {
    use std::f64::consts::TAU;
    let span = |from: Point3, to: Point3| -> (f64, f64) {
        let t_s = circle.project(from);
        let delta = (circle.project(to) - t_s).rem_euclid(TAU);
        let delta = if delta < 1e-12 { TAU } else { delta };
        (t_s, t_s + delta)
    };
    let (t_s, t_e) = span(start, end);
    if (circle.project(mid) - t_s).rem_euclid(TAU) <= t_e - t_s {
        (t_s, t_e, start, end)
    } else {
        let (t_s, t_e) = span(end, start);
        (t_s, t_e, end, start)
    }
}

/// The parts of `[t_s, t_e]` on `circle` that no existing Circle section
/// attached to `target` already spans. Sections are compared on the same
/// circle (centre, radius, carrier plane, either normal sign), their
/// endpoints and midpoint re-projected onto this circle's parameter so an
/// oppositely oriented carrier covers the same arc, and spans are taken
/// modulo the period.
fn uncovered_arc_spans(
    arena: &GfaArena,
    target: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    t_s: f64,
    t_e: f64,
    ang_tol: f64,
) -> Vec<(f64, f64)> {
    use std::f64::consts::TAU;
    let lin_tol = ang_tol * circle.radius();
    let mut spans = vec![(t_s, t_e)];
    for c in &arena.curves {
        let EdgeCurve::Circle(existing) = &c.curve else {
            continue;
        };
        if (c.face_a != target && c.face_b != target)
            || (existing.center() - circle.center()).length() > lin_tol
            || (existing.radius() - circle.radius()).abs() > lin_tol
            || existing.normal().cross(circle.normal()).length() > 1e-9
        {
            continue;
        }
        let (x0, x1) = c.t_range;
        let e0 = circle.project(existing.evaluate(x0));
        let e1 = circle.project(existing.evaluate(x1));
        let em = circle.project(existing.evaluate(0.5 * (x0 + x1)));
        let (c0, width) = {
            let w = (e1 - e0).rem_euclid(TAU);
            let w = if w < 1e-12 { TAU } else { w };
            if (em - e0).rem_euclid(TAU) <= w {
                (e0, w)
            } else {
                let w = (e0 - e1).rem_euclid(TAU);
                (e1, if w < 1e-12 { TAU } else { w })
            }
        };
        for k in -1..=1 {
            let lo = (c0 - t_s).rem_euclid(TAU) + t_s + f64::from(k) * TAU - ang_tol;
            let hi = lo + width + 2.0 * ang_tol;
            let mut next = Vec::with_capacity(spans.len() + 1);
            for &(a, b) in &spans {
                if hi <= a || lo >= b {
                    next.push((a, b));
                    continue;
                }
                if lo > a {
                    next.push((a, lo));
                }
                if hi < b {
                    next.push((hi, b));
                }
            }
            spans = next;
        }
    }
    spans.retain(|&(a, b)| b - a > ang_tol);
    spans
}

/// Emit the arc `[t0, t1]` of `circle` as one or more section edges, each
/// spanning strictly less than a half turn: the assembler's endpoint-keyed
/// edge merge would collapse two arcs between the same two vertices (a rim
/// split once by a single crossing) into one.
#[allow(clippy::too_many_arguments)]
fn create_arc_section_edges(
    topo: &mut Topology,
    arena: &mut GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    faces: [FaceId; 2],
    circle: &brepkit_math::curves::Circle3D,
    (p0, p1): (Point3, Point3),
    (t0, t1): (f64, f64),
    tol: Tolerance,
) {
    let span = t1 - t0;
    if span * circle.radius() < tol.linear {
        return;
    }
    let n_sub = (span / (std::f64::consts::PI * 0.999)).ceil().max(1.0) as usize;
    let mut prev = (p0, t0);
    for k in 1..=n_sub {
        let t = if k == n_sub {
            t1
        } else {
            span.madd(k as f64 / n_sub as f64, t0)
        };
        let p = if k == n_sub { p1 } else { circle.evaluate(t) };
        create_arc_section_edge(
            topo,
            arena,
            minted,
            faces,
            circle,
            (prev.0, p),
            (prev.1, t),
            tol,
        );
        prev = (p, t);
    }
}

/// Register one arc section (edge, pave block, curve, interference) the way
/// the regular FF phase registers its arcs.
#[allow(clippy::too_many_arguments)]
fn create_arc_section_edge(
    topo: &mut Topology,
    arena: &mut GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    faces: [FaceId; 2],
    circle: &brepkit_math::curves::Circle3D,
    (p_s, p_e): (Point3, Point3),
    (t_s, t_e): (f64, f64),
    tol: Tolerance,
) {
    let [face_a, face_b] = faces;
    let start_vid = find_or_create_vertex(topo, arena, minted, faces, p_s, tol);
    let end_vid = find_or_create_vertex(topo, arena, minted, faces, p_e, tol);
    let edge = Edge::new(start_vid, end_vid, EdgeCurve::Circle(circle.clone()));
    let edge_id = topo.add_edge(edge);

    let pb = PaveBlock::new(edge_id, Pave::new(start_vid, t_s), Pave::new(end_vid, t_e));
    let pb_id = arena.pave_blocks.alloc(pb);
    arena
        .edge_pave_blocks
        .entry(edge_id)
        .or_default()
        .push(pb_id);

    let n: usize = 16;
    let bbox = Aabb3::from_points(
        (0..=n).map(|k| circle.evaluate((t_e - t_s).madd(k as f64 / n as f64, t_s))),
    );
    let curve_index = arena.curves.len();
    arena.curves.push(IntersectionCurveDS {
        curve: EdgeCurve::Circle(circle.clone()),
        face_a,
        face_b,
        bbox,
        pave_blocks: vec![pb_id],
        t_range: (t_s, t_e),
    });
    arena.interference.ff.push(Interference::FF {
        f1: face_a,
        f2: face_b,
        curve_index,
    });

    log::debug!(
        "FF-coplanar: faces {face_a:?} and {face_b:?} arc section t=[{t_s:.4},{t_e:.4}] \
         (curve_index={curve_index}, edge={edge_id:?}, pb={pb_id:?})",
    );
}

/// Whether `target` already has a section that is the piece of an ellipse
/// boundary edge along `piece` from `p_start` to `p_end`, through `mid`:
/// with the piece's ends, and proved one curve with it along its whole
/// length (see [`crate::builder::face_splitter::curves_one_curve`]). A
/// circle or an ellipse through the piece's ends and middle can still part
/// from it between them, and the thin regions there would be lost. A curve
/// attached only to the other face of the pair does not count.
fn has_existing_section_along(
    arena: &GfaArena,
    topo: &Topology,
    target: FaceId,
    (piece_curve, p_start, p_end): (&EdgeCurve, Point3, Point3),
    mid: Point3,
    tol: Tolerance,
) -> bool {
    let near = |p: Point3, q: Point3| (p - q).length() <= tol.linear;
    let piece = Aabb3::from_points([p_start, p_end, mid]);
    arena.curves.iter().any(|curve| {
        if (curve.face_a != target && curve.face_b != target)
            || !curve.bbox.expanded(tol.linear).intersects(piece)
        {
            return false;
        }
        let Some(((s, _), (e, _))) = section_ends(topo, arena, curve) else {
            return false;
        };
        if !((near(s, p_start) && near(e, p_end)) || (near(s, p_end) && near(e, p_start))) {
            return false;
        }
        crate::builder::face_splitter::curves_one_curve(
            (piece_curve, p_start, p_end),
            (&curve.curve, s, e),
            tol.linear,
        )
    })
}

/// Register one section along an ellipse boundary edge's curve, from `span.0`
/// (at `p_s`) to `span.1` (at `p_e`), the way the regular FF phase registers
/// its sections.
#[allow(clippy::too_many_arguments)]
fn create_curve_section_edge(
    topo: &mut Topology,
    arena: &mut GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    faces: [FaceId; 2],
    along: &CurveSpan,
    (p_s, p_e): (Point3, Point3),
    (t_s, t_e): (f64, f64),
    tol: Tolerance,
) {
    if (p_e - p_s).length() < tol.linear {
        return;
    }
    let [face_a, face_b] = faces;
    let start_vid = find_or_create_vertex(topo, arena, minted, faces, p_s, tol);
    let end_vid = find_or_create_vertex(topo, arena, minted, faces, p_e, tol);
    let edge_id = topo.add_edge(Edge::new(start_vid, end_vid, along.curve.clone()));
    let pb = PaveBlock::new(edge_id, Pave::new(start_vid, t_s), Pave::new(end_vid, t_e));
    let pb_id = arena.pave_blocks.alloc(pb);
    arena
        .edge_pave_blocks
        .entry(edge_id)
        .or_default()
        .push(pb_id);
    let n: usize = 16;
    let bbox = Aabb3::from_points(
        (0..=n).map(|k| along.point((t_e - t_s).madd(k as f64 / n as f64, t_s))),
    );
    let curve_index = arena.curves.len();
    arena.curves.push(IntersectionCurveDS {
        curve: along.curve.clone(),
        face_a,
        face_b,
        bbox,
        pave_blocks: vec![pb_id],
        t_range: (t_s, t_e),
    });
    arena.interference.ff.push(Interference::FF {
        f1: face_a,
        f2: face_b,
        curve_index,
    });
    log::debug!(
        "FF-coplanar: faces {face_a:?} and {face_b:?} curve section t=[{t_s:.4},{t_e:.4}] \
         (curve_index={curve_index}, edge={edge_id:?}, pb={pb_id:?})",
    );
}

/// Create a section edge and register it in the GFA arena.
/// Create a CommonBlock linking leaf PaveBlocks of two coincident boundary edges.
///
/// For flush-face (touching) booleans, A's boundary edge and B's boundary edge
/// overlap at the shared face boundary. Linking their PaveBlocks via a
/// CommonBlock ensures they share the same split edge, enabling
/// `merge_duplicate_edges` to recognize them as the same geometric edge.
fn create_coplanar_common_block(
    arena: &mut GfaArena,
    a_edge: brepkit_topology::edge::EdgeId,
    b_edge: brepkit_topology::edge::EdgeId,
    tol: f64,
) {
    let get_leaves = |edge: brepkit_topology::edge::EdgeId| -> Vec<PaveBlockId> {
        arena
            .edge_pave_blocks
            .get(&edge)
            .map(|pbs| {
                pbs.iter()
                    .copied()
                    .filter(|&pb_id| {
                        arena
                            .pave_blocks
                            .get(pb_id)
                            .is_some_and(|pb| pb.children.is_empty())
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let a_leaves = get_leaves(a_edge);
    let b_leaves = get_leaves(b_edge);

    // For now, handle the simple case: both edges have exactly 1 leaf PB.
    // More complex cases (split edges with multiple children) need position
    // matching to pair the correct leaf PBs.
    if a_leaves.len() == 1 && b_leaves.len() == 1 {
        let a_pb = a_leaves[0];
        let b_pb = b_leaves[0];

        // Skip if both PBs are already in the same CB, or either
        // is in a different CB (merging CBs deferred to Phase 5).
        let a_cb = arena.pb_to_cb.get(&a_pb).copied();
        let b_cb = arena.pb_to_cb.get(&b_pb).copied();
        if (a_cb.is_some() && a_cb == b_cb) || a_cb.is_some() || b_cb.is_some() {
            return;
        }

        arena.create_common_block(vec![a_pb, b_pb], tol);

        log::debug!("coplanar CommonBlock: edge {a_edge:?} + {b_edge:?} (PBs {a_pb:?} + {b_pb:?})");
    }
}

#[allow(clippy::unnecessary_wraps)]
fn create_section_edge(
    topo: &mut Topology,
    arena: &mut GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    faces: [FaceId; 2],
    p3d_start: Point3,
    p3d_end: Point3,
    tol: Tolerance,
) -> Result<(), AlgoError> {
    let [face_a, face_b] = faces;
    let edge_length = (p3d_end - p3d_start).length();
    if edge_length < tol.linear {
        // Degenerate edge, skip
        return Ok(());
    }

    let start_vid = find_or_create_vertex(topo, arena, minted, faces, p3d_start, tol);
    let end_vid = find_or_create_vertex(topo, arena, minted, faces, p3d_end, tol);

    let edge = Edge::new(start_vid, end_vid, EdgeCurve::Line);
    let edge_id = topo.add_edge(edge);

    // EdgeCurve::Line uses normalized parameter space [0, 1].
    let start_pave = Pave::new(start_vid, 0.0);
    let end_pave = Pave::new(end_vid, 1.0);
    let pb = PaveBlock::new(edge_id, start_pave, end_pave);
    let pb_id = arena.pave_blocks.alloc(pb);

    // Register in edge_pave_blocks so ForceInterfEE can detect overlaps
    // between this section PB and boundary-edge PBs with the same
    // endpoints. This creates CommonBlocks → shared split edges →
    // manifold shell connectivity between coplanar sub-faces.
    arena
        .edge_pave_blocks
        .entry(edge_id)
        .or_default()
        .push(pb_id);

    let bbox = Aabb3 {
        min: Point3::new(
            p3d_start.x().min(p3d_end.x()),
            p3d_start.y().min(p3d_end.y()),
            p3d_start.z().min(p3d_end.z()),
        ),
        max: Point3::new(
            p3d_start.x().max(p3d_end.x()),
            p3d_start.y().max(p3d_end.y()),
            p3d_start.z().max(p3d_end.z()),
        ),
    };

    let curve_index = arena.curves.len();
    arena.curves.push(IntersectionCurveDS {
        curve: EdgeCurve::Line,
        face_a,
        face_b,
        bbox,
        pave_blocks: vec![pb_id],
        t_range: (0.0, 1.0),
    });

    arena.interference.ff.push(Interference::FF {
        f1: face_a,
        f2: face_b,
        curve_index,
    });

    log::debug!(
        "FF-coplanar: faces {face_a:?} and {face_b:?} section edge \
         (curve_index={curve_index}, edge={edge_id:?}, pb={pb_id:?})",
    );

    Ok(())
}

/// Find an existing vertex near the point (a pave vertex, one this phase
/// minted, or a vertex of either face: the pave index is built once after
/// phase VV and never sees later vertices), or create a new one.
fn find_or_create_vertex(
    topo: &mut Topology,
    arena: &GfaArena,
    minted: &mut Vec<(Point3, VertexId)>,
    faces: [FaceId; 2],
    point: Point3,
    tol: Tolerance,
) -> VertexId {
    if let Some(vid) = find_nearby_pave_vertex(topo, arena, point, tol) {
        return vid;
    }
    if let Some(&(_, vid)) = minted
        .iter()
        .find(|(p, _)| (*p - point).length() < tol.linear)
    {
        return vid;
    }
    for face in faces {
        if let Some(vid) = find_nearby_face_vertex(topo, face, point, tol) {
            return vid;
        }
    }
    let vid = topo.add_vertex(Vertex::new(point, tol.linear));
    minted.push((point, vid));
    vid
}

// ---------------------------------------------------------------------------
// 2D geometry helpers
// ---------------------------------------------------------------------------

/// Minimal plane frame for 3D ↔ 2D projection (same logic as
/// `builder::plane_frame::PlaneFrame` but kept local to avoid coupling).
struct PlaneFrame2D {
    origin: Point3,
    u_axis: Vec3,
    v_axis: Vec3,
    normal: Vec3,
}

impl PlaneFrame2D {
    /// The frame of the plane `normal . p = d` that both faces of a coplanar
    /// pair share: the normal's sign is fixed by its first non-zero
    /// component and the origin is the plane's foot of the world origin, so
    /// faces with anti-parallel normals project into one chart.
    fn canonical(normal: Vec3, d: f64) -> Self {
        let flip = [normal.x(), normal.y(), normal.z()]
            .into_iter()
            .find(|c| c.abs() > 1e-9)
            .is_some_and(|c| c < 0.0);
        let (n, d) = if flip {
            (normal * -1.0, -d)
        } else {
            (normal, d)
        };
        Self::new(n, Point3::new(n.x() * d, n.y() * d, n.z() * d))
    }

    fn new(normal: Vec3, origin: Point3) -> Self {
        let seed = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u_raw = normal.cross(seed);
        let u_axis = u_raw.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let v_axis = normal.cross(u_axis);
        Self {
            origin,
            u_axis,
            v_axis,
            normal: u_axis.cross(v_axis),
        }
    }

    fn project(&self, p: Point3) -> Point2 {
        let d = p - self.origin;
        Point2::new(d.dot(self.u_axis), d.dot(self.v_axis))
    }

    fn unproject(&self, p: Point2) -> Point3 {
        self.origin + self.u_axis * p.x() + self.v_axis * p.y()
    }
}

/// Check if a 2D point lies on a line segment within tolerance.
fn point_on_segment_2d(pt: Point2, a: Point2, b: Point2, tol: f64) -> bool {
    let ab = Point2::new(b.x() - a.x(), b.y() - a.y());
    let ap = Point2::new(pt.x() - a.x(), pt.y() - a.y());

    let ab_len_sq = ab.x() * ab.x() + ab.y() * ab.y();
    if ab_len_sq < tol * tol {
        // Degenerate segment — just check distance to endpoint
        return ap.x() * ap.x() + ap.y() * ap.y() <= tol * tol;
    }

    let t = (ap.x() * ab.x() + ap.y() * ab.y()) / ab_len_sq;
    if t < -tol || t > 1.0 + tol {
        return false;
    }

    let closest_x = a.x() + t.clamp(0.0, 1.0) * ab.x();
    let closest_y = a.y() + t.clamp(0.0, 1.0) * ab.y();
    let dx = pt.x() - closest_x;
    let dy = pt.y() - closest_y;

    dx * dx + dy * dy <= tol * tol
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn point_on_segment() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        assert!(point_on_segment_2d(Point2::new(0.5, 0.0), a, b, 1e-7));
        assert!(!point_on_segment_2d(Point2::new(0.5, 1.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(0.0, 0.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(1.0, 0.0), a, b, 1e-7));
    }

    use brepkit_math::curves::Circle3D;
    use brepkit_topology::face::Face;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A face on the plane `z = 0`, facing down, bounded by `outer` and by
    /// the holes `inner`, each a loop of edges.
    fn face_down(
        topo: &mut Topology,
        outer: Vec<OrientedEdge>,
        inner: Vec<Vec<OrientedEdge>>,
    ) -> FaceId {
        let outer = topo.add_wire(Wire::new(outer, true).unwrap());
        let inner = inner
            .into_iter()
            .map(|edges| topo.add_wire(Wire::new(edges, true).unwrap()))
            .collect();
        topo.add_face(Face::new(
            outer,
            inner,
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ))
    }

    /// A circle on `z = 0` about the origin, as one closed edge from its
    /// point at 2.4 rad.
    fn rim(topo: &mut Topology, radius: f64) -> OrientedEdge {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(2.4), 1e-7));
        OrientedEdge::new(
            topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle))),
            false,
        )
    }

    /// The ends of each section the arena holds, by its pave blocks.
    fn all_section_ends(topo: &Topology, arena: &GfaArena) -> Vec<(Point3, Point3)> {
        arena
            .curves
            .iter()
            .flat_map(|c| &c.pave_blocks)
            .filter_map(|&pb| arena.pave_blocks.get(pb))
            .map(|pb| {
                let at = |p: Pave| topo.vertex(arena.resolve_vertex(p.vertex)).unwrap().point();
                (at(pb.start), at(pb.end))
            })
            .collect()
    }

    /// The end of a ring of radii 2 and 3 about the z axis, and the bottom
    /// of a block from the origin to (5, 5): the block's edge along x
    /// crosses the ring's end from (2, 0, 0) to (3, 0, 0). The section of
    /// the ring's end on the block's side `y = 0` runs from (5, 0, 0) to
    /// (0, 0, 0), with the same middle. That piece is not on the section.
    #[test]
    fn a_piece_with_the_middle_of_a_longer_section_is_kept() {
        let tol = Tolerance::default();
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        let (outer, inner) = (rim(&mut topo, 3.0), rim(&mut topo, 2.0));
        let ring_end = face_down(&mut topo, vec![outer], vec![vec![inner]]);
        let corners = [(0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 0.0)]
            .map(|(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)));
        let sides = (0..4)
            .map(|k| {
                let edge = Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let block_bottom = face_down(&mut topo, sides, vec![]);
        let block_side_rim = rim(&mut topo, 9.0);
        let block_side = face_down(&mut topo, vec![block_side_rim], vec![]);

        let (from, to) = (Point3::new(5.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        create_section_edge(
            &mut topo,
            &mut arena,
            &mut Vec::new(),
            [ring_end, block_side],
            from,
            to,
            tol,
        )
        .unwrap();

        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let region_ring = face_region_2d(&topo, ring_end, &frame, 1e-7).unwrap();
        let region_block = face_region_2d(&topo, block_bottom, &frame, 1e-7).unwrap();
        process_coplanar_pair(
            &mut topo,
            &mut arena,
            &mut Vec::new(),
            (ring_end, &region_ring),
            (block_bottom, &region_block),
            &frame,
            tol,
        )
        .unwrap();

        let piece = (Point3::new(2.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
        let ends = all_section_ends(&topo, &arena);
        let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
        assert!(
            ends.iter()
                .any(|&(s, e)| (near(s, piece.0) && near(e, piece.1))
                    || (near(s, piece.1) && near(e, piece.0))),
            "no section from (2, 0, 0) to (3, 0, 0): {ends:?}"
        );
    }

    /// A piece with the two ends of a line section on its face is that
    /// section, either way. A piece in the middle of it is not, nor is a
    /// piece on a face the section does not split, nor the chord of an arc
    /// section.
    #[test]
    fn a_piece_on_a_section_with_its_ends_is_present() {
        let tol = Tolerance::default();
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        let target_rim = rim(&mut topo, 9.0);
        let target = face_down(&mut topo, vec![target_rim], vec![]);
        let other_rim = rim(&mut topo, 8.0);
        let other = face_down(&mut topo, vec![other_rim], vec![]);
        let (from, to) = (Point3::new(5.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        create_section_edge(
            &mut topo,
            &mut arena,
            &mut Vec::new(),
            [target, other],
            from,
            to,
            tol,
        )
        .unwrap();
        assert!(has_existing_section_at(
            &arena, &topo, target, to, from, tol
        ));
        assert!(has_existing_section_at(
            &arena, &topo, target, from, to, tol
        ));
        let middle = (Point3::new(2.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
        assert!(!has_existing_section_at(
            &arena, &topo, target, middle.0, middle.1, tol
        ));
        let elsewhere_rim = rim(&mut topo, 7.0);
        let elsewhere = face_down(&mut topo, vec![elsewhere_rim], vec![]);
        assert!(!has_existing_section_at(
            &arena, &topo, elsewhere, to, from, tol
        ));

        let circle =
            Circle3D::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let (a, b) = (0.2, 0.2 + std::f64::consts::FRAC_PI_2);
        let (p0, p1) = (circle.evaluate(a), circle.evaluate(b));
        create_arc_section_edges(
            &mut topo,
            &mut arena,
            &mut Vec::new(),
            [target, other],
            &circle,
            (p0, p1),
            (a, b),
            tol,
        );
        assert!(!has_existing_section_at(&arena, &topo, target, p0, p1, tol));
    }

    /// Records a section on `face` alone along `curve`, from `from` at the
    /// parameter `t.0` to `to` at `t.1`, as the regular FF phase does.
    fn record_section(
        topo: &mut Topology,
        arena: &mut GfaArena,
        face: FaceId,
        curve: EdgeCurve,
        (from, to): (Point3, Point3),
        t: (f64, f64),
    ) {
        let (a, b) = (
            topo.add_vertex(Vertex::new(from, 1e-7)),
            topo.add_vertex(Vertex::new(to, 1e-7)),
        );
        let edge = topo.add_edge(Edge::new(a, b, curve.clone()));
        let pb =
            arena
                .pave_blocks
                .alloc(PaveBlock::new(edge, Pave::new(a, t.0), Pave::new(b, t.1)));
        arena.curves.push(IntersectionCurveDS {
            curve,
            face_a: face,
            face_b: face,
            bbox: Aabb3::from_points([from, to]),
            pave_blocks: vec![pb],
            t_range: t,
        });
    }

    /// A line section as the regular FF phase records one, its parameters
    /// running over its length (0 to 5), is present for a piece with its
    /// two ends.
    #[test]
    fn a_line_section_with_parameters_over_its_length_is_present() {
        let tol = Tolerance::default();
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        let target_rim = rim(&mut topo, 9.0);
        let target = face_down(&mut topo, vec![target_rim], vec![]);
        let (from, to) = (Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 0.0, 0.0));
        let line = EdgeCurve::Line;
        record_section(&mut topo, &mut arena, target, line, (from, to), (0.0, 5.0));
        for (a, b) in [(from, to), (to, from)] {
            assert!(has_existing_section_at(&arena, &topo, target, a, b, tol));
        }
    }

    /// Cubic B-spline sections from (0, 0, 0) to (5, 0, 0). One runs along
    /// the line at an uneven speed: it is present for the piece with those
    /// ends. The other weaves across the line and passes through (2.5, 0,
    /// 0) at its middle parameter: it is not.
    #[test]
    fn a_spline_section_is_present_only_along_the_piece() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let tol = Tolerance::default();
        let p = Point3::new;
        let (from, to) = (p(0.0, 0.0, 0.0), p(5.0, 0.0, 0.0));
        let cubic = |inner: [Point3; 2]| {
            NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![from, inner[0], inner[1], to],
                vec![1.0; 4],
            )
            .unwrap()
        };
        let uneven = cubic([p(0.5, 0.0, 0.0), p(1.0, 0.0, 0.0)]);
        let weave = cubic([p(5.0 / 3.0, 1.0, 0.0), p(10.0 / 3.0, -1.0, 0.0)]);
        assert!((weave.evaluate(0.5) - p(2.5, 0.0, 0.0)).length() < 1e-12);
        for (curve, present) in [(uneven, true), (weave, false)] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            let target_rim = rim(&mut topo, 9.0);
            let target = face_down(&mut topo, vec![target_rim], vec![]);
            let section = EdgeCurve::NurbsCurve(curve);
            record_section(
                &mut topo,
                &mut arena,
                target,
                section,
                (from, to),
                (0.0, 1.0),
            );
            for (a, b) in [(from, to), (to, from)] {
                let found = has_existing_section_at(&arena, &topo, target, a, b, tol);
                assert_eq!(found, present);
            }
        }
    }

    /// The rational quadratic that traces `circle` from its parameter 0
    /// through `quarters` quarter turns, each a span with its corner
    /// weighted by cos 45 degrees.
    fn rational_arc(circle: &Circle3D, quarters: usize) -> brepkit_math::nurbs::curve::NurbsCurve {
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, SQRT_2};
        let along = |k: usize, reach: f64| {
            #[allow(clippy::cast_precision_loss)]
            let t = FRAC_PI_4 * k as f64;
            circle.center() + (circle.u_axis() * t.cos() + circle.v_axis() * t.sin()) * reach
        };
        let r = circle.radius();
        let points: Vec<Point3> = (0..=2 * quarters)
            .map(|k| along(k, if k % 2 == 0 { r } else { r * SQRT_2 }))
            .collect();
        let weights = (0..=2 * quarters)
            .map(|k| if k % 2 == 0 { 1.0 } else { FRAC_1_SQRT_2 })
            .collect();
        let mut knots = vec![0.0; 3];
        for q in 1..quarters {
            #[allow(clippy::cast_precision_loss)]
            let at = q as f64 / quarters as f64;
            knots.extend([at, at]);
        }
        knots.extend([1.0; 3]);
        brepkit_math::nurbs::curve::NurbsCurve::new(2, knots, points, weights).unwrap()
    }

    /// A rational quadratic on the quarter circle's control points whose
    /// weights spread a millionfold is a parabola, not the arc: its ends and
    /// quarter points lie within 6e-12 of the unit circle, yet near
    /// `t = 1e-6` it strays 0.06 off it. It keeps its chords.
    #[test]
    fn a_rational_quadratic_off_its_circle_keeps_its_chords() {
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)],
            vec![1e-12, 1e-6, 1.0],
        )
        .unwrap();
        let stray = (curve.evaluate(1e-6) - p(0.0, 0.0)).length() - 1.0;
        assert!(stray.abs() > 0.05, "{stray}");
        let mut topo = Topology::new();
        let [east, north, origin] =
            [p(1.0, 0.0), p(0.0, 1.0), p(0.0, 0.0)].map(|q| topo.add_vertex(Vertex::new(q, 1e-7)));
        let arc = topo.add_edge(Edge::new(east, north, EdgeCurve::NurbsCurve(curve)));
        let down = topo.add_edge(Edge::new(north, origin, EdgeCurve::Line));
        let back = topo.add_edge(Edge::new(origin, east, EdgeCurve::Line));
        let face = face_down(
            &mut topo,
            vec![
                OrientedEdge::new(arc, true),
                OrientedEdge::new(down, true),
                OrientedEdge::new(back, true),
            ],
            vec![],
        );
        let region = face_region_2d(&topo, face, &frame, 1e-7).unwrap();
        let edge = region.edges.iter().find(|e| e.eid == arc).unwrap();
        assert!(matches!(edge.shape, Shape2::Poly(_)), "read as an arc");
    }

    /// A rim stored as a rational quadratic, whole or a quarter turn, reads
    /// as the arc the same rim stored as a circle reads, either way round.
    #[test]
    fn a_rational_quadratic_rim_reads_as_its_arc() {
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let circle =
            Circle3D::new(Point3::new(1.0, -2.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        for quarters in [4, 1] {
            for forward in [true, false] {
                let mut topo = Topology::new();
                let start = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
                let end = if quarters == 4 {
                    start
                } else {
                    topo.add_vertex(Vertex::new(
                        circle.evaluate(std::f64::consts::FRAC_PI_2),
                        1e-7,
                    ))
                };
                let mut shape_of = |curve: EdgeCurve| {
                    let rim = topo.add_edge(Edge::new(start, end, curve));
                    let mut loop_edges = vec![OrientedEdge::new(rim, forward)];
                    if quarters == 1 {
                        let chord = topo.add_edge(Edge::new(end, start, EdgeCurve::Line));
                        loop_edges.push(OrientedEdge::new(chord, forward));
                        if !forward {
                            loop_edges.reverse();
                        }
                    }
                    let face = face_down(&mut topo, loop_edges, vec![]);
                    let region = face_region_2d(&topo, face, &frame, 1e-7).unwrap();
                    let edge = region.edges.into_iter().find(|e| e.eid == rim).unwrap();
                    match edge.shape {
                        Shape2::Arc(arc) => Some(arc),
                        Shape2::Seg | Shape2::Poly(_) => None,
                    }
                };
                let label = format!("{quarters} quarters, forward {forward}");
                let exact = shape_of(EdgeCurve::Circle(circle.clone())).expect(&label);
                let spline =
                    shape_of(EdgeCurve::NurbsCurve(rational_arc(&circle, quarters))).expect(&label);
                assert!(
                    (exact.center.x() - spline.center.x()).abs() < 1e-12
                        && (exact.center.y() - spline.center.y()).abs() < 1e-12,
                    "{label}: centre"
                );
                assert!(
                    (exact.radius - spline.radius).abs() < 1e-12,
                    "{label}: radius"
                );
                let turn = (exact.a0 - spline.a0).rem_euclid(std::f64::consts::TAU);
                assert!(
                    turn.min(std::f64::consts::TAU - turn) < 1e-12,
                    "{label}: start"
                );
                assert!((exact.sweep - spline.sweep).abs() < 1e-12, "{label}: sweep");
            }
        }
    }

    fn test_poses() -> [brepkit_math::mat::Mat4; 3] {
        use brepkit_math::mat::Mat4;
        [
            Mat4::identity(),
            Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            Mat4::translation(1.0, 0.5, -2.0) * Mat4::scale(-1.0, 1.0, 1.0),
        ]
    }

    /// A whole circle whose seam lies at its parameter pi comes in pieces of
    /// at most a half turn, from the seam round to the seam, in three poses.
    #[test]
    fn a_closed_circle_seamed_at_pi_comes_in_half_turns_from_its_seam() {
        use std::f64::consts::{PI, TAU};
        for pose in test_poses() {
            let centre = pose.mul_point(Point3::new(1.0, 2.0, 0.0));
            let normal = (pose.mul_point(Point3::new(0.0, 0.0, 1.0))
                - pose.mul_point(Point3::new(0.0, 0.0, 0.0)))
            .normalize()
            .unwrap();
            let circle = Circle3D::new(centre, normal, 2.0).unwrap();
            let seam = circle.evaluate(PI);
            let curve = EdgeCurve::Circle(circle);
            let pieces = half_turns(&curve, (seam, seam));
            assert!((pieces[0].1 - seam).length() < 1e-12, "{pose:?}");
            assert!((pieces[pieces.len() - 1].2 - seam).length() < 1e-12);
            let mut total = 0.0;
            for (k, (c, a, b)) in pieces.iter().enumerate() {
                let (t0, t1) = c.domain_with_endpoints(*a, *b);
                let sweep = t1 - t0;
                assert!(sweep > 0.0 && sweep <= PI, "piece {k}: {sweep}");
                total += sweep;
                if k + 1 < pieces.len() {
                    assert!((*b - pieces[k + 1].1).length() < 1e-12);
                }
            }
            assert!((total - TAU).abs() < 1e-9, "{total}");
        }
    }

    /// An arc of the ellipse of semi-axes 3 and 2 from (0.6, -/+ 1.96)
    /// through (3, 0), and an arc of the circle of radius 2 about (1, 0)
    /// through the same three points: the same ends and middle, two curves
    /// with thin regions between them. The circle as a section on a face
    /// does not stand for the ellipse's piece there, but the ellipse does,
    /// in three poses.
    #[test]
    fn an_ellipse_piece_through_a_circle_sections_ends_and_middle_is_kept() {
        use brepkit_math::curves::Ellipse3D;
        let tol = Tolerance::default();
        for pose in test_poses() {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            let target_rim = rim(&mut topo, 9.0);
            let target = face_down(&mut topo, vec![target_rim], vec![]);
            let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 0.0));
            let along = |v: Vec3| {
                pose.mul_point(Point3::new(v.x(), v.y(), v.z()))
                    - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
            };
            let normal = along(Vec3::new(0.0, 0.0, 1.0)).normalize().unwrap();
            let ellipse = Ellipse3D::with_axes(
                at(0.0, 0.0),
                normal,
                3.0,
                2.0,
                along(Vec3::new(1.0, 0.0, 0.0)).normalize().unwrap(),
                along(Vec3::new(0.0, 1.0, 0.0)).normalize().unwrap(),
            )
            .unwrap();
            let y0 = 2.0 * (1.0_f64 - 0.04).sqrt();
            let (from, to, middle) = (at(0.6, -y0), at(0.6, y0), at(3.0, 0.0));
            let circle = Circle3D::new(at(1.0, 0.0), normal, 2.0).unwrap();
            let round = EdgeCurve::Circle(circle);
            // The arc through the middle, read from whichever end its
            // parameter runs from.
            let (ends, span) = [(from, to), (to, from)]
                .into_iter()
                .map(|(p, q)| ((p, q), round.domain_with_endpoints(p, q)))
                .find(|&((p, q), (c0, c1))| {
                    (round.evaluate_with_endpoints(0.5 * (c0 + c1), p, q) - middle).length() < 1e-9
                })
                .unwrap();
            record_section(&mut topo, &mut arena, target, round, ends, span);
            let piece = EdgeCurve::Ellipse(ellipse.clone());
            assert!(!has_existing_section_along(
                &arena,
                &topo,
                target,
                (&piece, from, to),
                middle,
                tol
            ));
            let (e0, e1) = EdgeCurve::Ellipse(ellipse.clone()).domain_with_endpoints(from, to);
            record_section(
                &mut topo,
                &mut arena,
                target,
                EdgeCurve::Ellipse(ellipse),
                (from, to),
                (e0, e1),
            );
            assert!(has_existing_section_along(
                &arena,
                &topo,
                target,
                (&piece, from, to),
                middle,
                tol
            ));
        }
    }

    /// Two segments crossing inside both meet once, at the crossing; two
    /// meeting only at an end, or lying apart, meet nowhere the finder must
    /// report: in three poses.
    #[test]
    fn two_segments_crossing_inside_both_meet_at_the_crossing() {
        let tol = 1e-7;
        let line = EdgeCurve::Line;
        for pose in test_poses() {
            let p = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 0.0));
            let met = crate::builder::face_splitter::section_meetings(
                (&line, p(-1.0, 0.0), p(1.0, 0.0)),
                (&line, p(0.0, -1.0), p(0.0, 1.0)),
                tol,
            )
            .unwrap();
            assert_eq!(met.len(), 1, "{pose:?}");
            assert!((met[0] - p(0.0, 0.0)).length() < 1e-9, "{met:?}");
            for (a, b) in [
                ((p(0.0, 0.0), p(1.0, 0.0)), (p(0.0, 0.0), p(0.0, 1.0))),
                ((p(0.0, 0.0), p(1.0, 0.0)), (p(0.0, 1.0), p(1.0, 1.0))),
            ] {
                let met = crate::builder::face_splitter::section_meetings(
                    (&line, a.0, a.1),
                    (&line, b.0, b.1),
                    tol,
                )
                .unwrap();
                assert!(met.is_empty(), "{met:?}");
            }
        }
    }

    /// A face bounded by one closed ellipse of semi-axes 3 and 2, and an
    /// arc of the concentric ellipse of semi-axes 2.997 and 1.998 over its
    /// parameters 0 to pi / 3: the arc lies inside the face, though its
    /// middle lies outside the 16 chords the face's rim was once read as.
    /// It is a piece to split the face along.
    #[test]
    fn an_ellipse_arc_just_inside_an_ellipse_face_is_a_piece() {
        use brepkit_math::curves::Ellipse3D;
        let tol = Tolerance::default();
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let origin = Point3::new(0.0, 0.0, 0.0);
        let mut topo = Topology::new();
        let outer = Ellipse3D::with_axes(origin, up, 3.0, 2.0, x, y).unwrap();
        let seam = topo.add_vertex(Vertex::new(outer.evaluate(2.0), 1e-7));
        let rim = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Ellipse(outer)));
        let target = face_down(&mut topo, vec![OrientedEdge::new(rim, false)], vec![]);
        let inner = Ellipse3D::with_axes(origin, up, 2.997, 1.998, x, y).unwrap();
        let third = std::f64::consts::FRAC_PI_3;
        let (a, b, c) = (
            topo.add_vertex(Vertex::new(inner.evaluate(0.0), 1e-7)),
            topo.add_vertex(Vertex::new(inner.evaluate(third), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(1.0, 0.5, 0.0), 1e-7)),
        );
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Ellipse(inner)));
        let back = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
        let close = topo.add_edge(Edge::new(c, a, EdgeCurve::Line));
        let piece_face = face_down(
            &mut topo,
            vec![
                OrientedEdge::new(arc, true),
                OrientedEdge::new(back, true),
                OrientedEdge::new(close, true),
            ],
            vec![],
        );
        let arena = GfaArena::new();
        let mut target_region = face_region_2d(&topo, target, &frame, 1e-7).unwrap();
        attach_paves(&mut target_region, &topo, &arena);
        let mut piece_region = face_region_2d(&topo, piece_face, &frame, 1e-7).unwrap();
        attach_paves(&mut piece_region, &topo, &arena);
        let edge = piece_region.edges.iter().find(|e| e.eid == arc).unwrap();
        let mut out = Vec::new();
        clip_to_region(edge, (target, &target_region), &frame, tol.linear, &mut out).unwrap();
        assert_eq!(out.len(), 1, "{}", out.len());
        assert!(matches!(out[0], SectionPiece::Curve { .. }));
    }

    /// A closed ellipse edge, one edge with no paves, wholly inside a face
    /// of radius 9: it is cut into pieces round the whole turn from its seam,
    /// each a piece to split the face along.
    #[test]
    fn a_closed_ellipse_wholly_inside_a_face_is_cut_round_its_turn() {
        use brepkit_math::curves::Ellipse3D;
        let tol = Tolerance::default();
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let mut topo = Topology::new();
        let target_rim = rim(&mut topo, 9.0);
        let target = face_down(&mut topo, vec![target_rim], vec![]);
        let ellipse = Ellipse3D::with_axes(
            Point3::new(1.0, 2.0, 0.0),
            up,
            1.5,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let seam = topo.add_vertex(Vertex::new(ellipse.evaluate(0.7), 1e-7));
        let loop_edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Ellipse(ellipse)));
        let inner = face_down(&mut topo, vec![OrientedEdge::new(loop_edge, false)], vec![]);
        let arena = GfaArena::new();
        let mut target_region = face_region_2d(&topo, target, &frame, 1e-7).unwrap();
        attach_paves(&mut target_region, &topo, &arena);
        let mut inner_region = face_region_2d(&topo, inner, &frame, 1e-7).unwrap();
        attach_paves(&mut inner_region, &topo, &arena);
        let edge = inner_region
            .edges
            .iter()
            .find(|e| e.eid == loop_edge)
            .unwrap();
        let mut out = Vec::new();
        clip_to_region(edge, (target, &target_region), &frame, tol.linear, &mut out).unwrap();
        let sweep: f64 = out
            .iter()
            .map(|piece| match piece {
                SectionPiece::Curve { span, .. } => span.1 - span.0,
                _ => 0.0,
            })
            .sum();
        assert!(out.len() >= 2, "{}", out.len());
        assert!((sweep - std::f64::consts::TAU).abs() < 1e-9, "{sweep}");
    }

    /// A segment along x at z = 0 and one along y at z = 1 lie in no one
    /// plane: no meeting set is given for them. Two parallel segments at
    /// those heights lie in one plane and meet nowhere. In three poses.
    #[test]
    fn two_skew_segments_are_refused() {
        let tol = 1e-7;
        let line = EdgeCurve::Line;
        for pose in test_poses() {
            let p = |x: f64, y: f64, z: f64| pose.mul_point(Point3::new(x, y, z));
            let skew = crate::builder::face_splitter::section_meetings(
                (&line, p(-1.0, 0.0, 0.0), p(1.0, 0.0, 0.0)),
                (&line, p(0.0, -1.0, 1.0), p(0.0, 1.0, 1.0)),
                tol,
            );
            assert!(skew.is_none(), "{skew:?}");
            let parallel = crate::builder::face_splitter::section_meetings(
                (&line, p(-1.0, 0.0, 0.0), p(1.0, 0.0, 0.0)),
                (&line, p(-1.0, 0.0, 1.0), p(1.0, 0.0, 1.0)),
                tol,
            );
            assert!(parallel.as_ref().is_some_and(Vec::is_empty), "{parallel:?}");
        }
    }

    /// A face bounded by an open arc of the ellipse of semi-axes 1.5 and 1
    /// over `sweep` of its parameter from `start`, and two lines back to its
    /// centre, wholly inside a face of radius 9. The arc's half turns, and
    /// the sections it gives the larger face, each lie on the arc's own
    /// range, read from its stored start the way its parameter runs, and
    /// together cover the whole arc: for arcs of 3 pi / 2, a hair over pi
    /// and a hair under a turn, starting at 0 and at pi, in three poses in
    /// the plane.
    #[test]
    fn an_open_ellipse_arc_partitions_its_partner_along_itself() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::mat::Mat4;
        use std::f64::consts::{PI, TAU};
        let tol = Tolerance::default();
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let poses = [
            Mat4::identity(),
            Mat4::translation(3.0, -2.0, 0.0) * Mat4::rotation_z(0.6),
            Mat4::translation(1.0, 0.5, 0.0) * Mat4::scale(-1.0, 1.0, 1.0),
        ];
        for pose in poses {
            for sweep in [1.5 * PI, PI + 0.05, TAU - 0.05] {
                for start in [0.0, PI] {
                    let what = format!("sweep {sweep}, start {start}, {pose:?}");
                    let mut topo = Topology::new();
                    let target_rim = rim(&mut topo, 9.0);
                    let target = face_down(&mut topo, vec![target_rim], vec![]);
                    let along = |v: Vec3| {
                        pose.mul_point(Point3::new(v.x(), v.y(), v.z()))
                            - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
                    };
                    let centre = pose.mul_point(Point3::new(1.0, 2.0, 0.0));
                    let (u, v) = (
                        along(Vec3::new(1.0, 0.0, 0.0)),
                        along(Vec3::new(0.0, 1.0, 0.0)),
                    );
                    let ellipse = Ellipse3D::with_axes(centre, u.cross(v), 1.5, 1.0, u, v).unwrap();
                    let (a, b, c) = (
                        topo.add_vertex(Vertex::new(ellipse.evaluate(start), 1e-7)),
                        topo.add_vertex(Vertex::new(ellipse.evaluate(start + sweep), 1e-7)),
                        topo.add_vertex(Vertex::new(centre, 1e-7)),
                    );
                    let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Ellipse(ellipse.clone())));
                    let back = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
                    let close = topo.add_edge(Edge::new(c, a, EdgeCurve::Line));
                    // The face's normal is -z: its outline runs clockwise
                    // seen from +z, against the arc where the arc runs the
                    // other way.
                    let outline = if ellipse.normal().z() > 0.0 {
                        vec![
                            OrientedEdge::new(close, false),
                            OrientedEdge::new(back, false),
                            OrientedEdge::new(arc, false),
                        ]
                    } else {
                        vec![
                            OrientedEdge::new(arc, true),
                            OrientedEdge::new(back, true),
                            OrientedEdge::new(close, true),
                        ]
                    };
                    let pacman = face_down(&mut topo, outline, vec![]);
                    let mut arena = GfaArena::new();
                    let mut regions = Vec::new();
                    for face in [pacman, target] {
                        let mut region = face_region_2d(&topo, face, &frame, 1e-7).unwrap();
                        attach_paves(&mut region, &topo, &arena);
                        regions.push(region);
                    }
                    process_coplanar_pair(
                        &mut topo,
                        &mut arena,
                        &mut Vec::new(),
                        (pacman, &regions[0]),
                        (target, &regions[1]),
                        &frame,
                        tol,
                    )
                    .unwrap();
                    let on_arc = |p: Point3| (ellipse.project(p) - start).rem_euclid(TAU);
                    let covers = |pieces: Vec<(EdgeCurve, Point3, Point3)>, what: &str| {
                        let mut covered = 0.0;
                        for (curve, s, e) in pieces {
                            let (t0, t1) = curve.domain_with_endpoints(s, e);
                            covered += t1 - t0;
                            for k in 0..=8 {
                                let p = curve.evaluate_with_endpoints(
                                    (t1 - t0).madd(f64::from(k) / 8.0, t0),
                                    s,
                                    e,
                                );
                                let at = on_arc(p);
                                assert!(
                                    at <= sweep + 1e-9 || at >= TAU - 1e-9,
                                    "{what}: off the arc at {at}"
                                );
                            }
                        }
                        assert!((covered - sweep).abs() < 1e-9, "{what}: covered {covered}");
                    };
                    let edge = topo.edge(arc).unwrap();
                    let turns = half_turns(
                        edge.curve(),
                        (
                            topo.vertex(edge.start()).unwrap().point(),
                            topo.vertex(edge.end()).unwrap().point(),
                        ),
                    );
                    assert!(turns.len() >= 2, "{what}: {} half turns", turns.len());
                    for (curve, s, e) in &turns {
                        let (t0, t1) = curve.domain_with_endpoints(*s, *e);
                        assert!(
                            t1 - t0 <= 0.99 * PI + 1e-12,
                            "{what}: a turn of {}",
                            t1 - t0
                        );
                    }
                    covers(turns, &format!("{what}, half turns"));
                    let mut sections = Vec::new();
                    for curve in &arena.curves {
                        let EdgeCurve::Ellipse(_) = &curve.curve else {
                            continue;
                        };
                        let pb = arena.pave_blocks.get(curve.pave_blocks[0]).unwrap();
                        let edge = topo.edge(pb.original_edge).unwrap();
                        let (s, e) = (
                            topo.vertex(edge.start()).unwrap().point(),
                            topo.vertex(edge.end()).unwrap().point(),
                        );
                        sections.push((edge.curve().clone(), s, e));
                    }
                    covers(sections, &format!("{what}, sections"));
                }
            }
        }
    }

    /// Three poses that keep the plane z = 0: upright, turned in it and
    /// mirrored across x = 0.
    fn in_plane_poses() -> [brepkit_math::mat::Mat4; 3] {
        use brepkit_math::mat::Mat4;
        [
            Mat4::identity(),
            Mat4::translation(3.0, -2.0, 0.0) * Mat4::rotation_z(0.6),
            Mat4::translation(1.0, 0.5, 0.0) * Mat4::scale(-1.0, 1.0, 1.0),
        ]
    }

    /// A face of normal -z bounded by the whole ellipse of semi-axes `axes`
    /// about `centre` along x and y, placed by `pose`, from its point at
    /// `seam`. Its parameter runs counterclockwise about the normal of its
    /// own axes, which a mirror turns over.
    fn ellipse_face_down(
        topo: &mut Topology,
        pose: &brepkit_math::mat::Mat4,
        centre: (f64, f64),
        axes: (f64, f64),
        seam: f64,
    ) -> (FaceId, brepkit_math::curves::Ellipse3D) {
        let along = |v: Vec3| {
            pose.mul_point(Point3::new(v.x(), v.y(), v.z()))
                - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
        };
        let (u, v) = (
            along(Vec3::new(1.0, 0.0, 0.0)),
            along(Vec3::new(0.0, 1.0, 0.0)),
        );
        let ellipse = brepkit_math::curves::Ellipse3D::with_axes(
            pose.mul_point(Point3::new(centre.0, centre.1, 0.0)),
            u.cross(v),
            axes.0,
            axes.1,
            u,
            v,
        )
        .unwrap();
        let vertex = topo.add_vertex(Vertex::new(ellipse.evaluate(seam), 1e-7));
        let edge = topo.add_edge(Edge::new(
            vertex,
            vertex,
            EdgeCurve::Ellipse(ellipse.clone()),
        ));
        let forward = ellipse.normal().z() < 0.0;
        (
            face_down(topo, vec![OrientedEdge::new(edge, forward)], vec![]),
            ellipse,
        )
    }

    /// A face of normal -z bounded by the polygon through `corners`, given
    /// counterclockwise about +z, placed by `pose`.
    fn polygon_face_down(
        topo: &mut Topology,
        pose: &brepkit_math::mat::Mat4,
        corners: &[(f64, f64)],
    ) -> FaceId {
        let vertices: Vec<_> = corners
            .iter()
            .map(|&(x, y)| {
                topo.add_vertex(Vertex::new(pose.mul_point(Point3::new(x, y, 0.0)), 1e-7))
            })
            .collect();
        let edges: Vec<_> = (0..vertices.len())
            .map(|k| {
                topo.add_edge(Edge::new(
                    vertices[k],
                    vertices[(k + 1) % vertices.len()],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        let turned =
            pose.mul_point(Point3::new(1.0, 0.0, 0.0)) - pose.mul_point(Point3::new(0.0, 0.0, 0.0));
        let up = turned.cross(
            pose.mul_point(Point3::new(0.0, 1.0, 0.0)) - pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
        );
        let outline = if up.z() > 0.0 {
            edges
                .iter()
                .rev()
                .map(|&e| OrientedEdge::new(e, false))
                .collect()
        } else {
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
        };
        face_down(topo, outline, vec![])
    }

    /// The coplanar phase's sections between `faces`, as each section's
    /// curve and its two ends.
    fn coplanar_sections(
        topo: &mut Topology,
        faces: [FaceId; 2],
        tol: Tolerance,
    ) -> Vec<(EdgeCurve, Point3, Point3)> {
        let frame = PlaneFrame2D::canonical(Vec3::new(0.0, 0.0, -1.0), 0.0);
        let mut arena = GfaArena::new();
        let mut regions = Vec::new();
        for face in faces {
            let mut region = face_region_2d(topo, face, &frame, tol.linear).unwrap();
            attach_paves(&mut region, topo, &arena);
            regions.push(region);
        }
        process_coplanar_pair(
            topo,
            &mut arena,
            &mut Vec::new(),
            (faces[0], &regions[0]),
            (faces[1], &regions[1]),
            &frame,
            tol,
        )
        .unwrap();
        arena
            .curves
            .iter()
            .map(|curve| {
                let pb = arena.pave_blocks.get(curve.pave_blocks[0]).unwrap();
                let edge = topo.edge(pb.original_edge).unwrap();
                (
                    edge.curve().clone(),
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                )
            })
            .collect()
    }

    /// Whether an ellipse section runs over the parameter `t` of its curve,
    /// short of its ends.
    fn covers(sections: &[(EdgeCurve, Point3, Point3)], t: f64) -> bool {
        use std::f64::consts::TAU;
        sections.iter().any(|(curve, s, e)| {
            let EdgeCurve::Ellipse(_) = curve else {
                return false;
            };
            let (t0, t1) = curve.domain_with_endpoints(*s, *e);
            let at = (t - t0).rem_euclid(TAU);
            at > 0.0 && at < t1 - t0
        })
    }

    /// An existing section on the face 5 tolerances off an ellipse piece,
    /// its ends 5 tolerances off the piece's, is another curve: the finder
    /// proves the two apart, so the piece is not taken for it.
    #[test]
    fn a_section_five_tolerances_off_an_ellipse_piece_is_not_the_piece() {
        use brepkit_math::curves::Ellipse3D;
        let tol = Tolerance::default();
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        let target_rim = rim(&mut topo, 9.0);
        let target = face_down(&mut topo, vec![target_rim], vec![]);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let ellipse = Ellipse3D::with_axes(Point3::new(0.0, 0.0, 0.0), up, 3.0, 2.0, x, y).unwrap();
        let off = 5.0 * tol.linear;
        let wider =
            Ellipse3D::with_axes(Point3::new(0.0, 0.0, 0.0), up, 3.0 + off, 2.0 + off, x, y)
                .unwrap();
        let (from, to, middle) = (
            ellipse.evaluate(-1.0),
            ellipse.evaluate(1.0),
            ellipse.evaluate(0.0),
        );
        let near = (wider.evaluate(-1.0), wider.evaluate(1.0));
        assert!((near.0 - from).length() < 10.0 * tol.linear);
        let wider = EdgeCurve::Ellipse(wider);
        let span = wider.domain_with_endpoints(near.0, near.1);
        record_section(&mut topo, &mut arena, target, wider, near, span);
        let piece = EdgeCurve::Ellipse(ellipse);
        assert!(!has_existing_section_along(
            &arena,
            &topo,
            target,
            (&piece, from, to),
            middle,
            tol
        ));
    }

    /// The whole ellipse of semi-axes 3 and 2 about the origin, against a
    /// square of side 10 with a slot 5 tolerances wide cut down to y = 1
    /// along x = 0. The ellipse crosses the slot's two walls 5 tolerances
    /// apart: the stretch between them lies outside the square, and no
    /// section on the square runs over it, in three poses.
    #[test]
    fn an_ellipse_across_a_slot_five_tolerances_wide_gives_no_section_over_it() {
        use std::f64::consts::FRAC_PI_2;
        let tol = Tolerance::default();
        let w = 2.5 * tol.linear;
        for pose in in_plane_poses() {
            let mut topo = Topology::new();
            let (face, ellipse) = ellipse_face_down(&mut topo, &pose, (0.0, 0.0), (3.0, 2.0), 0.0);
            let square = polygon_face_down(
                &mut topo,
                &pose,
                &[
                    (-5.0, -5.0),
                    (5.0, -5.0),
                    (5.0, 5.0),
                    (w, 5.0),
                    (w, 1.0),
                    (-w, 1.0),
                    (-w, 5.0),
                    (-5.0, 5.0),
                ],
            );
            let sections = coplanar_sections(&mut topo, [face, square], tol);
            let top = ellipse.project(pose.mul_point(Point3::new(0.0, 2.0, 0.0)));
            assert!(
                (ellipse.evaluate(top) - pose.mul_point(Point3::new(0.0, 2.0, 0.0))).length()
                    < 1e-12
            );
            assert!(!covers(&sections, top), "{pose:?}: a section over the slot");
            assert!(covers(&sections, top + FRAC_PI_2), "{pose:?}");
        }
    }

    /// Two circles of radius 1000 touching at (1000, 0, 0), the second's
    /// plane turned 5e-10 about its own diameter along y: at the touch it
    /// stands 5 tolerances above the first's plane, though its ends lie in
    /// it. They lie in no one plane within the tolerance. Two circles of
    /// radius 1e-3 whose normals part by 2e-9 lie in one within 2e-12.
    #[test]
    fn two_circles_read_in_one_plane_by_their_points_not_their_normals() {
        use brepkit_math::curves::Circle3D;
        let tol = 1e-7;
        let up = Vec3::new(0.0, 0.0, 1.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let a = Circle3D::with_axes(Point3::new(0.0, 0.0, 0.0), up, 1000.0, x, y).unwrap();
        let tilt = 5e-10_f64;
        let tilted = Vec3::new(tilt.sin(), 0.0, tilt.cos());
        let b = Circle3D::with_axes(
            Point3::new(2000.0, 0.0, 0.0),
            tilted,
            1000.0,
            Vec3::new(-tilt.cos(), 0.0, tilt.sin()),
            Vec3::new(0.0, -1.0, 0.0),
        )
        .unwrap();
        let (b0, b1) = (b.evaluate(-1.5), b.evaluate(1.5));
        assert!(b0.z().abs() < 0.5 * tol && b1.z().abs() < 0.5 * tol);
        assert!((b.evaluate(0.0).z() - 5e-7).abs() < 1e-12);
        let a_arc = (a.evaluate(-0.5), a.evaluate(0.5));
        assert!((a.evaluate(0.0) - Point3::new(1000.0, 0.0, 0.0)).length() < 1e-12);
        let met = crate::builder::face_splitter::section_meetings(
            (&EdgeCurve::Circle(a), a_arc.0, a_arc.1),
            (&EdgeCurve::Circle(b), b0, b1),
            tol,
        );
        assert!(met.is_none(), "{met:?}");
        let small = Circle3D::with_axes(Point3::new(0.0, 0.0, 0.0), up, 1e-3, x, y).unwrap();
        let tilt = 2e-9_f64;
        let other = Circle3D::with_axes(
            Point3::new(1.5e-3, 0.0, 0.0),
            Vec3::new(tilt.sin(), 0.0, tilt.cos()),
            1e-3,
            Vec3::new(tilt.cos(), 0.0, -tilt.sin()),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let met = crate::builder::face_splitter::section_meetings(
            (
                &EdgeCurve::Circle(small.clone()),
                small.evaluate(-1.5),
                small.evaluate(1.5),
            ),
            (
                &EdgeCurve::Circle(other.clone()),
                other.evaluate(1.5),
                other.evaluate(1.5 + 3.0),
            ),
            tol,
        );
        assert!(met.as_ref().is_some_and(|m| m.len() == 2), "{met:?}");
    }

    /// An ellipse of semi-axes 1.5 and 1 inside a disc of radius 3, its far
    /// end on x 0.5 tolerances short of the rim, or 2 tolerances past it,
    /// with its seam on that end or across from it, in three poses. Short
    /// of the rim, the sections on the disc run round the whole ellipse.
    /// Past it, none runs over the far end, which lies outside the disc,
    /// and no section strays more than a tolerance past the rim.
    #[test]
    fn an_ellipse_grazing_a_disc_rim_is_kept_inside_and_dropped_outside() {
        use std::f64::consts::{PI, TAU};
        let tol = Tolerance::default();
        for pose in in_plane_poses() {
            let origin = pose.mul_point(Point3::new(0.0, 0.0, 0.0));
            for (gap, inside) in [(0.5 * tol.linear, true), (-2.0 * tol.linear, false)] {
                for seam in [0.0, PI] {
                    let what = format!("gap {gap:e}, seam {seam}, {pose:?}");
                    let mut topo = Topology::new();
                    let circle = Circle3D::new(origin, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
                    let v = topo.add_vertex(Vertex::new(circle.evaluate(2.4), 1e-7));
                    let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
                    let disc = face_down(&mut topo, vec![OrientedEdge::new(edge, false)], vec![]);
                    let (face, _) =
                        ellipse_face_down(&mut topo, &pose, (1.5 - gap, 0.0), (1.5, 1.0), seam);
                    let sections: Vec<_> = coplanar_sections(&mut topo, [face, disc], tol)
                        .into_iter()
                        .filter(|(c, _, _)| matches!(c, EdgeCurve::Ellipse(_)))
                        .collect();
                    let mut covered = 0.0;
                    for (curve, s, e) in &sections {
                        let (t0, t1) = curve.domain_with_endpoints(*s, *e);
                        covered += t1 - t0;
                        for k in 0..=64 {
                            let p = curve.evaluate_with_endpoints(
                                (t1 - t0).madd(f64::from(k) / 64.0, t0),
                                *s,
                                *e,
                            );
                            assert!(
                                (p - origin).length() <= 3.0 + tol.linear,
                                "{what}: a section past the rim at {p:?}"
                            );
                        }
                    }
                    if inside {
                        assert!((covered - TAU).abs() < 1e-9, "{what}: covered {covered}");
                    } else {
                        assert!(!covers(&sections, 0.0), "{what}: a section past the rim");
                        assert!(covered > TAU - 0.01, "{what}: covered {covered}");
                    }
                }
            }
        }
    }
}
