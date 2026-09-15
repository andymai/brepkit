//! Phase FF-Coplanar: coplanar face splitting.
//!
//! Handles the case where two faces from different solids lie on the same
//! plane and partially overlap. Phase FF skips these because parallel planes
//! have no intersection line. This phase runs after FF and creates section
//! edges by clipping one face's boundary edges (segments and arcs, exactly)
//! to the other face's region.

use brepkit_math::aabb::Aabb3;
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
    // Vertices minted by this phase, so a crossing computed once along an
    // arc and once along a line (or shared by two pairs) is one vertex.
    let mut minted: Vec<(Point3, VertexId)> = Vec::new();

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
                    slot.insert(face_region_2d(topo, fid, &frame, tol.linear)?);
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

    if points.is_empty() {
        Ok(Aabb3 {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(0.0, 0.0, 0.0),
        })
    } else {
        Ok(Aabb3::from_points(points))
    }
}

/// Check if a section curve already exists at this position for either face.
///
/// Searches `arena.curves` for any existing intersection curve involving
/// `face_a` or `face_b` whose endpoints match `p_start`/`p_end` within
/// tolerance. This prevents the coplanar phase from creating duplicate
/// section edges that already exist from the regular FF phase.
fn has_existing_section_at(
    arena: &GfaArena,
    face_a: FaceId,
    face_b: FaceId,
    p_start: Point3,
    p_end: Point3,
    tol: Tolerance,
) -> bool {
    for curve in &arena.curves {
        if curve.face_a != face_a
            && curve.face_a != face_b
            && curve.face_b != face_a
            && curve.face_b != face_b
        {
            continue;
        }

        let edge_min = Point3::new(
            p_start.x().min(p_end.x()),
            p_start.y().min(p_end.y()),
            p_start.z().min(p_end.z()),
        );
        let edge_max = Point3::new(
            p_start.x().max(p_end.x()),
            p_start.y().max(p_end.y()),
            p_start.z().max(p_end.z()),
        );
        let expanded = curve.bbox.expanded(tol.linear);
        if edge_min.x() > expanded.max.x()
            || edge_max.x() < expanded.min.x()
            || edge_min.y() > expanded.max.y()
            || edge_max.y() < expanded.min.y()
            || edge_min.z() > expanded.max.z()
            || edge_max.z() < expanded.min.z()
        {
            continue;
        }

        // Check endpoint match: midpoint of proposed edge must be near the
        // existing curve's midpoint. Use midpoint instead of endpoint to
        // handle reversed-direction curves.
        let mid = Point3::new(
            (p_start.x() + p_end.x()) * 0.5,
            (p_start.y() + p_end.y()) * 0.5,
            (p_start.z() + p_end.z()) * 0.5,
        );
        let curve_mid = Point3::new(
            (curve.bbox.min.x() + curve.bbox.max.x()) * 0.5,
            (curve.bbox.min.y() + curve.bbox.max.y()) * 0.5,
            (curve.bbox.min.z() + curve.bbox.max.z()) * 0.5,
        );
        if (mid - curve_mid).length() < tol.linear * 10.0 {
            return true;
        }
    }
    false
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
            clip_to_region(e, region_a, frame, tol.linear, &mut pieces);
        }
    }
    for e in &region_a.edges {
        if coincident_boundary_edge(e, &region_b.edges, tol.linear).is_none() {
            clip_to_region(e, region_b, frame, tol.linear, &mut pieces);
        }
    }
    let faces = [face_a, face_b];
    for piece in pieces {
        match piece {
            SectionPiece::Line(s, e) => {
                if !has_existing_section_at(arena, face_a, face_b, s, e, tol) {
                    create_section_edge(topo, arena, minted, faces, s, e, tol)?;
                }
            }
            SectionPiece::Arc {
                circle,
                start,
                end,
                mid,
            } => {
                // The wall sharing this arc meets the partner plane in the
                // same circle, so the regular FF phase usually emitted the
                // arc already, split at the same partner crossings; only
                // what it left uncovered is emitted here.
                let (t_s, t_e, p_s, p_e) = oriented_arc_span(&circle, start, end, mid);
                let ang_tol = tol.linear * 10.0 / circle.radius();
                for (a, b) in uncovered_arc_spans(arena, faces, &circle, t_s, t_e, ang_tol) {
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
}

/// A face's boundary in the plane's frame: every wire's edges, plus one
/// arc-true sampled polygon per wire for the even-odd region test.
struct Region2 {
    edges: Vec<BoundaryEdge>,
    loops: Vec<Vec<Point2>>,
}

/// A clipped piece of a boundary edge inside the partner face.
enum SectionPiece {
    Line(Point3, Point3),
    Arc {
        circle: brepkit_math::curves::Circle3D,
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
        arc.radius.mul_add(phi.cos(), arc.center.x()),
        arc.radius.mul_add(phi.sin(), arc.center.y()),
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

/// Interior samples of an arc (start and end excluded) for the region
/// polygon: chords within a hundred linear tolerances of the arc up to a
/// radius of about 34 mm, and within the 8192-segment cap's sagitta
/// (`r * (1 - cos(pi / 8192))`, 7e-6 at r = 100) beyond that.
fn arc_samples(arc: &Arc2, tol: f64) -> Vec<Point2> {
    let sag = (tol * 100.0).min(arc.radius * 0.5);
    let step = 2.0 * (1.0 - sag / arc.radius).acos();
    let len = arc.sweep.abs();
    let n = if step > 0.0 {
        ((len / step).ceil() as usize).clamp(8, 8192)
    } else {
        8
    };
    (1..n)
        .map(|k| arc_point(arc, len * k as f64 / n as f64))
        .collect()
}

fn lerp3(a: Point3, b: Point3, t: f64) -> Point3 {
    Point3::new(
        (b.x() - a.x()).mul_add(t, a.x()),
        (b.y() - a.y()).mul_add(t, a.y()),
        (b.z() - a.z()).mul_add(t, a.z()),
    )
}

/// Collect a face's boundary (outer and inner wires) in the frame.
fn face_region_2d(
    topo: &Topology,
    face_id: FaceId,
    frame: &PlaneFrame2D,
    tol: f64,
) -> Result<Region2, AlgoError> {
    use std::f64::consts::TAU;
    let face = topo.face(face_id)?;
    let mut edges = Vec::new();
    let mut loops = Vec::new();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wid)?;
        let mut polygon = Vec::new();
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
            polygon.push(p2_start);
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
                    let arc = Arc2 {
                        center,
                        radius,
                        a0,
                        sweep,
                        circle: circle.clone(),
                    };
                    polygon.extend(arc_samples(&arc, tol));
                    Shape2::Arc(arc)
                }
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                    let (t0, t1) = edge.curve().domain_with_endpoints(v_start, v_end);
                    let mut samples: Vec<Point2> = (1..POLY_SAMPLES)
                        .map(|k| {
                            let t = (t1 - t0).mul_add(k as f64 / POLY_SAMPLES as f64, t0);
                            frame.project(edge.curve().evaluate_with_endpoints(t, v_start, v_end))
                        })
                        .collect();
                    if !oe.is_forward() {
                        samples.reverse();
                    }
                    polygon.extend(samples.iter().copied());
                    let mut poly = Vec::with_capacity(POLY_SAMPLES + 1);
                    poly.push(p2_start);
                    poly.extend(samples);
                    poly.push(p2_end);
                    Shape2::Poly(poly)
                }
            };
            edges.push(BoundaryEdge {
                eid: oe.edge(),
                p2_start,
                p2_end,
                p3_start,
                p3_end,
                shape,
            });
        }
        if polygon.len() >= 3 {
            loops.push(polygon);
        }
    }
    Ok(Region2 { edges, loops })
}

/// Even-odd containment over every wire of the region.
fn inside_region(pt: Point2, region: &Region2) -> bool {
    region
        .loops
        .iter()
        .fold(false, |acc, lp| acc ^ point_in_polygon_2d(pt, lp))
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

/// Index of the partner edge that `e` lies on along its whole length.
fn coincident_boundary_edge(e: &BoundaryEdge, target: &[BoundaryEdge], tol: f64) -> Option<usize> {
    let probes = carrier_probes(e);
    target
        .iter()
        .position(|t| probes.iter().all(|&p| point_on_edge_2d(p, t, tol)))
}

/// Parameters `t` along the segment `a..b` where it crosses the target edge.
fn seg_crossings(a: Point2, b: Point2, t: &BoundaryEdge, tol: f64, out: &mut Vec<f64>) {
    match &t.shape {
        Shape2::Seg => {
            if let Some(x) = seg_seg_param(a, b, t.p2_start, t.p2_end) {
                out.push(x);
            }
        }
        Shape2::Poly(poly) => {
            for w in poly.windows(2) {
                if let Some(x) = seg_seg_param(a, b, w[0], w[1]) {
                    out.push(x);
                }
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

/// Parameter along `a..b` of its crossing with `c..d`, if the segments cross.
fn seg_seg_param(a: Point2, b: Point2, c: Point2, d: Point2) -> Option<f64> {
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
    let foot = Point2::new(dx.mul_add(t_foot, a.x()), dy.mul_add(t_foot, a.y()));
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
            (t, Point2::new(dx.mul_add(t, a.x()), dy.mul_add(t, a.y())))
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
    let mx = ex.mul_add(a, c1.x());
    let my = ey.mul_add(a, c1.y());
    if d >= r1 + r2 - tol || d <= (r1 - r2).abs() + tol {
        return vec![Point2::new(mx, my)];
    }
    let h = r1.mul_add(r1, -(a * a)).max(0.0).sqrt();
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
    target: &Region2,
    frame: &PlaneFrame2D,
    tol: f64,
    out: &mut Vec<SectionPiece>,
) {
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
                if on_boundary(mid2, &target.edges, tol) || !inside_region(mid2, target) {
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
                    circle: arc.circle.clone(),
                    start,
                    end,
                    mid: on_circle(mid2),
                });
            }
        }
        Shape2::Seg | Shape2::Poly(_) => {
            let (a, b) = (e.p2_start, e.p2_end);
            let d = Point2::new(b.x() - a.x(), b.y() - a.y());
            let seg_len = d.x().hypot(d.y());
            if seg_len < tol {
                return;
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
                let mid = Point2::new(d.x().mul_add(tm, a.x()), d.y().mul_add(tm, a.y()));
                if on_boundary(mid, &target.edges, tol) || !inside_region(mid, target) {
                    continue;
                }
                out.push(SectionPiece::Line(
                    lerp3(e.p3_start, e.p3_end, ta),
                    lerp3(e.p3_start, e.p3_end, tb),
                ));
            }
        }
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
/// involving either face already spans. Sections are compared on the same
/// circle (centre, radius, carrier plane) and their spans are taken modulo
/// the period.
fn uncovered_arc_spans(
    arena: &GfaArena,
    faces: [FaceId; 2],
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
        let shares_face = faces.contains(&c.face_a) || faces.contains(&c.face_b);
        if !shares_face
            || (existing.center() - circle.center()).length() > lin_tol
            || (existing.radius() - circle.radius()).abs() > lin_tol
            || existing.normal().cross(circle.normal()).length() > 1e-9
        {
            continue;
        }
        // The existing section's parameters are the other circle object's;
        // re-project its endpoints so both spans share one origin.
        let c0 = circle.project(existing.evaluate(c.t_range.0));
        let width = c.t_range.1 - c.t_range.0;
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
            span.mul_add(k as f64 / n_sub as f64, t0)
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
        (0..=n).map(|k| circle.evaluate((t_e - t_s).mul_add(k as f64 / n as f64, t_s))),
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

/// Ray-casting point-in-polygon test.
///
/// Returns `true` if `pt` is strictly inside `polygon` (CCW or CW vertex order).
fn point_in_polygon_2d(pt: Point2, polygon: &[Point2]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;

    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];

        let yi = pi.y();
        let yj = pj.y();
        let xi = pi.x();
        let xj = pj.x();

        if ((yi > pt.y()) != (yj > pt.y())) && (pt.x() < (xj - xi) * (pt.y() - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
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
    fn point_in_unit_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(point_in_polygon_2d(Point2::new(0.5, 0.5), &square));
        assert!(!point_in_polygon_2d(Point2::new(2.0, 0.5), &square));
        assert!(!point_in_polygon_2d(Point2::new(-0.1, 0.5), &square));
    }

    #[test]
    fn point_on_segment() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        assert!(point_on_segment_2d(Point2::new(0.5, 0.0), a, b, 1e-7));
        assert!(!point_on_segment_2d(Point2::new(0.5, 1.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(0.0, 0.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(1.0, 0.0), a, b, 1e-7));
    }
}
