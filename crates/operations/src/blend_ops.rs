//! Thin wrappers around `brepkit-blend` for the operations API.

use std::collections::{HashMap, HashSet};

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_math::madd::Madd;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::OperationsError;

/// The blended solid, when every requested edge was blended.
///
/// A [`BlendResult`] that left edges out still carries a closed solid (the
/// input itself when none was blended), which a caller expecting the whole
/// request would take for success.
///
/// # Errors
/// Returns [`OperationsError::InvalidInput`] naming how many edges failed
/// and why the first did, if any edge failed.
pub fn all_blended(result: BlendResult) -> Result<SolidId, OperationsError> {
    match result.failed.first() {
        None => Ok(result.solid),
        Some((edge, error)) => Err(OperationsError::InvalidInput {
            reason: format!(
                "{} edge(s) could not be blended (edge {}: {error})",
                result.failed.len(),
                edge.index()
            ),
        }),
    }
}

/// `result` as a solid of its own: the walking builders share the input's
/// unchanged edges and its cavity shells with the result, which would move
/// with it under an in-place transform of either. A blend reaching a cavity
/// is refused.
fn own_cavities(
    topo: &mut Topology,
    solid: SolidId,
    mut result: BlendResult,
    what: &str,
) -> Result<BlendResult, OperationsError> {
    if result.solid == solid {
        return Ok(result);
    }
    result.solid = crate::cavities::detach_walked(topo, solid, result.solid, what)?;
    Ok(result)
}

/// A walking-engine chamfer's result as a solid of its own, refused where
/// it is not a valid solid: the builder does not read what else lies in the
/// regions its chamfers cut, such as a hole in the triangle an end takes
/// from its end face, which it would leave outside that face. A pcurve the
/// chamfer gave an edge must also follow that edge on its face, which the
/// solid's checks do not read.
fn valid_chamfer(
    topo: &mut Topology,
    solid: SolidId,
    built: BlendResult,
) -> Result<BlendResult, OperationsError> {
    if built.solid != solid
        && let Some(issue) = pcurve_off_its_edge(topo, solid, built.solid)?
    {
        return Err(OperationsError::InvalidInput {
            reason: format!("the chamfer leaves an invalid solid: {issue}"),
        });
    }
    let result = own_cavities(topo, solid, built, "chamfer")?;
    if result.solid != solid {
        let report = crate::validate::validate_solid(topo, result.solid)?;
        if let Some(issue) = report
            .issues
            .iter()
            .find(|issue| issue.severity == crate::validate::Severity::Error)
        {
            return Err(OperationsError::InvalidInput {
                reason: format!("the chamfer leaves an invalid solid: {}", issue.description),
            });
        }
    }
    Ok(result)
}

/// The first edge of `result` that `source` does not have, whose pcurve on
/// a curved face of `result`, read through that face's surface, strays from
/// the edge: by more than a hundred times the fit the blend engine builds a
/// pcurve to, and not less than the solid checks' own `1e-4`.
fn pcurve_off_its_edge(
    topo: &Topology,
    source: SolidId,
    result: SolidId,
) -> Result<Option<String>, OperationsError> {
    let old: HashSet<EdgeId> = brepkit_topology::explorer::solid_edges(topo, source)?
        .into_iter()
        .collect();
    let size = crate::measure::solid_bounding_box(topo, result)
        .map_or(0.0, |aabb| (aabb.max - aabb.min).length());
    let tolerance = 1e-5_f64.madd(size, 1e-5).max(1e-4);
    for face in solid_faces(topo, result)? {
        let data = topo.face(face)?;
        for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            for oe in topo.wire(wire)?.edges() {
                if old.contains(&oe.edge()) {
                    continue;
                }
                let found = brepkit_check::validate::check_edge_same_parameter(
                    topo,
                    oe.edge(),
                    face,
                    tolerance,
                )
                .map_err(|e| OperationsError::InvalidInput {
                    reason: e.to_string(),
                })?;
                if let Some(issue) = found.first() {
                    return Ok(Some(format!(
                        "the pcurve of edge {} on face {}: {}",
                        oe.edge().index(),
                        face.index(),
                        issue.description
                    )));
                }
            }
        }
    }
    Ok(None)
}

/// Fillet edges with constant radius — the production engine.
///
/// Runs the corrected [`FilletBuilder`]: one ordered contour per G1 run,
/// canonical boundary-registry ownership, ordered-fan junction corners, and
/// closed periodic bands for circular rims. This is the engine the WASM
/// dispatch prefers; the deprecated `fillet_rolling_ball` remains callable
/// only as an explicit reference implementation.
///
/// A result that is not a valid solid is refused: the solid comes back with
/// each edge named in `failed`.
///
/// # Errors
/// Returns `OperationsError` if radius is non-positive, edges are empty,
/// or the blend computation fails.
pub fn fillet_v2(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<BlendResult, OperationsError> {
    if radius <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: "radius must be positive".into(),
        });
    }
    if edges.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    // The bore path only takes a round it builds exactly: one it declines, or
    // fails on, is left to the direct fillet, which reports its own errors.
    match fillet_over_bores(topo, solid, edges, radius) {
        Ok(Some(result)) => return Ok(result),
        Ok(None) => {}
        Err(e) => log::debug!("fillet over bores: {e}"),
    }
    fillet_directly(topo, solid, edges, radius)
}

/// `edges` of `solid` rounded by the walking engine as they are.
fn fillet_directly(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<BlendResult, OperationsError> {
    let mut builder = FilletBuilder::new(topo, solid);
    builder.add_edges(edges, radius);
    let built = builder.build()?;
    let result = own_cavities(topo, solid, built, "fillet")?;
    valid_fillet(topo, solid, result)
}

/// A walking-engine fillet's result, or `solid` with every edge the
/// builder rounded reported as failed where the result is not a valid
/// solid. The builder checks how its faces share edges, not where the
/// faces lie: two rounds that meet at a corner the corner solver does not
/// model can leave a face whose outline crosses itself.
fn valid_fillet(
    topo: &Topology,
    solid: SolidId,
    result: BlendResult,
) -> Result<BlendResult, OperationsError> {
    if result.solid == solid {
        return Ok(result);
    }
    let report = crate::validate::validate_solid(topo, result.solid)?;
    let Some(issue) = report
        .issues
        .iter()
        .find(|issue| issue.severity == crate::validate::Severity::Error)
    else {
        return Ok(result);
    };
    // An input the checks already reject (two blocks meeting along one
    // edge of four faces) gives them nothing to tell the fillet's own
    // faults by.
    if !crate::validate::validate_solid(topo, solid)?.is_valid() {
        return Ok(result);
    }
    let reason = format!("the fillet leaves an invalid solid: {}", issue.description);
    let mut failed = result.failed;
    failed.extend(result.succeeded.into_iter().map(|edge| {
        (
            edge,
            brepkit_blend::BlendError::PlanningFailure {
                reason: reason.clone(),
            },
        )
    }));
    Ok(BlendResult {
        solid,
        succeeded: Vec::new(),
        failed,
        is_partial: true,
    })
}

/// A straight through-bore: a cylinder wall with the material round it, from
/// a circle that is a hole in one plane face to a circle that is a hole in
/// another.
struct Bore {
    wall: FaceId,
    mouths: [(FaceId, brepkit_topology::wire::WireId); 2],
    centres: [Point3; 2],
    radius: f64,
}

/// The through-bores with a mouth on one of `faces`.
fn bores_opening_on(
    topo: &Topology,
    solid: SolidId,
    faces: &HashSet<FaceId>,
) -> Result<Vec<Bore>, OperationsError> {
    use brepkit_topology::face::FaceSurface;
    // Each single-circle hole of a plane face, by its edge.
    let mut holes = HashMap::new();
    let all = solid_faces(topo, solid)?;
    for &face_id in &all {
        let face = topo.face(face_id)?;
        if !matches!(face.surface(), FaceSurface::Plane { .. }) {
            continue;
        }
        for &wire_id in face.inner_wires() {
            let wire = topo.wire(wire_id)?;
            if let [oe] = wire.edges()
                && let EdgeCurve::Circle(circle) = topo.edge(oe.edge())?.curve()
            {
                holes.insert(
                    oe.edge(),
                    (face_id, wire_id, circle.center(), circle.radius()),
                );
            }
        }
    }
    let mut bores = Vec::new();
    for &face_id in &all {
        let face = topo.face(face_id)?;
        let FaceSurface::Cylinder(cylinder) = face.surface() else {
            continue;
        };
        if !face.inner_wires().is_empty() {
            continue;
        }
        let mut circles = Vec::new();
        let mut other = false;
        for oe in topo.wire(face.outer_wire())?.edges() {
            match topo.edge(oe.edge())?.curve() {
                EdgeCurve::Circle(_) => circles.push(oe.edge()),
                EdgeCurve::Line => {}
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => other = true,
            }
        }
        let [a, b] = circles[..] else { continue };
        if other || a == b {
            continue;
        }
        let (Some(&ha), Some(&hb)) = (holes.get(&a), holes.get(&b)) else {
            continue;
        };
        if !(faces.contains(&ha.0) || faces.contains(&hb.0)) {
            continue;
        }
        let radius = cylinder.radius();
        let along = hb.2 - ha.2;
        let tol = Tolerance::new().linear * (1.0 + radius + along.length());
        if (ha.3 - radius).abs() > tol
            || (hb.3 - radius).abs() > tol
            || along.cross(cylinder.axis()).length() > tol
            || along.length() <= tol
        {
            continue;
        }
        // A pin or a hub meets its faces in circles too, with its material
        // inside its wall: a bore's wall faces its axis.
        let on_wall = topo.vertex(topo.edge(a)?.start())?.point();
        if outward_normal_at(topo, face_id, on_wall)?.is_none_or(|n| n.dot(on_wall - ha.2) >= 0.0) {
            continue;
        }
        bores.push(Bore {
            wall: face_id,
            mouths: [(ha.0, ha.1), (hb.0, hb.1)],
            centres: [ha.2, hb.2],
            radius,
        });
    }
    Ok(bores)
}

/// How far past each mouth the cut back runs, in bore radii.
const MOUTH_MARGIN: f64 = 0.1;

/// Whether the band of a round of `radius` along one of `edges` reaches
/// over one of the bore's mouths, on a face of that edge.
fn band_reaches_a_mouth(
    topo: &Topology,
    bore: &Bore,
    edges: &[EdgeId],
    faces_of: &HashMap<EdgeId, Vec<FaceId>>,
    radius: f64,
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    for (&(mouth, _), &centre) in bore.mouths.iter().zip(&bore.centres) {
        for edge in edges {
            let Some(adjacent) = faces_of.get(edge) else {
                continue;
            };
            if !adjacent.contains(&mouth) {
                continue;
            }
            let &[a, b] = adjacent.as_slice() else {
                return Ok(true);
            };
            let other = if a == mouth { b } else { a };
            let (nearest, exact) = nearest_on_edge(topo, *edge, centre)?;
            let gap = (centre - nearest).length() - bore.radius;
            if gap <= reach_over(topo, mouth, other, nearest, exact, radius)? + tol {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The point of `edge_id` nearest `point`, and whether it is exact: a
/// line's and a circle's are, an ellipse's or a spline's is found
/// numerically.
fn nearest_on_edge(
    topo: &Topology,
    edge_id: EdgeId,
    point: Point3,
) -> Result<(Point3, bool), OperationsError> {
    use brepkit_geometry::extrema::point_to_curve;
    let edge = topo.edge(edge_id)?;
    let (start, end) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
    let (lo, hi) = (t0.min(t1), t0.max(t1));
    Ok(match edge.curve() {
        EdgeCurve::Line => {
            let along = end - start;
            let length_sq = along.dot(along);
            let s = if length_sq > 0.0 {
                ((point - start).dot(along) / length_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (start + along * s, true)
        }
        EdgeCurve::Circle(circle) => {
            let t = lo + (circle.project(point) - lo).rem_euclid(std::f64::consts::TAU);
            let nearest = if t <= hi {
                circle.evaluate(t)
            } else if (point - start).length() <= (point - end).length() {
                start
            } else {
                end
            };
            (nearest, true)
        }
        EdgeCurve::Ellipse(ellipse) => (point_to_curve(point, ellipse, lo, hi).point, false),
        EdgeCurve::NurbsCurve(curve) => (point_to_curve(point, curve, lo, hi).point, false),
    })
}

/// How far a round of `radius` reaches over the plane face `mouth` from
/// the point `at` of its edge, `other` the edge's other face. The ball
/// touches the plane `r tan(b / 2)` off the edge, `b` the turn between the
/// faces' outward normals, wherever the other face's section across the
/// edge is straight: a plane, or a cylinder or cone square to the plane.
/// Where it curves, or `at` is not exact, the reach is uncertain, and taken
/// as at least four radii.
fn reach_over(
    topo: &Topology,
    mouth: FaceId,
    other: FaceId,
    at: Point3,
    exact: bool,
    radius: f64,
) -> Result<f64, OperationsError> {
    use brepkit_topology::face::FaceSurface;
    let (Some(n_mouth), Some(n_other)) = (
        outward_normal_at(topo, mouth, at)?,
        outward_normal_at(topo, other, at)?,
    ) else {
        return Ok(f64::INFINITY);
    };
    let cos = n_mouth.dot(n_other);
    let reach = if 1.0 + cos <= Tolerance::new().angular {
        f64::INFINITY
    } else {
        radius * n_mouth.cross(n_other).length() / (1.0 + cos)
    };
    let square = |axis: Vec3| axis.cross(n_mouth).length() <= Tolerance::loose().angular;
    let straight = match topo.face(other)?.surface() {
        FaceSurface::Plane { .. } => true,
        FaceSurface::Cylinder(cylinder) => square(cylinder.axis()),
        FaceSurface::Cone(cone) => square(cone.axis()),
        FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => false,
    };
    Ok(if exact && straight {
        reach
    } else {
        reach.max(4.0 * radius)
    })
}

/// The outward unit normal of `face_id` at `point`, on it.
fn outward_normal_at(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<Option<Vec3>, OperationsError> {
    let face = topo.face(face_id)?;
    let surface = face.surface();
    let (u, v) = surface.project_point(point).unwrap_or((0.0, 0.0));
    let normal = surface.normal(u, v).normalize().ok();
    Ok(normal.map(|n| if face.is_reversed() { -n } else { n }))
}

/// Points along an edge at which its convexity is read.
const CONVEXITY_SAMPLES: u32 = 8;

/// Whether the solid is convex all along `edge_id` between its faces `a`
/// and `b`: at each sample, each face runs off the edge to the material's
/// side of the other's tangent plane.
fn is_convex(
    topo: &Topology,
    edge_id: EdgeId,
    a: FaceId,
    b: FaceId,
) -> Result<bool, OperationsError> {
    let edge = topo.edge(edge_id)?;
    let (start, end) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
    let tol = Tolerance::new().angular;
    for i in 0..CONVEXITY_SAMPLES {
        let t = t0 + (t1 - t0) * (f64::from(i) + 0.5) / f64::from(CONVEXITY_SAMPLES);
        let point = edge.curve().evaluate_with_endpoints(t, start, end);
        let tangent = edge.curve().tangent_with_endpoints(t, start, end);
        let along = if t1 < t0 { -tangent } else { tangent };
        let (Some(into_a), Some(into_b), Some(out_a), Some(out_b)) = (
            into_face(topo, a, edge_id, point, along)?,
            into_face(topo, b, edge_id, point, along)?,
            outward_normal_at(topo, a, point)?,
            outward_normal_at(topo, b, point)?,
        ) else {
            return Ok(false);
        };
        if into_b.dot(out_a) >= -tol || into_a.dot(out_b) >= -tol {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The unit direction from `point` on `edge_id` into `face_id`, square to the
/// edge in the face's tangent plane, where `along` is the edge's direction
/// from its start to its end. `None` where no wire of the face runs the edge.
fn into_face(
    topo: &Topology,
    face_id: FaceId,
    edge_id: EdgeId,
    point: Point3,
    along: Vec3,
) -> Result<Option<Vec3>, OperationsError> {
    let face = topo.face(face_id)?;
    let mut forward = None;
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        if let Some(oe) = topo
            .wire(wire_id)?
            .edges()
            .iter()
            .find(|oe| oe.edge() == edge_id)
        {
            forward = Some(oe.is_forward());
            break;
        }
    }
    let Some(forward) = forward else {
        return Ok(None);
    };
    let along = if forward { along } else { -along };
    // A wire winds about its surface's normal, and a reversed face turns
    // both, so the face lies to the left of its wire about the surface
    // normal either way.
    let surface = face.surface();
    let (u, v) = surface.project_point(point).unwrap_or((0.0, 0.0));
    Ok(surface.normal(u, v).cross(along).normalize().ok())
}

/// Whether a seam of `solid` (an edge that one wire runs both ways) may pass
/// through the bore's cutter: the cut makes a hole across the seam but
/// leaves the seam whole, so it runs through the hole, and the solid is
/// valid but its face's outline crosses its hole.
fn cutter_meets_a_seam(
    topo: &Topology,
    solid: SolidId,
    bore: &Bore,
) -> Result<bool, OperationsError> {
    let cutter = Cutter::of(bore.centres, bore.radius)?;
    for face_id in solid_faces(topo, solid)? {
        let face = topo.face(face_id)?;
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            let edges = topo.wire(wire_id)?.edges();
            for (i, oe) in edges.iter().enumerate() {
                if !edges[i + 1..].iter().any(|later| later.edge() == oe.edge()) {
                    continue;
                }
                let edge = topo.edge(oe.edge())?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                if cutter.may_meet(edge.curve(), a, b) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// A bore's cutter in its own frame: a point's coordinates are its offsets
/// along `x` and `y`, across the axis, and along `axis` from `base`. The
/// cutter is the points at most `radius` from the axis with an axial
/// coordinate from 0 to `length`, each grown by `tol`.
struct Cutter {
    base: Point3,
    x: Vec3,
    y: Vec3,
    axis: Vec3,
    length: f64,
    radius: f64,
    tol: f64,
}

impl Cutter {
    /// The cutter of a bore from mouth centre to mouth centre, `MOUTH_MARGIN`
    /// radii past each.
    fn of(centres: [Point3; 2], radius: f64) -> Result<Self, OperationsError> {
        let [start, end] = centres;
        let axis = (end - start).normalize()?;
        let (x, y) = square_to(axis)?;
        let margin = MOUTH_MARGIN * radius;
        Ok(Self {
            base: start - axis * margin,
            x,
            y,
            axis,
            length: (end - start).length() + 2.0 * margin,
            radius,
            tol: Tolerance::new().linear,
        })
    }

    fn coords(&self, p: Point3) -> [f64; 3] {
        let d = p - self.base;
        [d.dot(self.x), d.dot(self.y), d.dot(self.axis)]
    }

    /// Whether a curve from `a` to `b` may meet the cutter: exactly for a
    /// segment, and by a box in the cutter's frame that holds the curve for
    /// the others.
    fn may_meet(&self, curve: &EdgeCurve, a: Point3, b: Point3) -> bool {
        let span = || {
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            (t0.min(t1), t0.max(t1))
        };
        match curve {
            EdgeCurve::Line => self.segment_meets(a, b),
            EdgeCurve::Circle(circle) => self.box_meets(self.arc_box(
                circle.center(),
                circle.u_axis() * circle.radius(),
                circle.v_axis() * circle.radius(),
                span(),
            )),
            EdgeCurve::Ellipse(ellipse) => self.box_meets(self.arc_box(
                ellipse.center(),
                ellipse.u_axis() * ellipse.semi_major(),
                ellipse.v_axis() * ellipse.semi_minor(),
                span(),
            )),
            // A curve with positive weights lies in its control points'
            // hull; with any other weight, nothing bounds it here.
            EdgeCurve::NurbsCurve(nurbs) => {
                !nurbs.weights().iter().all(|&w| w > 0.0)
                    || self.box_meets(self.points_box(nurbs.control_points()))
            }
        }
    }

    /// Whether a set lying in the box `[lo, hi]` of the cutter's frame may
    /// meet the cutter: its axial span meets the cutter's, and the box's
    /// span across the axis comes within the radius of it.
    fn box_meets(&self, (lo, hi): ([f64; 3], [f64; 3])) -> bool {
        if hi[2] < -self.tol || lo[2] > self.length + self.tol {
            return false;
        }
        let off = |k: usize| lo[k].max(-hi[k]).max(0.0);
        off(0).hypot(off(1)) <= self.radius + self.tol
    }

    fn points_box(&self, points: &[Point3]) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &p in points {
            for (k, c) in self.coords(p).into_iter().enumerate() {
                lo[k] = lo[k].min(c);
                hi[k] = hi[k].max(c);
            }
        }
        (lo, hi)
    }

    /// The box in the cutter's frame of the arc `centre + u cos t + v sin t`
    /// for `t` in `span`: each coordinate is `c + p cos t + q sin t`, which
    /// is `c + m cos(t - f)` with `m = hypot(p, q)` and `f = atan2(q, p)`, so
    /// its extremes are at the ends of the span and at each `t` in it where
    /// `t - f` is a whole number of half turns.
    fn arc_box(&self, centre: Point3, u: Vec3, v: Vec3, span: (f64, f64)) -> ([f64; 3], [f64; 3]) {
        use std::f64::consts::{PI, TAU};
        let c = self.coords(centre);
        let (t0, t1) = span;
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for (k, w) in [self.x, self.y, self.axis].into_iter().enumerate() {
            let (p, q) = (u.dot(w), v.dot(w));
            let at = |t: f64| c[k] + p * t.cos() + q * t.sin();
            let (mut low, mut high) = (at(t0).min(at(t1)), at(t0).max(at(t1)));
            let (m, f) = (p.hypot(q), q.atan2(p));
            for (turn, value) in [(f, c[k] + m), (f + PI, c[k] - m)] {
                if t0 + (turn - t0).rem_euclid(TAU) <= t1 {
                    low = low.min(value);
                    high = high.max(value);
                }
            }
            lo[k] = low;
            hi[k] = high;
        }
        (lo, hi)
    }

    /// Whether the segment from `a` to `b` meets the cutter: the part of it
    /// within the cutter's axial span, then the point of that part nearest
    /// the axis.
    fn segment_meets(&self, a: Point3, b: Point3) -> bool {
        let p = self.coords(a);
        let q = self.coords(b);
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let (low, high) = (-self.tol, self.length + self.tol);
        let (mut s0, mut s1) = (0.0_f64, 1.0_f64);
        if d[2].abs() > 0.0 {
            let (e0, e1) = ((low - p[2]) / d[2], (high - p[2]) / d[2]);
            s0 = s0.max(e0.min(e1));
            s1 = s1.min(e0.max(e1));
            if s0 > s1 {
                return false;
            }
        } else if p[2] < low || p[2] > high {
            return false;
        }
        let across = d[0] * d[0] + d[1] * d[1];
        let s = if across > 0.0 {
            (-(p[0] * d[0] + p[1] * d[1]) / across).clamp(s0, s1)
        } else {
            s0
        };
        (p[0] + d[0] * s).hypot(p[1] + d[1] * s) <= self.radius + self.tol
    }
}

/// Two unit directions square to `axis` and to each other.
fn square_to(axis: Vec3) -> Result<(Vec3, Vec3), OperationsError> {
    let x = axis
        .cross(if axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        })
        .normalize()?;
    Ok((x, axis.cross(x)))
}

/// Whether the cut back carved material past one of the bore's mouths: a
/// vertex of the cut `solid` lies in the cutter's stretch past a mouth (past
/// the mouth's plane, up to the cutter's end, within its radius). Material
/// carved there leaves the cut's boundary on the cutter's wall or end there,
/// and each edge of that boundary has a vertex, a closed edge too. A vertex
/// there that no carve made bounds material that was there already.
fn carved_past_a_mouth(
    topo: &Topology,
    solid: SolidId,
    bore: &Bore,
) -> Result<bool, OperationsError> {
    use brepkit_topology::explorer::solid_vertices;
    let [start, end] = bore.centres;
    let axis = (end - start).normalize()?;
    let margin = MOUTH_MARGIN * bore.radius;
    let linear = Tolerance::new().linear;
    for vertex_id in solid_vertices(topo, solid)? {
        let vertex = topo.vertex(vertex_id)?;
        let tol = linear.max(vertex.tolerance());
        for (centre, out) in [(start, -axis), (end, axis)] {
            let from = vertex.point() - centre;
            let past = from.dot(out);
            if past > tol
                && past <= margin + tol
                && (from - out * past).length() <= bore.radius + tol
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// A fillet over through-bores, as when the bores were filled: a round
/// running over a bore's mouth (a bracket's edge rounded past a hole near
/// it) trims the face round the hole, which the trimmer refuses, or, where
/// the hole straddles the round's contact, crosses the trimmed outline.
/// Each bore whose mouth the round's band reaches, on a face of a picked
/// edge, is filled, the solid rounded, and the bore cut back through the
/// round.
/// `None` when there is no such bore, a picked edge is not convex, a cut
/// back would cross a seam or carves material past a mouth, or the round or
/// the cut does not build exactly.
fn fillet_over_bores(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<BlendResult>, OperationsError> {
    use brepkit_topology::explorer::face_edges;
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;

    let picked: HashSet<EdgeId> = edges.iter().copied().collect();
    let mut faces_of: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for face_id in solid_faces(topo, solid)? {
        for edge in face_edges(topo, face_id)? {
            if picked.contains(&edge) {
                let faces = faces_of.entry(edge).or_default();
                if !faces.contains(&face_id) {
                    faces.push(face_id);
                }
            }
        }
    }
    let beside: HashSet<FaceId> = faces_of.values().flatten().copied().collect();
    let mut bores = Vec::new();
    for bore in bores_opening_on(topo, solid, &beside)? {
        if band_reaches_a_mouth(topo, &bore, edges, &faces_of, radius)? {
            bores.push(bore);
        }
    }
    if bores.is_empty() {
        return Ok(None);
    }
    log::debug!("fillet over bores: {} bore(s) in reach", bores.len());
    // A concave round adds material over a mouth, past where the cut back
    // ends. A convex one only takes material off, so its filled solid has
    // none in the bore past either mouth.
    for &edge in edges {
        let convex = match faces_of.get(&edge).map(Vec::as_slice) {
            Some(&[a, b]) => is_convex(topo, edge, a, b)?,
            _ => false,
        };
        if !convex {
            log::debug!("fillet over bores: edge {} is not convex", edge.index());
            return Ok(None);
        }
    }
    // The solid with the bores filled: their walls left out, their mouths
    // no longer holes.
    let walls: HashSet<FaceId> = bores.iter().map(|b| b.wall).collect();
    let mut filled: HashMap<FaceId, Vec<_>> = HashMap::new();
    for bore in &bores {
        for &(face, wire) in &bore.mouths {
            filled.entry(face).or_default().push(wire);
        }
    }
    let (outer, inner) = {
        let data = topo.solid(solid)?;
        (data.outer_shell(), data.inner_shells().to_vec())
    };
    let mut shell_faces = Vec::new();
    for face_id in topo.shell(outer)?.faces().to_vec() {
        if walls.contains(&face_id) {
            continue;
        }
        let Some(gone) = filled.get(&face_id) else {
            shell_faces.push(face_id);
            continue;
        };
        let face = topo.face(face_id)?;
        let holes: Vec<_> = face
            .inner_wires()
            .iter()
            .copied()
            .filter(|w| !gone.contains(w))
            .collect();
        let mut plain = Face::new(face.outer_wire(), holes, face.surface().clone());
        plain.set_reversed(face.is_reversed());
        shell_faces.push(topo.add_face(plain));
    }
    if shell_faces.len() + walls.len() != topo.shell(outer)?.faces().len() {
        // A bore's wall in a cavity shell: left alone.
        return Ok(None);
    }
    let shell = topo.add_shell(Shell::new(shell_faces)?);
    let plugged = topo.add_solid(Solid::new(shell, inner));

    // The trimmer drops each pcurve of an edge it splits, whatever face
    // holds it, and the builder puts back only the filled solid's. The
    // faces the filled solid leaves out share its edges, so their pcurves
    // are put back here, whatever the round comes to.
    let left_out: HashSet<FaceId> = walls.iter().chain(filled.keys()).copied().collect();
    let mut saved = Vec::new();
    for &face in &left_out {
        for (edge, pcurve) in topo.pcurves().pcurves_for_face(face) {
            saved.push((edge, face, pcurve.clone()));
        }
    }
    let mut builder = FilletBuilder::new(topo, plugged);
    builder.add_edges(edges, radius);
    let built = builder.build();
    topo.pcurves_mut().remove_faces(&left_out);
    for (edge, face, pcurve) in saved {
        topo.pcurves_mut().set(edge, face, pcurve);
    }
    let built = match built {
        Ok(built) if built.failed.is_empty() => built,
        Ok(built) => {
            log::debug!(
                "fillet over bores: the filled solid's round failed: {:?}",
                built.failed.first()
            );
            return Ok(None);
        }
        Err(e) => {
            log::debug!("fillet over bores: the filled solid's round failed: {e}");
            return Ok(None);
        }
    };
    let succeeded = built.succeeded.clone();
    let built = own_cavities(topo, plugged, built, "fillet")?;
    for bore in &bores {
        if cutter_meets_a_seam(topo, built.solid, bore)? {
            log::debug!("fillet over bores: a bore's cut back would cross a seam");
            return Ok(None);
        }
    }

    // Each bore cut back, a little past both mouths.
    let mut result = built.solid;
    for bore in &bores {
        let [start, end] = bore.centres;
        let length = (end - start).length();
        let axis = (end - start).normalize()?;
        let margin = MOUTH_MARGIN * bore.radius;
        let tool = crate::primitives::make_cylinder(topo, bore.radius, length + 2.0 * margin)?;
        let (x, y) = square_to(axis)?;
        let base = start - axis * margin;
        let frame = brepkit_math::mat::Mat4([
            [x.x(), y.x(), axis.x(), base.x()],
            [x.y(), y.y(), axis.y(), base.y()],
            [x.z(), y.z(), axis.z(), base.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        crate::transform::transform_solid(topo, tool, &frame)?;
        let opts = crate::boolean::BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..crate::boolean::BooleanOptions::default()
        };
        let (cut, repair_declined) = crate::boolean::without_repair_unify(|| {
            crate::boolean::boolean_with_options(
                topo,
                crate::boolean::BooleanOp::Cut,
                result,
                tool,
                opts,
            )
        });
        match cut {
            Ok(cut) if !repair_declined => result = cut,
            Ok(_) => {
                log::debug!("fillet over bores: the bore's cut needs its faces unified");
                return Ok(None);
            }
            Err(e) => {
                log::debug!("fillet over bores: the bore's cut failed: {e}");
                return Ok(None);
            }
        }
    }
    for bore in &bores {
        if carved_past_a_mouth(topo, result, bore)? {
            log::debug!("fillet over bores: the cut back carved material past a mouth");
            return Ok(None);
        }
    }
    // A cut that leaves the solid unsound (faces disagreeing about an edge's
    // sense where the bore runs through a band) is not taken; the round is
    // then made as it is.
    if !crate::validate::validate_solid(topo, result)?.is_valid() {
        log::debug!("fillet over bores: the cut back is not a valid solid");
        return Ok(None);
    }
    Ok(Some(BlendResult {
        solid: result,
        succeeded,
        failed: Vec::new(),
        is_partial: false,
    }))
}

/// How the corner where three rounds meet is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerKind {
    /// A ball rolled between the three faces, unless the corner is set back.
    Rolling,
    /// A patch curving like each round where it meets it, which needs room: with
    /// no set-back given, each round stops one radius past where the ball stops it.
    G2,
}

/// Corner options for one vertex where three rounds meet.
#[derive(Debug, Clone)]
pub struct FilletCorner {
    /// The vertex the rounds meet at.
    pub vertex: VertexId,
    /// How far from the vertex each meeting edge's round stops. Either none or
    /// one for each of the three edges. A corner set back is always closed by the
    /// patch [`CornerKind::G2`] names, since no ball fits between its rounds.
    pub setbacks: Vec<(EdgeId, f64)>,
    /// How the corner is closed.
    pub kind: CornerKind,
}

fn other_end_direction(
    topo: &Topology,
    edge: EdgeId,
    vertex: VertexId,
) -> Result<Vec3, OperationsError> {
    let data = topo.edge(edge)?;
    if !matches!(data.curve(), EdgeCurve::Line) {
        return Err(OperationsError::InvalidInput {
            reason: "corner options need straight edges meeting at the vertex".into(),
        });
    }
    let far = if data.start() == vertex {
        data.end()
    } else {
        data.start()
    };
    Ok((topo.vertex(far)?.point() - topo.vertex(vertex)?.point()).normalize()?)
}

/// A corner's position and each meeting edge's direction and given set-back.
type CornerPlan = (Point3, [Vec3; 3], Option<[f64; 3]>);

fn plan_corner(
    topo: &Topology,
    edges: &[EdgeId],
    corner: &FilletCorner,
) -> Result<CornerPlan, OperationsError> {
    let mut meeting = Vec::new();
    for &edge in edges {
        let data = topo.edge(edge)?;
        if data.start() == corner.vertex || data.end() == corner.vertex {
            meeting.push(edge);
        }
    }
    let [a, b, c] = meeting.as_slice() else {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "corner options need three rounds meeting at the vertex, and {} do",
                meeting.len()
            ),
        });
    };
    let meeting = [*a, *b, *c];
    let mut directions = [Vec3::new(0.0, 0.0, 0.0); 3];
    for (direction, &edge) in directions.iter_mut().zip(&meeting) {
        *direction = other_end_direction(topo, edge, corner.vertex)?;
    }
    let distances = if corner.setbacks.is_empty() {
        None
    } else {
        if corner.setbacks.len() != 3 {
            return Err(OperationsError::InvalidInput {
                reason: format!(
                    "a corner takes a set-back for each of its three edges, not {}",
                    corner.setbacks.len()
                ),
            });
        }
        let mut distances = [0.0; 3];
        for (slot, edge) in distances.iter_mut().zip(meeting) {
            let &(_, distance) = corner
                .setbacks
                .iter()
                .find(|(given, _)| *given == edge)
                .ok_or_else(|| OperationsError::InvalidInput {
                    reason: "a corner's set-backs must name each edge meeting there".into(),
                })?;
            *slot = distance;
        }
        Some(distances)
    };
    Ok((topo.vertex(corner.vertex)?.point(), directions, distances))
}

/// Fillet edges with constant radius, with options at the corners where three
/// of them meet: how far back along each edge its round stops, and what closes
/// the hole that leaves.
///
/// A set-back corner is two faces meeting along a seam, tangent to each other,
/// to the three rounds and to the three flat faces, with the rounds' own
/// curvature where they meet it. Its rounds and flat faces keep their exact
/// surfaces, and each flat face gains a link between its two rounds' contact
/// lines, tangent to both.
///
/// # Errors
/// Returns `OperationsError` if the fillet fails, if a corner is not three
/// straight rounds rolled between flat faces, if a set-back is not past where
/// the ball stops its round or not short of its edge's far end, or if the
/// result does not validate.
pub fn fillet_v2_with_corners(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
    corners: &[FilletCorner],
) -> Result<SolidId, OperationsError> {
    let mut plans = Vec::with_capacity(corners.len());
    for corner in corners {
        plans.push((corner.kind, plan_corner(topo, edges, corner)?));
    }
    let mut result = all_blended(fillet_v2(topo, solid, edges, radius)?)?;
    for (kind, (at, directions, given)) in plans {
        let rolling =
            brepkit_blend::setback_corner::rolling_distances(topo, result, at, &directions)?
                .ok_or_else(|| OperationsError::InvalidInput {
                    reason: "corner options need a ball corner between three flat faces".into(),
                })?;
        let distances = match (given, kind) {
            (Some(distances), _) => distances,
            (None, CornerKind::Rolling) => continue,
            (None, CornerKind::G2) => rolling.map(|distance| distance + radius),
        };
        let setbacks = [
            (directions[0], distances[0]),
            (directions[1], distances[1]),
            (directions[2], distances[2]),
        ];
        let before: HashSet<FaceId> = solid_faces(topo, result)?.into_iter().collect();
        let (set_back, reach) =
            brepkit_blend::setback_corner::set_back_corner(topo, result, at, &setbacks)?;
        result = set_back;
        // The patches take off more than the fillet's own check of the cavities
        // saw. The rounds and flat faces rebuilt around them reach far past what
        // changed, so the cavities are read against `reach` alone.
        let made: Vec<FaceId> = solid_faces(topo, result)?
            .into_iter()
            .filter(|face| !before.contains(face))
            .collect();
        crate::cavities::clear_of_own_cavities_within(
            topo,
            result,
            &made,
            &[reach],
            "fillet corner",
        )?;
    }
    if !crate::validate::validate_solid(topo, result)?.is_valid() {
        return Err(OperationsError::NonManifoldResult);
    }
    Ok(result)
}

/// Chamfer edges with two distances (v2 engine), `d1` on each edge's first
/// face in the solid's face order and `d2` on its other face.
///
/// A pick of closed circular rims, or of chains of lines and arcs (see
/// [`crate::chamfer_chain::chamfer_tangent_chains`]), is cut exactly by
/// those paths first; any other pick goes to the walking engine.
///
/// # Errors
/// Returns `OperationsError` if distances are non-positive, edges are empty,
/// a rim's distance consumes its cap or wall, or the blend computation
/// fails.
pub fn chamfer_v2(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    d1: f64,
    d2: f64,
) -> Result<BlendResult, OperationsError> {
    if d1 <= 0.0 || d2 <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: "distances must be positive".into(),
        });
    }
    if edges.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    if let Some(pairs) = on_first_faces(topo, solid, edges)?
        && let Some(result) = crate::chamfer::chamfer_exactly_on(topo, solid, &pairs, d1, d2)?
    {
        return Ok(chamfered_whole(result, &pairs));
    }
    chamfer_v2_walking(topo, solid, edges, d1, d2)
}

/// [`chamfer_v2`] by the walking engine alone.
pub(crate) fn chamfer_v2_walking(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    d1: f64,
    d2: f64,
) -> Result<BlendResult, OperationsError> {
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_asymmetric(edges, d1, d2);
    let built = builder.build()?;
    valid_chamfer(topo, solid, built)
}

/// Chamfer edges with two distances (v2 engine), `d1` on each edge's
/// reference face and `d2` on its other face.
///
/// Rims and chains of lines and arcs are cut exactly first, as in
/// [`chamfer_v2`].
///
/// # Errors
/// Returns `OperationsError` if distances are non-positive, edges are empty,
/// an edge has two different reference faces, a reference face does not
/// border its edge, a rim's distance consumes its cap or wall, or the blend
/// computation fails.
pub fn chamfer_v2_on(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[(EdgeId, FaceId)],
    d1: f64,
    d2: f64,
) -> Result<BlendResult, OperationsError> {
    if d1 <= 0.0 || d2 <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: "distances must be positive".into(),
        });
    }
    if edges.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    let mut pairs: Vec<(EdgeId, FaceId)> = Vec::with_capacity(edges.len());
    for &(edge, face) in edges {
        match pairs.iter().find(|&&(e, _)| e == edge) {
            Some(&(_, other)) if other != face => {
                return Err(OperationsError::InvalidInput {
                    reason: format!("chamfer edge {edge:?} has two reference faces"),
                });
            }
            Some(_) => {}
            None => pairs.push((edge, face)),
        }
    }
    // A reference off its edge, or an edge off the outer shell, goes to the
    // walking engine, which refuses it.
    let adjacency = topo.build_adjacency(solid)?;
    let bordered = pairs.iter().all(|&(edge, face)| {
        let around = adjacency.faces_for_edge(edge);
        around.len() == 2 && around.contains(&face)
    });
    if bordered
        && let Some(result) = crate::chamfer::chamfer_exactly_on(topo, solid, &pairs, d1, d2)?
    {
        return Ok(chamfered_whole(result, &pairs));
    }
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_asymmetric_on(edges, d1, d2);
    let built = builder.build()?;
    valid_chamfer(topo, solid, built)
}

/// Chamfer edges with distance and angle (v2 engine): `distance` on each
/// edge's first face in the solid's face order, and `distance * tan(angle)`
/// on its other face, as [`chamfer_v2`] takes them.
///
/// # Errors
/// Returns `OperationsError` if distance is non-positive, angle is out of
/// range (0, pi/2), edges are empty, a rim's distance consumes its cap or
/// wall, or the blend computation fails.
pub fn chamfer_distance_angle(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    distance: f64,
    angle: f64,
) -> Result<BlendResult, OperationsError> {
    if distance <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: "distance must be positive".into(),
        });
    }
    if angle <= 0.0 || angle >= std::f64::consts::FRAC_PI_2 {
        return Err(OperationsError::InvalidInput {
            reason: "angle must be between 0 and \u{03c0}/2".into(),
        });
    }
    if edges.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    let other = distance * angle.tan();
    if let Some(pairs) = on_first_faces(topo, solid, edges)?
        && let Some(result) =
            crate::chamfer::chamfer_exactly_on(topo, solid, &pairs, distance, other)?
    {
        return Ok(chamfered_whole(result, &pairs));
    }
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_distance_angle(edges, distance, angle);
    let built = builder.build()?;
    valid_chamfer(topo, solid, built)
}

/// Each edge, once, with its first face in the outer shell's face order,
/// the face the walking chamfer sets `d1` on. `None` when an edge does not
/// lie between two faces of the outer shell.
fn on_first_faces(
    topo: &Topology,
    solid: SolidId,
    edges: &[EdgeId],
) -> Result<Option<Vec<(EdgeId, FaceId)>>, OperationsError> {
    let adjacency = topo.build_adjacency(solid)?;
    let mut pairs: Vec<(EdgeId, FaceId)> = Vec::with_capacity(edges.len());
    for &edge in edges {
        if pairs.iter().any(|&(e, _)| e == edge) {
            continue;
        }
        let [first, _] = adjacency.faces_for_edge(edge) else {
            return Ok(None);
        };
        pairs.push((edge, *first));
    }
    Ok(Some(pairs))
}

/// An exact chamfer of every edge in `pairs` as a blend result.
fn chamfered_whole(solid: SolidId, pairs: &[(EdgeId, FaceId)]) -> BlendResult {
    BlendResult {
        solid,
        succeeded: pairs.iter().map(|&(e, _)| e).collect(),
        failed: Vec::new(),
        is_partial: false,
    }
}

/// Fillet edges with per-edge radius laws (v2 walking engine).
///
/// # Errors
/// Returns `OperationsError` if `edge_laws` is empty or the blend
/// computation fails.
pub fn fillet_v2_variable(
    topo: &mut Topology,
    solid: SolidId,
    edge_laws: Vec<(EdgeId, brepkit_blend::radius_law::RadiusLaw)>,
) -> Result<BlendResult, OperationsError> {
    if edge_laws.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    // Group tangent-continuous runs with an identical constant radius into
    // one edge set. The builder still computes per-edge stripes inside a
    // set, so grouping does not remove in-run junctions by itself; it keeps
    // each run under one radius law and gives downstream corner handling a
    // consistent picture of which junctions are true radius-change or
    // angular corners.
    let mut remaining: Vec<(EdgeId, brepkit_blend::radius_law::RadiusLaw)> = edge_laws;
    let mut chains: Vec<(Vec<EdgeId>, brepkit_blend::radius_law::RadiusLaw)> = Vec::new();
    while let Some((seed, law)) = remaining.pop() {
        let seed_r = match law {
            brepkit_blend::radius_law::RadiusLaw::Constant(r) => Some(r),
            _ => None,
        };
        let mut chain = vec![seed];
        if let Some(r) = seed_r {
            loop {
                let mut grew = false;
                let mut i = 0;
                while i < remaining.len() {
                    let brepkit_blend::radius_law::RadiusLaw::Constant(cand_r) = remaining[i].1
                    else {
                        i += 1;
                        continue;
                    };
                    if (cand_r - r).abs() > 1e-9 {
                        i += 1;
                        continue;
                    }
                    if chain_extends_tangently(topo, &chain, remaining[i].0) {
                        chain.push(remaining[i].0);
                        remaining.swap_remove(i);
                        grew = true;
                    } else {
                        i += 1;
                    }
                }
                if !grew {
                    break;
                }
            }
        }
        chains.push((chain, law));
    }
    let mut builder = FilletBuilder::new(topo, solid);
    for (chain, law) in chains {
        builder.add_edges_with_law(&chain, law);
    }
    let built = builder.build()?;
    let result = own_cavities(topo, solid, built, "fillet")?;
    valid_fillet(topo, solid, result)
}

/// Whether `cand` shares an endpoint with some chain edge and meets it
/// tangentially (G1) there.
fn chain_extends_tangently(topo: &Topology, chain: &[EdgeId], cand: EdgeId) -> bool {
    use brepkit_math::vec::{Point3, Vec3};
    let ends = |eid: EdgeId| -> Option<(Point3, Point3, brepkit_topology::edge::EdgeCurve)> {
        let e = topo.edge(eid).ok()?;
        Some((
            topo.vertex(e.start()).ok()?.point(),
            topo.vertex(e.end()).ok()?.point(),
            e.curve().clone(),
        ))
    };
    let Some((cs, ce, cc)) = ends(cand) else {
        return false;
    };
    let tangent_at = |curve: &brepkit_topology::edge::EdgeCurve,
                      s: Point3,
                      e: Point3,
                      at_start: bool|
     -> Option<Vec3> {
        curve
            .tangent_with_endpoints(if at_start { 0.0 } else { 1.0 }, s, e)
            .normalize()
            .ok()
    };
    for &eid in chain {
        let Some((s, e, curve)) = ends(eid) else {
            continue;
        };
        for (cp, c_at_start) in [(cs, true), (ce, false)] {
            for (p, at_start) in [(s, true), (e, false)] {
                if (cp - p).length() > 1e-6 {
                    continue;
                }
                let (Some(t_chain), Some(t_cand)) = (
                    tangent_at(&curve, s, e, at_start),
                    tangent_at(&cc, cs, ce, c_at_start),
                ) else {
                    continue;
                };
                // Traversal directions at a shared vertex oppose when the
                // curves continue smoothly through it exactly when one is
                // at its start and the other at its end.
                let aligned = if at_start == c_at_start {
                    -t_chain.dot(t_cand)
                } else {
                    t_chain.dot(t_cand)
                };
                if aligned > 0.999 {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::vertex::Vertex;

    use super::{Cutter, nearest_on_edge};

    fn edge(topo: &mut Topology, a: Point3, b: Point3, curve: EdgeCurve) -> EdgeId {
        let a = topo.add_vertex(Vertex::new(a, 1e-7));
        let b = if b == topo.vertex(a).unwrap().point() {
            a
        } else {
            topo.add_vertex(Vertex::new(b, 1e-7))
        };
        topo.add_edge(Edge::new(a, b, curve))
    }

    fn assert_nearest(topo: &Topology, edge: EdgeId, point: [f64; 2], want: [f64; 2]) {
        let (nearest, exact) =
            nearest_on_edge(topo, edge, Point3::new(point[0], point[1], 0.0)).unwrap();
        assert!(exact);
        let want = Point3::new(want[0], want[1], 0.0);
        assert!(
            (nearest - want).length() < 1e-12,
            "{point:?}: {nearest:?}, want {want:?}"
        );
    }

    /// A line's nearest point is clamped to its ends, and an arc's, past
    /// its span, is the nearer of its ends.
    #[test]
    fn the_nearest_point_of_an_edge_lies_on_the_edge() {
        let mut topo = Topology::new();
        let o = Point3::new(0.0, 0.0, 0.0);
        let line = edge(&mut topo, o, Point3::new(10.0, 0.0, 0.0), EdgeCurve::Line);
        assert_nearest(&topo, line, [4.0, 3.0], [4.0, 0.0]);
        assert_nearest(&topo, line, [12.0, 3.0], [10.0, 0.0]);

        let z = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new_with_ref(o, z, 5.0, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let rim = Point3::new(5.0, 0.0, 0.0);
        let closed = edge(&mut topo, rim, rim, EdgeCurve::Circle(circle.clone()));
        assert_nearest(&topo, closed, [0.0, -2.0], [0.0, -5.0]);

        let top = Point3::new(0.0, 5.0, 0.0);
        let arc = edge(&mut topo, rim, top, EdgeCurve::Circle(circle));
        let diagonal = 5.0 * std::f64::consts::FRAC_1_SQRT_2;
        assert_nearest(&topo, arc, [1.0, 1.0], [diagonal, diagonal]);
        assert_nearest(&topo, arc, [0.0, -2.0], [5.0, 0.0]);
        assert_nearest(&topo, arc, [-2.0, 0.5], [0.0, 5.0]);
    }

    fn p(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }

    /// A cutter of radius 0.5 standing on (0, 10) from z = -0.05 to 10.05.
    fn upright_cutter() -> Cutter {
        Cutter::of([p(0.0, 10.0, 0.0), p(0.0, 10.0, 10.0)], 0.5).unwrap()
    }

    /// A polyline spline along y = 0 at z = 5 from x = -32 to 32, with a
    /// spike out to `tip` over the parameters 0.501 to 0.503.
    fn spiked(tip: Point3) -> (EdgeCurve, Point3, Point3) {
        let (a, b) = (p(-32.0, 0.0, 5.0), p(32.0, 0.0, 5.0));
        let points = vec![a, p(-0.1, 0.0, 5.0), tip, p(0.1, 0.0, 5.0), b];
        let knots = vec![0.0, 0.0, 0.501, 0.502, 0.503, 1.0, 1.0];
        let curve = NurbsCurve::new(1, knots, points, vec![1.0; 5]).unwrap();
        (EdgeCurve::NurbsCurve(curve), a, b)
    }

    /// The spike reaches the cutter's axis between two of 65 even samples,
    /// each farther from the axis than the radius and the longest chord
    /// between samples together. The spline's control points bound it, so
    /// it is found; a spike that stops short of the cutter is not.
    #[test]
    fn a_spline_seam_through_the_cutter_between_samples_meets_it() {
        let cutter = upright_cutter();
        let (seam, a, b) = spiked(p(0.0, 10.0, 5.0));
        let (t0, t1) = seam.domain_with_endpoints(a, b);
        let samples: Vec<Point3> = (0..=64)
            .map(|k| seam.evaluate_with_endpoints(t0 + (t1 - t0) * f64::from(k) / 64.0, a, b))
            .collect();
        let chord = samples
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).length())
            .fold(0.0, f64::max);
        let off_axis = |q: Point3| q.x().hypot(q.y() - 10.0);
        assert!(samples.iter().all(|&q| off_axis(q) > 0.5 + chord));
        assert!(cutter.may_meet(&seam, a, b));

        let (seam, a, b) = spiked(p(0.0, 5.0, 5.0));
        assert!(!cutter.may_meet(&seam, a, b));
    }

    /// A segment meets the cutter exactly where some point of it lies within
    /// the radius of the axis inside the cutter's length.
    #[test]
    fn a_line_seam_meets_the_cutter_exactly() {
        let cutter = upright_cutter();
        let line = EdgeCurve::Line;
        assert!(cutter.may_meet(&line, p(-5.0, 10.0, 5.0), p(5.0, 10.0, 5.0)));
        assert!(!cutter.may_meet(&line, p(-5.0, 10.6, 2.0), p(5.0, 10.6, 8.0)));
        assert!(cutter.may_meet(&line, p(-5.0, 10.4, 2.0), p(5.0, 10.4, 8.0)));
        assert!(!cutter.may_meet(&line, p(-5.0, 10.0, 11.0), p(5.0, 10.0, 12.0)));
        assert!(cutter.may_meet(&line, p(0.0, 10.0, 12.0), p(0.0, 10.3, -3.0)));
    }

    /// A rod's top rim rounded by 2: the band's seam is the quarter circle
    /// about (3, 0, 4) from (5, 0, 4) to (3, 0, 6). A cutter on (4.2, 0)
    /// crosses it, one a radian round the rod does not, and one on (1.8, 0)
    /// meets the whole circle but not the quarter.
    #[test]
    fn a_circle_seam_meets_the_cutter_within_its_span() {
        let rim = Circle3D::new(p(3.0, 0.0, 4.0), Vec3::new(0.0, 1.0, 0.0), 2.0).unwrap();
        let (a, b) = (p(5.0, 0.0, 4.0), p(3.0, 0.0, 6.0));
        let quarter =
            if (rim.evaluate(rim.project(a) + std::f64::consts::FRAC_PI_2) - b).length() < 1e-9 {
                (EdgeCurve::Circle(rim), a, b)
            } else {
                (EdgeCurve::Circle(rim), b, a)
            };
        let cutter = |x: f64, y: f64| Cutter::of([p(x, y, 0.0), p(x, y, 6.0)], 0.3).unwrap();
        let (seam, a, b) = quarter;
        assert!(cutter(4.2, 0.0).may_meet(&seam, a, b));
        assert!(!cutter(4.2 * 1.0_f64.cos(), 4.2 * 1.0_f64.sin()).may_meet(&seam, a, b));
        assert!(!cutter(1.8, 0.0).may_meet(&seam, a, b));
    }

    /// A closed ellipse in the plane x = 0 about a point on the cutter's axis
    /// crosses the axis, and one in z = 5 whose centre is 4 off the axis, its
    /// semi-major axis 3, stays clear.
    #[test]
    fn an_ellipse_seam_meets_the_cutter_by_its_box() {
        let cutter = upright_cutter();
        let across = Ellipse3D::new(p(0.0, 10.0, 5.0), Vec3::new(1.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let at = across.evaluate(0.0);
        assert!(cutter.may_meet(&EdgeCurve::Ellipse(across), at, at));
        let beside = Ellipse3D::new(p(0.0, 14.0, 5.0), Vec3::new(0.0, 0.0, 1.0), 3.0, 1.0).unwrap();
        let at = beside.evaluate(0.0);
        assert!(!cutter.may_meet(&EdgeCurve::Ellipse(beside), at, at));
    }

    /// A cylinder of radius 3 and height 4 with its wall in its B-spline
    /// form, and its bottom rim chamfered 0.4 and 0.25 by the builder.
    fn chamfered_b_spline_rim(topo: &mut Topology) -> (super::SolidId, super::BlendResult) {
        let solid = crate::primitives::make_cylinder(topo, 3.0, 4.0).unwrap();
        let rim = brepkit_topology::explorer::solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
            })
            .unwrap();
        for face in super::solid_faces(topo, solid).unwrap() {
            if topo.face(face).unwrap().surface().is_planar() {
                continue;
            }
            let wall = brepkit_heal::custom::convert_to_bspline::face_bspline_surface(topo, face)
                .unwrap()
                .unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(brepkit_topology::face::FaceSurface::Nurbs(wall));
        }
        let mut builder = super::ChamferBuilder::new(topo, solid);
        builder.add_edges_asymmetric(&[rim], 0.4, 0.25);
        let built = builder.build().unwrap();
        assert!(built.failed.is_empty(), "{:?}", built.failed);
        (solid, built)
    }

    /// A chamfer's contact whose pcurve on the B-spline wall runs from u = 1
    /// to 2 on the knot domain 0 to 1, a whole turn off: read through the
    /// wall, it strays far from its edge. The solid checks pass the result,
    /// and the chamfer's gate refuses it. The result as built passes both.
    #[test]
    fn a_chamfer_whose_pcurve_leaves_its_edge_is_refused() {
        use brepkit_math::curves2d::{Curve2D, Line2D, NurbsCurve2D};
        use brepkit_math::vec::Vec2;
        use brepkit_topology::face::FaceSurface;
        use brepkit_topology::pcurve::PCurve;

        let mut topo = Topology::new();
        let (solid, built) = chamfered_b_spline_rim(&mut topo);
        super::valid_chamfer(&mut topo, solid, built).unwrap();

        let (solid, built) = chamfered_b_spline_rim(&mut topo);
        let old: std::collections::HashSet<EdgeId> =
            brepkit_topology::explorer::solid_edges(&topo, solid)
                .unwrap()
                .into_iter()
                .collect();
        let mut moved = None;
        'faces: for face in super::solid_faces(&topo, built.solid).unwrap() {
            let f = topo.face(face).unwrap();
            let FaceSurface::Nurbs(wall) = f.surface() else {
                continue;
            };
            if !wall.is_periodic_u() || wall.degree_v() != 1 {
                continue;
            }
            let (u0, u1) = wall.domain_u();
            let turn = Vec2::new(u1 - u0, 0.0);
            for oe in topo.wire(f.outer_wire()).unwrap().edges() {
                let Some(pcurve) = topo.pcurves().get(oe.edge(), face) else {
                    continue;
                };
                if old.contains(&oe.edge()) {
                    continue;
                }
                let curve = match pcurve.curve() {
                    Curve2D::Line(line) => {
                        Curve2D::Line(Line2D::new(line.origin() + turn, line.direction()).unwrap())
                    }
                    Curve2D::Nurbs(nurbs) => Curve2D::Nurbs(
                        NurbsCurve2D::new(
                            nurbs.degree(),
                            nurbs.knots().to_vec(),
                            nurbs.control_points().iter().map(|&q| q + turn).collect(),
                            nurbs.weights().to_vec(),
                        )
                        .unwrap(),
                    ),
                    Curve2D::Circle(_) | Curve2D::Ellipse(_) => continue,
                };
                moved = Some((
                    oe.edge(),
                    face,
                    PCurve::new(curve, pcurve.t_start(), pcurve.t_end()),
                ));
                break 'faces;
            }
        }
        let (edge, face, pcurve) = moved.expect("a contact's pcurve on the wall");
        topo.pcurves_mut().set(edge, face, pcurve);
        assert!(
            crate::validate::validate_solid(&topo, built.solid)
                .unwrap()
                .is_valid()
        );
        let error = super::valid_chamfer(&mut topo, solid, built)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("pcurve"), "the gate gave {error:?}");
    }
}
