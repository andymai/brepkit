//! Thin wrappers around `brepkit-blend` for the operations API.

use std::collections::HashSet;

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
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

/// Fillet edges with constant radius — the production engine.
///
/// Runs the corrected [`FilletBuilder`]: one ordered contour per G1 run,
/// canonical boundary-registry ownership, ordered-fan junction corners, and
/// closed periodic bands for circular rims. This is the engine the WASM
/// dispatch prefers; the deprecated `fillet_rolling_ball` remains callable
/// only as an explicit reference implementation.
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
    if let Ok(Some(result)) = fillet_over_bores(topo, solid, edges, radius) {
        return Ok(result);
    }
    let mut builder = FilletBuilder::new(topo, solid);
    builder.add_edges(edges, radius);
    let built = builder.build()?;
    own_cavities(topo, solid, built, "fillet")
}

/// A straight through-bore: a cylinder wall from a circle that is a hole in
/// one plane face to a circle that is a hole in another.
struct Bore {
    wall: FaceId,
    mouths: [(FaceId, brepkit_topology::wire::WireId); 2],
    centres: [Point3; 2],
    radius: f64,
}

/// The through-bores opening on `faces` whose mouth comes within `reach`
/// of a point of `near`.
fn bores_opening_on(
    topo: &Topology,
    solid: SolidId,
    faces: &HashSet<FaceId>,
    near: &[Point3],
    reach: f64,
) -> Result<Vec<Bore>, OperationsError> {
    use brepkit_topology::face::FaceSurface;
    // Each single-circle hole of a plane face, by its edge.
    let mut holes = std::collections::HashMap::new();
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
                _ => other = true,
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
        let tol = 1e-7 * (1.0 + radius + along.length());
        if (ha.3 - radius).abs() > tol
            || (hb.3 - radius).abs() > tol
            || along.cross(cylinder.axis()).length() > tol
            || along.length() <= tol
        {
            continue;
        }
        let gap = |centre: Point3| {
            near.iter()
                .map(|&p| (p - centre).length())
                .fold(f64::INFINITY, f64::min)
                - radius
        };
        if gap(ha.2).min(gap(hb.2)) > reach {
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

/// A fillet over through-bores, as when the bores were filled: a round
/// running over a bore's mouth (a bracket's edge rounded past a hole near
/// it) trims the face round the hole, which the trimmer refuses, or, where
/// the hole straddles the round's contact, crosses the trimmed outline.
/// Each bore opening within the round's reach on a face of the picked edges
/// is filled, the solid rounded, and the bore cut back through the round.
/// `None` when there is no such bore, or the round or the cut does not
/// build exactly.
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
    let mut beside = HashSet::new();
    for face_id in solid_faces(topo, solid)? {
        if face_edges(topo, face_id)?
            .iter()
            .any(|e| picked.contains(e))
        {
            beside.insert(face_id);
        }
    }
    // Points along the picked edges: a round reaches a few radii from its
    // edge at most (r cot(a / 2) on a face at an acute corner a).
    let mut near = Vec::new();
    for &edge_id in edges {
        let edge = topo.edge(edge_id)?;
        let (start, end) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        for i in 0..=64 {
            let t = t0 + (t1 - t0) * f64::from(i) / 64.0;
            near.push(edge.curve().evaluate_with_endpoints(t, start, end));
        }
    }
    let bores = bores_opening_on(topo, solid, &beside, &near, 4.0 * radius)?;
    if bores.is_empty() {
        return Ok(None);
    }
    log::debug!("fillet over bores: {} bore(s) in reach", bores.len());
    // The solid with the bores filled: their walls left out, their mouths
    // no longer holes.
    let walls: HashSet<FaceId> = bores.iter().map(|b| b.wall).collect();
    let mut filled: std::collections::HashMap<FaceId, Vec<_>> = std::collections::HashMap::new();
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

    let mut builder = FilletBuilder::new(topo, plugged);
    builder.add_edges(edges, radius);
    let built = match builder.build() {
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

    // Each bore cut back, a little past both mouths (outside the solid
    // there, as the bore opens on both).
    let mut result = built.solid;
    for bore in &bores {
        let [start, end] = bore.centres;
        let along = end - start;
        let length = along.length();
        let axis = along * (1.0 / length);
        let margin = 0.1 * bore.radius;
        let tool = crate::primitives::make_cylinder(topo, bore.radius, length + 2.0 * margin)?;
        let x = axis.cross(if axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        });
        let x = x * (1.0 / x.length());
        let y = axis.cross(x);
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
            ..crate::boolean::BooleanOptions::default()
        };
        match crate::boolean::boolean_with_options(
            topo,
            crate::boolean::BooleanOp::Cut,
            result,
            tool,
            opts,
        ) {
            Ok(cut) => result = cut,
            Err(e) => {
                log::debug!("fillet over bores: the bore's cut failed: {e}");
                return Ok(None);
            }
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

/// Chamfer edges with two distances (v2 engine).
///
/// # Errors
/// Returns `OperationsError` if distances are non-positive, edges are empty,
/// or the blend computation fails.
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
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_asymmetric(edges, d1, d2);
    let built = builder.build()?;
    own_cavities(topo, solid, built, "chamfer")
}

/// Chamfer edges with two distances (v2 engine), `d1` on each edge's
/// reference face and `d2` on its other face.
///
/// # Errors
/// Returns `OperationsError` if distances are non-positive, edges are empty,
/// a reference face does not border its edge, or the blend computation
/// fails.
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
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_asymmetric_on(edges, d1, d2);
    let built = builder.build()?;
    own_cavities(topo, solid, built, "chamfer")
}

/// Chamfer edges with distance and angle (v2 engine).
///
/// # Errors
/// Returns `OperationsError` if distance is non-positive, angle is out of
/// range (0, pi/2), edges are empty, or the blend computation fails.
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
    let mut builder = ChamferBuilder::new(topo, solid);
    builder.add_edges_distance_angle(edges, distance, angle);
    let built = builder.build()?;
    own_cavities(topo, solid, built, "chamfer")
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
    own_cavities(topo, solid, built, "fillet")
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
