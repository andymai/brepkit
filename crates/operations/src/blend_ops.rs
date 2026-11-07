//! Thin wrappers around `brepkit-blend` for the operations API.

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
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
    let mut builder = FilletBuilder::new(topo, solid);
    builder.add_edges(edges, radius);
    let built = builder.build()?;
    own_cavities(topo, solid, built, "fillet")
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
        let distances = match (given, kind) {
            (Some(distances), _) => distances,
            (None, CornerKind::Rolling) => continue,
            (None, CornerKind::G2) => {
                brepkit_blend::setback_corner::rolling_distances(topo, result, at, &directions)?
                    .ok_or_else(|| OperationsError::InvalidInput {
                        reason: "corner options need a ball corner between three flat faces".into(),
                    })?
                    .map(|rolling| rolling + radius)
            }
        };
        let setbacks = [
            (directions[0], distances[0]),
            (directions[1], distances[1]),
            (directions[2], distances[2]),
        ];
        result = brepkit_blend::setback_corner::set_back_corner(topo, result, at, &setbacks)?;
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
