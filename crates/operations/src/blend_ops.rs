//! Thin wrappers around `brepkit-blend` for the operations API.

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::{Solid, SolidId};

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

/// `result` with the cavities of `solid` as its own. The walking builders
/// rebuild only the outer shell and carry the cavity shells over as they
/// are, so a blend reaching a cavity is refused, and each kept cavity is
/// copied: a shell shared with the input would move with it under an
/// in-place transform of either.
fn own_cavities(
    topo: &mut Topology,
    solid: SolidId,
    mut result: BlendResult,
    what: &str,
) -> Result<BlendResult, OperationsError> {
    if result.solid == solid || topo.solid(solid)?.inner_shells().is_empty() {
        return Ok(result);
    }
    let outer = topo.solid(result.solid)?.outer_shell();
    let bare = topo.add_solid(Solid::new(outer, Vec::new()));
    result.solid = crate::cavities::keep_cavities(topo, solid, bare, what)?;
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    /// A box's top front edge, 40 long, filleted at a radius running 2 to 5:
    /// the round stays inside the box, rolling against the two faces' insides,
    /// and the solid loses `(1 − π/4) ∫ r²` along the edge.
    #[test]
    fn a_variable_fillet_rounds_a_convex_edge_inside_it() {
        use brepkit_blend::radius_law::RadiusLaw;

        let mut topo = brepkit_topology::Topology::new();
        let block = crate::primitives::make_box(&mut topo, 40.0, 30.0, 20.0).unwrap();
        let edge = brepkit_topology::explorer::solid_edges(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                [edge.start(), edge.end()].iter().all(|&v| {
                    let p = topo.vertex(v).unwrap().point();
                    p.y().abs() < 1e-9 && (p.z() - 20.0).abs() < 1e-9
                })
            })
            .unwrap();
        let law = RadiusLaw::Linear {
            start: 2.0,
            end: 5.0,
        };
        let rounded = super::all_blended(
            super::fillet_v2_variable(&mut topo, block, vec![(edge, law)]).unwrap(),
        )
        .unwrap();

        let bounds = crate::measure::solid_bounding_box(&topo, rounded).unwrap();
        assert!(
            bounds.min.y() > -1e-9 && bounds.max.z() < 20.0 + 1e-9,
            "the round leaves the box: {bounds:?}"
        );
        let volume = crate::measure::solid_volume(&topo, rounded, 0.01).unwrap();
        let removed = (1.0 - std::f64::consts::FRAC_PI_4) * 40.0 * (4.0 + 10.0 + 25.0) / 3.0;
        assert!(
            (volume - (24000.0 - removed)).abs() < 1e-2 * removed,
            "volume {volume}, expected {}",
            24000.0 - removed
        );
    }
}
