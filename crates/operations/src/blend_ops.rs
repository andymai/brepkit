//! Thin wrappers around `brepkit-blend` for the operations API.

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

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
    Ok(builder.build()?)
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
    Ok(builder.build()?)
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
    Ok(builder.build()?)
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
    Ok(builder.build()?)
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
