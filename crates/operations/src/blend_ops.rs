//! Thin wrappers around `brepkit-blend` for the operations API.

use brepkit_blend::BlendResult;
use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Fillet edges with constant radius (v2 walking-based engine).
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
