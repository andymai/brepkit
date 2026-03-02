//! Edge chamfering (cutting edges at an angle).

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

/// Chamfer one or more edges of a solid.
///
/// # Errors
///
/// Returns an error if any edge is not part of the solid.
pub fn chamfer(
    _topo: &mut Topology,
    _solid: SolidId,
    _edges: &[EdgeId],
    _distance: f64,
) -> Result<SolidId, crate::OperationsError> {
    Err(crate::OperationsError::InvalidInput {
        reason: "chamfer not yet implemented".into(),
    })
}
