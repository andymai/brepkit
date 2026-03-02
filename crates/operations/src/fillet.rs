//! Edge filleting (rounding edges with a constant or variable radius).

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

/// Fillet one or more edges of a solid.
///
/// # Errors
///
/// Returns an error if any edge is not part of the solid.
pub fn fillet(
    _topo: &mut Topology,
    _solid: SolidId,
    _edges: &[EdgeId],
    _radius: f64,
) -> Result<SolidId, crate::OperationsError> {
    Err(crate::OperationsError::InvalidInput {
        reason: "fillet not yet implemented".into(),
    })
}
