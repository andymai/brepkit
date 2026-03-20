//! Global self-intersection detection and removal.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::error::OffsetError;

/// Detect and remove global self-intersections in the offset solid.
///
/// # Errors
///
/// Returns [`OffsetError::SelfIntersection`] if self-intersections cannot be resolved.
pub fn remove_self_intersections(
    _topo: &mut Topology,
    _solid: SolidId,
) -> Result<SolidId, OffsetError> {
    todo!("Phase 9: self-intersection removal")
}
