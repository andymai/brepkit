//! Global self-intersection detection and removal.
//!
//! When an offset distance exceeds the local radius of curvature at
//! concave features, the offset shell folds back on itself. This module
//! detects such regions and removes them, producing a valid offset solid.
//!
//! The current implementation returns an error when invoked. Full
//! BOP-based SI removal will be added in a follow-up.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::error::OffsetError;

/// Detect and remove global self-intersections in the offset solid.
///
/// # Errors
///
/// Always returns [`OffsetError::SelfIntersection`] — self-intersection
/// removal is not yet implemented. Callers should only invoke this when
/// the `remove_self_intersections` option is explicitly enabled.
pub fn remove_self_intersections(
    _topo: &mut Topology,
    _solid: SolidId,
) -> Result<SolidId, OffsetError> {
    Err(OffsetError::SelfIntersection {
        reason: "self-intersection removal not yet implemented".into(),
    })
}
