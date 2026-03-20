//! Global self-intersection detection and removal.
//!
//! When an offset distance exceeds the local radius of curvature at
//! concave features, the offset shell folds back on itself. This module
//! detects such regions and removes them, producing a valid offset solid.
//!
//! The current implementation is a passthrough — it returns the solid
//! unchanged. Full BOP-based SI removal will be added in a follow-up.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::error::OffsetError;

/// Detect and remove global self-intersections in the offset solid.
///
/// Currently a passthrough: returns the input solid unchanged. Full
/// self-intersection removal via the GFA boolean engine will be
/// implemented when the pipeline handles non-planar geometry.
///
/// # Errors
///
/// Returns [`OffsetError::SelfIntersection`] if self-intersections
/// cannot be resolved (not yet triggered).
#[allow(clippy::unnecessary_wraps)]
pub fn remove_self_intersections(
    _topo: &mut Topology,
    solid: SolidId,
) -> Result<SolidId, OffsetError> {
    // TODO: implement BOP-based SI removal
    // 1. Detect inverted face normals
    // 2. Build bounding box, intersect offset solid with it
    // 3. Filter inverted regions
    log::debug!("self_int: SI removal not yet implemented, returning solid unchanged");
    Ok(solid)
}
