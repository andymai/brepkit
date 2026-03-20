//! Rolling-ball arc joint construction at convex edges.

use brepkit_topology::Topology;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Build arc-fillet joint faces at convex edges when `JointType::Arc`
/// is selected.
///
/// # Errors
///
/// Returns [`OffsetError`] if an arc joint cannot be constructed.
pub fn build_arc_joints(_topo: &mut Topology, _data: &mut OffsetData) -> Result<(), OffsetError> {
    Err(OffsetError::InvalidInput {
        reason: "arc joint construction is not yet implemented; use JointType::Intersection"
            .to_string(),
    })
}
