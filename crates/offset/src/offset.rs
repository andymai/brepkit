//! Offset surface construction for each face.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Construct the offset surface for every non-excluded face.
///
/// # Errors
///
/// Returns [`OffsetError`] if a surface cannot be offset.
pub fn build_offset_faces(
    _topo: &Topology,
    _solid: SolidId,
    _data: &mut OffsetData,
) -> Result<(), OffsetError> {
    todo!("Phase 2: offset surface construction")
}
