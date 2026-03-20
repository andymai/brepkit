//! Final shell and solid assembly from offset faces and wire loops.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Assemble the final offset solid from trimmed offset faces, joint
/// faces, and wire loops.
///
/// # Errors
///
/// Returns [`OffsetError::AssemblyFailed`] if the solid cannot be assembled.
pub fn assemble_solid(_topo: &mut Topology, _data: &OffsetData) -> Result<SolidId, OffsetError> {
    todo!("Phase 8: shell and solid assembly")
}
