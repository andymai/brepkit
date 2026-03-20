//! 3D intersection of adjacent offset faces.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Intersect pairs of adjacent offset faces in 3D to find new edge curves.
///
/// # Errors
///
/// Returns [`OffsetError::IntersectionFailed`] if a face pair cannot be intersected.
pub fn intersect_faces_3d(
    _topo: &Topology,
    _solid: SolidId,
    _data: &mut OffsetData,
) -> Result<(), OffsetError> {
    todo!("Phase 3: 3D face-face intersection")
}
