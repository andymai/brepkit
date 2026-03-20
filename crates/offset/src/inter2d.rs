//! 2D intersection of offset PCurves in parameter space.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Intersect offset PCurves in the parameter space of each face to find
/// split points on edges.
///
/// # Errors
///
/// Returns [`OffsetError`] if a PCurve intersection fails.
pub fn intersect_pcurves_2d(
    _topo: &Topology,
    _solid: SolidId,
    _data: &mut OffsetData,
) -> Result<(), OffsetError> {
    todo!("Phase 4: 2D PCurve intersection")
}
