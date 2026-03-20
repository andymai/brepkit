//! Wire loop construction from trimmed intersection edges.

use brepkit_topology::Topology;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Build closed wire loops for each offset face from the trimmed
/// intersection curves and split edges.
///
/// # Errors
///
/// Returns [`OffsetError`] if a wire loop cannot be closed.
pub fn build_wire_loops(_topo: &mut Topology, _data: &mut OffsetData) -> Result<(), OffsetError> {
    todo!("Phase 7: wire loop construction")
}
