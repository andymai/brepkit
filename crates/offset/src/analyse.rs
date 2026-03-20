//! Edge and vertex convexity classification.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Classify every edge of the solid as Convex, Concave, or Tangent,
/// and derive vertex classifications from incident edges.
///
/// # Errors
///
/// Returns [`OffsetError::AnalysisFailed`] if an edge cannot be classified.
pub fn analyse_edges(
    _topo: &Topology,
    _solid: SolidId,
    _data: &mut OffsetData,
) -> Result<(), OffsetError> {
    todo!("Phase 1: edge analysis")
}
