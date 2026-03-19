//! Build 2D parametric curves for split edges on their faces.
//!
//! Each split edge that lies on a face needs a pcurve (2D representation
//! in the face's UV parameter space). This phase computes those pcurves
//! and registers them in the topology's [`PCurveRegistry`].
//!
//! Currently a stub — pcurve computation will be added when the Builder's
//! wire splitter needs them.
//!
//! [`PCurveRegistry`]: brepkit_topology::pcurve::PCurveRegistry

use brepkit_topology::Topology;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// Compute pcurves for all split edges on their faces.
///
/// Currently a stub — pcurve computation will be added when the
/// Builder's wire splitter needs them.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
#[allow(clippy::unnecessary_wraps)] // Result kept for API consistency with other phases
pub fn perform(_topo: &mut Topology, _arena: &GfaArena) -> Result<(), AlgoError> {
    // TODO: For each face's pave_blocks_on and pave_blocks_sc,
    // project the split edge's 3D curve onto the face surface
    // to get a 2D pcurve, then register it.
    log::debug!("MakePCurves: stub — pcurves will be computed on demand");
    Ok(())
}
