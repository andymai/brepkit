//! Assemble selected faces into shells and solids.
//!
//! Delegates to [`builder_solid::build_solid`] for OCCT-style 4-phase
//! shell assembly with edge connectivity, dihedral angle selection,
//! and Growth/Hole classification.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::bop::SelectedFace;
use crate::error::AlgoError;

/// Assemble selected faces into a solid.
///
/// Delegates to [`builder_solid::build_solid`], which implements an
/// OCCT-style BuilderSolid assembly:
/// 1. Build shells via edge-connectivity flood-fill (with dihedral
///    angle selection at non-manifold edges)
/// 2. Classify shells as Growth/Hole via signed volume
/// 3. Assemble into Solid with inner shells
///
/// Note: Phase 1 (free-edge removal) is currently disabled pending
/// full edge-identity sharing via CommonBlocks.
///
/// # Errors
///
/// Returns `AlgoError::AssemblyFailed` if no faces are selected or
/// shell construction fails.
pub fn assemble_solid(
    topo: &mut Topology,
    selected: &[SelectedFace],
) -> Result<SolidId, AlgoError> {
    super::builder_solid::build_solid(topo, selected)
}
