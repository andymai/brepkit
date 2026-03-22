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
/// Uses the 4-phase BuilderSolid algorithm:
/// 1. Remove faces with free edges
/// 2. Build shells via edge-connectivity flood-fill
/// 3. Classify shells as Growth/Hole
/// 4. Assemble into Solid with inner shells
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
