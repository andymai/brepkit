//! Assemble selected faces into shells and solids.

use brepkit_topology::Topology;
use brepkit_topology::face::Face;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

use crate::bop::SelectedFace;
use crate::error::AlgoError;

/// Assemble selected faces into a solid.
///
/// Takes the faces selected by the BOP and builds shell -> solid.
/// Faces from solid B in a Cut operation are reversed (normals flipped).
/// For now, creates a single shell from all selected faces.
///
/// # Errors
///
/// Returns `AlgoError::AssemblyFailed` if no faces are selected or
/// shell construction fails.
pub fn assemble_solid(
    topo: &mut Topology,
    selected: &[SelectedFace],
) -> Result<SolidId, AlgoError> {
    if selected.is_empty() {
        return Err(AlgoError::AssemblyFailed("no faces selected".into()));
    }

    let mut result_faces = Vec::with_capacity(selected.len());

    for sf in selected {
        if sf.reversed {
            // Create a reversed copy of the face
            let face = topo.face(sf.face_id)?;
            let surface = face.surface().clone();
            let outer_wire = face.outer_wire();
            let inner_wires = face.inner_wires().to_vec();

            let reversed_face = Face::new_reversed(outer_wire, inner_wires, surface);
            let reversed_id = topo.add_face(reversed_face);
            result_faces.push(reversed_id);
        } else {
            result_faces.push(sf.face_id);
        }
    }

    // Build shell from all faces
    let shell = Shell::new(result_faces)
        .map_err(|e| AlgoError::AssemblyFailed(format!("shell creation failed: {e}")))?;
    let shell_id = topo.add_shell(shell);

    // Build solid with single outer shell, no inner shells (voids)
    let solid = Solid::new(shell_id, vec![]);
    let solid_id = topo.add_solid(solid);

    log::debug!(
        "assemble_solid: created solid {solid_id:?} with {} faces",
        selected.len(),
    );

    Ok(solid_id)
}
