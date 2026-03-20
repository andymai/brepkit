//! Remove internal (hole) wires from faces.

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Remove all inner (hole) wires from faces in a solid.
///
/// Returns the number of wires removed.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn remove_internal_wires(topo: &mut Topology, solid_id: SolidId) -> Result<usize, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut removed = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let n_inner = face.inner_wires().len();
        if n_inner > 0 {
            let face_mut = topo.face_mut(fid)?;
            face_mut.inner_wires_mut().clear();
            removed += n_inner;
        }
    }

    Ok(removed)
}
