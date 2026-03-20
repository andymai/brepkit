//! Final shell and solid assembly from offset faces and wire loops.

use brepkit_topology::Topology;
use brepkit_topology::face::Face;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

use crate::data::{OffsetData, OffsetStatus};
use crate::error::OffsetError;

/// Assemble the final offset solid from trimmed offset faces, joint
/// faces, and wire loops.
///
/// For each non-excluded offset face that has reconstructed wire loops,
/// a new [`Face`] is created with the offset surface and wires. All
/// new faces (including any joint faces from Phase 6) are collected
/// into a [`Shell`], which is then wrapped in a [`Solid`].
///
/// # Errors
///
/// Returns [`OffsetError::AssemblyFailed`] if no faces could be
/// assembled or the shell construction fails.
pub fn assemble_solid(topo: &mut Topology, data: &OffsetData) -> Result<SolidId, OffsetError> {
    let mut new_faces = Vec::new();

    // Create a face for each successfully offset face that has wire loops.
    for (face_id, offset_face) in &data.offset_faces {
        if offset_face.status == OffsetStatus::Excluded {
            continue;
        }

        let Some(wires) = data.face_wires.get(face_id) else {
            // No wires built for this face — skip with a warning.
            // In a production build this might log; for now we silently skip.
            continue;
        };

        if wires.is_empty() {
            continue;
        }

        let outer_wire = wires[0];
        let inner_wires = wires[1..].to_vec();

        let face = Face::new(outer_wire, inner_wires, offset_face.surface.clone());
        let face_id = topo.add_face(face);
        new_faces.push(face_id);
    }

    // Include any joint faces created during Phase 6 (arc joints).
    for &joint_face in &data.joint_faces {
        new_faces.push(joint_face);
    }

    if new_faces.is_empty() {
        return Err(OffsetError::AssemblyFailed {
            reason: "no faces could be assembled for the offset solid".to_string(),
        });
    }

    let shell = Shell::new(new_faces)?;
    let shell_id = topo.add_shell(shell);

    let solid = Solid::new(shell_id, vec![]);
    let solid_id = topo.add_solid(solid);

    Ok(solid_id)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions};
    use brepkit_topology::Topology;

    fn run_full_pipeline(topo: &mut Topology, solid: SolidId, distance: f64) -> SolidId {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        crate::inter3d::intersect_faces_3d(topo, solid, &mut data).unwrap();
        crate::inter2d::intersect_pcurves_2d(topo, solid, &mut data).unwrap();
        crate::loops::build_wire_loops(topo, &mut data).unwrap();
        assemble_solid(topo, &data).unwrap()
    }

    #[test]
    fn box_offset_produces_valid_solid() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = run_full_pipeline(&mut topo, solid, 0.5);

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6, "offset box should have 6 faces");
    }

    #[test]
    fn box_offset_faces_have_wires() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = run_full_pipeline(&mut topo, solid, 0.5);

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            assert_eq!(wire.edges().len(), 4, "each face should have 4 edges");
        }
    }

    #[test]
    fn box_offset_end_to_end() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = crate::offset_solid(
            &mut topo,
            solid,
            0.5,
            OffsetOptions {
                remove_self_intersections: false,
                ..Default::default()
            },
        )
        .unwrap();

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6);
    }
}
