//! Shell fixing — face-level fixes, orientation consistency.

use brepkit_topology::Topology;
use brepkit_topology::shell::ShellId;

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Fix a shell: run face-level fixes, then repair orientation consistency.
///
/// 1. Runs [`analyze_shell`](crate::analysis::shell::analyze_shell) to detect
///    boundary edges, non-manifold edges, and orientation inconsistencies.
/// 2. Iterates all faces and calls [`fix_face`](super::face::fix_face) on each.
/// 3. If `config.fix_orientation` permits, traverses the shell via BFS and
///    flips faces whose shared-edge directions disagree with their neighbors.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::too_many_lines)]
pub fn fix_shell(
    topo: &mut Topology,
    shell_id: ShellId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = FixResult::ok();

    let analysis = crate::analysis::shell::analyze_shell(topo, shell_id)?;

    if !analysis.boundary_edges.is_empty() {
        ctx.warn(format!(
            "shell has {} boundary (free) edges",
            analysis.boundary_edges.len()
        ));
    }
    if !analysis.non_manifold_edges.is_empty() {
        ctx.warn(format!(
            "shell has {} non-manifold edges",
            analysis.non_manifold_edges.len()
        ));
    }

    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    for &fid in &face_ids {
        let face_result = super::face::fix_face(topo, fid, ctx, config)?;
        result.merge(&face_result);
    }

    let should_fix_orientation = config
        .fix_orientation
        .should_fix(!analysis.orientation_consistent);

    if should_fix_orientation {
        let orientation_result = fix_orientation(topo, shell_id, ctx)?;
        result.merge(&orientation_result);
    }

    Ok(result)
}

/// Check and repair effective face orientation within a shell.
///
/// For each edge shared by exactly two faces, the two effective edge senses
/// should oppose. A breadth-first traversal anchors the first face and toggles
/// neighbors whose raw wire sense XOR reversal flag disagrees.
fn fix_orientation(
    topo: &mut Topology,
    shell_id: ShellId,
    ctx: &mut HealContext,
) -> Result<FixResult, HealError> {
    let face_ids = topo.shell(shell_id)?.faces().to_vec();
    let Some(&seed) = face_ids.first() else {
        return Ok(FixResult::ok());
    };
    let flipped_count =
        brepkit_topology::orientation::propagate_orientation(topo, &face_ids, &[seed])?;

    if flipped_count > 0 {
        ctx.info(format!(
            "flipped {flipped_count} faces for orientation consistency"
        ));
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: flipped_count,
        })
    } else {
        Ok(FixResult::ok())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::orientation::propagate_orientation;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    #[test]
    fn propagation_repairs_effective_face_senses() {
        let mut topo = brepkit_topology::Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let face_ids = topo.shell(shell_id).unwrap().faces().to_vec();
        assert!(
            crate::analysis::shell::analyze_shell(&topo, shell_id)
                .unwrap()
                .orientation_consistent
        );

        let wrong_face = *face_ids.last().unwrap();
        let was_reversed = topo.face(wrong_face).unwrap().is_reversed();
        topo.face_mut(wrong_face)
            .unwrap()
            .set_reversed(!was_reversed);
        assert!(
            !crate::analysis::shell::analyze_shell(&topo, shell_id)
                .unwrap()
                .orientation_consistent
        );

        let flipped = propagate_orientation(&mut topo, &face_ids, &face_ids[..1]).unwrap();
        assert!(flipped > 0);
        assert!(
            crate::analysis::shell::analyze_shell(&topo, shell_id)
                .unwrap()
                .orientation_consistent
        );
    }
}
