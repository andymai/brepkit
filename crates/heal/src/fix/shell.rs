//! Shell fixing — face-level fixes, orientation consistency.

use std::collections::HashMap;

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

    // ── 1. Run analysis to detect issues ────────────────────────────
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

    // ── 2. Fix each face individually ───────────────────────────────
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    for &fid in &face_ids {
        let face_result = super::face::fix_face(topo, fid, ctx, config)?;
        result.merge(&face_result);
    }

    // ── 3. Fix orientation consistency ──────────────────────────────
    let should_fix_orientation = config
        .fix_orientation
        .should_fix(!analysis.orientation_consistent);

    if should_fix_orientation {
        let orientation_result = fix_orientation(topo, shell_id, ctx)?;
        result.merge(&orientation_result);
    }

    Ok(result)
}

/// Check and repair face orientation consistency within a shell.
///
/// For each edge shared by exactly two faces, the two faces should
/// traverse that edge in opposite directions. If they traverse it in the
/// same direction, one face is flipped (its `reversed` flag is toggled).
///
/// Uses a BFS traversal starting from the first face: each visited face
/// is considered "correctly oriented", and neighbors that disagree are
/// flipped to match.
#[allow(clippy::too_many_lines)]
fn fix_orientation(
    topo: &mut Topology,
    shell_id: ShellId,
    ctx: &mut HealContext,
) -> Result<FixResult, HealError> {
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    if face_ids.is_empty() {
        return Ok(FixResult::ok());
    }

    // Map face arena index -> position in face_ids slice.
    let _face_idx_map: HashMap<usize, usize> = face_ids
        .iter()
        .enumerate()
        .map(|(i, fid)| (fid.index(), i))
        .collect();

    // Snapshot: collect all (face_position, edge_index, is_forward) triples.
    let mut face_edge_info: Vec<(usize, usize, bool)> = Vec::new();

    for (i, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                face_edge_info.push((i, oe.edge().index(), oe.is_forward()));
            }
        }
    }

    // Build edge-to-face adjacency from snapshots.
    // Key: edge index. Value: vec of (face_position, is_forward).
    let mut edge_faces: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
    for &(face_pos, edge_idx, is_forward) in &face_edge_info {
        edge_faces
            .entry(edge_idx)
            .or_default()
            .push((face_pos, is_forward));
    }

    // Build per-face edge list for efficient BFS neighbor lookup.
    let n = face_ids.len();
    let mut face_edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for &(face_pos, edge_idx, is_forward) in &face_edge_info {
        face_edges[face_pos].push((edge_idx, is_forward));
    }

    // BFS: start from face 0, flip neighbors that disagree.
    let mut visited = vec![false; n];
    let mut needs_flip = vec![false; n];
    let mut queue = std::collections::VecDeque::new();

    visited[0] = true;
    queue.push_back(0);

    while let Some(current) = queue.pop_front() {
        for &(edge_idx, is_forward) in &face_edges[current] {
            if let Some(neighbors) = edge_faces.get(&edge_idx) {
                for &(neighbor_pos, neighbor_fwd) in neighbors {
                    if neighbor_pos == current || visited[neighbor_pos] {
                        continue;
                    }
                    visited[neighbor_pos] = true;

                    // In a consistently oriented shell, adjacent faces
                    // traverse the shared edge in opposite directions.
                    let current_effective_fwd = if needs_flip[current] {
                        !is_forward
                    } else {
                        is_forward
                    };

                    if current_effective_fwd == neighbor_fwd {
                        // Same direction: neighbor needs to be flipped.
                        needs_flip[neighbor_pos] = true;
                    }

                    queue.push_back(neighbor_pos);
                }
            }
        }
    }

    // Apply flips.
    let mut flipped_count = 0usize;
    for (i, &fid) in face_ids.iter().enumerate() {
        if needs_flip[i] {
            let face = topo.face_mut(fid)?;
            let current_reversed = face.is_reversed();
            face.set_reversed(!current_reversed);
            flipped_count += 1;
        }
    }

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
