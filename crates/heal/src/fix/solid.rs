//! Solid fixing — top-level fix orchestration.
//!
//! Orchestrates shell, face, wire, and edge fixes, plus solid-level
//! repairs such as coincident vertex merging and small face removal.

use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::Edge;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Fix a solid: orchestrates shell, face, wire, and edge fixes.
///
/// 1. Fixes the outer shell (face-level + orientation).
/// 2. If `config.fix_coincident_vertices` permits, merges coincident
///    vertices across the solid.
/// 3. If `config.fix_small_faces` permits, removes degenerate small faces.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_solid(
    topo: &mut Topology,
    solid_id: SolidId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = FixResult::ok();

    // ── 1. Fix outer shell ──────────────────────────────────────────
    let solid = topo.solid(solid_id)?;
    let shell_id = solid.outer_shell();

    let shell_result = super::shell::fix_shell(topo, shell_id, ctx, config)?;
    result.merge(&shell_result);

    // ── 2. Fix inner shells (voids) ─────────────────────────────────
    let solid = topo.solid(solid_id)?;
    let inner_shells: Vec<_> = solid.inner_shells().to_vec();
    for shell_id in inner_shells {
        let inner_result = super::shell::fix_shell(topo, shell_id, ctx, config)?;
        result.merge(&inner_result);
    }

    // ── 3. Merge coincident vertices ────────────────────────────────
    // Always detected (cheap), applied per config mode.
    let should_merge = config.fix_coincident_vertices.should_fix(true);
    if should_merge {
        let merge_result = merge_coincident_vertices(topo, solid_id, ctx)?;
        result.merge(&merge_result);
    }

    // ── 4. Remove small faces ───────────────────────────────────────
    let should_fix_small = config.fix_small_faces.should_fix(true);
    if should_fix_small {
        let small_result = super::small_face::fix_small_faces(topo, solid_id, ctx, config)?;
        result.merge(&small_result);
    }

    Ok(result)
}

/// Merge coincident vertices across a solid.
///
/// For each pair of vertices within `ctx.tolerance.linear` distance,
/// the higher-index vertex is replaced by the lower-index one. Edge
/// start/end references are updated via the `ReShape` replacement tracker.
///
/// Ported from `brepkit-operations` `heal::merge_coincident_vertices`.
#[allow(clippy::too_many_lines)]
fn merge_coincident_vertices(
    topo: &mut Topology,
    solid_id: SolidId,
    ctx: &mut HealContext,
) -> Result<FixResult, HealError> {
    let tol = ctx.tolerance.linear;
    let tol_sq = tol * tol;

    // Collect all unique vertex IDs and positions from the solid.
    let solid_data = topo.solid(solid_id)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut vertex_ids: Vec<VertexId> = Vec::new();
    let mut positions: Vec<Point3> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                for &vid in &[edge.start(), edge.end()] {
                    if seen.insert(vid.index()) {
                        let point = topo.vertex(vid)?.point();
                        vertex_ids.push(vid);
                        positions.push(point);
                    }
                }
            }
        }
    }

    // Build merge map: higher-index merges into lower-index (canonical).
    let num_verts = vertex_ids.len();
    let mut merge_to: HashMap<usize, VertexId> = HashMap::new();
    let mut merged_count = 0usize;

    for i in 0..num_verts {
        if merge_to.contains_key(&vertex_ids[i].index()) {
            continue;
        }
        for j in (i + 1)..num_verts {
            if merge_to.contains_key(&vertex_ids[j].index()) {
                continue;
            }
            let dist_sq = (positions[i] - positions[j]).length_squared();
            if dist_sq < tol_sq {
                merge_to.insert(vertex_ids[j].index(), vertex_ids[i]);
                merged_count += 1;
            }
        }
    }

    if merged_count == 0 {
        return Ok(FixResult::ok());
    }

    // Record vertex replacements in the reshape tracker.
    for (&from_idx, &to_vid) in &merge_to {
        // Reconstruct the VertexId from index — find it in our collected list.
        if let Some(&from_vid) = vertex_ids.iter().find(|v| v.index() == from_idx) {
            ctx.reshape.replace_vertex(from_vid, to_vid);
        }
    }

    // Also apply directly to edges (snapshot then allocate pattern).
    let mut edge_ids = Vec::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                edge_ids.push(oe.edge());
            }
        }
    }
    edge_ids.sort_by_key(|e| e.index());
    edge_ids.dedup_by_key(|e| e.index());

    // Snapshot edge updates.
    let updates: Vec<_> = edge_ids
        .iter()
        .filter_map(|&eid| {
            let edge = topo.edge(eid).ok()?;
            let new_start = merge_to
                .get(&edge.start().index())
                .copied()
                .unwrap_or_else(|| edge.start());
            let new_end = merge_to
                .get(&edge.end().index())
                .copied()
                .unwrap_or_else(|| edge.end());
            if new_start != edge.start() || new_end != edge.end() {
                Some((eid, new_start, new_end, edge.curve().clone()))
            } else {
                None
            }
        })
        .collect();

    // Apply updates.
    for (eid, new_start, new_end, curve) in updates {
        let edge = topo.edge_mut(eid)?;
        *edge = Edge::new(new_start, new_end, curve);
    }

    ctx.info(format!("merged {merged_count} coincident vertices"));

    Ok(FixResult {
        status: Status::DONE2,
        actions_taken: merged_count,
    })
}
