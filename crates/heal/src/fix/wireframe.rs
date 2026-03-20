//! Wireframe fixing — repair missing or misaligned edges in shells.
//!
//! Detects free edges (edges used by only one face) and attempts to sew
//! geometrically coincident pairs by merging their vertices. This
//! effectively closes gaps in the shell boundary where edges should be
//! shared but were created as separate entities.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;
use brepkit_topology::vertex::VertexId;

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Fix wireframe issues in a shell.
///
/// 1. Identifies free edges (edges referenced by exactly one face).
/// 2. Attempts to sew geometrically coincident free-edge pairs by
///    merging their vertices so the edges become shared boundaries.
/// 3. Remaining unsewn free edges are logged as warnings.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_wireframe(
    topo: &mut Topology,
    shell_id: ShellId,
    ctx: &mut HealContext,
    _config: &FixConfig,
) -> Result<FixResult, HealError> {
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Build edge usage map: edge index -> list of face IDs that reference it.
    let mut edge_face_count: HashMap<usize, (EdgeId, Vec<FaceId>)> = HashMap::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let idx = oe.edge().index();
                edge_face_count
                    .entry(idx)
                    .or_insert_with(|| (oe.edge(), Vec::new()))
                    .1
                    .push(fid);
            }
        }
    }

    // Detect free edges (used by exactly one face).
    let mut free_edges: Vec<EdgeId> = Vec::new();
    for (edge_id, faces) in edge_face_count.values() {
        if faces.len() == 1 {
            free_edges.push(*edge_id);
        }
    }

    if free_edges.is_empty() {
        return Ok(FixResult::ok());
    }

    ctx.warn(format!(
        "detected {} free edges in shell (wireframe gaps)",
        free_edges.len()
    ));

    // Attempt to sew coincident free-edge pairs.
    let sewn = sew_free_edges(topo, &free_edges, ctx)?;

    if sewn > 0 {
        ctx.info(format!("sewn {sewn} free-edge pairs by merging vertices"));
    }

    // Log remaining unsewn free edges.
    let remaining = free_edges.len().saturating_sub(sewn * 2);
    if remaining > 0 {
        ctx.warn(format!(
            "{remaining} free edges remain after sewing (may need manual repair)"
        ));
    }

    if sewn > 0 {
        Ok(FixResult {
            status: Status::DONE4,
            actions_taken: sewn,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Snapshot of a free edge's geometric data.
struct FreeEdgeSnapshot {
    /// The edge ID.
    edge_id: EdgeId,
    /// Start vertex ID.
    start_vid: VertexId,
    /// End vertex ID.
    end_vid: VertexId,
    /// Start vertex position.
    start_pos: brepkit_math::vec::Point3,
    /// End vertex position.
    end_pos: brepkit_math::vec::Point3,
    /// Whether this edge has been paired already.
    paired: bool,
}

/// Attempt to sew geometrically coincident free-edge pairs.
///
/// For each pair of free edges, checks if their endpoint positions match
/// within tolerance (in either same or reversed orientation). If a match
/// is found, merges vertices so the edges share the same vertex handles.
///
/// Returns the number of pairs sewn.
#[allow(clippy::needless_pass_by_ref_mut)]
fn sew_free_edges(
    topo: &mut Topology,
    free_edges: &[EdgeId],
    ctx: &mut HealContext,
) -> Result<usize, HealError> {
    let tolerance = ctx.tolerance.linear;

    // Snapshot all free edge geometry.
    let mut snapshots: Vec<FreeEdgeSnapshot> = Vec::with_capacity(free_edges.len());
    for &eid in free_edges {
        let edge = topo.edge(eid)?;
        let start_vid = edge.start();
        let end_vid = edge.end();
        let start_pos = topo.vertex(start_vid)?.point();
        let end_pos = topo.vertex(end_vid)?.point();

        snapshots.push(FreeEdgeSnapshot {
            edge_id: eid,
            start_vid,
            end_vid,
            start_pos,
            end_pos,
            paired: false,
        });
    }

    // Find coincident pairs and record vertex merges.
    // A merge is (victim_vertex, target_vertex) — victim gets replaced by target.
    let mut merges: Vec<(EdgeId, VertexId, VertexId, VertexId, VertexId)> = Vec::new();

    for i in 0..snapshots.len() {
        if snapshots[i].paired {
            continue;
        }
        for j in (i + 1)..snapshots.len() {
            if snapshots[j].paired {
                continue;
            }

            let a = &snapshots[i];
            let b = &snapshots[j];

            // Check same-direction match: a.start ~ b.start AND a.end ~ b.end
            let same_start = (a.start_pos - b.start_pos).length() < tolerance;
            let same_end = (a.end_pos - b.end_pos).length() < tolerance;

            // Check reversed match: a.start ~ b.end AND a.end ~ b.start
            let rev_start = (a.start_pos - b.end_pos).length() < tolerance;
            let rev_end = (a.end_pos - b.start_pos).length() < tolerance;

            if same_start && same_end {
                // Same direction: merge b's vertices into a's.
                merges.push((b.edge_id, b.start_vid, a.start_vid, b.end_vid, a.end_vid));
                snapshots[i].paired = true;
                snapshots[j].paired = true;
                break;
            } else if rev_start && rev_end {
                // Reversed: merge b's start→a's end, b's end→a's start.
                merges.push((b.edge_id, b.start_vid, a.end_vid, b.end_vid, a.start_vid));
                snapshots[i].paired = true;
                snapshots[j].paired = true;
                break;
            }
        }
    }

    // Apply vertex merges: update edge b to use edge a's vertices.
    for &(edge_id, _old_start, new_start, _old_end, new_end) in &merges {
        let edge = topo.edge_mut(edge_id)?;
        edge.set_start(new_start);
        edge.set_end(new_end);
    }

    Ok(merges.len())
}
