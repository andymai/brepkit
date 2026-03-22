//! Create topology edges from finalized pave blocks.
//!
//! This is the single point where `&mut Topology` is written during the
//! PaveFiller pipeline. For each leaf pave block that does not yet have a
//! `split_edge`, a new [`Edge`] is created in the topology.
//!
//! **CommonBlock-aware:** When a PaveBlock belongs to a CommonBlock, a single
//! edge is created for the entire group. All PBs in the CB reference the
//! same edge entity.

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};

use crate::ds::{CommonBlockId, GfaArena, PaveBlockId};
use crate::error::AlgoError;

/// Create topology edges for all leaf pave blocks.
///
/// For each pave block without children (leaf) and without an existing
/// `split_edge`, a new [`Edge`] is created in the topology. The curve
/// type is inherited from the original edge.
///
/// **CommonBlock handling:** When a PB belongs to a CommonBlock that has
/// already been processed, the existing edge is reused. When processing
/// a CB for the first time, the canonical PB creates the edge and all
/// members share it.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &mut Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Track processed CommonBlocks to avoid creating duplicate edges
    let mut processed_cbs: HashSet<CommonBlockId> = HashSet::new();

    // Collect all leaf pave block IDs that need edges
    let leaf_ids: Vec<_> = arena
        .pave_blocks
        .iter()
        .filter(|(_, pb)| pb.children.is_empty() && pb.split_edge.is_none())
        .map(|(id, _)| id)
        .collect();

    for pb_id in leaf_ids {
        // Check if this PB is part of a CommonBlock
        if let Some(&cb_id) = arena.pb_to_cb.get(&pb_id) {
            if !processed_cbs.insert(cb_id) {
                // CB already processed — reuse its split edge
                let split_edge = arena.common_blocks.get(cb_id).and_then(|cb| cb.split_edge);
                if let Some(edge_id) = split_edge {
                    if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                        pb.split_edge = Some(edge_id);
                    }
                }
                continue;
            }

            // First PB in this CB — use canonical PB to create the edge
            let canonical_pb_id = arena
                .common_blocks
                .get(cb_id)
                .and_then(|cb| cb.pave_blocks.first().copied())
                .unwrap_or(pb_id);

            let edge_id = create_split_edge(topo, arena, canonical_pb_id)?;

            // Set split_edge on the CB and ALL PBs in the group
            let all_pbs: Vec<PaveBlockId> = arena
                .common_blocks
                .get(cb_id)
                .map(|cb| cb.pave_blocks.clone())
                .unwrap_or_default();

            if let Some(cb) = arena.common_blocks.get_mut(cb_id) {
                cb.split_edge = Some(edge_id);
            }
            for &member_pb in &all_pbs {
                if let Some(pb) = arena.pave_blocks.get_mut(member_pb) {
                    pb.split_edge = Some(edge_id);
                }
            }

            log::debug!(
                "MakeSplitEdges: created edge {edge_id:?} for CommonBlock {cb_id:?} \
                 ({} PaveBlocks)",
                all_pbs.len()
            );
        } else {
            // No CB — create individual split edge as before
            let edge_id = create_split_edge(topo, arena, pb_id)?;
            if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                pb.split_edge = Some(edge_id);
            }
            log::debug!("MakeSplitEdges: created edge {edge_id:?} for pave block {pb_id:?}");
        }
    }

    Ok(())
}

/// Create a single split edge from a pave block's data.
fn create_split_edge(
    topo: &mut Topology,
    arena: &GfaArena,
    pb_id: PaveBlockId,
) -> Result<EdgeId, AlgoError> {
    let (original_edge_id, start_vertex, end_vertex) = {
        let pb = arena.pave_blocks.get(pb_id).ok_or_else(|| {
            AlgoError::FaceSplitFailed(format!(
                "MakeSplitEdges: pave block {pb_id:?} not found in arena"
            ))
        })?;
        let start_v = arena.resolve_vertex(pb.start.vertex);
        let end_v = arena.resolve_vertex(pb.end.vertex);
        (pb.original_edge, start_v, end_v)
    };

    let curve = topo.edge(original_edge_id)?.curve().clone();
    let new_edge = Edge::new(start_vertex, end_vertex, curve);
    Ok(topo.add_edge(new_edge))
}
