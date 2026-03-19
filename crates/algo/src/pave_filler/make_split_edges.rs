//! Create topology edges from finalized pave blocks.
//!
//! This is the single point where `&mut Topology` is written during the
//! PaveFiller pipeline. For each leaf pave block that does not yet have a
//! `split_edge`, a new [`Edge`] is created in the topology.

use brepkit_topology::Topology;
use brepkit_topology::edge::Edge;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// Create topology edges for all leaf pave blocks.
///
/// For each pave block without children (leaf) and without an existing
/// `split_edge`, a new [`Edge`] is created in the topology. The curve
/// type is inherited from the original edge. For `Line` edges the
/// geometry is fully determined by the endpoint vertices. For curves
/// (`Circle`, `Ellipse`, `NurbsCurve`) the parent curve is reused
/// (sub-curve extraction is deferred to a follow-up).
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &mut Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect all leaf pave block IDs that need edges
    let leaf_ids: Vec<_> = arena
        .pave_blocks
        .iter()
        .filter(|(_, pb)| pb.children.is_empty() && pb.split_edge.is_none())
        .map(|(id, _)| id)
        .collect();

    for pb_id in leaf_ids {
        // Snapshot data needed from the pave block and topology
        let (original_edge_id, start_vertex, end_vertex) = {
            let pb = match arena.pave_blocks.get(pb_id) {
                Some(pb) => pb,
                None => continue,
            };
            let start_v = arena.resolve_vertex(pb.start.vertex);
            let end_v = arena.resolve_vertex(pb.end.vertex);
            (pb.original_edge, start_v, end_v)
        };

        // Clone the original edge's curve
        let curve = topo.edge(original_edge_id)?.curve().clone();

        let new_edge = Edge::new(start_vertex, end_vertex, curve);
        let new_edge_id = topo.add_edge(new_edge);

        // Record the split edge in the pave block
        if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
            pb.split_edge = Some(new_edge_id);
        }

        log::debug!(
            "MakeSplitEdges: created edge {new_edge_id:?} for pave block {pb_id:?} \
             (original {original_edge_id:?})",
        );
    }

    Ok(())
}
