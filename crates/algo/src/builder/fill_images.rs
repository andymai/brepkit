//! Map original edges to their split images (leaf pave blocks).
//!
//! After the PaveFiller has split edges at intersection points, each
//! original edge maps to one or more leaf pave blocks, each of which
//! has a `split_edge` pointing to the new topology edge.

use std::collections::HashMap;

use brepkit_topology::edge::EdgeId;

use crate::ds::GfaArena;

/// For each original edge, collect its leaf pave block split-edge IDs,
/// sorted by start parameter along the original edge.
///
/// Returns a map from original edge ID to a list of new edge IDs that
/// replace it. Edges with no splits map to their original split-edge
/// (the single leaf pave block). The split edges are in parameter order
/// so that wire reconstruction can iterate them in sequence.
#[must_use]
pub fn fill_edge_images(arena: &GfaArena) -> HashMap<EdgeId, Vec<EdgeId>> {
    let mut images: HashMap<EdgeId, Vec<EdgeId>> = HashMap::new();

    for (&original_edge, pb_ids) in &arena.edge_pave_blocks {
        let leaves = arena.collect_leaf_pave_blocks(pb_ids);

        // Collect (start_parameter, split_edge) for sorting
        let mut split_with_param: Vec<(f64, EdgeId)> = Vec::new();

        for leaf_id in leaves {
            if let Some(pb) = arena.pave_blocks.get(leaf_id) {
                if let Some(se) = pb.split_edge {
                    split_with_param.push((pb.start.parameter, se));
                }
            }
        }

        if split_with_param.is_empty() {
            images.insert(original_edge, vec![original_edge]);
            continue;
        }

        // Sort by start parameter for correct wire order
        split_with_param.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let split_edges: Vec<EdgeId> = split_with_param.into_iter().map(|(_, eid)| eid).collect();
        images.insert(original_edge, split_edges);
    }

    images
}
