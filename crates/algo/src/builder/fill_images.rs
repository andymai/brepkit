//! Map original edges to their split images (leaf pave blocks).
//!
//! After the PaveFiller has split edges at intersection points, each
//! original edge maps to one or more leaf pave blocks, each of which
//! has a `split_edge` pointing to the new topology edge.

use std::collections::HashMap;

use brepkit_topology::edge::EdgeId;

use crate::ds::GfaArena;

/// For each original edge, collect its leaf pave block split-edge IDs.
///
/// Returns a map from original edge ID to a list of new edge IDs that
/// replace it. Edges with no splits map to their original split-edge
/// (the single leaf pave block).
#[must_use]
pub fn fill_edge_images(arena: &GfaArena) -> HashMap<EdgeId, Vec<EdgeId>> {
    let mut images: HashMap<EdgeId, Vec<EdgeId>> = HashMap::new();

    for (&original_edge, pb_ids) in &arena.edge_pave_blocks {
        let leaves = arena.collect_leaf_pave_blocks(pb_ids);
        let mut split_edges = Vec::new();

        for leaf_id in leaves {
            if let Some(pb) = arena.pave_blocks.get(leaf_id) {
                if let Some(se) = pb.split_edge {
                    split_edges.push(se);
                }
            }
        }

        if split_edges.is_empty() {
            // No split happened — the original edge is its own image.
            // This can occur for edges not involved in any intersection.
            split_edges.push(original_edge);
        }

        images.insert(original_edge, split_edges);
    }

    images
}
