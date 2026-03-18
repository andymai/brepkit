//! Finalize pave block structure by splitting at extra paves.
//!
//! For each edge's pave blocks, if a pave block has accumulated `extra_paves`
//! from intersection phases (VE, EE, EF), this phase splits it into child
//! pave blocks and updates `edge_pave_blocks` to reference the leaves.

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// Split all pave blocks at their extra paves.
///
/// After this, each pave block represents a contiguous edge segment
/// with no pending intersection points. The `edge_pave_blocks` map
/// is updated to reference leaf (unsplit) pave blocks only.
///
/// # Errors
///
/// Returns [`AlgoError`] if arena operations fail.
#[allow(clippy::unnecessary_wraps)] // Result kept for API consistency with other phases
pub fn perform(arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect edges that need processing (can't iterate and mutate simultaneously)
    let edges: Vec<_> = arena.edge_pave_blocks.keys().copied().collect();

    for edge_id in edges {
        let pb_ids: Vec<_> = arena
            .edge_pave_blocks
            .get(&edge_id)
            .cloned()
            .unwrap_or_default();

        let mut new_pb_ids = Vec::new();
        for pb_id in pb_ids {
            let has_extras = arena
                .pave_blocks
                .get(pb_id)
                .is_some_and(|pb| !pb.extra_paves.is_empty());

            if has_extras {
                // Snapshot all data we need before mutating
                let (original_edge, start, end, extra_paves) = {
                    let pb = match arena.pave_blocks.get(pb_id) {
                        Some(pb) => pb,
                        None => continue,
                    };
                    (pb.original_edge, pb.start, pb.end, pb.extra_paves.clone())
                };

                // Sort and deduplicate extra paves by parameter
                let mut sorted_paves = extra_paves;
                sorted_paves.sort_by(|a, b| {
                    a.parameter
                        .partial_cmp(&b.parameter)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted_paves.dedup_by(|a, b| (a.parameter - b.parameter).abs() < 1e-10);

                // Create child pave blocks for each segment
                let mut prev_pave = start;
                let mut children = Vec::new();

                for pave in &sorted_paves {
                    // Skip paves that coincide with the boundaries
                    if (pave.parameter - start.parameter).abs() < 1e-10
                        || (pave.parameter - end.parameter).abs() < 1e-10
                    {
                        continue;
                    }

                    let child = crate::ds::PaveBlock::new(original_edge, prev_pave, *pave);
                    let child_id = arena.pave_blocks.alloc(child);
                    children.push(child_id);
                    prev_pave = *pave;
                }

                // Final segment from last split point to end
                let last_child = crate::ds::PaveBlock::new(original_edge, prev_pave, end);
                let last_id = arena.pave_blocks.alloc(last_child);
                children.push(last_id);

                // Record children on the parent
                if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                    pb.children.clone_from(&children);
                }

                log::debug!(
                    "MakeBlocks: edge {edge_id:?} pave block {pb_id:?} split into {} children",
                    children.len(),
                );

                new_pb_ids.extend(children);
            } else {
                // No extras — keep the original pave block as a leaf
                new_pb_ids.push(pb_id);
            }
        }

        arena.edge_pave_blocks.insert(edge_id, new_pb_ids);
    }

    Ok(())
}
