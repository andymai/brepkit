//! Post-split EE overlap detection — creates CommonBlocks for coincident
//! leaf PaveBlocks from different original edges.
//!
//! Runs after `make_blocks` (which splits PaveBlocks at extra paves),
//! iterating leaf PaveBlocks to find pairs with matching 3D endpoints
//! and compatible curve geometry. This follows OCCT's `ForceInterfEE` pattern.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};

use crate::ds::{GfaArena, PaveBlockId};
use crate::error::AlgoError;

/// Detect overlapping leaf PaveBlocks and group them into CommonBlocks.
///
/// Two leaf PaveBlocks from different original edges overlap if:
/// 1. Their start/end vertex positions are within tolerance
/// 2. Their edge curves have compatible geometry (same line direction,
///    same circle, etc.)
///
/// # Errors
///
/// Returns [`AlgoError`] if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn perform(topo: &Topology, tol: Tolerance, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect all leaf PBs with their 3D endpoint data
    let all_edge_pbs: Vec<(EdgeId, Vec<PaveBlockId>)> = arena
        .edge_pave_blocks
        .iter()
        .map(|(&eid, pbs)| (eid, arena.collect_leaf_pave_blocks(pbs)))
        .collect();

    // Flatten to (pb_id, original_edge, start_pos, end_pos)
    let mut leaf_data: Vec<(
        PaveBlockId,
        EdgeId,
        brepkit_math::vec::Point3,
        brepkit_math::vec::Point3,
    )> = Vec::new();

    for (orig_edge, leaf_pbs) in &all_edge_pbs {
        for &pb_id in leaf_pbs {
            let pb = match arena.pave_blocks.get(pb_id) {
                Some(pb) => pb,
                None => continue,
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);
            let start_pos = topo.vertex(sv)?.point();
            let end_pos = topo.vertex(ev)?.point();
            leaf_data.push((pb_id, *orig_edge, start_pos, end_pos));
        }
    }

    // Find overlapping pairs: O(n²) but n is typically small (< 100 leaf PBs)
    let mut overlap_map: HashMap<PaveBlockId, Vec<PaveBlockId>> = HashMap::new();
    let n = leaf_data.len();

    for i in 0..n {
        let (pb_i, edge_i, start_i, end_i) = &leaf_data[i];
        for j in (i + 1)..n {
            let (pb_j, edge_j, start_j, end_j) = &leaf_data[j];

            // Must be from different original edges
            if edge_i == edge_j {
                continue;
            }

            // Already in same CB
            if arena.pb_to_cb.contains_key(pb_i)
                && arena.pb_to_cb.get(pb_i) == arena.pb_to_cb.get(pb_j)
            {
                continue;
            }

            // Check endpoint match (either same-direction or reversed)
            let fwd_match = (*start_i - *start_j).length() < tol.linear
                && (*end_i - *end_j).length() < tol.linear;
            let rev_match = (*start_i - *end_j).length() < tol.linear
                && (*end_i - *start_j).length() < tol.linear;

            if !fwd_match && !rev_match {
                continue;
            }

            // Check curve compatibility
            let curve_i = topo.edge(*edge_i)?.curve();
            let curve_j = topo.edge(*edge_j)?.curve();
            if !curves_compatible(curve_i, curve_j, tol) {
                continue;
            }

            // Record overlap
            overlap_map.entry(*pb_i).or_default().push(*pb_j);
            overlap_map.entry(*pb_j).or_default().push(*pb_i);
        }
    }

    // Build transitive closure and create CommonBlocks
    let mut visited: HashSet<PaveBlockId> = HashSet::new();

    for &(pb_id, _, _, _) in &leaf_data {
        if visited.contains(&pb_id) || !overlap_map.contains_key(&pb_id) {
            continue;
        }

        // BFS to find connected component
        let mut group = Vec::new();
        let mut queue = vec![pb_id];
        while let Some(current) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }
            group.push(current);
            if let Some(neighbors) = overlap_map.get(&current) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }

        if group.len() >= 2 {
            let cb_id = arena.create_common_block(group.clone(), tol.linear);
            log::debug!(
                "ForceInterfEE: created CommonBlock {cb_id:?} with {} PaveBlocks",
                group.len()
            );
        }
    }

    Ok(())
}

/// Check if two edge curves are geometrically compatible (same type + parameters).
fn curves_compatible(a: &EdgeCurve, b: &EdgeCurve, tol: Tolerance) -> bool {
    match (a, b) {
        (EdgeCurve::Line, EdgeCurve::Line) => true,
        (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) => {
            (ca.radius() - cb.radius()).abs() < tol.linear
                && (ca.center() - cb.center()).length() < tol.linear
        }
        (EdgeCurve::Ellipse(ea), EdgeCurve::Ellipse(eb)) => {
            (ea.semi_major() - eb.semi_major()).abs() < tol.linear
                && (ea.semi_minor() - eb.semi_minor()).abs() < tol.linear
                && (ea.center() - eb.center()).length() < tol.linear
        }
        _ => false, // Different types or NURBS: no overlap detection yet
    }
}
