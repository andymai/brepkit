//! Free bounds analysis — detect unshared (free) edges in a shell.
//!
//! Free boundary edges are those used by only one face. This module
//! groups them into connected loops by following shared vertices.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::shell::ShellId;

use crate::HealError;

/// Find free boundary loops in a shell.
///
/// Free edges are those referenced by exactly one face. The returned
/// value groups free edges into connected loops: each inner `Vec` is a
/// chain of edge IDs that share vertices end-to-end.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn find_free_bounds(topo: &Topology, shell_id: ShellId) -> Result<Vec<Vec<EdgeId>>, HealError> {
    let shell = topo.shell(shell_id)?;

    // Count how many times each edge is used across all face wires.
    let mut edge_use_count: HashMap<usize, (EdgeId, u32)> = HashMap::new();

    for &face_id in shell.faces() {
        let face = topo.face(face_id)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                edge_use_count
                    .entry(oe.edge().index())
                    .or_insert_with(|| (oe.edge(), 0))
                    .1 += 1;
            }
        }
    }

    // Collect free edges (used by exactly 1 face).
    let free_edges: Vec<EdgeId> = edge_use_count
        .values()
        .filter(|(_, count)| *count == 1)
        .map(|(eid, _)| *eid)
        .collect();

    if free_edges.is_empty() {
        return Ok(Vec::new());
    }

    // Build vertex-to-edge adjacency for free edges.
    // Each free edge contributes two vertex connections.
    let mut vertex_to_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();
    let free_set: HashSet<usize> = free_edges.iter().map(|e| e.index()).collect();

    for &eid in &free_edges {
        let edge = topo.edge(eid)?;
        vertex_to_edges
            .entry(edge.start().index())
            .or_default()
            .push(eid);
        vertex_to_edges
            .entry(edge.end().index())
            .or_default()
            .push(eid);
    }

    // Group free edges into connected loops by walking the adjacency.
    let mut visited: HashSet<usize> = HashSet::new();
    let mut loops = Vec::new();

    for &eid in &free_edges {
        if visited.contains(&eid.index()) {
            continue;
        }

        let mut chain = Vec::new();
        let mut stack = vec![eid];

        while let Some(current) = stack.pop() {
            if !visited.insert(current.index()) {
                continue;
            }
            chain.push(current);

            let edge = topo.edge(current)?;
            for vid_idx in [edge.start().index(), edge.end().index()] {
                if let Some(neighbors) = vertex_to_edges.get(&vid_idx) {
                    for &neighbor in neighbors {
                        if free_set.contains(&neighbor.index())
                            && !visited.contains(&neighbor.index())
                        {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        if !chain.is_empty() {
            loops.push(chain);
        }
    }

    Ok(loops)
}
