//! Shell sewing — merge open shells by stitching free edges.
//!
//! Finds pairs of free boundary edges that are geometrically coincident
//! and merges their vertices to close gaps.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::shell::ShellId;

use crate::HealError;

/// Sew free boundary edges in a shell by merging coincident vertices.
///
/// Returns the number of edges sewn.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn sew_shell(
    topo: &mut Topology,
    shell_id: ShellId,
    tolerance: f64,
) -> Result<usize, HealError> {
    let tol_sq = tolerance * tolerance;

    // Build edge usage count.
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut edge_usage: HashMap<usize, usize> = HashMap::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            *edge_usage.entry(oe.edge().index()).or_insert(0) += 1;
        }
    }

    // Collect actual EdgeIds from wires (not just indices).
    let mut all_edge_ids: Vec<EdgeId> = Vec::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            all_edge_ids.push(oe.edge());
        }
    }

    // Free edges: used by exactly 1 face.
    let free_edges: Vec<EdgeId> = all_edge_ids
        .iter()
        .filter(|eid| edge_usage.get(&eid.index()).copied().unwrap_or(0) == 1)
        .copied()
        .collect();

    if free_edges.len() < 2 {
        return Ok(0);
    }

    // Try to match free edges by vertex proximity.
    let mut sewn = 0;
    let mut matched: Vec<bool> = vec![false; free_edges.len()];

    for i in 0..free_edges.len() {
        if matched[i] {
            continue;
        }
        let ei = topo.edge(free_edges[i])?;
        let si = topo.vertex(ei.start())?.point();
        let ti = topo.vertex(ei.end())?.point();

        for j in (i + 1)..free_edges.len() {
            if matched[j] {
                continue;
            }
            let ej = topo.edge(free_edges[j])?;
            let sj = topo.vertex(ej.start())?.point();
            let tj = topo.vertex(ej.end())?.point();

            // Check forward matching: si≈sj and ti≈tj
            let fwd = (si - sj).length_squared() < tol_sq && (ti - tj).length_squared() < tol_sq;
            // Check reverse matching: si≈tj and ti≈sj
            let rev = (si - tj).length_squared() < tol_sq && (ti - sj).length_squared() < tol_sq;

            if fwd || rev {
                // Merge vertices of edge j into edge i.
                let (_merge_s, _merge_t) = if fwd {
                    (ej.start(), ej.end())
                } else {
                    (ej.end(), ej.start())
                };
                let target_s = ei.start();
                let target_t = ei.end();

                // Update edge j's vertices to match edge i.
                let curve_j = topo.edge(free_edges[j])?.curve().clone();
                let ej_mut = topo.edge_mut(free_edges[j])?;
                *ej_mut = brepkit_topology::edge::Edge::new(
                    if fwd { target_s } else { target_t },
                    if fwd { target_t } else { target_s },
                    curve_j,
                );

                matched[i] = true;
                matched[j] = true;
                sewn += 1;
                break;
            }
        }
    }

    Ok(sewn)
}
