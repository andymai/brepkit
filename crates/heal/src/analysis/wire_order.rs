//! Wire ordering — compute optimal edge ordering for a wire.
//!
//! Determines the best permutation of edges to form a connected chain,
//! accounting for edge orientations.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::wire::WireId;

use crate::HealError;

/// Result of computing the optimal wire order.
#[derive(Debug, Clone)]
pub struct WireOrderResult {
    /// Reordered edge list (indices into the original edge array).
    pub order: Vec<usize>,
    /// Whether any edges needed to be flipped.
    pub flips: Vec<bool>,
    /// Maximum gap found between consecutive edges after reordering.
    pub max_gap: f64,
    /// Whether the wire could be fully connected.
    pub is_connected: bool,
}

/// Compute the optimal edge ordering for a wire.
///
/// Given a wire with potentially unordered edges, finds a permutation
/// that minimizes gaps between consecutive edges. Uses a greedy
/// nearest-neighbor approach: starting from the first edge, repeatedly
/// picks the unused edge whose start is closest to the current end.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn compute_wire_order(topo: &Topology, wire_id: WireId) -> Result<WireOrderResult, HealError> {
    let wire = topo.wire(wire_id)?;
    let edges = wire.edges();
    let n = edges.len();

    if n <= 1 {
        return Ok(WireOrderResult {
            order: (0..n).collect(),
            flips: vec![false; n],
            max_gap: 0.0,
            is_connected: true,
        });
    }

    // Snapshot all edge endpoint positions.
    let mut starts: Vec<Point3> = Vec::with_capacity(n);
    let mut ends: Vec<Point3> = Vec::with_capacity(n);

    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let s = topo.vertex(oe.oriented_start(edge))?.point();
        let e = topo.vertex(oe.oriented_end(edge))?.point();
        starts.push(s);
        ends.push(e);
    }

    // Greedy nearest-neighbor ordering.
    let mut used = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut flips = Vec::with_capacity(n);
    let mut max_gap = 0.0_f64;

    // Start with edge 0 in its original orientation.
    order.push(0);
    flips.push(false);
    used[0] = true;
    let mut current_end = ends[0];

    for _ in 1..n {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        let mut best_flip = false;

        for j in 0..n {
            if used[j] {
                continue;
            }
            // Try forward: current_end -> starts[j]
            let d_fwd = (starts[j] - current_end).length_squared();
            if d_fwd < best_dist {
                best_dist = d_fwd;
                best_idx = j;
                best_flip = false;
            }
            // Try reversed: current_end -> ends[j]
            let d_rev = (ends[j] - current_end).length_squared();
            if d_rev < best_dist {
                best_dist = d_rev;
                best_idx = j;
                best_flip = true;
            }
        }

        used[best_idx] = true;
        order.push(best_idx);
        flips.push(best_flip);
        let gap = best_dist.sqrt();
        if gap > max_gap {
            max_gap = gap;
        }
        current_end = if best_flip {
            starts[best_idx]
        } else {
            ends[best_idx]
        };
    }

    let is_connected = max_gap < 1e-6;

    Ok(WireOrderResult {
        order,
        flips,
        max_gap,
        is_connected,
    })
}
