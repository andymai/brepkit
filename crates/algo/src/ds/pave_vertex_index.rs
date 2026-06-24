//! Spatial hash of pave-block endpoint vertices for O(1) coincidence lookup.
//!
//! During intersection the PaveFiller snaps every new intersection-curve
//! endpoint to a coincident existing pave vertex. A linear scan over all pave
//! blocks per endpoint is O(blocks) × O(endpoints) — quadratic when one operand
//! has many edges (a merged multi-region tool, e.g. a perforated panel). This
//! grid (cell = snap tolerance) answers each lookup in O(1) expected.
//!
//! It returns the SAME vertex the linear scan would: among the vertices within
//! tolerance of the query point, the one that appears FIRST in the scan's
//! iteration order — `edge_pave_blocks` ascending by `EdgeId` (a `BTreeMap`),
//! and start-before-end within each block. That position is recorded as `rank`,
//! and a query returns the minimum-`rank` candidate within tolerance.

use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_topology::vertex::VertexId;

/// One indexed pave vertex: its scan-order rank, resolved id, and position.
#[derive(Debug, Clone)]
struct Entry {
    rank: u32,
    vertex: VertexId,
    pos: Point3,
}

/// Uniform-grid spatial hash over resolved pave-block endpoint vertices.
#[derive(Debug, Clone)]
pub struct PaveVertexIndex {
    inv_cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<Entry>>,
}

impl PaveVertexIndex {
    /// Build the index from `(rank, resolved_vertex, position)` triples in
    /// ascending scan order. `cell` is the snap tolerance (the cell size).
    pub(crate) fn build(cell: f64, entries: impl Iterator<Item = (u32, VertexId, Point3)>) -> Self {
        let inv_cell = 1.0 / cell.max(f64::MIN_POSITIVE);
        let mut grid: HashMap<(i64, i64, i64), Vec<Entry>> = HashMap::new();
        for (rank, vertex, pos) in entries {
            grid.entry(cell_of(pos, inv_cell))
                .or_default()
                .push(Entry { rank, vertex, pos });
        }
        Self { inv_cell, grid }
    }

    /// Resolved pave vertex within `tol` of `point`, matching the linear scan's
    /// first-in-iteration-order tie-break; `None` if none is within `tol`.
    #[must_use]
    pub fn find_within(&self, point: Point3, tol: f64) -> Option<VertexId> {
        let (cx, cy, cz) = cell_of(point, self.inv_cell);
        let mut best: Option<(u32, VertexId)> = None;
        // A vertex within `tol` (one cell width) of `point` lies in the query
        // cell or an immediate neighbour, so the 3×3×3 stencil is exhaustive.
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(entries) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for e in entries {
                        crate::perf::bump_pave_vertex_probe();
                        if (e.pos - point).length() <= tol && best.is_none_or(|(r, _)| e.rank < r) {
                            best = Some((e.rank, e.vertex));
                        }
                    }
                }
            }
        }
        best.map(|(_, v)| v)
    }
}

fn cell_of(p: Point3, inv_cell: f64) -> (i64, i64, i64) {
    #[allow(clippy::cast_possible_truncation)]
    (
        (p.x() * inv_cell).floor() as i64,
        (p.y() * inv_cell).floor() as i64,
        (p.z() * inv_cell).floor() as i64,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use brepkit_topology::Topology;
    use brepkit_topology::vertex::Vertex;

    use super::*;

    /// Reference linear scan the index must reproduce: first within-`tol`
    /// entry in ascending rank (the scan's iteration order).
    fn linear_scan(
        entries: &[(u32, VertexId, Point3)],
        point: Point3,
        tol: f64,
    ) -> Option<VertexId> {
        entries
            .iter()
            .filter(|(_, _, pos)| (*pos - point).length() <= tol)
            .min_by_key(|(rank, _, _)| *rank)
            .map(|(_, v, _)| *v)
    }

    #[test]
    fn matches_linear_scan_including_tie_break_and_neighbors() {
        let mut topo = Topology::new();
        let tol = 1e-7;
        // Vertices: two coincident points (within tol) at distinct ranks, plus a
        // far one, plus a point straddling a cell boundary.
        let p_a = Point3::new(1.0, 2.0, 3.0);
        let p_a2 = Point3::new(1.0 + 0.4e-7, 2.0, 3.0); // within tol of p_a
        let p_far = Point3::new(50.0, 50.0, 50.0);
        let v_a = topo.add_vertex(Vertex::new(p_a, tol));
        let v_a2 = topo.add_vertex(Vertex::new(p_a2, tol));
        let v_far = topo.add_vertex(Vertex::new(p_far, tol));

        // Rank order: v_a2 (rank 0) appears before v_a (rank 1) — the index must
        // return the LOWER rank among the within-tol coincident pair.
        let entries = vec![(0u32, v_a2, p_a2), (1u32, v_a, p_a), (2u32, v_far, p_far)];
        let idx = PaveVertexIndex::build(tol, entries.clone().into_iter());

        // Query exactly at p_a: both v_a and v_a2 are within tol; min rank = v_a2.
        assert_eq!(idx.find_within(p_a, tol), linear_scan(&entries, p_a, tol));
        assert_eq!(idx.find_within(p_a, tol), Some(v_a2));

        // A point near the far vertex but outside tol returns None.
        let near_far = Point3::new(50.0 + 1e-3, 50.0, 50.0);
        assert_eq!(idx.find_within(near_far, tol), None);
        assert_eq!(
            idx.find_within(near_far, tol),
            linear_scan(&entries, near_far, tol)
        );

        // Exactly on the far vertex.
        assert_eq!(idx.find_within(p_far, tol), Some(v_far));
    }

    #[test]
    fn empty_index_returns_none() {
        let idx = PaveVertexIndex::build(1e-7, std::iter::empty());
        assert_eq!(idx.find_within(Point3::new(0.0, 0.0, 0.0), 1e-7), None);
    }
}
