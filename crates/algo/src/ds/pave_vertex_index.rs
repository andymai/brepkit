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

use std::cell::RefCell;
use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_topology::vertex::VertexId;

use super::PaveBlockId;

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

/// Every pave-block vertex, ends and extra paves, by position, for the
/// nearest pave vertex whose tolerance ball, or a caller's reach, holds a
/// point. A vertex qualifies within the larger of the two, so a query with a
/// reach up to one cell reads the cells around the point and the vertices
/// whose tolerance has passed a cell, which are kept apart.
///
/// Built after VV, when the pave blocks' ends and the same-domain mapping
/// are fixed. Extra paves join through [`super::GfaArena::add_extra_pave`]
/// and a raised tolerance through [`Self::note_tolerance`].
#[derive(Debug, Clone)]
pub struct PaveBallIndex {
    cell: f64,
    inv_cell: f64,
    entries: Vec<BallEntry>,
    grid: HashMap<(i64, i64, i64), Vec<usize>>,
    by_vertex: HashMap<VertexId, Vec<usize>>,
    /// Each pave block's place in the scan the index stands in for.
    block_order: HashMap<PaveBlockId, u32>,
    /// Entries whose vertex's tolerance has passed `cell`, and a flag per
    /// entry for them. Interior so a tolerance raised through a shared
    /// arena reaches them.
    loose: RefCell<(Vec<usize>, Vec<bool>)>,
}

#[derive(Debug, Clone)]
struct BallEntry {
    /// The scan's order: the block's place, then 0 and 1 for its ends and 2
    /// on for its extra paves.
    key: (u32, u32),
    vertex: VertexId,
}

impl PaveBallIndex {
    pub(crate) fn new(cell: f64) -> Self {
        Self {
            cell,
            inv_cell: 1.0 / cell.max(f64::MIN_POSITIVE),
            entries: Vec::new(),
            grid: HashMap::new(),
            by_vertex: HashMap::new(),
            block_order: HashMap::new(),
            loose: RefCell::new((Vec::new(), Vec::new())),
        }
    }

    pub(crate) fn set_block_order(&mut self, block: PaveBlockId, order: u32) {
        self.block_order.insert(block, order);
    }

    pub(crate) fn block_order(&self, block: PaveBlockId) -> Option<u32> {
        self.block_order.get(&block).copied()
    }

    pub(crate) fn push(&mut self, key: (u32, u32), vertex: VertexId, pos: Point3, tolerance: f64) {
        let at = self.entries.len();
        self.entries.push(BallEntry { key, vertex });
        self.grid
            .entry(cell_of(pos, self.inv_cell))
            .or_default()
            .push(at);
        self.by_vertex.entry(vertex).or_default().push(at);
        let tight = tolerance <= self.cell && pos_is_finite(pos);
        let loose = !tight;
        let mut held = self.loose.borrow_mut();
        held.1.push(loose);
        if loose {
            held.0.push(at);
        }
    }

    /// Record that `vertex`'s tolerance rose to `tolerance`.
    pub(crate) fn note_tolerance(&self, vertex: VertexId, tolerance: f64) {
        if tolerance <= self.cell {
            return;
        }
        let Some(at) = self.by_vertex.get(&vertex) else {
            return;
        };
        let mut held = self.loose.borrow_mut();
        for &i in at {
            if !held.1[i] {
                held.1[i] = true;
                held.0.push(i);
            }
        }
    }

    /// Whether [`Self::nearest_within`] can answer for `point` and `reach`:
    /// a reach past one cell, or a point that is not finite, needs the scan.
    pub(crate) fn answers(&self, point: Point3, reach: f64) -> bool {
        reach <= self.cell && pos_is_finite(point)
    }

    /// The nearest pave vertex within the larger of its tolerance and
    /// `reach` of `point`, among those whose ball is wider than `linear`,
    /// the first in scan order among equals. Only for a query the index
    /// [`answers`](Self::answers).
    pub(crate) fn nearest_within(
        &self,
        topo: &brepkit_topology::Topology,
        point: Point3,
        reach: f64,
        linear: f64,
    ) -> Option<VertexId> {
        let (cx, cy, cz) = cell_of(point, self.inv_cell);
        let mut found: Vec<usize> = self.loose.borrow().0.clone();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(at) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        found.extend_from_slice(at);
                    }
                }
            }
        }
        found.sort_unstable_by_key(|&i| self.entries[i].key);
        found.dedup();
        let mut best: Option<(f64, VertexId)> = None;
        for i in found {
            let vertex = self.entries[i].vertex;
            let Ok(v) = topo.vertex(vertex) else {
                continue;
            };
            let ball = v.tolerance().max(reach);
            let d = (v.point() - point).length();
            if ball > linear && d <= ball && best.is_none_or(|(b, _)| d < b) {
                best = Some((d, vertex));
            }
        }
        best.map(|(_, v)| v)
    }
}

fn pos_is_finite(p: Point3) -> bool {
    p.x().is_finite() && p.y().is_finite() && p.z().is_finite()
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

    /// The ball index must answer as the scan over every vertex in its order
    /// does: the nearest vertex within the larger of its tolerance and the
    /// reach, wider than the linear tolerance, the first among equals; with
    /// a tolerance raised after the build, an extra pave added later, and a
    /// far vertex whose own ball reaches the point.
    #[test]
    fn ball_index_matches_the_scan() {
        let linear = 1e-7;
        let cell = 1e-4;
        let mut topo = Topology::new();
        let mut all: Vec<((u32, u32), VertexId)> = Vec::new();
        let mut index = PaveBallIndex::new(cell);
        let put = |topo: &mut Topology,
                   index: &mut PaveBallIndex,
                   all: &mut Vec<((u32, u32), VertexId)>,
                   key: (u32, u32),
                   p: Point3,
                   t: f64| {
            let v = topo.add_vertex(Vertex::new(p, t));
            index.push(key, v, p, t);
            all.push((key, v));
            v
        };
        // Two vertices the same distance from the query, and a tight one
        // beside them.
        let q = Point3::new(10.0, 10.0, 10.0);
        put(
            &mut topo,
            &mut index,
            &mut all,
            (3, 0),
            Point3::new(10.0 + 2e-6, 10.0, 10.0),
            5e-6,
        );
        put(
            &mut topo,
            &mut index,
            &mut all,
            (1, 1),
            Point3::new(10.0 - 2e-6, 10.0, 10.0),
            5e-6,
        );
        put(
            &mut topo,
            &mut index,
            &mut all,
            (0, 0),
            Point3::new(10.0, 10.0 + 5e-8, 10.0),
            linear,
        );
        // A vertex a millimetre off whose tolerance will be raised past it.
        let raised = put(
            &mut topo,
            &mut index,
            &mut all,
            (2, 0),
            Point3::new(11.0, 10.0, 10.0),
            1e-6,
        );
        // A far vertex built loose.
        put(
            &mut topo,
            &mut index,
            &mut all,
            (5, 2),
            Point3::new(10.0, 12.0, 10.0),
            3.0,
        );

        let scan = |topo: &Topology, all: &[((u32, u32), VertexId)], q: Point3, reach: f64| {
            let mut sorted = all.to_vec();
            sorted.sort_by_key(|e| e.0);
            let mut best: Option<(f64, VertexId)> = None;
            for (_, v) in sorted {
                let vx = topo.vertex(v).unwrap();
                let ball = vx.tolerance().max(reach);
                let d = (vx.point() - q).length();
                if ball > linear && d <= ball && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, v));
                }
            }
            best.map(|(_, v)| v)
        };
        // A second point a unit off, which only the loose vertices' balls
        // reach.
        let q2 = Point3::new(10.0, 10.0, 11.0);
        let check = |topo: &Topology, index: &PaveBallIndex, all: &[((u32, u32), VertexId)]| {
            for p in [q, q2] {
                for reach in [linear, 3e-6, cell] {
                    assert!(index.answers(p, reach));
                    assert_eq!(
                        index.nearest_within(topo, p, reach, linear),
                        scan(topo, all, p, reach)
                    );
                }
                assert!(!index.answers(p, 2.0 * cell));
            }
        };
        check(&topo, &index, &all);

        // Raise the far vertex's tolerance past its distance; the index
        // hears of it.
        topo.vertex_mut(raised).unwrap().set_tolerance(1.5);
        index.note_tolerance(raised, 1.5);
        check(&topo, &index, &all);
        assert_eq!(
            index.nearest_within(&topo, q2, linear, linear),
            Some(raised)
        );

        // An extra pave joining after the build, nearer than the rest.
        put(
            &mut topo,
            &mut index,
            &mut all,
            (4, 3),
            Point3::new(10.0, 10.0, 10.0 + 1e-6),
            2e-6,
        );
        check(&topo, &index, &all);
        assert_eq!(
            index.nearest_within(&topo, q, linear, linear),
            Some(all.last().unwrap().1)
        );
    }
}
