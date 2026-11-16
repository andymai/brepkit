//! Links up from each edge to the wires that hold it, and from each wire to
//! the faces that hold it.

use std::collections::HashMap;

use crate::arena::Arena;
use crate::edge::EdgeId;
use crate::face::{Face, FaceId};
use crate::wire::{Wire, WireId};

/// An empty slot.
const NONE: usize = usize::MAX;

/// The links of a [`Topology`](crate::Topology), by arena index.
///
/// The links of each wire and face are the ones it holds now, with one
/// exception: the wire or face that `wire_mut` or `face_mut` gave out last
/// has no links, and a reader reads it as it is now. `add_wire` and
/// `add_face` record the links of a new entity, [`Owners::hand_out`] removes
/// those of the entity it gives out and records those of the one before, and
/// a rollback removes the links of each value it drops or puts back and
/// records those of each value it restores. A wire that holds an edge two
/// times has two links to it.
#[derive(Debug, Default, Clone)]
pub struct Owners {
    /// The first two wires of each edge.
    edge_wires: Vec<[usize; 2]>,
    /// The other wires of an edge.
    more_edge_wires: HashMap<usize, Vec<usize>>,
    /// The first face of each wire.
    wire_faces: Vec<[usize; 1]>,
    /// The other faces of a wire.
    more_wire_faces: HashMap<usize, Vec<usize>>,
    /// The wire or face that `wire_mut` or `face_mut` gave out last.
    pending: Option<Pending>,
}

/// A wire or face that an edit can change.
#[derive(Debug, Clone, Copy)]
pub enum Pending {
    Wire(WireId),
    Face(FaceId),
}

impl Owners {
    /// Records the links of the wire or face given out last, removes those
    /// of `next`, and makes `next` the one given out last.
    ///
    /// The `&mut` borrow of the one given out last ends before this call, so
    /// its edit is complete.
    pub fn hand_out(&mut self, next: Pending, wires: &Arena<Wire>, faces: &Arena<Face>) {
        self.fold(wires, faces);
        match next {
            Pending::Wire(wire) => {
                if let Some(data) = wires.get(wire) {
                    self.remove_wire(wire, data);
                }
            }
            Pending::Face(face) => {
                if let Some(data) = faces.get(face) {
                    self.remove_face(face, data);
                }
            }
        }
        self.pending = Some(next);
    }

    /// Makes room for the links of `edges` more edges and `wires` more wires.
    pub fn reserve(&mut self, edges: usize, wires: usize) {
        self.edge_wires.reserve(edges);
        self.wire_faces.reserve(wires);
    }

    /// Records the links of the wire or face given out last.
    pub fn fold(&mut self, wires: &Arena<Wire>, faces: &Arena<Face>) {
        match self.pending.take() {
            Some(Pending::Wire(wire)) => {
                if let Some(data) = wires.get(wire) {
                    self.add_wire(wire, data);
                }
            }
            Some(Pending::Face(face)) => {
                if let Some(data) = faces.get(face) {
                    self.add_face(face, data);
                }
            }
            None => {}
        }
    }

    /// Records a link from each edge of `data` to `wire`.
    pub fn add_wire(&mut self, wire: WireId, data: &Wire) {
        for oe in data.edges() {
            link(
                &mut self.edge_wires,
                &mut self.more_edge_wires,
                oe.edge().index(),
                wire.index(),
            );
        }
    }

    /// Removes the links from the edges of `data` to `wire`.
    pub fn remove_wire(&mut self, wire: WireId, data: &Wire) {
        for oe in data.edges() {
            unlink(
                &mut self.edge_wires,
                &mut self.more_edge_wires,
                oe.edge().index(),
                wire.index(),
            );
        }
    }

    /// Records a link from each wire of `data` to `face`.
    pub fn add_face(&mut self, face: FaceId, data: &Face) {
        for wire in face_wires(data) {
            link(
                &mut self.wire_faces,
                &mut self.more_wire_faces,
                wire.index(),
                face.index(),
            );
        }
    }

    /// Removes the links from the wires of `data` to `face`.
    pub fn remove_face(&mut self, face: FaceId, data: &Face) {
        for wire in face_wires(data) {
            unlink(
                &mut self.wire_faces,
                &mut self.more_wire_faces,
                wire.index(),
                face.index(),
            );
        }
    }

    /// The faces that hold `edge` in one of their wires, in id order.
    ///
    /// A link counts only where the entities hold it now. So a link that is
    /// not true cannot give a wrong face.
    pub fn faces_of_edge(
        &self,
        edge: EdgeId,
        wires: &Arena<Wire>,
        faces: &Arena<Face>,
    ) -> Vec<FaceId> {
        let holds_edge = |index: usize| {
            wires
                .id_from_index(index)
                .filter(|&id| wires.get(id).is_some_and(|w| wire_holds(w, edge)))
        };
        let mut held_by: Vec<WireId> = Vec::new();
        let pending_wire = match self.pending {
            Some(Pending::Wire(wire)) => Some(wire.index()),
            _ => None,
        };
        for index in
            linked(&self.edge_wires, &self.more_edge_wires, edge.index()).chain(pending_wire)
        {
            if !held_by.iter().any(|w| w.index() == index)
                && let Some(wire) = holds_edge(index)
            {
                held_by.push(wire);
            }
        }

        let mut found: Vec<FaceId> = Vec::new();
        for wire in held_by {
            for index in linked(&self.wire_faces, &self.more_wire_faces, wire.index()) {
                if let Some(face) = faces.id_from_index(index)
                    && faces
                        .get(face)
                        .is_some_and(|f| face_wires(f).any(|w| w == wire))
                    && !found.contains(&face)
                {
                    found.push(face);
                }
            }
        }
        if let Some(Pending::Face(face)) = self.pending
            && let Some(data) = faces.get(face)
            && face_wires(data).any(|w| holds_edge(w.index()).is_some())
            && !found.contains(&face)
        {
            found.push(face);
        }
        found.sort_unstable_by_key(|face| face.index());
        found
    }
}

/// Adds a link from `from` to `to`.
fn link<const N: usize>(
    slots: &mut Vec<[usize; N]>,
    more: &mut HashMap<usize, Vec<usize>>,
    from: usize,
    to: usize,
) {
    if slots.len() <= from {
        slots.resize(from + 1, [NONE; N]);
    }
    if let Some(slot) = slots[from].iter_mut().find(|slot| **slot == NONE) {
        *slot = to;
    } else {
        more.entry(from).or_default().push(to);
    }
}

/// Removes each link from `from` to `to`.
fn unlink<const N: usize>(
    slots: &mut [[usize; N]],
    more: &mut HashMap<usize, Vec<usize>>,
    from: usize,
    to: usize,
) {
    if let Some(set) = slots.get_mut(from) {
        for slot in set.iter_mut().filter(|slot| **slot == to) {
            *slot = NONE;
        }
    }
    if let Some(rest) = more.get_mut(&from) {
        rest.retain(|&other| other != to);
        if rest.is_empty() {
            more.remove(&from);
        }
    }
}

/// The links of `from`.
fn linked<'a, const N: usize>(
    slots: &'a [[usize; N]],
    more: &'a HashMap<usize, Vec<usize>>,
    from: usize,
) -> impl Iterator<Item = usize> + 'a {
    slots
        .get(from)
        .into_iter()
        .flatten()
        .copied()
        .filter(|&to| to != NONE)
        .chain(more.get(&from).into_iter().flatten().copied())
}

fn wire_holds(wire: &Wire, edge: EdgeId) -> bool {
    wire.edges().iter().any(|oe| oe.edge() == edge)
}

fn face_wires(face: &Face) -> impl Iterator<Item = WireId> + '_ {
    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use brepkit_math::vec::{Point3, Vec3};

    use crate::Topology;
    use crate::edge::{Edge, EdgeCurve, EdgeId};
    use crate::face::{Face, FaceId, FaceSurface};
    use crate::vertex::Vertex;
    use crate::wire::{OrientedEdge, Wire, WireId};

    /// The faces of each edge, read from every face of the topology.
    fn walk(topo: &Topology) -> BTreeMap<usize, Vec<FaceId>> {
        let mut faces: BTreeMap<usize, Vec<FaceId>> = BTreeMap::new();
        for (face, data) in topo.faces().iter() {
            for wire in super::face_wires(data) {
                let Some(wire) = topo.wires().get(wire) else {
                    continue;
                };
                for oe in wire.edges() {
                    let list = faces.entry(oe.edge().index()).or_default();
                    if !list.contains(&face) {
                        list.push(face);
                    }
                }
            }
        }
        for list in faces.values_mut() {
            list.sort_unstable_by_key(|face| face.index());
        }
        faces
    }

    fn assert_links(topo: &Topology, context: &str) {
        let walked = walk(topo);
        for (edge, _) in topo.edges().iter() {
            let expected = walked.get(&edge.index()).cloned().unwrap_or_default();
            assert_eq!(
                topo.faces_of_edge(edge),
                expected,
                "{context}: edge {}",
                edge.index()
            );
        }
        assert_exact(topo, context);
    }

    /// The stored links are the ones each wire and face holds now, apart
    /// from the one given out last, which has none.
    fn assert_exact(topo: &Topology, context: &str) {
        let owners = topo.owners();
        let mut edge_wires: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (wire, data) in topo.wires().iter() {
            if !matches!(owners.pending, Some(super::Pending::Wire(w)) if w == wire) {
                for oe in data.edges() {
                    edge_wires
                        .entry(oe.edge().index())
                        .or_default()
                        .push(wire.index());
                }
            }
        }
        let mut wire_faces: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (face, data) in topo.faces().iter() {
            if !matches!(owners.pending, Some(super::Pending::Face(f)) if f == face) {
                for wire in super::face_wires(data) {
                    wire_faces
                        .entry(wire.index())
                        .or_default()
                        .push(face.index());
                }
            }
        }
        let stored = |slots: &[[usize; 2]], more, from| {
            let mut links: Vec<usize> = super::linked(slots, more, from).collect();
            links.sort_unstable();
            links
        };
        for from in 0..owners.edge_wires.len().max(topo.num_edges()) {
            let mut expected = edge_wires.remove(&from).unwrap_or_default();
            expected.sort_unstable();
            assert_eq!(
                stored(&owners.edge_wires, &owners.more_edge_wires, from),
                expected,
                "{context}: the stored wires of edge {from}"
            );
        }
        for from in 0..owners.wire_faces.len().max(topo.num_wires()) {
            let mut expected = wire_faces.remove(&from).unwrap_or_default();
            expected.sort_unstable();
            let mut links: Vec<usize> =
                super::linked(&owners.wire_faces, &owners.more_wire_faces, from).collect();
            links.sort_unstable();
            assert_eq!(
                links, expected,
                "{context}: the stored faces of wire {from}"
            );
        }
    }

    fn plane() -> FaceSurface {
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        }
    }

    fn square(topo: &mut Topology) -> (FaceId, WireId, [EdgeId; 4]) {
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .map(|(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)));
        let edges = [0, 1, 2, 3]
            .map(|i| topo.add_edge(Edge::new(corners[i], corners[(i + 1) % 4], EdgeCurve::Line)));
        let wire = topo
            .add_wire(Wire::new(edges.map(|e| OrientedEdge::new(e, true)).to_vec(), true).unwrap());
        let face = topo.add_face(Face::new(wire, Vec::new(), plane()));
        (face, wire, edges)
    }

    #[test]
    fn an_edge_on_two_faces_names_both() {
        let mut topo = Topology::new();
        let (first, wire, edges) = square(&mut topo);
        let second = topo.add_face(Face::new(wire, Vec::new(), plane()));
        assert_eq!(topo.faces_of_edge(edges[0]), vec![first, second]);
        assert_links(&topo, "two faces on one wire");
    }

    #[test]
    fn an_edit_moves_the_links_while_its_borrow_is_the_last() {
        let mut topo = Topology::new();
        let (face, wire, edges) = square(&mut topo);
        let (_, _, others) = square(&mut topo);
        topo.wire_mut(wire).unwrap().edges_mut()[0] = OrientedEdge::new(others[0], true);
        assert!(topo.faces_of_edge(edges[0]).is_empty());
        assert!(topo.faces_of_edge(others[0]).contains(&face));
        assert_links(&topo, "a wire edit before the next borrow");

        let (_, hole, hole_edges) = square(&mut topo);
        topo.face_mut(face).unwrap().inner_wires_mut().push(hole);
        assert!(topo.faces_of_edge(hole_edges[2]).contains(&face));
        assert_links(&topo, "a face edit before the next borrow");
    }

    #[test]
    fn a_failed_transaction_takes_its_links_with_it() {
        let mut topo = Topology::new();
        let (face, wire, edges) = square(&mut topo);
        let result: Result<(), ()> = topo.transaction(|topo| {
            let (_, _, others) = square(topo);
            topo.wire_mut(wire).unwrap().edges_mut()[1] = OrientedEdge::new(others[1], true);
            Err(())
        });
        assert!(result.is_err());
        assert_eq!(topo.faces_of_edge(edges[1]), vec![face]);
        assert_links(&topo, "after the rollback");
        let (again, _, reused) = square(&mut topo);
        assert_eq!(topo.faces_of_edge(reused[1]), vec![again]);
        assert_links(&topo, "ids that the rollback freed, in use again");
    }

    /// A small generator, so that each seed gives the same steps.
    struct Steps(u64);

    impl Steps {
        fn next(&mut self, below: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            usize::try_from(self.0 % below as u64).unwrap()
        }

        fn edge(&mut self, topo: &Topology) -> EdgeId {
            topo.edge_id_from_index(self.next(topo.num_edges()))
                .unwrap()
        }

        fn wire(&mut self, topo: &Topology) -> WireId {
            topo.wire_id_from_index(self.next(topo.num_wires()))
                .unwrap()
        }

        fn face(&mut self, topo: &Topology) -> FaceId {
            topo.face_id_from_index(self.next(topo.num_faces()))
                .unwrap()
        }
    }

    fn step(topo: &mut Topology, steps: &mut Steps, depth: usize) {
        match steps.next(if depth < 2 { 10 } else { 8 }) {
            0 => {
                square(topo);
            }
            1 => {
                let count = 1 + steps.next(4);
                let edges = (0..count)
                    .map(|_| OrientedEdge::new(steps.edge(topo), steps.next(2) == 0))
                    .collect();
                topo.add_wire(Wire::new(edges, false).unwrap());
            }
            2 => {
                let outer = steps.wire(topo);
                let inner = (0..steps.next(3)).map(|_| steps.wire(topo)).collect();
                topo.add_face(Face::new(outer, inner, plane()));
            }
            3 | 4 => {
                let wire = steps.wire(topo);
                let edge = steps.edge(topo);
                let forward = steps.next(2) == 0;
                let edges = topo.wire_mut(wire).unwrap().edges_mut();
                let at = steps.next(edges.len());
                edges[at] = OrientedEdge::new(edge, forward);
            }
            5 => {
                let face = steps.face(topo);
                let wire = steps.wire(topo);
                topo.face_mut(face).unwrap().set_outer_wire(wire);
            }
            6 => {
                let face = steps.face(topo);
                let wire = steps.wire(topo);
                let clear = steps.next(3) == 0;
                let inner = topo.face_mut(face).unwrap().inner_wires_mut();
                if clear {
                    inner.clear();
                } else {
                    inner.push(wire);
                }
            }
            7 => {
                let face = steps.face(topo);
                let _ = topo.face_mut(face).unwrap();
            }
            8 => {
                let keep = steps.next(2) == 0;
                let _: Result<(), ()> = topo.transaction(|topo| {
                    for _ in 0..=steps.next(5) {
                        step(topo, steps, depth + 1);
                        assert_links(topo, "in a transaction");
                    }
                    if keep { Ok(()) } else { Err(()) }
                });
            }
            _ => {
                let mark = topo.open_checkpoint();
                for _ in 0..=steps.next(5) {
                    step(topo, steps, depth + 1);
                    assert_links(topo, "in a checkpoint");
                }
                if steps.next(2) == 0 {
                    assert!(topo.restore_checkpoint(mark));
                }
                assert!(topo.close_checkpoint(mark));
            }
        }
    }

    #[test]
    fn the_links_match_a_walk_of_every_face_through_edits_and_rollbacks() {
        for seed in 1..=60_u64 {
            let mut topo = Topology::new();
            square(&mut topo);
            let mut steps = Steps(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            for i in 0..150 {
                step(&mut topo, &mut steps, 0);
                assert_links(&topo, &format!("seed {seed}, step {i}"));
            }
        }
    }
}
