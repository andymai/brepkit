//! Central context holding all topological arenas.
//!
//! [`Topology`] is the single owner of every arena. All operations that
//! create or query topological entities take a reference to this struct.

use std::collections::HashMap;

use crate::TopologyError;
use crate::adjacency::AdjacencyIndex;
use crate::arena::Arena;
use crate::compound::{Compound, CompoundId};
use crate::compsolid::{CompSolid, CompSolidId};
use crate::edge::{Edge, EdgeId};
use crate::face::{Face, FaceId};
use crate::pcurve::PCurveRegistry;
use crate::shell::{Shell, ShellId};
use crate::solid::{Solid, SolidId};
use crate::vertex::{Vertex, VertexId};
use crate::wire::{Wire, WireId};

/// Central context owning all topological entity arenas.
///
/// Arena fields are private to enforce invariants through the public API.
/// Use the typed accessor methods for lookups and the `add_*` methods
/// for allocation.
#[derive(Debug, Default)]
pub struct Topology {
    /// All vertices in the model.
    vertices: Arena<Vertex>,
    /// All edges in the model.
    edges: Arena<Edge>,
    /// All wires in the model.
    wires: Arena<Wire>,
    /// All faces in the model.
    faces: Arena<Face>,
    /// All shells in the model.
    shells: Arena<Shell>,
    /// All solids in the model.
    solids: Arena<Solid>,
    /// All compounds in the model.
    compounds: Arena<Compound>,
    /// All comp-solids in the model.
    compsolids: Arena<CompSolid>,
    /// `PCurves`: 2D parametric curves mapping edges to face surface parameters.
    pcurves: PCurveRegistry,
    /// One record for each open [`Topology::transaction`] and
    /// [`Topology::open_checkpoint`], innermost last.
    journals: Vec<Journal>,
    /// The number of checkpoints opened, which names the next one.
    checkpoints_opened: u64,
}

/// A checkpoint that [`Topology::open_checkpoint`] opened: the place of its
/// record among the open records, and its number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    depth: usize,
    number: u64,
}

/// A copy holds the entities and the pcurves. It does not hold the record of
/// an open transaction or checkpoint, because none is open on the copy.
impl Clone for Topology {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            edges: self.edges.clone(),
            wires: self.wires.clone(),
            faces: self.faces.clone(),
            shells: self.shells.clone(),
            solids: self.solids.clone(),
            compounds: self.compounds.clone(),
            compsolids: self.compsolids.clone(),
            pcurves: self.pcurves.clone(),
            journals: Vec::new(),
            checkpoints_opened: 0,
        }
    }
}

/// The number of entities in each arena.
#[derive(Debug, Default, Clone, Copy)]
struct Counts {
    vertices: usize,
    edges: usize,
    wires: usize,
    faces: usize,
    shells: usize,
    solids: usize,
    compounds: usize,
    compsolids: usize,
}

/// What one open transaction or checkpoint needs to put the topology back:
/// the arena sizes at its start, and the first value of each older entity
/// that it changed.
#[derive(Debug, Default, Clone)]
struct Journal {
    /// The number of the checkpoint the record is for, `None` for a
    /// transaction.
    checkpoint: Option<u64>,
    base: Counts,
    pcurve_mark: usize,
    vertices: HashMap<usize, Vertex>,
    edges: HashMap<usize, Edge>,
    wires: HashMap<usize, Wire>,
    faces: HashMap<usize, Face>,
    shells: HashMap<usize, Shell>,
    solids: HashMap<usize, Solid>,
    compounds: HashMap<usize, Compound>,
    compsolids: HashMap<usize, CompSolid>,
}

/// Rolls an open transaction back if a panic unwinds through it.
struct TransactionGuard<'a> {
    topo: &'a mut Topology,
    /// The place of the transaction's record among the open records.
    depth: usize,
    open: bool,
}

impl Drop for TransactionGuard<'_> {
    fn drop(&mut self) {
        if self.open {
            self.topo.end_from(self.depth, false);
        }
    }
}

/// Generates an immutable arena accessor method on [`Topology`].
///
/// Usage: `arena_get!(method_name, arena_field, EntityType, IdType, ErrorVariant)`
macro_rules! arena_get {
    ($method:ident, $field:ident, $T:ty, $Id:ty, $err:ident) => {
        /// Returns a shared reference to the entity with the given ID.
        ///
        /// # Errors
        ///
        /// Returns a not-found error if the ID is invalid.
        pub fn $method(&self, id: $Id) -> Result<&$T, TopologyError> {
            self.$field.get(id).ok_or(TopologyError::$err(id))
        }
    };
}

/// Generates a mutable arena accessor method on [`Topology`].
///
/// Usage: `arena_get_mut!(method_name, arena_field, EntityType, IdType, ErrorVariant)`
macro_rules! arena_get_mut {
    ($method:ident, $field:ident, $T:ty, $Id:ty, $err:ident) => {
        /// Returns an exclusive reference to the entity with the given ID.
        ///
        /// # Errors
        ///
        /// Returns a not-found error if the ID is invalid.
        pub fn $method(&mut self, id: $Id) -> Result<&mut $T, TopologyError> {
            if let Some(journal) = self.journals.last_mut()
                && id.index() < journal.base.$field
                && let Some(value) = self.$field.get(id)
            {
                journal
                    .$field
                    .entry(id.index())
                    .or_insert_with(|| value.clone());
            }
            self.$field.get_mut(id).ok_or(TopologyError::$err(id))
        }
    };
}

/// Generates allocation, read-only arena access, count, and index
/// reconstruction methods for a single entity type.
macro_rules! arena_api {
    (
        add = $add:ident,
        arena = $arena:ident,
        arena_fn = $arena_fn:ident,
        count = $count:ident,
        id_from_index = $id_from_index:ident,
        T = $T:ty,
        Id = $Id:ty
    ) => {
        /// Allocates a new entity in the arena and returns its typed handle.
        pub fn $add(&mut self, value: $T) -> $Id {
            self.$arena.alloc(value)
        }

        /// Returns a shared reference to the arena for iteration and queries.
        #[must_use]
        pub fn $arena_fn(&self) -> &Arena<$T> {
            &self.$arena
        }

        /// Returns the number of entities in this arena.
        #[must_use]
        pub fn $count(&self) -> usize {
            self.$arena.len()
        }

        /// Reconstructs a typed ID from a raw index, returning `None` if
        /// out of bounds. Intended for FFI boundaries (e.g. WASM).
        #[must_use]
        pub fn $id_from_index(&self, index: usize) -> Option<$Id> {
            self.$arena.id_from_index(index)
        }
    };
}

impl Topology {
    /// Creates a new, empty topology context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves capacity for the given number of additional entities in the
    /// six primary entity arenas (vertices, edges, wires, faces, shells, solids).
    ///
    /// Does **not** cover compounds, comp-solids, or the pcurve registry.
    ///
    /// Useful for pre-allocating before bulk insertion (e.g. boolean assembly)
    /// to avoid repeated reallocations.
    pub fn reserve(
        &mut self,
        vertices: usize,
        edges: usize,
        wires: usize,
        faces: usize,
        shells: usize,
        solids: usize,
    ) {
        self.vertices.reserve(vertices);
        self.edges.reserve(edges);
        self.wires.reserve(wires);
        self.faces.reserve(faces);
        self.shells.reserve(shells);
        self.solids.reserve(solids);
    }

    arena_get!(vertex, vertices, Vertex, VertexId, VertexNotFound);
    arena_get_mut!(vertex_mut, vertices, Vertex, VertexId, VertexNotFound);

    arena_get!(edge, edges, Edge, EdgeId, EdgeNotFound);
    arena_get_mut!(edge_mut, edges, Edge, EdgeId, EdgeNotFound);

    arena_get!(wire, wires, Wire, WireId, WireNotFound);
    arena_get_mut!(wire_mut, wires, Wire, WireId, WireNotFound);

    arena_get!(face, faces, Face, FaceId, FaceNotFound);
    arena_get_mut!(face_mut, faces, Face, FaceId, FaceNotFound);

    arena_get!(shell, shells, Shell, ShellId, ShellNotFound);
    arena_get_mut!(shell_mut, shells, Shell, ShellId, ShellNotFound);

    arena_get!(solid, solids, Solid, SolidId, SolidNotFound);
    arena_get_mut!(solid_mut, solids, Solid, SolidId, SolidNotFound);

    arena_get!(compound, compounds, Compound, CompoundId, CompoundNotFound);
    arena_get_mut!(
        compound_mut,
        compounds,
        Compound,
        CompoundId,
        CompoundNotFound
    );

    arena_get!(
        compsolid,
        compsolids,
        CompSolid,
        CompSolidId,
        CompSolidNotFound
    );
    arena_get_mut!(
        compsolid_mut,
        compsolids,
        CompSolid,
        CompSolidId,
        CompSolidNotFound
    );

    arena_api!(
        add = add_vertex,
        arena = vertices,
        arena_fn = vertices,
        count = num_vertices,
        id_from_index = vertex_id_from_index,
        T = Vertex,
        Id = VertexId
    );

    arena_api!(
        add = add_edge,
        arena = edges,
        arena_fn = edges,
        count = num_edges,
        id_from_index = edge_id_from_index,
        T = Edge,
        Id = EdgeId
    );

    arena_api!(
        add = add_wire,
        arena = wires,
        arena_fn = wires,
        count = num_wires,
        id_from_index = wire_id_from_index,
        T = Wire,
        Id = WireId
    );

    arena_api!(
        add = add_face,
        arena = faces,
        arena_fn = faces,
        count = num_faces,
        id_from_index = face_id_from_index,
        T = Face,
        Id = FaceId
    );

    arena_api!(
        add = add_shell,
        arena = shells,
        arena_fn = shells,
        count = num_shells,
        id_from_index = shell_id_from_index,
        T = Shell,
        Id = ShellId
    );

    arena_api!(
        add = add_solid,
        arena = solids,
        arena_fn = solids,
        count = num_solids,
        id_from_index = solid_id_from_index,
        T = Solid,
        Id = SolidId
    );

    arena_api!(
        add = add_compound,
        arena = compounds,
        arena_fn = compounds,
        count = num_compounds,
        id_from_index = compound_id_from_index,
        T = Compound,
        Id = CompoundId
    );

    arena_api!(
        add = add_compsolid,
        arena = compsolids,
        arena_fn = compsolids,
        count = num_compsolids,
        id_from_index = compsolid_id_from_index,
        T = CompSolid,
        Id = CompSolidId
    );

    /// Allocates an empty-result solid: a solid backed by a faceless
    /// [`Shell::empty`].
    ///
    /// Booleans whose algebraic outcome is the empty set (e.g. the
    /// intersection of disjoint solids) return this so the result is a
    /// valid, queryable handle reporting zero faces and zero volume,
    /// distinct from a malformed-input error. A shell cannot otherwise
    /// hold zero faces, so this is the only path that produces one.
    pub fn add_empty_solid(&mut self) -> SolidId {
        let shell = self.add_shell(Shell::empty());
        self.add_solid(Solid::new(shell, Vec::new()))
    }

    /// Returns `true` when `solid` is an empty-result sentinel — its
    /// outer shell is faceless and it has no inner shells (see
    /// [`Topology::add_empty_solid`]).
    #[must_use]
    pub fn is_empty_solid(&self, solid: SolidId) -> bool {
        self.solids.get(solid).is_some_and(|s| {
            s.inner_shells().is_empty()
                && self
                    .shells
                    .get(s.outer_shell())
                    .is_some_and(Shell::is_empty)
        })
    }

    /// Returns a shared reference to the pcurve registry.
    #[must_use]
    pub fn pcurves(&self) -> &PCurveRegistry {
        &self.pcurves
    }

    /// Returns an exclusive reference to the pcurve registry.
    pub fn pcurves_mut(&mut self) -> &mut PCurveRegistry {
        &mut self.pcurves
    }

    /// Runs `edit`, and puts the topology back as it was if `edit` fails.
    ///
    /// On an error, each entity that `edit` changed gets its first value
    /// again, each entity and pcurve that `edit` added is dropped, and each
    /// pcurve that it removed or replaced comes back. An id that `edit` made
    /// is then not valid. The same applies when a panic unwinds through
    /// `edit`.
    ///
    /// The cost is the entities that `edit` changes and adds. It does not
    /// depend on how many entities the topology holds. Transactions can nest.
    ///
    /// The record of the transaction is part of the topology. Do not put
    /// another topology, or another pcurve registry, in the place of this
    /// one in `edit`: the record goes with the old one, and the undo with it.
    ///
    /// # Errors
    ///
    /// Returns the error of `edit`.
    pub fn transaction<T, E>(
        &mut self,
        edit: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, E> {
        let depth = self.journals.len();
        let journal = self.journal(None);
        self.journals.push(journal);
        let mut guard = TransactionGuard {
            topo: self,
            depth,
            open: true,
        };
        let result = edit(guard.topo);
        guard.open = false;
        guard.topo.end_from(depth, result.is_ok());
        result
    }

    /// Starts to record the changes that [`Topology::restore_checkpoint`]
    /// needs to put the topology back as it is now.
    ///
    /// While a checkpoint is open, the first change to each entity keeps a
    /// copy of its old value, and the pcurve registry keeps a record of each
    /// change. A restore costs the entities and pcurves changed and added
    /// since the checkpoint opened, and not the size of the topology.
    /// Checkpoints can nest, and a transaction can run in a checkpoint. A
    /// checkpoint opened in a transaction closes when the transaction ends,
    /// with its changes kept or undone as the transaction's are.
    pub fn open_checkpoint(&mut self) -> Checkpoint {
        self.checkpoints_opened += 1;
        let mark = Checkpoint {
            depth: self.journals.len(),
            number: self.checkpoints_opened,
        };
        let journal = self.journal(Some(mark.number));
        self.journals.push(journal);
        mark
    }

    /// Puts the topology back as it was when `mark` opened: each entity
    /// changed since then gets its old value, and each entity and pcurve
    /// added since then is dropped. The checkpoints opened after `mark`
    /// close, and `mark` stays open.
    ///
    /// Returns `false`, and changes nothing, if `mark` is not open or a
    /// transaction that opened after it is still open. In a transaction
    /// that opened before it, `mark` works on the changes after it alone.
    #[must_use]
    pub fn restore_checkpoint(&mut self, mark: Checkpoint) -> bool {
        if !self.is_open(mark) {
            return false;
        }
        self.end_from(mark.depth, false);
        let journal = self.journal(Some(mark.number));
        self.journals.push(journal);
        true
    }

    /// Closes `mark` and the checkpoints opened after it, and keeps the
    /// changes.
    ///
    /// Returns `false`, and changes nothing, if `mark` is not open or a
    /// transaction that opened after it is still open. In a transaction
    /// that opened before it, `mark` works on the changes after it alone.
    #[must_use]
    pub fn close_checkpoint(&mut self, mark: Checkpoint) -> bool {
        if !self.is_open(mark) {
            return false;
        }
        self.end_from(mark.depth, true);
        true
    }

    /// Closes the record at `depth` and each one opened after it, innermost
    /// first: keeps their changes, or undoes them.
    fn end_from(&mut self, depth: usize, keep: bool) {
        while self.journals.len() > depth {
            self.end_transaction(keep);
        }
    }

    /// Whether `mark` is open, with no transaction open after it. A
    /// transaction's record must stay the innermost until its end.
    fn is_open(&self, mark: Checkpoint) -> bool {
        self.journals
            .get(mark.depth)
            .is_some_and(|journal| journal.checkpoint == Some(mark.number))
            && self
                .journals
                .iter()
                .skip(mark.depth)
                .all(|journal| journal.checkpoint.is_some())
    }

    /// A new record that starts now, for a transaction or the checkpoint
    /// `checkpoint`.
    fn journal(&mut self, checkpoint: Option<u64>) -> Journal {
        Journal {
            checkpoint,
            base: Counts {
                vertices: self.vertices.len(),
                edges: self.edges.len(),
                wires: self.wires.len(),
                faces: self.faces.len(),
                shells: self.shells.len(),
                solids: self.solids.len(),
                compounds: self.compounds.len(),
                compsolids: self.compsolids.len(),
            },
            pcurve_mark: self.pcurves.begin_undo(),
            ..Journal::default()
        }
    }

    /// Closes the innermost transaction or checkpoint: keeps its changes, or
    /// undoes them.
    fn end_transaction(&mut self, keep: bool) {
        let Some(journal) = self.journals.pop() else {
            return;
        };
        if keep {
            if let Some(outer) = self.journals.last_mut() {
                macro_rules! hand_over {
                    ($field:ident) => {
                        for (index, value) in journal.$field {
                            if index < outer.base.$field {
                                outer.$field.entry(index).or_insert(value);
                            }
                        }
                    };
                }
                hand_over!(vertices);
                hand_over!(edges);
                hand_over!(wires);
                hand_over!(faces);
                hand_over!(shells);
                hand_over!(solids);
                hand_over!(compounds);
                hand_over!(compsolids);
            }
        } else {
            macro_rules! put_back {
                ($field:ident) => {
                    self.$field.truncate(journal.base.$field);
                    for (index, value) in journal.$field {
                        if let Some(id) = self.$field.id_from_index(index)
                            && let Some(slot) = self.$field.get_mut(id)
                        {
                            *slot = value;
                        }
                    }
                };
            }
            put_back!(vertices);
            put_back!(edges);
            put_back!(wires);
            put_back!(faces);
            put_back!(shells);
            put_back!(solids);
            put_back!(compounds);
            put_back!(compsolids);
            self.pcurves.undo_to(journal.pcurve_mark);
        }
        if self.journals.is_empty() {
            self.pcurves.end_undo();
        }
    }

    /// Builds an adjacency index for the given solid.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] if any referenced entity does not exist.
    pub fn build_adjacency(&self, solid: SolidId) -> Result<AdjacencyIndex, TopologyError> {
        AdjacencyIndex::build(self, solid)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::Point3;

    use super::*;

    #[test]
    fn allocate_and_lookup_vertex() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let v = topo.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn clone_preserves_entities() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let snapshot = topo.clone();

        topo.add_vertex(Vertex::new(Point3::new(4.0, 5.0, 6.0), 1e-7));
        assert_eq!(topo.num_vertices(), 2);

        assert_eq!(snapshot.num_vertices(), 1);
        let v = snapshot.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn restore_from_clone() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let snapshot = topo.clone();

        topo.add_vertex(Vertex::new(Point3::new(9.0, 9.0, 9.0), 1e-7));

        topo = snapshot;
        assert_eq!(topo.num_vertices(), 1);
        let v = topo.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_id_returns_error() {
        use crate::arena::Id;
        let topo = Topology::new();
        let mut dummy_arena: Arena<Vertex> = Arena::new();
        let vid = dummy_arena.alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 0.0));
        let _ = Id::<Vertex>::index(vid);
        assert!(topo.vertex(vid).is_err());
    }

    #[test]
    fn arena_accessors_and_counts() {
        let mut topo = Topology::new();
        assert_eq!(topo.num_vertices(), 0);
        assert!(topo.vertices().is_empty());

        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));
        assert_eq!(topo.num_vertices(), 1);
        assert!(topo.vertices().get(vid).is_some());
    }

    #[test]
    fn id_from_index_roundtrip() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let reconstructed = topo.vertex_id_from_index(vid.index()).unwrap();
        assert_eq!(reconstructed, vid);
        assert!(topo.vertex_id_from_index(999).is_none());
    }

    #[test]
    fn reserve_preserves_existing_entities() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));
        assert_eq!(topo.num_vertices(), 1);

        topo.reserve(100, 50, 25, 25, 2, 2);
        assert_eq!(topo.num_vertices(), 1);

        let v = topo.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);

        let vid2 = topo.add_vertex(Vertex::new(Point3::new(4.0, 5.0, 6.0), 1e-7));
        assert_eq!(topo.num_vertices(), 2);
        let v2 = topo.vertex(vid2).unwrap();
        assert!((v2.point().x() - 4.0).abs() < f64::EPSILON);
    }

    use crate::edge::EdgeCurve;
    use crate::face::FaceSurface;
    use crate::pcurve::PCurve;
    use crate::wire::OrientedEdge;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2, Vec3};

    fn at(x: f64) -> Vertex {
        Vertex::new(Point3::new(x, 0.0, 0.0), 1e-7)
    }

    fn x_of(topo: &Topology, v: VertexId) -> f64 {
        topo.vertex(v).unwrap().point().x()
    }

    fn pcurve(end: f64) -> PCurve {
        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        PCurve::new(Curve2D::Line(line), 0.0, end)
    }

    fn pcurve_end(topo: &Topology, edge: EdgeId, face: FaceId) -> Option<f64> {
        topo.pcurves().get(edge, face).map(PCurve::t_end)
    }

    fn wire_of(topo: &mut Topology, edges: &[EdgeId]) -> WireId {
        let edges = edges.iter().map(|&e| OrientedEdge::new(e, true)).collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    fn face_on(topo: &mut Topology, wire: WireId, holes: Vec<WireId>) -> FaceId {
        topo.add_face(Face::new(
            wire,
            holes,
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    /// A triangle face, its vertices and its edges.
    fn triangle(topo: &mut Topology) -> (FaceId, [VertexId; 3], [EdgeId; 3]) {
        let v = [at(0.0), at(1.0), at(2.0)].map(|vertex| topo.add_vertex(vertex));
        let e = [(0, 1), (1, 2), (2, 0)]
            .map(|(a, b)| topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line)));
        let wire = wire_of(topo, &e);
        (face_on(topo, wire, vec![]), v, e)
    }

    fn counts(topo: &Topology) -> [usize; 5] {
        [
            topo.num_vertices(),
            topo.num_edges(),
            topo.num_wires(),
            topo.num_faces(),
            topo.pcurves().len(),
        ]
    }

    #[test]
    fn a_restore_puts_back_what_changed_since_its_checkpoint() {
        let mut topo = Topology::new();
        let (face, [v0, ..], [e0, e1, _]) = triangle(&mut topo);
        topo.pcurves_mut().set(e0, face, pcurve(1.0));
        let before = counts(&topo);

        let mark = topo.open_checkpoint();
        for round in 0..2 {
            *topo.vertex_mut(v0).unwrap() = at(5.0 + f64::from(round));
            let made = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
            let wire = wire_of(&mut topo, &[made, e1]);
            let copy = face_on(&mut topo, wire, vec![]);
            topo.pcurves_mut().set(e0, face, pcurve(6.0));
            topo.pcurves_mut().set(made, copy, pcurve(7.0));
            assert!(topo.restore_checkpoint(mark));
            assert_eq!(counts(&topo), before, "round {round}");
            assert!(x_of(&topo, v0).abs() < 1e-12, "round {round}");
            assert_eq!(pcurve_end(&topo, e0, face), Some(1.0), "round {round}");
        }

        assert!(topo.close_checkpoint(mark));
        assert!(!topo.restore_checkpoint(mark));
        *topo.vertex_mut(v0).unwrap() = at(3.0);
        assert!((x_of(&topo, v0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn nested_checkpoints_restore_in_their_order() {
        let mut topo = Topology::new();
        let (_, [v0, ..], _) = triangle(&mut topo);
        let first = topo.open_checkpoint();
        *topo.vertex_mut(v0).unwrap() = at(1.0);
        let second = topo.open_checkpoint();
        let at_second = counts(&topo);
        *topo.vertex_mut(v0).unwrap() = at(2.0);
        topo.add_vertex(at(9.0));

        assert!(topo.restore_checkpoint(second));
        assert!((x_of(&topo, v0) - 1.0).abs() < 1e-12);
        assert_eq!(counts(&topo), at_second);

        *topo.vertex_mut(v0).unwrap() = at(3.0);
        assert!(topo.restore_checkpoint(first));
        assert!(x_of(&topo, v0).abs() < 1e-12);
        assert!(!topo.restore_checkpoint(second));

        // A checkpoint opened in the place of a closed one is a new one.
        assert!(topo.close_checkpoint(first));
        let third = topo.open_checkpoint();
        assert!(!topo.close_checkpoint(first));
        assert!(topo.close_checkpoint(third));
    }

    #[test]
    fn a_transaction_runs_in_a_checkpoint() {
        let mut topo = Topology::new();
        let (face, [v0, ..], [e0, ..]) = triangle(&mut topo);
        let before = counts(&topo);
        let mark = topo.open_checkpoint();

        let failed: Result<(), ()> = topo.transaction(|topo| {
            *topo.vertex_mut(v0).unwrap() = at(5.0);
            assert!(!topo.restore_checkpoint(mark));
            assert!(!topo.close_checkpoint(mark));
            Err(())
        });
        assert!(failed.is_err());
        assert!(x_of(&topo, v0).abs() < 1e-12);

        let kept: Result<(), ()> = topo.transaction(|topo| {
            *topo.vertex_mut(v0).unwrap() = at(7.0);
            topo.add_vertex(at(8.0));
            topo.pcurves_mut().set(e0, face, pcurve(2.0));
            Ok(())
        });
        assert!(kept.is_ok());
        assert!((x_of(&topo, v0) - 7.0).abs() < 1e-12);

        assert!(topo.restore_checkpoint(mark));
        assert_eq!(counts(&topo), before);
        assert!(x_of(&topo, v0).abs() < 1e-12);
        assert_eq!(pcurve_end(&topo, e0, face), None);
        assert!(topo.close_checkpoint(mark));
    }

    #[test]
    fn a_checkpoint_opened_in_a_transaction_ends_with_it() {
        let mut topo = Topology::new();
        let (_, [v0, ..], _) = triangle(&mut topo);
        let before = counts(&topo);

        let mut inner = None;
        let failed: Result<(), ()> = topo.transaction(|topo| {
            *topo.vertex_mut(v0).unwrap() = at(1.0);
            inner = Some(topo.open_checkpoint());
            *topo.vertex_mut(v0).unwrap() = at(2.0);
            topo.add_vertex(at(3.0));
            Err(())
        });
        assert!(failed.is_err());
        assert!(x_of(&topo, v0).abs() < 1e-12);
        assert_eq!(counts(&topo), before);
        assert!(!topo.restore_checkpoint(inner.unwrap()));

        let kept: Result<(), ()> = topo.transaction(|topo| {
            inner = Some(topo.open_checkpoint());
            *topo.vertex_mut(v0).unwrap() = at(4.0);
            Ok(())
        });
        assert!(kept.is_ok());
        assert!((x_of(&topo, v0) - 4.0).abs() < 1e-12);
        assert!(!topo.close_checkpoint(inner.unwrap()));

        // No record stays open: a checkpoint after them restores alone.
        let mark = topo.open_checkpoint();
        *topo.vertex_mut(v0).unwrap() = at(5.0);
        assert!(topo.restore_checkpoint(mark));
        assert!((x_of(&topo, v0) - 4.0).abs() < 1e-12);
        assert!(topo.close_checkpoint(mark));

        // In the transaction, the checkpoint restores and closes on its own
        // changes, and the transaction's failure still undoes them all.
        let failed: Result<(), ()> = topo.transaction(|topo| {
            *topo.vertex_mut(v0).unwrap() = at(7.0);
            let inner = topo.open_checkpoint();
            *topo.vertex_mut(v0).unwrap() = at(8.0);
            topo.add_vertex(at(9.0));
            assert!(topo.restore_checkpoint(inner));
            assert!((x_of(topo, v0) - 7.0).abs() < 1e-12);
            assert_eq!(counts(topo), before);
            *topo.vertex_mut(v0).unwrap() = at(10.0);
            assert!(topo.close_checkpoint(inner));
            Err(())
        });
        assert!(failed.is_err());
        assert!((x_of(&topo, v0) - 4.0).abs() < 1e-12);
        assert_eq!(counts(&topo), before);

        // A panic in the transaction undoes the checkpoint's changes too.
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), ()> = topo.transaction(|topo| {
                topo.open_checkpoint();
                *topo.vertex_mut(v0).unwrap() = at(6.0);
                std::panic::resume_unwind(Box::new("stop"));
            });
        }));
        assert!(unwound.is_err());
        assert!((x_of(&topo, v0) - 4.0).abs() < 1e-12);
        let mark = topo.open_checkpoint();
        assert!(topo.restore_checkpoint(mark));
        assert!(topo.close_checkpoint(mark));
    }
}
