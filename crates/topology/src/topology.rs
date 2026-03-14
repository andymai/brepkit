//! Central context holding all topological arenas.
//!
//! [`Topology`] is the single owner of every arena. All operations that
//! create or query topological entities take a reference to this struct.

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
#[derive(Debug, Default, Clone)]
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

    // ── Single-entity lookup (by ID → Result) ─────────────────────

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

    // ── Allocation + arena access + count + index reconstruction ──

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

    // ── PCurve registry ───────────────────────────────────────────

    /// Returns a shared reference to the pcurve registry.
    #[must_use]
    pub fn pcurves(&self) -> &PCurveRegistry {
        &self.pcurves
    }

    /// Returns an exclusive reference to the pcurve registry.
    pub fn pcurves_mut(&mut self) -> &mut PCurveRegistry {
        &mut self.pcurves
    }

    // ── Adjacency ─────────────────────────────────────────────────

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

        // Add more entities after the snapshot
        topo.add_vertex(Vertex::new(Point3::new(4.0, 5.0, 6.0), 1e-7));
        assert_eq!(topo.num_vertices(), 2);

        // Snapshot still has exactly 1 vertex
        assert_eq!(snapshot.num_vertices(), 1);
        let v = snapshot.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn restore_from_clone() {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let snapshot = topo.clone();

        // Mutate after snapshot
        topo.add_vertex(Vertex::new(Point3::new(9.0, 9.0, 9.0), 1e-7));

        // Restore from snapshot
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
}
