//! Central context holding all topological arenas.
//!
//! [`Topology`] is the single owner of every arena. All operations that
//! create or query topological entities take a reference to this struct.

use crate::TopologyError;
use crate::arena::Arena;
use crate::compound::{Compound, CompoundId};
use crate::edge::{Edge, EdgeId};
use crate::face::{Face, FaceId};
use crate::pcurve::PCurveRegistry;
use crate::shell::{Shell, ShellId};
use crate::solid::{Solid, SolidId};
use crate::vertex::{Vertex, VertexId};
use crate::wire::{Wire, WireId};

/// Central context owning all topological entity arenas.
///
/// Fields are `pub` because every operation needs direct arena access for
/// allocation and lookup; wrapping in getters would be pure boilerplate
/// with no invariant to protect.
#[derive(Debug, Default)]
pub struct Topology {
    /// All vertices in the model.
    pub vertices: Arena<Vertex>,
    /// All edges in the model.
    pub edges: Arena<Edge>,
    /// All wires in the model.
    pub wires: Arena<Wire>,
    /// All faces in the model.
    pub faces: Arena<Face>,
    /// All shells in the model.
    pub shells: Arena<Shell>,
    /// All solids in the model.
    pub solids: Arena<Solid>,
    /// All compounds in the model.
    pub compounds: Arena<Compound>,
    /// `PCurves`: 2D parametric curves mapping edges to face surface parameters.
    pub pcurves: PCurveRegistry,
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

impl Topology {
    /// Creates a new, empty topology context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::Point3;

    use super::*;

    #[test]
    fn allocate_and_lookup_vertex() {
        let mut topo = Topology::new();
        let vid = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

        let v = topo.vertex(vid).unwrap();
        assert!((v.point().x() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_id_returns_error() {
        use crate::arena::Id;
        let topo = Topology::new();
        // Fabricate an ID that doesn't exist — index 999.
        // Safety: Id is just a usize + PhantomData, we construct one via alloc trick.
        let mut dummy_arena: Arena<Vertex> = Arena::new();
        // Alloc 1000 entries would be wasteful, instead just test with the empty topology.
        // Any ID will fail because there are no vertices.
        let vid = dummy_arena.alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 0.0));
        // vid has index 0, but the *topology's* arena is empty, so lookup should fail.
        let _ = Id::<Vertex>::index(vid); // just to suppress unused import warning
        assert!(topo.vertex(vid).is_err());
    }
}
