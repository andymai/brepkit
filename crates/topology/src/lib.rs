//! # brepkit-topology
//!
//! Arena-allocated boundary representation (B-Rep) data structures.
//! Layer L1, depending only on `brepkit-math`.
//!
//! # Topology and geometry are separate
//!
//! A B-Rep solid is defined by its boundary: the surfaces, edges, and vertices
//! that form its skin. brepkit keeps *how things connect* apart from *where
//! they are in space*.
//!
//! - **Topology**: [`Vertex`](vertex::Vertex) to [`Edge`](edge::Edge) to
//!   [`Wire`](wire::Wire) to [`Face`](face::Face) to [`Shell`](shell::Shell)
//!   to [`Solid`](solid::Solid).
//! - **Geometry**: points, curves, and surfaces, all owned by `brepkit-math`.
//!
//! A [`Face`](face::Face) knows which wires bound it (topology) and which
//! [`FaceSurface`](face::FaceSurface) defines its shape (geometry). Keeping
//! the two apart is what lets a boolean reason about connectivity without
//! re-deriving it from coordinates every time.
//!
//! # Arena allocation
//!
//! Every entity lives in a central [`Arena`] and is referenced by a typed
//! [`Id<T>`](arena::Id) handle rather than a pointer or an `Rc`. This keeps
//! traversal cache-friendly, drops reference-counting overhead, gives O(1)
//! lookup, and makes ownership unambiguous: the arena owns everything, and a
//! handle is just an index.
//!
//! The consequence to know about is aliasing. You cannot hold a shared borrow
//! of the arena while taking a mutable one, so read what you need into locals
//! first, then allocate:
//!
//! ```
//! use brepkit_math::vec::Point3;
//! use brepkit_topology::Topology;
//! use brepkit_topology::vertex::Vertex;
//!
//! let mut topo = Topology::new();
//! let original = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));
//!
//! // Snapshot the read, then allocate. Doing both in one expression would
//! // borrow the arena immutably and mutably at the same time.
//! let position = topo.vertex(original)?.point();
//! let copy = topo.add_vertex(Vertex::new(position, 1e-7));
//!
//! assert_eq!(topo.vertex(copy)?.point().x(), 1.0);
//! # Ok::<(), brepkit_topology::TopologyError>(())
//! ```
//!
//! # Surfaces and curves are enums
//!
//! [`FaceSurface`](face::FaceSurface) is one of `Plane`, `Cylinder`, `Cone`,
//! `Sphere`, `Torus`, or `Nurbs`. [`EdgeCurve`](edge::EdgeCurve) is one of
//! `Line`, `Circle`, `Ellipse`, or `NurbsCurve`. Analytic types are
//! deliberately special-cased rather than collapsed into NURBS:
//!
//! - Operations preserve them. A cylinder cut by a plane stays a cylinder, so
//!   face counts stay flat across chained booleans instead of growing at every
//!   step.
//! - Intersections take exact closed-form paths where a pair allows one,
//!   falling back to NURBS marching only when no analytic solution exists.
//! - STEP export writes them as native surface entities, so a round-trip is
//!   lossless rather than an approximation.
//!
//! Both enums are exhaustive, with no `_ =>` wildcards in production code.
//! That is a deliberate trade: adding a variant is a breaking change for every
//! downstream matcher, but the compiler finds every site that needs updating.
//! Prefer the delegate methods (`evaluate`, `normal`, `type_tag`, and the
//! rest, defined in `brepkit_math::traits`) over matching variants directly,
//! since code that goes through a delegate keeps compiling when a variant is
//! added.
//!
//! # A solid is not just its outer shell
//!
//! This is the trap that costs the most time. A [`Solid`](solid::Solid) has an
//! outer shell *and* zero or more inner shells, which are the cavity walls
//! left by a hollowing operation or a boolean cut that opened a void. Code
//! that reaches through `outer_shell()` and iterates its faces compiles, runs,
//! and gives the right answer on every model without a cavity. On a hollow
//! part it silently skips the interior and reports a volume or a face count
//! that is quietly wrong. (A bounding box survives, because a cavity sits
//! inside the outer shell and cannot extend it.)
//!
//! Use [`explorer::solid_faces`], which flattens outer and inner shells into
//! one list:
//!
//! ```
//! use brepkit_topology::Topology;
//! use brepkit_topology::explorer::solid_faces;
//!
//! let mut topo = Topology::new();
//! let solid = topo.add_empty_solid();
//!
//! // Covers cavity faces too. Walking `outer_shell()` by hand does not.
//! let faces = solid_faces(&topo, solid)?;
//! assert!(faces.is_empty());
//! # Ok::<(), brepkit_topology::TopologyError>(())
//! ```
//!
//! The exception is work that is genuinely per-shell: orientation fixes,
//! sewing, and any guard whose job is to reason about one shell at a time.
//! Those should keep iterating shell by shell. The rule is about scope. If the
//! question is about the solid (how many faces, what does it weigh, what type
//! is this feature), flatten. If the question is about a shell, do not.
//!
//! [`explorer::solid_edges`] and [`explorer::solid_vertices`] follow the same
//! convention.
//!
//! # See also
//!
//! - [`brepkit_math`](https://docs.rs/brepkit-math): the geometry these
//!   structures carry, and the tolerance model they compare with.
//! - [`brepkit_operations`](https://docs.rs/brepkit-operations): the modeling
//!   operations that build and consume this topology.
//! - [brepjs.dev](https://brepjs.dev/concepts/topology): the same hierarchy
//!   from the TypeScript side.

pub mod adjacency;
pub mod arena;
pub mod builder;
pub mod compound;
pub mod compsolid;
pub mod edge;
pub mod explorer;
pub mod face;
pub mod orientation;

pub mod pcurve;
pub mod shell;
pub mod solid;
#[cfg(feature = "test-utils")]
pub mod test_utils;
pub mod topology;
pub mod validation;
pub mod vertex;
pub mod wire;

pub use arena::Arena;
pub use compound::CompoundId;
pub use compsolid::CompSolidId;
pub use edge::EdgeId;
pub use face::FaceId;
pub use shell::ShellId;
pub use solid::SolidId;
pub use topology::Topology;
pub use vertex::VertexId;
pub use wire::{OrientedEdge, WireId};

/// Errors from topology operations.
#[derive(Debug, thiserror::Error)]
pub enum TopologyError {
    /// A referenced vertex ID does not exist in the arena.
    #[error("vertex {0:?} not found")]
    VertexNotFound(vertex::VertexId),

    /// A referenced edge ID does not exist in the arena.
    #[error("edge {0:?} not found")]
    EdgeNotFound(edge::EdgeId),

    /// A referenced wire ID does not exist in the arena.
    #[error("wire {0:?} not found")]
    WireNotFound(wire::WireId),

    /// A referenced face ID does not exist in the arena.
    #[error("face {0:?} not found")]
    FaceNotFound(face::FaceId),

    /// A referenced shell ID does not exist in the arena.
    #[error("shell {0:?} not found")]
    ShellNotFound(shell::ShellId),

    /// A referenced solid ID does not exist in the arena.
    #[error("solid {0:?} not found")]
    SolidNotFound(solid::SolidId),

    /// A referenced compound ID does not exist in the arena.
    #[error("compound {0:?} not found")]
    CompoundNotFound(compound::CompoundId),

    /// A referenced comp-solid ID does not exist in the arena.
    #[error("compsolid {0:?} not found")]
    CompSolidNotFound(compsolid::CompSolidId),

    /// A wire does not form a closed loop.
    #[error("wire is not closed")]
    WireNotClosed,

    /// The topology is not manifold.
    #[error("non-manifold topology: {reason}")]
    NonManifold {
        /// Description of the manifold violation.
        reason: String,
    },

    /// An empty collection was provided where at least one element is required.
    #[error("empty {entity} — at least one element is required")]
    Empty {
        /// The kind of entity that was empty.
        entity: &'static str,
    },

    /// A wire's edge geometry does not lie within tolerance of any single
    /// plane, so a planar face cannot be constructed from it.
    #[error("wire is not planar")]
    NotPlanar,
}
