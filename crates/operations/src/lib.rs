//! # brepkit-operations
//!
//! CAD modeling operations for B-Rep solids, and the entry point for Rust
//! consumers of brepkit. Layer L3, depending on `brepkit-math`,
//! `brepkit-topology`, `brepkit-geometry`, `brepkit-algo`, `brepkit-blend`,
//! `brepkit-heal`, `brepkit-check`, `brepkit-offset`, and `brepkit-sketch`.
//!
//! # Getting started
//!
//! ```
//! use brepkit_operations::boolean::{boolean, BooleanOp};
//! use brepkit_operations::measure::solid_volume;
//! use brepkit_operations::primitives::{make_box, make_cylinder};
//! use brepkit_topology::Topology;
//!
//! let mut topo = Topology::new();
//!
//! // Primitives are anchored at the origin, so this cylinder rounds off the
//! // block's corner. Use `transform_solid` to place it somewhere else.
//! let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
//! let cutter = make_cylinder(&mut topo, 5.0, 15.0)?;
//! let notched = boolean(&mut topo, BooleanOp::Cut, block, cutter)?;
//!
//! // A quarter-cylinder of radius 5 and height 10 is gone from the corner.
//! let expected = 30.0 * 20.0 * 10.0 - 0.25 * std::f64::consts::PI * 25.0 * 10.0;
//! let volume = solid_volume(&topo, notched, 0.01)?;
//! assert!((volume - expected).abs() / expected < 1e-3);
//! # Ok::<(), brepkit_operations::OperationsError>(())
//! ```
//!
//! # Conventions
//!
//! Every operation takes the [`Topology`](brepkit_topology::Topology) arena as
//! `&mut` and returns a typed handle into it, so results compose without
//! copying geometry. Nothing panics: every public operation returns a
//! [`Result`], and `unwrap`, `expect`, and `panic!` are denied by lint across
//! the workspace.
//!
//! Primitives are anchored at the origin. Place them with
//! [`transform`] rather than expecting a position argument.
//!
//! # Exact geometry, and when it degrades
//!
//! Booleans run on an exact path that preserves analytic and NURBS surfaces.
//! A cylinder cut by a plane stays a cylinder, so face counts stay flat across
//! chained operations instead of compounding: a nine-step compound boolean
//! settles around 72 faces where a mesh-based approach would reach several
//! thousand.
//!
//! Some configurations defeat that path and fall back to a mesh-based boolean
//! built on co-refinement. The usual causes are coincident-face contact,
//! coaxial analytic surfaces, razor-thin geometry, and very high face counts.
//! The fallback returns a usable, non-degenerate solid, but the curved faces
//! come back tessellated and the result is not guaranteed watertight.
//!
//! The fallback does not announce itself in the return value, which matters
//! most for export pipelines: a STEP file written from a fallback result
//! carries triangles where it should carry a cylinder. Snapshot
//! [`boolean::mesh_fallback_count`] around the chain and refuse the output
//! when it grew.
//!
//! ```
//! use brepkit_operations::boolean::{boolean, mesh_fallback_count, BooleanOp};
//! use brepkit_operations::primitives::{make_box, make_cylinder};
//! use brepkit_operations::validate::validate_solid;
//! use brepkit_topology::Topology;
//!
//! let mut topo = Topology::new();
//! let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
//! let cutter = make_cylinder(&mut topo, 5.0, 15.0)?;
//!
//! let before = mesh_fallback_count();
//! let notched = boolean(&mut topo, BooleanOp::Cut, block, cutter)?;
//!
//! // This cut takes the exact path, so the counter is unmoved and the
//! // rounded wall is still a real cylinder.
//! assert_eq!(mesh_fallback_count(), before);
//!
//! // Topological checks: wire closure, shell watertightness, Euler
//! // characteristic, face orientation, and duplicate faces.
//! assert!(validate_solid(&topo, notched)?.is_valid());
//! # Ok::<(), brepkit_operations::OperationsError>(())
//! ```
//!
//! # Verifying a result
//!
//! Three checks, in increasing cost, and they catch different things:
//!
//! 1. [`validate::validate_solid`] reports topological defects: an unclosed
//!    wire, a shell with a free edge, a wrong Euler characteristic, an
//!    inconsistently oriented face. Cheap, and the right default.
//! 2. [`measure::solid_volume`] against a closed-form expectation catches
//!    geometric errors that leave the topology intact, which is the failure
//!    mode a boolean is most likely to produce. Pass a tight deflection:
//!    a coarse one under-counts curved faces and will disagree with itself
//!    across values.
//! 3. [`heal::heal_solid`] repairs what the first two find, merging
//!    coincident vertices, dropping degenerate edges, closing wire gaps, and
//!    fixing face orientation.
//!
//! A solid that passes `validate_solid` is well-formed, not necessarily
//! correct. Volume is what distinguishes the two.
//!
//! # Module families
//!
//! | Family | Modules | Purpose |
//! |--------|---------|---------|
//! | **Core** | [`primitives`], [`extrude`], [`revolve`], [`sweep`], [`loft`], [`pipe`], [`helix`] | Shape creation |
//! | **Transform** | [`transform`], [`copy`], [`mirror`], [`pattern`] | Spatial operations |
//! | **Boolean** | [`boolean`], [`mesh_boolean`] | Set operations |
//! | **Blend** | [`fillet`], [`chamfer`], [`blend_ops`] | Edge smoothing |
//! | **Offset** | [`offset_face`], [`offset_trim`], [`offset_v2`], [`offset_wire`] | Wall thickness |
//! | **Surface** | [`fill_face`], [`thicken`], [`shell_op`], [`draft`], [`section`], [`split`] | Surface/solid modification |
//! | **Repair** | [`heal`], [`defeature`], [`sew`], [`untrim`] | Shape fixing |
//! | **Analysis** | [`measure`], [`distance`], [`classify`], [`validate`], [`query`], [`feature_recognition`] | Interrogation |
//! | **Tessellation** | [`tessellate`] | Mesh generation |
//! | **Infrastructure** | [`assembly`], [`compound_ops`], [`evolution`], [`sketch`] | Utilities |
//!
//! # See also
//!
//! - [`brepkit_io`](https://docs.rs/brepkit-io): reading and writing STEP and
//!   the mesh formats.
//! - [`brepkit_topology`](https://docs.rs/brepkit-topology): the arena every
//!   operation here takes, and the surface enums it stores.
//! - [brepjs.dev](https://brepjs.dev): concepts, task recipes, and the
//!   TypeScript API built on this kernel.

use brepkit_math::vec::{Point3, Vec3};

pub mod extrude;
pub mod helix;
pub mod loft;
pub mod pipe;
pub mod primitives;
pub mod projection;
pub mod revolve;
pub mod sweep;

pub mod copy;
pub mod mirror;
pub mod pattern;
pub mod transform;

pub mod boolean;
pub mod mesh_boolean;

pub mod blend_ops;
pub mod chamfer;
pub mod fillet;

pub mod offset_face;
pub mod offset_trim;
pub mod offset_v2;
pub mod offset_wire;

pub mod draft;
pub mod fill_face;
pub mod section;
pub mod shell_op;
pub mod split;
pub mod thicken;

pub mod defeature;
pub mod heal;
pub mod sew;
pub mod untrim;

pub mod classify;
pub mod distance;
pub mod feature_recognition;
pub mod measure;
pub mod query;
pub mod validate;

pub mod tessellate;

pub mod assembly;
pub(crate) mod cap;
pub mod compound_ops;
pub mod evolution;
pub mod sketch;
pub(crate) mod winding;

#[cfg(test)]
pub(crate) mod test_helpers;

/// Compute `n · p` treating a `Point3` as a direction vector.
///
/// Equivalent to the dot product `n.x*p.x + n.y*p.y + n.z*p.z`, used
/// for the plane equation `n · point = d`.
fn dot_normal_point(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// Errors from modeling operations.
#[derive(Debug, thiserror::Error)]
pub enum OperationsError {
    /// The input shape is invalid for this operation.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Description of what is wrong.
        reason: String,
    },

    /// The operation produced a non-manifold result.
    #[error("non-manifold result")]
    NonManifoldResult,

    /// The operation produced an empty result (no geometry).
    ///
    /// Boolean operations return this when the algebraic outcome is the
    /// empty set: `Cut(A, B)` when `A ⊆ B`, or any operation on
    /// pre-collapsed inputs. Distinguishable from [`InvalidInput`] so
    /// callers can apply empty-operand identity rules without
    /// string-matching the error message.
    ///
    /// [`InvalidInput`]: Self::InvalidInput
    #[error("empty result: {reason}")]
    EmptyResult {
        /// Description of the empty-result scenario.
        reason: String,
    },

    /// A referenced topology entity was not found.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math error occurred during the operation.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),

    /// A GFA algorithm error occurred.
    #[error("algo: {0}")]
    Algo(#[from] brepkit_algo::error::AlgoError),

    /// A blend (fillet/chamfer v2) error occurred.
    #[error("blend: {0}")]
    Blend(#[from] brepkit_blend::BlendError),

    /// A check (classification/validation/distance) error occurred.
    #[error("check: {0}")]
    Check(#[from] brepkit_check::CheckError),

    /// A geometry conversion error occurred.
    #[error("geometry: {0}")]
    Geometry(#[from] brepkit_geometry::error::GeomError),
}
