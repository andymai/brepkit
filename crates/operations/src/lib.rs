//! # brepkit-operations
//!
//! CAD modeling operations for B-Rep solids. Layer L3, depending on
//! `brepkit-math`, `brepkit-topology`, `brepkit-geometry`, `brepkit-algo`,
//! `brepkit-blend`, `brepkit-heal`, `brepkit-check`, and `brepkit-offset`.
//!
//! # Module families
//!
//! | Family | Modules | Purpose |
//! |--------|---------|---------|
//! | **Core** | [`primitives`], [`extrude`], [`revolve`], [`sweep`], [`loft`], [`pipe`], [`helix`] | Shape creation |
//! | **Transform** | [`transform`], [`copy`], [`mirror`], [`pattern`] | Spatial operations |
//! | **Boolean** | [`boolean`], [`mesh_boolean`] | Set operations |
//! | **Blend** | [`fillet`], [`chamfer`], [`blend_ops`] | Edge smoothing |
//! | **Offset** | [`offset_face`], [`offset_solid`], [`offset_trim`], [`offset_v2`], [`offset_wire`] | Wall thickness |
//! | **Surface** | [`fill_face`], [`thicken`], [`shell_op`], [`draft`], [`section`], [`split`] | Surface/solid modification |
//! | **Repair** | [`heal`], [`defeature`], [`sew`], [`untrim`] | Shape fixing |
//! | **Analysis** | [`measure`], [`distance`], [`classify`], [`validate`], [`query`], [`feature_recognition`] | Interrogation |
//! | **Tessellation** | [`tessellate`] | Mesh generation |
//! | **Infrastructure** | [`assembly`], [`compound_ops`], [`evolution`], [`sketch`] | Utilities |

use brepkit_math::vec::{Point3, Vec3};

// ── Core: shape creation ─────────────────────────────────────────
pub mod extrude;
pub mod helix;
pub mod loft;
pub mod pipe;
pub mod primitives;
pub mod revolve;
pub mod sweep;

// ── Transform: spatial operations ────────────────────────────────
pub mod copy;
pub mod mirror;
pub mod pattern;
pub mod transform;

// ── Boolean: set operations ──────────────────────────────────────
pub mod boolean;
pub mod mesh_boolean;

// ── Blend: edge smoothing ────────────────────────────────────────
pub mod blend_ops;
pub mod chamfer;
pub mod fillet;

// ── Offset: wall thickness ───────────────────────────────────────
pub mod offset_face;
pub mod offset_solid;
pub mod offset_trim;
pub mod offset_v2;
pub mod offset_wire;

// ── Surface & solid modification ─────────────────────────────────
pub mod draft;
pub mod fill_face;
pub mod section;
pub mod shell_op;
pub mod split;
pub mod thicken;

// ── Repair: shape fixing ─────────────────────────────────────────
pub mod defeature;
pub mod heal;
pub mod sew;
pub mod untrim;

// ── Analysis: interrogation ──────────────────────────────────────
pub mod classify;
pub mod distance;
pub mod feature_recognition;
pub mod measure;
pub mod query;
pub mod validate;

// ── Tessellation: mesh generation ────────────────────────────────
pub mod tessellate;

// ── Infrastructure ───────────────────────────────────────────────
pub mod assembly;
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
