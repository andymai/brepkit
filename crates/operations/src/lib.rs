//! # brepkit-operations
//!
//! CAD modeling operations: booleans, fillets, chamfers, extrusions,
//! sweeps, and tessellation.
//!
//! This is layer L2, depending on `brepkit-math` and `brepkit-topology`.

use brepkit_math::vec::{Point3, Vec3};

pub mod assembly;
pub mod boolean;
pub mod chamfer;
pub mod classify;
pub mod compound_ops;
pub mod copy;
pub mod defeature;
pub mod distance;
pub mod draft;
pub mod extrude;
pub mod feature_recognition;
pub mod fillet;
pub mod heal;
pub mod helix;
pub mod loft;
pub mod measure;
pub mod mirror;
pub mod offset_face;
pub mod offset_wire;
pub mod pattern;
pub mod pipe;
pub mod primitives;
pub mod revolve;
pub mod section;
pub mod sew;
pub mod shell_op;
pub mod split;
pub mod sweep;
pub mod tessellate;
pub mod thicken;
pub mod transform;
pub mod validate;

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
}
