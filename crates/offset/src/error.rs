//! Error types for the offset engine.

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;

/// Errors from solid offset operations.
#[derive(Debug, thiserror::Error)]
pub enum OffsetError {
    /// A topology operation failed.
    #[error("topology error: {0}")]
    Topology(#[from] brepkit_topology::TopologyError),

    /// A math operation failed.
    #[error("math error: {0}")]
    Math(#[from] brepkit_math::MathError),

    /// The input parameters are invalid.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Description of why the input is invalid.
        reason: String,
    },

    /// Edge analysis could not determine convexity.
    #[error("analysis failed for edge {edge:?}: {reason}")]
    AnalysisFailed {
        /// The edge that could not be classified.
        edge: EdgeId,
        /// Description of the failure.
        reason: String,
    },

    /// Intersection of two offset faces failed.
    #[error("intersection failed between faces {face_a:?} and {face_b:?}: {reason}")]
    IntersectionFailed {
        /// First face in the intersection pair.
        face_a: FaceId,
        /// Second face in the intersection pair.
        face_b: FaceId,
        /// Description of the failure.
        reason: String,
    },

    /// A self-intersection was detected in the offset shell.
    #[error("self-intersection: {reason}")]
    SelfIntersection {
        /// Description of the self-intersection.
        reason: String,
    },

    /// Final shell assembly failed.
    #[error("assembly failed: {reason}")]
    AssemblyFailed {
        /// Description of the assembly failure.
        reason: String,
    },

    /// The offset distance exceeds the local curvature, collapsing the solid.
    #[error("offset distance collapses the solid")]
    CollapsedSolid,
}
