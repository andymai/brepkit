//! Walking-based fillet and chamfer engine.
//!
//! This crate implements blend surface computation using a
//! Newton-Raphson walking algorithm. It produces G1-continuous fillet
//! and chamfer surfaces for all combinations of analytic and NURBS faces.

#[allow(dead_code)]
pub(crate) mod adaptive_tolerance;
pub(crate) mod analytic;
pub(crate) mod blend_func;
pub(crate) mod builder_utils;
pub mod chamfer_builder;
pub(crate) mod corner;
pub mod fillet_builder;
#[allow(dead_code)]
pub(crate) mod g1_chain;
pub mod radius_law;
pub(crate) mod section;
pub(crate) mod spherical_triangle;
pub(crate) mod spine;
pub(crate) mod stripe;
pub(crate) mod trimmer;
pub(crate) mod walker;

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Error type for blend operations.
#[derive(Debug, thiserror::Error)]
pub enum BlendError {
    /// No initial solution found at the spine start.
    #[error("no start solution at edge {edge:?}, t={t}")]
    StartSolutionFailure {
        /// The edge where the start solution failed.
        edge: EdgeId,
        /// The parameter value at the failure point.
        t: f64,
    },

    /// Walker diverged during marching.
    #[error("walking failure at edge {edge:?}, t={t}, residual={residual}")]
    WalkingFailure {
        /// The edge where walking failed.
        edge: EdgeId,
        /// The parameter value at the failure point.
        t: f64,
        /// The residual norm at failure.
        residual: f64,
    },

    /// Generated surface is twisted or self-intersecting.
    #[error("twisted surface on stripe {stripe_idx}")]
    TwistedSurface {
        /// Index of the stripe that is twisted.
        stripe_idx: usize,
    },

    /// Radius too large for the edge geometry.
    #[error("radius too large for edge {edge:?}: max={max_radius}")]
    RadiusTooLarge {
        /// The edge for which the radius is too large.
        edge: EdgeId,
        /// The maximum allowable radius.
        max_radius: f64,
    },

    /// Face trimming failed.
    #[error("trimming failure on face {face:?}")]
    TrimmingFailure {
        /// The face where trimming failed.
        face: FaceId,
    },

    /// Corner solver failed at vertex.
    #[error("corner failure at vertex {vertex:?}")]
    CornerFailure {
        /// The vertex where the corner solver failed.
        vertex: VertexId,
    },

    /// Surface type not supported.
    #[error("unsupported surface on face {face:?}: {surface_tag}")]
    UnsupportedSurface {
        /// The face with the unsupported surface.
        face: FaceId,
        /// A description of the unsupported surface type.
        surface_tag: String,
    },

    /// Topology error from underlying operations.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// Math error from underlying computations.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),
}

/// Result of a blend operation.
pub struct BlendResult {
    /// The resulting solid.
    pub solid: SolidId,
    /// Edges that were successfully blended.
    pub succeeded: Vec<EdgeId>,
    /// Edges that failed with diagnostic info.
    pub failed: Vec<(EdgeId, BlendError)>,
    /// Whether this is a partial result (some edges failed).
    pub is_partial: bool,
}
