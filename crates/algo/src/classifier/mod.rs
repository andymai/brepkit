//! Face classification -- determines if a sub-face is inside/outside
//! the opposing solid.
//!
//! Two strategies:
//! - **Analytic**: O(1) point-in-solid for convex analytic solids.
//! - **Ray cast**: Multi-ray fallback for general solids.

mod analytic;
mod ray_cast;

pub use analytic::{AnalyticClassifier, classify_analytic, try_build_analytic_classifier};
pub use ray_cast::{classify_ray_cast, compute_solid_bbox, point_in_face_3d};

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Classify a point relative to a solid -- dispatch to the best available
/// strategy.
///
/// Tries the analytic classifier first (O(1) for convex analytic solids),
/// then falls back to ray casting.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate.
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    // Try analytic first (O(1) for convex analytic solids)
    if let Some(class) = classify_analytic(topo, solid, point) {
        return Ok(class);
    }

    // Fall back to ray casting
    classify_ray_cast(topo, solid, point)
}
