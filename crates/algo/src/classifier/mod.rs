//! Face classification — determines if a sub-face is inside/outside
//! the opposing solid.
//!
//! Two strategies:
//! - **Analytic**: O(1) point-in-solid for convex analytic solids.
//! - **Ray cast**: Multi-ray BVH fallback for general solids.
//!
//! Phase 4 starts with a stub classifier that returns `Unknown`.
//! The real classifiers will be ported from `operations/boolean/classify.rs`.

mod analytic;
mod ray_cast;

pub use analytic::classify_analytic;
pub use ray_cast::classify_ray_cast;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Classify a point relative to a solid — dispatch to the best available
/// strategy.
///
/// Currently delegates to ray casting. Analytic fast paths will be
/// added as classifiers are ported.
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
