//! Ray-cast point-in-solid classification.
//!
//! Shoots rays from a sample point and counts boundary crossings
//! to determine inside/outside status.
//!
//! This is a stub — the full implementation will be ported from
//! `operations/boolean/classify.rs` in a follow-up.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Classify a point by ray casting against the solid's faces.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate after multiple ray directions.
#[allow(unused_variables, clippy::unnecessary_wraps)]
pub fn classify_ray_cast(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    // Stub: return Unknown for now.
    // TODO: port ray-cast classifier from operations/boolean/classify.rs
    log::debug!("classify_ray_cast: stub returning Unknown for point {point:?} vs solid {solid:?}");
    Ok(FaceClass::Unknown)
}
