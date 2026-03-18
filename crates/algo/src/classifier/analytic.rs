//! Analytic O(1) point-in-solid classification.
//!
//! For convex solids composed entirely of analytic surfaces (plane,
//! cylinder, cone, sphere, torus), a point can be classified by
//! testing the signed distance to each face. If the point is on the
//! negative side of ALL faces, it is inside.
//!
//! This is a stub — the full implementation will be ported from
//! `operations/boolean/classify.rs` in a follow-up.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;

/// Try to classify a point using analytic geometry.
///
/// Returns `Some(FaceClass)` if the solid is a convex analytic solid
/// and the point can be classified without tessellation. Returns `None`
/// if the solid is not suitable for analytic classification.
#[must_use]
#[allow(unused_variables)]
pub fn classify_analytic(topo: &Topology, solid: SolidId, point: Point3) -> Option<FaceClass> {
    // Stub: always fall back to ray casting for now.
    // TODO: port analytic classifier from operations/boolean/classify.rs
    None
}
