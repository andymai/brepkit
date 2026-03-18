//! Same-domain face detection and merging.
//!
//! When two faces from opposing solids share the same underlying surface
//! (coplanar planes, coincident cylinders, etc.), they are "same-domain"
//! faces. These need special handling:
//!
//! - **CoplanarSame**: normals point the same way — kept in fuse/intersect.
//! - **CoplanarOpposite**: normals point opposite — kept in cut (for B faces).
//!
//! This is a stub — full same-domain merging will be implemented in a
//! follow-up when face splitting is complete.

use std::collections::HashMap;
use std::hash::BuildHasher;

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

use super::SubFace;
use crate::ds::{GfaArena, Rank};

/// Detect same-domain face pairs and update their classification.
///
/// Scans all sub-faces looking for pairs where both faces lie on the
/// same geometric surface. Updates classification to `CoplanarSame`
/// or `CoplanarOpposite` based on normal alignment.
///
/// # Errors
///
/// Returns [`AlgoError`] if topology lookups fail.
#[allow(unused_variables)]
pub fn detect_same_domain<S: BuildHasher>(
    topo: &Topology,
    arena: &GfaArena,
    sub_faces: &[SubFace],
    face_ranks: &HashMap<FaceId, Rank, S>,
) {
    // Stub: same-domain detection deferred to follow-up.
    log::debug!("detect_same_domain: stub, no same-domain faces detected");
}
