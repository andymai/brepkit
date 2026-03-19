//! Split faces using `FaceInfo` data from the PaveFiller.
//!
//! For each face that has section or interior pave blocks, build
//! sub-faces. Faces with no intersection data pass through unchanged.

use std::collections::HashMap;
use std::hash::BuildHasher;

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;

use crate::ds::{GfaArena, Rank};

use super::SubFace;
use super::face_class::FaceClass;

/// Build sub-faces for all faces that have intersection data.
///
/// For Phase 4, faces without section edges pass through as a single
/// sub-face. Faces with section edges are recorded but not geometrically
/// split yet (splitting deferred to a later phase).
pub fn fill_images_faces<S: BuildHasher, S2: BuildHasher>(
    arena: &GfaArena,
    _edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
    face_ranks: &HashMap<FaceId, Rank, S2>,
) -> Vec<SubFace> {
    let mut sub_faces = Vec::new();

    for (&face_id, &rank) in face_ranks {
        let fi = arena.face_info(face_id);
        let has_intersections = fi.is_some_and(FaceInfo::has_intersections);

        if has_intersections {
            log::debug!(
                "fill_images_faces: face {face_id:?} (rank {rank:?}) has intersections, \
                 deferring split"
            );
        }

        // Phase 4 minimal: record face as single sub-face regardless
        // of intersection state. Classification determines IN/OUT.
        sub_faces.push(SubFace {
            parent_face: face_id,
            face_id,
            classification: FaceClass::Unknown,
            rank,
        });
    }

    sub_faces
}

use crate::ds::FaceInfo;
