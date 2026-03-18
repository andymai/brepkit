//! Intersection curve from face-face intersection.

use brepkit_math::aabb::Aabb3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceId;

use super::pave::PaveBlockId;

/// An intersection curve from face-face intersection, with its
/// pave blocks and associated metadata.
#[derive(Debug, Clone)]
pub struct IntersectionCurveDS {
    /// The 3D curve geometry.
    pub curve: EdgeCurve,
    /// First face of the intersecting pair.
    pub face_a: FaceId,
    /// Second face of the intersecting pair.
    pub face_b: FaceId,
    /// Bounding box of the curve.
    pub bbox: Aabb3,
    /// Pave blocks split from this curve.
    pub pave_blocks: Vec<PaveBlockId>,
    /// Parameter range on the curve.
    pub t_range: (f64, f64),
}
