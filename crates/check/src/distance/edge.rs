//! Edge-to-edge distance computation.

use brepkit_math::vec::Point3;

/// Minimum distance between two line segments [p0, p1] and [q0, q1].
///
/// Returns (distance, closest\_on\_seg1, closest\_on\_seg2).
#[allow(clippy::similar_names)]
pub fn segment_segment_distance(
    p0: Point3,
    p1: Point3,
    q0: Point3,
    q1: Point3,
) -> (f64, Point3, Point3) {
    brepkit_geometry::extrema::segment_segment_distance(p0, p1, q0, q1)
}
