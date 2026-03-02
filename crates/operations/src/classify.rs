//! Point-in-solid classification via ray casting.
//!
//! Determines whether a 3D point is inside, outside, or on the boundary
//! of a solid. This is equivalent to OCCT's `BRepClass3d_SolidClassifier`.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Result of classifying a point relative to a solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointClassification {
    /// The point is inside the solid.
    Inside,
    /// The point is outside the solid.
    Outside,
    /// The point is on the boundary (within tolerance).
    OnBoundary,
}

/// Classifies a point relative to a solid using ray casting.
///
/// Shoots a ray from `point` and counts crossings with the solid's
/// boundary faces. Uses tessellation for robust intersection with
/// curved faces.
///
/// `deflection` controls tessellation quality for NURBS faces.
/// `tolerance` is the distance threshold for "on boundary" classification.
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // First check: is the point within tolerance of any face?
    if is_on_boundary(topo, shell.faces(), point, deflection, tolerance)? {
        return Ok(PointClassification::OnBoundary);
    }

    // Ray direction chosen to be unlikely to hit edges/vertices exactly.
    // Using an irrational direction avoids alignment with common geometry.
    let ray_dir = Vec3::new(
        0.573_576_436_351_046,   // 1/sqrt(3) + small offset
        0.740_535_693_464_567_5, // golden ratio / sqrt(5) + offset
        0.350_889_803_483_932_2, // 1/e + offset
    );

    let crossings = count_ray_crossings(topo, shell.faces(), point, ray_dir, deflection)?;

    if crossings % 2 == 1 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Checks if a point is within `tolerance` of any face boundary.
fn is_on_boundary(
    topo: &Topology,
    faces: &[FaceId],
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<bool, OperationsError> {
    let tol_sq = tolerance * tolerance;

    for &fid in faces {
        let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;

        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];

            let dist_sq = point_triangle_distance_sq(point, v0, v1, v2);
            if dist_sq < tol_sq {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// Counts the number of times a ray crosses the solid's boundary.
fn count_ray_crossings(
    topo: &Topology,
    faces: &[FaceId],
    origin: Point3,
    direction: Vec3,
    deflection: f64,
) -> Result<u32, OperationsError> {
    let mut crossings = 0u32;

    for &fid in faces {
        let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;

        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];

            if ray_triangle_intersect(origin, direction, v0, v1, v2) {
                crossings += 1;
            }
        }
    }

    Ok(crossings)
}

/// Möller–Trumbore ray-triangle intersection test.
///
/// Returns true if the ray `origin + t * direction` (t > 0) intersects
/// the triangle (v0, v1, v2).
fn ray_triangle_intersect(
    origin: Point3,
    direction: Vec3,
    v0: Point3,
    v1: Point3,
    v2: Point3,
) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let pvec = direction.cross(edge2);
    let det = edge1.dot(pvec);

    // Ray is parallel to triangle.
    if det.abs() < 1e-12 {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = origin - v0;
    let bary_u = inv_det * tvec.dot(pvec);

    if !(0.0..=1.0).contains(&bary_u) {
        return false;
    }

    let qvec = tvec.cross(edge1);
    let bary_v = inv_det * direction.dot(qvec);

    if bary_v < 0.0 || bary_u + bary_v > 1.0 {
        return false;
    }

    let ray_t = inv_det * edge2.dot(qvec);

    // Only count forward intersections
    ray_t > 1e-10
}

/// Computes the squared distance from a point to a triangle.
///
/// Uses the projection method: project point onto the triangle's plane,
/// then check if the projection is inside the triangle. If outside,
/// compute distance to the nearest edge.
fn point_triangle_distance_sq(point: Point3, v0: Point3, v1: Point3, v2: Point3) -> f64 {
    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let v0_to_p = point - v0;

    let d00 = edge0.dot(edge0);
    let d01 = edge0.dot(edge1);
    let d11 = edge1.dot(edge1);
    let d20 = v0_to_p.dot(edge0);
    let d21 = v0_to_p.dot(edge1);

    let denom = d00.mul_add(d11, -(d01 * d01));
    if denom.abs() < 1e-20 {
        // Degenerate triangle — use distance to first vertex
        return v0_to_p.length_squared();
    }

    let inv_denom = 1.0 / denom;
    let u = d11.mul_add(d20, -(d01 * d21)) * inv_denom;
    let v = d00.mul_add(d21, -(d01 * d20)) * inv_denom;

    // If projection is inside the triangle
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        let projected = v0 + edge0 * u + edge1 * v;
        let diff = point - projected;
        return diff.length_squared();
    }

    // Otherwise, find closest point on edges
    let d_e0 = point_segment_distance_sq(point, v0, v1);
    let d_e1 = point_segment_distance_sq(point, v1, v2);
    let d_e2 = point_segment_distance_sq(point, v2, v0);

    d_e0.min(d_e1).min(d_e2)
}

/// Squared distance from a point to a line segment.
fn point_segment_distance_sq(point: Point3, seg_start: Point3, seg_end: Point3) -> f64 {
    let seg = seg_end - seg_start;
    let seg_len_sq = seg.length_squared();

    if seg_len_sq < 1e-20 {
        return (point - seg_start).length_squared();
    }

    let t = ((point - seg_start).dot(seg) / seg_len_sq).clamp(0.0, 1.0);
    let closest = seg_start + seg * t;
    (point - closest).length_squared()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_on_boundary_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box is centered at origin, so face at z=1.0
        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 1.0), 0.1, 1e-3).unwrap();
        assert_eq!(result, PointClassification::OnBoundary);
    }

    #[test]
    fn point_outside_negative_direction() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point(&topo, solid, Point3::new(-5.0, -5.0, -5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_near_corner() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Point inside near corner
        let result = classify_point(&topo, solid, Point3::new(0.9, 0.9, 0.9), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    // ── Ray-triangle tests ────────────────────────────

    #[test]
    fn ray_hits_triangle() {
        assert!(ray_triangle_intersect(
            Point3::new(0.25, 0.25, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ));
    }

    #[test]
    fn ray_misses_triangle() {
        assert!(!ray_triangle_intersect(
            Point3::new(2.0, 2.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ));
    }

    #[test]
    fn ray_backward_no_hit() {
        // Ray going the wrong way
        assert!(!ray_triangle_intersect(
            Point3::new(0.25, 0.25, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ));
    }

    // ── Point-triangle distance tests ─────────────────

    #[test]
    fn point_on_triangle() {
        let dist = point_triangle_distance_sq(
            Point3::new(0.25, 0.25, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(dist < 1e-20);
    }

    #[test]
    fn point_above_triangle() {
        let dist = point_triangle_distance_sq(
            Point3::new(0.25, 0.25, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!((dist - 9.0).abs() < 1e-10);
    }
}
