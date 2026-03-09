//! Point-in-solid classification via ray casting and generalized winding numbers.
//!
//! Determines whether a 3D point is inside, outside, or on the boundary
//! of a solid. This is equivalent to OCCT's `BRepClass3d_SolidClassifier`.
//!
//! Three classifiers are provided:
//! - [`classify_point`]: ray casting (fast, fragile on mesh defects)
//! - [`classify_point_winding`]: generalized winding numbers (robust to gaps)
//! - [`classify_point_robust`]: winding numbers with ray-casting fallback

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use std::f64::consts::PI;

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

    // Two perpendicular irrational ray directions for dual-ray consensus.
    // Using two directions handles edge-on degeneracies where a single ray
    // might give an ambiguous count due to tangent faces.
    let ray_dirs = [
        Vec3::new(
            0.573_576_436_351_046,
            0.740_535_693_464_567_5,
            0.350_889_803_483_932_2,
        ),
        Vec3::new(
            -0.350_889_803_483_932_2,
            0.573_576_436_351_046,
            0.740_535_693_464_567_5,
        ),
    ];

    let mut inside_votes = 0u32;
    for &dir in &ray_dirs {
        let crossings = count_ray_crossings(topo, shell.faces(), point, dir, deflection)?;
        if crossings % 2 == 1 {
            inside_votes += 1;
        }
    }

    // Majority vote: both rays must agree the point is inside
    if inside_votes >= 2 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Classifies a point relative to a solid using generalized winding numbers.
///
/// For each triangle on the solid's boundary, computes the signed solid angle
/// subtended at the query point. The sum divided by 4pi gives the winding
/// number: > 0.5 means inside, < 0.5 means outside.
///
/// This method is inherently robust to mesh defects (small gaps, non-manifold
/// edges) because it integrates a continuous function rather than counting
/// discrete crossings.
///
/// `deflection` controls tessellation quality for NURBS faces.
/// `tolerance` is the distance threshold for "on boundary" classification.
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_winding(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let (winding, on_boundary) = compute_winding_number(topo, solid, point, deflection, tolerance)?;
    if on_boundary {
        return Ok(PointClassification::OnBoundary);
    }
    if winding > 0.5 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Robust point classification combining winding numbers and ray casting.
///
/// Tries generalized winding numbers first (more robust to mesh defects),
/// then falls back to ray casting if the winding number is ambiguous
/// (within 0.1 of the 0.5 threshold).
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_robust(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let (winding, on_boundary) = compute_winding_number(topo, solid, point, deflection, tolerance)?;
    if on_boundary {
        return Ok(PointClassification::OnBoundary);
    }

    // Confident classification: winding number is far from the 0.5 threshold
    if winding > 0.6 {
        return Ok(PointClassification::Inside);
    }
    if winding < 0.4 {
        return Ok(PointClassification::Outside);
    }

    // Ambiguous region (0.4..=0.6): fall back to ray casting
    classify_point(topo, solid, point, deflection, tolerance)
}

/// Computes the generalized winding number of a point relative to a solid.
///
/// Returns `(winding_number, is_on_boundary)`. The winding number is the sum
/// of signed solid angles subtended by each boundary triangle, divided by 4pi.
/// A value near 1.0 indicates the point is inside; near 0.0 indicates outside.
///
/// Uses the formula from Jacobson, Kavan, and Sorkine-Hornung (2013).
#[allow(clippy::similar_names)]
fn compute_winding_number(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<(f64, bool), OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // First check: is the point within tolerance of any face?
    if is_on_boundary(topo, shell.faces(), point, deflection, tolerance)? {
        return Ok((0.0, true));
    }

    let mut total_omega = 0.0;

    for &fid in shell.faces() {
        let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;

        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let a = mesh.positions[i0];
            let mut b = mesh.positions[i1];
            let mut c = mesh.positions[i2];

            // Ensure consistent outward winding by comparing the triangle's
            // geometric normal with the tessellation's vertex normal.
            let tri_normal = (b - a).cross(c - a);
            let mesh_normal = mesh.normals[i0];
            if tri_normal.dot(mesh_normal) < 0.0 {
                std::mem::swap(&mut b, &mut c);
            }

            // Vectors from point to triangle vertices
            let pa = a - point;
            let pb = b - point;
            let pc = c - point;

            let la = pa.length();
            let lb = pb.length();
            let lc = pc.length();

            // Point coincides with a vertex — treat as on boundary
            if la < tolerance || lb < tolerance || lc < tolerance {
                return Ok((0.0, true));
            }

            // Normalize
            let pa_n = pa * (1.0 / la);
            let pb_n = pb * (1.0 / lb);
            let pc_n = pc * (1.0 / lc);

            // Signed solid angle via the Van Oosterom-Strackee formula
            let numerator = pa_n.dot(pb_n.cross(pc_n));
            let denominator = 1.0 + pa_n.dot(pb_n) + pb_n.dot(pc_n) + pc_n.dot(pa_n);

            let omega = 2.0 * f64::atan2(numerator, denominator);
            total_omega += omega;
        }
    }

    // Use abs(): the vertex-swap above ensures consistent winding per
    // triangle, but across faces the orientation may vary (common after
    // boolean operations). abs() makes classification robust to mixed
    // orientations while still detecting inside vs outside.
    let winding = (total_omega / (4.0 * PI)).abs();
    Ok((winding, false))
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

/// Watertight ray-triangle intersection test.
///
/// Returns true if the ray `origin + t * direction` (t > 0) intersects
/// the triangle (v0, v1, v2). Delegates to the watertight algorithm from
/// `brepkit_math` which guarantees no cracks or double-hits on shared edges.
fn ray_triangle_intersect(
    origin: Point3,
    direction: Vec3,
    v0: Point3,
    v1: Point3,
    v2: Point3,
) -> bool {
    brepkit_math::ray_triangle::watertight_ray_triangle_intersect(origin, direction, v0, v1, v2)
        .is_some()
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

        // Box extends from (0,0,0) to (2,2,2); center is (1,1,1).
        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
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

        // Box extends from (0,0,0) to (2,2,2), so face at z=2.0
        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 2.0), 0.1, 1e-3).unwrap();
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

    // ── Winding number tests ─────────────────────────

    #[test]
    fn winding_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box extends from (0,0,0) to (2,2,2); center is (1,1,1).
        let result =
            classify_point_winding(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn winding_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_winding(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    // ── Robust classifier tests ──────────────────────

    #[test]
    fn robust_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box extends from (0,0,0) to (2,2,2); center is (1,1,1).
        let result =
            classify_point_robust(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn robust_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_robust(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }
}
