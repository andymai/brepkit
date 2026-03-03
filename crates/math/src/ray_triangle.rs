//! Watertight ray-triangle intersection (Woop, Benthin, Wald 2013).
//!
//! This module implements the watertight ray-triangle intersection algorithm
//! that guarantees no cracks or double-hits on shared edges between adjacent
//! triangles. The key idea is to transform coordinates so the ray is
//! axis-aligned along +Z, then evaluate edge functions using identical
//! floating-point operations on shared vertices.

use crate::vec::{Point3, Vec3};

/// Result of a watertight ray-triangle intersection test.
#[derive(Debug, Clone, Copy)]
pub struct RayTriangleHit {
    /// Distance along the ray to the hit point (t parameter).
    pub t: f64,
    /// Barycentric coordinate u.
    pub u: f64,
    /// Barycentric coordinate v.
    pub v: f64,
}

/// Deterministic tie-breaking for zero-valued edge functions.
///
/// When a 2D edge function `p x q` evaluates to exactly zero, this function
/// returns a small positive or negative value based on the edge direction.
/// The sign is chosen deterministically so that for the complementary edge
/// (`q x p`), the opposite sign is returned. This ensures shared edges are
/// assigned to exactly one triangle.
fn edge_tiebreak(px: f64, py: f64, qx: f64, qy: f64) -> f64 {
    // Use the signs of the constituent products as a tiebreaker.
    // For edge function `px*qy - py*qx`, when result is zero, look at
    // the individual terms to pick a deterministic sign.
    let a = px * qy;
    let b = py * qx;
    if a > b {
        f64::MIN_POSITIVE
    } else if a < b {
        -f64::MIN_POSITIVE
    } else {
        // Both products are equal (and possibly zero). Use coordinate signs.
        // This order is deterministic: we break ties by comparing coordinates.
        if px > qx {
            f64::MIN_POSITIVE
        } else if px < qx {
            -f64::MIN_POSITIVE
        } else if py > qy {
            f64::MIN_POSITIVE
        } else {
            -f64::MIN_POSITIVE
        }
    }
}

/// Watertight ray-triangle intersection (Woop, Benthin, Wald 2013).
///
/// Tests whether a ray from `origin` in direction `dir` hits the triangle
/// `(v0, v1, v2)`. Returns `Some(hit)` with the parametric distance and
/// barycentric coordinates, or `None` if no intersection.
///
/// This algorithm guarantees watertight results on shared edges: a ray
/// hitting a shared edge between two triangles will report exactly one
/// intersection, with no cracks or double-hits. This is achieved by using
/// identical floating-point operations on shared vertices.
///
/// The barycentric coordinates satisfy `hit_point = (1-u-v)*v0 + u*v1 + v*v2`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn watertight_ray_triangle_intersect(
    origin: Point3,
    dir: Vec3,
    v0: Point3,
    v1: Point3,
    v2: Point3,
) -> Option<RayTriangleHit> {
    let d = dir.0;

    // Step 1: Find the largest absolute component of dir (kz), then set kx, ky.
    let abs_x = d[0].abs();
    let abs_y = d[1].abs();
    let abs_z = d[2].abs();

    let kz = if abs_x > abs_y && abs_x > abs_z {
        0
    } else if abs_y > abs_z {
        1
    } else {
        2
    };
    let mut kx = (kz + 1) % 3;
    let mut ky = (kx + 1) % 3;

    // Swap kx and ky if dir[kz] is negative to preserve winding order.
    if d[kz] < 0.0 {
        std::mem::swap(&mut kx, &mut ky);
    }

    // Step 2: Shear constants.
    let sz = 1.0 / d[kz];
    let sx = d[kx] * sz;
    let sy = d[ky] * sz;

    // Step 3: Translate vertices relative to ray origin.
    let a = (v0 - origin).0;
    let b = (v1 - origin).0;
    let c = (v2 - origin).0;

    // Step 4: Shear and permute.
    let ax = a[kx] - sx * a[kz];
    let ay = a[ky] - sy * a[kz];
    let bx = b[kx] - sx * b[kz];
    let by = b[ky] - sy * b[kz];
    let cx = c[kx] - sx * c[kz];
    let cy = c[ky] - sy * c[kz];

    // Step 5: Edge function values (2D cross products).
    let mut u = cx.mul_add(by, -(cy * bx));
    let mut v = ax.mul_add(cy, -(ay * cx));
    let mut w = bx.mul_add(ay, -(by * ax));

    // Step 6: Deterministic tie-breaking for zero edge functions.
    // When an edge function is exactly zero, assign a deterministic sign
    // based on the edge direction to guarantee that shared edges/vertices
    // are claimed by exactly one triangle.
    if u == 0.0 {
        u = edge_tiebreak(cx, cy, bx, by);
    }
    if v == 0.0 {
        v = edge_tiebreak(ax, ay, cx, cy);
    }
    if w == 0.0 {
        w = edge_tiebreak(bx, by, ax, ay);
    }

    // Step 7: Sign check — U, V, W must all share the same sign.
    if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
        return None;
    }

    // Step 8: Determinant.
    let det = u + v + w;
    if det == 0.0 {
        return None;
    }

    // Step 9: Compute scaled t.
    let az = sz * a[kz];
    let bz = sz * b[kz];
    let cz = sz * c[kz];
    let t_scaled = u.mul_add(az, v.mul_add(bz, w * cz));

    // Step 10: Check that t is positive (ray goes forward).
    // If det > 0, t_scaled must be > 0; if det < 0, t_scaled must be < 0.
    if (det > 0.0 && t_scaled <= 0.0) || (det < 0.0 && t_scaled >= 0.0) {
        return None;
    }

    // Step 11: Final values.
    let inv_det = 1.0 / det;
    Some(RayTriangleHit {
        t: t_scaled * inv_det,
        u: v * inv_det,
        v: w * inv_det,
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_lossless,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn tri() -> (Point3, Point3, Point3) {
        (
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn ray_hits_triangle() {
        let (v0, v1, v2) = tri();
        let origin = Point3::new(0.0, 0.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);

        let hit = watertight_ray_triangle_intersect(origin, dir, v0, v1, v2).expect("should hit");
        assert!((hit.t - 1.0).abs() < EPS);
    }

    #[test]
    fn ray_misses_triangle() {
        let (v0, v1, v2) = tri();
        let origin = Point3::new(10.0, 10.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);

        assert!(watertight_ray_triangle_intersect(origin, dir, v0, v1, v2).is_none());
    }

    #[test]
    fn ray_parallel_to_triangle() {
        let (v0, v1, v2) = tri();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);

        assert!(watertight_ray_triangle_intersect(origin, dir, v0, v1, v2).is_none());
    }

    #[test]
    fn shared_edge_exactly_one_hit() {
        // Two triangles sharing edge from (0,0,0) to (1,0,0).
        let shared_a = Point3::new(0.0, 0.0, 0.0);
        let shared_b = Point3::new(1.0, 0.0, 0.0);
        let tri1_c = Point3::new(0.5, 1.0, 0.0);
        let tri2_c = Point3::new(0.5, -1.0, 0.0);

        // Ray aimed at the midpoint of the shared edge.
        let origin = Point3::new(0.5, 0.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);

        let hit1 = watertight_ray_triangle_intersect(origin, dir, shared_a, shared_b, tri1_c);
        let hit2 = watertight_ray_triangle_intersect(origin, dir, shared_a, shared_b, tri2_c);

        let count = hit1.is_some() as u32 + hit2.is_some() as u32;
        assert_eq!(count, 1, "shared edge must report exactly one hit");
    }

    #[test]
    fn ray_hits_vertex() {
        // Four triangles forming a complete fan around the origin vertex,
        // covering all quadrants so the fan is closed.
        let shared = Point3::new(0.0, 0.0, 0.0);
        let tris = [
            (
                shared,
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ),
            (
                shared,
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ),
            (
                shared,
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
            ),
            (
                shared,
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
        ];

        let origin = Point3::new(0.0, 0.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);

        let count: u32 = tris
            .iter()
            .map(|&(a, b, c)| {
                watertight_ray_triangle_intersect(origin, dir, a, b, c).is_some() as u32
            })
            .sum();

        assert_eq!(
            count, 1,
            "vertex shared by 4 triangles must report exactly one hit"
        );
    }

    #[test]
    fn backface_not_hit() {
        let (v0, v1, v2) = tri();
        // Ray going away from the triangle.
        let origin = Point3::new(0.0, 0.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        assert!(watertight_ray_triangle_intersect(origin, dir, v0, v1, v2).is_none());
    }

    #[test]
    fn barycentric_coordinates_valid() {
        let (v0, v1, v2) = tri();
        let origin = Point3::new(0.0, 0.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);

        let hit = watertight_ray_triangle_intersect(origin, dir, v0, v1, v2).expect("should hit");

        assert!(hit.u >= -EPS, "u = {} should be >= 0", hit.u);
        assert!(hit.v >= -EPS, "v = {} should be >= 0", hit.v);
        assert!(
            hit.u + hit.v <= 1.0 + EPS,
            "u + v = {} should be <= 1",
            hit.u + hit.v,
        );

        // Verify that barycentric coords reconstruct the hit point.
        let w = 1.0 - hit.u - hit.v;
        let px = w * v0.x() + hit.u * v1.x() + hit.v * v2.x();
        let py = w * v0.y() + hit.u * v1.y() + hit.v * v2.y();
        let pz = w * v0.z() + hit.u * v1.z() + hit.v * v2.z();

        let expected = origin.0[2] + hit.t * dir.0[2];
        assert!(
            (pz - expected).abs() < EPS,
            "pz = {pz}, expected = {expected}"
        );
        // The hit point should be at origin + t*dir.
        assert!((px - origin.x()).abs() < EPS);
        assert!((py - origin.y()).abs() < EPS);
    }
}
