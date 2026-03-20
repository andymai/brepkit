//! Minimum distance and extrema between shapes.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

pub(crate) mod analytic;

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Which topological element supports a closest point.
#[derive(Debug, Clone, Copy)]
pub enum SupportElement {
    /// Closest point is on a face.
    Face(FaceId, f64, f64),
}

/// A single distance solution.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    /// The minimum distance.
    pub distance: f64,
    /// Closest point on shape A (or the query point).
    pub point_a: Point3,
    /// Closest point on shape B.
    pub point_b: Point3,
}

/// Compute the minimum distance from a point to a solid.
///
/// Uses BVH over face AABBs for acceleration. Dispatches per face type:
/// planar (point-to-polygon), analytic (closed-form), NURBS (Newton projection).
///
/// # Errors
///
/// Returns an error if any topology entity is missing.
#[allow(clippy::too_many_lines)]
pub fn point_to_solid(
    topo: &Topology,
    point: Point3,
    solid: SolidId,
) -> Result<DistanceResult, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build AABBs and BVH.
    let face_aabbs: Vec<(usize, Aabb3)> = face_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &fid)| crate::util::face_aabb(topo, fid).ok().map(|aabb| (i, aabb)))
        .collect();
    let bvh = Bvh::build(&face_aabbs);

    let mut best_dist = f64::INFINITY;
    let mut best_point = point;

    // Sort candidates by AABB distance to point for early termination.
    let mut candidates: Vec<usize> = (0..face_aabbs.len()).collect();
    candidates.sort_by(|&a, &b| {
        let da = face_aabbs[a].1.distance_squared_to_point(point);
        let db = face_aabbs[b].1.distance_squared_to_point(point);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use BVH closest hint to potentially reorder.
    if let Some(closest_idx) = bvh.query_closest(point) {
        if let Some(pos) = candidates
            .iter()
            .position(|&i| face_aabbs[i].0 == closest_idx)
        {
            candidates.swap(0, pos);
        }
    }

    for idx in candidates {
        let aabb_dist_sq = face_aabbs[idx].1.distance_squared_to_point(point);
        if aabb_dist_sq > best_dist * best_dist {
            break; // sorted, so all remaining are farther
        }
        let face_idx = face_aabbs[idx].0;
        let fid = face_ids[face_idx];
        if let Ok(Some((dist, closest))) = point_to_face(topo, point, fid) {
            if dist < best_dist {
                best_dist = dist;
                best_point = closest;
            }
        }
    }

    Ok(DistanceResult {
        distance: best_dist,
        point_a: point,
        point_b: best_point,
    })
}

/// Compute distance from a point to a single face, dispatching by surface type.
///
/// # Errors
///
/// Returns an error if the face lookup fails.
pub fn point_to_face(
    topo: &Topology,
    point: Point3,
    face_id: FaceId,
) -> Result<Option<(f64, Point3)>, CheckError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            let polygon = crate::util::face_polygon(topo, face_id)?;
            Ok(point_to_polygon_distance(point, &polygon, *normal, *d))
        }
        FaceSurface::Cylinder(cyl) => Ok(Some(analytic::point_to_cylinder(point, cyl))),
        FaceSurface::Cone(cone) => Ok(Some(analytic::point_to_cone(point, cone))),
        FaceSurface::Sphere(sph) => Ok(Some(analytic::point_to_sphere(point, sph))),
        FaceSurface::Torus(tor) => Ok(Some(analytic::point_to_torus(point, tor))),
        FaceSurface::Nurbs(nurbs) => {
            match brepkit_math::nurbs::projection::project_point_to_surface(nurbs, point, 1e-7) {
                Ok(proj) => Ok(Some((proj.distance, proj.point))),
                Err(_) => Ok(None),
            }
        }
    }
}

/// Point-to-polygon distance for planar faces.
///
/// Projects the point onto the plane, checks if inside polygon, otherwise
/// finds the closest point on polygon edges.
fn point_to_polygon_distance(
    point: Point3,
    polygon: &[Point3],
    normal: Vec3,
    d: f64,
) -> Option<(f64, Point3)> {
    if polygon.len() < 3 {
        return None;
    }

    // Project point onto plane.
    let (_, projected) = analytic::point_to_plane(point, normal, d);

    // Check if projected point is inside polygon.
    if crate::util::point_in_polygon_3d(&projected, polygon, &normal) {
        let dist = (point - projected).length();
        return Some((dist, projected));
    }

    // Otherwise, find closest point on polygon edges.
    let mut best_dist = f64::INFINITY;
    let mut best_pt = polygon[0];
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (dist, closest) = point_to_segment(point, polygon[i], polygon[j]);
        if dist < best_dist {
            best_dist = dist;
            best_pt = closest;
        }
    }
    Some((best_dist, best_pt))
}

/// Distance from point to line segment.
fn point_to_segment(point: Point3, a: Point3, b: Point3) -> (f64, Point3) {
    let ab = b - a;
    let ap = point - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-30 {
        return ((point - a).length(), a);
    }
    let t = (ap.dot(ab) / len_sq).clamp(0.0, 1.0);
    let closest = Point3::new(
        ab.x().mul_add(t, a.x()),
        ab.y().mul_add(t, a.y()),
        ab.z().mul_add(t, a.z()),
    );
    ((point - closest).length(), closest)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface, ToroidalSurface};

    #[test]
    fn point_to_sphere_outside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let (dist, closest) = analytic::point_to_sphere(Point3::new(3.0, 0.0, 0.0), &sphere);
        assert!(
            (dist - 2.0).abs() < 1e-10,
            "distance should be 2.0, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn point_to_sphere_inside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let (dist, closest) = analytic::point_to_sphere(Point3::new(0.5, 0.0, 0.0), &sphere);
        assert!(
            (dist - 0.5).abs() < 1e-10,
            "distance should be 0.5, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn point_to_cylinder_outside() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let (dist, closest) = analytic::point_to_cylinder(Point3::new(2.0, 0.0, 0.0), &cyl);
        assert!(
            (dist - 1.0).abs() < 1e-10,
            "distance should be 1.0, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn point_to_plane_above() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let (dist, closest) = analytic::point_to_plane(Point3::new(0.0, 0.0, 5.0), normal, d);
        assert!(
            (dist - 5.0).abs() < 1e-10,
            "distance should be 5.0, got {dist}"
        );
        assert!(closest.x().abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn point_to_torus_outside() {
        // Torus at origin with major_radius=3, minor_radius=1, Z-axis.
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        // Point at (6, 0, 0): major circle closest is (3,0,0), tube dist = 3, minor_r = 1.
        let (dist, closest) = analytic::point_to_torus(Point3::new(6.0, 0.0, 0.0), &torus);
        assert!(
            (dist - 2.0).abs() < 1e-10,
            "distance should be 2.0, got {dist}"
        );
        assert!((closest.x() - 4.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }
}
