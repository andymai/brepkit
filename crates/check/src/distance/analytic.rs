//! Closed-form point-to-surface distance for analytic surfaces.
//!
//! These are thin wrappers around `brepkit_geometry::extrema` that extract
//! `(distance, point)` from a [`SurfaceProjection`].

use brepkit_geometry::extrema::{
    SurfaceProjection, point_to_cone as geo_point_to_cone,
    point_to_cylinder as geo_point_to_cylinder, point_to_plane as geo_point_to_plane,
    point_to_sphere as geo_point_to_sphere, point_to_torus as geo_point_to_torus,
};
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};

/// Extract `(distance, point)` from a [`SurfaceProjection`].
fn extract(proj: SurfaceProjection) -> (f64, Point3) {
    (proj.distance, proj.point)
}

/// Closest point on a cylinder to a given point.
///
/// Returns `(distance, closest_point)`.
pub fn point_to_cylinder(point: Point3, cyl: &CylindricalSurface) -> (f64, Point3) {
    extract(geo_point_to_cylinder(point, cyl))
}

/// Closest point on a cone to a given point.
///
/// Returns `(distance, closest_point)`.
pub fn point_to_cone(point: Point3, cone: &ConicalSurface) -> (f64, Point3) {
    extract(geo_point_to_cone(point, cone))
}

/// Closest point on a sphere to a given point.
///
/// Returns `(distance, closest_point)`.
pub fn point_to_sphere(point: Point3, sphere: &SphericalSurface) -> (f64, Point3) {
    extract(geo_point_to_sphere(point, sphere))
}

/// Closest point on a torus to a given point.
///
/// Returns `(distance, closest_point)`.
pub fn point_to_torus(point: Point3, torus: &ToroidalSurface) -> (f64, Point3) {
    extract(geo_point_to_torus(point, torus))
}

/// Perpendicular distance from a point to an infinite plane.
///
/// The plane is defined by `normal · x = d` where `normal` is unit length.
/// Returns `(distance, closest_point)`.
pub fn point_to_plane(point: Point3, normal: Vec3, d: f64) -> (f64, Point3) {
    // Convert plane equation `normal · x = d` to an origin point on the plane.
    // For unit `normal`, the nearest origin is `normal * d`.
    let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);
    extract(geo_point_to_plane(point, origin, normal))
}
