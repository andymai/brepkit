//! Closed-form point-to-surface distance for analytic surfaces.

use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};

/// Closest point on a cylinder to a given point.
///
/// Returns `(distance, closest_point)`.
pub fn point_to_cylinder(point: Point3, cyl: &CylindricalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - cyl.origin().x(),
        point.y() - cyl.origin().y(),
        point.z() - cyl.origin().z(),
    );
    // Project onto axis to get height parameter.
    let h = pv.dot(cyl.axis());
    // Radial vector: pv - h * axis.
    let radial = Vec3::new(
        pv.x() - h * cyl.axis().x(),
        pv.y() - h * cyl.axis().y(),
        pv.z() - h * cyl.axis().z(),
    );
    let r_len = radial.length();

    let closest = if r_len < 1e-15 {
        // Point is on the axis — pick any radial direction.
        cyl.evaluate(0.0, h)
    } else {
        // Closest point is at the same height, on the surface.
        let scale = cyl.radius() / r_len;
        Point3::new(
            cyl.origin().x() + radial.x() * scale + h * cyl.axis().x(),
            cyl.origin().y() + radial.y() * scale + h * cyl.axis().y(),
            cyl.origin().z() + radial.z() * scale + h * cyl.axis().z(),
        )
    };
    ((point - closest).length(), closest)
}

/// Closest point on a cone to a given point.
///
/// Projects the point onto the generatrix line from the apex.
/// Returns `(distance, closest_point)`.
pub fn point_to_cone(point: Point3, cone: &ConicalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - cone.apex().x(),
        point.y() - cone.apex().y(),
        point.z() - cone.apex().z(),
    );
    let h = pv.dot(cone.axis());

    // Radial component perpendicular to axis.
    let radial = Vec3::new(
        pv.x() - h * cone.axis().x(),
        pv.y() - h * cone.axis().y(),
        pv.z() - h * cone.axis().z(),
    );
    let r_len = radial.length();

    if h <= 0.0 && r_len < 1e-15 {
        // Very close to apex.
        return ((point - cone.apex()).length(), cone.apex());
    }

    // Project onto the cone's generatrix direction.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let v = h.mul_add(sin_a, r_len * cos_a);

    if v <= 0.0 {
        // Closest point is the apex.
        return ((point - cone.apex()).length(), cone.apex());
    }

    // Cone surface point at parameter v along the generatrix.
    let cone_r = v * cos_a;
    let cone_h = v * sin_a;

    let closest = if r_len < 1e-15 {
        Point3::new(
            cone.apex().x() + cone_h * cone.axis().x(),
            cone.apex().y() + cone_h * cone.axis().y(),
            cone.apex().z() + cone_h * cone.axis().z(),
        )
    } else {
        let radial_dir_x = radial.x() / r_len;
        let radial_dir_y = radial.y() / r_len;
        let radial_dir_z = radial.z() / r_len;
        Point3::new(
            cone.apex().x() + cone_h * cone.axis().x() + cone_r * radial_dir_x,
            cone.apex().y() + cone_h * cone.axis().y() + cone_r * radial_dir_y,
            cone.apex().z() + cone_h * cone.axis().z() + cone_r * radial_dir_z,
        )
    };

    ((point - closest).length(), closest)
}

/// Closest point on a sphere to a given point.
///
/// Radial projection from the sphere center.
/// Returns `(distance, closest_point)`.
pub fn point_to_sphere(point: Point3, sphere: &SphericalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - sphere.center().x(),
        point.y() - sphere.center().y(),
        point.z() - sphere.center().z(),
    );
    let dist_to_center = pv.length();

    if dist_to_center < 1e-15 {
        // Point is at the center — closest surface point is arbitrary.
        let closest = Point3::new(
            sphere.center().x() + sphere.radius(),
            sphere.center().y(),
            sphere.center().z(),
        );
        return (sphere.radius(), closest);
    }

    let scale = sphere.radius() / dist_to_center;
    let closest = Point3::new(
        sphere.center().x() + pv.x() * scale,
        sphere.center().y() + pv.y() * scale,
        sphere.center().z() + pv.z() * scale,
    );
    ((dist_to_center - sphere.radius()).abs(), closest)
}

/// Closest point on a torus to a given point.
///
/// Projects onto the major circle first, then onto the minor circle.
/// Returns `(distance, closest_point)`.
pub fn point_to_torus(point: Point3, torus: &ToroidalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - torus.center().x(),
        point.y() - torus.center().y(),
        point.z() - torus.center().z(),
    );

    let z_axis = torus.z_axis();
    let h = pv.dot(z_axis);

    // Radial projection in the equatorial plane.
    let radial = Vec3::new(
        pv.x() - h * z_axis.x(),
        pv.y() - h * z_axis.y(),
        pv.z() - h * z_axis.z(),
    );
    let r_len = radial.length();

    let major_r = torus.major_radius();
    let minor_r = torus.minor_radius();

    // Closest point on major circle.
    let (major_closest_x, major_closest_y, major_closest_z) = if r_len < 1e-15 {
        // On the axis — pick any direction.
        (
            torus.center().x() + major_r,
            torus.center().y(),
            torus.center().z(),
        )
    } else {
        let scale = major_r / r_len;
        (
            torus.center().x() + radial.x() * scale,
            torus.center().y() + radial.y() * scale,
            torus.center().z() + radial.z() * scale,
        )
    };

    // Vector from major circle point to query point.
    let tube_vec = Vec3::new(
        point.x() - major_closest_x,
        point.y() - major_closest_y,
        point.z() - major_closest_z,
    );
    let tube_dist = tube_vec.length();

    if tube_dist < 1e-15 {
        // Point is on the major circle — closest torus point is minor_r away.
        let dir = if r_len < 1e-15 {
            z_axis
        } else {
            Vec3::new(radial.x() / r_len, radial.y() / r_len, radial.z() / r_len)
        };
        let closest = Point3::new(
            major_closest_x + minor_r * dir.x(),
            major_closest_y + minor_r * dir.y(),
            major_closest_z + minor_r * dir.z(),
        );
        return (minor_r, closest);
    }

    let tube_scale = minor_r / tube_dist;
    let closest = Point3::new(
        major_closest_x + tube_vec.x() * tube_scale,
        major_closest_y + tube_vec.y() * tube_scale,
        major_closest_z + tube_vec.z() * tube_scale,
    );
    ((tube_dist - minor_r).abs(), closest)
}

/// Perpendicular distance from a point to an infinite plane.
///
/// The plane is defined by `normal . x = d` where `normal` is unit length.
/// Returns `(distance, closest_point)`.
pub fn point_to_plane(point: Point3, normal: Vec3, d: f64) -> (f64, Point3) {
    let pv = Vec3::new(point.x(), point.y(), point.z());
    let dist = normal.dot(pv) - d;
    let closest = Point3::new(
        point.x() - dist * normal.x(),
        point.y() - dist * normal.y(),
        point.z() - dist * normal.z(),
    );
    (dist.abs(), closest)
}
