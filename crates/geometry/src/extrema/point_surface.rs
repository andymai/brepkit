//! Point-to-surface projection algorithms.
//!
//! Provides analytic fast paths for all standard analytic surface types
//! ([`CylindricalSurface`], [`ConicalSurface`], [`SphericalSurface`],
//! [`ToroidalSurface`]) and a generic Newton-Raphson solver for any
//! [`ParametricSurface`].

use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};

use super::SurfaceProjection;

// ── Analytic fast paths ──────────────────────────────────────────────────────

/// Project a point onto an infinite plane.
///
/// The plane is defined by an `origin` point and a unit `normal` vector.
/// The u and v parameters returned are measured in the plane's local frame;
/// since we don't track a basis here, they are set to 0.
///
/// Returns a [`SurfaceProjection`] with the perpendicular foot and the signed
/// distance (`distance` is always non-negative).
#[must_use]
pub fn point_to_plane(point: Point3, origin: Point3, normal: Vec3) -> SurfaceProjection {
    // Signed distance along the normal: d = normal · (point - origin)
    let diff = Vec3::new(
        point.x() - origin.x(),
        point.y() - origin.y(),
        point.z() - origin.z(),
    );
    let d = normal.dot(diff);
    let closest = Point3::new(
        point.x() - d * normal.x(),
        point.y() - d * normal.y(),
        point.z() - d * normal.z(),
    );
    SurfaceProjection {
        distance: d.abs(),
        point: closest,
        u: 0.0,
        v: 0.0,
    }
}

/// Project a point onto an infinite cylinder.
///
/// Projects the point onto the cylinder's axis to find the height parameter
/// `v`, then computes the radial closest point at that height. When the
/// point lies on the axis (degenerate case) the u-parameter is 0.
#[must_use]
pub fn point_to_cylinder(point: Point3, cyl: &CylindricalSurface) -> SurfaceProjection {
    let pv = Vec3::new(
        point.x() - cyl.origin().x(),
        point.y() - cyl.origin().y(),
        point.z() - cyl.origin().z(),
    );
    // Height along axis (v parameter).
    let h = pv.dot(cyl.axis());
    // Radial vector: pv - h * axis.
    let radial = Vec3::new(
        pv.x() - h * cyl.axis().x(),
        pv.y() - h * cyl.axis().y(),
        pv.z() - h * cyl.axis().z(),
    );
    let r_len = radial.length();

    if r_len < 1e-15 {
        // Point is on the axis — pick u = 0 arbitrarily.
        let closest = cyl.evaluate(0.0, h);
        SurfaceProjection {
            distance: (point - closest).length(),
            point: closest,
            u: 0.0,
            v: h,
        }
    } else {
        // u = atan2(radial · y_axis, radial · x_axis)
        let u = (radial.dot(cyl.y_axis())).atan2(radial.dot(cyl.x_axis()));
        let scale = cyl.radius() / r_len;
        let closest = Point3::new(
            cyl.origin().x() + radial.x() * scale + h * cyl.axis().x(),
            cyl.origin().y() + radial.y() * scale + h * cyl.axis().y(),
            cyl.origin().z() + radial.z() * scale + h * cyl.axis().z(),
        );
        SurfaceProjection {
            distance: (point - closest).length(),
            point: closest,
            u,
            v: h,
        }
    }
}

/// Project a point onto a cone.
///
/// Handles the apex case, points interior to the cone that project to the apex,
/// and the general case where the point projects onto a generator line.
#[must_use]
pub fn point_to_cone(point: Point3, cone: &ConicalSurface) -> SurfaceProjection {
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
        // Very close to apex — return apex with u=0, v=0.
        return SurfaceProjection {
            distance: (point - cone.apex()).length(),
            point: cone.apex(),
            u: 0.0,
            v: 0.0,
        };
    }

    // Project onto the cone's generatrix direction.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let v = h.mul_add(sin_a, r_len * cos_a);

    if v <= 0.0 {
        // Closest point is the apex.
        return SurfaceProjection {
            distance: (point - cone.apex()).length(),
            point: cone.apex(),
            u: 0.0,
            v: 0.0,
        };
    }

    // Cone surface point at parameter v along the generatrix.
    let cone_r = v * cos_a;
    let cone_h = v * sin_a;

    let (closest, u) = if r_len < 1e-15 {
        let closest = Point3::new(
            cone.apex().x() + cone_h * cone.axis().x(),
            cone.apex().y() + cone_h * cone.axis().y(),
            cone.apex().z() + cone_h * cone.axis().z(),
        );
        (closest, 0.0_f64)
    } else {
        let radial_dir_x = radial.x() / r_len;
        let radial_dir_y = radial.y() / r_len;
        let radial_dir_z = radial.z() / r_len;
        let closest = Point3::new(
            cone.apex().x() + cone_h * cone.axis().x() + cone_r * radial_dir_x,
            cone.apex().y() + cone_h * cone.axis().y() + cone_r * radial_dir_y,
            cone.apex().z() + cone_h * cone.axis().z() + cone_r * radial_dir_z,
        );
        let radial_vec = Vec3::new(radial_dir_x, radial_dir_y, radial_dir_z);
        let u = radial_vec
            .dot(cone.y_axis())
            .atan2(radial_vec.dot(cone.x_axis()));
        (closest, u)
    };

    SurfaceProjection {
        distance: (point - closest).length(),
        point: closest,
        u,
        v,
    }
}

/// Project a point onto a sphere.
///
/// Projects radially from the sphere center. When the point is at the center,
/// an arbitrary surface point is returned.
#[must_use]
pub fn point_to_sphere(point: Point3, sphere: &SphericalSurface) -> SurfaceProjection {
    let pv = Vec3::new(
        point.x() - sphere.center().x(),
        point.y() - sphere.center().y(),
        point.z() - sphere.center().z(),
    );
    let dist_to_center = pv.length();

    if dist_to_center < 1e-15 {
        // Point at center — arbitrary direction, u=0, v=0.
        let closest = Point3::new(
            sphere.center().x() + sphere.radius(),
            sphere.center().y(),
            sphere.center().z(),
        );
        return SurfaceProjection {
            distance: sphere.radius(),
            point: closest,
            u: 0.0,
            v: 0.0,
        };
    }

    let scale = sphere.radius() / dist_to_center;
    let closest = Point3::new(
        sphere.center().x() + pv.x() * scale,
        sphere.center().y() + pv.y() * scale,
        sphere.center().z() + pv.z() * scale,
    );

    // u = azimuth angle, v = elevation angle.
    let (u, v) = sphere.project_point(point);

    SurfaceProjection {
        distance: (dist_to_center - sphere.radius()).abs(),
        point: closest,
        u,
        v,
    }
}

/// Project a point onto a torus.
///
/// Projects onto the major circle first, then onto the minor circle tube.
#[must_use]
pub fn point_to_torus(point: Point3, torus: &ToroidalSurface) -> SurfaceProjection {
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
    let (major_cx, major_cy, major_cz, u) = if r_len < 1e-15 {
        // On the axis — pick u = 0.
        (
            torus.center().x() + major_r * torus.x_axis().x(),
            torus.center().y() + major_r * torus.x_axis().y(),
            torus.center().z() + major_r * torus.x_axis().z(),
            0.0_f64,
        )
    } else {
        let scale = major_r / r_len;
        let u = radial.dot(torus.y_axis()).atan2(radial.dot(torus.x_axis()));
        (
            torus.center().x() + radial.x() * scale,
            torus.center().y() + radial.y() * scale,
            torus.center().z() + radial.z() * scale,
            u,
        )
    };

    // Vector from major circle point to query point.
    let tube_vec = Vec3::new(
        point.x() - major_cx,
        point.y() - major_cy,
        point.z() - major_cz,
    );
    let tube_dist = tube_vec.length();

    if tube_dist < 1e-15 {
        // Point is on the major circle — closest point is minor_r away.
        let dir = if r_len < 1e-15 {
            torus.x_axis()
        } else {
            Vec3::new(radial.x() / r_len, radial.y() / r_len, radial.z() / r_len)
        };
        let closest = Point3::new(
            major_cx + minor_r * dir.x(),
            major_cy + minor_r * dir.y(),
            major_cz + minor_r * dir.z(),
        );
        return SurfaceProjection {
            distance: minor_r,
            point: closest,
            u,
            v: 0.0,
        };
    }

    let tube_scale = minor_r / tube_dist;
    let closest = Point3::new(
        major_cx + tube_vec.x() * tube_scale,
        major_cy + tube_vec.y() * tube_scale,
        major_cz + tube_vec.z() * tube_scale,
    );

    // v = angle of tube vector relative to radial/z directions.
    // v = atan2(tube component along z_axis, tube component along radial dir)
    let v = if r_len < 1e-15 {
        tube_vec.dot(z_axis).atan2(tube_vec.dot(torus.x_axis()))
    } else {
        let radial_dir = Vec3::new(radial.x() / r_len, radial.y() / r_len, radial.z() / r_len);
        tube_vec.dot(z_axis).atan2(tube_vec.dot(radial_dir))
    };

    SurfaceProjection {
        distance: (tube_dist - minor_r).abs(),
        point: closest,
        u,
        v,
    }
}

// ── Generic Newton-Raphson solver ────────────────────────────────────────────

/// Maximum Newton iterations.
const MAX_ITER: usize = 50;

/// Convergence tolerance on the parameter update magnitude.
const PARAM_TOL: f64 = 1e-10;

/// Grid size for initial sampling phase.
const GRID_N: usize = 8;

/// Project a point onto any [`ParametricSurface`] over `[u_range] × [v_range]`.
///
/// **Algorithm:**
/// 1. Sample the surface on an [`GRID_N`]×[`GRID_N`] grid to find the global
///    closest sample (avoids local-minimum traps on non-convex surfaces).
/// 2. Refine the best sample using Newton-Raphson on the two-variable
///    stationarity conditions:
///    - `dot(S(u,v) - P, ∂S/∂u) = 0`
///    - `dot(S(u,v) - P, ∂S/∂v) = 0`
///
/// Parameters are clamped to their ranges after each Newton step.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn point_to_surface<S: ParametricSurface>(
    point: Point3,
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> SurfaceProjection {
    let (u0, u1) = u_range;
    let (v0, v1) = v_range;

    // ── Phase 1: grid search ──────────────────────────────────────────────────
    let mut best_u = (u0 + u1) * 0.5;
    let mut best_v = (v0 + v1) * 0.5;
    let mut best_dist_sq = f64::INFINITY;

    for iu in 0..GRID_N {
        let u = u0 + (u1 - u0) * (iu as f64) / ((GRID_N - 1) as f64);
        for iv in 0..GRID_N {
            let v = v0 + (v1 - v0) * (iv as f64) / ((GRID_N - 1) as f64);
            let p = surface.evaluate(u, v);
            let diff = p - point;
            let d2 = diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z();
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_u = u;
                best_v = v;
            }
        }
    }

    // ── Phase 2: Newton-Raphson ───────────────────────────────────────────────
    // Solve the 2×2 system:
    //   f1(u,v) = dot(S(u,v) - P, Su) = 0
    //   f2(u,v) = dot(S(u,v) - P, Sv) = 0
    //
    // Using Gauss-Newton: J^T J Δx = -J^T f  where J_ij = ∂fi/∂xj
    //   J11 ≈ |Su|^2,  J12 = Su·Sv
    //   J21 = Su·Sv,   J22 ≈ |Sv|^2
    let mut u = best_u;
    let mut v = best_v;

    for _ in 0..MAX_ITER {
        let p = surface.evaluate(u, v);
        let diff = Vec3::new(p.x() - point.x(), p.y() - point.y(), p.z() - point.z());
        let su = surface.partial_u(u, v);
        let sv = surface.partial_v(u, v);

        let f1 = diff.dot(su);
        let f2 = diff.dot(sv);

        let j11 = su.dot(su);
        let j12 = su.dot(sv);
        let j22 = sv.dot(sv);
        let det = j11 * j22 - j12 * j12;

        if det.abs() < f64::EPSILON {
            break;
        }

        let du = (f1 * j22 - f2 * j12) / det;
        let dv = (f2 * j11 - f1 * j12) / det;

        let u_new = (u - du).clamp(u0, u1);
        let v_new = (v - dv).clamp(v0, v1);

        if (u_new - u).abs() < PARAM_TOL && (v_new - v).abs() < PARAM_TOL {
            u = u_new;
            v = v_new;
            break;
        }
        u = u_new;
        v = v_new;
    }

    let closest = surface.evaluate(u, v);
    let diff = closest - point;
    let distance = (diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z()).sqrt();

    SurfaceProjection {
        distance,
        point: closest,
        u,
        v,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::items_after_statements
    )]

    use std::f64::consts::{FRAC_PI_2, TAU};

    use super::*;
    use brepkit_math::vec::Vec3;

    // Helpers
    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── point_to_plane ───────────────────────────────────────────────────────

    #[test]
    fn plane_perpendicular_projection() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let point = Point3::new(3.0, 4.0, 5.0);
        let proj = point_to_plane(point, origin, normal);
        assert!(approx(proj.distance, 5.0, 1e-12), "dist={}", proj.distance);
        assert!(approx(proj.point.x(), 3.0, 1e-12));
        assert!(approx(proj.point.y(), 4.0, 1e-12));
        assert!(approx(proj.point.z(), 0.0, 1e-12));
    }

    #[test]
    fn plane_point_on_plane_zero_distance() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        // Point on the plane (y=2).
        let point = Point3::new(5.0, 2.0, 7.0);
        let proj = point_to_plane(point, origin, normal);
        assert!(proj.distance < 1e-12, "dist={}", proj.distance);
    }

    // ── point_to_sphere ──────────────────────────────────────────────────────

    #[test]
    fn sphere_point_outside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let point = Point3::new(0.0, 0.0, 5.0);
        let proj = point_to_sphere(point, &sphere);
        // Expected: distance = 5 - 2 = 3, closest = (0,0,2).
        assert!(approx(proj.distance, 3.0, 1e-12), "dist={}", proj.distance);
        assert!(approx(proj.point.z(), 2.0, 1e-12));
        assert!(proj.point.x().abs() < 1e-12);
        assert!(proj.point.y().abs() < 1e-12);
    }

    #[test]
    fn sphere_point_inside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0).unwrap();
        let point = Point3::new(0.0, 3.0, 0.0);
        let proj = point_to_sphere(point, &sphere);
        // Distance from center = 3; distance to surface = 5 - 3 = 2.
        assert!(approx(proj.distance, 2.0, 1e-12), "dist={}", proj.distance);
        // Closest must be on the sphere.
        let r = (proj.point.x() * proj.point.x()
            + proj.point.y() * proj.point.y()
            + proj.point.z() * proj.point.z())
        .sqrt();
        assert!(approx(r, 5.0, 1e-12), "not on sphere: r={r}");
    }

    #[test]
    fn sphere_point_at_center_returns_surface_point() {
        let sphere = SphericalSurface::new(Point3::new(1.0, 2.0, 3.0), 4.0).unwrap();
        let proj = point_to_sphere(sphere.center(), &sphere);
        assert!(approx(proj.distance, 4.0, 1e-12), "dist={}", proj.distance);
    }

    // ── point_to_cylinder ────────────────────────────────────────────────────

    #[test]
    fn cylinder_point_perpendicular_to_axis() {
        // Cylinder along Z-axis, radius 3; point at (5,0,2) (perpendicular to axis at h=2).
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let point = Point3::new(5.0, 0.0, 2.0);
        let proj = point_to_cylinder(point, &cyl);
        // Distance from axis = 5, so distance to cylinder = 5 - 3 = 2.
        assert!(approx(proj.distance, 2.0, 1e-12), "dist={}", proj.distance);
        // Closest must be on the cylinder surface.
        let ox = proj.point.x();
        let oy = proj.point.y();
        let r = (ox * ox + oy * oy).sqrt();
        assert!(approx(r, 3.0, 1e-12), "not on cylinder: r={r}");
        // Height preserved.
        assert!(approx(proj.point.z(), 2.0, 1e-12), "z={}", proj.point.z());
    }

    #[test]
    fn cylinder_point_on_axis_uses_u_zero() {
        // Point on the axis — should pick u=0 and distance = radius.
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();
        let point = Point3::new(0.0, 0.0, 4.0);
        let proj = point_to_cylinder(point, &cyl);
        assert!(approx(proj.distance, 2.0, 1e-12), "dist={}", proj.distance);
        assert!(approx(proj.u, 0.0, 1e-12), "u={}", proj.u);
    }

    // ── point_to_torus ───────────────────────────────────────────────────────

    #[test]
    fn torus_point_on_major_circle_returns_minor_radius() {
        // Torus with major=3, minor=1; point on the major circle at (3,0,0).
        let torus = ToroidalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            3.0, // major
            1.0, // minor
        )
        .unwrap();
        let point = Point3::new(3.0, 0.0, 0.0);
        let proj = point_to_torus(point, &torus);
        assert!(approx(proj.distance, 1.0, 1e-12), "dist={}", proj.distance);
    }

    #[test]
    fn torus_point_on_surface_zero_distance() {
        // Point on the torus surface: major=3, minor=1.
        // Surface point at u=0, v=0: (major+minor, 0, 0) = (4,0,0).
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let point = torus.evaluate(0.0, 0.0);
        let proj = point_to_torus(point, &torus);
        assert!(proj.distance < 1e-10, "dist={}", proj.distance);
    }

    // ── point_to_surface (generic) ───────────────────────────────────────────

    #[test]
    fn generic_cylinder_matches_analytic() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let point = Point3::new(5.0, 0.0, 2.0);

        let analytic = point_to_cylinder(point, &cyl);
        let generic = point_to_surface(point, &cyl, (0.0, TAU), (-10.0, 10.0));

        assert!(
            approx(analytic.distance, generic.distance, 1e-4),
            "analytic={} generic={}",
            analytic.distance,
            generic.distance
        );
    }

    #[test]
    fn generic_sphere_matches_analytic() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let point = Point3::new(0.0, 0.0, 5.0);

        let analytic = point_to_sphere(point, &sphere);
        let generic = point_to_surface(point, &sphere, (0.0, TAU), (-FRAC_PI_2, FRAC_PI_2));

        assert!(
            approx(analytic.distance, generic.distance, 1e-4),
            "analytic={} generic={}",
            analytic.distance,
            generic.distance
        );
    }
}
