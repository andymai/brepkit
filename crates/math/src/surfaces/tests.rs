use super::*;
use std::f64::consts::{FRAC_PI_2, PI};

const TOL: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

fn point_approx_eq(a: Point3, b: Point3) -> bool {
    approx_eq(a.x(), b.x()) && approx_eq(a.y(), b.y()) && approx_eq(a.z(), b.z())
}

// ── Cylinder tests ────────────────────────────────

#[test]
fn cylinder_at_origin() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    let p = cyl.evaluate(0.0, 0.0);
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 2.0));

    let p = cyl.evaluate(0.0, 5.0);
    assert!(approx_eq(p.z(), 5.0));
}

#[test]
fn cylinder_normal_is_radial() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let n = cyl.normal(0.0, 0.0);
    // Normal should be perpendicular to axis
    assert!(approx_eq(n.dot(Vec3::new(0.0, 0.0, 1.0)), 0.0));
    // And unit length
    assert!(approx_eq(n.length(), 1.0));
}

#[test]
fn cylinder_zero_radius_error() {
    assert!(
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,)
            .is_err()
    );
}

// ── Cone tests ────────────────────────────────────

#[test]
fn cone_apex() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 4.0,
    )
    .unwrap();

    // At v=0, all points should be at the apex
    let p = cone.evaluate(0.0, 0.0);
    assert!(point_approx_eq(p, Point3::new(0.0, 0.0, 0.0)));
}

#[test]
fn cone_radius_at() {
    let half_angle = PI / 6.0; // 30 degrees
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        half_angle,
    )
    .unwrap();

    let r = cone.radius_at(10.0);
    assert!(approx_eq(r, 10.0 * half_angle.cos()));
}

#[test]
fn cone_invalid_half_angle() {
    assert!(
        ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,).is_err()
    );

    assert!(
        ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            FRAC_PI_2,
        )
        .is_err()
    );
}

// ── Sphere tests ──────────────────────────────────

#[test]
fn sphere_north_pole() {
    let s = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
    let p = s.evaluate(0.0, FRAC_PI_2);
    assert!(point_approx_eq(p, Point3::new(0.0, 0.0, 3.0)));
}

#[test]
fn sphere_equator() {
    let s = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();

    // u=0, v=0 → along x-axis
    let p = s.evaluate(0.0, 0.0);
    assert!(approx_eq(p.x(), 2.0));
    assert!(approx_eq(p.y(), 0.0));
    assert!(approx_eq(p.z(), 0.0));
}

#[test]
fn sphere_normal_is_radial() {
    let sphere = SphericalSurface::new(Point3::new(1.0, 2.0, 3.0), 5.0).unwrap();

    let param_u = 1.0;
    let param_v = 0.5;
    let point = sphere.evaluate(param_u, param_v);
    let normal = sphere.normal(param_u, param_v);

    // Normal should point from center to surface point
    let expected_dir = point - sphere.center();
    let expected_norm = expected_dir.normalize().expect("non-zero");
    assert!(approx_eq(normal.dot(expected_norm), 1.0));
}

#[test]
fn sphere_zero_radius_error() {
    assert!(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 0.0).is_err());
}

// ── Torus tests ───────────────────────────────────

#[test]
fn torus_outer_point() {
    let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();

    // u=0, v=0 → outermost point at (R+r, 0, 0) = (7, 0, 0) along x
    let p = t.evaluate(0.0, 0.0);
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 7.0));
}

#[test]
fn torus_inner_point() {
    let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();

    // v=π → innermost point at distance R-r = 3
    let p = t.evaluate(0.0, PI);
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));
}

#[test]
fn torus_zero_radius_error() {
    assert!(ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 0.0, 1.0).is_err());
    assert!(ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 0.0).is_err());
}

// ── Revolution surface tests ──────────────────────

#[test]
fn revolution_cylinder_like() {
    // A revolution surface with constant radius = cylinder
    let rev = RevolutionSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        vec![3.0, 3.0],
        vec![0.0, 10.0],
    )
    .unwrap();

    let p = rev.evaluate(0.0, 0.0);
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));

    let p = rev.evaluate(0.0, 1.0);
    assert!(approx_eq(p.z(), 10.0));
}

// ── Additional coverage tests ─────────────────────

#[test]
fn cylinder_with_x_aligned_axis() {
    // Tests the other branch of the axis-candidate selection (a.x().abs() >= 0.9)
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0).unwrap();
    assert!(approx_eq(cyl.radius(), 2.0));
    let p = cyl.evaluate(0.0, 3.0);
    assert!(approx_eq(p.x(), 3.0));
}

#[test]
fn cylinder_accessors() {
    let cyl =
        CylindricalSurface::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
    assert!(approx_eq(cyl.origin().x(), 1.0));
    assert!(approx_eq(cyl.axis().z(), 1.0));
    assert!(approx_eq(cyl.radius(), 5.0));
}

#[test]
fn cone_normal() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 4.0,
    )
    .unwrap();
    let n = cone.normal(0.0, 1.0);
    // Normal should be non-zero and perpendicular to surface at that point
    assert!(n.length() > 0.5);
}

#[test]
fn cone_accessors() {
    let cone = ConicalSurface::new(
        Point3::new(1.0, 2.0, 3.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 6.0,
    )
    .unwrap();
    assert!(point_approx_eq(cone.apex(), Point3::new(1.0, 2.0, 3.0)));
    assert!(approx_eq(cone.axis().z(), 1.0));
    assert!(approx_eq(cone.half_angle(), PI / 6.0));
}

#[test]
fn cone_with_x_axis() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        PI / 4.0,
    )
    .unwrap();
    let p = cone.evaluate(0.0, 1.0);
    assert!((p - Point3::new(0.0, 0.0, 0.0)).length() > 0.5);
}

#[test]
fn sphere_with_custom_axis() {
    let s = SphericalSurface::with_axis(Point3::new(0.0, 0.0, 0.0), 2.0, Vec3::new(1.0, 0.0, 0.0))
        .unwrap();
    assert!(approx_eq(s.radius(), 2.0));
    // North pole should be along x
    let p = s.evaluate(0.0, FRAC_PI_2);
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 2.0));
}

#[test]
fn sphere_with_axis_zero_radius_error() {
    assert!(
        SphericalSurface::with_axis(Point3::new(0.0, 0.0, 0.0), 0.0, Vec3::new(0.0, 0.0, 1.0))
            .is_err()
    );
}

#[test]
fn torus_normal() {
    let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();
    let n = t.normal(0.0, 0.0);
    // At u=0, v=0 (outermost point), normal should point radially outward
    assert!(n.length() > 0.5);
}

#[test]
fn torus_accessors() {
    let t = ToroidalSurface::new(Point3::new(1.0, 2.0, 3.0), 5.0, 2.0).unwrap();
    assert!(point_approx_eq(t.center(), Point3::new(1.0, 2.0, 3.0)));
    assert!(approx_eq(t.major_radius(), 5.0));
    assert!(approx_eq(t.minor_radius(), 2.0));
}

#[test]
fn sphere_aabb_at_origin() {
    let s = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
    let bb = s.aabb();
    assert!(point_approx_eq(bb.min, Point3::new(-3.0, -3.0, -3.0)));
    assert!(point_approx_eq(bb.max, Point3::new(3.0, 3.0, 3.0)));
}

#[test]
fn sphere_aabb_offcenter() {
    // The census case: sphere r=6 centered at (5,5,5).
    let s = SphericalSurface::new(Point3::new(5.0, 5.0, 5.0), 6.0).unwrap();
    let bb = s.aabb();
    assert!(point_approx_eq(bb.min, Point3::new(-1.0, -1.0, -1.0)));
    assert!(point_approx_eq(bb.max, Point3::new(11.0, 11.0, 11.0)));
}

#[test]
fn torus_aabb_canonical() {
    // Axis +z: radial reach R+r=13 in x/y, tube ±r=3 in z.
    let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
    let bb = t.aabb();
    assert!(point_approx_eq(bb.min, Point3::new(-13.0, -13.0, -3.0)));
    assert!(point_approx_eq(bb.max, Point3::new(13.0, 13.0, 3.0)));
}

#[test]
fn torus_aabb_x_axis() {
    // Axis along +x: tube ±r=3 in x, radial reach R+r=13 in y/z.
    let t = ToroidalSurface::with_axis(
        Point3::new(0.0, 0.0, 0.0),
        10.0,
        3.0,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    let bb = t.aabb();
    assert!(point_approx_eq(bb.min, Point3::new(-3.0, -13.0, -13.0)));
    assert!(point_approx_eq(bb.max, Point3::new(3.0, 13.0, 13.0)));
}

#[test]
fn revolution_with_x_axis() {
    // Tests the other candidate branch for revolution surfaces
    let rev = RevolutionSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        vec![2.0, 2.0],
        vec![0.0, 5.0],
    )
    .unwrap();
    assert!(point_approx_eq(rev.origin(), Point3::new(0.0, 0.0, 0.0)));
    assert!(approx_eq(rev.axis().x(), 1.0));
}

#[test]
fn revolution_empty_input_error() {
    assert!(
        RevolutionSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            vec![],
            vec![],
        )
        .is_err()
    );
}

#[test]
fn revolution_mismatched_lengths_error() {
    assert!(
        RevolutionSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            vec![1.0, 2.0],
            vec![0.0],
        )
        .is_err()
    );
}

#[test]
fn revolution_midpoint_evaluation() {
    let rev = RevolutionSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        vec![1.0, 3.0, 1.0],
        vec![0.0, 5.0, 10.0],
    )
    .unwrap();
    // At v=0.5 (midpoint of generatrix): radius should be 3.0, height 5.0
    let p = rev.evaluate(0.0, 0.5);
    assert!(approx_eq(p.z(), 5.0));
}

#[test]
fn cylinder_evaluate_at_quarter_circle() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
    let p = cyl.evaluate(FRAC_PI_2, 0.0);
    // At u=π/2, should be at 90° around the cylinder
    assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));
}

#[test]
fn cylinder_project_point_roundtrip() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    // Evaluate at known parameters, then project back.
    let (u_orig, v_orig) = (1.0, 3.0);
    let pt = cyl.evaluate(u_orig, v_orig);
    let (u_proj, v_proj) = cyl.project_point(pt);

    assert!(
        approx_eq(u_proj, u_orig),
        "u should roundtrip: expected {u_orig}, got {u_proj}"
    );
    assert!(
        approx_eq(v_proj, v_orig),
        "v should roundtrip: expected {v_orig}, got {v_proj}"
    );
}

#[test]
fn cylinder_project_external_point() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    // Project a point and verify its v-parameter (axial position).
    let (_, v) = cyl.project_point(Point3::new(5.0, 0.0, 2.0));
    assert!(approx_eq(v, 2.0), "v should be 2, got {v}");

    // Verify that evaluating at the projected parameters gives a point
    // on the cylinder surface closest to the original point.
    let (u, v) = cyl.project_point(Point3::new(3.0, 4.0, 7.0));
    let on_surface = cyl.evaluate(u, v);
    assert!(
        approx_eq(on_surface.z(), 7.0),
        "projected z should be 7.0, got {}",
        on_surface.z()
    );
    // The surface point should be at distance = radius from the axis.
    let radial_dist = (on_surface.x() * on_surface.x() + on_surface.y() * on_surface.y()).sqrt();
    assert!(
        approx_eq(radial_dist, 1.0),
        "surface point should be at radius 1.0, got {radial_dist}"
    );
}

#[test]
fn sphere_project_point_roundtrip() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();

    let (u_orig, v_orig) = (1.5, 0.3);
    let pt = sphere.evaluate(u_orig, v_orig);
    let (u_proj, v_proj) = sphere.project_point(pt);

    assert!(
        approx_eq(u_proj, u_orig),
        "u should roundtrip: expected {u_orig}, got {u_proj}"
    );
    assert!(
        approx_eq(v_proj, v_orig),
        "v should roundtrip: expected {v_orig}, got {v_proj}"
    );
}

#[test]
fn cone_project_point_roundtrip() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 6.0, // 30° half-angle
    )
    .unwrap();

    let (u_orig, v_orig) = (1.2, 2.5);
    let pt = cone.evaluate(u_orig, v_orig);
    let (u_proj, v_proj) = cone.project_point(pt);

    assert!(
        approx_eq(u_proj, u_orig),
        "cone u should roundtrip: expected {u_orig}, got {u_proj}"
    );
    assert!(
        approx_eq(v_proj, v_orig),
        "cone v should roundtrip: expected {v_orig}, got {v_proj}"
    );
}

#[test]
fn cone_project_external_point() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 4.0, // 45° half-angle
    )
    .unwrap();

    // A point above the cone should project with correct v.
    let (_, v) = cone.project_point(Point3::new(5.0, 0.0, 5.0));
    assert!(v > 0.0, "v should be positive above cone apex, got {v}");
}

#[test]
fn torus_project_point_roundtrip() {
    let torus = ToroidalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        3.0, // major radius
        1.0, // minor radius
    )
    .unwrap();

    let (u_orig, v_orig) = (0.8, 1.5);
    let pt = torus.evaluate(u_orig, v_orig);
    let (u_proj, v_proj) = torus.project_point(pt);

    assert!(
        approx_eq(u_proj, u_orig),
        "torus u should roundtrip: expected {u_orig}, got {u_proj}"
    );
    assert!(
        approx_eq(v_proj, v_orig),
        "torus v should roundtrip: expected {v_orig}, got {v_proj}"
    );
}

#[test]
fn torus_project_external_point() {
    let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0).unwrap();

    // Point on the outer equator (x=6, y=0, z=0) should project to u=0, v=0.
    let (u, v) = torus.project_point(Point3::new(6.0, 0.0, 0.0));
    assert!(approx_eq(u, 0.0), "outer equator u should be 0, got {u}");
    assert!(approx_eq(v, 0.0), "outer equator v should be 0, got {v}");

    // Point directly above the tube center at u=0 → v=π/2.
    let (_, v) = torus.project_point(Point3::new(5.0, 0.0, 1.5));
    assert!(
        approx_eq(v, FRAC_PI_2),
        "above tube center v should be π/2, got {v}"
    );
}

// ── Partial derivative tests ─────────────────────

use crate::traits::ParametricSurface;

#[test]
fn cylinder_partial_u_is_tangential() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
    let u = 1.0;
    let v = 2.0;
    let du = ParametricSurface::partial_u(&cyl, u, v);
    let n = cyl.normal(u, v);
    // du should be perpendicular to outward normal
    assert!(du.dot(n).abs() < TOL, "du·n = {} (expected ~0)", du.dot(n));
    // |du| should equal radius
    assert!(
        approx_eq(du.length(), 3.0),
        "|du| = {} (expected 3.0)",
        du.length()
    );
}

#[test]
fn cylinder_partial_v_is_axial() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let dv = ParametricSurface::partial_v(&cyl, 0.5, 1.0);
    assert!(
        approx_eq(dv.x(), 0.0) && approx_eq(dv.y(), 0.0) && approx_eq(dv.z(), 1.0),
        "dv should equal axis (0,0,1), got ({}, {}, {})",
        dv.x(),
        dv.y(),
        dv.z()
    );
}

#[test]
fn sphere_partials_are_perpendicular() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0).unwrap();
    let u = 1.2;
    let v = 0.4;
    let du = ParametricSurface::partial_u(&sphere, u, v);
    let dv = ParametricSurface::partial_v(&sphere, u, v);
    assert!(
        du.dot(dv).abs() < TOL,
        "du·dv = {} (expected ~0)",
        du.dot(dv)
    );
}

#[test]
fn torus_partial_u_magnitude() {
    let big_r = 5.0;
    let small_r = 2.0;
    let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r, small_r).unwrap();
    let u = 0.7;
    let v = 1.3;
    let du = ParametricSurface::partial_u(&torus, u, v);
    let expected_mag = big_r + small_r * v.cos();
    assert!(
        approx_eq(du.length(), expected_mag),
        "|du| = {} (expected {})",
        du.length(),
        expected_mag
    );
}

#[test]
fn cone_partial_v_is_unit_at_v1() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        PI / 6.0,
    )
    .unwrap();
    // dv is independent of v; it should be a unit vector (cos²a + sin²a = 1)
    let dv = ParametricSurface::partial_v(&cone, 0.0, 1.0);
    assert!(
        approx_eq(dv.length(), 1.0),
        "|dv| = {} (expected 1.0)",
        dv.length()
    );
}
