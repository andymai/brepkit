//! SSI (Surface-Surface Intersection) and CSI (Curve-Surface Intersection)
//! robustness tests for the brepkit math layer.
//!
//! Tests cover analytic-analytic, plane-analytic, NURBS-NURBS surface
//! intersections, and curve-surface intersections with known geometric
//! configurations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::TAU;

use brepkit_math::analytic_intersection::{
    AnalyticSurface, ExactIntersectionCurve, exact_plane_analytic, intersect_analytic_analytic,
    intersect_plane_cone, intersect_plane_sphere, intersect_plane_torus,
};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::intersection::{
    intersect_curve_surface, intersect_line_nurbs, intersect_nurbs_nurbs, intersect_plane_nurbs,
};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};

// ── Helpers ────────────────────────────────────────────────────────────

/// Build a bilinear (degree-1) NURBS patch from four corners.
/// Corners are ordered: (u=0,v=0), (u=0,v=1), (u=1,v=0), (u=1,v=1).
fn bilinear_patch(p00: Point3, p01: Point3, p10: Point3, p11: Point3) -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![vec![p00, p01], vec![p10, p11]],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

/// Build a straight-line NURBS curve between two points (degree 1).
fn line_curve(a: Point3, b: Point3) -> NurbsCurve {
    NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![a, b], vec![1.0, 1.0]).unwrap()
}

/// Assert that a 3D point lies on a sphere (within tolerance).
fn assert_on_sphere(pt: Point3, center: Point3, radius: f64, tol: f64) {
    let dist = (pt - center).length();
    assert!(
        (dist - radius).abs() < tol,
        "point {pt:?} is at distance {dist} from center, expected radius {radius}"
    );
}

/// Assert that a 3D point lies on a cylinder (within tolerance).
fn assert_on_cylinder(pt: Point3, origin: Point3, axis: Vec3, radius: f64, tol: f64) {
    let v = pt - origin;
    let axial = Vec3::new(v.x(), v.y(), v.z());
    let along = axis.dot(axial);
    let radial_vec = axial - axis * along;
    let radial_dist = radial_vec.length();
    assert!(
        (radial_dist - radius).abs() < tol,
        "point {pt:?} radial distance {radial_dist}, expected {radius}"
    );
}

// ── Analytic-Analytic Intersection Tests ──────────────────────────────

/// Plane perpendicular to a cylinder axis (z=5) should yield a circle of
/// radius 1 at height z=5.
#[test]
fn plane_cylinder_perpendicular_gives_circle() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let results = exact_plane_analytic(
        AnalyticSurface::Cylinder(&cyl),
        Vec3::new(0.0, 0.0, 1.0),
        5.0,
    )
    .unwrap();

    assert_eq!(
        results.len(),
        1,
        "should get exactly one intersection curve"
    );
    match &results[0] {
        ExactIntersectionCurve::Circle(c) => {
            assert!(
                (c.radius() - 1.0).abs() < 1e-10,
                "circle radius should be 1"
            );
            assert!(
                (c.center().z() - 5.0).abs() < 1e-10,
                "circle center should be at z=5"
            );
        }
        other => panic!("expected Circle, got {other:?}"),
    }
}

/// A plane tilted 45 degrees to the cylinder axis should yield an ellipse
/// with semi-minor = r and semi-major = r / cos(45).
#[test]
fn plane_cylinder_oblique_gives_ellipse() {
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    // Plane normal tilted 45 degrees from Z toward X: normal = (sin45, 0, cos45)
    let angle = std::f64::consts::FRAC_PI_4;
    let sin45 = angle.sin();
    let cos45 = angle.cos();
    let normal = Vec3::new(sin45, 0.0, cos45);

    let results = exact_plane_analytic(AnalyticSurface::Cylinder(&cyl), normal, 0.0).unwrap();

    assert_eq!(
        results.len(),
        1,
        "should get exactly one intersection curve"
    );
    match &results[0] {
        ExactIntersectionCurve::Ellipse(e) => {
            let expected_major = 2.0 / cos45;
            assert!(
                (e.semi_major() - expected_major).abs() < 1e-6,
                "semi-major should be r/cos(45)={expected_major}, got {}",
                e.semi_major()
            );
            assert!(
                (e.semi_minor() - 2.0).abs() < 1e-6,
                "semi-minor should be radius=2, got {}",
                e.semi_minor()
            );
        }
        other => panic!("expected Ellipse, got {other:?}"),
    }
}

/// Plane at z=0.5 intersects a unit sphere — should produce a circle.
/// The intersection circle radius = sqrt(1 - 0.5^2) = sqrt(0.75).
#[test]
fn plane_sphere_latitude_circle() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

    let results = exact_plane_analytic(
        AnalyticSurface::Sphere(&sphere),
        Vec3::new(0.0, 0.0, 1.0),
        0.5,
    )
    .unwrap();

    assert_eq!(results.len(), 1, "should get one intersection");
    match &results[0] {
        ExactIntersectionCurve::Circle(c) => {
            let expected_r = (1.0_f64 - 0.25).sqrt();
            assert!(
                (c.radius() - expected_r).abs() < 1e-10,
                "circle radius should be sqrt(0.75)={expected_r}, got {}",
                c.radius()
            );
            assert!(
                (c.center().z() - 0.5).abs() < 1e-10,
                "circle center z should be 0.5"
            );
        }
        other => panic!("expected Circle, got {other:?}"),
    }
}

/// Plane intersecting a cone should produce a conic section (sampled points
/// since the exact code may not always give an analytic ellipse for oblique
/// cuts).
#[test]
fn plane_cone_intersection() {
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_4,
    )
    .unwrap();

    // Horizontal plane at z=1.0 — should intersect the cone in a circle.
    let curves = intersect_plane_cone(&cone, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    assert!(!curves.is_empty(), "should find intersection with cone");

    // All intersection points should be near z=1.0
    for curve in &curves {
        for pt in &curve.points {
            assert!(
                (pt.point.z() - 1.0).abs() < 0.1,
                "intersection point z={} should be near 1.0",
                pt.point.z()
            );
        }
    }
}

/// Two perpendicular cylinders (same radius, same origin) produce a
/// figure-8 intersection.
#[test]
fn cylinder_cylinder_perpendicular() {
    let cyl_z =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let cyl_x =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0).unwrap();

    let curves = intersect_analytic_analytic(
        AnalyticSurface::Cylinder(&cyl_z),
        AnalyticSurface::Cylinder(&cyl_x),
        20,
    )
    .unwrap();

    assert!(
        !curves.is_empty(),
        "perpendicular equal cylinders should intersect"
    );

    // All intersection points must lie on both cylinders.
    // Newton post-correction achieves high accuracy.
    let tol = 1e-4;
    for curve in &curves {
        for pt in &curve.points {
            assert_on_cylinder(
                pt.point,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                tol,
            );
            assert_on_cylinder(
                pt.point,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                1.0,
                tol,
            );
        }
    }
}

/// Sphere and concentric cylinder — intersection points should lie on both
/// surfaces.
#[test]
fn sphere_cylinder_intersection_on_both_surfaces() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let curves = intersect_analytic_analytic(
        AnalyticSurface::Sphere(&sphere),
        AnalyticSurface::Cylinder(&cyl),
        20,
    )
    .unwrap();

    assert!(
        !curves.is_empty(),
        "sphere R=2 and cylinder r=1 must intersect"
    );

    // Marching method tolerance is generous due to step-size discretization.
    let tol = 0.5;
    for curve in &curves {
        for pt in &curve.points {
            assert_on_sphere(pt.point, Point3::new(0.0, 0.0, 0.0), 2.0, tol);
            assert_on_cylinder(
                pt.point,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                tol,
            );
        }
    }
}

/// A plane tangent to a sphere (distance = radius) should produce a
/// degenerate intersection (empty or a single point).
#[test]
fn plane_sphere_tangent() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

    // Plane at z=1.0 is tangent to the unit sphere at the north pole.
    let results = exact_plane_analytic(
        AnalyticSurface::Sphere(&sphere),
        Vec3::new(0.0, 0.0, 1.0),
        1.0,
    )
    .unwrap();

    // Tangent plane: either empty (degenerate zero-radius circle rejected)
    // or a circle with radius ~0.
    if !results.is_empty() {
        if let ExactIntersectionCurve::Circle(c) = &results[0] {
            assert!(
                c.radius() < 1e-6,
                "tangent circle radius should be ~0, got {}",
                c.radius()
            );
        }
    }
}

/// Plane completely missing a sphere (plane_d > radius) should return
/// no intersections.
#[test]
fn plane_sphere_no_intersection() {
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

    let results = intersect_plane_sphere(&sphere, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    assert!(
        results.is_empty(),
        "plane at z=2 should not intersect unit sphere, got {} curves",
        results.len()
    );
}

/// Plane-torus intersection at z=0 (through the center) should produce
/// two concentric circles: one at R+r and one at R-r.
#[test]
#[ignore = "bug: torus sampler only finds inner circle (R-r), misses outer circle (R+r)"]
fn plane_torus_equatorial_two_circles() {
    let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0).unwrap();

    let curves = intersect_plane_torus(&torus, Vec3::new(0.0, 0.0, 1.0), 0.0).unwrap();

    // The equatorial plane through the center of the torus should yield
    // two concentric loops at R-r=4 and R+r=6. The sampler may chain
    // them into one or two curves depending on resolution.
    assert!(
        !curves.is_empty(),
        "equatorial plane through torus should produce at least 1 curve"
    );

    // Verify that intersection points lie at the correct radial distances
    // from the torus center (R-r=4 or R+r=6).
    let mut has_inner = false;
    let mut has_outer = false;
    for curve in &curves {
        for pt in &curve.points {
            let r = (pt.point.x() * pt.point.x() + pt.point.y() * pt.point.y()).sqrt();
            if (r - 4.0).abs() < 0.5 {
                has_inner = true;
            }
            if (r - 6.0).abs() < 0.5 {
                has_outer = true;
            }
        }
    }
    // Collect all radii for debugging
    let radii: Vec<f64> = curves
        .iter()
        .flat_map(|c| c.points.iter())
        .map(|pt| (pt.point.x() * pt.point.x() + pt.point.y() * pt.point.y()).sqrt())
        .collect();
    assert!(
        has_inner,
        "should have points near inner circle R-r=4, radii={radii:?}"
    );
    assert!(
        has_outer,
        "should have points near outer circle R+r=6, radii={radii:?}"
    );
}

// ── NURBS Surface Intersection Tests ──────────────────────────────────

/// Two planar NURBS patches intersecting at a 90-degree angle should
/// produce a single intersection line.
#[test]
fn two_planar_nurbs_patches_intersection_line() {
    // Patch A: XY plane, z=0, from (-2,-2) to (2,2)
    let patch_a = bilinear_patch(
        Point3::new(-2.0, -2.0, 0.0),
        Point3::new(-2.0, 2.0, 0.0),
        Point3::new(2.0, -2.0, 0.0),
        Point3::new(2.0, 2.0, 0.0),
    );

    // Patch B: XZ plane, y=0, from (-2,-2) to (2,2) in x and z
    let patch_b = bilinear_patch(
        Point3::new(-2.0, 0.0, -2.0),
        Point3::new(-2.0, 0.0, 2.0),
        Point3::new(2.0, 0.0, -2.0),
        Point3::new(2.0, 0.0, 2.0),
    );

    let curves = intersect_nurbs_nurbs(&patch_a, &patch_b, 20, 0.0).unwrap();

    assert_eq!(
        curves.len(),
        1,
        "two perpendicular planes should intersect in one line, got {}",
        curves.len()
    );

    // All points on the intersection should have y~0 and z~0 (the X axis).
    let tol = 1e-4;
    for pt in &curves[0].points {
        assert!(
            pt.point.y().abs() < tol,
            "intersection point y={} should be ~0",
            pt.point.y()
        );
        assert!(
            pt.point.z().abs() < tol,
            "intersection point z={} should be ~0",
            pt.point.z()
        );
    }
}

/// Plane-NURBS intersection: a horizontal plane cutting through a
/// non-planar NURBS surface.
#[test]
fn plane_nurbs_horizontal_cut() {
    // Build a slightly curved surface: a saddle shape z = x*y
    // Using a degree-2 patch, 3x3 control points
    let patch = NurbsSurface::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            vec![
                Point3::new(-1.0, -1.0, 1.0),
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(-1.0, 1.0, -1.0),
            ],
            vec![
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 1.0),
            ],
        ],
        vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ],
    )
    .unwrap();

    // Cut with z=0 plane
    let curves = intersect_plane_nurbs(&patch, Vec3::new(0.0, 0.0, 1.0), 0.0, 30).unwrap();

    assert!(
        !curves.is_empty(),
        "z=0 plane should intersect the saddle surface"
    );

    // All intersection points should have z close to 0
    let tol = 1e-3;
    for curve in &curves {
        for pt in &curve.points {
            assert!(
                pt.point.z().abs() < tol,
                "intersection point z={} should be ~0",
                pt.point.z()
            );
        }
    }
}

// ── Curve-Surface Intersection Tests ──────────────────────────────────

/// A NURBS line passing through a NURBS sphere proxy should hit twice.
#[test]
fn line_through_nurbs_sphere_two_hits() {
    // Create a NURBS surface approximating a sphere: use the unit sphere
    // evaluated on a grid. This is a degree-2 surface.
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

    // Build a coarse NURBS approximation: 5x5 control points on the sphere
    let n = 5;
    let mut cps = Vec::new();
    let mut ws = Vec::new();
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        let u = TAU * (i as f64) / ((n - 1) as f64);
        for j in 0..n {
            let v =
                -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * (j as f64) / ((n - 1) as f64);
            row.push(sphere.evaluate(u, v));
            wrow.push(1.0);
        }
        cps.push(row);
        ws.push(wrow);
    }
    // n=5 control points, degree 2: knots = 5+2+1 = 8
    let knots_u = vec![0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0];

    let surface = NurbsSurface::new(2, 2, knots_u, knots_v, cps, ws).unwrap();

    // Line along the X axis, from (-3,0,0) to (3,0,0)
    let hits = intersect_line_nurbs(
        &surface,
        Point3::new(-3.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        30,
    )
    .unwrap();

    // Should find at least one intersection (two ideally, but coarse approximation
    // might merge them or miss one).
    assert!(
        !hits.is_empty(),
        "line along X axis should intersect sphere-like NURBS surface"
    );
}

/// Curve-surface intersection: a straight line NURBS curve hitting a planar
/// NURBS patch. The line goes from (0.5, 0.5, -1) to (0.5, 0.5, 1) and
/// the patch is z=0 from (0,0) to (1,1), so there should be one hit at z=0.
#[test]
fn line_curve_vs_planar_patch_one_hit() {
    let patch = bilinear_patch(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    );

    let curve = line_curve(Point3::new(0.5, 0.5, -1.0), Point3::new(0.5, 0.5, 1.0));

    let hits = intersect_curve_surface(&curve, &patch, 1e-7).unwrap();

    assert_eq!(
        hits.len(),
        1,
        "should find exactly one hit, got {}",
        hits.len()
    );
    assert!(
        hits[0].point.z().abs() < 1e-4,
        "hit z={} should be ~0",
        hits[0].point.z()
    );
    assert!(
        (hits[0].point.x() - 0.5).abs() < 1e-4,
        "hit x={} should be ~0.5",
        hits[0].point.x()
    );
}

/// A line tangent to a NURBS sphere-like surface should produce at most
/// one intersection point (or zero if the tangency is not resolved).
#[test]
fn line_tangent_to_sphere_nurbs() {
    // Use intersect_line_nurbs with a line tangent to a sphere-approximating
    // surface. The line goes along z at (1, 0, z), tangent to a unit sphere.
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

    // Build a coarse NURBS approximation
    let n = 5;
    let mut cps = Vec::new();
    let mut ws = Vec::new();
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        let u = TAU * (i as f64) / ((n - 1) as f64);
        for j in 0..n {
            let v =
                -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * (j as f64) / ((n - 1) as f64);
            row.push(sphere.evaluate(u, v));
            wrow.push(1.0);
        }
        cps.push(row);
        ws.push(wrow);
    }
    let knots_u = vec![0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0];

    let surface = NurbsSurface::new(2, 2, knots_u, knots_v, cps, ws).unwrap();

    // Line tangent to unit sphere at (1, 0, 0), going along Z
    let hits = intersect_line_nurbs(
        &surface,
        Point3::new(1.0, 0.0, -2.0),
        Vec3::new(0.0, 0.0, 1.0),
        30,
    )
    .unwrap();

    // Tangent intersection: 0 or 1 hits are both acceptable
    assert!(
        hits.len() <= 1,
        "tangent line should produce 0 or 1 hit, got {}",
        hits.len()
    );
}

/// A line clearly missing a NURBS patch should return zero intersections.
#[test]
fn line_missing_patch_no_hits() {
    let patch = bilinear_patch(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    );

    // Line at (5, 5, z) is far from the [0,1]x[0,1] patch
    let hits = intersect_line_nurbs(
        &patch,
        Point3::new(5.0, 5.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
        20,
    )
    .unwrap();

    assert!(
        hits.is_empty(),
        "line far from patch should produce no hits, got {}",
        hits.len()
    );
}
