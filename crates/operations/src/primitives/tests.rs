#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_topology::Topology;

use super::*;
use crate::test_helpers::{assert_euler_genus0, assert_volume_near, euler_characteristic};

#[test]
fn make_box_unit_cube() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert_eq!(sh.faces().len(), 6, "box should have 6 faces");

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 1.0).abs() < 1e-10,
        "unit box volume should be 1.0, got {vol}"
    );

    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_box_rectangular() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 24.0).abs() < 1e-10,
        "2x3x4 box volume should be 24.0, got {vol}"
    );

    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_box_corner_at_origin() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    // Box extends from (0,0,0) to (2,2,2), so center of mass is at (1,1,1).
    let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
    assert!(
        (com.x() - 1.0).abs() < 1e-6,
        "com x should be ~1, got {}",
        com.x()
    );
    assert!(
        (com.y() - 1.0).abs() < 1e-6,
        "com y should be ~1, got {}",
        com.y()
    );
    assert!(
        (com.z() - 1.0).abs() < 1e-6,
        "com z should be ~1, got {}",
        com.z()
    );
}

#[test]
fn make_box_zero_dimension_error() {
    let mut topo = Topology::new();
    assert!(make_box(&mut topo, 0.0, 1.0, 1.0).is_err());
    assert!(make_box(&mut topo, 1.0, 0.0, 1.0).is_err());
    assert!(make_box(&mut topo, 1.0, 1.0, 0.0).is_err());
}

#[test]
fn make_box_negative_dimension_error() {
    let mut topo = Topology::new();
    assert!(make_box(&mut topo, -1.0, 1.0, 1.0).is_err());
}

#[test]
fn make_box_manifold_edges() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    brepkit_topology::validation::validate_shell_manifold(sh, &topo)
        .expect("box should be manifold");
}

#[test]
fn make_cylinder_basic() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // Analytic cylinder: 1 lateral + 2 caps = 3 faces
    assert_eq!(sh.faces().len(), 3, "cylinder should have 3 faces");

    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_cylinder_volume() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
    let expected = PI * 1.0_f64.powi(2) * 2.0;
    assert!(
        (vol - expected).abs() / expected < 1e-6,
        "cylinder volume should be ~{expected}, got {vol} (error: {:.1}%)",
        (vol - expected).abs() / expected * 100.0
    );
}

#[test]
fn make_cylinder_zero_radius_error() {
    let mut topo = Topology::new();
    assert!(make_cylinder(&mut topo, 0.0, 1.0).is_err());
}

#[test]
fn make_cylinder_zero_height_error() {
    let mut topo = Topology::new();
    assert!(make_cylinder(&mut topo, 1.0, 0.0).is_err());
}

#[test]
fn make_cone_frustum() {
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(!sh.faces().is_empty(), "cone should have faces");

    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_cone_frustum_volume() {
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
    // V = πh/3 * (r1² + r1*r2 + r2²)
    let expected =
        std::f64::consts::PI * 3.0 / 3.0 * (2.0_f64.powi(2) + 2.0 * 1.0 + 1.0_f64.powi(2));
    let rel_error = (vol - expected).abs() / expected;
    assert!(
        rel_error < 0.01,
        "cone frustum volume should be ~{expected:.3}, got {vol:.3} (error: {rel_error:.1}%)",
    );
}

#[test]
fn make_cone_both_zero_error() {
    let mut topo = Topology::new();
    assert!(make_cone(&mut topo, 0.0, 0.0, 1.0).is_err());
}

#[test]
fn make_cone_negative_radius_error() {
    let mut topo = Topology::new();
    assert!(make_cone(&mut topo, -1.0, 1.0, 1.0).is_err());
}

#[test]
fn make_sphere_basic() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 1.0, 8).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(!sh.faces().is_empty(), "sphere should have faces");

    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_sphere_volume() {
    let mut topo = Topology::new();
    // Use high segment count for better approximation
    let solid = make_sphere(&mut topo, 1.0, 32).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
    let expected = 4.0 / 3.0 * PI;
    // Polygonal approximation — allow 5% tolerance
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "sphere volume should be ~{expected}, got {vol} (error: {:.1}%)",
        (vol - expected).abs() / expected * 100.0
    );
}

#[test]
fn make_sphere_center_of_mass_at_origin() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 1.0, 16).unwrap();

    let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
    assert!(
        com.x().abs() < 1e-6,
        "sphere com x should be ~0, got {}",
        com.x()
    );
    assert!(
        com.y().abs() < 1e-6,
        "sphere com y should be ~0, got {}",
        com.y()
    );
    assert!(
        com.z().abs() < 1e-6,
        "sphere com z should be ~0, got {}",
        com.z()
    );
}

#[test]
fn make_sphere_two_hemispheres() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 1.0, 8).unwrap();
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert_eq!(sh.faces().len(), 2, "sphere should have 2 hemisphere faces");
}

#[test]
fn make_sphere_zero_radius_error() {
    let mut topo = Topology::new();
    assert!(make_sphere(&mut topo, 0.0, 8).is_err());
}

#[test]
fn make_sphere_few_segments_error() {
    let mut topo = Topology::new();
    assert!(make_sphere(&mut topo, 1.0, 2).is_err());
}

#[test]
fn make_torus_basic() {
    let mut topo = Topology::new();
    let solid = make_torus(&mut topo, 3.0, 1.0, 8).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(!sh.faces().is_empty(), "torus should have faces");

    assert_eq!(
        euler_characteristic(&topo, solid),
        0,
        "torus should be genus-1 (χ=0)"
    );
}

#[test]
fn make_torus_volume() {
    let mut topo = Topology::new();
    let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
    // V = 2π²Rr² where R=major, r=minor
    let expected = 2.0 * PI * PI * 3.0 * 1.0;
    // Polygonal approximation — allow 5% tolerance
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "torus volume should be ~{expected}, got {vol} (error: {:.1}%)",
        (vol - expected).abs() / expected * 100.0
    );
}

#[test]
fn make_torus_self_intersecting_error() {
    let mut topo = Topology::new();
    // minor >= major → self-intersecting
    assert!(make_torus(&mut topo, 1.0, 1.0, 8).is_err());
    assert!(make_torus(&mut topo, 1.0, 2.0, 8).is_err());
}

#[test]
fn make_torus_zero_radius_error() {
    let mut topo = Topology::new();
    assert!(make_torus(&mut topo, 0.0, 1.0, 8).is_err());
    assert!(make_torus(&mut topo, 3.0, 0.0, 8).is_err());
}

#[test]
fn cylinder_seam_continuity() {
    // Cylinder surface should evaluate to the same point at u=0 and u=2π
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

    for fid in &faces {
        let face = topo.face(*fid).unwrap();
        if let brepkit_topology::face::FaceSurface::Cylinder(cyl) = face.surface() {
            let p0 = cyl.evaluate(0.0, 0.5);
            let p2pi = cyl.evaluate(std::f64::consts::TAU, 0.5);
            let dist = ((p0.x() - p2pi.x()).powi(2)
                + (p0.y() - p2pi.y()).powi(2)
                + (p0.z() - p2pi.z()).powi(2))
            .sqrt();
            assert!(
                dist < 1e-10,
                "cylinder seam gap: u=0 and u=2π should coincide, dist={dist}"
            );
        }
    }
}

#[test]
fn sphere_pole_degenerate() {
    // Sphere should have degenerate edges at poles (start == end vertex)
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 1.0, 8).unwrap();

    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
    // Sphere has 2 hemisphere faces
    assert_eq!(faces.len(), 2, "sphere should have 2 faces");
}

#[test]
fn torus_seam_u_and_v() {
    // Torus surface should be periodic in both u and v
    let mut topo = Topology::new();
    let solid = make_torus(&mut topo, 3.0, 1.0, 8).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

    for fid in &faces {
        let face = topo.face(*fid).unwrap();
        if let brepkit_topology::face::FaceSurface::Torus(tor) = face.surface() {
            // Check u-periodicity
            let p00 = tor.evaluate(0.0, 0.5);
            let p2pi_0 = tor.evaluate(std::f64::consts::TAU, 0.5);
            let dist_u = ((p00.x() - p2pi_0.x()).powi(2)
                + (p00.y() - p2pi_0.y()).powi(2)
                + (p00.z() - p2pi_0.z()).powi(2))
            .sqrt();
            assert!(dist_u < 1e-10, "torus u-seam gap: dist={dist_u}");

            // Check v-periodicity
            let p0_0 = tor.evaluate(0.5, 0.0);
            let p0_2pi = tor.evaluate(0.5, std::f64::consts::TAU);
            let dist_v = ((p0_0.x() - p0_2pi.x()).powi(2)
                + (p0_0.y() - p0_2pi.y()).powi(2)
                + (p0_0.z() - p0_2pi.z()).powi(2))
            .sqrt();
            assert!(dist_v < 1e-10, "torus v-seam gap: dist={dist_v}");
        }
    }
}

#[test]
fn cone_apex_singular() {
    // A cone with top_radius=0 has an apex vertex
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();
    let verts = brepkit_topology::explorer::solid_vertices(&topo, solid).unwrap();

    // The apex should be at (0, 0, 3.0) — the top of the cone
    let has_apex = verts.iter().any(|vid| {
        let v = topo.vertex(*vid).unwrap();
        let p = v.point();
        (p.x().powi(2) + p.y().powi(2)).sqrt() < 1e-10 && (p.z() - 3.0).abs() < 1e-10
    });
    assert!(has_apex, "cone should have apex vertex at (0,0,3)");
}

#[test]
fn make_box_very_thin() {
    // Thin plate: dx=1e-4, dy=1, dz=1
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 1e-4, 1.0, 1.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 1e-4).abs() < 1e-12,
        "thin box volume should be 1e-4, got {vol}"
    );
    assert_euler_genus0(&topo, solid);
}

#[test]
fn make_box_very_large() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 1e6, 1.0, 1.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 1e6).abs() / 1e6 < 1e-10,
        "large box volume should be 1e6, got {vol}"
    );
}

#[test]
fn make_cylinder_high_aspect_ratio() {
    // Very thin, very tall cylinder
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 0.001, 1000.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    let expected = PI * 0.001_f64.powi(2) * 1000.0;
    let rel_error = (vol - expected).abs() / expected;
    assert!(
        rel_error < 0.01,
        "high aspect cylinder volume: got {vol:.6e}, expected {expected:.6e} (error: {:.1}%)",
        rel_error * 100.0
    );
}

#[test]
fn convex_hull_unit_cube() {
    let mut topo = Topology::new();
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    let solid = make_convex_hull(&mut topo, &points).unwrap();
    assert_volume_near(&topo, solid, 1.0, 1e-10);
}

#[test]
fn convex_hull_larger_box() {
    let mut topo = Topology::new();
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(3.0, 0.0, 0.0),
        Point3::new(0.0, 4.0, 0.0),
        Point3::new(3.0, 4.0, 0.0),
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(3.0, 0.0, 5.0),
        Point3::new(0.0, 4.0, 5.0),
        Point3::new(3.0, 4.0, 5.0),
    ];
    let solid = make_convex_hull(&mut topo, &points).unwrap();
    assert_volume_near(&topo, solid, 60.0, 1e-10);
}

#[test]
fn convex_hull_minkowski_sum_volume() {
    // Minkowski sum of [0,1]^3 with [0,1]^3 = [0,2]^3, volume = 8.
    let cube: Vec<Point3> = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];

    let mut sum_points = Vec::with_capacity(64);
    for &a in &cube {
        for &b in &cube {
            sum_points.push(Point3::new(a.x() + b.x(), a.y() + b.y(), a.z() + b.z()));
        }
    }

    let mut topo = Topology::new();
    let solid = make_convex_hull(&mut topo, &sum_points).unwrap();
    assert_volume_near(&topo, solid, 8.0, 1e-8);
}

#[test]
fn convex_hull_minkowski_displaced_cubes_volume() {
    // Cube A at origin [0,1]^3, Cube B at (5,5,5) to (6,6,6).
    // Hull of pairwise sums spans [5,7]^3, volume = 8.
    let cube_a = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    let cube_b = vec![
        Point3::new(5.0, 5.0, 5.0),
        Point3::new(6.0, 5.0, 5.0),
        Point3::new(5.0, 6.0, 5.0),
        Point3::new(6.0, 6.0, 5.0),
        Point3::new(5.0, 5.0, 6.0),
        Point3::new(6.0, 5.0, 6.0),
        Point3::new(5.0, 6.0, 6.0),
        Point3::new(6.0, 6.0, 6.0),
    ];

    let mut sum_points = Vec::with_capacity(64);
    for &a in &cube_a {
        for &b in &cube_b {
            sum_points.push(Point3::new(a.x() + b.x(), a.y() + b.y(), a.z() + b.z()));
        }
    }

    let mut topo = Topology::new();
    let solid = make_convex_hull(&mut topo, &sum_points).unwrap();
    assert_volume_near(&topo, solid, 8.0, 1e-8);
}

#[test]
fn minkowski_sum_unit_boxes_volume_8() {
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let sum = make_minkowski_sum(&mut topo, a, b).unwrap();
    // Two unit boxes sum to a 2×2×2 box (volume 8), independent of where
    // make_box places its origin.
    assert_volume_near(&topo, sum, 8.0, 1e-6);
}

#[test]
fn minkowski_sum_box10_box2_volume_1728() {
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let sum = make_minkowski_sum(&mut topo, a, b).unwrap();
    // 10-box ⊕ 2-box = 12-box, volume 1728.
    assert_volume_near(&topo, sum, 1728.0, 1e-4);
}

#[test]
fn convex_hull_with_interior_points_volume() {
    // 8 cube corners + interior points — volume should be 1.0.
    let mut points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    points.push(Point3::new(0.5, 0.5, 0.5));
    points.push(Point3::new(0.2, 0.3, 0.7));

    let mut topo = Topology::new();
    let solid = make_convex_hull(&mut topo, &points).unwrap();
    assert_volume_near(&topo, solid, 1.0, 1e-10);
}

#[test]
fn convex_hull_rejects_coplanar_points() {
    let mut topo = Topology::new();
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ];
    assert!(make_convex_hull(&mut topo, &points).is_err());
}

#[test]
fn convex_hull_rejects_too_few_points() {
    let mut topo = Topology::new();
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    assert!(make_convex_hull(&mut topo, &points).is_err());
}
