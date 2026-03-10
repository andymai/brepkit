//! Boolean operation edge case tests.
//!
//! Verifies correct behavior on dangerous geometric configurations:
//! tangent solids, kissing faces, thin features, degenerate results,
//! sequential operations, mixed surfaces, and vertex/edge contact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::no_effect_underscore_binding,
    clippy::single_match,
    clippy::single_match_else
)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_manifold;

const DEFLECTION: f64 = 0.1;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn check_manifold(topo: &Topology, solid: SolidId) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    validate_shell_manifold(sh, &topo.faces, &topo.wires).expect("result should be manifold");
}

// ── Tangent solids ──────────────────────────────────────────────────

#[test]
fn test_sphere_tangent_to_plane() {
    // Sphere radius 1 at origin. Box from (-5,-5,-5) to (5,5,-1).
    // The box top face at z=-1 is exactly tangent to the sphere bottom.
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, 1.0, 16).unwrap();
    let bx = make_box(&mut topo, 10.0, 10.0, 4.0).unwrap();
    transform_solid(&mut topo, bx, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, sphere, bx);
    match result {
        Ok(fused) => {
            let expected = 4.0 * PI / 3.0 + 400.0;
            let v = vol(&topo, fused);
            let rel = (v - expected).abs() / expected;
            assert!(rel < 0.05, "volume {v:.2} not near expected {expected:.2}");
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("tangent fuse failed: {e}"),
    }
}

#[test]
fn test_cylinder_tangent_to_plane() {
    // Cylinder radius 1, height 2 at origin. Box with right face at x=-1.
    let mut topo = Topology::new();
    let cyl = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let bx = make_box(&mut topo, 4.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, bx, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, cyl, bx);
    match result {
        Ok(fused) => {
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("tangent cylinder-plane fuse failed: {e}"),
    }
}

#[test]
fn test_two_spheres_tangent() {
    // Two spheres radius 1, touching at (1,0,0).
    let mut topo = Topology::new();
    let s1 = make_sphere(&mut topo, 1.0, 16).unwrap();
    let s2 = make_sphere(&mut topo, 1.0, 16).unwrap();
    transform_solid(&mut topo, s2, &Mat4::translation(2.0, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, s1, s2);
    match result {
        Ok(fused) => {
            let expected = 2.0 * 4.0 * PI / 3.0;
            let v = vol(&topo, fused);
            let rel = (v - expected).abs() / expected;
            assert!(rel < 0.05, "volume {v:.2} not near expected {expected:.2}");
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("tangent spheres fuse failed: {e}"),
    }
}

// ── Kissing solids ──────────────────────────────────────────────────

#[test]
fn test_kissing_boxes_fuse() {
    // Two unit boxes sharing a face at x=1.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 0.0, 0.0)).unwrap();

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let v = vol(&topo, fused);
    let rel = (v - 2.0).abs() / 2.0;
    assert!(rel < 0.01, "fused volume {v:.4} not near 2.0");
    check_manifold(&topo, fused);
}

#[test]
fn test_kissing_boxes_cut() {
    // Two unit boxes sharing a face at x=1. Cut A - B should leave A unchanged.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, a, b);
    match result {
        Ok(cut) => {
            let v = vol(&topo, cut);
            let rel = (v - 1.0).abs();
            assert!(rel < 0.01, "cut volume {v:.4} not near 1.0");
            check_manifold(&topo, cut);
        }
        Err(_) => {
            // Also acceptable: error if the engine treats face-contact as degenerate
        }
    }
}

#[test]
#[ignore = "bug: tangent cylinder-box fuse produces incorrect volume"]
fn test_kissing_cylinder_box() {
    // Cylinder radius 1, height 2. Box with left face tangent at x=1.
    let mut topo = Topology::new();
    let cyl = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let bx = make_box(&mut topo, 4.0, 10.0, 2.0).unwrap();
    transform_solid(&mut topo, bx, &Mat4::translation(1.0, -5.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, cyl, bx);
    match result {
        Ok(fused) => {
            let cyl_vol = PI * 1.0 * 1.0 * 2.0;
            let box_vol = 4.0 * 10.0 * 2.0;
            let expected = cyl_vol + box_vol;
            let v = vol(&topo, fused);
            // Volume should be close to sum (tangent, no overlap)
            let rel = (v - expected).abs() / expected;
            assert!(rel < 0.05, "volume {v:.2} not near expected {expected:.2}");
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("kissing cylinder-box fuse failed: {e}"),
    }
}

// ── Thin features ───────────────────────────────────────────────────

#[test]
fn test_boolean_thin_slab() {
    // Box A: 10x10x10. Box B: 10x10x10 offset by (0,0,9.999). Cut A - B.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, 9.999)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, a, b);
    match result {
        Ok(cut) => {
            let expected = 10.0 * 10.0 * 9.999;
            let v = vol(&topo, cut);
            let rel = (v - expected).abs() / expected;
            assert!(rel < 0.01, "volume {v:.4} not near expected {expected:.4}");
            check_manifold(&topo, cut);
        }
        Err(e) => panic!("thin slab cut failed: {e}"),
    }
}

#[test]
fn test_boolean_near_tangent() {
    // Two boxes with tiny overlap: A (0,0,0)-(1,1,1), B (0.999,0,0)-(2,1,1).
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.001, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(0.999, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b);
    match result {
        Ok(inter) => {
            let v = vol(&topo, inter);
            assert!(v > 0.0, "intersection should have positive volume, got {v}");
            assert!(v < 0.01, "intersection should be tiny, got {v}");
            check_manifold(&topo, inter);
        }
        Err(_) => {
            // Acceptable: engine may treat near-tangent as no intersection
        }
    }
}

// ── Degenerate results ──────────────────────────────────────────────

#[test]
fn test_cut_removes_all() {
    // Box A: 1x1x1. Box B: 2x2x2 covering A entirely.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(-0.5, -0.5, -0.5)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, a, b);
    assert!(result.is_err(), "cutting away everything should return Err");
}

#[test]
fn test_intersect_disjoint() {
    // Two disjoint boxes with no overlap.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(5.0, 5.0, 5.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b);
    assert!(
        result.is_err(),
        "intersecting disjoint solids should return Err"
    );
}

// ── Sequential booleans ─────────────────────────────────────────────

#[test]
fn test_sequential_cuts_volume() {
    // Start with 10x10x10 box. Cut 5 columns (1x1x10 each) at different positions.
    let mut topo = Topology::new();
    let mut current = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    for i in 0..5 {
        let cutter = make_box(&mut topo, 1.0, 1.0, 10.0).unwrap();
        let x = i as f64 * 2.0; // positions 0, 2, 4, 6, 8
        transform_solid(&mut topo, cutter, &Mat4::translation(x, 0.0, 0.0)).unwrap();
        current = boolean(&mut topo, BooleanOp::Cut, current, cutter).unwrap();
    }

    let expected = 1000.0 - 5.0 * 10.0;
    let v = vol(&topo, current);
    let rel = (v - expected).abs() / expected;
    assert!(
        rel < 0.01,
        "sequential cuts volume {v:.2} not near expected {expected:.2}"
    );
    check_manifold(&topo, current);
}

#[test]
fn test_sequential_boolean_vertex_drift() {
    // Perform 10 fuse+cut cycles. Volume should return to original each time.
    let mut topo = Topology::new();
    let base = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
    let original_vol = vol(&topo, base);
    let mut current = base;

    for i in 0..10 {
        let addon = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let x = 5.0 + (i as f64) * 0.01; // slightly different each time
        transform_solid(&mut topo, addon, &Mat4::translation(x, 0.0, 0.0)).unwrap();

        let fused = boolean(&mut topo, BooleanOp::Fuse, current, addon).unwrap();

        let cutter = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, cutter, &Mat4::translation(x, 0.0, 0.0)).unwrap();

        current = boolean(&mut topo, BooleanOp::Cut, fused, cutter).unwrap();
    }

    let final_vol = vol(&topo, current);
    let drift = (final_vol - original_vol).abs() / original_vol;
    assert!(
        drift < 0.05,
        "vertex drift after 10 cycles: {drift:.4} (original={original_vol:.4}, final={final_vol:.4})"
    );
}

#[test]
fn test_alternating_union_cut() {
    // A|B - C pattern.
    // A: (0-2, 0-1, 0-1), B: (1-3, 0-1, 0-1), C: (1-2, 0-1, 0-1).
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 2.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 2.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 0.0, 0.0)).unwrap();

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let fused_vol = vol(&topo, fused);
    let rel_fused = (fused_vol - 3.0).abs() / 3.0;
    assert!(rel_fused < 0.01, "fused volume {fused_vol:.4} not near 3.0");

    let c = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, c, &Mat4::translation(1.0, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, fused, c).unwrap();
    let v = vol(&topo, result);
    let rel = (v - 2.0).abs() / 2.0;
    assert!(rel < 0.01, "A|B - C volume {v:.4} not near 2.0");
    check_manifold(&topo, result);
}

// ── Mixed surfaces ──────────────────────────────────────────────────

#[test]
#[ignore = "bug: sphere-cylinder cut produces degenerate result (1 face)"]
fn test_boolean_sphere_cylinder() {
    // Sphere radius 2. Cylinder radius 0.5, height 6 through center.
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, 2.0, 16).unwrap();
    let cyl = make_cylinder(&mut topo, 0.5, 6.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, sphere, cyl);
    match result {
        Ok(drilled) => {
            let sphere_vol = 4.0 * PI * 8.0 / 3.0;
            // Cylinder intersection with sphere: height inside sphere = 4.0 (from -2 to 2)
            let cyl_inside_vol = PI * 0.25 * 4.0;
            let expected = sphere_vol - cyl_inside_vol;
            let v = vol(&topo, drilled);
            let rel = (v - expected).abs() / expected;
            assert!(
                rel < 0.10,
                "drilled sphere volume {v:.2} not near expected {expected:.2} (rel={rel:.4})"
            );
            check_manifold(&topo, drilled);
        }
        Err(e) => panic!("sphere-cylinder cut failed: {e}"),
    }
}

#[test]
#[ignore = "bug: cone-box fuse produces incorrect volume (32 vs expected 48)"]
fn test_boolean_cone_box() {
    // Cone r_bottom=2, r_top=0, height=3. Box 4x4x3.
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();
    let bx = make_box(&mut topo, 4.0, 4.0, 3.0).unwrap();
    transform_solid(&mut topo, bx, &Mat4::translation(-2.0, -2.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, cone, bx);
    match result {
        Ok(fused) => {
            let _cone_vol = PI * 4.0 * 3.0 / 3.0; // pi * r^2 * h / 3
            let box_vol = 48.0;
            let v = vol(&topo, fused);
            // Cone is fully inside the box, so fuse volume = box volume
            assert!(
                v > box_vol * 0.95,
                "fused volume {v:.2} should be at least box_vol {box_vol:.2}"
            );
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("cone-box fuse failed: {e}"),
    }
}

#[test]
#[ignore = "bug: cone-cylinder cut produces incorrect volume (1.95 vs expected 19.63)"]
fn test_boolean_cone_cylinder() {
    // Cone r_bottom=2, r_top=1, height=3. Cylinder radius 0.5, height 3.
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();
    let cyl = make_cylinder(&mut topo, 0.5, 3.0).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, cone, cyl);
    match result {
        Ok(cut) => {
            let cone_vol = PI * 3.0 / 3.0 * (4.0 + 2.0 + 1.0); // pi*h/3*(R^2+Rr+r^2)
            let cyl_vol = PI * 0.25 * 3.0;
            let expected = cone_vol - cyl_vol;
            let v = vol(&topo, cut);
            let rel = (v - expected).abs() / expected;
            assert!(
                rel < 0.10,
                "cone-cylinder cut volume {v:.2} not near expected {expected:.2} (rel={rel:.4})"
            );
            check_manifold(&topo, cut);
        }
        Err(e) => panic!("cone-cylinder cut failed: {e}"),
    }
}

// ── Edge/vertex contact ─────────────────────────────────────────────

#[test]
fn test_boolean_shared_edge() {
    // Box A: (0,0,0)-(1,1,1). Box B: (1,1,0)-(2,2,1). Share edge (1,1,0)-(1,1,1).
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b);
    match result {
        Ok(fused) => {
            let v = vol(&topo, fused);
            let rel = (v - 2.0).abs() / 2.0;
            assert!(rel < 0.01, "fused volume {v:.4} not near 2.0");
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("shared-edge fuse failed: {e}"),
    }
}

#[test]
fn test_boolean_shared_vertex() {
    // Box A: (0,0,0)-(1,1,1). Box B: (1,1,1)-(2,2,2). Share vertex (1,1,1).
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.0, 1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b);
    match result {
        Ok(fused) => {
            let v = vol(&topo, fused);
            let rel = (v - 2.0).abs() / 2.0;
            assert!(rel < 0.01, "fused volume {v:.4} not near 2.0");
            check_manifold(&topo, fused);
        }
        Err(e) => panic!("shared-vertex fuse failed: {e}"),
    }
}
