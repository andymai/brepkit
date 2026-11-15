//! Scale-dependent tests.
//!
//! Verify that boolean operations, tessellation, and volume measurement
//! work correctly across geometric scales from micrometers to kilometers,
//! catching any hardcoded epsilon values.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_manifold;

fn vol(topo: &Topology, solid: SolidId, deflection: f64) -> f64 {
    solid_volume(topo, solid, deflection).unwrap()
}

fn check_manifold(topo: &Topology, solid: SolidId) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    validate_shell_manifold(sh, topo).expect("result should be manifold");
}

/// Fuse two overlapping boxes of size `s` with `s/2` overlap along X.
/// Returns the fused solid and the expected volume (1.5 * s^3).
fn fuse_overlapping_boxes(topo: &mut Topology, s: f64) -> (SolidId, f64) {
    let a = make_box(topo, s, s, s).unwrap();
    let b = make_box(topo, s, s, s).unwrap();
    transform_solid(topo, b, &Mat4::translation(s / 2.0, 0.0, 0.0)).unwrap();
    let fused = boolean(topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = 1.5 * s * s * s;
    (fused, expected)
}

// ── Millimeter scale ────────────────────────────────────────────────

#[test]
fn test_boolean_at_mm_scale() {
    let mut topo = Topology::new();
    let s = 0.1; // 100 um boxes
    let (fused, expected) = fuse_overlapping_boxes(&mut topo, s);
    let v = vol(&topo, fused, s * 0.01);
    let rel = (v - expected).abs() / expected;
    assert!(
        rel < 0.01,
        "mm-scale: volume {v:.8} not within 1% of expected {expected:.8} (error {:.2}%)",
        rel * 100.0
    );
    check_manifold(&topo, fused);
}

// ── Meter scale ─────────────────────────────────────────────────────

#[test]
fn test_boolean_at_m_scale() {
    let mut topo = Topology::new();
    let s = 10.0;
    let (fused, expected) = fuse_overlapping_boxes(&mut topo, s);
    let v = vol(&topo, fused, s * 0.01);
    let rel = (v - expected).abs() / expected;
    assert!(
        rel < 0.01,
        "m-scale: volume {v:.4} not within 1% of expected {expected:.4} (error {:.2}%)",
        rel * 100.0
    );
    check_manifold(&topo, fused);
}

// ── Kilometer scale ─────────────────────────────────────────────────

#[test]
fn test_boolean_at_km_scale() {
    let mut topo = Topology::new();
    let s = 1000.0;
    let (fused, expected) = fuse_overlapping_boxes(&mut topo, s);
    let v = vol(&topo, fused, s * 0.01);
    let rel = (v - expected).abs() / expected;
    assert!(
        rel < 0.01,
        "km-scale: volume {v:.2} not within 1% of expected {expected:.2} (error {:.2}%)",
        rel * 100.0
    );
    check_manifold(&topo, fused);
}

// ── Micrometer tessellation ─────────────────────────────────────────

#[test]
fn test_tessellation_at_micro_scale() {
    let mut topo = Topology::new();
    let radius = 0.001;
    let height = 0.01;
    let cyl = make_cylinder(&mut topo, radius, height).unwrap();
    let mesh = tessellate_solid(&topo, cyl, 0.0001).unwrap();

    let num_triangles = mesh.indices.len() / 3;
    assert!(
        num_triangles > 0,
        "micro-scale cylinder produced 0 triangles"
    );

    // Verify all triangles have positive area.
    for tri in mesh.indices.chunks_exact(3) {
        let p0 = mesh.positions[tri[0] as usize];
        let p1 = mesh.positions[tri[1] as usize];
        let p2 = mesh.positions[tri[2] as usize];
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let area = e1.cross(e2).length() * 0.5;
        assert!(
            area > 0.0,
            "degenerate triangle at micro scale: area={area}"
        );
    }
}

// ── Cross-scale tolerance consistency ───────────────────────────────

#[test]
fn test_tolerance_scaling() {
    let scales = [0.01, 1.0, 1000.0];

    for &s in &scales {
        let mut topo = Topology::new();
        let (fused, expected) = fuse_overlapping_boxes(&mut topo, s);
        let v = vol(&topo, fused, s * 0.01);
        let rel = (v - expected).abs() / expected;
        assert!(
            rel < 0.01,
            "scale {s}: volume {v} not within 1% of expected {expected} (error {:.2}%)",
            rel * 100.0
        );
        check_manifold(&topo, fused);
    }
}

// ── Small uniform scale ─────────────────────────────────────────────

/// A uniform scale by `s` has the determinant `s³`, which is below a length
/// tolerance from `s = 0.004` down. Each such scale has an inverse.
#[test]
fn a_small_uniform_scale_keeps_the_solid() {
    use brepkit_operations::copy::copy_and_transform_solid;
    use brepkit_operations::primitives::make_sphere;

    for s in [0.005, 0.002, 0.0011, 0.001, 1e-4, 1e-5, 1e-6, 1e-7] {
        let m = Mat4::scale(s, s, s);
        let cube = s * s * s;

        let mut topo = Topology::new();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let c = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let sp = make_sphere(&mut topo, 4.0, 16).unwrap();
        let truths = [
            (b, 1000.0),
            (c, std::f64::consts::PI * 90.0),
            (sp, std::f64::consts::PI * 256.0 / 3.0),
        ];
        for (solid, truth) in truths {
            let copy = copy_and_transform_solid(&mut topo, solid, &m);
            assert!(copy.is_ok(), "copy at scale {s}: {copy:?}");
            let copy = copy.unwrap();
            let moved = transform_solid(&mut topo, solid, &m);
            assert!(moved.is_ok(), "transform at scale {s}: {moved:?}");
            for scaled in [solid, copy] {
                let v = vol(&topo, scaled, s * 0.01);
                let rel = (v - truth * cube).abs() / (truth * cube);
                assert!(
                    rel < 1e-9,
                    "scale {s}: volume {v}, expected {}, relative error {rel:e}",
                    truth * cube
                );
                check_manifold(&topo, scaled);
            }
        }
    }
}

#[test]
fn a_matrix_with_no_inverse_is_refused() {
    use brepkit_operations::copy::copy_and_transform_solid;

    let flat = Mat4::scale(1.0, 1.0, 0.0);
    let tiny_flat = Mat4::scale(1e-7, 1e-7, 0.0);
    let tinier_flat = Mat4::scale(1e-80, 1e-80, 0.0);
    let not_finite = Mat4::scale(f64::NAN, 1.0, 1.0);
    for m in [flat, tiny_flat, tinier_flat, not_finite] {
        let mut topo = Topology::new();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        assert!(copy_and_transform_solid(&mut topo, b, &m).is_err());
        assert!(transform_solid(&mut topo, b, &m).is_err());
        let face = brepkit_topology::explorer::solid_faces(&topo, b).unwrap()[0];
        let wire = topo.face(face).unwrap().outer_wire();
        assert!(brepkit_operations::transform::transform_face(&mut topo, face, &m).is_err());
        assert!(brepkit_operations::transform::transform_wire(&mut topo, wire, &m).is_err());
        assert!((vol(&topo, b, 0.1) - 1000.0).abs() < 1e-9);
    }
}

/// A scale about a centre far from the origin has a large translation,
/// which must not change whether the matrix has an inverse.
#[test]
fn a_small_scale_about_a_far_centre_keeps_the_solid() {
    use brepkit_operations::copy::copy_and_transform_solid;

    for s in [0.002, 1e-7] {
        let centre = 100_000.0;
        let m = Mat4::translation(centre, 0.0, 0.0)
            * Mat4::scale(s, s, s)
            * Mat4::translation(-centre, 0.0, 0.0);
        let mut topo = Topology::new();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, b, &Mat4::translation(centre, 0.0, 0.0)).unwrap();
        let copy = copy_and_transform_solid(&mut topo, b, &m);
        assert!(copy.is_ok(), "copy at scale {s}: {copy:?}");
        let moved = transform_solid(&mut topo, b, &m);
        assert!(moved.is_ok(), "transform at scale {s}: {moved:?}");
        for scaled in [b, copy.unwrap()] {
            let v = vol(&topo, scaled, s * 0.01);
            let truth = 1000.0 * s * s * s;
            // The corners sit at 1e5, where a coordinate holds steps of
            // about 1e-11: five digits of a box edge of 1e-6.
            assert!(
                (v - truth).abs() / truth < 1e-3,
                "scale {s}: volume {v}, expected {truth}"
            );
            check_manifold(&topo, scaled);
        }
    }
}
