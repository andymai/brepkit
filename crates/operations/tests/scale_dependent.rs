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
    validate_shell_manifold(sh, &topo.faces, &topo.wires).expect("result should be manifold");
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
