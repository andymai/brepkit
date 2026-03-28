//! Integration tests for the offset pipeline.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_offset::{OffsetOptions, offset_solid};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_topology::Topology;

fn offset_opts() -> OffsetOptions {
    OffsetOptions {
        remove_self_intersections: false,
        ..Default::default()
    }
}

#[test]
fn offset_box_outward_face_count() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let result = offset_solid(&mut topo, solid, 0.5, offset_opts()).unwrap();
    let shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert_eq!(shell.faces().len(), 6, "offset box should have 6 faces");
}

#[test]
fn offset_box_inward_face_count() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let result = offset_solid(&mut topo, solid, -0.5, offset_opts()).unwrap();
    let shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert_eq!(shell.faces().len(), 6);
}

#[test]
fn offset_rectangular_box() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 3.0, 5.0, 7.0).unwrap();
    let result = offset_solid(&mut topo, solid, 1.0, offset_opts()).unwrap();
    let shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert_eq!(shell.faces().len(), 6);
}

// Volume tests — for planar offsets, Phase 3 computes exact intersection
// line endpoints using the offset distance as margin (no percentage-based
// approximation), and the wire builder clips edges at exact line-line
// intersection corners.  Corner vertices use the first line's point
// directly (no midpoint averaging) since coplanar lines intersect exactly.
// This gives exact volumes for box offsets.

#[test]
fn offset_box_outward_exact_volume() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let result = offset_solid(&mut topo, solid, 0.5, offset_opts()).unwrap();
    let vol = solid_volume(&topo, result, 0.01).unwrap();
    let expected = 27.0_f64; // (2+1)^3
    let error_pct = ((vol - expected) / expected * 100.0).abs();
    assert!(
        error_pct < 0.01,
        "outward offset volume {vol:.10} should be {expected}, error {error_pct:.6}%"
    );
}

#[test]
fn offset_box_inward_exact_volume() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let result = offset_solid(&mut topo, solid, -0.5, offset_opts()).unwrap();
    let vol = solid_volume(&topo, result, 0.01).unwrap();
    let expected = 27.0_f64; // (4-1)^3
    let error_pct = ((vol - expected) / expected * 100.0).abs();
    assert!(
        error_pct < 0.01,
        "inward offset volume {vol:.10} should be {expected}, error {error_pct:.6}%"
    );
}

#[test]
fn offset_rectangular_box_exact_volume() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 3.0, 5.0, 7.0).unwrap();
    let result = offset_solid(&mut topo, solid, 1.0, offset_opts()).unwrap();
    let vol = solid_volume(&topo, result, 0.01).unwrap();
    let expected = 5.0 * 7.0 * 9.0; // 315
    let error_pct = ((vol - expected) / expected * 100.0).abs();
    assert!(
        error_pct < 0.01,
        "rectangular offset volume {vol:.10} should be {expected}, error {error_pct:.6}%"
    );
}

#[test]
fn offset_rectangular_box_inward_exact_volume() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 3.0, 5.0, 7.0).unwrap();
    let result = offset_solid(&mut topo, solid, -0.5, offset_opts()).unwrap();
    let vol = solid_volume(&topo, result, 0.01).unwrap();
    let expected = 2.0 * 4.0 * 6.0; // 48
    let error_pct = ((vol - expected) / expected * 100.0).abs();
    assert!(
        error_pct < 0.01,
        "inward rectangular offset volume {vol:.10} should be {expected}, error {error_pct:.6}%"
    );
}

#[test]
fn offset_box_outward_volume_larger_than_original() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let original_vol = solid_volume(&topo, solid, 0.1).unwrap();

    let result = offset_solid(&mut topo, solid, 0.5, offset_opts()).unwrap();
    let offset_vol = solid_volume(&topo, result, 0.1).unwrap();
    assert!(
        offset_vol > original_vol,
        "outward offset volume ({offset_vol}) should exceed original ({original_vol})"
    );
}

#[test]
fn offset_box_inward_volume_smaller_than_original() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let original_vol = solid_volume(&topo, solid, 0.1).unwrap();

    let result = offset_solid(&mut topo, solid, -0.5, offset_opts()).unwrap();
    let offset_vol = solid_volume(&topo, result, 0.1).unwrap();
    assert!(
        offset_vol < original_vol,
        "inward offset volume ({offset_vol}) should be less than original ({original_vol})"
    );
}

// ── Cylinder offset tests ──────────────────────────────────────

#[test]
fn offset_cylinder_outward_produces_solid() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();
    let result = offset_solid(&mut topo, solid, 0.5, offset_opts()).unwrap();
    let shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert!(
        shell.faces().len() >= 3,
        "offset cylinder should have at least 3 faces, got {}",
        shell.faces().len()
    );
}

#[test]
fn offset_cylinder_volume_increases() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();
    let original_vol = solid_volume(&topo, solid, 0.1).unwrap();

    if let Ok(result) = offset_solid(&mut topo, solid, 0.5, offset_opts()) {
        let offset_vol = solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            offset_vol > original_vol,
            "outward offset volume ({offset_vol}) should exceed original ({original_vol})"
        );
    }
}

// ── Thick solid (shell) tests ─────────────────────────────────

#[test]
fn thick_solid_box_produces_hollow() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();
    let exclude = vec![faces[0]];

    let result =
        brepkit_offset::thick_solid(&mut topo, solid, -0.2, &exclude, offset_opts()).unwrap();

    let result_shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert!(
        result_shell.faces().len() >= 9,
        "thick solid should have >= 9 faces, got {}",
        result_shell.faces().len()
    );

    let vol = solid_volume(&topo, result, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "thick solid should have positive volume, got {vol}"
    );
}

// ── Sphere offset tests ────────────────────────────────────────

#[test]
fn offset_sphere_outward_produces_solid() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 3.0, 16_usize).unwrap();
    let result = offset_solid(&mut topo, solid, 0.5, offset_opts()).unwrap();
    let shell = topo
        .shell(topo.solid(result).unwrap().outer_shell())
        .unwrap();
    assert!(!shell.faces().is_empty(), "offset sphere should have faces");
}
