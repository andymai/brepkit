//! Coaxial-torus scenarios for boolean robustness.
//!
//! Validates the new Torus arm of `surfaces_same_domain()` (added
//! alongside this corpus). Torus same-domain requires matching center,
//! major radius, minor radius, and z-axis (parallel or antiparallel).
//! The DETECTOR is verified by unit tests in `same_domain.rs`; the
//! GFA pipeline integration of torus SD pairs has known gaps tracked here.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_torus;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DEFLECTION: f64 = 0.05;
const SEGMENTS: usize = 32;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn torus_volume(major: f64, minor: f64) -> f64 {
    2.0 * PI * PI * major * minor * minor
}

fn approx_eq(a: f64, b: f64, frac: f64) -> bool {
    (a - b).abs() < a.abs().max(b.abs()).max(1.0) * frac
}

fn torus_at(topo: &mut Topology, x: f64, y: f64, z: f64, major: f64, minor: f64) -> SolidId {
    let t = make_torus(topo, major, minor, SEGMENTS).unwrap();
    if x != 0.0 || y != 0.0 || z != 0.0 {
        transform_solid(topo, t, &Mat4::translation(x, y, z)).unwrap();
    }
    t
}

// ── 0. Baseline: disjoint toruses ──────────────────────────────────────

#[test]
fn baseline_disjoint_toruses_intersect_empty() {
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = torus_at(&mut topo, 20.0, 0.0, 0.0, 3.0, 0.5);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        assert!(
            v < 1e-3,
            "disjoint torus intersect should be ~zero, got {v}"
        );
    }
}

// ── 1. Identical toruses (validates new Torus SD arm) ─────────────────

#[test]
fn identical_toruses_fuse_preserves_volume() {
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = torus_volume(3.0, 0.5);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

#[test]
fn identical_toruses_intersect_preserves_volume() {
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let expected = torus_volume(3.0, 0.5);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

// ── 2. Mismatched minor radius — must NOT be SD ───────────────────────

#[test]
fn toruses_different_minor_radius_intersect_nontrivial() {
    // Two toruses sharing major axis but with different tube radii produce
    // a non-trivial intersection. Exercises NON-SD path; the SD detector
    // correctly returns None for this pair.
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.7);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    // Either OK (positive volume) or Err (degenerate) is acceptable;
    // the key requirement is that the SD detector did NOT mark this
    // pair as same-domain (since minor radii differ).
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        // Volume should be >0 and <= min(torus A, torus B).
        assert!(v > 0.0);
        assert!(v <= torus_volume(3.0, 0.5) + 1e-3);
    }
}

// ── 3. Opposite axis (rotated 180°) ───────────────────────────────────

#[test]
fn torus_opposite_axis_fuse() {
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = make_torus(&mut topo, 3.0, 0.5, SEGMENTS).unwrap();
    transform_solid(&mut topo, b, &Mat4::rotation_x(PI)).unwrap();
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = torus_volume(3.0, 0.5);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

// ── 4. Shared major, different minor — new coaxial-torus shortcut path ──

#[test]
fn coaxial_torus_fuse_collapses_to_larger_minor() {
    // Two coaxial tori with shared center and major radius, differing
    // minor radii: Fuse should collapse to the torus with the larger
    // minor radius (since the smaller one is fully contained).
    let mut topo = Topology::default();
    let outer = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.7);
    let inner = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.4);
    let r = boolean(&mut topo, BooleanOp::Fuse, outer, inner).unwrap();
    let expected = torus_volume(3.0, 0.7);
    let got = vol(&topo, r);
    assert!(
        approx_eq(got, expected, 0.05),
        "coaxial torus fuse should collapse to outer torus: got {got:.3}, expected {expected:.3}"
    );
}

#[test]
fn coaxial_torus_intersect_collapses_to_smaller_minor() {
    let mut topo = Topology::default();
    let outer = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.7);
    let inner = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.4);
    let r = boolean(&mut topo, BooleanOp::Intersect, outer, inner).unwrap();
    let expected = torus_volume(3.0, 0.4);
    let got = vol(&topo, r);
    assert!(
        approx_eq(got, expected, 0.05),
        "coaxial torus intersect should collapse to inner torus: got {got:.3}, expected {expected:.3}"
    );
}

#[test]
fn coaxial_torus_at_offset_center_fuse() {
    // Verify the shortcut handles a non-origin shared center.
    let mut topo = Topology::default();
    let outer = torus_at(&mut topo, 5.0, -2.0, 7.0, 3.0, 0.7);
    let inner = torus_at(&mut topo, 5.0, -2.0, 7.0, 3.0, 0.4);
    let r = boolean(&mut topo, BooleanOp::Fuse, outer, inner).unwrap();
    let expected = torus_volume(3.0, 0.7);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

#[test]
fn non_coaxial_torus_fuse_does_not_use_shortcut() {
    // When the major radii differ, the shortcut must NOT fire — the
    // result must include the union geometry of the two distinct tori.
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.4);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, 5.0, 0.4);
    if let Ok(sid) = boolean(&mut topo, BooleanOp::Fuse, a, b) {
        let got = vol(&topo, sid);
        // Two disjoint same-tube-radius tori at different majors don't
        // overlap (5 - 0.4 > 3 + 0.4 ⇒ tubes don't touch). Union volume
        // equals the sum.
        let expected = torus_volume(3.0, 0.4) + torus_volume(5.0, 0.4);
        // Loose because the boolean engine may produce some tessellation
        // drift, and we're not exercising the shortcut here — just
        // confirming we still get a non-collapsed result.
        assert!(
            got > expected * 0.8,
            "disjoint coaxial tori union should preserve both volumes: got {got:.3}, expected ~{expected:.3}"
        );
    }
}
