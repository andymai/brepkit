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

// ── 2. Mismatched major radius — must NOT be SD or shortcut ───────────

#[test]
fn toruses_different_major_radius_intersect_nontrivial() {
    // Two toruses sharing center and axis but with different MAJOR radii
    // (minor radii also differ here). Exercises the GFA non-SD path; the
    // coaxial-torus shortcut bails because `same_major` is false, and the
    // SD detector correctly returns None since the toroidal surfaces
    // differ. Originally this test used matching majors with mismatched
    // minors, but #556 added a coaxial-torus Fuse/Intersect shortcut that
    // would silently route those exact inputs through the fast path,
    // hiding the GFA coverage gap. Switched to mismatched majors so the
    // test continues to validate the GFA pipeline.
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 0.5);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 0.7);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    // Either OK (positive volume) or Err (degenerate) is acceptable;
    // the key requirement is that neither the SD detector nor the
    // coaxial-torus shortcut hijacked this pair.
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        // Volume should be >0 and <= the smaller-tube torus volume.
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
fn different_major_torus_fuse_does_not_use_shortcut() {
    // The two tori share center and axis (so they ARE coaxial) but have
    // different *major* radii. The shortcut must NOT fire — its
    // `same_major` guard rejects this pair — so the result must include
    // the union geometry of two distinct tori.
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

/// Two coaxial tori whose tubes overlap meet in two circles about their axis,
/// and their common part is the lens between them: each slice at height `z`
/// is the annulus between the tubes' inner and outer walls there.
#[test]
fn coaxial_tori_meet_in_an_exact_lens() {
    let (big_a, small_a, big_b, small_b) = (3.0_f64, 0.5_f64, 4.0_f64, 0.7_f64);
    let mut topo = Topology::default();
    let a = torus_at(&mut topo, 0.0, 0.0, 0.0, big_a, small_a);
    let b = torus_at(&mut topo, 0.0, 0.0, 0.0, big_b, small_b);
    let lens = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();

    let report = brepkit_operations::validate::validate_solid(&topo, lens).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let faces = brepkit_topology::explorer::solid_faces(&topo, lens).unwrap();
    assert_eq!(faces.len(), 2);

    // Simpson over the height the tubes share.
    let slice = |z: f64| {
        let wall = |big: f64, small: f64| {
            let w = small.mul_add(small, -(z * z)).max(0.0).sqrt();
            (big - w, big + w)
        };
        let (lo_a, hi_a) = wall(big_a, small_a);
        let (lo_b, hi_b) = wall(big_b, small_b);
        let (lo, hi) = (lo_a.max(lo_b), hi_a.min(hi_b));
        if hi > lo {
            PI * (hi * hi - lo * lo)
        } else {
            0.0
        }
    };
    let n = 20_000;
    let top = small_a.min(small_b);
    let h = 2.0 * top / f64::from(n);
    let mut sum = slice(-top) + slice(top);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * slice(-top + h * f64::from(k));
    }
    let truth = sum * h / 3.0;
    let volume = solid_volume(&topo, lens, DEFLECTION).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "volume {volume}, truth {truth}"
    );
    let mesh = brepkit_operations::tessellate::tessellate_solid(&topo, lens, 0.01).unwrap();
    assert!(brepkit_operations::tessellate::is_watertight(&mesh));
}
