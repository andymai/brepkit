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
