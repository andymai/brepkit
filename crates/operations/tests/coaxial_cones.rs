//! Coaxial-cone scenarios for boolean robustness.
//!
//! Cone same-domain requires matching apex, axis (parallel or
//! antiparallel), and half-angle — see `same_domain.rs`. The DETECTOR
//! is verified by unit tests there; the GFA pipeline integration of
//! cone SD pairs has known gaps tracked here.
//!
//! NOTE: `make_cone` produces a frustum (different bottom and top
//! radius) — for true cones, use `top_radius=0.0`. Some tests here
//! use frustums to exercise typical CAD workflows.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cone;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DEFLECTION: f64 = 0.05;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn frustum_volume(r1: f64, r2: f64, h: f64) -> f64 {
    PI * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0
}

fn approx_eq(a: f64, b: f64, frac: f64) -> bool {
    (a - b).abs() < a.abs().max(b.abs()).max(1.0) * frac
}

fn cone_at_z(topo: &mut Topology, z: f64, r1: f64, r2: f64, h: f64) -> SolidId {
    let c = make_cone(topo, r1, r2, h).unwrap();
    if z != 0.0 {
        transform_solid(topo, c, &Mat4::translation(0.0, 0.0, z)).unwrap();
    }
    c
}

// ── 0. Baseline ────────────────────────────────────────────────────────

#[test]
fn baseline_disjoint_cones_intersect_empty() {
    let mut topo = Topology::default();
    let a = cone_at_z(&mut topo, 0.0, 1.0, 0.5, 1.0);
    let b = cone_at_z(&mut topo, 10.0, 1.0, 0.5, 1.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        assert!(v < 1e-3, "disjoint cone intersect should be ~zero, got {v}");
    }
}

// ── 1. Identical frustums ──────────────────────────────────────────────

#[test]
fn identical_cones_fuse_preserves_volume() {
    let mut topo = Topology::default();
    let a = cone_at_z(&mut topo, 0.0, 1.0, 0.5, 1.0);
    let b = cone_at_z(&mut topo, 0.0, 1.0, 0.5, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = frustum_volume(1.0, 0.5, 1.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 2. Cap-on-cap stack (frustum + inverse) ────────────────────────────

#[test]
#[ignore = "Gap: cone cap-on-cap stack — cap planes are SD; lateral cone surfaces \
            share apex/axis/half-angle so are also SD via apex line. GFA fails."]
fn cone_cap_on_cap_stack_fuse() {
    // Frustum A: r1=1 (z=0), r2=0.5 (z=1).
    // Frustum B: r1=0.5 (z=1), r2=0.25 (z=2). Same apex/axis/half-angle.
    let mut topo = Topology::default();
    let a = cone_at_z(&mut topo, 0.0, 1.0, 0.5, 1.0);
    let b = cone_at_z(&mut topo, 1.0, 0.5, 0.25, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = frustum_volume(1.0, 0.25, 2.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 3. Coaxial overlap (lateral SD on cone surface) ───────────────────

#[test]
fn cone_coaxial_partial_z_overlap_fuse() {
    let mut topo = Topology::default();
    let a = cone_at_z(&mut topo, 0.0, 1.0, 0.5, 2.0);
    let b = cone_at_z(&mut topo, 1.0, 0.75, 0.25, 2.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = frustum_volume(1.0, 0.25, 3.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 4. True cone (zero top radius) — apex-sharing scenarios ───────────

#[test]
fn identical_true_cones_fuse() {
    let mut topo = Topology::default();
    let a = cone_at_z(&mut topo, 0.0, 1.0, 0.0, 2.0);
    let b = cone_at_z(&mut topo, 0.0, 1.0, 0.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = frustum_volume(1.0, 0.0, 2.0); // = π·1²·2/3
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}
