//! Concentric-sphere scenarios for boolean robustness.
//!
//! Sphere same-domain requires matching center and radius; the SD
//! detector returns `Some(true)` (always same-direction since spheres
//! have no axis). Like cylinders, the DETECTOR works correctly
//! (see `same_domain.rs::sphere_*` unit tests) but the GFA pipeline
//! integration of sphere SD pairs has known gaps tracked here.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_sphere;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DEFLECTION: f64 = 0.05;
const SEGMENTS: usize = 32;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn sphere_volume(r: f64) -> f64 {
    4.0 * PI * r * r * r / 3.0
}

fn approx_eq(a: f64, b: f64, frac: f64) -> bool {
    (a - b).abs() < a.abs().max(b.abs()).max(1.0) * frac
}

fn sphere_at(topo: &mut Topology, x: f64, y: f64, z: f64, radius: f64) -> SolidId {
    let s = make_sphere(topo, radius, SEGMENTS).unwrap();
    if x != 0.0 || y != 0.0 || z != 0.0 {
        transform_solid(topo, s, &Mat4::translation(x, y, z)).unwrap();
    }
    s
}

// ── 0. Baseline: disjoint spheres ──────────────────────────────────────

#[test]
fn baseline_disjoint_spheres_intersect_empty() {
    let mut topo = Topology::default();
    let a = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let b = sphere_at(&mut topo, 5.0, 0.0, 0.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        assert!(
            v < 1e-3,
            "disjoint sphere intersect should be ~zero, got {v}"
        );
    }
}

// ── 1. Identical spheres (degenerate SD) ──────────────────────────────

#[test]
fn identical_spheres_fuse_preserves_volume() {
    let mut topo = Topology::default();
    let a = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let b = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = sphere_volume(1.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

#[test]
fn identical_spheres_intersect_preserves_volume() {
    let mut topo = Topology::default();
    let a = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let b = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let expected = sphere_volume(1.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}

// ── 2. Concentric different radii (NOT same-domain — must NOT merge) ──

#[test]
fn concentric_spheres_different_radii_fuse() {
    let mut topo = Topology::default();
    let outer = sphere_at(&mut topo, 0.0, 0.0, 0.0, 2.0);
    let inner = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, outer, inner).unwrap();
    let expected = sphere_volume(2.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 3. Sub-tolerance shifted center (should be SD) ────────────────────

#[test]
fn spheres_sub_tolerance_shifted_fuse() {
    let mut topo = Topology::default();
    let a = sphere_at(&mut topo, 0.0, 0.0, 0.0, 1.0);
    let b = sphere_at(&mut topo, 4e-8, 0.0, 0.0, 1.0); // < linear tol 1e-7
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let expected = sphere_volume(1.0);
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.05));
}
