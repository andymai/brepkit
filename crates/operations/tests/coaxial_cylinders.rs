//! Coaxial-cylinder scenarios for boolean robustness.
//!
//! Tests the same-domain detector and GFA boolean pipeline against
//! coaxial cylinder configurations. Cylinder same-domain requires
//! matching axis line (origin along axis + parallel direction) and
//! matching radius — see `algo/src/builder/same_domain.rs`.
//!
//! NOTE: cylinder unit tests in `same_domain.rs` show the DETECTOR works
//! correctly. The end-to-end booleans below currently fail because the
//! GFA pipeline integration of cylinder SD pairs has known gaps (see
//! existing `boolean_stress.rs` ignored cylinder tests). This corpus
//! documents the territory for future PRs to address.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_manifold;

const DEFLECTION: f64 = 0.05;

fn check_manifold(topo: &Topology, solid: SolidId) -> usize {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(
        validate_shell_manifold(sh, topo).is_ok(),
        "result should be manifold"
    );
    sh.faces().len()
}

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn approx_eq(a: f64, b: f64, frac: f64) -> bool {
    (a - b).abs() < a.abs().max(b.abs()).max(1.0) * frac
}

fn cylinder_at_z(topo: &mut Topology, z: f64, radius: f64, height: f64) -> SolidId {
    let c = make_cylinder(topo, radius, height).unwrap();
    if z != 0.0 {
        transform_solid(topo, c, &Mat4::translation(0.0, 0.0, z)).unwrap();
    }
    c
}

// ── 0. Baseline: non-coincident cylinder boolean (must pass) ──────────

#[test]
fn baseline_cylinder_disjoint_intersect_empty_or_zero() {
    // Two cylinders far apart on z-axis — intersect is empty / zero.
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 1.0);
    let b = cylinder_at_z(&mut topo, 10.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    if let Ok(sid) = r {
        let v = vol(&topo, sid);
        assert!(v < 1e-6, "disjoint intersect should be ~zero, got {v}");
    }
    // Err is also acceptable for a degenerate-empty intersect.
}

// ── 1. Identical cylinders (degenerate SD case) ───────────────────────

#[test]
#[ignore = "Gap: identical-cylinder fuse/intersect — SD detector identifies the pair \
            but GFA pipeline integration produces wrong topology (volume off ~18%)."]
fn identical_cylinders_fuse_preserves_volume() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let b = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 2.0;
    let got = vol(&topo, r);
    assert!(
        approx_eq(got, expected, 0.02),
        "identical-cylinder fuse vol {got}, expected {expected}"
    );
}

#[test]
#[ignore = "Gap: identical-cylinder fuse/intersect — SD detector identifies the pair \
            but GFA pipeline integration produces wrong topology (volume off ~15%)."]
fn identical_cylinders_intersect_preserves_volume() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let b = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 2.0;
    let got = vol(&topo, r);
    assert!(
        approx_eq(got, expected, 0.02),
        "identical-cylinder intersect vol {got}, expected {expected}"
    );
}

// ── 2. Cap-on-cap stack (coplanar cap faces + lateral SD) ─────────────

#[test]
#[ignore = "Gap: cylinder cap-on-cap stack — SD pair found on cap planes but lateral \
            cylinder surfaces are also SD across the joint, and GFA fails to merge cleanly."]
fn cylinder_cap_on_cap_stack_fuse() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 1.0);
    let b = cylinder_at_z(&mut topo, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 2.0;
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

#[test]
#[ignore = "Gap: 3-cylinder cap-stack chain — same root cause as 2-cylinder cap-stack."]
fn cylinder_cap_on_cap_three_stack_fuse() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 1.0);
    let b = cylinder_at_z(&mut topo, 1.0, 1.0, 1.0);
    let c = cylinder_at_z(&mut topo, 2.0, 1.0, 1.0);
    let ab = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let abc = boolean(&mut topo, BooleanOp::Fuse, ab, c).unwrap();
    let _faces = check_manifold(&topo, abc);
    let expected = PI * 1.0 * 1.0 * 3.0;
    let got = vol(&topo, abc);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 3. Coaxial overlap (lateral SD case) ──────────────────────────────

#[test]
#[ignore = "Gap: coaxial cylinder partial-Z overlap — lateral SD region is shared \
            but GFA does not produce a clean merged solid."]
fn cylinder_coaxial_partial_z_overlap_fuse() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let b = cylinder_at_z(&mut topo, 1.0, 1.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 3.0;
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

#[test]
#[ignore = "Gap: coaxial containment fuse — inner cylinder SD with outer along its \
            full Z range should collapse to outer, but GFA fails."]
fn cylinder_coaxial_full_z_containment_fuse() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 4.0);
    let b = cylinder_at_z(&mut topo, 1.0, 1.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 4.0;
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

#[test]
#[ignore = "Gap: coaxial containment intersect — should yield inner cylinder, \
            but GFA fails on this SD configuration."]
fn cylinder_coaxial_full_z_containment_intersect() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 4.0);
    let b = cylinder_at_z(&mut topo, 1.0, 1.0, 2.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 2.0;
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}

// ── 4. Opposite-axis (rotated 180°) ───────────────────────────────────

#[test]
#[ignore = "Gap: opposite-axis cylinder fuse — SD detector reports `same_orientation=false`, \
            but GFA pipeline does not yet honour the flag for cylinders."]
fn cylinder_opposite_axis_fuse() {
    let mut topo = Topology::default();
    let a = cylinder_at_z(&mut topo, 0.0, 1.0, 2.0);
    let b = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let flip = Mat4::translation(0.0, 0.0, 2.0) * Mat4::rotation_x(std::f64::consts::PI);
    transform_solid(&mut topo, b, &flip).unwrap();
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let expected = PI * 1.0 * 1.0 * 2.0;
    let got = vol(&topo, r);
    assert!(approx_eq(got, expected, 0.03));
}
