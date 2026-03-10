//! Boolean operation mathematical invariant tests.
//!
//! Verifies conservation laws, commutativity, and algebraic identities
//! that must hold for any correct boolean implementation.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::copy::copy_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::test_utils::make_unit_cube_manifold_at;
use brepkit_topology::validation::validate_shell_manifold;

const DEFLECTION: f64 = 0.1;

fn vol(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn check_manifold(topo: &Topology, solid: brepkit_topology::solid::SolidId) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    validate_shell_manifold(sh, &topo.faces, &topo.wires).expect("result should be manifold");
}

// -- Volume conservation --------------------------------------------------

#[test]
fn volume_conservation_overlapping_boxes() {
    // V(A) + V(B) = V(A|B) + V(A&B)
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let vol_a = vol(&topo, a);
    let vol_b = vol(&topo, b);

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let vol_fused = vol(&topo, fused);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a2, b2).unwrap();
    let vol_inter = vol(&topo, inter);

    let lhs = vol_a + vol_b;
    let rhs = vol_fused + vol_inter;
    let rel_error = (lhs - rhs).abs() / lhs;
    assert!(
        rel_error < 0.01,
        "conservation violated: V(A)+V(B)={lhs:.6}, V(A|B)+V(A&B)={rhs:.6} (error: {:.2}%)",
        rel_error * 100.0
    );
}

#[test]
fn volume_conservation_3d_offset() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let vol_a = vol(&topo, a);
    let vol_b = vol(&topo, b);

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let vol_fused = vol(&topo, fused);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a2, b2).unwrap();
    let vol_inter = vol(&topo, inter);

    let lhs = vol_a + vol_b;
    let rhs = vol_fused + vol_inter;
    let rel_error = (lhs - rhs).abs() / lhs;
    assert!(
        rel_error < 0.01,
        "conservation violated: V(A)+V(B)={lhs:.6}, V(A|B)+V(A&B)={rhs:.6} (error: {:.2}%)",
        rel_error * 100.0
    );
}

// -- Commutativity --------------------------------------------------------

#[test]
fn fuse_commutativity() {
    let mut topo = Topology::new();
    let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b1 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let r1 = boolean(&mut topo, BooleanOp::Fuse, a1, b1).unwrap();
    let v1 = vol(&topo, r1);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let r2 = boolean(&mut topo, BooleanOp::Fuse, b2, a2).unwrap();
    let v2 = vol(&topo, r2);

    let rel_error = (v1 - v2).abs() / v1;
    assert!(
        rel_error < 0.001,
        "fuse should be commutative: V(A|B)={v1:.6} vs V(B|A)={v2:.6}"
    );
}

#[test]
fn intersect_commutativity() {
    let mut topo = Topology::new();
    let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b1 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let r1 = boolean(&mut topo, BooleanOp::Intersect, a1, b1).unwrap();
    let v1 = vol(&topo, r1);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let r2 = boolean(&mut topo, BooleanOp::Intersect, b2, a2).unwrap();
    let v2 = vol(&topo, r2);

    let rel_error = (v1 - v2).abs() / v1;
    assert!(
        rel_error < 0.001,
        "intersect should be commutative: V(A&B)={v1:.6} vs V(B&A)={v2:.6}"
    );
}

// -- Cut complement -------------------------------------------------------

#[test]
fn cut_complement_identity() {
    // V(A-B) = V(A) - V(A&B)
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let vol_a = vol(&topo, a);

    let cut = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    let vol_cut = vol(&topo, cut);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a2, b2).unwrap();
    let vol_inter = vol(&topo, inter);

    let expected = vol_a - vol_inter;
    let rel_error = (vol_cut - expected).abs() / vol_a;
    assert!(
        rel_error < 0.01,
        "cut complement: V(A-B)={vol_cut:.6}, V(A)-V(A&B)={expected:.6}"
    );
}

// -- Anti-commutativity ---------------------------------------------------

#[test]
fn anti_commutativity_identity() {
    // V(A-B) + V(B-A) + 2*V(A&B) = V(A) + V(B)
    let mut topo = Topology::new();

    let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b1 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let vol_a = vol(&topo, a1);
    let vol_b = vol(&topo, b1);

    let cut_ab = boolean(&mut topo, BooleanOp::Cut, a1, b1).unwrap();
    let vol_a_minus_b = vol(&topo, cut_ab);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let cut_ba = boolean(&mut topo, BooleanOp::Cut, b2, a2).unwrap();
    let vol_b_minus_a = vol(&topo, cut_ba);

    let a3 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b3 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a3, b3).unwrap();
    let vol_inter = vol(&topo, inter);

    let lhs = vol_a_minus_b + vol_b_minus_a + 2.0 * vol_inter;
    let rhs = vol_a + vol_b;
    let rel_error = (lhs - rhs).abs() / rhs;
    assert!(
        rel_error < 0.01,
        "anti-commutativity: LHS={lhs:.6}, RHS={rhs:.6} (error: {:.2}%)",
        rel_error * 100.0
    );
}

// -- Self-boolean identities ----------------------------------------------

#[test]
fn identical_solids_fuse_preserves_volume() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = copy_solid(&mut topo, a).unwrap();
    let vol_a = vol(&topo, a);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let vol_result = vol(&topo, result);

    let rel_error = (vol_result - vol_a).abs() / vol_a;
    assert!(
        rel_error < 0.01,
        "A|A should equal V(A): got {vol_result:.6}, expected {vol_a:.6}"
    );
}

#[test]
fn identical_solids_intersect_preserves_volume() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = copy_solid(&mut topo, a).unwrap();
    let vol_a = vol(&topo, a);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let vol_result = vol(&topo, result);

    let rel_error = (vol_result - vol_a).abs() / vol_a;
    assert!(
        rel_error < 0.01,
        "A&A should equal V(A): got {vol_result:.6}, expected {vol_a:.6}"
    );
}

// -- Manifold and Euler checks on boolean results -------------------------

#[test]
fn boolean_results_are_manifold() {
    let mut topo = Topology::new();

    let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b1 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let fused = boolean(&mut topo, BooleanOp::Fuse, a1, b1).unwrap();
    check_manifold(&topo, fused);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let cut = boolean(&mut topo, BooleanOp::Cut, a2, b2).unwrap();
    check_manifold(&topo, cut);

    let a3 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b3 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a3, b3).unwrap();
    check_manifold(&topo, inter);
}

// -- Mixed analytic conservation ------------------------------------------

#[test]
fn conservation_cylinder_box() {
    // V(A) + V(B) = V(A|B) + V(A&B) for cylinder + box
    let mut topo = Topology::new();
    let base = make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
    let cyl = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let vol_a = vol(&topo, base);
    let vol_b = vol(&topo, cyl);

    let fused = boolean(&mut topo, BooleanOp::Fuse, base, cyl).unwrap();
    let vol_fused = vol(&topo, fused);

    let base2 = make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
    let cyl2 = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    transform_solid(&mut topo, cyl2, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let inter = boolean(&mut topo, BooleanOp::Intersect, base2, cyl2).unwrap();
    let vol_inter = vol(&topo, inter);

    let lhs = vol_a + vol_b;
    let rhs = vol_fused + vol_inter;
    let rel_error = (lhs - rhs).abs() / lhs;
    assert!(
        rel_error < 0.05,
        "conservation (cyl+box): V(A)+V(B)={lhs:.3}, V(A|B)+V(A&B)={rhs:.3} (error: {:.2}%)",
        rel_error * 100.0
    );
}
