//! Boolean operation mathematical invariant tests.
//!
//! Verifies conservation laws, commutativity, and algebraic identities
//! that must hold for any correct boolean implementation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_operations::OperationsError;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::copy::copy_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer;
use brepkit_topology::solid::SolidId;
use brepkit_topology::test_utils::make_unit_cube_manifold_at;
use brepkit_topology::validation::validate_shell_manifold;

const DEFLECTION: f64 = 0.1;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn check_manifold(topo: &Topology, solid: SolidId) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    validate_shell_manifold(sh, topo).expect("result should be manifold");
}

#[allow(clippy::cast_possible_wrap)]
fn euler_characteristic(topo: &Topology, solid: SolidId) -> i64 {
    let (f, e, v) = explorer::solid_entity_counts(topo, solid).unwrap();
    (v as i64) - (e as i64) + (f as i64)
}

fn assert_euler_genus0(topo: &Topology, solid: SolidId) {
    let chi = euler_characteristic(topo, solid);
    assert_eq!(
        chi, 2,
        "expected Euler characteristic V-E+F = 2 (genus-0), got {chi}"
    );
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

// -- Euler characteristic on boolean results ------------------------------

#[test]
fn boolean_results_euler_genus0() {
    let mut topo = Topology::new();

    let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b1 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let fused = boolean(&mut topo, BooleanOp::Fuse, a1, b1).unwrap();
    assert_euler_genus0(&topo, fused);

    let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b2 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let cut = boolean(&mut topo, BooleanOp::Cut, a2, b2).unwrap();
    assert_euler_genus0(&topo, cut);

    let a3 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b3 = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a3, b3).unwrap();
    assert_euler_genus0(&topo, inter);
}

#[test]
fn boolean_3d_euler_genus0() {
    let mut topo = Topology::new();

    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    assert_euler_genus0(&topo, fused);
}

// -- Edge-on-edge contact -------------------------------------------------

#[test]
fn fuse_edge_on_edge_boxes() {
    // Two unit cubes sharing only an edge (diagonal contact).
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.0, 1.0, 0.0);

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, fused);
    // No overlap — fused volume should equal sum of both.
    let v = vol(&topo, fused);
    let rel_error = (v - 2.0).abs() / 2.0;
    assert!(
        rel_error < 0.01,
        "edge-on-edge fuse: got {v:.6}, expected 2.0"
    );
}

// -- Vertex-on-face contact -----------------------------------------------

#[test]
fn fuse_vertex_on_face_boxes() {
    // Box A at origin, Box B positioned so its corner touches A's face.
    // B at (1.0, 0.5, 0.5) — B's min-corner (1.0, 0.5, 0.5) touches
    // A's +X face, but the boxes don't overlap.
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.5, 0.5);

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, fused);
    let v = vol(&topo, fused);
    let rel_error = (v - 2.0).abs() / 2.0;
    assert!(
        rel_error < 0.01,
        "vertex-on-face fuse: got {v:.6}, expected 2.0"
    );
}

// -- Identical solids cut -------------------------------------------------

#[test]
fn identical_solids_cut_errors_or_empty() {
    // Cutting a solid from itself should produce an error or near-zero volume.
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = copy_solid(&mut topo, a).unwrap();

    match boolean(&mut topo, BooleanOp::Cut, a, b) {
        Err(_) => {} // Expected: error for empty result
        Ok(result) => {
            // If it succeeds, volume should be near zero.
            let v = vol(&topo, result);
            assert!(
                v.abs() < 0.01,
                "A-A should be empty or near-zero: got volume {v:.6}"
            );
        }
    }
}

// -- Near-coincident-face robustness (brepjs parity) ---------------------
//
// Both scenarios put face coincidence within the default `Tolerance::new()
// .linear = 1e-7` band. They mirror brepjs parity tests that exposed the
// classifier failing to handle near-coincident face geometry consistently.
//
// These tests use the same empty-result contract as `cut_box_by_large_sphere_containment`
// in `boolean/tests.rs`: a boolean returning `Err` is treated as an empty result
// (volume 0) so the invariant can be evaluated set-theoretically.

fn safe_vol(topo: &Topology, result: Result<SolidId, OperationsError>) -> f64 {
    match result {
        Ok(s) => solid_volume(topo, s, DEFLECTION).unwrap_or(0.0),
        Err(OperationsError::EmptyResult { .. }) => 0.0,
        Err(e) => panic!("unexpected boolean error (only EmptyResult should map to 0): {e:?}"),
    }
}

#[test]
fn cut_then_fuse_back_recovers_volume_near_coincident() {
    // INVARIANT: vol((A - B) ∪ (A ∩ B)) === vol(A)
    //
    // A = 0.5000001³ box at origin; B = 0.5³ box at origin.
    // B is nearly contained in A — the +x/+y/+z faces are 1e-7 apart, exactly
    // the default linear tolerance. Empty-result contract: `(A - B)` may
    // legitimately return `Err` (the difference is sub-tolerance and treated
    // as empty); when that happens, `(empty) ∪ (A ∩ B) = (A ∩ B)`.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.500_000_1, 0.500_000_1, 0.500_000_1).unwrap();
    let b = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let vol_a = vol(&topo, a);

    let a_cut = make_box(&mut topo, 0.500_000_1, 0.500_000_1, 0.500_000_1).unwrap();
    let b_cut = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let diff_opt: Option<SolidId> = boolean(&mut topo, BooleanOp::Cut, a_cut, b_cut).ok();
    let vol_diff = diff_opt.map(|s| vol(&topo, s)).unwrap_or(0.0);

    let inter = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let vol_inter = vol(&topo, inter);

    // Build the (A-B) ∪ (A∩B) — when diff is empty, just use inter.
    let vol_fused = match diff_opt {
        Some(diff) => {
            let fused = boolean(&mut topo, BooleanOp::Fuse, diff, inter).unwrap();
            vol(&topo, fused)
        }
        None => vol_inter,
    };

    let rel_error = (vol_fused - vol_a).abs() / vol_a;
    assert!(
        rel_error < 1e-4,
        "(A-B) ∪ (A∩B) should reconstruct A: got vol_fused={vol_fused:.10}, vol_a={vol_a:.10} \
         (rel error {:.4}%). vol(A-B)={vol_diff:.3e}, vol(A∩B)={vol_inter:.10}",
        rel_error * 100.0
    );
}

#[test]
fn inclusion_exclusion_near_coincident_faces() {
    // INVARIANT: vol(A ∪ B) + vol(A ∩ B) === vol(A) + vol(B)
    //
    // A = 0.5³ box at origin. B = 0.5³ box shifted by (0, 0.5 - 1e-7, 0).
    // A's +y face (y=0.5) and B's -y face (y=0.4999999) are 1e-7 apart,
    // exactly the default linear tolerance. Either operation may legitimately
    // return an empty-result error; account for that with the safe_vol helper.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let offset = 0.5_f64 - 1e-7;
    transform_solid(&mut topo, b, &Mat4::translation(0.0, offset, 0.0)).unwrap();
    let vol_a = vol(&topo, a);
    let vol_b = vol(&topo, b);

    let a2 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b2 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    transform_solid(&mut topo, b2, &Mat4::translation(0.0, offset, 0.0)).unwrap();
    let fused_result = boolean(&mut topo, BooleanOp::Fuse, a2, b2);
    let vol_fused = safe_vol(&topo, fused_result);

    let a3 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b3 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    transform_solid(&mut topo, b3, &Mat4::translation(0.0, offset, 0.0)).unwrap();
    let inter_result = boolean(&mut topo, BooleanOp::Intersect, a3, b3);
    let vol_inter = safe_vol(&topo, inter_result);

    let lhs = vol_a + vol_b;
    let rhs = vol_fused + vol_inter;
    let rel_error = (lhs - rhs).abs() / lhs;
    assert!(
        rel_error < 1e-4,
        "inclusion-exclusion: V(A)+V(B)={lhs:.10}, V(A∪B)+V(A∩B)={rhs:.10} \
         (vol_fused={vol_fused:.10}, vol_inter={vol_inter:.3e}, rel error {:.4}%)",
        rel_error * 100.0
    );
}

// -- Containment robustness (brepjs parity) ------------------------------
//
// A ⊂ B sharing the origin corner. Mathematically:
//   intersect(A, B) = A         (vol 0.125)
//   cut(A, B)       = ∅         (Err per empty-result contract)
//   fuse(∅, A)      = A         (vol 0.125)
//
// brepjs counterexample for cut-then-fuse-back: `[0.5, 0.9531210881257514,
// 1.7801e-308]` decodes to A=unitCube(0.5,0.5,0.5), B=unitCube(0.95,0.95,0.95)
// offset by (1.78e-308, 0, 0) ≈ origin. Failure mode on brepjs: vol_fused=0,
// meaning intersect(A,B) didn't return A.

#[test]
fn cut_contained_solid_returns_empty_result() {
    // A ⊂ B → Cut(A, B) is empty per the empty-result contract.
    // Without the explicit containment-Cut shortcut, GFA fabricates a
    // degenerate vol=0 solid that callers mistake for a real result.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b = make_box(&mut topo, 0.95, 0.95, 0.95).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, a, b);
    assert!(
        matches!(result, Err(OperationsError::EmptyResult { .. })),
        "Cut(A ⊂ B, B) should return EmptyResult, got {result:?}"
    );
}

#[test]
fn intersect_contained_solid_returns_contained() {
    // A ⊂ B with shared origin corner. intersect(A, B) must return ≈ A.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b = make_box(&mut topo, 0.95, 0.95, 0.95).unwrap();
    let inter = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let v = vol(&topo, inter);
    let expected = 0.125_f64;
    let rel_error = (v - expected).abs() / expected;
    assert!(
        rel_error < 1e-5,
        "intersect(A ⊂ B) should return A: got vol={v:.10}, expected {expected:.10} \
         (rel error {:.4}%)",
        rel_error * 100.0
    );
}

#[test]
fn cut_then_fuse_back_containment() {
    // Full invariant: vol((A-B) ∪ (A∩B)) ≈ vol(A), with A ⊂ B.
    // (A-B) is empty (A is fully cut away), so applying the empty-operand
    // identity `empty ∪ X = X` leaves (A∩B), which the containment
    // shortcut returns as ≈ A.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b = make_box(&mut topo, 0.95, 0.95, 0.95).unwrap();
    let vol_a = vol(&topo, a);

    let a_cut = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let b_cut = make_box(&mut topo, 0.95, 0.95, 0.95).unwrap();
    let diff_result = boolean(&mut topo, BooleanOp::Cut, a_cut, b_cut);
    assert!(
        matches!(diff_result, Err(OperationsError::EmptyResult { .. })),
        "Cut(A ⊂ B, B) should return EmptyResult, got {diff_result:?}"
    );

    let inter = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    let vol_inter = vol(&topo, inter);

    // With (A-B) empty, the invariant reduces to vol_inter ≈ vol_a.
    let rel_error = (vol_inter - vol_a).abs() / vol_a;
    assert!(
        rel_error < 1e-5,
        "empty ∪ (A∩B) should reconstruct A (A ⊂ B): vol_inter={vol_inter:.10}, \
         vol_a={vol_a:.10} (rel error {:.4}%)",
        rel_error * 100.0
    );
}

// -- Fuse(thin_shell, larger_solid_that_contains_it) ----------------------
//
// Brepjs counterexample for `cut-then-fuse-back recovers original volume`:
//   A = unitCube(0.5000001)  // [0, 0.5000001]³, vol ≈ 0.125000075
//   B = unitCube(0.5)        // [0, 0.5]³, vol = 0.125
//   diff   = Cut(A, B)       // thin L-shell, vol ≈ 7.5e-8
//   inter  = Intersect(A, B) // identical-shortcut: copy of A, vol ≈ 0.125000075
//   fused  = Fuse(diff, inter)
//   expect: vol(fused) ≈ vol(A) since diff ⊂ inter
//   actual: vol(fused) ≈ vol(diff) — fuse collapses to just the thin shell.

#[test]
#[ignore = "GFA face selection is correct (same-domain orientation now honors face reversal), but assembly leaves an open shell: diff's outer faces keep long unsplit edges that don't match the collinear sliver edges on adjacent split faces (18 free edges), so the wrapper falls back to mesh boolean. Needs a collinear-edge refinement pass in GFA assembly plus closed-shell-aware wrapper acceptance."]
fn fuse_thin_shell_with_containing_solid_preserves_larger_volume() {
    // Uses 0.5001 (above tol.linear) so the identical-Cut shortcut doesn't
    // fire and Cut(A,B) actually goes through GFA, producing the L-shell.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 0.500_1, 0.500_1, 0.500_1).unwrap();
    let b = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let vol_a = vol(&topo, a);

    let diff = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    let vol_diff = vol(&topo, diff);

    let a2 = make_box(&mut topo, 0.500_1, 0.500_1, 0.500_1).unwrap();
    let b2 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    let inter = boolean(&mut topo, BooleanOp::Intersect, a2, b2).unwrap();
    let vol_inter = vol(&topo, inter);

    let fused = boolean(&mut topo, BooleanOp::Fuse, diff, inter).unwrap();
    let vol_fused = vol(&topo, fused);

    // diff ⊂ inter (≈ A), so Fuse should return ≈ A.
    let rel_error = (vol_fused - vol_a).abs() / vol_a;
    assert!(
        rel_error < 1e-4,
        "Fuse(thin_shell, A_copy) should return ≈ vol(A): vol_fused={vol_fused:.10}, \
         vol_a={vol_a:.10}, vol_diff={vol_diff:.3e}, vol_inter={vol_inter:.10} \
         (rel error {:.4}%)",
        rel_error * 100.0
    );
}

// -- Cut cylinder from box ------------------------------------------------

#[test]
fn cut_cylinder_from_box_volume() {
    // Cylinder centered in box, protruding above and below.
    let mut topo = Topology::new();
    let base = make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
    let cyl = make_cylinder(&mut topo, 0.5, 4.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let vol_base = vol(&topo, base);
    let vol_cyl_in_box = std::f64::consts::PI * 0.5 * 0.5 * 2.0; // πr²h, h clamped to box height

    let result = boolean(&mut topo, BooleanOp::Cut, base, cyl).unwrap();
    check_manifold(&topo, result);

    let v = vol(&topo, result);
    let expected = vol_base - vol_cyl_in_box;
    let rel_error = (v - expected).abs() / expected;
    assert!(
        rel_error < 0.05,
        "cut cyl from box: got {v:.4}, expected {expected:.4} (error: {:.2}%)",
        rel_error * 100.0
    );
}
