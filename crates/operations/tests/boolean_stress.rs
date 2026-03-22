//! Boolean operation stress tests.
//!
//! Comprehensive suite covering edge cases: coplanar faces, tangent solids,
//! thin walls, near-miss geometry, mixed analytic surfaces, fully-contained
//! solids, and chained operations. Each test verifies face count, manifoldness,
//! and (where possible) volume correctness.

#![allow(clippy::unwrap_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean, boolean_with_options};
use brepkit_operations::copy::copy_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::test_utils::make_unit_cube_manifold_at;
use brepkit_topology::validation::validate_shell_manifold;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Verify the result is manifold and return face count.
fn check_manifold(topo: &Topology, solid: SolidId) -> usize {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(
        validate_shell_manifold(sh, topo).is_ok(),
        "result should be manifold"
    );
    sh.faces().len()
}

const DEFLECTION: f64 = 0.1;

/// Verify volume is approximately equal to expected.
fn assert_volume(topo: &Topology, solid: SolidId, expected: f64, tol_frac: f64) {
    let vol = solid_volume(topo, solid, DEFLECTION).unwrap();
    let diff = (vol - expected).abs();
    assert!(
        diff < expected * tol_frac,
        "volume {vol:.6} not within {:.1}% of expected {expected:.6} (diff={diff:.6})",
        tol_frac * 100.0
    );
}

/// Make a box at a given position.
fn box_at(topo: &mut Topology, x: f64, y: f64, z: f64, sx: f64, sy: f64, sz: f64) -> SolidId {
    let b = make_box(topo, sx, sy, sz).unwrap();
    transform_solid(topo, b, &Mat4::translation(x, y, z)).unwrap();
    b
}

// ===========================================================================
// 1. Coplanar face tests (shared face planes)
// ===========================================================================

#[test]
fn coplanar_fuse_shared_face() {
    // Two unit cubes sharing a face: [0,1]^3 and [1,2]×[0,1]^2.
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let faces = check_manifold(&topo, result);
    assert!(
        faces >= 6,
        "fused cubes should have at least 6 faces, got {faces}"
    );
}

#[test]
fn coplanar_cut_shared_face() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn coplanar_intersect_shared_face() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

    // Should produce an empty result (just touching, no volume overlap)
    // or a degenerate face — either way the operation should not panic.
    let _result = boolean(&mut topo, BooleanOp::Intersect, a, b);
}

#[test]
fn coplanar_offset_shared_face_y() {
    // Cubes sharing a face in Y direction.
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.0, 1.0, 0.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn coplanar_offset_shared_face_z() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 2. Fully-contained solid tests
// ===========================================================================

#[test]
fn fully_contained_fuse() {
    // Small box inside big box — fuse should produce the big box.
    let mut topo = Topology::new();
    let big = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let small = box_at(&mut topo, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, big, small).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn fully_contained_cut() {
    // Cut small box from inside big box — should create hollow box.
    let mut topo = Topology::new();
    let big = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let small = box_at(&mut topo, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Cut, big, small).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn fully_contained_intersect() {
    // Intersect small inside big — should produce the small box.
    let mut topo = Topology::new();
    let big = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let small = box_at(&mut topo, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, big, small).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 3. Near-miss / barely-touching tests
// ===========================================================================

#[test]
fn near_miss_disjoint_fuse() {
    // Two cubes separated by a tiny gap (1e-5).
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 1.00001, 0.0, 0.0);

    // Should produce a valid result (no intersection segments).
    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn near_miss_barely_overlapping() {
    // Two cubes overlapping by a tiny amount (1e-5).
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.99999, 0.0, 0.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn near_miss_cut_barely_touching() {
    let mut topo = Topology::new();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.99999, 0.0, 0.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 4. Thin wall tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn thin_wall_cut() {
    // Cut a thin slice from a box — result has very thin geometry.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let b = box_at(&mut topo, 4.9, 0.0, 0.0, 0.2, 10.0, 10.0); // thin cut through middle

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn thin_wall_intersect() {
    // Intersection producing a thin slab.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let b = box_at(&mut topo, 4.5, -1.0, -1.0, 1.0, 12.0, 12.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 5. Box volume verification tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn volume_fuse_overlapping_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    // Total: 2×2×2 + 2×2×2 - overlap(1×2×2) = 8 + 8 - 4 = 12
    assert_volume(&topo, result, 12.0, 0.01);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn volume_cut_overlapping_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
    // A - overlap = 8 - 4 = 4
    assert_volume(&topo, result, 4.0, 0.01);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn volume_intersect_overlapping_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
    // Overlap: 1×2×2 = 4
    assert_volume(&topo, result, 4.0, 0.01);
}

// ===========================================================================
// 6. Different axis overlaps
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_overlap_y_axis() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 12.0, 0.01);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_overlap_z_axis() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 12.0, 0.01);
}

#[test]
fn fuse_overlap_all_axes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    // overlap: 1×1×1 = 1, total = 8+8-1 = 15
    assert_volume(&topo, result, 15.0, 0.01);
}

// ===========================================================================
// 7. Symmetric / commutative property tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_commutative() {
    let mut topo = Topology::new();
    let a1 = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b1 = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let r1 = boolean(&mut topo, BooleanOp::Fuse, a1, b1).unwrap();
    let v1 = solid_volume(&topo, r1, DEFLECTION).unwrap();

    let a2 = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b2 = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let r2 = boolean(&mut topo, BooleanOp::Fuse, b2, a2).unwrap();
    let v2 = solid_volume(&topo, r2, DEFLECTION).unwrap();

    assert!(
        (v1 - v2).abs() < 0.01,
        "fuse should be commutative: {v1} vs {v2}"
    );
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn intersect_commutative() {
    let mut topo = Topology::new();
    let a1 = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b1 = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let r1 = boolean(&mut topo, BooleanOp::Intersect, a1, b1).unwrap();
    let v1 = solid_volume(&topo, r1, DEFLECTION).unwrap();

    let a2 = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b2 = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let r2 = boolean(&mut topo, BooleanOp::Intersect, b2, a2).unwrap();
    let v2 = solid_volume(&topo, r2, DEFLECTION).unwrap();

    assert!(
        (v1 - v2).abs() < 0.01,
        "intersect should be commutative: {v1} vs {v2}"
    );
}

// ===========================================================================
// 8. Volume identity tests: A ∪ B = vol(A) + vol(B) - vol(A ∩ B)
// ===========================================================================

#[test]
fn volume_identity_inclusion_exclusion() {
    let mut topo = Topology::new();

    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
    let b = box_at(&mut topo, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0);

    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let vol_fuse = solid_volume(&topo, fused, DEFLECTION).unwrap();

    // Recreate for intersection (boolean consumed the topology state).
    let a2 = box_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
    let b2 = box_at(&mut topo, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0);
    let inter = boolean(&mut topo, BooleanOp::Intersect, a2, b2).unwrap();
    let vol_inter = solid_volume(&topo, inter, DEFLECTION).unwrap();

    // vol(A) + vol(B) - vol(A∩B) should equal vol(A∪B)
    let vol_a = 27.0; // 3^3
    let vol_b = 27.0;
    let expected_fuse = vol_a + vol_b - vol_inter;

    assert!(
        (vol_fuse - expected_fuse).abs() < 0.5,
        "inclusion-exclusion: fuse={vol_fuse:.2}, expected={expected_fuse:.2}, inter={vol_inter:.2}"
    );
}

// ===========================================================================
// 9. Chained operation tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn chained_fuse_three_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let ab = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, ab);

    let c = box_at(&mut topo, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let abc = boolean(&mut topo, BooleanOp::Fuse, ab, c).unwrap();
    check_manifold(&topo, abc);
    // 3 boxes in a row: 4×2×2 = 16
    assert_volume(&topo, abc, 16.0, 0.02);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn chained_cut_multiple_holes() {
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let hole1 = box_at(&mut topo, 1.0, 1.0, -1.0, 2.0, 2.0, 12.0);

    let r1 = boolean(&mut topo, BooleanOp::Cut, base, hole1).unwrap();
    check_manifold(&topo, r1);

    let hole2 = box_at(&mut topo, 5.0, 5.0, -1.0, 2.0, 2.0, 12.0);
    let r2 = boolean(&mut topo, BooleanOp::Cut, r1, hole2).unwrap();
    check_manifold(&topo, r2);
}

#[test]
fn chained_fuse_then_cut() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
    let b = box_at(&mut topo, 2.0, 0.0, 0.0, 3.0, 3.0, 3.0);
    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();

    let cutter = box_at(&mut topo, 1.0, 1.0, -1.0, 2.0, 1.0, 5.0);
    let result = boolean(&mut topo, BooleanOp::Cut, fused, cutter).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 10. Disjoint tests
// ===========================================================================

#[test]
fn disjoint_fuse_far_apart() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 100.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn disjoint_cut_no_effect() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 8.0, 0.01);
}

#[test]
fn disjoint_intersect_empty() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    // Should error (empty result).
    let result = boolean(&mut topo, BooleanOp::Intersect, a, b);
    assert!(result.is_err(), "disjoint intersect should fail");
}

// ===========================================================================
// 11. Mixed analytic surface tests (cylinder + box)
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn cut_cylinder_from_box() {
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let cyl = make_cylinder(&mut topo, 1.0, 6.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, base, cyl).unwrap();
    check_manifold(&topo, result);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_cylinder_and_box() {
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 2.0);
    let cyl = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, base, cyl).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn intersect_cylinder_and_box() {
    // Cylinder r=1, h=6 at origin (z=0..6), translated to (2,2,-1)
    // → z=-1..5, fully through box (z=0..4). Intersection = cylinder section
    // from z=0..4, expected volume = π·r²·h = π·1·4 ≈ 12.566.
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let cyl = make_cylinder(&mut topo, 1.0, 6.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Intersect, base, cyl).unwrap();
    check_manifold(&topo, result);

    let expected = std::f64::consts::PI * 1.0 * 1.0 * 4.0;
    // Analytic boolean with exact cylinder surface — tighter tolerance.
    assert_volume(&topo, result, expected, 0.05);
}

// ===========================================================================
// 12. Large overlap ratio tests
// ===========================================================================

#[test]
fn large_overlap_90_percent() {
    // B covers 90% of A.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let b = box_at(&mut topo, -1.0, -1.0, -1.0, 10.0, 10.0, 10.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn small_overlap_1_percent() {
    // B overlaps just a corner of A.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    let b = box_at(&mut topo, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 13. Asymmetric box sizes
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_asymmetric_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    let b = box_at(&mut topo, 0.5, 0.5, 0.5, 3.0, 1.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn cut_asymmetric_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 5.0, 3.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.5, -1.0, 2.0, 1.0, 4.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 14. Self-boolean tests
// ===========================================================================

#[test]
fn self_fuse() {
    // Fusing a solid with a copy of itself.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = copy_solid(&mut topo, a).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 8.0, 0.05);
}

#[test]
fn self_intersect() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = copy_solid(&mut topo, a).unwrap();

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 8.0, 0.05);
}

// ===========================================================================
// 15. Configurable deflection tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn options_fine_deflection() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let opts = BooleanOptions {
        deflection: 0.01,
        ..Default::default()
    };
    let result = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, opts).unwrap();
    check_manifold(&topo, result);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn options_coarse_deflection() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let opts = BooleanOptions {
        deflection: 1.0,
        ..Default::default()
    };
    let result = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, opts).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 16. Edge-aligned overlaps
// ===========================================================================

#[test]
fn edge_aligned_overlap_x() {
    // Two boxes sharing an edge along X.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0);

    // Just touching along Y=2 face — should be disjoint-like for intersect.
    let fuse = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, fuse);
}

// ===========================================================================
// 17. Multiple operations on same base
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn multiple_cuts_from_same_base() {
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 10.0, 10.0, 2.0);

    // Cut 4 holes in a grid pattern.
    let h1 = box_at(&mut topo, 1.0, 1.0, -1.0, 2.0, 2.0, 4.0);
    let r1 = boolean(&mut topo, BooleanOp::Cut, base, h1).unwrap();
    check_manifold(&topo, r1);

    let h2 = box_at(&mut topo, 5.0, 1.0, -1.0, 2.0, 2.0, 4.0);
    let r2 = boolean(&mut topo, BooleanOp::Cut, r1, h2).unwrap();
    check_manifold(&topo, r2);

    let h3 = box_at(&mut topo, 1.0, 5.0, -1.0, 2.0, 2.0, 4.0);
    let r3 = boolean(&mut topo, BooleanOp::Cut, r2, h3).unwrap();
    check_manifold(&topo, r3);

    let h4 = box_at(&mut topo, 5.0, 5.0, -1.0, 2.0, 2.0, 4.0);
    let r4 = boolean(&mut topo, BooleanOp::Cut, r3, h4).unwrap();
    check_manifold(&topo, r4);
}

// ===========================================================================
// 18. Corner-only overlap
// ===========================================================================

#[test]
fn corner_overlap_fuse() {
    // Two boxes overlapping only at a corner region.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    // overlap: 0.5^3 = 0.125, total = 8+8-0.125 = 15.875
    assert_volume(&topo, result, 15.875, 0.02);
}

#[test]
fn corner_overlap_cut() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 7.875, 0.02);
}

#[test]
fn corner_overlap_intersect() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 0.125, 0.05);
}

// ===========================================================================
// 19. Half-overlap (50%) tests
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn half_overlap_fuse() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 12.0, 0.01);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn half_overlap_cut() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 4.0, 0.01);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn half_overlap_intersect() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 4.0, 0.01);
}

// ===========================================================================
// 20. Identity operations
// ===========================================================================

#[test]
fn cut_no_overlap_preserves_volume() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
    let b = box_at(&mut topo, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 27.0, 0.01);
}

// ===========================================================================
// 21. L-shaped boolean
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn l_shape_fuse() {
    // Create L-shape by fusing two boxes.
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    // 3+3-1 = 5 volume
    assert_volume(&topo, result, 5.0, 0.02);
}

// ===========================================================================
// 22. T-shaped boolean
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn t_shape_fuse() {
    // Create T-shape by fusing horizontal and vertical boxes.
    let mut topo = Topology::new();
    let horizontal = box_at(&mut topo, 0.0, 0.0, 0.0, 6.0, 1.0, 1.0);
    let vertical = box_at(&mut topo, 2.0, 0.0, 0.0, 2.0, 3.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, horizontal, vertical).unwrap();
    check_manifold(&topo, result);
    // 6 + 6 - 2 = 10 volume
    assert_volume(&topo, result, 10.0, 0.02);
}

// ===========================================================================
// 23. Cross-shaped boolean
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn cross_shape_fuse() {
    let mut topo = Topology::new();
    let h = box_at(&mut topo, 0.0, 1.0, 0.0, 4.0, 2.0, 1.0);
    let v = box_at(&mut topo, 1.0, 0.0, 0.0, 2.0, 4.0, 1.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, h, v).unwrap();
    check_manifold(&topo, result);
    // 8 + 8 - 4 = 12
    assert_volume(&topo, result, 12.0, 0.02);
}

// ===========================================================================
// 24. Cylinder boolean tests
// ===========================================================================

#[test]
fn fuse_two_cylinders() {
    let mut topo = Topology::new();
    let a = make_cylinder(&mut topo, 1.0, 3.0).unwrap();
    let b = make_cylinder(&mut topo, 1.0, 3.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
}

#[test]
fn cut_cylinder_from_cylinder() {
    let mut topo = Topology::new();
    let a = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let b = make_cylinder(&mut topo, 1.0, 5.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    check_manifold(&topo, result);
}

// ===========================================================================
// 25. Box-box with volume verification (varied sizes)
// ===========================================================================

#[test]
fn volume_tiny_overlap() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0);
    let b = box_at(&mut topo, 4.8, 4.8, 4.8, 5.0, 5.0, 5.0);

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
    check_manifold(&topo, result);
    // overlap: 0.2^3 = 0.008
    assert_volume(&topo, result, 0.008, 0.10);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn volume_large_boxes() {
    let mut topo = Topology::new();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
    let b = box_at(&mut topo, 50.0, 0.0, 0.0, 100.0, 100.0, 100.0);

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    check_manifold(&topo, result);
    assert_volume(&topo, result, 1_500_000.0, 0.01);
}

// ===========================================================================
// 26. Cylinder-box volume verification
// ===========================================================================

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn cut_cylinder_from_box_volume() {
    // Box 4×4×4 at origin (z=0..4), cylinder r=1, h=6 at z=0..6
    // (make_cylinder places base at z=0), translated to (2,2,-1) → z=-1..5.
    // Overlap height = 4 (clamped to box z=0..4).
    // Expected: box_vol - π·r²·h = 64 - π·1²·4 ≈ 64 - 12.566 ≈ 51.434
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
    let cyl = make_cylinder(&mut topo, 1.0, 6.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, base, cyl).unwrap();
    check_manifold(&topo, result);

    let box_vol = 64.0;
    let cyl_overlap = std::f64::consts::PI * 1.0 * 1.0 * 4.0; // π·r²·h
    let expected = box_vol - cyl_overlap;
    // Analytic boolean with exact cylinder surface — tighter tolerance.
    assert_volume(&topo, result, expected, 0.05);
}

#[test]
#[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
fn fuse_cylinder_and_box_volume() {
    // Box 4×4×2 at origin (z=0..2), cylinder r=1, h=4 at origin
    // (z=0..4), translated to (2,2,-1) → z=-1..3, sticks out 1 above + 1 below.
    // Overlap height = 2 (z=0..2).
    // Expected: box_vol + cylinder_vol - overlap
    //         = 32 + π·1²·4 - π·1²·2 ≈ 32 + 12.566 - 6.283 ≈ 38.283
    let mut topo = Topology::new();
    let base = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 2.0);
    let cyl = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -1.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, base, cyl).unwrap();
    check_manifold(&topo, result);

    let box_vol = 32.0;
    let cyl_vol = std::f64::consts::PI * 1.0 * 1.0 * 4.0;
    let overlap = std::f64::consts::PI * 1.0 * 1.0 * 2.0;
    let expected = box_vol + cyl_vol - overlap;
    // Analytic boolean with exact cylinder surface — tighter tolerance.
    assert_volume(&topo, result, expected, 0.05);
}
