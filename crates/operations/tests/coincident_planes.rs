//! Coincident-plane scenarios for boolean robustness.
//!
//! Topology-organized stress corpus targeting the same-domain detector
//! and its integration with the GFA boolean pipeline. Complements
//! `boolean_stress.rs` and `boolean_edge_cases.rs` with scenarios that
//! specifically exercise plane-plane same-domain handling at the
//! face / edge / vertex / partial-overlap / nested / sliver levels.

#![allow(clippy::unwrap_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_manifold;

const DEFLECTION: f64 = 0.1;

fn check_manifold(topo: &Topology, solid: SolidId) -> usize {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(
        validate_shell_manifold(sh, topo).is_ok(),
        "result should be manifold"
    );
    sh.faces().len()
}

fn assert_volume_close(topo: &Topology, solid: SolidId, expected: f64) {
    let vol = solid_volume(topo, solid, DEFLECTION).unwrap();
    let diff = (vol - expected).abs();
    assert!(
        diff < expected.max(1.0) * 0.01,
        "volume {vol:.6} not within 1% of expected {expected:.6}"
    );
}

fn box_at(topo: &mut Topology, x: f64, y: f64, z: f64, sx: f64, sy: f64, sz: f64) -> SolidId {
    let b = make_box(topo, sx, sy, sz).unwrap();
    transform_solid(topo, b, &Mat4::translation(x, y, z)).unwrap();
    b
}

// ── 1. Face-on-face stack (most common SD case) ────────────────────────

#[test]
fn face_on_face_unit_stack_fuse() {
    // [0,1]^3 stacked under [0,1]^2 × [1,2] — full face coincidence.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let faces = check_manifold(&topo, r);
    assert_eq!(faces, 10, "face-stack fuse should drop the shared face");
    assert_volume_close(&topo, r, 2.0);
}

#[test]
fn face_on_face_three_box_chain_fuse() {
    // Three boxes sharing two faces total ([0,1]∪[1,2]∪[2,3] in x).
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let c = box_at(&mut topo, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let ab = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let abc = boolean(&mut topo, BooleanOp::Fuse, ab, c).unwrap();
    let _faces = check_manifold(&topo, abc);
    assert_volume_close(&topo, abc, 3.0);
}

#[test]
fn face_on_face_tall_thin_stack_fuse() {
    // 1×1×0.001 face-stacked on 1×1×1 — exercises sliver-thin top.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 0.0, 1.0, 1.0, 1.0, 0.001);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 1.001);
}

// ── 2. Partial face overlap (offset stacks) ────────────────────────────

#[test]
#[ignore = "Gap: partial face overlap (offset stack) currently produces non-manifold output. \
            SD detector requires identical edge sets; partial overlap goes through FF intersection \
            path which has issues here. Tracked as a same-domain robustness gap."]
fn partial_face_overlap_quarter_offset_fuse() {
    // B sits half-on, half-off A in x. Shared face is partial.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 2.0);
}

#[test]
#[ignore = "Gap: passes today but exercises the same partial-face-overlap GFA path \
            as `partial_face_overlap_quarter_offset_fuse`. Ignored to avoid silent \
            flakiness — should be unignored together with quarter_offset_fuse once \
            the partial-overlap gap closes."]
fn partial_face_overlap_diagonal_offset_fuse() {
    // B offset diagonally — only a quarter of the upper face is shared.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 2.0);
}

// ── 3. Fully nested coplanar (small face inside larger) ────────────────

#[test]
#[ignore = "Gap: fully-nested coplanar fuse produces volume drift (~2% surplus). \
            Tessellation-based volume reads ~17.33 instead of 17.0, suggesting either \
            an internal-face leak in the result topology or winding inconsistency \
            in the assembled shell."]
fn fully_nested_coplanar_fuse() {
    // Small box stacked on the centre of a larger one, top face of A
    // strictly contains bottom face of B.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 1.0);
    let b = box_at(&mut topo, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 16.0 + 1.0);
}

#[test]
fn fully_nested_coplanar_cut() {
    // Same shape but cut B from A — leaves L-shape with embossed top.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 4.0, 4.0, 1.0);
    let b = box_at(&mut topo, 1.5, 1.5, 0.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 16.0 - 1.0);
}

// ── 4. Edge-on-edge contact (1D coincidence) ───────────────────────────

#[test]
fn edge_on_edge_offset_corner_fuse() {
    // B touches A at the top edge, offset in z so they share a line, not a face.
    // [0,1]^3 and [1,2]×[0,1]×[1,2] meet only along edge x=1, z=1, y∈[0,1].
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    // Edge-only contact — fuse should report a non-manifold result or
    // a compound; we just require that it does not panic and that any
    // returned solid has positive volume (single-component or merged).
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b);
    if let Ok(sid) = r {
        let v = solid_volume(&topo, sid, DEFLECTION).unwrap();
        assert!(v > 0.0, "edge-contact fuse volume should be positive");
        assert!(v <= 2.001, "edge-contact fuse volume should not exceed sum");
    }
}

// ── 5. Vertex-on-vertex contact (0D coincidence) ───────────────────────

#[test]
fn vertex_on_vertex_diagonal_fuse() {
    // Boxes meet at exactly one corner: [0,1]^3 vs [1,2]^3.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b);
    if let Ok(sid) = r {
        let v = solid_volume(&topo, sid, DEFLECTION).unwrap();
        assert!(v > 0.0, "vertex-contact fuse volume should be positive");
        assert!(
            v <= 2.001,
            "vertex-contact fuse volume should not exceed sum"
        );
    }
}

// ── 6. Sub-tolerance gap (sliver between two faces) ───────────────────

#[test]
fn sub_tolerance_gap_fuse_treated_as_coincident() {
    // Faces separated by 0.4× linear tolerance — should round to coincident.
    let gap = 4e-8; // < default linear tol 1e-7
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 0.0, 1.0 + gap, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    let v = solid_volume(&topo, r, DEFLECTION).unwrap();
    assert!(
        (v - 2.0).abs() < 1e-3,
        "sub-tolerance gap fuse volume {v} should be ~2.0"
    );
}

#[test]
fn super_tolerance_gap_fuse_disjoint() {
    // Faces separated by 100× linear tolerance — should NOT be coincident.
    let gap = 1e-5;
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 0.0, 1.0 + gap, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b);
    // Either a compound with two parts or two separate solids — we
    // accept any non-panicking outcome here. The KEY assertion is that
    // a coincidence-like result isn't fabricated.
    if let Ok(sid) = r {
        let v = solid_volume(&topo, sid, DEFLECTION).unwrap();
        assert!(
            v <= 2.0 + 1e-3,
            "super-tolerance gap fuse volume {v} should not exceed sum"
        );
    }
}

// ── 7. Multi-axis coincidence (faces shared along several axes) ───────

#[test]
#[ignore = "Gap: L-shape fuse (multi-axis face share) produces non-manifold output. \
            Two boxes meeting along an L-junction expose the same partial-overlap path \
            as `partial_face_overlap_quarter_offset_fuse`."]
fn multi_axis_face_share_l_shape_fuse() {
    // L-shape: A=[0,2]×[0,1]^2, B=[0,1]×[1,2]×[0,1]. Share x∈[0,1], y=1.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0);
    let b = box_at(&mut topo, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let _faces = check_manifold(&topo, r);
    assert_volume_close(&topo, r, 3.0);
}

// ── 8. Symmetry / commutativity of SD detection ───────────────────────

#[test]
fn face_on_face_fuse_is_commutative() {
    // A∪B and B∪A should have the same volume and face count.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let ab = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let mut topo2 = Topology::default();
    let a2 = box_at(&mut topo2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b2 = box_at(&mut topo2, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let ba = boolean(&mut topo2, BooleanOp::Fuse, b2, a2).unwrap();
    let v_ab = solid_volume(&topo, ab, DEFLECTION).unwrap();
    let v_ba = solid_volume(&topo2, ba, DEFLECTION).unwrap();
    assert!(
        (v_ab - v_ba).abs() < 1e-6,
        "fuse should be commutative: {v_ab} vs {v_ba}"
    );
}

// ── 9. Anti-symmetric coplanar (opposite normals) ─────────────────────

#[test]
fn coplanar_opposite_normals_intersect_zero_volume() {
    // Two boxes that touch only on a face — intersect should be the face,
    // which has zero volume. Operation should not panic; result either
    // returns Err (degenerate) or zero-volume solid.
    let mut topo = Topology::default();
    let a = box_at(&mut topo, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let b = box_at(&mut topo, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    let r = boolean(&mut topo, BooleanOp::Intersect, a, b);
    if let Ok(sid) = r {
        let v = solid_volume(&topo, sid, DEFLECTION).unwrap_or(0.0);
        assert!(
            v < 1e-3,
            "coplanar-only intersect should have ~zero volume, got {v}"
        );
    }
}
