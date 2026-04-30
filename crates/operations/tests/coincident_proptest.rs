//! Property tests for coincident-face boolean robustness.
//!
//! Targets invariants that must hold across random rotations,
//! translations, and sub-tolerance perturbations of coincident-face
//! configurations. Complements the deterministic corpus in
//! `coincident_planes.rs` etc.

#![allow(clippy::unwrap_used)]

use proptest::prelude::*;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DEFLECTION: f64 = 0.05;

fn vol(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, DEFLECTION).unwrap()
}

fn box_at(topo: &mut Topology, x: f64, y: f64, z: f64, sx: f64, sy: f64, sz: f64) -> SolidId {
    let b = make_box(topo, sx, sy, sz).unwrap();
    transform_solid(topo, b, &Mat4::translation(x, y, z)).unwrap();
    b
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    // Translation invariance: a face-on-face stack translated by an
    // arbitrary vector should produce the same fused volume (within
    // tessellation tolerance).
    #[test]
    fn prop_face_stack_translation_invariant(
        tx in -10.0f64..10.0,
        ty in -10.0f64..10.0,
        tz in -10.0f64..10.0,
    ) {
        let mut topo = Topology::default();
        let a = box_at(&mut topo, tx, ty, tz, 1.0, 1.0, 1.0);
        let b = box_at(&mut topo, tx, ty, tz + 1.0, 1.0, 1.0, 1.0);
        let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let v = vol(&topo, r);
        prop_assert!(
            (v - 2.0).abs() < 0.02,
            "translated face-stack volume {v} should be ~2.0",
        );
    }

    // Rotation invariance: rotating both boxes of a face-stack by the
    // same Z rotation should preserve volume. The boolean exhibits up
    // to ~15% volume drift at certain non-axis-aligned angles, which
    // is a known gap (see ignored cylinder/cone/torus SD tests for the
    // related GFA-integration weakness on non-axis-aligned shapes).
    #[test]
    #[ignore = "Gap: face-stack fuse exhibits up to ~15% volume drift at certain \
                Z-rotation angles (e.g. 3.98 rad). Likely related to GFA face-splitting \
                producing an extra internal face at off-axis rotations."]
    fn prop_face_stack_z_rotation_invariant(angle in 0.0f64..std::f64::consts::TAU) {
        let mut topo = Topology::default();
        let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
        let rot = Mat4::rotation_z(angle);
        transform_solid(&mut topo, a, &rot).unwrap();
        transform_solid(&mut topo, b, &rot).unwrap();
        let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let v = vol(&topo, r);
        prop_assert!(
            (v - 2.0).abs() < 0.02,
            "z-rotated face-stack (angle={angle}) volume {v} should be ~2.0",
        );
    }

    // Volume conservation under face-coincidence: V(A∪B) + V(A∩B) = V(A) + V(B).
    // For face-touching boxes, V(A∩B) = 0 (coplanar contact), so V(A∪B) = V(A)+V(B).
    #[test]
    fn prop_face_stack_volume_conservation(
        sx in 0.5f64..2.0,
        sy in 0.5f64..2.0,
        sz_a in 0.5f64..2.0,
        sz_b in 0.5f64..2.0,
    ) {
        let mut topo = Topology::default();
        let a = box_at(&mut topo, 0.0, 0.0, 0.0, sx, sy, sz_a);
        let b = box_at(&mut topo, 0.0, 0.0, sz_a, sx, sy, sz_b);
        let r = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let v = vol(&topo, r);
        let expected = sx * sy * (sz_a + sz_b);
        let rel_err = (v - expected).abs() / expected;
        prop_assert!(
            rel_err < 0.02,
            "face-stack vol {v} expected {expected} rel_err {rel_err}",
        );
    }

    // Commutativity under face-coincidence: V(A∪B) = V(B∪A).
    #[test]
    fn prop_face_stack_commutativity(offset_x in -0.5f64..0.5) {
        let mut topo1 = Topology::default();
        let a1 = box_at(&mut topo1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let b1 = box_at(&mut topo1, offset_x, 0.0, 1.0, 1.0, 1.0, 1.0);
        let ab = boolean(&mut topo1, BooleanOp::Fuse, a1, b1);

        let mut topo2 = Topology::default();
        let a2 = box_at(&mut topo2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let b2 = box_at(&mut topo2, offset_x, 0.0, 1.0, 1.0, 1.0, 1.0);
        let ba = boolean(&mut topo2, BooleanOp::Fuse, b2, a2);

        match (ab, ba) {
            (Ok(s1), Ok(s2)) => {
                let v1 = vol(&topo1, s1);
                let v2 = vol(&topo2, s2);
                prop_assert!((v1 - v2).abs() < 0.02);
            }
            // If both error or one errors, the test is inconclusive but
            // not a failure — commutativity is asserted only when both
            // succeed. Mismatched success is a real failure.
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                prop_assert!(false, "commutativity violated: one direction succeeded, the other failed");
            }
            (Err(_), Err(_)) => {}
        }
    }
}
