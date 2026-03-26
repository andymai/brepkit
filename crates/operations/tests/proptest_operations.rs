//! Property-based tests for brepkit operations layer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use proptest::prelude::*;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::test_utils::make_unit_cube_manifold_at;

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::copy::copy_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_operations::transform::transform_solid;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    // 1. V(A) + V(B) = V(A|B) + V(A&B)
    #[test]
    fn prop_boolean_volume_conservation(offset in 0.1f64..0.9) {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, offset, 0.0, 0.0);

        let va = solid_volume(&topo, a, 0.01).unwrap();
        let vb = solid_volume(&topo, b, 0.01).unwrap();

        // Need fresh copies for each boolean since booleans consume topology
        let a_fuse = copy_solid(&mut topo, a).unwrap();
        let b_fuse = copy_solid(&mut topo, b).unwrap();
        let fused = boolean(&mut topo, BooleanOp::Fuse, a_fuse, b_fuse).unwrap();
        let v_union = solid_volume(&topo, fused, 0.01).unwrap();

        let a_int = copy_solid(&mut topo, a).unwrap();
        let b_int = copy_solid(&mut topo, b).unwrap();
        let intersected = boolean(&mut topo, BooleanOp::Intersect, a_int, b_int).unwrap();
        let v_inter = solid_volume(&topo, intersected, 0.01).unwrap();

        let lhs = va + vb;
        let rhs = v_union + v_inter;
        let rel_err = (lhs - rhs).abs() / lhs;
        prop_assert!(rel_err < 0.02, "volume conservation failed: lhs={lhs}, rhs={rhs}, rel_err={rel_err}");
    }

    // 2. fuse(A,B) volume == fuse(B,A) volume
    #[test]
    fn prop_boolean_commutativity(offset in 0.1f64..0.9) {
        let mut topo = Topology::new();

        let a1 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b1 = make_unit_cube_manifold_at(&mut topo, offset, 0.0, 0.0);
        let fuse_ab = boolean(&mut topo, BooleanOp::Fuse, a1, b1).unwrap();
        let v_ab = solid_volume(&topo, fuse_ab, 0.01).unwrap();

        let a2 = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b2 = make_unit_cube_manifold_at(&mut topo, offset, 0.0, 0.0);
        let fuse_ba = boolean(&mut topo, BooleanOp::Fuse, b2, a2).unwrap();
        let v_ba = solid_volume(&topo, fuse_ba, 0.01).unwrap();

        let rel_err = (v_ab - v_ba).abs() / v_ab;
        prop_assert!(rel_err < 0.001, "commutativity failed: v_ab={v_ab}, v_ba={v_ba}, rel_err={rel_err}");
    }

    // 3. rotation + translation preserves volume
    #[test]
    fn prop_transform_preserves_volume(theta in 0.0f64..std::f64::consts::TAU, tx in -10.0f64..10.0) {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let vol_before = solid_volume(&topo, solid, 0.01).unwrap();

        let mat = Mat4::translation(tx, 0.0, 0.0) * Mat4::rotation_z(theta);
        transform_solid(&mut topo, solid, &mat).unwrap();
        let vol_after = solid_volume(&topo, solid, 0.01).unwrap();

        let rel_err = (vol_before - vol_after).abs() / vol_before;
        prop_assert!(rel_err < 1e-6, "volume changed after transform: before={vol_before}, after={vol_after}, rel_err={rel_err}");
    }

    // 4. copy produces identical volume
    #[test]
    fn prop_copy_produces_equal_volume(dx in 0.5f64..5.0, dy in 0.5f64..5.0, dz in 0.5f64..5.0) {
        let mut topo = Topology::new();
        let original = make_box(&mut topo, dx, dy, dz).unwrap();
        let vol_orig = solid_volume(&topo, original, 0.01).unwrap();

        let copied = copy_solid(&mut topo, original).unwrap();
        let vol_copy = solid_volume(&topo, copied, 0.01).unwrap();

        let rel_err = (vol_orig - vol_copy).abs() / vol_orig;
        prop_assert!(rel_err < 1e-10, "copy volume mismatch: orig={vol_orig}, copy={vol_copy}, rel_err={rel_err}");
    }

    // 5. all tessellation triangles have positive area
    #[test]
    fn prop_tessellate_positive_area(r in 0.5f64..5.0, h in 0.5f64..5.0) {
        let mut topo = Topology::new();
        let cyl = make_cylinder(&mut topo, r, h).unwrap();
        let mesh = tessellate_solid(&topo, cyl, 0.1).unwrap();

        let n_tris = mesh.indices.len() / 3;
        for i in 0..n_tris {
            let i0 = mesh.indices[i * 3] as usize;
            let i1 = mesh.indices[i * 3 + 1] as usize;
            let i2 = mesh.indices[i * 3 + 2] as usize;
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];
            let e1 = Vec3::new(p1.x() - p0.x(), p1.y() - p0.y(), p1.z() - p0.z());
            let e2 = Vec3::new(p2.x() - p0.x(), p2.y() - p0.y(), p2.z() - p0.z());
            let cross = e1.cross(e2);
            let area = 0.5 * (cross.x() * cross.x() + cross.y() * cross.y() + cross.z() * cross.z()).sqrt();
            prop_assert!(area > 0.0, "degenerate triangle {i}: area={area}");
        }
    }

    // 6. box volume matches dx*dy*dz exactly
    #[test]
    fn prop_box_volume_exact(dx in 0.1f64..100.0, dy in 0.1f64..100.0, dz in 0.1f64..100.0) {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, dx, dy, dz).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();

        let expected = dx * dy * dz;
        let rel_err = (vol - expected).abs() / expected;
        prop_assert!(rel_err < 1e-8, "box volume mismatch: got={vol}, expected={expected}, rel_err={rel_err}");
    }

    // 7. cylinder volume matches pi*r^2*h within 1%
    #[test]
    fn prop_cylinder_volume(r in 0.5f64..5.0, h in 0.5f64..5.0) {
        let mut topo = Topology::new();
        let cyl = make_cylinder(&mut topo, r, h).unwrap();
        let vol = solid_volume(&topo, cyl, 0.01).unwrap();

        let expected = std::f64::consts::PI * r * r * h;
        let rel_err = (vol - expected).abs() / expected;
        prop_assert!(rel_err < 0.01, "cylinder volume mismatch: got={vol}, expected={expected}, rel_err={rel_err}");
    }

    // 8. V(A-B) = V(A) - V(A&B)
    #[test]
    #[ignore = "GFA pipeline limitation — old boolean pipeline removed"]
    fn prop_cut_complement(offset in 0.1f64..0.9) {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, offset, 0.0, 0.0);

        let va = solid_volume(&topo, a, 0.01).unwrap();

        // Cut A - B
        let a_cut = copy_solid(&mut topo, a).unwrap();
        let b_cut = copy_solid(&mut topo, b).unwrap();
        let cut_result = boolean(&mut topo, BooleanOp::Cut, a_cut, b_cut).unwrap();
        let v_cut = solid_volume(&topo, cut_result, 0.01).unwrap();

        // Intersect A & B
        let a_int = copy_solid(&mut topo, a).unwrap();
        let b_int = copy_solid(&mut topo, b).unwrap();
        let int_result = boolean(&mut topo, BooleanOp::Intersect, a_int, b_int).unwrap();
        let v_inter = solid_volume(&topo, int_result, 0.01).unwrap();

        let expected_cut = va - v_inter;
        let rel_err = (v_cut - expected_cut).abs() / va;
        prop_assert!(rel_err < 0.02, "cut complement failed: v_cut={v_cut}, expected={expected_cut}, rel_err={rel_err}");
    }

    // 9. tessellation produces watertight mesh (every edge has a matching reverse)
    #[test]
    fn prop_tessellate_watertight_box(dx in 0.5f64..10.0, dy in 0.5f64..10.0, dz in 0.5f64..10.0) {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, dx, dy, dz).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // Snap positions to 6 decimal places for matching
        let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
        let key = |p: &brepkit_math::vec::Point3| -> (i64, i64, i64) {
            (snap(p.x()), snap(p.y()), snap(p.z()))
        };

        // Map snapped positions to canonical vertex IDs
        let mut pos_to_id: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut next_id = 0usize;
        let canonical: Vec<usize> = mesh.positions.iter().map(|p| {
            let k = key(p);
            *pos_to_id.entry(k).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        }).collect();

        // Count directed edges
        let mut edge_counts: HashMap<(usize, usize), i32> = HashMap::new();
        let n_tris = mesh.indices.len() / 3;
        for i in 0..n_tris {
            let tri = [
                canonical[mesh.indices[i * 3] as usize],
                canonical[mesh.indices[i * 3 + 1] as usize],
                canonical[mesh.indices[i * 3 + 2] as usize],
            ];
            for j in 0..3 {
                let v0 = tri[j];
                let v1 = tri[(j + 1) % 3];
                *edge_counts.entry((v0, v1)).or_insert(0) += 1;
            }
        }

        // Every directed edge (a,b) should have a matching (b,a)
        for (&(a, b), &count) in &edge_counts {
            let reverse = edge_counts.get(&(b, a)).copied().unwrap_or(0);
            prop_assert!(
                reverse == count,
                "non-watertight: edge ({a},{b}) appears {count} times, reverse ({b},{a}) appears {reverse} times"
            );
        }
    }

    // 10. boolean fuse of two overlapping scaled cubes: volume = 1.5 * s^3
    #[test]
    fn prop_scale_invariant_boolean(s in 1.0f64..100.0) {
        let mut topo = Topology::new();
        let a = make_box(&mut topo, s, s, s).unwrap();
        let b = make_box(&mut topo, s, s, s).unwrap();

        // Translate b by s/2 along x
        let mat = Mat4::translation(s / 2.0, 0.0, 0.0);
        transform_solid(&mut topo, b, &mat).unwrap();

        let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let v_fuse = solid_volume(&topo, fused, 0.01).unwrap();

        let expected = 1.5 * s * s * s;
        let rel_err = (v_fuse - expected).abs() / expected;
        prop_assert!(rel_err < 0.01, "scale-invariant boolean failed: v_fuse={v_fuse}, expected={expected}, rel_err={rel_err}");
    }
}
