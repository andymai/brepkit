//! A box holed by a cylinder, cut or intersected by a plane that crosses
//! the hole: the plane's face takes the box's outline as one loop and the
//! hole's rim as another inside it. The outline's lines came with no UVs and
//! a placeholder pcurve, so the loop was laid out at the frame's origin, the
//! rim was handed to the face outside the outline, and the ring between them
//! was never built; the boolean fell back to a mesh.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn poses() -> [Mat4; 3] {
    [
        Mat4::identity(),
        Mat4::translation(4.0, -9.0, 2.0) * Mat4::rotation_x(90.0_f64.to_radians()),
        Mat4::translation(7.0, -3.0, 11.0)
            * Mat4::rotation_z(30.0_f64.to_radians())
            * Mat4::rotation_x(50.0_f64.to_radians()),
    ]
}

fn placed_box(topo: &mut Topology, d: [f64; 3], at: [f64; 3], pose: Mat4) -> SolidId {
    let b = make_box(topo, d[0], d[1], d[2]).unwrap();
    transform_solid(topo, b, &(pose * Mat4::translation(at[0], at[1], at[2]))).unwrap();
    b
}

fn placed_cylinder(topo: &mut Topology, r: f64, h: f64, at: [f64; 3], pose: Mat4) -> SolidId {
    let c = make_cylinder(topo, r, h).unwrap();
    transform_solid(topo, c, &(pose * Mat4::translation(at[0], at[1], at[2]))).unwrap();
    c
}

fn check(
    topo: &Topology,
    s: SolidId,
    pose: Mat4,
    want: f64,
    inside: &[[f64; 3]],
    outside: &[[f64; 3]],
) {
    assert_eq!(solid_faces(topo, s).unwrap().len(), 7);
    let report = validate_solid(topo, s).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let volume = solid_volume(topo, s, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-9 * want,
        "volume {volume}, want {want}"
    );
    let at = |p: &[f64; 3]| pose.mul_point(Point3::new(p[0], p[1], p[2]));
    let classify = |p| classify_point(topo, s, p, &ClassifyOptions::default()).unwrap();
    for p in inside {
        assert_eq!(classify(at(p)), PointClassification::Inside, "{p:?}");
    }
    for p in outside {
        assert_eq!(classify(at(p)), PointClassification::Outside, "{p:?}");
    }
}

#[test]
fn a_plane_across_a_through_hole_cuts_it_exactly() {
    for (k, pose) in poses().into_iter().enumerate() {
        for (hole, z) in [([30.0, 20.0], 10.0), ([12.0, 9.0], 7.0)] {
            let mut topo = Topology::new();
            let b = placed_box(&mut topo, [60.0, 40.0, 20.0], [0.0; 3], pose);
            let c = placed_cylinder(&mut topo, 5.0, 40.0, [hole[0], hole[1], -10.0], pose);
            let holed = boolean(&mut topo, BooleanOp::Cut, b, c).unwrap();
            let top = placed_box(&mut topo, [200.0, 200.0, 100.0], [-100.0, -100.0, z], pose);
            let before = mesh_fallback_count();
            let cut = boolean(&mut topo, BooleanOp::Cut, holed, top).unwrap();
            assert_eq!(mesh_fallback_count(), before, "pose {k}, hole {hole:?}");
            check(
                &topo,
                cut,
                pose,
                (2400.0 - PI * 25.0) * z,
                &[[59.0, 1.0, z - 1.0], [hole[0] + 6.0, hole[1], 1.0]],
                &[
                    [hole[0], hole[1], z / 2.0],
                    [59.0, 1.0, z + 1.0],
                    [hole[0] + 4.0, hole[1], z / 2.0],
                ],
            );
        }
    }
}

#[test]
fn a_slab_on_a_blind_holes_floor_keeps_the_hole_through() {
    for (k, pose) in poses().into_iter().enumerate() {
        let mut topo = Topology::new();
        let b = placed_box(&mut topo, [40.0, 30.0, 20.0], [0.0; 3], pose);
        let c = placed_cylinder(&mut topo, 2.5, 12.5, [20.0, 15.0, 10.0], pose);
        let holed = boolean(&mut topo, BooleanOp::Cut, b, c).unwrap();
        let slab = placed_box(
            &mut topo,
            [200.0, 200.0, 100.0],
            [-100.0, -100.0, 10.0],
            pose,
        );
        let before = mesh_fallback_count();
        let kept = boolean(&mut topo, BooleanOp::Intersect, holed, slab).unwrap();
        assert_eq!(mesh_fallback_count(), before, "pose {k}");
        check(
            &topo,
            kept,
            pose,
            12000.0 - PI * 6.25 * 10.0,
            &[[1.0, 1.0, 15.0], [23.0, 15.0, 11.0]],
            &[[20.0, 15.0, 15.0], [20.0, 15.0, 9.0], [1.0, 1.0, 9.0]],
        );
    }
}
