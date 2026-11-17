//! Two rods of a ball's radius that end at its centre, along -x and +y, with
//! the ball turned about x. Turned near a quarter turn, the ball's equator
//! lies within a hundredth of a radian of the second rod's end cap.
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The ball turned `spin` about x, a rod along -x and a rod along +y, both
/// of the ball's radius and ending at its centre, then `pose` on all three.
fn scene(topo: &mut Topology, spin: f64, pose: &Mat4) -> [SolidId; 3] {
    let ball = make_sphere(topo, 1.0, 32).unwrap();
    transform_solid(topo, ball, &Mat4::rotation_x(spin)).unwrap();
    let along_x = make_cylinder(topo, 1.0, 5.0).unwrap();
    transform_solid(
        topo,
        along_x,
        &(Mat4::rotation_z(PI / 2.0)
            * Mat4::rotation_x(PI / 2.0)
            * Mat4::translation(0.0, 0.0, -5.0)),
    )
    .unwrap();
    let along_y = make_cylinder(topo, 1.0, 5.0).unwrap();
    transform_solid(
        topo,
        along_y,
        &(Mat4::rotation_x(-PI / 2.0) * Mat4::rotation_z(-PI / 2.0)),
    )
    .unwrap();
    for s in [ball, along_x, along_y] {
        transform_solid(topo, s, pose).unwrap();
    }
    [ball, along_x, along_y]
}

/// Two rods of length 5, less the quarter of a bicylinder they share, and the
/// quarter of the ball outside both.
fn expected() -> f64 {
    10.0 * PI - 4.0 / 3.0 + PI / 3.0
}

fn turned() -> Mat4 {
    Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1)
}

#[test]
fn turned_mirrored_two_rods_stay_exact() {
    let pose = Mat4::scale(1.0, -1.0, 1.0) * turned();
    let mut topo = Topology::new();
    let [ball, a, b] = scene(&mut topo, 1.57, &pose);
    let opts = BooleanOptions {
        mesh_fallback: false,
        unify_faces: false,
        ..BooleanOptions::default()
    };
    let first = boolean::boolean_with_options(&mut topo, BooleanOp::Fuse, ball, a, opts).unwrap();
    let fused = boolean::boolean_with_options(&mut topo, BooleanOp::Fuse, first, b, opts).unwrap();
    assert!(validate_solid(&topo, fused).unwrap().is_valid());
    let v = solid_volume(&topo, fused, 0.0005).unwrap();
    assert!((v - expected()).abs() < 1e-6 * expected(), "volume {v}");
}

#[test]
fn two_rods_at_a_ball_mesh_closed() {
    for (name, spin, pose) in [
        ("1.56 upright", 1.56, Mat4::identity()),
        ("1.58 upright", 1.58, Mat4::identity()),
    ] {
        let mut topo = Topology::new();
        let [ball, a, b] = scene(&mut topo, spin, &pose);
        let first = boolean::boolean(&mut topo, BooleanOp::Fuse, ball, a).unwrap();
        let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, first, b).unwrap();
        assert!(validate_solid(&topo, fused).unwrap().is_valid(), "{name}");
        let mesh = tessellate_solid(&topo, fused, 0.005).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
    }
}

#[test]
fn a_bar_cut_through_two_rods_at_a_ball_stays_valid() {
    // Ball turned (0, 1.2, 0.4) about (x, y, z); the bar runs along y
    // through z = 0.8, x = 0.3 with radius 0.12.
    let pose = turned();
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, 1.0, 32).unwrap();
    transform_solid(
        &mut topo,
        ball,
        &(Mat4::rotation_z(0.4) * Mat4::rotation_y(1.2)),
    )
    .unwrap();
    let [_, a, b] = scene(&mut topo, 0.0, &Mat4::identity());
    let bar = make_cylinder(&mut topo, 0.12, 4.0).unwrap();
    transform_solid(
        &mut topo,
        bar,
        &(Mat4::rotation_x(-PI / 2.0)
            * Mat4::rotation_z(-PI / 2.0)
            * Mat4::translation(0.8, 0.3, -2.0)),
    )
    .unwrap();
    for s in [ball, a, b, bar] {
        transform_solid(&mut topo, s, &pose).unwrap();
    }
    let first = boolean::boolean(&mut topo, BooleanOp::Fuse, ball, a).unwrap();
    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, first, b).unwrap();
    let cut = boolean::boolean(&mut topo, BooleanOp::Cut, fused, bar).unwrap();
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, cut, 0.005).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "open mesh");
}
