//! The check crate's integrator bounds and clips a curved face's `(u, v)`
//! from points along its wire. A wire through a sphere's pole or a cone's
//! apex, where `u` is arbitrary, keeps the face's own range.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_check::properties::{PropertiesOptions, solid_volume};
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cone, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

/// A unit ball less its positive octant, as made and turned four ways: the
/// wall keeping 270 degrees above the equator runs down a meridian from the
/// pole and up another, and reads its volume `7π/6`.
#[test]
fn a_ball_less_an_octant_reads_its_volume() {
    let truth = 7.0 * PI / 6.0;
    for (name, pose) in [
        ("as made", Mat4::identity()),
        (
            "turned 37 degrees about z",
            Mat4::rotation_z(37f64.to_radians()),
        ),
        (
            "turned 200 degrees about z",
            Mat4::rotation_z(200f64.to_radians()),
        ),
        ("turned over about x", Mat4::rotation_x(PI)),
        ("turned a quarter about y", Mat4::rotation_y(FRAC_PI_2)),
    ] {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 1.0, 16).unwrap();
        let corner = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let rest = boolean(&mut topo, BooleanOp::Cut, ball, corner).unwrap();
        transform_solid(&mut topo, rest, &pose).unwrap();
        let volume = solid_volume(&topo, rest, &PropertiesOptions::default()).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{name}: volume {volume}, truth {truth}"
        );
    }
}

/// A pointed cone less the quadrant `x, y > 0`: its wall runs from the apex
/// down one cut and up the other, and reads its volume `π / 2`.
#[test]
fn a_cone_less_a_quadrant_reads_its_volume() {
    let truth = PI / 2.0;
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 1.0, 0.0, 2.0).unwrap();
    let quadrant = make_box(&mut topo, 2.0, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, quadrant, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let rest = boolean(&mut topo, BooleanOp::Cut, cone, quadrant).unwrap();
    let volume = solid_volume(&topo, rest, &PropertiesOptions::default()).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "volume {volume}, truth {truth}"
    );
}
