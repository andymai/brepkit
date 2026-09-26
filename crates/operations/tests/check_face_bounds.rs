//! The check crate's integrator bounds a curved face's `(u, v)` from points
//! along its wire. A wire through a sphere's pole, where `u` is arbitrary,
//! keeps the face's own range.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_check::properties::{PropertiesOptions, solid_volume};
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

/// A unit ball less its positive octant, turned four ways: the wall keeping
/// 270 degrees above the equator runs down a meridian from the pole and up
/// another, and reads its volume `7π/6`.
#[test]
fn a_turned_ball_less_an_octant_reads_its_volume() {
    let truth = 7.0 * PI / 6.0;
    for (name, pose) in [
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
