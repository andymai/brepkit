//! A rectangular pocket cut into the side of a pointed cone.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// The r=3, h=6 cone less its part of the box x -0.5..0.5, y 1..5,
/// z 1..2: across the box's width the cone's chord at height z (radius
/// `3 - z/2`) integrates in closed form, and Simpson's rule takes the
/// height.
fn truth() -> f64 {
    const N: u32 = 2000;
    let across = |z: f64| {
        let (r, a) = (3.0 - z / 2.0, 0.5);
        a * (r * r - a * a).sqrt() + r * r * (a / r).asin() - 2.0 * a
    };
    let h = 1.0 / f64::from(N);
    let mut sum = across(1.0) + across(2.0);
    for k in 1..N {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * across(1.0 + h * f64::from(k));
    }
    PI * 9.0 * 6.0 / 3.0 - sum * h / 3.0
}

/// The pocket turned `turn` about the cone's axis.
fn pocket(turn: f64) {
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 3.0, 0.0, 6.0).unwrap();
    let block = make_box(&mut topo, 1.0, 4.0, 1.0).unwrap();
    let place = Mat4::rotation_z(turn) * Mat4::translation(-0.5, 1.0, 1.0);
    transform_solid(&mut topo, block, &place).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, cone, block).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mut census: BTreeMap<&str, usize> = BTreeMap::new();
    for face in solid_faces(&topo, result).unwrap() {
        *census
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(census, BTreeMap::from([("cone", 1), ("plane", 6)]));

    // The wall's normal makes a fixed angle with the axis, so the pocket
    // takes sqrt(5) times its footprint on the base: the part of the strip
    // |x| <= 0.5 between the radii 2 and 2.5 where the wall is 1 to 2 high.
    let strip = |r: f64| 0.5 * (r * r - 0.25).sqrt() + r * r * (0.5 / r).asin();
    let wall_truth = 5.0_f64.sqrt() * (9.0 * PI - strip(2.5) + strip(2.0));
    let wall = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .find(|&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .unwrap();
    let area = face_area(&topo, wall, 0.001).unwrap();
    assert!(
        (area - wall_truth).abs() < 1e-9 * wall_truth,
        "wall area {area}, truth {wall_truth}"
    );

    let volume = truth();
    let exact = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (exact - volume).abs() < 1e-7 * volume,
        "volume {exact}, truth {volume}"
    );
    assert!(is_watertight(
        &tessellate_solid(&topo, result, 0.01).unwrap()
    ));
    let meshed = oriented_solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (meshed - volume).abs() < 1e-3 * volume,
        "mesh volume {meshed}, truth {volume}"
    );

    let at = |x: f64, y: f64, z: f64| Mat4::rotation_z(turn).mul_point(Point3::new(x, y, z));
    let classify =
        |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
    assert_eq!(classify(at(0.0, 1.8, 1.5)), PointClassification::Outside);
    for kept in [at(0.0, 0.5, 1.5), at(0.0, 1.5, 2.5), at(0.8, 1.5, 1.5)] {
        assert_eq!(classify(kept), PointClassification::Inside, "{kept:?}");
    }
}

#[test]
fn pocket_into_a_pointed_cone() {
    pocket(0.0);
}

#[test]
fn pocket_into_a_pointed_cone_turned() {
    pocket(FRAC_PI_2);
}
