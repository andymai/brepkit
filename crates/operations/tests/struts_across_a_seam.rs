//! Two long struts joined by two posts into a ring, where each post enters a
//! long strut's wall across that wall's seam.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const LONG: f64 = 25.0;
const POST: f64 = 8.0;
const LONG_START: f64 = 55.0;
const LONG_LENGTH: f64 = 390.0;
const HALF_GAP: f64 = 120.0;
const POSTS: [f64; 2] = [70.0, 430.0];

fn cylinder(topo: &mut Topology, radius: f64, height: f64, place: &Mat4) -> SolidId {
    let solid = make_cylinder(topo, radius, height).unwrap();
    transform_solid(topo, solid, place).unwrap();
    solid
}

fn fuse(topo: &mut Topology, a: SolidId, b: SolidId) -> SolidId {
    let options = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    boolean_with_options(topo, BooleanOp::Fuse, a, b, options).unwrap()
}

fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let n = 200_000;
    let h = (b - a) / f64::from(n);
    let mut sum = f(a) + f(b);
    for i in 1..n {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(h.mul_add(f64::from(i), a));
    }
    sum * h / 3.0
}

/// Each post ends on the axis of each long strut, so each post loses half of
/// its crossing with each long strut.
fn union_volume() -> f64 {
    let pi = std::f64::consts::PI;
    let root = |r: f64, x: f64| (r * r - x * x).max(0.0).sqrt();
    let half_crossing = simpson(|y| 2.0 * root(POST, y) * root(LONG, y), -POST, POST);
    2.0 * pi * LONG * LONG * LONG_LENGTH + 2.0 * (pi * POST * POST * 2.0 * HALF_GAP)
        - 4.0 * half_crossing
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

fn edges_not_used_twice(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

/// The long struts along `x`, turned `spin` about their own axes, and the two
/// posts along `z` from the lower strut's axis to the upper strut's axis.
fn ring(topo: &mut Topology, pose: &Mat4, spin: f64) -> [SolidId; 4] {
    let along_x = Mat4::rotation_x(spin) * Mat4::rotation_y(FRAC_PI_2);
    let upper = cylinder(
        topo,
        LONG,
        LONG_LENGTH,
        &(*pose * Mat4::translation(LONG_START, 0.0, HALF_GAP) * along_x),
    );
    let lower = cylinder(
        topo,
        LONG,
        LONG_LENGTH,
        &(*pose * Mat4::translation(LONG_START, 0.0, -HALF_GAP) * along_x),
    );
    let first = cylinder(
        topo,
        POST,
        2.0 * HALF_GAP,
        &(*pose * Mat4::translation(POSTS[0], 0.0, -HALF_GAP)),
    );
    let second = cylinder(
        topo,
        POST,
        2.0 * HALF_GAP,
        &(*pose * Mat4::translation(POSTS[1], 0.0, -HALF_GAP)),
    );
    [upper, lower, first, second]
}

fn check(topo: &Topology, fused: SolidId, pose: &Mat4, name: &str) {
    let report = validate_solid(topo, fused).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    assert_eq!(edges_not_used_twice(topo, fused), 0, "{name}");
    assert_eq!(
        census(topo, fused),
        BTreeMap::from([("cylinder", 4), ("plane", 4)]),
        "{name}"
    );
    let expected = union_volume();
    let volume = solid_volume(topo, fused, 0.005).unwrap();
    assert!(
        (volume - expected).abs() < 1e-6 * expected,
        "{name}: volume {volume} should be {expected}"
    );
    let class = |x: f64, y: f64, z: f64| {
        classify_point(
            topo,
            fused,
            pose.mul_point(Point3::new(x, y, z)),
            0.01,
            1e-7,
        )
        .unwrap()
    };
    for post in POSTS {
        for (p, want) in [
            ((post, 0.0, 0.0), PointClassification::Inside),
            (
                (post, 0.0, HALF_GAP - LONG - 1.0),
                PointClassification::Inside,
            ),
            (
                (post, 0.0, -HALF_GAP + LONG + 1.0),
                PointClassification::Inside,
            ),
            ((post + POST + 1.0, 0.0, 0.0), PointClassification::Outside),
            ((post, POST + 1.0, 0.0), PointClassification::Outside),
            (
                (post, 0.0, HALF_GAP + LONG - 1.0),
                PointClassification::Inside,
            ),
            (
                (post, 0.0, HALF_GAP + LONG + 1.0),
                PointClassification::Outside,
            ),
        ] {
            assert_eq!(class(p.0, p.1, p.2), want, "{name} at {p:?}");
        }
    }
    assert_eq!(
        class(250.0, 0.0, 0.0),
        PointClassification::Outside,
        "{name}"
    );
}

fn poses() -> [(&'static str, Mat4); 4] {
    [
        ("as built", Mat4::identity()),
        ("turned", Mat4::rotation_x(0.41) * Mat4::rotation_z(1.13)),
        (
            "shifted",
            Mat4::translation(2540.0, 150.0, 1170.0) * Mat4::rotation_x(-0.5),
        ),
        ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
    ]
}

#[test]
fn a_second_post_across_a_wall_seam_fuses_exactly() {
    for (name, pose) in poses() {
        for spin in [0.0, FRAC_PI_2, PI] {
            let name = format!("{name}, spin {spin}");
            let mut topo = Topology::new();
            let [upper, lower, first, second] = ring(&mut topo, &pose, spin);
            let longs = fuse(&mut topo, upper, lower);
            let one = fuse(&mut topo, longs, first);
            let fused = fuse(&mut topo, one, second);
            check(&topo, fused, &pose, &name);
        }
    }
}

#[test]
fn a_second_post_across_a_wall_seam_fuses_exactly_in_the_engine() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let [upper, lower, first, second] = ring(&mut topo, &pose, 0.0);
        let longs = fuse(&mut topo, upper, lower);
        let one = fuse(&mut topo, longs, first);
        let fused =
            brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, one, second)
                .unwrap();
        check(&topo, fused, &pose, name);
    }
}
