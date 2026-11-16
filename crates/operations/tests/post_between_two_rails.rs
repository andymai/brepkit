//! A post that crosses one rail and ends inside another, fused onto the two
//! rails at once (a stair's baluster between its two handrails).
//!
//! The thin rail passes through the post, so the post's wall takes two holes.
//! The post's top ends inside the thick rail, so a loop round the post's wall
//! cuts its top off. The wall splits into a band with the two holes, the band
//! above the loop, and the two discs the holes cut out.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const THIN: f64 = 12.0;
const THICK: f64 = 22.0;
const POST: f64 = 16.0;
const RAIL_LENGTH: f64 = 300.0;
const POST_BOTTOM: f64 = -60.0;
const RAIL_GAP: f64 = 100.0;

fn cylinder(topo: &mut Topology, radius: f64, height: f64, place: &Mat4) -> SolidId {
    let solid = make_cylinder(topo, radius, height).unwrap();
    transform_solid(topo, solid, place).unwrap();
    solid
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

/// The volume of the union from the cylinders' volumes less their overlaps:
/// the thin rail through the post, and the post's top in the thick rail.
fn union_volume() -> f64 {
    let pi = std::f64::consts::PI;
    let root = |r: f64, x: f64| (r * r - x * x).max(0.0).sqrt();
    let through = simpson(|x| 4.0 * root(THIN, x) * root(POST, x), -THIN, THIN);
    let top = simpson(|x| 2.0 * root(POST, x) * root(THICK, x), -POST, POST);
    pi * (THIN * THIN + THICK * THICK) * RAIL_LENGTH + pi * POST * POST * (RAIL_GAP - POST_BOTTOM)
        - through
        - top
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

#[test]
fn a_post_through_one_rail_into_another_fuses_onto_both() {
    let poses = [
        ("as built", Mat4::identity()),
        ("turned", Mat4::rotation_x(0.41) * Mat4::rotation_z(1.13)),
        (
            "shifted",
            Mat4::translation(2540.0, 150.0, 1170.0) * Mat4::rotation_x(-0.5),
        ),
        ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
    ];
    let along_y = Mat4::rotation_x(-std::f64::consts::FRAC_PI_2);
    let expected = union_volume();
    for (name, pose) in poses {
        let mut topo = Topology::new();
        let start = Mat4::translation(0.0, -RAIL_LENGTH / 2.0, 0.0);
        let thin = cylinder(&mut topo, THIN, RAIL_LENGTH, &(pose * start * along_y));
        let thick = cylinder(
            &mut topo,
            THICK,
            RAIL_LENGTH,
            &(pose * Mat4::translation(0.0, -RAIL_LENGTH / 2.0, RAIL_GAP) * along_y),
        );
        let post = cylinder(
            &mut topo,
            POST,
            RAIL_GAP - POST_BOTTOM,
            &(pose * Mat4::translation(0.0, 0.0, POST_BOTTOM)),
        );
        let rails = boolean::boolean(&mut topo, BooleanOp::Fuse, thick, thin).unwrap();
        let before = boolean::mesh_fallback_count();
        let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, rails, post).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "mesh fallback, {name}"
        );
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert_eq!(
            census(&topo, fused),
            BTreeMap::from([("cylinder", 4), ("plane", 5)]),
            "{name}"
        );
        assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
        let volume = solid_volume(&topo, fused, 0.005).unwrap();
        assert!(
            (volume - expected).abs() < 1e-6 * expected,
            "{name}: volume {volume} should be {expected}"
        );
        let class = |x: f64, y: f64, z: f64| {
            classify_point(
                &topo,
                fused,
                pose.mul_point(Point3::new(x, y, z)),
                0.01,
                1e-7,
            )
            .unwrap()
        };
        for (p, want) in [
            ((0.0, 0.0, 50.0), PointClassification::Inside),
            ((0.0, 0.0, -50.0), PointClassification::Inside),
            ((0.0, POST + 1.0, 0.0), PointClassification::Inside),
            ((0.0, -POST - 1.0, 0.0), PointClassification::Inside),
            (
                (0.0, 0.0, RAIL_GAP + THICK - 1.0),
                PointClassification::Inside,
            ),
            ((0.0, 0.0, POST_BOTTOM - 1.0), PointClassification::Outside),
            ((POST - 1.0, 0.0, THIN + 1.0), PointClassification::Inside),
            ((POST + 1.0, 0.0, THIN + 1.0), PointClassification::Outside),
            ((POST + 1.0, 20.0, 50.0), PointClassification::Outside),
        ] {
            assert_eq!(class(p.0, p.1, p.2), want, "{name} at {p:?}");
        }
    }
}

#[test]
fn a_post_through_a_hollow_rail_plugs_the_bore() {
    // The thin rail is a tube with a bore of radius `BORE`. The post's wall
    // takes the tube's outer loop and, inside it, the bore's loop on each
    // side: the wall keeps the disc inside the bore's loop and loses the ring
    // between the two loops.
    const BORE: f64 = 8.0;
    let poses = [
        ("as built", Mat4::identity()),
        ("turned", Mat4::rotation_x(0.41) * Mat4::rotation_z(1.13)),
        ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
    ];
    let along_y = Mat4::rotation_x(-std::f64::consts::FRAC_PI_2);
    let pi = std::f64::consts::PI;
    let root = |r: f64, x: f64| (r * r - x * x).max(0.0).sqrt();
    let through = |r: f64| simpson(|x| 4.0 * root(r, x) * root(POST, x), -r, r);
    let top = simpson(|x| 2.0 * root(POST, x) * root(THICK, x), -POST, POST);
    let expected = pi * (THIN * THIN - BORE * BORE + THICK * THICK) * RAIL_LENGTH
        + pi * POST * POST * (RAIL_GAP - POST_BOTTOM)
        - (through(THIN) - through(BORE))
        - top;
    for (name, pose) in poses {
        let mut topo = Topology::new();
        let start = pose * Mat4::translation(0.0, -RAIL_LENGTH / 2.0, 0.0) * along_y;
        let thin = cylinder(&mut topo, THIN, RAIL_LENGTH, &start);
        let bore = cylinder(&mut topo, BORE, RAIL_LENGTH, &start);
        let tube = boolean::boolean(&mut topo, BooleanOp::Cut, thin, bore).unwrap();
        let thick = cylinder(
            &mut topo,
            THICK,
            RAIL_LENGTH,
            &(pose * Mat4::translation(0.0, -RAIL_LENGTH / 2.0, RAIL_GAP) * along_y),
        );
        let post = cylinder(
            &mut topo,
            POST,
            RAIL_GAP - POST_BOTTOM,
            &(pose * Mat4::translation(0.0, 0.0, POST_BOTTOM)),
        );
        let rails = boolean::boolean(&mut topo, BooleanOp::Fuse, thick, tube).unwrap();
        let before = boolean::mesh_fallback_count();
        let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, rails, post).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "mesh fallback, {name}"
        );
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert_eq!(
            census(&topo, fused),
            BTreeMap::from([("cylinder", 8), ("plane", 5)]),
            "{name}"
        );
        assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
        let volume = solid_volume(&topo, fused, 0.005).unwrap();
        assert!(
            (volume - expected).abs() < 1e-6 * expected,
            "{name}: volume {volume} should be {expected}"
        );
        let class = |x: f64, y: f64, z: f64| {
            classify_point(
                &topo,
                fused,
                pose.mul_point(Point3::new(x, y, z)),
                0.01,
                1e-7,
            )
            .unwrap()
        };
        for (p, want) in [
            ((0.0, 0.0, 0.0), PointClassification::Inside),
            ((0.0, POST - 1.0, 0.0), PointClassification::Inside),
            ((0.0, POST + 1.0, 0.0), PointClassification::Outside),
            ((0.0, POST + 1.0, BORE + 1.0), PointClassification::Inside),
            ((0.0, 0.0, 50.0), PointClassification::Inside),
        ] {
            assert_eq!(class(p.0, p.1, p.2), want, "{name} at {p:?}");
        }
    }
}
