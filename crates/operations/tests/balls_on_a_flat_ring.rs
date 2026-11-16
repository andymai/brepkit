//! A flat ring with balls across it (a bearing cage's band and its ball
//! pockets): each ball crosses the ring's two walls and its top and bottom.
//!
//! The walls' sections with a ball run from the ring's top to its bottom, and
//! the rims cross the ball between their samples, so each rim takes a vertex
//! where it crosses the ball's face, and the sections end on it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_sphere;
use brepkit_operations::revolve::revolve;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::compound::Compound;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const INNER: f64 = 5.55;
const OUTER: f64 = 9.15;
const HALF: f64 = 0.35;
const BALL: f64 = 2.4145;
const PITCH: f64 = 7.35;

/// The ring: a rectangle in the `xz` plane turned about `z`.
fn ring(topo: &mut Topology) -> SolidId {
    let corners = [
        Point3::new(INNER, 0.0, -HALF),
        Point3::new(OUTER, 0.0, -HALF),
        Point3::new(OUTER, 0.0, HALF),
        Point3::new(INNER, 0.0, HALF),
    ];
    let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    revolve(
        topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        TAU,
    )
    .unwrap()
}

/// A ball on the pitch circle, `turn` round from the `x` axis.
fn ball(topo: &mut Topology, turn: f64) -> SolidId {
    let ball = make_sphere(topo, BALL, 32).unwrap();
    transform_solid(
        topo,
        ball,
        &(Mat4::rotation_z(turn) * Mat4::translation(PITCH, 0.0, 0.0)),
    )
    .unwrap();
    ball
}

/// The area two circles of radii `a` and `b`, `d` apart, share.
fn lens(a: f64, b: f64, d: f64) -> f64 {
    if d >= a + b {
        return 0.0;
    }
    if d <= (a - b).abs() {
        return PI * a.min(b).powi(2);
    }
    let p = a * a * ((d * d + a * a - b * b) / (2.0 * d * a)).acos();
    let q = b * b * ((d * d + b * b - a * a) / (2.0 * d * b)).acos();
    let r = 0.5 * ((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)).sqrt();
    p + q - r
}

/// The volume one ball shares with the ring: its slices through the ring's
/// height, each a disc less what lies inside the ring's hole.
fn shared() -> f64 {
    let slice = |z: f64| {
        let s = (BALL * BALL - z * z).sqrt();
        lens(s, OUTER, PITCH) - lens(s, INNER, PITCH)
    };
    let n = 20_000;
    let h = 2.0 * HALF / f64::from(n);
    let mut sum = slice(-HALF) + slice(HALF);
    for i in 1..n {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * slice(h.mul_add(f64::from(i), -HALF));
    }
    sum * h / 3.0
}

fn union_volume(balls: u32) -> f64 {
    let ring = PI * (OUTER * OUTER - INNER * INNER) * 2.0 * HALF;
    let ball = 4.0 / 3.0 * PI * BALL.powi(3);
    ring + f64::from(balls) * (ball - shared())
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

fn poses() -> [(&'static str, Mat4); 4] {
    [
        ("as built", Mat4::identity()),
        ("turned", Mat4::rotation_x(0.41) * Mat4::rotation_z(1.13)),
        (
            "shifted",
            Mat4::translation(300.0, -7.3, 0.123) * Mat4::rotation_y(0.3),
        ),
        ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
    ]
}

#[test]
fn a_ball_across_a_flat_ring_fuses_onto_it() {
    let expected = union_volume(1);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let band = ring(&mut topo);
        let one = ball(&mut topo, 0.0);
        transform_solid(&mut topo, band, &pose).unwrap();
        transform_solid(&mut topo, one, &pose).unwrap();
        let before = boolean::mesh_fallback_count();
        let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, band, one).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "mesh fallback, {name}"
        );
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert_eq!(
            census(&topo, fused),
            BTreeMap::from([("cylinder", 2), ("plane", 2), ("sphere", 2)]),
            "{name}"
        );
        assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
        let volume = solid_volume(&topo, fused, 0.001).unwrap();
        assert!(
            (volume - expected).abs() < 1e-5 * expected,
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
            ((PITCH, 0.0, 2.0), PointClassification::Inside),
            ((PITCH, 0.0, -2.0), PointClassification::Inside),
            ((PITCH + 2.2, 0.0, 0.0), PointClassification::Inside),
            ((PITCH - 2.2, 0.0, 0.0), PointClassification::Inside),
            ((0.0, 7.35, 0.0), PointClassification::Inside),
            ((0.0, 7.35, 0.5), PointClassification::Outside),
            ((PITCH, 0.0, 2.5), PointClassification::Outside),
            ((PITCH, 2.3, 0.9), PointClassification::Outside),
            ((0.0, 0.0, 0.0), PointClassification::Outside),
        ] {
            assert_eq!(class(p.0, p.1, p.2), want, "{name} at {p:?}");
        }
    }
}

#[test]
fn seven_balls_round_a_flat_ring_fuse_in_one() {
    // A 608 bearing's cage band and the seven pockets round it.
    let expected = union_volume(7);
    let mut topo = Topology::new();
    let mut solids = vec![ring(&mut topo)];
    for i in 0..7 {
        solids.push(ball(&mut topo, TAU * f64::from(i) / 7.0));
    }
    let compound = topo.add_compound(Compound::new(solids));
    let before = boolean::mesh_fallback_count();
    let fused = fuse_all(&mut topo, compound).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        census(&topo, fused),
        BTreeMap::from([("cylinder", 14), ("plane", 14), ("sphere", 14)])
    );
    assert_eq!(edges_not_used_twice(&topo, fused), 0);
    let volume = solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (volume - expected).abs() < 1e-5 * expected,
        "volume {volume} should be {expected}"
    );
}
