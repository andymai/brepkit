//! A torus against a ball or a rod sharing its axis: the two meet in circles
//! about the axis, where their cross-sections in a half-plane through the axis
//! cross, and each boolean keeps an exact solid.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const RING: (f64, f64) = (4.0, 1.5);

/// Simpson over the tube's height of the area each slice keeps.
fn slices(keep: impl Fn(f64, f64, f64) -> f64) -> f64 {
    let (big, small) = RING;
    let n = 20_000;
    let h = 2.0 * small / f64::from(n);
    let slice = |z: f64| {
        let w = small.mul_add(small, -(z * z)).max(0.0).sqrt();
        keep(z, big - w, big + w)
    };
    let mut sum = slice(-small) + slice(small);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * slice(-small + h * f64::from(k));
    }
    sum * h / 3.0
}

fn check(topo: &Topology, piece: SolidId, faces: usize, truth: f64, label: &str) {
    let report = validate_solid(topo, piece).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    assert_eq!(
        solid_faces(topo, piece).unwrap().len(),
        faces,
        "{label}: faces"
    );
    let mesh = tessellate_solid(topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let meshed = oriented_solid_volume(topo, piece, 0.005).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-2 * truth,
        "{label}: mesh volume {meshed}, truth {truth}"
    );
}

fn at(topo: &Topology, piece: SolidId, x: f64, z: f64) -> PointClassification {
    classify_point(
        topo,
        piece,
        Point3::new(x, 0.0, z),
        &ClassifyOptions::default(),
    )
    .unwrap()
}

/// A ball of radius 3 in the ring's hole reaches into the tube's inner side.
#[test]
fn ball_in_a_rings_hole() {
    let (big, small) = RING;
    let ball = 3.0_f64;
    let inside = slices(|z, lo, hi| {
        let reach = ball.mul_add(ball, -(z * z)).max(0.0).sqrt();
        let hi = hi.min(reach);
        if hi > lo {
            PI * (hi * hi - lo * lo)
        } else {
            0.0
        }
    });
    let ring = 2.0 * PI * PI * big * small * small;
    let sphere = 4.0 / 3.0 * PI * ball.powi(3);
    for (op, truth) in [
        (BooleanOp::Fuse, ring + sphere - inside),
        (BooleanOp::Cut, ring - inside),
        (BooleanOp::Intersect, inside),
    ] {
        let label = format!("{op:?}");
        let mut topo = Topology::new();
        let a = make_torus(&mut topo, big, small, 32).unwrap();
        let b = make_sphere(&mut topo, ball, 32).unwrap();
        let piece = boolean(&mut topo, op, a, b).unwrap();
        check(&topo, piece, 3, truth, &label);
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        // The ball's chordal equator leaves a zone of its surface measured
        // short of exact; the lens inside both is exact.
        let bound = if op == BooleanOp::Intersect {
            1e-6
        } else {
            1e-3
        };
        assert!(
            (volume - truth).abs() < bound * truth,
            "{label}: volume {volume}, truth {truth}"
        );
        // In the tube past the ball, and in the ball's middle.
        let (far, middle) = match op {
            BooleanOp::Fuse => (PointClassification::Inside, PointClassification::Inside),
            BooleanOp::Cut => (PointClassification::Inside, PointClassification::Outside),
            BooleanOp::Intersect => (PointClassification::Outside, PointClassification::Outside),
        };
        assert_eq!(at(&topo, piece, 5.0, 0.0), far, "{label}: far side");
        assert_eq!(at(&topo, piece, 0.0, 0.0), middle, "{label}: middle");
        // Off the ball's equator plane, where its hemispheres meet on chords.
        assert_eq!(
            at(&topo, piece, 2.8, 0.3),
            if op == BooleanOp::Cut {
                PointClassification::Outside
            } else {
                PointClassification::Inside
            },
            "{label}: tube inside the ball"
        );
    }
}

/// A rod of radius 4.2 through the ring's hole cuts its tube along two
/// circles; the tube's part past the wall is a disc segment revolved.
#[test]
fn rod_through_a_rings_tube() {
    let (big, small) = RING;
    let rod = 4.2_f64;
    let gap = rod - big;
    let segment =
        small * small * (gap / small).acos() - gap * small.mul_add(small, -(gap * gap)).sqrt();
    let reach = 2.0 / 3.0 * small.mul_add(small, -(gap * gap)).powf(1.5) / segment;
    let outside = 2.0 * PI * segment * (big + reach);
    let ring = 2.0 * PI * PI * big * small * small;
    let cylinder = PI * rod * rod * 10.0;
    for (op, faces, truth) in [
        (BooleanOp::Fuse, 5, cylinder + outside),
        (BooleanOp::Cut, 2, outside),
        (BooleanOp::Intersect, 2, ring - outside),
    ] {
        let label = format!("{op:?}");
        let mut topo = Topology::new();
        let a = make_torus(&mut topo, big, small, 32).unwrap();
        let b = make_cylinder(&mut topo, rod, 10.0).unwrap();
        transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, -5.0)).unwrap();
        let piece = boolean(&mut topo, op, a, b).unwrap();
        check(&topo, piece, faces, truth, &label);
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "{label}: volume {volume}, truth {truth}"
        );
        let past_wall = if op == BooleanOp::Intersect {
            PointClassification::Outside
        } else {
            PointClassification::Inside
        };
        assert_eq!(
            at(&topo, piece, 5.0, 0.0),
            past_wall,
            "{label}: past the wall"
        );
    }
}

/// A rod of radius 2 stands in the ring's hole without touching it.
#[test]
fn rod_in_a_rings_hole() {
    let (big, small) = RING;
    let ring = 2.0 * PI * PI * big * small * small;
    let rod = PI * 4.0 * 10.0;
    let mut topo = Topology::new();
    let a = make_torus(&mut topo, big, small, 32).unwrap();
    let b = make_cylinder(&mut topo, 2.0, 10.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, -5.0)).unwrap();
    let both = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    let volume = solid_volume(&topo, both, 0.01).unwrap();
    assert!(
        (volume - ring - rod).abs() < 1e-9 * (ring + rod),
        "fuse: volume {volume}"
    );
    assert_eq!(at(&topo, both, big, 0.0), PointClassification::Inside);
    assert_eq!(at(&topo, both, 0.0, 0.0), PointClassification::Inside);

    let mut topo = Topology::new();
    let a = make_torus(&mut topo, big, small, 32).unwrap();
    let b = make_cylinder(&mut topo, 2.0, 10.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, -5.0)).unwrap();
    let rest = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
    assert_eq!(solid_faces(&topo, rest).unwrap().len(), 1);
    let volume = solid_volume(&topo, rest, 0.01).unwrap();
    assert!((volume - ring).abs() < 1e-9 * ring, "cut: volume {volume}");
}
