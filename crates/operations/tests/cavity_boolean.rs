//! A cube with a ball-shaped cavity against boxes: the cavity is an inner
//! shell, and every boolean keeps the part of it the result holds.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 3.0;

/// The cube `|x|, |y|, |z| < 10` less the ball of radius 3 at its centre.
fn hollow_cube(topo: &mut Topology) -> SolidId {
    let cube = make_box(topo, 20.0, 20.0, 20.0).unwrap();
    transform_solid(topo, cube, &Mat4::translation(-10.0, -10.0, -10.0)).unwrap();
    let ball = make_sphere(topo, RADIUS, 32).unwrap();
    boolean(topo, BooleanOp::Cut, cube, ball).unwrap()
}

/// The ball's part above `z = z0`.
fn ball_above(z0: f64) -> f64 {
    let h = (RADIUS - z0).clamp(0.0, 2.0 * RADIUS);
    PI * h * h * (3.0 * RADIUS - h) / 3.0
}

/// Checks a result is exact (a handful of faces, not a mesh), valid, and
/// within `1e-6` of `truth`.
fn check(topo: &Topology, result: SolidId, truth: f64, label: &str) {
    let faces = solid_faces(topo, result).unwrap().len();
    assert!(faces <= 16, "{label}: fell back to a mesh ({faces} faces)");
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let volume = solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
}

/// The hollow cube against the box over `z < z0`, from below the cavity,
/// through it (its equator plane included), to past it, in every op: each
/// result is exact, valid and within `1e-6` of the volumes the cavity's
/// part above `z0` leaves. The cube is a box by its outer shell, and a
/// classifier built from that alone read the cavity as solid (every op took
/// the box-pair shortcut); the cavity's shell is a hole whether its faces
/// have a corner fan or not (a cap and its disc, or the two hemispheres
/// cornered only on their equator); and the two hemispheres, the halves of
/// one sphere sharing every edge, are not a doubled face.
#[test]
fn a_cavity_survives_booleans_with_a_box_below_a_plane() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    for z0 in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.5] {
        let above = ball_above(z0);
        let below = ball - above;
        let slab = 400.0 * (z0 + 10.0);
        for (op, swapped, truth) in [
            (BooleanOp::Intersect, false, slab - below),
            (BooleanOp::Cut, false, 8000.0 - slab - above),
            (BooleanOp::Cut, true, below),
            (BooleanOp::Fuse, false, 8000.0 - above),
        ] {
            if truth < 1e-9 {
                continue;
            }
            let label = format!("z < {z0}: {op:?}{}", if swapped { " swapped" } else { "" });
            let mut topo = Topology::new();
            let hollow = hollow_cube(&mut topo);
            let other = make_box(&mut topo, 20.0, 20.0, z0 + 10.0).unwrap();
            transform_solid(&mut topo, other, &Mat4::translation(-10.0, -10.0, -10.0)).unwrap();
            let result = if swapped {
                boolean(&mut topo, op, other, hollow)
            } else {
                boolean(&mut topo, op, hollow, other)
            }
            .unwrap();
            check(&topo, result, truth, &label);
        }
    }
}

/// The hollow cube against the same cube shifted 1 along x, which holds the
/// whole cavity, in every op: each result is exact, valid, within `1e-6` of
/// its volume, and reads a point beside the cavity's centre on the right
/// side.
#[test]
fn a_cavity_survives_booleans_with_a_box_holding_it() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    for (op, swapped, truth, centre) in [
        (
            BooleanOp::Intersect,
            false,
            7600.0 - ball,
            PointClassification::Outside,
        ),
        (BooleanOp::Cut, false, 400.0, PointClassification::Outside),
        (
            BooleanOp::Cut,
            true,
            400.0 + ball,
            PointClassification::Inside,
        ),
        (BooleanOp::Fuse, false, 8400.0, PointClassification::Inside),
    ] {
        let label = format!("{op:?}{}", if swapped { " swapped" } else { "" });
        let mut topo = Topology::new();
        let hollow = hollow_cube(&mut topo);
        let other = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(&mut topo, other, &Mat4::translation(-9.0, -10.0, -10.0)).unwrap();
        let result = if swapped {
            boolean(&mut topo, op, other, hollow)
        } else {
            boolean(&mut topo, op, hollow, other)
        }
        .unwrap();
        check(&topo, result, truth, &label);
        let got = classify_point(
            &topo,
            result,
            Point3::new(0.3, 0.2, -1.0),
            &ClassifyOptions::default(),
        )
        .unwrap();
        assert_eq!(got, centre, "{label}: beside the cavity's centre");
    }
}
