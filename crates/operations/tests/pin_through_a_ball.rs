//! A tapered pin through a ball off the ball's centre, upright, tilted and
//! laid across the ball's equator: in every pose the Cut and the Intersect
//! are exact and take what the pin holds of the ball.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const BALL: f64 = 3.0;
/// The pin: radius 1.2 at its base tapering to 0.4 over its length of 10.
const BASE: f64 = 1.2;
const TIP: f64 = 0.4;
const LENGTH: f64 = 10.0;

/// The area two coplanar discs of radii `r1` and `r2`, `d` apart, share.
fn lens(r1: f64, r2: f64, d: f64) -> f64 {
    if r1 <= 0.0 || r2 <= 0.0 || d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return PI * r1.min(r2).powi(2);
    }
    let a = r1 * r1 * ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos();
    let b = r2 * r2 * ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos();
    let c = 0.5 * ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();
    a + b - c
}

/// The ball's part inside the pin from `base` along the unit `axis`: across
/// the axis, the ball's section and the pin's are coplanar discs whose
/// centres lie the ball centre's distance from the axis apart.
fn inside(base: Point3, axis: Vec3) -> f64 {
    let to_centre = Point3::new(0.0, 0.0, 0.0) - base;
    let along = to_centre.dot(axis);
    let d = (to_centre - axis * along).length();
    let n = 200_000;
    let h = LENGTH / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let s = h * f64::from(i);
        let ball = (BALL * BALL - (s - along).powi(2)).max(0.0).sqrt();
        let pin = (TIP - BASE).mul_add(s / LENGTH, BASE);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * lens(ball, pin, d);
    }
    total * h / 3.0
}

#[test]
fn a_pin_through_a_ball_is_exact() {
    let turn = Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3);
    let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
    let ball = 4.0 / 3.0 * PI * BALL.powi(3);
    for (name, place) in [
        ("upright", Mat4::translation(1.0, 0.5, -5.0)),
        (
            "tilted",
            Mat4::translation(0.5, 0.3, 0.0)
                * Mat4::rotation_x(0.6)
                * Mat4::rotation_y(0.4)
                * Mat4::translation(0.0, 0.0, -5.0),
        ),
        (
            "across",
            Mat4::translation(-5.0, 0.0, 0.3) * Mat4::rotation_y(PI / 2.0),
        ),
    ] {
        let base = place.mul_point(Point3::new(0.0, 0.0, 0.0));
        let axis = (place.mul_point(Point3::new(0.0, 0.0, 1.0)) - base)
            .normalize()
            .unwrap();
        let taken = inside(base, axis);
        // On the axis where it passes nearest the ball's centre, and in the
        // ball two from the centre, square to the axis and to that point:
        // at least two from the axis, past the pin's radius.
        let centre = Point3::new(0.0, 0.0, 0.0);
        let foot = base + axis * (centre - base).dot(axis);
        let away = centre + axis.cross(foot - centre).normalize().unwrap() * 2.0;
        for pose in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth) in [
                (BooleanOp::Cut, ball - taken),
                (BooleanOp::Intersect, taken),
            ] {
                let label = format!("{name} {pose} {op:?}");
                let mut topo = Topology::new();
                let mut a = make_sphere(&mut topo, BALL, 32).unwrap();
                let mut b = make_cone(&mut topo, BASE, TIP, LENGTH).unwrap();
                transform_solid(&mut topo, b, &place).unwrap();
                match pose {
                    "turned" => {
                        transform_solid(&mut topo, a, &turn).unwrap();
                        transform_solid(&mut topo, b, &turn).unwrap();
                    }
                    "mirrored" => {
                        a = mirror(&mut topo, a, at, normal).unwrap();
                        b = mirror(&mut topo, b, at, normal).unwrap();
                    }
                    "scaled" => {
                        let flip = Mat4::scale(-1.0, 1.0, 1.0);
                        transform_solid(&mut topo, a, &flip).unwrap();
                        transform_solid(&mut topo, b, &flip).unwrap();
                    }
                    _ => {}
                }
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 6, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"cone") && tags.contains(&"sphere"),
                    "{label}: surfaces {tags:?}"
                );
                assert!(
                    validate_solid(&topo, result).unwrap().is_valid(),
                    "{label}: invalid"
                );
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let volume = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-4,
                    "{label}: volume {volume}, truth {truth}"
                );
                let placed = |p: Point3| match pose {
                    "turned" => turn.mul_point(p),
                    "mirrored" => {
                        let unit = normal.normalize().unwrap();
                        p - unit * (2.0 * (p - at).dot(unit))
                    }
                    "scaled" => Point3::new(-p.x(), p.y(), p.z()),
                    _ => p,
                };
                let (on_axis, beside) = match op {
                    BooleanOp::Cut => (PointClassification::Outside, PointClassification::Inside),
                    _ => (PointClassification::Inside, PointClassification::Outside),
                };
                for (p, want) in [(foot, on_axis), (away, beside)] {
                    let got = classify_point(&topo, result, placed(p), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
            }
        }
    }
}
