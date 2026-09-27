//! A rod across a pointed cone, a frustum or a cylinder, poking out through
//! its wall on the side of the wall's seam and on the far side, or passing
//! through it, upright, turned, mirrored through a slanted plane and
//! mirrored by a transform: the Cut and the Intersect are exact and remove
//! what the rod takes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// The rod's radius, and the height of its axis.
const ROD: f64 = 0.6;
const ROD_Z: f64 = 1.0;

/// The part of a solid of revolution about `z`, radius `radius(z)`, inside
/// the rod along `y` through `(x0, ., ROD_Z)`: at each height, the chord
/// strip `2 sqrt(r^2 - x^2)` integrated in closed form across the rod's
/// width, then Simpson over the rod's height (in its angle, which smooths
/// the ends).
fn rod_volume(x0: f64, radius: impl Fn(f64) -> f64) -> f64 {
    let strip = |r: f64, x: f64| {
        let x = x.clamp(-r, r);
        x.mul_add((r * r - x * x).max(0.0).sqrt(), r * r * (x / r).asin())
    };
    let n = 4000;
    let h = PI / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let phi = f64::from(i).mul_add(h, -PI / 2.0);
        let (z, half) = (ROD.mul_add(phi.sin(), ROD_Z), ROD * phi.cos());
        let r = radius(z);
        let (a, b) = ((x0 - half).max(-r), (x0 + half).min(r));
        let f = if b > a {
            (strip(r, b) - strip(r, a)) * half
        } else {
            0.0
        };
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * f;
    }
    total * h / 3.0
}

/// The solid, the rod axis's `x`, the solid's volume and its radius at `z`.
type Case<'a> = (&'a str, f64, f64, &'a dyn Fn(f64) -> f64);

#[test]
fn a_rod_through_a_wall_is_exact_on_either_side_of_its_seam() {
    let turn = Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3);
    let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
    let cone_radius = |z: f64| 0.5 * (3.0 - z);
    let frustum_radius = |z: f64| 0.25f64.mul_add(-(z + 3.0), 3.0);
    let cylinder_radius = |_: f64| 2.0;
    let cases: [Case<'_>; 6] = [
        ("cone", 0.5, PI * 9.0 * 6.0 / 3.0, &cone_radius),
        ("cone", -0.5, PI * 9.0 * 6.0 / 3.0, &cone_radius),
        ("cone", 0.0, PI * 9.0 * 6.0 / 3.0, &cone_radius),
        (
            "frustum",
            0.5,
            PI * 6.0 * (9.0 + 4.5 + 2.25) / 3.0,
            &frustum_radius,
        ),
        ("cylinder", 1.7, PI * 4.0 * 6.0, &cylinder_radius),
        ("cylinder", -1.7, PI * 4.0 * 6.0, &cylinder_radius),
    ];
    for (solid, x0, whole, radius) in cases {
        let taken = rod_volume(x0, radius);
        for pose in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth) in [
                (BooleanOp::Cut, whole - taken),
                (BooleanOp::Intersect, taken),
            ] {
                let label = format!("{solid} x0 {x0} {pose} {op:?}");
                let mut topo = Topology::new();
                let mut a = match solid {
                    "cone" => make_cone(&mut topo, 3.0, 0.0, 6.0).unwrap(),
                    "frustum" => make_cone(&mut topo, 3.0, 1.5, 6.0).unwrap(),
                    _ => make_cylinder(&mut topo, 2.0, 6.0).unwrap(),
                };
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let mut b = make_cylinder(&mut topo, ROD, 20.0).unwrap();
                let place = Mat4::translation(x0, 10.0, ROD_Z)
                    * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
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
                let faces = solid_faces(&topo, result).unwrap().len();
                assert!(faces <= 6, "{label}: {faces} faces");
                assert!(
                    validate_solid(&topo, result).unwrap().is_valid(),
                    "{label}: invalid"
                );
                let volume = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-4,
                    "{label}: volume {volume}, truth {truth}"
                );
                // The rod's axis inside the solid, and the solid's axis below
                // the rod, placed like the operands.
                let placed = |p: Point3| match pose {
                    "turned" => turn.mul_point(p),
                    "mirrored" => {
                        let unit = normal.normalize().unwrap();
                        p - unit * (2.0 * (p - at).dot(unit))
                    }
                    "scaled" => Point3::new(-p.x(), p.y(), p.z()),
                    _ => p,
                };
                let (in_rod, below) = match op {
                    BooleanOp::Cut => (PointClassification::Outside, PointClassification::Inside),
                    _ => (PointClassification::Inside, PointClassification::Outside),
                };
                for (p, want) in [
                    (Point3::new(x0, 0.0, ROD_Z), in_rod),
                    (Point3::new(0.0, 0.0, -2.0), below),
                ] {
                    let got = classify_point(&topo, result, placed(p), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
            }
        }
    }
}

/// The part of `make_torus(4, 1.5)` inside the rod along `y` through
/// `(x0, ., z0)`: at each height the ring's section is an annulus, whose
/// chord strips are the outer disc's less the inner's.
fn rod_in_ring(x0: f64, z0: f64) -> f64 {
    let strip = |r: f64, a: f64, b: f64| {
        let f = |x: f64| {
            let x = x.clamp(-r, r);
            x.mul_add((r * r - x * x).max(0.0).sqrt(), r * r * (x / r).asin())
        };
        let (a, b) = (a.max(-r), b.min(r));
        if b > a { f(b) - f(a) } else { 0.0 }
    };
    let n = 4000;
    let h = PI / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let phi = f64::from(i).mul_add(h, -PI / 2.0);
        let (z, half) = (ROD.mul_add(phi.sin(), z0), ROD * phi.cos());
        let s = z.mul_add(-z, 2.25).max(0.0).sqrt();
        let (a, b) = (x0 - half, x0 + half);
        let f = (strip(4.0 + s, a, b) - strip(4.0 - s, a, b)) * half;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * f;
    }
    total * h / 3.0
}

/// A rod through a ring's tube on both sides of its hole, low enough that
/// every ruling of the rod meets the tube: each op is exact.
#[test]
fn a_rod_through_a_rings_tube_is_exact() {
    use brepkit_operations::primitives::make_torus;
    let turn = Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3);
    let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
    let (ring, rod) = (18.0 * PI * PI, PI * ROD * ROD * 20.0);
    for (x0, z0) in [(0.5, 0.3), (1.0, -0.4)] {
        let taken = rod_in_ring(x0, z0);
        for pose in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth) in [
                (BooleanOp::Cut, ring - taken),
                (BooleanOp::Intersect, taken),
                (BooleanOp::Fuse, ring + rod - taken),
            ] {
                let label = format!("x0 {x0} z0 {z0} {pose} {op:?}");
                let mut topo = Topology::new();
                let mut a = make_torus(&mut topo, 4.0, 1.5, 32).unwrap();
                let mut b = make_cylinder(&mut topo, ROD, 20.0).unwrap();
                let place =
                    Mat4::translation(x0, 10.0, z0) * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
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
                let faces = solid_faces(&topo, result).unwrap().len();
                assert!(faces <= 8, "{label}: {faces} faces");
                assert!(
                    validate_solid(&topo, result).unwrap().is_valid(),
                    "{label}: invalid"
                );
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
                // In the tube and the rod; in the tube away from the rod; in
                // the rod within the ring's hole.
                let (both, ring_only, rod_only) = match op {
                    BooleanOp::Cut => (
                        PointClassification::Outside,
                        PointClassification::Inside,
                        PointClassification::Outside,
                    ),
                    BooleanOp::Intersect => (
                        PointClassification::Inside,
                        PointClassification::Outside,
                        PointClassification::Outside,
                    ),
                    BooleanOp::Fuse => (
                        PointClassification::Inside,
                        PointClassification::Inside,
                        PointClassification::Inside,
                    ),
                };
                for (p, want) in [
                    (Point3::new(x0, -4.0, z0), both),
                    (Point3::new(-4.0, 0.0, 0.0), ring_only),
                    (Point3::new(x0, 0.0, z0), rod_only),
                ] {
                    let got = classify_point(&topo, result, placed(p), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
            }
        }
    }
}
