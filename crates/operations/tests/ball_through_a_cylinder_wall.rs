//! A ball bulging through a cylinder's side wall: the section loop crosses
//! the ball's equator, so each hemisphere keeps its own arc of it, on and off
//! the wall's seam line. In every pose each operation is exact, valid and
//! watertight, and holds the volume integrated from the lens-shaped sections.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const WALL: f64 = 2.0;

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

/// The ball's part inside the cylinder: at each height along the axis, the
/// two sections are discs the ball centre's distance from the axis apart.
fn inside(d: f64, r: f64) -> f64 {
    let n = 200_000;
    let h = 2.0 * r / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h.mul_add(f64::from(i), -r);
        let ball = (r * r - z * z).max(0.0).sqrt();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * lens(ball, WALL, d);
    }
    total * h / 3.0
}

fn pose_of(name: &str) -> Mat4 {
    match name {
        "turned" => {
            Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3)
        }
        "scaled" => Mat4::scale(-1.0, 1.0, 1.0),
        _ => Mat4::identity(),
    }
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1))
}

fn posed(topo: &mut Topology, solid: SolidId, name: &str) -> SolidId {
    if name == "mirrored" {
        let (at, normal) = mirror_plane();
        return mirror(topo, solid, at, normal).unwrap();
    }
    transform_solid(topo, solid, &pose_of(name)).unwrap();
    solid
}

fn placed(p: Point3, name: &str) -> Point3 {
    if name == "mirrored" {
        let (at, normal) = mirror_plane();
        let unit = normal.normalize().unwrap();
        return p - unit * (2.0 * (p - at).dot(unit));
    }
    pose_of(name).mul_point(p)
}

#[test]
fn a_ball_through_a_cylinder_wall_is_exact() {
    let cylinder = PI * WALL * WALL * 6.0;
    for (x, y, z, r) in [
        (1.0_f64, 0.8_f64, 1.2_f64, 1.1_f64),
        (1.0, 0.8, 0.0, 1.1),
        (1.28, 0.0, 1.2, 1.1),
        (1.5, 0.0, 0.0, 1.0),
    ] {
        let d = x.hypot(y);
        let both = inside(d, r);
        let ball = 4.0 / 3.0 * PI * r.powi(3);
        let centre = Point3::new(x, y, z);
        // Midway between the wall and the ball's far side, on the ray from
        // the axis through the centre: in the ball, outside the cylinder.
        let bulge = Point3::new(x, y, z) + Vec3::new(x, y, 0.0) * ((0.5 * (WALL + d + r) - d) / d);
        for name in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth, at_centre, in_bulge) in [
                (
                    BooleanOp::Cut,
                    cylinder - both,
                    PointClassification::Outside,
                    PointClassification::Outside,
                ),
                (
                    BooleanOp::Intersect,
                    both,
                    PointClassification::Inside,
                    PointClassification::Outside,
                ),
                (
                    BooleanOp::Fuse,
                    cylinder + ball - both,
                    PointClassification::Inside,
                    PointClassification::Inside,
                ),
            ] {
                let label = format!("ball {r} at ({x}, {y}, {z}), {name} {op:?}");
                let mut topo = Topology::new();
                let a = make_cylinder(&mut topo, WALL, 6.0).unwrap();
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_sphere(&mut topo, r, 32).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(x, y, z)).unwrap();
                let (a, b) = (posed(&mut topo, a, name), posed(&mut topo, b, name));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"sphere")
                        && tags.contains(&"cylinder")
                        && !tags.contains(&"nurbs"),
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
                for (p, want) in [(centre, at_centre), (bulge, in_bulge)] {
                    let got = classify_point(&topo, result, placed(p, name), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
            }
        }
    }
}
