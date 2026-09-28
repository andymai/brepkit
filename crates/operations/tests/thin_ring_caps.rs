//! A cap whose ring between its circular rim and a hole is thinner than the
//! sag of the rim's chords: a thin-walled tube, the tube bored part way, and
//! a ball through a cylinder's or a frustum's cap near its rim. In every pose
//! each operation keeps the ring, is exact and valid, and holds the
//! closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const POSES: [&str; 4] = ["upright", "turned", "mirrored", "scaled"];

fn turn() -> Mat4 {
    Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1))
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, solid, &turn()).unwrap();
            solid
        }
        "mirrored" => {
            let (at, normal) = mirror_plane();
            mirror(topo, solid, at, normal).unwrap()
        }
        "scaled" => {
            transform_solid(topo, solid, &Mat4::scale(-1.0, 1.0, 1.0)).unwrap();
            solid
        }
        _ => solid,
    }
}

fn placed(p: Point3, pose: &str) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "mirrored" => {
            let (at, normal) = mirror_plane();
            let unit = normal.normalize().unwrap();
            p - unit * (2.0 * (p - at).dot(unit))
        }
        "scaled" => Point3::new(-p.x(), p.y(), p.z()),
        _ => p,
    }
}

fn check(
    topo: &Topology,
    result: SolidId,
    label: &str,
    truth: f64,
    points: &[(Point3, PointClassification)],
    pose: &str,
) {
    let faces = solid_faces(topo, result).unwrap();
    assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
    let tags: Vec<&str> = faces
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect();
    assert!(!tags.contains(&"nurbs"), "{label}: surfaces {tags:?}");
    assert!(
        validate_solid(topo, result).unwrap().is_valid(),
        "{label}: invalid"
    );
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let volume = solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6,
        "{label}: volume {volume}, truth {truth}"
    );
    for &(p, want) in points {
        let got = classify_point(topo, result, placed(p, pose), 0.01, 1e-7).unwrap();
        assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
    }
}

/// A cylinder of height 10 less a coaxial bore, through it or 5 deep: the
/// top cap's ring is 5% of the rim's radius wide.
#[test]
fn a_thin_walled_tube_is_exact() {
    for (outer, inner) in [(10.0_f64, 9.5_f64), (2.0, 1.9)] {
        let wall = Point3::new(0.5 * (outer + inner), 0.0, 7.0);
        for (depth, truth) in [
            (12.0, PI * (outer * outer - inner * inner) * 10.0),
            (6.0, PI * outer.mul_add(outer * 10.0, -inner * inner * 5.0)),
        ] {
            for pose in POSES {
                let label = format!("{outer}/{inner}, bore {depth}, {pose}");
                let mut topo = Topology::new();
                let a = make_cylinder(&mut topo, outer, 10.0).unwrap();
                let b = make_cylinder(&mut topo, inner, depth).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, 11.0 - depth)).unwrap();
                let (a, b) = (posed(&mut topo, a, pose), posed(&mut topo, b, pose));
                let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
                let points = [
                    (wall, PointClassification::Inside),
                    (Point3::new(0.0, 0.0, 7.0), PointClassification::Outside),
                ];
                check(&topo, result, &label, truth, &points, pose);
            }
        }
    }
}

/// The volume a solid of revolution about `z` over `-3 < z < 3`, of radius
/// `radius(z)`, shares with a ball of radius `r` centred on its axis at `zc`.
fn shared(radius: impl Fn(f64) -> f64, r: f64, zc: f64) -> f64 {
    let (lo, hi) = ((zc - r).max(-3.0), (zc + r).min(3.0));
    let n = 200_000;
    let h = (hi - lo) / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h.mul_add(f64::from(i), lo);
        let ball = (r * r - (z - zc).powi(2)).max(0.0).sqrt();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * PI * ball.min(radius(z)).powi(2);
    }
    total * h / 3.0
}

/// A ball inside a cap's rim crossing the cap near it: a cylinder of radius
/// 1.5, whose ring is 0.051 and 0.053 wide, and the frustum from radius 3 to
/// 1.5, whose rings are 0.058 and 0.041 wide.
#[test]
fn a_ball_through_a_caps_rim_is_exact() {
    for (frustum, r, zc) in [
        (false, 1.47_f64, 2.75_f64),
        (false, 1.45, 2.9),
        (true, 1.5265, 2.5),
        (true, 1.5265, 2.55),
    ] {
        let radius = |z: f64| {
            if frustum {
                (1.5 - 3.0f64).mul_add((z + 3.0) / 6.0, 3.0)
            } else {
                1.5
            }
        };
        let body = if frustum {
            PI * 2.0 * 1.5f64.mul_add(1.5, 3.0f64.mul_add(1.5, 9.0))
        } else {
            PI * 1.5 * 1.5 * 6.0
        };
        let ball = 4.0 / 3.0 * PI * r.powi(3);
        let both = shared(radius, r, zc);
        for pose in POSES {
            for (op, truth, centre) in [
                (BooleanOp::Cut, body - both, PointClassification::Outside),
                (BooleanOp::Intersect, both, PointClassification::Inside),
                (
                    BooleanOp::Fuse,
                    body + ball - both,
                    PointClassification::Inside,
                ),
            ] {
                let shape = if frustum { "frustum" } else { "cylinder" };
                let label = format!("{shape}, ball {r} at {zc}, {pose} {op:?}");
                let mut topo = Topology::new();
                let a = if frustum {
                    make_cone(&mut topo, 3.0, 1.5, 6.0).unwrap()
                } else {
                    make_cylinder(&mut topo, 1.5, 6.0).unwrap()
                };
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_sphere(&mut topo, r, 32).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, zc)).unwrap();
                let (a, b) = (posed(&mut topo, a, pose), posed(&mut topo, b, pose));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let points = [(Point3::new(0.0, 0.0, zc), centre)];
                check(&topo, result, &label, truth, &points, pose);
            }
        }
    }
}
