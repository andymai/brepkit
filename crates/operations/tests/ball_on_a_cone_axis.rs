//! A ball on a cone's axis meets the cone's wall in circles: through the
//! middle of a pointed cone, swallowing or touching its apex, across a
//! frustum's wall and top, across the wall and the base, beside a frustum's
//! wall without touching it, just past tangency with a pointed cone's wall,
//! and touching only a frustum's cap. In every pose each operation is
//! exact, valid and watertight, and holds the section-integrated volume.
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
use brepkit_topology::solid::SolidId;

/// The volume a cone of base radius 3 at `z = -3` and radius `top` at
/// `z = 3` shares with a ball of radius `r` centred on its axis at `zc`:
/// their sections are concentric discs.
fn shared(top: f64, r: f64, zc: f64) -> f64 {
    let (lo, hi) = ((zc - r).max(-3.0), (zc + r).min(3.0));
    let n = 200_000;
    let h = (hi - lo) / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h.mul_add(f64::from(i), lo);
        let cone = (top - 3.0).mul_add((z + 3.0) / 6.0, 3.0);
        let ball = (r * r - (z - zc).powi(2)).max(0.0).sqrt();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * PI * cone.min(ball).powi(2);
    }
    total * h / 3.0
}

/// Where a pose takes a solid: turned, flipped over, mirrored through a
/// slanted plane, or mirrored by a negative scale.
fn pose_of(name: &str) -> Mat4 {
    match name {
        "turned" => {
            Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3)
        }
        "flipped" => Mat4::rotation_x(PI),
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
fn a_ball_on_a_cones_axis_is_exact() {
    for (top, r, zc) in [
        (0.0_f64, 2.0_f64, 0.0),
        (0.0, 1.3, 2.5),
        (0.0, 0.5, 2.5),
        (1.5, 2.0, 2.0),
        (0.0, 2.4, -2.0),
        (1.5, 1.45, 2.5),
        // Just past tangency with the wall (whose distance from the axis
        // point is 3 / sqrt(5)): a band 1e-4 deep leaves the cone.
        (0.0, 3.0 / 5.0_f64.sqrt() + 1e-4, 0.0),
        // Touching only the frustum's cap, leaving rings of it 0.006 and
        // 0.0012 wide round its rim.
        (1.5, 1.5148, 2.75),
        (1.5, 1.5195, 2.75),
    ] {
        let cone = PI * 2.0 * top.mul_add(top, 3.0f64.mul_add(top, 9.0));
        let ball = 4.0 / 3.0 * PI * r.powi(3);
        let both = shared(top, r, zc);
        for name in ["upright", "turned", "flipped", "mirrored", "scaled"] {
            for (op, truth, centre) in [
                (BooleanOp::Cut, cone - both, PointClassification::Outside),
                (BooleanOp::Intersect, both, PointClassification::Inside),
                (
                    BooleanOp::Fuse,
                    cone + ball - both,
                    PointClassification::Inside,
                ),
            ] {
                let label = format!("top {top}, ball {r} at {zc}, {name} {op:?}");
                let mut topo = Topology::new();
                let a = make_cone(&mut topo, 3.0, top, 6.0).unwrap();
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_sphere(&mut topo, r, 32).unwrap();
                // Flipped over, the ball is also spun about its centre so its
                // poles leave the cone's axis.
                let spin = if name == "flipped" { 0.5 } else { 0.0 };
                let place = Mat4::translation(0.0, 0.0, zc) * Mat4::rotation_y(spin);
                transform_solid(&mut topo, b, &place).unwrap();
                let (a, b) = (posed(&mut topo, a, name), posed(&mut topo, b, name));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"sphere") && !tags.contains(&"nurbs"),
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
                    (volume - truth).abs() < 1e-6,
                    "{label}: volume {volume}, truth {truth}"
                );
                let at = placed(Point3::new(0.0, 0.0, zc), name);
                let got = classify_point(&topo, result, at, 0.01, 1e-7).unwrap();
                assert_eq!(got, centre, "{label}: the ball's centre reads {got:?}");
            }
        }
    }
}
