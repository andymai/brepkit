//! A ball off a cone's axis that only some of the cone's generators reach
//! (beside a pointed cone's wall, across its seam, near its base, and across
//! a frustum's wall), and one holding a pointed cone's apex. In every pose
//! each operation is exact, valid and watertight, and holds the volume
//! integrated from the lens-shaped sections.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_sphere};
use brepkit_operations::tessellate::{TriangleMesh, is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// The cone's radius at height `z`: 3 at `z = -3` to `top` at `z = 3`.
fn cone_radius(top: f64, z: f64) -> f64 {
    (top - 3.0).mul_add((z + 3.0) / 6.0, 3.0)
}

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

/// The ball's part inside the cone: at each height the two sections are
/// discs the ball centre's distance from the axis apart.
fn shared(top: f64, centre: Point3, r: f64) -> f64 {
    let d = centre.x().hypot(centre.y());
    let (lo, hi) = ((centre.z() - r).max(-3.0), (centre.z() + r).min(3.0));
    let n = 200_000;
    let h = (hi - lo) / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h.mul_add(f64::from(i), lo);
        let ball = (r * r - (z - centre.z()).powi(2)).max(0.0).sqrt();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * lens(cone_radius(top, z), ball, d);
    }
    total * h / 3.0
}

fn mesh_area(mesh: &TriangleMesh) -> f64 {
    mesh.indices
        .chunks(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
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
fn a_ball_beside_a_cone_is_exact() {
    for (top, centre, r) in [
        (0.0_f64, Point3::new(1.0, 0.8, 1.2), 1.1_f64),
        (0.0, Point3::new(0.0, 1.8, -1.0), 1.0),
        (0.0, Point3::new(1.5, 0.0, 0.0), 0.8),
        (1.5, Point3::new(0.5, 2.0, 0.5), 0.9),
        (0.0, Point3::new(-1.2, -1.2, -2.0), 1.2),
    ] {
        let cone = PI * 2.0 * top.mul_add(top, 3.0f64.mul_add(top, 9.0));
        let ball = 4.0 / 3.0 * PI * r.powi(3);
        let both = shared(top, centre, r);
        let d = centre.x().hypot(centre.y());
        let in_cone =
            |p: Point3| p.z() > -3.0 && p.z() < 3.0 && p.x().hypot(p.y()) < cone_radius(top, p.z());
        let in_ball = |p: Point3| (p - centre).length() < r;
        // The ball's centre, and a point toward the axis inside both.
        let reach = 0.5 * ((d - cone_radius(top, centre.z())).max(0.0) + r);
        let toward = centre - Vec3::new(centre.x(), centre.y(), 0.0) * (reach / d);
        let points = [centre, toward];
        for name in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth) in [
                (BooleanOp::Cut, cone - both),
                (BooleanOp::Intersect, both),
                (BooleanOp::Fuse, cone + ball - both),
            ] {
                let label = format!("ball {r} at {centre:?} beside top {top}, {name} {op:?}");
                let mut topo = Topology::new();
                let a = make_cone(&mut topo, 3.0, top, 6.0).unwrap();
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_sphere(&mut topo, r, 32).unwrap();
                transform_solid(
                    &mut topo,
                    b,
                    &Mat4::translation(centre.x(), centre.y(), centre.z()),
                )
                .unwrap();
                let (a, b) = (posed(&mut topo, a, name), posed(&mut topo, b, name));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"sphere") && tags.contains(&"cone") && !tags.contains(&"nurbs"),
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
                for p in points {
                    // A point on either surface reads OnBoundary.
                    let off_cone = (p.x().hypot(p.y()) - cone_radius(top, p.z())).abs();
                    if off_cone < 1e-3 || ((p - centre).length() - r).abs() < 1e-3 {
                        continue;
                    }
                    let inside = match op {
                        BooleanOp::Cut => in_cone(p) && !in_ball(p),
                        BooleanOp::Intersect => in_cone(p) && in_ball(p),
                        BooleanOp::Fuse => in_cone(p) || in_ball(p),
                    };
                    let want = if inside {
                        PointClassification::Inside
                    } else {
                        PointClassification::Outside
                    };
                    let got = classify_point(&topo, result, placed(p, name), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
                // A pointed cone's faces mesh on their own too, as a per-face
                // export takes them: the wall the ball bites into within 1%,
                // and a lens the section alone bounds within its chords' 3%.
                for &f in &faces {
                    if top > 0.0 || topo.face(f).unwrap().surface().type_tag() != "cone" {
                        continue;
                    }
                    let area = mesh_area(&tessellate(&topo, f, 0.01).unwrap());
                    let exact = face_area(&topo, f, 0.01).unwrap();
                    let bound = if op == BooleanOp::Intersect {
                        0.03
                    } else {
                        0.01
                    };
                    assert!(
                        (area - exact).abs() < bound * exact,
                        "{label}: cone face meshes {area} of {exact}"
                    );
                }
                // Mirrored afterwards, its wires run the other way round.
                transform_solid(&mut topo, result, &Mat4::scale(-1.0, 1.0, 1.0)).unwrap();
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open mesh mirrored");
            }
        }
    }
}

/// A ball holding the pointed cone's apex off its axis: every generator
/// leaves it once ahead of the apex, one loop round the cone, and the cone's
/// piece in the ball is a tip that loop bounds.
#[test]
fn a_ball_holding_the_apex_is_exact() {
    for (centre, r) in [
        (Point3::new(0.5, 0.0, 2.5), 2.0_f64),
        (Point3::new(-0.4, 0.3, 2.0), 1.5),
    ] {
        let cone = PI * 2.0 * 9.0;
        let ball = 4.0 / 3.0 * PI * r.powi(3);
        let both = shared(0.0, centre, r);
        let in_cone =
            |p: Point3| p.z() > -3.0 && p.z() < 3.0 && p.x().hypot(p.y()) < cone_radius(0.0, p.z());
        let in_ball = |p: Point3| (p - centre).length() < r;
        // Below the apex on the axis, in both; low on the axis; and past the
        // apex in the ball.
        let points = [
            Point3::new(0.0, 0.0, 2.5),
            Point3::new(0.0, 0.0, -2.5),
            Point3::new(centre.x(), centre.y(), centre.z() + 0.9 * r),
        ];
        for name in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth) in [
                (BooleanOp::Cut, cone - both),
                (BooleanOp::Intersect, both),
                (BooleanOp::Fuse, cone + ball - both),
            ] {
                let label = format!("ball {r} at {centre:?} holding the apex, {name} {op:?}");
                let mut topo = Topology::new();
                let a = make_cone(&mut topo, 3.0, 0.0, 6.0).unwrap();
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_sphere(&mut topo, r, 32).unwrap();
                transform_solid(
                    &mut topo,
                    b,
                    &Mat4::translation(centre.x(), centre.y(), centre.z()),
                )
                .unwrap();
                let (a, b) = (posed(&mut topo, a, name), posed(&mut topo, b, name));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"sphere") && tags.contains(&"cone") && !tags.contains(&"nurbs"),
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
                for p in points {
                    let off_cone = (p.x().hypot(p.y()) - cone_radius(0.0, p.z())).abs();
                    if off_cone < 1e-3 || ((p - centre).length() - r).abs() < 1e-3 {
                        continue;
                    }
                    let inside = match op {
                        BooleanOp::Cut => in_cone(p) && !in_ball(p),
                        BooleanOp::Intersect => in_cone(p) && in_ball(p),
                        BooleanOp::Fuse => in_cone(p) || in_ball(p),
                    };
                    let want = if inside {
                        PointClassification::Inside
                    } else {
                        PointClassification::Outside
                    };
                    let got = classify_point(&topo, result, placed(p, name), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
                // The tip meshes on its own too, as a per-face export takes
                // it.
                for &f in &faces {
                    if topo.face(f).unwrap().surface().type_tag() != "cone" {
                        continue;
                    }
                    let area = mesh_area(&tessellate(&topo, f, 0.01).unwrap());
                    let exact = face_area(&topo, f, 0.01).unwrap();
                    assert!(
                        (area - exact).abs() < 0.01 * exact,
                        "{label}: cone face meshes {area} of {exact}"
                    );
                }
                transform_solid(&mut topo, result, &Mat4::scale(-1.0, 1.0, 1.0)).unwrap();
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open mesh mirrored");
            }
        }
    }
}
