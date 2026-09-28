//! A rod parallel to a pointed cone's axis, through its wall beside the
//! axis, clear of the wall's seam or straddling it: every ruling of the rod
//! meets the cone once, so the section winds the rod as one closed loop. In
//! every pose each operation is exact, valid and watertight, and holds the
//! volume integrated from the lens-shaped sections.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_cylinder};
use brepkit_operations::tessellate::{TriangleMesh, is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const ROD: f64 = 0.6;

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

/// The rod's part inside the cone of radius 3 at `z = -3` to its apex at
/// `z = 3`: at each height the two sections are discs `d` apart.
fn shared(d: f64) -> f64 {
    let n = 200_000;
    let h = 6.0 / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h.mul_add(f64::from(i), -3.0);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * lens(0.5 * (3.0 - z), ROD, d);
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
fn a_rod_along_a_cone_is_exact() {
    let cone = PI * 9.0 * 6.0 / 3.0;
    let rod = PI * ROD * ROD * 20.0;
    for (x, y) in [(1.2_f64, 0.5_f64), (0.0, 1.3), (-1.3, 0.0), (0.5, -1.2)] {
        let d = x.hypot(y);
        let both = shared(d);
        // On the rod's axis: inside the cone at z = 0, outside it at z = 2.
        let (low, high) = (Point3::new(x, y, 0.0), Point3::new(x, y, 2.0));
        for name in ["upright", "turned", "mirrored", "scaled"] {
            for (op, truth, at_low, at_high) in [
                (
                    BooleanOp::Cut,
                    cone - both,
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
                    cone + rod - both,
                    PointClassification::Inside,
                    PointClassification::Inside,
                ),
            ] {
                let label = format!("rod at ({x}, {y}), {name} {op:?}");
                let mut topo = Topology::new();
                let a = make_cone(&mut topo, 3.0, 0.0, 6.0).unwrap();
                transform_solid(&mut topo, a, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                let b = make_cylinder(&mut topo, ROD, 20.0).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(x, y, -10.0)).unwrap();
                let (a, b) = (posed(&mut topo, a, name), posed(&mut topo, b, name));
                let result = boolean(&mut topo, op, a, b).unwrap();
                let faces = solid_faces(&topo, result).unwrap();
                assert!(faces.len() <= 8, "{label}: {} faces", faces.len());
                let tags: Vec<&str> = faces
                    .iter()
                    .map(|&f| topo.face(f).unwrap().surface().type_tag())
                    .collect();
                assert!(
                    tags.contains(&"cone")
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
                for (p, want) in [(low, at_low), (high, at_high)] {
                    let got = classify_point(&topo, result, placed(p, name), 0.01, 1e-7).unwrap();
                    assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
                }
                // The wall the rod passes through meshes on its own too, as a
                // per-face export takes it. The Intersect's cone faces are
                // trimmed patches, whose per-face mesh is an open roadmap row.
                for &f in &faces {
                    if op == BooleanOp::Intersect
                        || topo.face(f).unwrap().surface().type_tag() != "cone"
                    {
                        continue;
                    }
                    let area = mesh_area(&tessellate(&topo, f, 0.01).unwrap());
                    let exact = face_area(&topo, f, 0.01).unwrap();
                    assert!(
                        (area - exact).abs() < 0.01 * exact,
                        "{label}: cone face meshes {area} of {exact}"
                    );
                }
            }
        }
    }
}
