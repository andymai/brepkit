//! A cylinder less a box over one corner of its top, and the corner itself,
//! in every pose: upright, turned, mirrored through a slanted plane, and
//! mirrored by a transform. A mirror turns the cylinder's rims against its
//! u, and each result must still come out exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

/// The disc of radius 2 beyond `x = 1` and `y = 1.2`, times the 2.2 of
/// height the box takes from the cylinder's top.
fn corner_volume() -> f64 {
    let f = |x: f64| 0.5 * x * (4.0 - x * x).sqrt() + 2.0 * (0.5 * x).asin();
    let area = f(1.6) - f(1.0) - 1.2 * 0.6;
    area * 2.2
}

#[test]
fn a_cylinders_box_corner_is_exact_in_every_pose() {
    let cylinder = std::f64::consts::PI * 4.0 * 6.0;
    let turn = Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3);
    let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
    for pose in ["upright", "turned", "mirrored", "scaled"] {
        for (op, truth) in [
            (BooleanOp::Cut, cylinder - corner_volume()),
            (BooleanOp::Intersect, corner_volume()),
        ] {
            let mut topo = Topology::new();
            let mut c = make_cylinder(&mut topo, 2.0, 6.0).unwrap();
            transform_solid(&mut topo, c, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
            let mut b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.2, 0.8)).unwrap();
            match pose {
                "turned" => {
                    transform_solid(&mut topo, c, &turn).unwrap();
                    transform_solid(&mut topo, b, &turn).unwrap();
                }
                "mirrored" => {
                    c = mirror(&mut topo, c, at, normal).unwrap();
                    b = mirror(&mut topo, b, at, normal).unwrap();
                }
                "scaled" => {
                    let flip = Mat4::scale(-1.0, 1.0, 1.0);
                    transform_solid(&mut topo, c, &flip).unwrap();
                    transform_solid(&mut topo, b, &flip).unwrap();
                }
                _ => {}
            }
            let result = boolean(&mut topo, op, c, b).unwrap();
            let faces = solid_faces(&topo, result).unwrap();
            let walls = faces
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .count();
            assert!(
                faces.len() <= 8 && walls > 0,
                "{pose} {op:?}: {} faces, {walls} of them cylinder",
                faces.len()
            );
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{pose} {op:?}: invalid"
            );
            // A point in the corner the box takes, and one in the body it
            // leaves, placed like the operands.
            let place = |p: Point3| match pose {
                "turned" => turn.mul_point(p),
                "mirrored" => {
                    let unit = normal.normalize().unwrap();
                    p - unit * (2.0 * (p - at).dot(unit))
                }
                "scaled" => Point3::new(-p.x(), p.y(), p.z()),
                _ => p,
            };
            let (in_corner, in_body) = match op {
                BooleanOp::Cut => (PointClassification::Outside, PointClassification::Inside),
                _ => (PointClassification::Inside, PointClassification::Outside),
            };
            for (p, want) in [
                (Point3::new(1.2, 1.4, 2.0), in_corner),
                (Point3::new(-1.0, -0.5, 0.0), in_body),
            ] {
                let got = classify_point(&topo, result, place(p), 0.01, 1e-7).unwrap();
                assert_eq!(got, want, "{pose} {op:?}: {p:?} reads {got:?}");
            }
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth.max(1.0),
                "{pose} {op:?}: volume {volume}, truth {truth}"
            );
        }
    }
}
