//! Two discs about one centre revolved part of a turn about the same axis:
//! each a band of a torus capped by its two end discs. The smaller cut from
//! the larger leaves the band of a ring torus, which is exact, valid and
//! watertight in every pose, holds Pappus's volume, and keeps its material
//! between the two tubes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_circle_edge;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::wire::{OrientedEdge, Wire};

const POSES: [&str; 4] = ["upright", "turned", "mirrored", "scaled"];

/// The discs' centre and the axis they revolve about, 20 from it.
const CENTRE: Point3 = Point3::new(30.0, 0.0, 0.0);
const AXIS_AT: Point3 = Point3::new(30.0, 20.0, 0.0);

fn turn() -> Mat4 {
    Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(31.0, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1))
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

/// The disc of radius `r` about [`CENTRE`] in the plane `x = 30`, revolved
/// `angle` about the line through [`AXIS_AT`] along `z`.
fn revolved_disc(topo: &mut Topology, r: f64, angle: f64) -> SolidId {
    let normal = Vec3::new(1.0, 0.0, 0.0);
    let rim = make_circle_edge(topo, CENTRE, normal, r, 1e-7).unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 30.0 },
    ));
    revolve(topo, face, AXIS_AT, Vec3::new(0.0, 0.0, 1.0), angle).unwrap()
}

/// A point at distance `from_axis` from the tube's centre circle, `angle`
/// round the revolution from the discs' plane, level with the axis's foot.
fn at(angle: f64, from_axis: f64) -> Point3 {
    let reach = 20.0 - from_axis;
    Point3::new(30.0 + reach * angle.sin(), 20.0 - reach * angle.cos(), 0.0)
}

#[test]
fn a_disc_revolved_part_way_less_a_smaller_one_is_exact() {
    for angle in [PI / 2.0, PI, 1.7 * PI] {
        for pose in POSES {
            let label = format!("angle {angle:.3}, {pose}");
            let mut topo = Topology::new();
            let outer = revolved_disc(&mut topo, 5.0, angle);
            let inner = revolved_disc(&mut topo, 4.0, angle);
            let (a, b) = (posed(&mut topo, outer, pose), posed(&mut topo, inner, pose));
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
            assert_eq!(
                mesh_fallback_count(),
                before,
                "{label}: fell back to a mesh"
            );
            let faces = solid_faces(&topo, result).unwrap();
            assert_eq!(faces.len(), 4, "{label}: {} faces", faces.len());
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            // Pappus: the ring's area (25 - 16) pi times the path its centre
            // travels at radius 20.
            let truth = 9.0 * PI * 20.0 * angle;
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let half = 0.5 * angle;
            for (p, want) in [
                (at(half, 4.5), PointClassification::Inside),
                (at(half, -4.5), PointClassification::Inside),
                (at(half, 0.0), PointClassification::Outside),
                (at(half, 3.0), PointClassification::Outside),
                (at(half, 5.5), PointClassification::Outside),
            ] {
                let got = classify_point(&topo, result, placed(p, pose), 0.01, 1e-7).unwrap();
                assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
            }
        }
    }
}
