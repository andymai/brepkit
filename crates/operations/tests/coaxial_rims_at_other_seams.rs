//! Coaxial walls and rings whose seams lie at different angles: a ring
//! standing on a tube's rim, a cup cut by a ring extruded from two circles,
//! and two equal rods overlapping along their axis, the second turned about
//! it. In every pose each operation is exact, valid and watertight, holds the
//! closed-form volume, and keeps its material where it belongs.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::make_cylinder;
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

/// A cylinder of radius `r` from `z0` to `z1`, turned `degrees` about `z`.
fn rod(topo: &mut Topology, r: f64, z0: f64, z1: f64, degrees: f64) -> SolidId {
    let c = make_cylinder(topo, r, z1 - z0).unwrap();
    let place = Mat4::rotation_z(degrees.to_radians()) * Mat4::translation(0.0, 0.0, z0);
    transform_solid(topo, c, &place).unwrap();
    c
}

/// The tube between radii `r0` and `r1` from `z0` to `z1`, cut from a rod.
fn tube(topo: &mut Topology, r0: f64, r1: f64, z0: f64, z1: f64, degrees: f64) -> SolidId {
    let outer = rod(topo, r1, z0, z1, degrees);
    let inner = rod(topo, r0, z0 - 1.0, z1 + 1.0, degrees);
    boolean(topo, BooleanOp::Cut, outer, inner).unwrap()
}

/// The ring between radii `r0` and `r1` at `z0`, a face of two whole
/// circles, extruded up to `z1` and turned `degrees` about `z`.
fn extruded_ring(topo: &mut Topology, r0: f64, r1: f64, z0: f64, z1: f64, degrees: f64) -> SolidId {
    let (centre, up) = (Point3::new(0.0, 0.0, z0), Vec3::new(0.0, 0.0, 1.0));
    let outer = make_circle_edge(topo, centre, up, r1, 1e-7).unwrap();
    let inner = make_circle_edge(topo, centre, up, r0, 1e-7).unwrap();
    let outer = topo.add_wire(Wire::new(vec![OrientedEdge::new(outer, true)], true).unwrap());
    let inner = topo.add_wire(Wire::new(vec![OrientedEdge::new(inner, false)], true).unwrap());
    let face = topo.add_face(Face::new(
        outer,
        vec![inner],
        FaceSurface::Plane { normal: up, d: z0 },
    ));
    let solid = brepkit_operations::extrude::extrude(topo, face, up, z1 - z0).unwrap();
    transform_solid(topo, solid, &Mat4::rotation_z(degrees.to_radians())).unwrap();
    solid
}

/// A cylinder of radius 20 and height 12 shelled 2 thick, its top open.
fn cup(topo: &mut Topology) -> SolidId {
    let c = make_cylinder(topo, 20.0, 12.0).unwrap();
    let top = solid_faces(topo, c)
        .unwrap()
        .into_iter()
        .find(|&f| {
            matches!(topo.face(f).unwrap().surface(),
                FaceSurface::Plane { normal, d } if normal.z() > 0.9 && (d - 12.0).abs() < 1e-9)
        })
        .unwrap();
    brepkit_operations::shell_op::shell(topo, c, 2.0, &[top]).unwrap()
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
    assert!(
        validate_solid(topo, result).unwrap().is_valid(),
        "{label}: invalid"
    );
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let volume = solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    for &(p, want) in points {
        let got = classify_point(topo, result, placed(p, pose), 0.01, 1e-7).unwrap();
        assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
    }
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId, label: &str) -> SolidId {
    let before = mesh_fallback_count();
    let result = boolean(topo, op, a, b).unwrap();
    assert_eq!(
        mesh_fallback_count(),
        before,
        "{label}: fell back to a mesh"
    );
    result
}

/// A ring from radius 18.2 to 19, or 19.5 to 19.9, standing on the top of a
/// tube from 18 to 20: the tube's top splits into three rings and the middle
/// one meets the ring's foot.
#[test]
fn a_ring_standing_on_a_tubes_rim_fuses_exactly() {
    for (r0, r1) in [(18.2_f64, 19.0_f64), (19.5, 19.9)] {
        for degrees in [0.0, 30.0] {
            for pose in POSES {
                let label = format!("ring {r0}..{r1} turned {degrees}, {pose}");
                let mut topo = Topology::new();
                let wall = tube(&mut topo, 18.0, 20.0, 0.0, 12.0, 0.0);
                let ring = tube(&mut topo, r0, r1, 12.0, 13.5, degrees);
                let (a, b) = (posed(&mut topo, wall, pose), posed(&mut topo, ring, pose));
                let result = exact(&mut topo, BooleanOp::Fuse, a, b, &label);
                let truth = PI
                    * (20.0f64.mul_add(20.0, -18.0 * 18.0) * 12.0 + r1.mul_add(r1, -r0 * r0) * 1.5);
                let mid = 0.5 * (r0 + r1);
                let points = [
                    (Point3::new(mid, 0.0, 12.7), PointClassification::Inside),
                    (Point3::new(0.0, 19.0, 6.0), PointClassification::Inside),
                    (
                        Point3::new(r0 - 0.05, 0.0, 12.7),
                        PointClassification::Outside,
                    ),
                    (
                        Point3::new(0.0, r1 + 0.05, 12.7),
                        PointClassification::Outside,
                    ),
                ];
                check(&topo, result, &label, truth, &points, pose);
            }
        }
    }
}

/// A cup cut by a ring from radius 19.2 to 20 over `10.3 < z < 13`, the
/// ring extruded from a face of two circles and turned so its seam lies at
/// several angles from the cup's.
#[test]
fn a_cup_cut_by_an_extruded_ring_is_exact_at_any_seam() {
    let truth = PI * (1560.0 - 20.0f64.mul_add(20.0, -19.2 * 19.2) * 1.7);
    for degrees in [0.0, 30.0, 90.0, 200.0] {
        for pose in POSES {
            let label = format!("ring turned {degrees}, {pose}");
            let mut topo = Topology::new();
            let body = cup(&mut topo);
            let ring = extruded_ring(&mut topo, 19.2, 20.0, 10.3, 13.0, degrees);
            let (a, b) = (posed(&mut topo, body, pose), posed(&mut topo, ring, pose));
            let result = exact(&mut topo, BooleanOp::Cut, a, b, &label);
            let points = [
                (Point3::new(19.6, 0.0, 11.0), PointClassification::Outside),
                (Point3::new(-19.6, 0.0, 9.0), PointClassification::Inside),
                (Point3::new(0.0, 18.6, 11.0), PointClassification::Inside),
                (Point3::new(0.0, 0.0, 6.0), PointClassification::Outside),
            ];
            check(&topo, result, &label, truth, &points, pose);
        }
    }
}

/// A rod of radius 1 over `0 < z < 1` less the same rod over
/// `0.5 < z < 1.5`, turned about its axis.
#[test]
fn a_rod_less_a_turned_coaxial_rod_is_exact() {
    for degrees in [0.0, 30.0, 90.0, 200.0] {
        for pose in POSES {
            let label = format!("rod turned {degrees}, {pose}");
            let mut topo = Topology::new();
            let a = rod(&mut topo, 1.0, 0.0, 1.0, 0.0);
            let b = rod(&mut topo, 1.0, 0.5, 1.5, degrees);
            let (a, b) = (posed(&mut topo, a, pose), posed(&mut topo, b, pose));
            let result = exact(&mut topo, BooleanOp::Cut, a, b, &label);
            let points = [
                (Point3::new(0.5, 0.2, 0.25), PointClassification::Inside),
                (Point3::new(0.5, 0.2, 0.75), PointClassification::Outside),
            ];
            check(&topo, result, &label, PI * 0.5, &points, pose);
        }
    }
}
