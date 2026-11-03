//! A ring between radii 4 and 5 split along its diameter into two halves of
//! two lines and two arcs each, both revolved part of a turn about the same
//! axis, then fused: the halves meet along their flat sides and close each
//! other's half circles into whole ones. In every pose the fuse is exact,
//! valid and watertight, holds Pappus's volume, and keeps the ring's material.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
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
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const POSES: [&str; 4] = ["upright", "turned", "mirrored", "scaled"];

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

/// The half of the ring between radii 4 and 5 about (30, 0, 0) in the plane
/// `x = 30` on the side `z >= 0` when `up`, else `z <= 0`, revolved `angle`
/// about the line through (30, 20, 0) along `z`.
fn half_ring(topo: &mut Topology, up: bool, angle: f64) -> SolidId {
    let centre = Point3::new(30.0, 0.0, 0.0);
    let at =
        |topo: &mut Topology, y: f64| topo.add_vertex(Vertex::new(Point3::new(30.0, y, 0.0), 1e-7));
    let (outer_a, outer_b, inner_b, inner_a) =
        (at(topo, 5.0), at(topo, -5.0), at(topo, -4.0), at(topo, 4.0));
    // Each arc runs counterclockwise about its normal from +y to -y, so the
    // normal's sign picks the half it sweeps.
    let normal = Vec3::new(if up { 1.0 } else { -1.0 }, 0.0, 0.0);
    let outer = Circle3D::new(centre, normal, 5.0).unwrap();
    let inner = Circle3D::new(centre, normal, 4.0).unwrap();
    let edges = [
        (
            topo.add_edge(Edge::new(outer_a, outer_b, EdgeCurve::Circle(outer))),
            true,
        ),
        (
            topo.add_edge(Edge::new(outer_b, inner_b, EdgeCurve::Line)),
            true,
        ),
        (
            topo.add_edge(Edge::new(inner_a, inner_b, EdgeCurve::Circle(inner))),
            false,
        ),
        (
            topo.add_edge(Edge::new(inner_a, outer_a, EdgeCurve::Line)),
            true,
        ),
    ];
    let wire = topo.add_wire(
        Wire::new(
            edges
                .iter()
                .map(|&(e, forward)| OrientedEdge::new(e, forward))
                .collect(),
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(1.0, 0.0, 0.0),
            d: 30.0,
        },
    ));
    revolve(
        topo,
        face,
        Point3::new(30.0, 20.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        angle,
    )
    .unwrap()
}

/// A point at distance `from_axis` from the tubes' centre circle toward the
/// axis and `z` above the profile's plane, `angle` round the revolution.
fn at(angle: f64, from_axis: f64, z: f64) -> Point3 {
    let reach = 20.0 - from_axis;
    Point3::new(30.0 + reach * angle.sin(), 20.0 - reach * angle.cos(), z)
}

#[test]
fn two_revolved_half_rings_fuse_into_the_ring() {
    for angle in [PI / 2.0, PI, 1.7 * PI] {
        for pose in POSES {
            let label = format!("angle {angle:.3}, {pose}");
            let mut topo = Topology::new();
            let upper = half_ring(&mut topo, true, angle);
            let lower = half_ring(&mut topo, false, angle);
            let (a, b) = (posed(&mut topo, upper, pose), posed(&mut topo, lower, pose));
            // The fuse drops the flat faces where the halves meet and keeps
            // every tube face and the four half annuli capping the ends.
            let mut tube_faces = 0;
            for solid in [a, b] {
                for face in solid_faces(&topo, solid).unwrap() {
                    if matches!(topo.face(face).unwrap().surface(), FaceSurface::Torus(_)) {
                        tube_faces += 1;
                    }
                }
            }
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
            assert_eq!(
                mesh_fallback_count(),
                before,
                "{label}: fell back to a mesh"
            );
            let faces = solid_faces(&topo, result).unwrap();
            assert_eq!(faces.len(), tube_faces + 4, "{label}: faces");
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let truth = 9.0 * PI * 20.0 * angle;
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let half = 0.5 * angle;
            for (p, want) in [
                (at(half, 4.5, 0.0), PointClassification::Inside),
                (at(half, 0.0, 4.5), PointClassification::Inside),
                (at(half, 0.0, -4.5), PointClassification::Inside),
                (at(half, 0.0, 0.0), PointClassification::Outside),
                (at(half, 0.0, 5.5), PointClassification::Outside),
            ] {
                let got = classify_point(&topo, result, placed(p, pose), 0.01, 1e-7).unwrap();
                assert_eq!(got, want, "{label}: {p:?} reads {got:?}");
            }
        }
    }
}
