//! Two prisms of one half disc stacked end to end, fused. Where they meet,
//! each brings an arc and a diameter between the same two points, and the
//! two copies of each lie on each other: the fuse is exact, valid and
//! watertight, and holds the half cylinder's volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
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

/// The half disc of radius 2 on the side `y >= 0` at height `z`, extruded 1
/// up.
fn half_disc_prism(topo: &mut Topology, z: f64) -> SolidId {
    let right = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, z), 1e-7));
    let left = topo.add_vertex(Vertex::new(Point3::new(-2.0, 0.0, z), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let arc = topo.add_edge(Edge::new(right, left, EdgeCurve::Circle(circle)));
    let diameter = topo.add_edge(Edge::new(left, right, EdgeCurve::Line));
    let wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(arc, true),
                OrientedEdge::new(diameter, true),
            ],
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap()
}

#[test]
fn two_stacked_half_disc_prisms_fuse_into_one() {
    for pose in ["upright", "turned", "mirrored"] {
        let mut topo = Topology::new();
        let lower = half_disc_prism(&mut topo, 0.0);
        let upper = half_disc_prism(&mut topo, 1.0);
        let (a, b) = match pose {
            "turned" => {
                let turn = Mat4::translation(0.4, -0.3, 0.2)
                    * Mat4::rotation_x(0.7)
                    * Mat4::rotation_z(0.3);
                transform_solid(&mut topo, lower, &turn).unwrap();
                transform_solid(&mut topo, upper, &turn).unwrap();
                (lower, upper)
            }
            "mirrored" => {
                let (at, normal) = (Point3::new(0.5, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
                (
                    mirror(&mut topo, lower, at, normal).unwrap(),
                    mirror(&mut topo, upper, at, normal).unwrap(),
                )
            }
            _ => (lower, upper),
        };
        let before = mesh_fallback_count();
        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        assert_eq!(mesh_fallback_count(), before, "{pose}: fell back to a mesh");
        let faces = solid_faces(&topo, result).unwrap().len();
        assert!(faces <= 6, "{pose}: {faces} faces");
        assert!(
            validate_solid(&topo, result).unwrap().is_valid(),
            "{pose}: invalid"
        );
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{pose}: open or non-manifold mesh");
        let truth = 4.0 * PI;
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{pose}: volume {volume}, truth {truth}"
        );
    }
}
