//! A solid's bounding box reaches the farthest points of its trimmed sphere
//! and torus faces, and no farther.

#![allow(clippy::unwrap_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_bounding_box;
use brepkit_operations::primitives::{make_box, make_sphere, make_torus};
use brepkit_operations::revolve::revolve;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The box holds the solid's true extent and reaches past it by at most
/// `slack` (a NURBS edge is bounded by its Bezier control points).
fn assert_box(topo: &Topology, solid: SolidId, want: [[f64; 3]; 2], slack: f64, label: &str) {
    let bb = solid_bounding_box(topo, solid).unwrap();
    let min = [bb.min.x(), bb.min.y(), bb.min.z()];
    let max = [bb.max.x(), bb.max.y(), bb.max.z()];
    for k in 0..3 {
        let short = (min[k] - want[0][k]).max(want[1][k] - max[k]);
        let past = (want[0][k] - min[k]).max(max[k] - want[1][k]);
        assert!(
            short < 1e-9 && past < slack,
            "{label}: box {min:?} to {max:?}, want {want:?}"
        );
    }
}

/// A quarter ball of radius 3 swept from a half-disc a quarter turn about z.
#[test]
fn a_quarter_ball_is_boxed_to_its_quarter() {
    let mut topo = Topology::default();
    let origin = Point3::new(0.0, 0.0, 0.0);
    let circle = Circle3D::new(origin, Vec3::new(0.0, -1.0, 0.0), 3.0).unwrap();
    let south = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, -3.0), 1e-7));
    let north = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 3.0), 1e-7));
    let arc = topo.add_edge(Edge::new(south, north, EdgeCurve::Circle(circle)));
    let axis = topo.add_edge(Edge::new(north, south, EdgeCurve::Line));
    let edges = vec![OrientedEdge::new(arc, true), OrientedEdge::new(axis, true)];
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, -1.0, 0.0),
        d: 0.0,
    };
    let profile = topo.add_face(Face::new(wire, vec![], plane));
    let z = Vec3::new(0.0, 0.0, 1.0);
    let quarter = revolve(&mut topo, profile, origin, z, FRAC_PI_2).unwrap();
    assert_box(
        &topo,
        quarter,
        [[0.0, 0.0, -3.0], [3.0, 3.0, 3.0]],
        1e-9,
        "quarter ball",
    );

    let ball = make_sphere(&mut topo, 3.0, 16).unwrap();
    assert_box(&topo, ball, [[-3.0; 3], [3.0; 3]], 1e-9, "ball");
}

/// The torus of radii 3 and 1 within a box from x = `x0` to 4.5 and
/// |y| <= 1: the piece reaches the tube's outer equator at x = 4 between
/// its section edges, and its top where the tube is still past `x0`.
#[test]
fn a_torus_piece_is_boxed_to_its_tube() {
    for (x0, top) in [(2.0, 1.0), (2.5, 1.0), (3.5, 3.0_f64.sqrt() / 2.0)] {
        let mut topo = Topology::default();
        let torus = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();
        let block = make_box(&mut topo, 4.5 - x0, 2.0, 4.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(x0, -1.0, -2.0)).unwrap();
        let piece = boolean(&mut topo, BooleanOp::Intersect, torus, block).unwrap();
        let want = [[x0, -1.0, -top], [4.0, 1.0, top]];
        assert_box(&topo, piece, want, 1e-4, &format!("piece past x = {x0}"));
    }
    let mut topo = Topology::default();
    let torus = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();
    assert_box(
        &topo,
        torus,
        [[-4.0, -4.0, -1.0], [4.0, 4.0, 1.0]],
        1e-9,
        "torus",
    );
}
