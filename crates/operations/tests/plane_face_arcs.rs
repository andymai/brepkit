//! The boolean engine's ray cast reads a plane face's circular edges exactly:
//! an open arc as its chord and the circular segment between them, a closed
//! circle as its disc.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// A plate with a 120 degree concave notch (an arc about `(0, -1)`, radius
/// 2) and a finger entering the notch through its mouth, so the plate's own
/// boundary crosses the arc's chord, extruded 0.2 and turned. Every point of
/// a grid through the finger reads inside, and a small box cut out of the
/// finger stays exact and removes its volume.
#[test]
fn a_finger_across_a_notchs_chord_reads_inside() {
    let mut topo = Topology::new();
    let root3 = 3.0_f64.sqrt();
    let outline = [
        (4.0, 6.0),
        (-4.0, 6.0),
        (-4.0, 0.0),
        (-root3, 0.0),
        (root3, 0.0),
        (2.5, 0.0),
        (2.5, -1.0),
        (0.3, -1.0),
        (0.3, 0.6),
        (-0.3, 0.6),
        (-0.3, -1.5),
        (3.0, -1.5),
        (3.0, 0.0),
        (4.0, 0.0),
    ];
    let corners: Vec<_> = outline
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for (i, &a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % corners.len()];
        let curve = if i == 3 {
            let notch = Circle3D::new(Point3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0), 2.0);
            EdgeCurve::Circle(notch.unwrap())
        } else {
            EdgeCurve::Line
        };
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(a, b, curve)),
            true,
        ));
    }
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    let face = topo.add_face(Face::new(wire, vec![], plane));
    let plate = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
    let volume = solid_volume(&topo, plate, 0.01).unwrap();
    let pose = Mat4::rotation_x(-0.6155) * Mat4::rotation_z(std::f64::consts::FRAC_PI_4);
    transform_solid(&mut topo, plate, &pose).unwrap();
    for i in 0..40 {
        for j in 0..40 {
            let x = 0.58f64.mul_add(f64::from(i) / 39.0, -0.29);
            let y = 1.5f64.mul_add(f64::from(j) / 39.0, -0.95);
            let p = pose.mul_point(Point3::new(x, y, 0.1));
            let got = brepkit_algo::classifier::classify_ray_cast(&topo, plate, p).unwrap();
            assert_eq!(got, brepkit_algo::FaceClass::Inside, "({x}, {y}, 0.1)");
        }
    }
    let cutter = make_box(&mut topo, 0.3, 0.3, 2.0).unwrap();
    transform_solid(
        &mut topo,
        cutter,
        &(pose * Mat4::translation(-0.15, 0.1, -1.0)),
    )
    .unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, plate, cutter).unwrap();
    assert!(
        solid_faces(&topo, result).unwrap().len() <= 24,
        "fell back to a mesh"
    );
    let cut = solid_volume(&topo, result, 0.01).unwrap();
    let truth = 0.3f64.mul_add(-0.3 * 0.2, volume);
    assert!((cut - truth).abs() < 1e-6, "volume {cut}, truth {truth}");
}

/// A cylinder turned off its axis reads points just under its top cap, out
/// to 0.02 from its rim, inside: the cap is its disc, not a polygon of chords.
#[test]
fn a_turned_cylinder_reads_its_cap_to_the_rim() {
    let mut topo = Topology::new();
    let cylinder = make_cylinder(&mut topo, 5.0, 10.0).unwrap();
    let pose = Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    transform_solid(&mut topo, cylinder, &pose).unwrap();
    for i in 0..72 {
        for j in 0..6 {
            for k in 0..10 {
                let angle = std::f64::consts::TAU * f64::from(i) / 72.0;
                let rho = 0.01f64.mul_add(f64::from(j), 4.93);
                let z = 0.003f64.mul_add(-f64::from(k), 9.98);
                let p = pose.mul_point(Point3::new(rho * angle.cos(), rho * angle.sin(), z));
                let got = brepkit_algo::classifier::classify_ray_cast(&topo, cylinder, p).unwrap();
                assert_eq!(got, brepkit_algo::FaceClass::Inside, "{p:?}");
            }
        }
    }
}
