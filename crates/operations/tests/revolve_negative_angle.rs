//! A negative revolve angle turns the profile the other way about the axis,
//! the same as the positive angle about the reversed axis: a U-bolt's arch,
//! a disc swept half a turn over the pipe it holds.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::revolve::revolve;
use brepkit_operations::sheet::revolve_wire;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_circle_edge;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The arch's tube: a disc of radius 4 in `z = 0`, 25.2 out along `x`.
const REACH: f64 = 25.2;
const TUBE: f64 = 4.0;

fn disc(topo: &mut Topology) -> FaceId {
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let rim = make_circle_edge(topo, Point3::new(REACH, 0.0, 0.0), normal, TUBE, 1e-7).unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ))
}

fn classify(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    p: Point3,
) -> PointClassification {
    classify_point(topo, solid, p, 0.01, 1e-7).unwrap()
}

#[test]
fn a_disc_revolved_a_negative_half_turn_arches_the_other_way() {
    // Pappus: the tube's area times the path of its centre.
    let expected = PI * TUBE * TUBE * PI * REACH;
    let top = Point3::new(0.0, 0.0, REACH);
    let bottom = Point3::new(0.0, 0.0, -REACH);
    for (angle, over, under) in [(-PI, top, bottom), (PI, bottom, top)] {
        let mut topo = Topology::new();
        let face = disc(&mut topo);
        let arch = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            angle,
        )
        .unwrap();
        let report = validate_solid(&topo, arch).unwrap();
        assert!(report.is_valid(), "angle {angle}: {:?}", report.issues);
        let volume = solid_volume(&topo, arch, 0.001).unwrap();
        assert!(
            (volume - expected).abs() < 1e-4 * expected,
            "angle {angle}: volume {volume} should be {expected}"
        );
        assert_eq!(
            classify(&topo, arch, over),
            PointClassification::Inside,
            "angle {angle}"
        );
        assert_eq!(
            classify(&topo, arch, under),
            PointClassification::Outside,
            "angle {angle}"
        );
    }
}

#[test]
fn a_negative_whole_turn_is_the_whole_ring() {
    let expected = PI * TUBE * TUBE * TAU * REACH;
    let mut topo = Topology::new();
    let face = disc(&mut topo);
    let ring = revolve(
        &mut topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        -TAU,
    )
    .unwrap();
    assert!(validate_solid(&topo, ring).unwrap().is_valid());
    let volume = solid_volume(&topo, ring, 0.001).unwrap();
    assert!(
        (volume - expected).abs() < 1e-4 * expected,
        "volume {volume} should be {expected}"
    );
    assert_eq!(solid_faces(&topo, ring).unwrap().len(), 1);
}

#[test]
fn a_revolve_by_nothing_or_by_more_than_a_turn_is_refused() {
    for angle in [0.0, -0.0, TAU + 0.1, -TAU - 0.1, f64::NAN] {
        let mut topo = Topology::new();
        let face = disc(&mut topo);
        let result = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            angle,
        );
        assert!(
            matches!(result, Err(OperationsError::InvalidInput { .. })),
            "angle {angle}: {result:?}"
        );
    }
}

#[test]
fn a_wire_revolved_a_negative_quarter_turn_sweeps_the_other_quarter() {
    // A line 3 from the axis and 4 long sweeps a quarter of a cylinder's wall.
    let expected = 3.0 * FRAC_PI_2 * 4.0;
    for (angle, side) in [(-FRAC_PI_2, -1.0), (FRAC_PI_2, 1.0)] {
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(3.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(3.0, 0.0, 4.0), 1e-7));
        let line = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(line, true)], false).unwrap());
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            angle,
        )
        .unwrap();
        let faces = topo.shell(shell).unwrap().faces().to_vec();
        let area: f64 = faces
            .iter()
            .map(|&f| face_area(&topo, f, 1e-4).unwrap())
            .sum();
        assert!(
            (area - expected).abs() < 1e-6 * expected,
            "angle {angle}: area {area} should be {expected}"
        );
        for &f in &faces {
            let face = topo.face(f).unwrap();
            for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                for v in [edge.start(), edge.end()] {
                    let y = topo.vertex(v).unwrap().point().y();
                    assert!(y * side >= -1e-9, "angle {angle}: a vertex at y = {y}");
                }
            }
        }
    }
}
