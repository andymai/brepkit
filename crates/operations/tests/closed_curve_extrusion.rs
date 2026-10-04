//! Profiles bounded by one closed curve, extruded: a crush-rib magnet bore as
//! the layout tool draws it (one spline interpolated through a ring of points
//! that returns to its start, a wave of eight ribs about a nominal radius) and
//! an elliptical boss. Each extrudes to exact faces that mesh closed and cut
//! a block exactly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::fitting::interpolate;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_face_from_wire, make_nurbs_edge};
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const HEIGHT: f64 = 2.4;

/// The ribbed ring: radius 3.25 at the peaks, 0.3 deeper at the troughs,
/// sampled 64 times from the wave's inflection and closed on its start.
fn ring() -> Vec<Point3> {
    let (radius, depth, ribs) = (3.25_f64, 0.3_f64, 8.0_f64);
    let phase = PI / (2.0 * ribs);
    (0..=64)
        .map(|i| {
            let theta = TAU.mul_add(f64::from(i % 64) / 64.0, phase);
            let r = (depth / 2.0).mul_add((ribs * theta).cos(), radius - depth / 2.0);
            Point3::new(r * theta.cos(), r * theta.sin(), 0.0)
        })
        .collect()
}

/// The bore, and the area its spline encloses.
fn bore(topo: &mut Topology) -> (SolidId, f64) {
    let points = ring();
    let curve = interpolate(&points, 3).unwrap();
    let (t0, t1) = curve.domain();
    let area = 0.5
        * (0..20_000)
            .map(|k| {
                let p = curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 20_000.0);
                let q = curve.evaluate(t0 + (t1 - t0) * f64::from(k + 1) / 20_000.0);
                p.x().mul_add(q.y(), -(q.x() * p.y()))
            })
            .sum::<f64>();
    let edge = make_nurbs_edge(topo, points[0], points[64], curve, 1e-7);
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    let face = make_face_from_wire(topo, wire).unwrap();
    (
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap(),
        area,
    )
}

/// Valid, watertight, exact (a NURBS face, a handful of faces rather than a
/// mesh fallback's hundreds of planes), and of the given volume.
fn assert_exact(topo: &Topology, solid: SolidId, volume: f64, label: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{label}"
    );
    let faces = solid_faces(topo, solid).unwrap();
    assert!(
        faces.len() < 60
            && faces
                .iter()
                .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_))),
        "{label}: {} faces",
        faces.len()
    );
    let measured = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 1e-4 * volume,
        "{label}: {measured} against {volume}"
    );
}

#[test]
fn a_closed_spline_edge_has_one_vertex() {
    let mut topo = Topology::new();
    let points = ring();
    let curve = interpolate(&points, 3).unwrap();
    let edge = make_nurbs_edge(&mut topo, points[0], points[64], curve, 1e-7);
    let edge = topo.edge(edge).unwrap();
    assert_eq!(edge.start(), edge.end());
}

/// The bore cuts the block from its underside, as a cavity, and through its
/// top: 1.4 of its 2.4 height inside the block.
#[test]
fn a_ribbed_bore_cuts_a_block_exactly() {
    for (lift, inside) in [(0.0, HEIGHT), (1.0, HEIGHT), (3.6, 1.4)] {
        let label = format!("bore at {lift}");
        let mut topo = Topology::new();
        let (bore, area) = bore(&mut topo);
        assert_exact(&topo, bore, area * HEIGHT, &label);
        transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, lift)).unwrap();
        let block = make_box(&mut topo, 12.0, 12.0, 5.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(-6.0, -6.0, 0.0)).unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
        assert_exact(&topo, cut, area.mul_add(-inside, 720.0), &label);
    }
}

#[test]
fn an_elliptical_boss_cuts_a_block_exactly() {
    let mut topo = Topology::new();
    let ellipse = brepkit_math::curves::Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();
    let vertex = topo.add_vertex(Vertex::new(ellipse.evaluate(0.0), 1e-7));
    let edge = topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Ellipse(ellipse)));
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    let face = make_face_from_wire(&mut topo, wire).unwrap();
    let boss = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap();
    let volume = PI * 6.0 * HEIGHT;
    assert_exact(&topo, boss, volume, "boss");
    let block = make_box(&mut topo, 12.0, 12.0, 5.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-6.0, -6.0, 0.0)).unwrap();
    let cut = boolean(&mut topo, BooleanOp::Cut, block, boss).unwrap();
    assert_exact(&topo, cut, 720.0 - volume, "cut");
}
