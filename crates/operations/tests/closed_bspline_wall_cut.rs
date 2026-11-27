//! A prism on a closed B-spline profile as a file holds it: one wall closed
//! round its profile (a single rational B-spline closed along u), bounded by
//! its two rims and its seam each way, and two plane caps bounded by one rim
//! each. A box cut through its wall stays exact, as the same cut does on the
//! prism brepkit extrudes, whose wall comes in pieces.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_planar_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const HEIGHT: f64 = 10.0;

/// A closed cubic B-spline round an oval about the origin in z = 0, run
/// counter-clockwise seen from +z, clamped at its start on +x.
fn oval() -> NurbsCurve {
    let pts: Vec<Point3> = [
        (20.0, 0.0),
        (20.0, 12.0),
        (0.0, 14.0),
        (-20.0, 12.0),
        (-20.0, -12.0),
        (0.0, -14.0),
        (20.0, -12.0),
        (20.0, 0.0),
    ]
    .iter()
    .map(|&(x, y)| Point3::new(x, y, 0.0))
    .collect();
    let n = pts.len();
    let mut knots = vec![0.0; 4];
    #[allow(clippy::cast_precision_loss)]
    knots.extend((1..n - 3).map(|k| k as f64 / (n - 3) as f64));
    knots.extend([1.0; 4]);
    NurbsCurve::new(3, knots, pts, vec![1.0; n]).unwrap()
}

/// The prism with its wall one face closed round the profile.
fn closed_prism(topo: &mut Topology) -> SolidId {
    let curve = oval();
    let up = |p: Point3| Point3::new(p.x(), p.y(), p.z() + HEIGHT);
    let wall = NurbsSurface::new(
        curve.degree(),
        1,
        curve.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        curve
            .control_points()
            .iter()
            .map(|&p| vec![p, up(p)])
            .collect(),
        curve.weights().iter().map(|&w| vec![w, w]).collect(),
    )
    .unwrap();
    let top_curve = NurbsCurve::new(
        curve.degree(),
        curve.knots().to_vec(),
        curve.control_points().iter().map(|&p| up(p)).collect(),
        curve.weights().to_vec(),
    )
    .unwrap();
    let p0 = curve.control_points()[0];
    let a = topo.add_vertex(Vertex::new(p0, 1e-7));
    let b = topo.add_vertex(Vertex::new(up(p0), 1e-7));
    let low = topo.add_edge(Edge::new(a, a, EdgeCurve::NurbsCurve(curve)));
    let high = topo.add_edge(Edge::new(b, b, EdgeCurve::NurbsCurve(top_curve)));
    let seam = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
    let wall_wire = Wire::new(
        vec![
            OrientedEdge::new(low, true),
            OrientedEdge::new(seam, true),
            OrientedEdge::new(high, false),
            OrientedEdge::new(seam, false),
        ],
        true,
    )
    .unwrap();
    let wall_wire = topo.add_wire(wall_wire);
    // The profile runs counter-clockwise seen from +z, so the wall's
    // natural normal (along the profile, crossed with up) faces out.
    let wall = topo.add_face(Face::new(wall_wire, vec![], FaceSurface::Nurbs(wall)));
    let bottom_wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(low, false)], true).unwrap());
    let bottom = topo.add_face(Face::new(
        bottom_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let top_wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(high, true)], true).unwrap());
    let top = topo.add_face(Face::new(
        top_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: HEIGHT,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![bottom, wall, top]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// The prism brepkit extrudes from the same profile, its wall in pieces.
fn extruded_prism(topo: &mut Topology) -> SolidId {
    let curve = oval();
    let p0 = curve.control_points()[0];
    let v = topo.add_vertex(Vertex::new(p0, 1e-7));
    let e = topo.add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(curve)));
    let w = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
    let f = make_planar_face_from_wire(topo, w).unwrap();
    extrude(topo, f, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap()
}

fn cut_by_box(topo: &mut Topology, prism: SolidId, at: (f64, f64, f64)) -> SolidId {
    let block = make_box(topo, 20.0, 10.0, 10.0).unwrap();
    transform_solid(topo, block, &Mat4::translation(at.0, at.1, at.2)).unwrap();
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    boolean_with_options(topo, BooleanOp::Cut, prism, block, exact).unwrap()
}

/// A box through the closed wall, across a cap or between them, cuts as it
/// does the extruded prism: exact, valid, watertight, the same volume.
#[test]
fn a_box_cut_through_a_closed_bspline_wall_stays_exact() {
    for at in [
        (10.0, -5.0, 5.0),
        (10.0, -5.0, -5.0),
        (-30.0, 4.0, 2.0),
        (-5.0, 8.0, 2.0),
    ] {
        let mut topo = Topology::new();
        let prism = closed_prism(&mut topo);
        assert!(validate_solid(&topo, prism).unwrap().is_valid(), "{at:?}");
        let cut = cut_by_box(&mut topo, prism, at);
        let report = validate_solid(&topo, cut).unwrap();
        assert!(report.is_valid(), "{at:?}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, cut, 0.05).unwrap()),
            "{at:?}"
        );
        let volume = solid_volume(&topo, cut, 0.001).unwrap();

        let mut own = Topology::new();
        let prism = extruded_prism(&mut own);
        let cut = cut_by_box(&mut own, prism, at);
        let expected = solid_volume(&own, cut, 0.001).unwrap();
        assert!(
            (volume - expected).abs() < 1e-6 * expected,
            "{at:?}: {volume} against {expected}"
        );
    }
}
