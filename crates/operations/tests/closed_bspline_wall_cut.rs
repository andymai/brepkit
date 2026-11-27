//! A prism on a closed B-spline profile as a file holds it: one wall closed
//! round its profile (a single rational B-spline closed along u), bounded by
//! its two rims and its seam each way, and two plane caps bounded by one rim
//! each. A band of a longer closed wall bounds only its own part of it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
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

/// The prism with its wall one face closed round the profile. The wall's
/// surface reaches `beyond` past each cap, so with `beyond > 0` the face is
/// a band of it between two closed curves round its middle.
fn closed_prism(topo: &mut Topology, beyond: f64) -> SolidId {
    let curve = oval();
    let up = |p: Point3| Point3::new(p.x(), p.y(), p.z() + HEIGHT);
    let at = |p: Point3, z: f64| Point3::new(p.x(), p.y(), z);
    let wall = NurbsSurface::new(
        curve.degree(),
        1,
        curve.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        curve
            .control_points()
            .iter()
            .map(|&p| vec![at(p, -beyond), at(p, HEIGHT + beyond)])
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

/// A band of a longer closed wall, between two closed curves round its
/// middle, bounds only its own part of the wall: points past the caps, where
/// the wall's surface runs on, classify outside.
#[test]
fn a_band_of_a_closed_bspline_wall_bounds_only_its_part() {
    use brepkit_operations::classify::{PointClassification, classify_point};
    let mut topo = Topology::new();
    let prism = closed_prism(&mut topo, 5.0);
    assert!(validate_solid(&topo, prism).unwrap().is_valid());
    for (p, inside) in [
        (Point3::new(0.0, 0.0, 5.0), true),
        (Point3::new(15.0, 3.0, 9.0), true),
        (Point3::new(0.0, 0.0, 12.0), false),
        (Point3::new(0.0, 0.0, -2.0), false),
        (Point3::new(25.0, 0.0, 5.0), false),
        (Point3::new(5.0, 2.0, 13.0), false),
    ] {
        let got = classify_point(&topo, prism, p, 0.01, 1e-7).unwrap();
        let expected = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(got, expected, "{p:?}");
    }
}
