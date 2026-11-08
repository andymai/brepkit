//! A round along a curved B-spline edge: a walked band through its
//! sections.
//!
//! The part is a prism on a quarter sector whose arc is one parabolic
//! B-spline span, closed by two radii, and extruded `H` up `z`. The arc's
//! top edge is rounded: the band between the top and the B-spline wall is
//! walked and fitted, and it ends square on the two radial walls.
//!
//! The band took the walked sections as its control points, so along a
//! curved edge it cut inside them: its contact on the wall stood inside the
//! wall, and the part came out 0.2 light at r = 1. It now
//! interpolates them.
//!
//! The volume taken off: the top is a plane, the wall stands square to it
//! and the round ends square on the radial walls, so the round takes the
//! corner's section, `(1 - pi/4) r^2`, along the edge, less the inset's
//! shortening on the curve's inner side: `(5/6 - pi/4) r^3` for each radian
//! the edge turns.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const H: f64 = 6.0;
const R: f64 = 20.0;

fn arc() -> NurbsCurve {
    let p = |x: f64, y: f64| Point3::new(x * R, y * R, 0.0);
    NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)],
        vec![1.0; 3],
    )
    .unwrap()
}

/// The arc's length and the plan's area, the sector closed at the origin.
fn arc_length_and_area() -> (f64, f64) {
    let curve = arc();
    let n = 30_000;
    let (mut length, mut area) = (0.0, 0.0);
    let mut last = curve.evaluate(0.0);
    for i in 1..=n {
        let q = curve.evaluate(f64::from(i) / f64::from(n));
        length += (q - last).length();
        area += 0.5 * (last.x() * q.y() - q.x() * last.y());
        last = q;
    }
    (length, area)
}

fn round_the_arc(r: f64) {
    let mut topo = Topology::new();
    let [start, end, centre] = [
        Point3::new(R, 0.0, 0.0),
        Point3::new(0.0, R, 0.0),
        Point3::new(0.0, 0.0, 0.0),
    ]
    .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let edges = [
        Edge::new(start, end, EdgeCurve::NurbsCurve(arc())),
        Edge::new(end, centre, EdgeCurve::Line),
        Edge::new(centre, start, EdgeCurve::Line),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    let solid = extrude(&mut topo, face, normal, H).unwrap();
    // The arc's top edge.
    let top: Vec<_> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && [edge.start(), edge.end()]
                    .iter()
                    .all(|&v| (topo.vertex(v).unwrap().point().z() - H).abs() < 1e-9)
        })
        .collect();
    assert_eq!(top.len(), 1);
    let result = fillet_v2(&mut topo, solid, &top, r).unwrap();
    let case = format!("r = {r}");
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 6, "{case}");
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    let (length, area) = arc_length_and_area();
    let removed = (1.0 - PI / 4.0) * r * r * length - (5.0 / 6.0 - PI / 4.0) * r.powi(3) * 0.5 * PI;
    let expected = H * area - removed;
    let volume = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (volume - expected).abs() < 1e-4 * removed,
        "{case}: volume {volume}, expected {expected} ({removed} taken off)"
    );
}

#[test]
fn a_round_along_a_curved_spline_edge_takes_off_its_corner() {
    for r in [1.0, 2.0] {
        round_the_arc(r);
    }
}
