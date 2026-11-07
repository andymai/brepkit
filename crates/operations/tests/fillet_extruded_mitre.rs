//! Two rounds along a prism's top outline meeting at a corner whose walls
//! are not square to each other, the corner's vertical edge left sharp.
//!
//! The rounds meet as a moulding mitred at the corner: each round is its
//! cross-section swept along its edge up to the plane bisecting the
//! corner, and the two meet along their section by that plane, from the
//! top face down to the pinch a radius below the corner. Before, only a
//! square corner was mitred, and other convex corners were closed with a
//! corner patch meant for concave corners, which was wrong.
//!
//! The prism is a kite: corner `V` of angle `theta` between the rounded
//! edges, the edges' far ends square to the walls after them, so each
//! round ends square on its end wall and notches it. Along edge `a`, a
//! point of the round's cross-section at inset `w` from its wall runs from
//! the bisector, `w cot(theta / 2)` from `V`, to the far end, `L` from `V`.
//! The cross-section (a square of side `r` less its quarter disc) has area
//! `r^2 (1 - pi / 4)` and first moment in `w` of `r^3 (5 / 6 - pi / 4)`,
//! so the two rounds take off
//! `2 L r^2 (1 - pi / 4) - 2 cot(theta / 2) r^3 (5 / 6 - pi / 4)`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const L: f64 = 10.0;
const H: f64 = 6.0;

/// The polygon `points` in `z = 0`, wound counter-clockwise, extruded `H`
/// up `z`.
fn prism(topo: &mut Topology, points: &[Point3]) -> SolidId {
    let ids: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], EdgeCurve::Line);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The edge of `solid` from `a` to `b` (either way round).
fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            (near(p, a) && near(q, b)) || (near(p, b) && near(q, a))
        })
        .expect("the edge")
}

/// The kite with its corner `V` at the origin, of angle `theta` between
/// its edges `V -> A` and `V -> B`, each `L` long, square to the walls
/// `A -> C` and `B -> C`.
fn kite(theta: f64) -> [Point3; 4] {
    let h = theta / 2.0;
    [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(L * h.cos(), -L * h.sin(), 0.0),
        Point3::new(L / h.cos(), 0.0, 0.0),
        Point3::new(L * h.cos(), L * h.sin(), 0.0),
    ]
}

#[test]
fn rounds_meeting_at_a_corner_not_square_are_mitred() {
    for theta_deg in [60.0_f64, 75.0, 120.0, 140.0] {
        let theta = theta_deg.to_radians();
        for r in [1.0, 2.0] {
            let case = format!("theta = {theta_deg}, r = {r}");
            let mut topo = Topology::new();
            let [v, b, _, a] = kite(theta);
            let solid = prism(&mut topo, &kite(theta));
            let up = Vec3::new(0.0, 0.0, H);
            let edges = [
                edge_between(&topo, solid, v + up, a + up),
                edge_between(&topo, solid, v + up, b + up),
            ];
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let faces_before = solid_faces(&topo, solid).unwrap().len();

            let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            assert_eq!(
                solid_faces(&topo, result.solid).unwrap().len(),
                faces_before + 2,
                "{case}: the two rounds alone are added, no corner patch"
            );

            let cot = 1.0 / (theta / 2.0).tan();
            let removed =
                2.0 * L * r * r * (1.0 - PI / 4.0) - 2.0 * cot * r.powi(3) * (5.0 / 6.0 - PI / 4.0);
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            assert!(
                (lost - removed).abs() <= 1e-6 * before,
                "{case}: lost {lost}, the mitred rounds {removed}"
            );
        }
    }
}
