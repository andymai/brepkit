//! A round a little wider than the bend at its outline's tip.
//!
//! The slab's outline is a quadratic B-spline nose, from `(-A, 0)` up to a
//! tip at `(0, H / 2)` and down to `(A, 0)`, closed by a straight edge, and
//! extruded `T` up `z`. At its tip the nose bends with radius `A² / H`. The
//! nose's top edge is rounded: with a radius below the tip's bend the ball
//! rolls round the tip; with one past it, it cannot, and its contact on the
//! top face runs back past the tip, loops and crosses itself. The round now
//! cuts that crossing out: up to the section through the crossing point the
//! round is as walked, on from the other section through it too, and between
//! them the band runs from that point down to the wall's contact, tangent to
//! the wall. It had been refused ("turns back on itself").
//!
//! The volume a round takes off grows with its radius as its section's piece
//! does, `r² (1 - π/4)`, along a contact that barely changes: past the tip's
//! bend it keeps growing at that rate, within the little the tip's cut adds.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const A: f64 = 30.0;
const H: f64 = 40.0;
const T: f64 = 30.0;

/// The tip's bend radius.
const BEND: f64 = A * A / H;

fn slab(topo: &mut Topology) -> SolidId {
    let (left, right) = (Point3::new(-A, 0.0, 0.0), Point3::new(A, 0.0, 0.0));
    let nose = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![right, Point3::new(0.0, H, 0.0), left],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    let [vl, vr] = [left, right].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let edges = [
        Edge::new(vl, vr, EdgeCurve::Line),
        Edge::new(vr, vl, EdgeCurve::NurbsCurve(nose)),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let up = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal: up, d: 0.0 },
    ));
    extrude(topo, face, up, T).unwrap()
}

/// The nose's top edge.
fn nose_top(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && (topo.vertex(edge.start()).unwrap().point().z() - T).abs() < 1e-9
        })
        .expect("the nose's top edge")
}

/// The volume the round takes off, with its result checked.
fn taken_off(r: f64) -> f64 {
    let mut topo = Topology::new();
    let solid = slab(&mut topo);
    let edge = nose_top(&topo, solid);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    before - solid_volume(&topo, result.solid, 1e-3).unwrap()
}

#[test]
fn a_round_past_its_tip_s_bend_cuts_its_contact_s_crossing() {
    let piece = |r: f64| r * r * (1.0 - std::f64::consts::FRAC_PI_4);
    let below = [BEND - 1.5, BEND - 0.5].map(|r| (r, taken_off(r)));
    // Below the bend the removed volume runs along a nearly fixed length.
    let along = (below[1].1 - below[0].1) / (piece(below[1].0) - piece(below[0].0));
    for r in [BEND + 0.5, BEND + 1.5, BEND + 3.5] {
        let lost = taken_off(r);
        let want = below[1].1 + along * (piece(r) - piece(below[1].0));
        assert!(
            (lost - want).abs() <= 0.02 * want,
            "r = {r}: lost {lost}, the rate below the bend gives {want}"
        );
    }
}
