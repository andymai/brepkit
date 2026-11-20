//! A sliver face on a spline surface's rim, meshed.
//!
//! A boolean can leave a face of no area: its outline one curve run there
//! and back, as two edges parameterised differently, both on one rim of
//! the face's B-spline surface. In (u, v) the two edges lie along one line,
//! so every sample of either lies on a chord of the other, and halving the
//! chords (to keep a chord's (u, v) line on its curve) never ends that: it
//! doubled the samples each round, a face of a few hundred samples reached
//! some 25,000, and the triangulation of that outline ran for half an hour
//! (a printed robot's leg, after a cut).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::Solid;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const POLE: [(f64, f64); 4] = [(0.0, 0.0), (60.0, 40.0), (120.0, -40.0), (180.0, 0.0)];

fn bezier(points: [(f64, f64); 4], weights: [f64; 4]) -> NurbsCurve {
    NurbsCurve::new(
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        points
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect(),
        weights.to_vec(),
    )
    .unwrap()
}

#[test]
fn a_sliver_along_a_spline_rim_keeps_few_samples() {
    // The rim v = 1 is the curve; the surface hangs below it.
    let net: Vec<Vec<Point3>> = POLE
        .iter()
        .map(|&(x, y)| vec![Point3::new(x, y, -20.0), Point3::new(x, y, 0.0)])
        .collect();
    let surface = NurbsSurface::new(
        3,
        1,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        net,
        vec![vec![1.0; 2]; 4],
    )
    .unwrap();
    // There along the rim, and back along it with its parameter moved by
    // weights in a geometric run (the same curve, sampled elsewhere).
    let there = bezier(POLE, [1.0; 4]);
    let mut back_points = POLE;
    back_points.reverse();
    let c = 0.6_f64;
    let back = bezier(back_points, [1.0, c, c * c, c * c * c]);

    let mut topo = Topology::new();
    let [a, b] = [Point3::new(0.0, 0.0, 0.0), Point3::new(180.0, 0.0, 0.0)]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let e0 = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(there)));
    let e1 = topo.add_edge(Edge::new(b, a, EdgeCurve::NurbsCurve(back)));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(surface)));

    // (The solid mesher samples each edge once for the faces sharing it,
    // and halves those shared samples: a shell of the sliver alone.)
    let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));

    let started = std::time::Instant::now();
    let mesh = tessellate_solid(&topo, solid, 0.5).unwrap();
    // Each edge holds the deflection in well under a hundred samples.
    assert!(
        mesh.positions.len() < 400,
        "{} points on a sliver of two edges ({:?})",
        mesh.positions.len(),
        started.elapsed()
    );
}
