//! A long box (one side four times another) rounded on its edges has the
//! rolling ball's corners, as a near-cube has: `A' B' C' + 2r (A' B' + B' C'
//! + C' A') + pi r^2 (A' + B' + C') + 4/3 pi r^3` for the shrunk sides
//! `A' = a - 2r` and so on, when every edge is rounded.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;

/// The edges along `y` of the box `a x b x c` from `make_box`.
fn edges_along_y(topo: &Topology, edges: &[EdgeId]) -> Vec<usize> {
    edges
        .iter()
        .enumerate()
        .filter(|&(_, &e)| {
            let edge = topo.edge(e).unwrap();
            let p = topo.vertex(edge.start()).unwrap().point();
            let q = topo.vertex(edge.end()).unwrap().point();
            (p.y() - q.y()).abs() > 1.0
        })
        .map(|(i, _)| i)
        .collect()
}

/// Rounds the edges of the box `a x b x c` by `r`, with or without one
/// edge along `y`, and checks the result has `balls` ball corners, is valid
/// and watertight, and has the volume `truth`.
fn assert_rounds(drop: bool, truth: f64, balls: usize) {
    let (a, b, c, r) = (A, B, C, R);
    let mut topo = Topology::new();
    let block = make_box(&mut topo, a, b, c).unwrap();
    let mut edges = solid_edges(&topo, block).unwrap();
    if drop {
        let along_y = edges_along_y(&topo, &edges);
        edges.remove(along_y[0]);
    }
    let result = fillet_v2(&mut topo, block, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let spheres = solid_faces(&topo, result.solid)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .count();
    assert_eq!(spheres, balls, "sharp edge {drop}: the ball corners");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "sharp edge {drop}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "sharp edge {drop}: mesh not watertight"
    );
    let vol = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (vol - truth).abs() < 1e-7 * truth,
        "sharp edge {drop}: volume {vol}, truth {truth}"
    );
}

const A: f64 = 33.46;
const B: f64 = 27.33;
const C: f64 = 111.26;
const R: f64 = 2.0;

/// The box shrunk by `R` on every side, swept by a ball of radius `R`.
fn all_rounded() -> f64 {
    let (sa, sb, sc) = (A - 2.0 * R, B - 2.0 * R, C - 2.0 * R);
    2.0f64.mul_add(R * sc.mul_add(sa, sa.mul_add(sb, sb * sc)), sa * sb * sc)
        + PI * R * R * (sa + sb + sc)
        + 4.0 / 3.0 * PI * R * R * R
}

/// The long box rounded on every edge. Its corners had been left without
/// set-backs (stripes more than twice as long as another at a corner
/// were), the corner patch then joined sections a radius from the vertex,
/// and the mesh was open with 184 too much volume.
#[test]
fn a_long_box_rounded_on_every_edge() {
    assert_rounds(false, all_rounded(), 8);
}
