//! A long bar rounded on both edges of one end: each round's contacts end
//! on the side faces' long edges a radius from the end, which the trim must
//! split there. Their ends were found by a fixed forty-step ternary search
//! over the edge's length, which on a bar some thousands long stopped up to
//! 3e-6 off the contact's end, past the 1e-6 it is held to, so the split was
//! skipped: each long face kept its whole long edge beside a connector to
//! the corner, the side faces could not be notched, and the incidence audit
//! refused the fillet. Whether it did turned on where the search landed, so
//! a 4000 bar failed and an 8000 one passed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};

const WIDTH: f64 = 64.5;
const THICKNESS: f64 = 28.0;
const R: f64 = 13.0;

/// Rounds the two edges along `x` at the `z = 0` end of a
/// `WIDTH x THICKNESS x length` box by `R`, the box turned about `x` by
/// `turn` first, and checks the result.
fn round_the_end(length: f64, turn: f64) {
    let mut topo = Topology::new();
    let bar = make_box(&mut topo, WIDTH, THICKNESS, length).unwrap();
    let ends: Vec<_> = solid_edges(&topo, bar)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let p = topo.vertex(edge.start()).unwrap().point();
            let q = topo.vertex(edge.end()).unwrap().point();
            (p.x() - q.x()).abs() > 1.0 && p.z().abs() < 1e-9 && q.z().abs() < 1e-9
        })
        .collect();
    assert_eq!(ends.len(), 2);
    transform_solid(&mut topo, bar, &Mat4::rotation_x(turn)).unwrap();
    let result = fillet_v2(&mut topo, bar, &ends, R).unwrap();
    let case = format!("length {length}, turned {turn}");
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 8, "{case}");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    let truth = (2.0 * (1.0 - PI / 4.0) * R * R).mul_add(-WIDTH, WIDTH * THICKNESS * length);
    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - truth).abs() < 1e-9 * truth,
        "{case}: volume {vol}, truth {truth}"
    );
}

#[test]
fn a_long_bar_rounded_on_both_edges_of_its_end() {
    for length in [200.0, 4000.0, 8000.0, 11260.0, 30000.0] {
        round_the_end(length, 0.0);
        round_the_end(length, 1.2566);
    }
}
