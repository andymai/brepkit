//! A long bar rounded on both edges of one end: each round's contacts end
//! on the side faces' long edges a radius from the end, which the trim must
//! split there. A contact's end must meet the edge within 1e-6, so its
//! parameter on an edge some thousands long is found to the edge's own
//! precision. Short of that, each long face keeps its whole long edge
//! beside a connector to the corner, the side faces cannot be notched, and
//! the incidence audit refuses the fillet. A long edge stored as a straight
//! B-spline running against the edge, as a file can store it, has its
//! parameter running down the edge, and is searched the same way.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};

const WIDTH: f64 = 64.5;
const THICKNESS: f64 = 28.0;
const R: f64 = 13.0;

/// Rounds the two edges along `x` at the `z = 0` end of a
/// `WIDTH x THICKNESS x length` box by `R`, the box turned about `x` by
/// `turn` first, and checks the result. With `as_splines`, the box's long
/// edges are straight B-splines running from each edge's end to its start.
fn round_the_end(length: f64, turn: f64, as_splines: bool) {
    let mut topo = Topology::new();
    let bar = make_box(&mut topo, WIDTH, THICKNESS, length).unwrap();
    if as_splines {
        for edge_id in solid_edges(&topo, bar).unwrap() {
            let edge = topo.edge(edge_id).unwrap();
            let p = topo.vertex(edge.start()).unwrap().point();
            let q = topo.vertex(edge.end()).unwrap().point();
            if (p.z() - q.z()).abs() > 1.0 {
                let backward =
                    NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![q, p], vec![1.0, 1.0])
                        .unwrap();
                topo.edge_mut(edge_id)
                    .unwrap()
                    .set_curve(EdgeCurve::NurbsCurve(backward));
            }
        }
    }
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
    let case = format!("length {length}, turned {turn}, as splines {as_splines}");
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
        round_the_end(length, 0.0, false);
        round_the_end(length, 1.2566, false);
    }
}

#[test]
fn a_long_bar_with_backward_spline_edges_rounded_at_its_end() {
    for length in [200.0, 4000.0] {
        round_the_end(length, 0.0, true);
        round_the_end(length, 1.2566, true);
    }
}
