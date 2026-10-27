//! The planar chamfer, the flat-bevel fillet and the variable-radius fillet
//! rebuild a solid's outer shell alone. A 10 cube with a 3 cube cavity keeps
//! its cavity through each, and each refuses a cut that would reach a cavity
//! 0.05 behind the front face.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

/// A 10 cube less a 3 cube cavity at `corner`, and its top edge along x at
/// y = 0.
fn hollow_block(topo: &mut Topology, corner: (f64, f64, f64)) -> (SolidId, EdgeId) {
    let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let cube = make_box(topo, 3.0, 3.0, 3.0).unwrap();
    transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
    let hollow = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let edge = solid_edges(topo, hollow)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    (hollow, edge)
}

const CLEAR: (f64, f64, f64) = (3.5, 3.5, 3.5);
const NEAR: (f64, f64, f64) = (3.5, 0.05, 6.5);

/// The result keeps one valid cavity, and its volume.
fn kept(topo: &Topology, result: SolidId) -> f64 {
    assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    solid_volume(topo, result, 0.001).unwrap()
}

#[test]
fn a_chamfer_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
    let volume = kept(&topo, result);
    assert!((volume - (1000.0 - 27.0 - 5.0)).abs() < 1e-6, "{volume}");
}

#[test]
fn a_chamfer_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    assert!(chamfer(&mut topo, hollow, &[edge], 1.0).is_err());
}

/// The hollow block loses what a plain block does to the same fillet.
#[test]
fn a_variable_fillet_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result =
        fillet_variable(&mut topo, hollow, &[(edge, FilletRadiusLaw::Constant(1.0))]).unwrap();
    let volume = kept(&topo, result);

    let mut plain_topo = Topology::new();
    let plain = make_box(&mut plain_topo, 10.0, 10.0, 10.0).unwrap();
    let plain_edge = solid_edges(&plain_topo, plain)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = plain_topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = plain_topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    let plain_result = fillet_variable(
        &mut plain_topo,
        plain,
        &[(plain_edge, FilletRadiusLaw::Constant(1.0))],
    )
    .unwrap();
    let plain_volume = solid_volume(&plain_topo, plain_result, 0.001).unwrap();
    assert!(plain_volume < 1000.0, "{plain_volume}");
    assert!(
        (volume + 27.0 - plain_volume).abs() < 1e-6,
        "{volume} vs {plain_volume}"
    );
}

#[test]
fn a_variable_fillet_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    let law = FilletRadiusLaw::Linear {
        start: 0.5,
        end: 1.0,
    };
    assert!(fillet_variable(&mut topo, hollow, &[(edge, law)]).is_err());
}

/// The flat bevel takes off less than the chamfer's triangle across the
/// edge and more than nothing.
#[test]
#[allow(deprecated)]
fn a_bevel_fillet_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = brepkit_operations::fillet::fillet(&mut topo, hollow, &[edge], 1.0).unwrap();
    let volume = kept(&topo, result);
    assert!((968.0..973.0).contains(&volume), "{volume}");
}

#[test]
#[allow(deprecated)]
fn a_bevel_fillet_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    assert!(brepkit_operations::fillet::fillet(&mut topo, hollow, &[edge], 1.0).is_err());
}

/// The result's cavity is its own: moving the result in place leaves the
/// input's cavity where it was.
#[test]
fn a_kept_cavity_is_the_results_own() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
    transform_solid(&mut topo, result, &Mat4::translation(0.0, 0.0, 100.0)).unwrap();
    let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
    for &face in topo.shell(cavity).unwrap().faces() {
        for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
            let z = topo.vertex(v).unwrap().point().z();
            assert!(
                (3.5 - 1e-9..=6.5 + 1e-9).contains(&z),
                "input cavity moved to z {z}"
            );
        }
    }
    let report = validate_solid(&topo, hollow).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}
