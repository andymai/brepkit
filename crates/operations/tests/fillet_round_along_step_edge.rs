//! A round whose width lands exactly on a step: a block's edge rounded
//! where a notch cut into one of its faces stops a radius from the edge, so
//! the round's contact on that face runs along the notch wall's own edges.
//! The face beyond the contact keeps no spur along those edges and back
//! along the contact: the round meets the notch wall's faces along the
//! wall's own edges, and the mesh is closed along them.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const R: f64 = 6.0;
const HEIGHT: f64 = 20.0;

/// A box `dx x dy x dz` with its low corner at `at`.
fn block(topo: &mut Topology, at: [f64; 3], size: [f64; 3]) -> SolidId {
    let solid = make_box(topo, size[0], size[1], size[2]).unwrap();
    transform_solid(topo, solid, &Mat4::translation(at[0], at[1], at[2])).unwrap();
    solid
}

/// Rounds the block's edge along `z` at `x = 0, y = 20` by `R` after the
/// notches `cuts` (each a low corner and a size) are cut from it, and checks
/// the result against `truth`.
fn round_beside(cuts: &[([f64; 3], [f64; 3])], truth: f64) {
    let mut topo = Topology::new();
    let mut part = block(&mut topo, [0.0; 3], [40.0, 20.0, HEIGHT]);
    for &(at, size) in cuts {
        let notch = block(&mut topo, at, size);
        part = boolean(&mut topo, BooleanOp::Cut, part, notch).unwrap();
    }
    let edge: Vec<_> = solid_edges(&topo, part)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.x().abs() < 1e-9 && (p.y() - 20.0).abs() < 1e-9
            })
        })
        .collect();
    assert_eq!(edge.len(), 1, "{cuts:?}");
    let result = fillet_v2(&mut topo, part, &edge, R).unwrap();
    assert!(result.failed.is_empty(), "{cuts:?}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{cuts:?}: {:?}", report.issues);
    for deflection in [0.5, 0.05] {
        let mesh = tessellate_solid(&topo, result.solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "{cuts:?}: mesh not watertight at {deflection}"
        );
    }
    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - truth).abs() < 1e-9 * truth,
        "{cuts:?}: volume {vol}, truth {truth}"
    );
}

/// The material the round takes off the corner, whole along the edge (the
/// notches stand clear of the corner's square).
fn rounded_off() -> f64 {
    (1.0 - PI / 4.0) * R * R * HEIGHT
}

#[test]
fn a_round_whose_contact_runs_along_a_notch_wall() {
    // One notch from the bottom, its wall a radius from the edge: the
    // contact runs along the wall's edge, then on through the face.
    let notch = ([R, 14.0, -5.0], [24.0, 11.0, 22.0]);
    round_beside(
        &[notch],
        40.0 * 20.0 * HEIGHT - 24.0 * 6.0 * 17.0 - rounded_off(),
    );
}

#[test]
fn a_round_whose_contact_runs_along_a_stepped_notch_wall() {
    // A deeper notch at the bottom under the first: the wall's edge along
    // the contact is in two pieces, one to each notch's wall.
    let upper = ([R, 14.0, -5.0], [24.0, 11.0, 22.0]);
    let lower = ([R, 10.0, -5.0], [24.0, 15.0, 8.0]);
    let truth = 40.0 * 20.0 * HEIGHT - 24.0 * 6.0 * 17.0 - 24.0 * 4.0 * 3.0 - rounded_off();
    round_beside(&[upper, lower], truth);
}
