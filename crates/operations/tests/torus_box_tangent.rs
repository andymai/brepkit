//! A box whose face touches a torus at one point on its outer equator: the
//! torus piece inside the box is sampled at that point, on the box, and its
//! other samples have to stay on the piece (whose loops run across the
//! tube's seam) for it to read as inside.

#![allow(clippy::unwrap_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// Booleans of the torus (major radius 3, minor 1) with a box spanning
/// x from 2.5 to `x1`.
fn run(op: BooleanOp, x1: f64) -> (Vec<&'static str>, f64) {
    let mut topo = Topology::default();
    let torus = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();
    let block = make_box(&mut topo, x1 - 2.5, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(2.5, -1.0, -2.0)).unwrap();
    let result = boolean(&mut topo, op, torus, block).unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    let kinds = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect();
    (kinds, solid_volume(&topo, result, 0.002).unwrap())
}

#[test]
fn a_box_touching_the_outer_equator_cuts_exactly() {
    // The box reaching past x = 4 holds the same piece of the torus, so the
    // touching box's results differ from it only by the box's own volume.
    let past = 0.1 * 2.0 * 4.0;
    for (op, extra) in [
        (BooleanOp::Intersect, 0.0),
        (BooleanOp::Cut, 0.0),
        (BooleanOp::Fuse, -past),
    ] {
        let (kinds, touching) = run(op, 4.0);
        let (_, beyond) = run(op, 4.1);
        assert!(
            kinds.contains(&"torus") && kinds.len() < 20,
            "{op:?}: {} faces {kinds:?}",
            kinds.len()
        );
        assert!(
            (touching - (beyond + extra)).abs() < 1e-6 * beyond,
            "{op:?}: volume {touching} against {}",
            beyond + extra
        );
    }
}
