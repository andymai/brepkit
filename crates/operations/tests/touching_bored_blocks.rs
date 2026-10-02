//! Two blocks meeting face to face at `x = 0`, each bored along `x` by the
//! same radius 1 cylinder, the second block's top edge crossing the bore: a
//! hinge's knuckles end to end with the pin's bore through both. They only
//! touch, so the fuse adds their volumes and the cut leaves the first whole,
//! exactly, in four turns about the bore.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A block from `x0` for 4 along `x`, 6 wide in `y`, from `z = -3` up to
/// `ztop`, bored by a radius 1 cylinder on the `x` axis.
fn bored_block(topo: &mut Topology, x0: f64, ztop: f64) -> SolidId {
    let block = make_box(topo, 4.0, 6.0, ztop + 3.0).unwrap();
    transform_solid(topo, block, &Mat4::translation(x0, -3.0, -3.0)).unwrap();
    let bore = make_cylinder(topo, 1.0, 6.0).unwrap();
    let along_x = Mat4::translation(x0 - 1.0, 0.0, 0.0) * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(topo, bore, &along_x).unwrap();
    boolean(topo, BooleanOp::Cut, block, bore).unwrap()
}

/// The bored block's volume: its box less the part of the bore's disc below
/// `ztop`, over its length.
fn bored_volume(ztop: f64) -> f64 {
    let d = ztop.clamp(-1.0, 1.0);
    let below = PI - (d.acos() - d * (1.0 - d * d).sqrt());
    4.0 * 6.0 * (ztop + 3.0) - 4.0 * below
}

fn op_is_exact(op: BooleanOp, ztop: f64, expected: f64) {
    for k in 0..4 {
        let mut topo = Topology::new();
        let a = bored_block(&mut topo, 0.0, 3.0);
        let b = bored_block(&mut topo, -4.0, ztop);
        let turn = Mat4::rotation_x(f64::from(k) * 0.7);
        transform_solid(&mut topo, a, &turn).unwrap();
        transform_solid(&mut topo, b, &turn).unwrap();
        let before = mesh_fallback_count();
        let result = boolean(&mut topo, op, a, b).unwrap();
        assert_eq!(mesh_fallback_count(), before, "mesh fallback, turn {k}");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "turn {k}: {:?}", report.issues);
        let vol = solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            (vol - expected).abs() < 1e-6 * expected,
            "turn {k}: volume {vol}, expected {expected}"
        );
    }
}

#[test]
fn touching_bored_blocks_fuse_exactly() {
    op_is_exact(BooleanOp::Fuse, 0.5, bored_volume(3.0) + bored_volume(0.5));
}

#[test]
fn a_bored_block_touching_one_whose_top_is_below_the_bore_axis_cuts_nothing() {
    op_is_exact(BooleanOp::Cut, -0.4, bored_volume(3.0));
}
