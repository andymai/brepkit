//! `compound_cut` when no cut stays exact: a block with two bores, cut by a
//! pin across each bore's mouth. Each pin's own cut falls back to a mesh;
//! the batch must be cut once, with a single fallback.
//!
//! Kept in its own file: the fallback counter is process-wide.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A block (40 x 6 x 70) less two bores of radius 2.25 through it, at
/// `x = -10` and `x = 10`, their seams turned 270 degrees, and a pin of
/// radius 1.7 standing across each bore's mouth, its seam turned 180
/// degrees: each pin's cut falls back to the mesh boolean
/// (`a_pin_across_a_bores_mouth_with_both_seams_at_the_fold` in
/// `pin_through_hole_mouth.rs`, its bore at `x = 0`).
fn block_and_pins(topo: &mut Topology) -> (SolidId, [SolidId; 2]) {
    let mut part = make_box(topo, 40.0, 6.0, 70.0).unwrap();
    transform_solid(topo, part, &Mat4::translation(-20.0, -360.0, 751.0)).unwrap();
    let mut pins = Vec::new();
    for x in [-10.0, 10.0] {
        let bore = make_cylinder(topo, 2.25, 10.0).unwrap();
        let across = Mat4::translation(x, -352.0, 796.0)
            * Mat4::rotation_x(std::f64::consts::FRAC_PI_2)
            * Mat4::rotation_z(270f64.to_radians());
        transform_solid(topo, bore, &across).unwrap();
        part = boolean::boolean(topo, BooleanOp::Cut, part, bore).unwrap();
        let pin = make_cylinder(topo, 1.7, 444.0).unwrap();
        let standing = Mat4::translation(x, -355.0, 706.2) * Mat4::rotation_z(180f64.to_radians());
        transform_solid(topo, pin, &standing).unwrap();
        pins.push(pin);
    }
    (part, [pins[0], pins[1]])
}

#[test]
fn compound_cut_takes_one_fallback_when_no_cut_stays_exact() {
    for k in 0..2 {
        let mut topo = Topology::new();
        let (part, pins) = block_and_pins(&mut topo);
        let before = boolean::mesh_fallback_count();
        boolean::boolean(&mut topo, BooleanOp::Cut, part, pins[k]).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count() - before,
            1,
            "premise: each pin's own cut must fall back, or this pins nothing"
        );
    }

    let mut topo = Topology::new();
    let (part, pins) = block_and_pins(&mut topo);
    let before = boolean::mesh_fallback_count();
    boolean::compound_cut(&mut topo, part, &pins, BooleanOptions::default()).unwrap();
    assert_eq!(boolean::mesh_fallback_count() - before, 1);
}
