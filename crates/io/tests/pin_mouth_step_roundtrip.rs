//! A pin cut across a bore's mouth leaves edges that are pieces of the
//! closed loop where the pin's wall meets the bore's, some stored from the
//! loop's later point back to its earlier one. Written to STEP, such an edge
//! must say it runs against its curve (`same_sense = .F.`): written with the
//! curve, a reader takes the loop from the edge's start the other way round,
//! and the part it reads back is a different solid. (The builder now stores
//! a piece of a closed loop with its curve, so this cut may have no such edge
//! left; the writer's own tests cover one, and this one checks the part
//! reads back whole.)

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

#[test]
fn edges_running_against_their_loop_are_written_against_it() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 40.0, 6.0, 70.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-20.0, -360.0, 751.0)).unwrap();
    let hole = make_cylinder(&mut topo, 2.25, 10.0).unwrap();
    transform_solid(
        &mut topo,
        hole,
        &(Mat4::translation(0.0, -352.0, 796.0) * Mat4::rotation_x(FRAC_PI_2)),
    )
    .unwrap();
    let part = boolean(&mut topo, BooleanOp::Cut, block, hole).unwrap();
    let pin = make_cylinder(&mut topo, 1.7, 444.0).unwrap();
    transform_solid(&mut topo, pin, &Mat4::translation(0.0, -355.0, 706.2)).unwrap();
    let before = mesh_fallback_count();
    let cut = boolean(&mut topo, BooleanOp::Cut, part, pin).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let volume = solid_volume(&topo, cut, 0.001).unwrap();

    let text = write_step(&topo, &[cut]).unwrap();
    let mut back = Topology::new();
    let read = read_step(&text, &mut back).unwrap();
    assert_eq!(read.len(), 1);
    let again = solid_volume(&back, read[0], 0.001).unwrap();
    assert!(
        (again - volume).abs() < 1e-6 * volume,
        "read back {again}, written {volume}"
    );
}
