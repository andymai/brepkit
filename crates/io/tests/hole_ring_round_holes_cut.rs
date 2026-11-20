//! A cylinder cut through a plate read from STEP whose ring of round holes
//! its circle crosses. On the plate's faces the circle's arcs run from hole
//! to hole, each ending on a hole's rim: the holes must join the
//! arrangement there, or every arc dangles, is dropped, and the face comes
//! out whole with the holes attached as they were.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_cylinder_through_a_ring_of_round_holes_cuts_exactly() {
    let mut topo = Topology::new();
    let part = load(&mut topo, "hole_ring_cut_part.step");
    let tool = load(&mut topo, "hole_ring_cut_tool.step");

    let res = boolean(&mut topo, BooleanOp::Cut, part, tool).unwrap();

    // a few dozen exact faces, not the mesh fallback's thousand and more
    let faces = solid_faces(&topo, res).unwrap().len();
    assert!(faces < 100, "the cut fell back to a mesh ({faces} faces)");
    // the reference result's volume
    let vol = solid_volume(&topo, res, 0.01).unwrap();
    assert!(
        (vol - 10_041.288_311_272).abs() < 1e-6 * vol,
        "cut volume {vol}"
    );
}
