//! A cylinder cut from a part read from STEP, its wall running through a
//! ring of small holes whose axes are parallel to its own. Two cylinders
//! with parallel axes meet in straight lines along them: exact, they split
//! the small holes' walls and lie on the big wall end to end. Marched, each
//! line came back as overlapping fitted pieces short of both rims, and the
//! walls split into nothing.

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
fn a_cylinder_through_a_ring_of_parallel_holes_cuts_exactly() {
    let mut topo = Topology::new();
    let part = load(&mut topo, "ring_cut_part.step");
    let tool = load(&mut topo, "ring_cut_tool.step");

    let res = boolean(&mut topo, BooleanOp::Cut, part, tool).unwrap();

    // 33 faces as the reference result has them: not the mesh fallback
    assert_eq!(solid_faces(&topo, res).unwrap().len(), 33, "face count");
    // the reference result's volume
    let vol = solid_volume(&topo, res, 0.01).unwrap();
    assert!(
        (vol - 13_634.301_853_595).abs() < 1e-6 * vol,
        "cut volume {vol}"
    );
}
