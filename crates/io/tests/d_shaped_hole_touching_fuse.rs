//! A plate 0.41 thick whose side face has a D-shaped hole (a flat and an
//! arc between the same two vertices), fused with a ring sector that only
//! touches it, along a short wall and a face the two share a plane with.
//!
//! The face splitter dropped any hole of fewer than three vertices without
//! a closed edge, so the side face, split by the touching arc along its own
//! outline, came out whole, the hole capped. The hole's wall then had no
//! rim to join, was dropped with the face across its top, and the fuse was
//! 55% heavy, silently: the result was valid. A two-vertex hole with a
//! curved edge bounds an area and is kept.
//!
//! The operands are read from STEP, written by brepkit's own writer.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{name}.step"));
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn touching_fuse_keeps_a_d_shaped_hole() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "d_hole_plate");
    let ring = load(&mut topo, "d_hole_ring");
    let (a, b) = (
        solid_volume(&topo, plate, 1e-3).unwrap(),
        solid_volume(&topo, ring, 1e-3).unwrap(),
    );
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, ring).unwrap();
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert!(is_watertight(
        &tessellate_solid(&topo, fused, 0.01).unwrap()
    ));
    // The plate's eight faces and the ring's six, less the plate's short
    // wall, pressed wholly against the ring's inner wall. Without the hole,
    // its wall and the plate's top beside it went too.
    assert_eq!(solid_faces(&topo, fused).unwrap().len(), 13);
    // They only touch, so the fuse holds exactly both.
    let v = solid_volume(&topo, fused, 1e-3).unwrap();
    assert!(
        (v - (a + b)).abs() <= 1e-6 * (a + b),
        "fuse {v}, operands {a} + {b}"
    );
}
