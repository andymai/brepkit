//! A box sunk 1 into a drilled plate's face, its floor and roof crossing
//! holes: inside its footprint each hole is filled to the box's depth, and a
//! hole the floor crosses keeps, below the floor, a short strip of its wall
//! and the floor's chord strip across it. That pair meets the rest through
//! edges sharing their ends with the hole's arc and the box edge's chord, and
//! the shell grouping, counting edges by their ends, filled the count with the
//! arc and the chord first and left each pair as a cavity of its own. Both
//! operands are read from STEP (built from primitives the fuse does not hit
//! it); the fuse is exact, valid and at the operands' volumes less their
//! overlap.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_box_sunk_into_a_drilled_plate_fuses_exactly() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "box_sunk_plate.step");
    let block = load(&mut topo, "box_sunk_box.step");
    let (plate_volume, block_volume) = (
        solid_volume(&topo, plate, 0.001).unwrap(),
        solid_volume(&topo, block, 0.001).unwrap(),
    );
    let common = boolean(&mut topo, BooleanOp::Intersect, plate, block).unwrap();
    let overlap = solid_volume(&topo, common, 0.001).unwrap();

    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let want = plate_volume + block_volume - overlap;
    let volume = solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}
