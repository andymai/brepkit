//! A blind bore re-cut through its end: a part with a channel under it and
//! a bore of radius 1.45 along the channel, its axis 0.095 above the
//! channel's ceiling, ending blind at the channel's end wall (y = -48), and
//! a pin along the bore's axis cut on through the end block to y = -51.
//! The end wall's top edge is notched where the bore stands above the
//! ceiling, so the notch arc and the pin's arc below the ceiling share both
//! ends, and the wall's piece outside the pin is bounded by the lower arc.
//! The notch's pcurve traced the shorter way round (the lower arc), the face
//! trace paired the notch with the pin's arc as twins, and the walker
//! followed the notch, so the wall came out whole and the cut fell back to
//! the mesh boolean. Both operands are read from STEP, as a file gives them
//! (built from primitives, the part does not reproduce it); the cut is
//! exact, valid, and removes just the pin's run through the end block.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;
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
fn a_blind_bore_recut_through_its_end_is_exact() {
    let mut topo = Topology::new();
    let part = load(&mut topo, "blind_bore_recut_part.step");
    let pin = load(&mut topo, "blind_bore_recut_pin.step");
    let part_volume = solid_volume(&topo, part, 0.001).unwrap();

    let before = mesh_fallback_count();
    let cut = boolean(&mut topo, BooleanOp::Cut, part, pin).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    // The pin's full disc through the 3 of end block the bore stopped short of.
    let want = part_volume - PI * 1.45 * 1.45 * 3.0;
    let volume = solid_volume(&topo, cut, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}
