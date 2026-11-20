//! A box fused into a holed plate read from STEP, its side cutting one of
//! the plate's holes along a chord off the hole's axis. The hole's wall is
//! a cylinder pointing out of the plate, as a file often writes it: its
//! near rim, walked against its curve, is laid a turn off the window its
//! seam copies bound, and the sections ending on it sit at their own u in
//! neither. Laid out in one window the wall splits, and the fuse is exact.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_box_cutting_a_hole_along_a_chord_fuses_exactly() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "hole_chord_plate.step");
    let block = load(&mut topo, "hole_chord_box.step");
    let (v_plate, v_block) = (
        solid_volume(&topo, plate, 0.01).unwrap(),
        solid_volume(&topo, block, 0.01).unwrap(),
    );

    let res = boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();

    // every hole's wall still a cylinder: not the mesh fallback
    let walls = solid_faces(&topo, res)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(walls, 9, "the fuse fell back to a mesh");
    // The box overlaps the plate over 20 x 5 x 20, less the holes there:
    // the r = 2.25 hole at (0, 753) whole, and the two at x = ±12.124 cut
    // along chords at x = ±10, each the circle's segment beyond its chord,
    // all 5 deep.
    let (r, d) = (2.25_f64, 12.124_355_652_982_143 - 10.0);
    let segment = r * r * (d / r).acos() - d * (r * r - d * d).sqrt();
    let disc = std::f64::consts::PI * r * r;
    let overlap = 20.0 * 5.0 * 20.0 - (disc + 2.0 * segment) * 5.0;
    let expect = v_plate + v_block - overlap;
    let vol = solid_volume(&topo, res, 0.01).unwrap();
    assert!(
        (vol - expect).abs() < 1e-4 * expect,
        "fused volume {vol}, expected {expect}"
    );
}
