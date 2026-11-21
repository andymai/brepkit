//! A plate with round holes read from STEP, fused with a box sunk 1 mm into
//! it whose top face crosses one hole. That hole's wall keeps a full band
//! below the box and the part above the box's top: one piece, low in one
//! stretch of u and full height elsewhere. The box's bottom face cuts the
//! wall in an arc leaving u = 0 towards -u, which was laid out a turn the
//! long way round, and the piece's interior sample, taken from a polygon
//! that folded its half-turn rim halves, landed inside the box. The wall
//! was dropped and the fuse fell back to a mesh.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::oriented_solid_volume;
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
fn a_hole_wall_crossed_by_a_sunk_boxs_top_keeps_its_outside_part() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "hole_band_plate.step");
    let block = load(&mut topo, "hole_band_box.step");

    let res = boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();

    let faces = solid_faces(&topo, res).unwrap();
    let walls = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        walls,
        4,
        "a hole wall was dropped or the fuse fell back ({} faces)",
        faces.len()
    );
    // the reference result's volume
    let vol = oriented_solid_volume(&topo, res, 0.01).unwrap();
    assert!((vol - 5_731.261_3).abs() < 1e-5 * vol, "fused volume {vol}");
}
