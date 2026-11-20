//! A part with a ring of small round holes read from STEP, cut by a big
//! cylinder whose wall passes close to the holes' seams. Each crossed hole's
//! wall splits into pieces whose rims are laid out in half-turn halves; the
//! interior sample of the largest piece came from a polygon that folded
//! those halves, landed in its neighbour, and the piece was dropped (32 free
//! edges, so the cut fell back to a mesh).

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
fn a_ring_of_holes_cut_near_their_seams_stays_exact() {
    let mut topo = Topology::new();
    let part = load(&mut topo, "seam_ring_part.step");
    let tool = load(&mut topo, "seam_ring_tool.step");

    let res = boolean(&mut topo, BooleanOp::Cut, part, tool).unwrap();

    let faces = solid_faces(&topo, res).unwrap();
    let walls = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert!(
        faces.len() < 100 && walls >= 20,
        "a wall piece was dropped or the cut fell back ({} faces, {walls} cylinders)",
        faces.len()
    );
    let vol = oriented_solid_volume(&topo, res, 0.01).unwrap();
    // the reference result's volume
    assert!((vol - 8_973.578_6).abs() < 2e-4 * vol, "cut volume {vol}");
}
