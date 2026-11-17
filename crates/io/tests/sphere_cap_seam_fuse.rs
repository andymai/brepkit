//! A part read from STEP with a spherical cap, fused with a box away from
//! it. The cap's wire runs its rim and its seam meridian up to the pole and
//! back down, as a file writes a cap. Two cleanups read that seam as an
//! out-and-back spur: one stripped it, the other dropped the cap as a slit
//! face, and the shell came out open.

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
fn a_spherical_caps_seam_to_its_pole_survives_a_fuse() {
    let mut topo = Topology::new();
    let part = load(&mut topo, "sphere_cap_part.step");
    let block = load(&mut topo, "sphere_cap_box.step");

    let res = boolean(&mut topo, BooleanOp::Fuse, part, block).unwrap();

    let faces = solid_faces(&topo, res).unwrap();
    assert!(
        faces.len() < 100
            && faces
                .iter()
                .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_))),
        "the cap was dropped or the fuse fell back ({} faces)",
        faces.len()
    );
    // the reference result's volume
    let vol = solid_volume(&topo, res, 0.01).unwrap();
    assert!(
        (vol - 373_412.050_258_555).abs() < 1e-6 * vol,
        "fused volume {vol}"
    );
}
