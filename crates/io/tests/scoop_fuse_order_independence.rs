//! Regression: the scoop-fuse-into-compartmented-bin result must be the same
//! whether the bin or the scoop is operand A.
//!
//! The scoop's bottom face (z=0) is coincident with the compartmented bin's
//! bottom cap. The same-domain selector deduplicates that coincident pair to
//! one representative face. The representative must be picked by geometry (the
//! larger, containing face) rather than by which operand is A — otherwise a
//! Fuse with the bin as A keeps the full cap while the swapped order keeps a
//! tiny scoop fragment and drops the cap, leaving the floor non-manifold and
//! forcing the mesh-boolean fallback (which tessellates the corner cylinders
//! away). Both orders must keep the 8 analytic corner cylinders.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn read_one(name: &str, topo: &mut Topology) -> SolidId {
    let text = std::fs::read_to_string(fixture(name)).unwrap();
    let solids = brepkit_io::step::reader::read_step(&text, topo).unwrap();
    assert_eq!(solids.len(), 1, "expected exactly one solid in {name}");
    solids[0]
}

fn cylinder_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count()
}

fn build_comp_bin(topo: &mut Topology) -> (SolidId, SolidId) {
    let bin_box = read_one("scoop_bin_box.step", topo);
    let cavity_0 = read_one("scoop_cavity_0.step", topo);
    let cavity_1 = read_one("scoop_cavity_1.step", topo);
    let scoop = read_one("scoop_scoop_0.step", topo);
    let comp_bin = boolean(topo, BooleanOp::Cut, bin_box, cavity_0).unwrap();
    let comp_bin = boolean(topo, BooleanOp::Cut, comp_bin, cavity_1).unwrap();
    (comp_bin, scoop)
}

/// Both `Fuse(comp_bin, scoop)` and `Fuse(scoop, comp_bin)` must keep the 8
/// analytic corner cylinders — i.e. neither order falls back to mesh boolean.
/// Before the same-domain representative fix, the swapped order dropped the
/// coincident bottom cap and the result mesh-fell-back (0 cylinders).
#[test]
fn scoop_fuse_is_operand_order_independent() {
    let cyl_bin_first = {
        let mut topo = Topology::new();
        let (comp_bin, scoop) = build_comp_bin(&mut topo);
        let r = boolean(&mut topo, BooleanOp::Fuse, comp_bin, scoop).unwrap();
        cylinder_count(&topo, r)
    };
    let cyl_scoop_first = {
        let mut topo = Topology::new();
        let (comp_bin, scoop) = build_comp_bin(&mut topo);
        let r = boolean(&mut topo, BooleanOp::Fuse, scoop, comp_bin).unwrap();
        cylinder_count(&topo, r)
    };

    assert_eq!(
        cyl_bin_first, 8,
        "Fuse(bin, scoop) must keep 8 analytic corner cylinders"
    );
    assert_eq!(
        cyl_scoop_first, 8,
        "Fuse(scoop, bin) must keep 8 analytic corner cylinders \
         (operand-order independence: the coincident bottom cap must survive)"
    );
}
