//! Captured-operand pin for a ring-shaped custom bin's interior fillet
//! (the gridfinity layout tool's `binGenerator.export.interiorFillet`, "a
//! ring-shaped custom bin keeps its hole open"): the bin less its pocket
//! fused with the fillet material, whose straight wall fillets meet each
//! pocket corner's cylinder only at the corner's seam.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

#[test]
fn a_ring_bin_takes_its_fillet_material_exactly() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "ring_bin_base.bin");
    let material = load(&mut topo, "ring_bin_fillet_material.bin");

    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, bin, material).unwrap();
    let bin_only = boolean(&mut topo, BooleanOp::Cut, bin, material).unwrap();
    assert_eq!(mesh_fallback_count(), before, "a boolean fell back");
    assert!(validate_solid(&topo, fused).unwrap().is_valid());

    // The fuse is the bin's own part plus the whole fillet material.
    let volume = |s: SolidId| oriented_solid_volume(&topo, s, 0.001).unwrap();
    let (got, want) = (volume(fused), volume(bin_only) + volume(material));
    assert!((got - want).abs() <= 0.05, "fuse {got}, expected {want}");
}
