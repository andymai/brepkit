//! A ring-shaped custom bin from the gridfinity tool
//! (`binGenerator.export.interiorFillet.test.ts`, "a ring-shaped custom bin
//! keeps its hole open"): the bin's block, a rounded square round a rounded
//! square hole, less its cavity, a thinner ring standing on a 1.2 floor.
//!
//! The block's top face is a ring round the hole, and the cavity's two walls
//! cut it in two section loops, both enclosing the hole. The hole went to the
//! outer loop, so the strip between it and the inner loop came out as a disc
//! over the hole, its sample point classified the hole, and the cut left the
//! hole's rim free and fell back to a mesh.
//!
//! Data: `ring_bin_block.bin` and `ring_bin_cavity.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let fallbacks = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, op, a, b).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "{op:?} fell back"
    );
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert!(
        uses.values().all(|&n| n == 2),
        "{op:?} left free or over-shared edges"
    );
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{op:?}: mesh not watertight"
    );
    result
}

#[test]
fn a_ring_bin_less_its_cavity_keeps_the_rim_round_its_hole() {
    let mut topo = Topology::new();
    let block = load(&mut topo, "ring_bin_block.bin");
    let cavity = load(&mut topo, "ring_bin_cavity.bin");
    for operand in [block, cavity] {
        let report = validate_solid(&topo, operand).unwrap();
        assert!(report.is_valid(), "operand: {:?}", report.issues);
    }
    let cut = exact(&mut topo, BooleanOp::Cut, block, cavity);
    let fused = exact(&mut topo, BooleanOp::Fuse, block, cavity);
    let common = exact(&mut topo, BooleanOp::Intersect, block, cavity);

    // Closed forms: the block's ring area (outer 125.5 square less its 3.75
    // corners, less the 42.5 hole and its 3.75 corners) times 16.25; the
    // cavity's ring (123.1 square, 2.55 corners, less a 44.9 hole, 3.75
    // corners) times 16.05, of which 15.05 lies in the block.
    let corner = |side: f64, r: f64| (4.0 - std::f64::consts::PI).mul_add(-r * r, side * side);
    let block_ring = corner(125.5, 3.75) - corner(42.5, 3.75);
    let cavity_ring = corner(123.1, 2.55) - corner(44.9, 3.75);
    let (v_block, v_cavity) = (block_ring * 16.25, cavity_ring * 16.05);
    let overlap = cavity_ring * 15.05;
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    for (label, got, want) in [
        ("cut", volume(cut), v_block - overlap),
        ("fuse", volume(fused), v_block + v_cavity - overlap),
        ("intersect", volume(common), overlap),
    ] {
        assert!(
            (got - want).abs() <= 1e-6 * want,
            "{label} volume {got}, expected {want}"
        );
    }
}
