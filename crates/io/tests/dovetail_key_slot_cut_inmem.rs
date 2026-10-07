//! A baseplate cut by its fourth dovetail key slot, from the gridfinity
//! tool's `baseplateGenerator.scenario.dovetailKey.test.ts`.
//!
//! The plate (566 faces) already carries the first three slots; the tool is a
//! vertical key-slot prism (z -5.65 to 1) with rounded corners. One r 0.4
//! corner cylinder meets a cell's 45 degree pocket cone only in its last
//! 0.02 mm, where the cone reaches the plate's top: of that section's samples
//! only the last lies on the cylinder's quarter face.
//!
//! Data: `dovetail_key_slot_plate.bin` and `dovetail_key_slot_tool.bin`.

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
fn a_key_slot_cut_through_a_pocket_cone_rim_is_exact() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "dovetail_key_slot_plate.bin");
    let tool = load(&mut topo, "dovetail_key_slot_tool.bin");
    let cut = exact(&mut topo, BooleanOp::Cut, plate, tool);
    let common = exact(&mut topo, BooleanOp::Intersect, plate, tool);
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    let removed = volume(plate) - volume(cut);
    let inside = volume(common);
    assert!(
        (removed - inside).abs() <= 0.01,
        "the cut removes {removed}, the tool's part inside the plate is {inside}"
    );
}
