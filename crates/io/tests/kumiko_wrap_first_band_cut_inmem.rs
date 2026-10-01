//! The gridfinity tool's kumiko corner wrap (`slideRailBuilder.test.ts`, "is
//! not carved away by a kumiko wrap either"), captured on a brepkit-wasm built
//! from main: the first corner band the export compound-cuts by its 19 slot
//! boxes (194 faces, 160 of them NURBS), with the first two boxes. Neither the
//! batched cut nor the first box's own cut stays exact, so the compound cut
//! must take one mesh fallback for the batch. Cutting box by box after the
//! first box degrades sends every later box against a mesh of tens of
//! thousands of faces: two fallbacks and 50 s natively for these two boxes,
//! and a tool test past its 40-minute cap for all 19.
//!
//! Data: `kumiko_wrap_first_band.bin` (cut base),
//! `kumiko_wrap_first_band_box_<1..2>.bin` (the first two captured boxes).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOptions};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

#[test]
fn kumiko_wrap_first_band_compound_cut_takes_one_fallback() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band.bin");
    let mut census: HashMap<&str, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, band).unwrap() {
        *census
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    assert_eq!(
        (census["cylinder"], census["nurbs"], census["plane"]),
        (14, 160, 20),
        "fixture drifted: {census:?}"
    );
    let boxes = [
        load(&mut topo, "kumiko_wrap_first_band_box_1.bin"),
        load(&mut topo, "kumiko_wrap_first_band_box_2.bin"),
    ];
    let before = boolean::mesh_fallback_count();
    let cut = boolean::compound_cut(&mut topo, band, &boxes, BooleanOptions::default()).unwrap();
    let fallbacks = boolean::mesh_fallback_count() - before;
    assert!(
        fallbacks <= 1,
        "{fallbacks} mesh fallbacks: the boxes were cut one by one through a mesh"
    );
    assert!(
        !brepkit_topology::explorer::solid_faces(&topo, cut)
            .unwrap()
            .is_empty()
    );
}
