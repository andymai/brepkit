//! The gridfinity tool's kumiko corner wrap (`slideRailBuilder.test.ts`, "is
//! not carved away by a kumiko wrap either"), captured on brepkit-wasm 4.1.5
//! with the kernel's calls wrapped to save each boolean's operands: the
//! corner band, now cut exactly by its helical struts (138 faces, 126 of them
//! NURBS strut-wall pieces), is compound-cut by 19 tilted slot boxes. Cut
//! alone, 12 boxes stay exact and 7 fall back to planar meshes of 16,583 to
//! 19,532 faces, the raw cut leaving 5 to 36 free edges where a box meets the
//! strut walls; natively the compound cut takes 113 s and returns a
//! 7,227-face blob. Later compound cuts in the export consume such blobs,
//! and in the tool the wasm kernel panics with a hash table capacity
//! overflow after 1,826 s, where 3.3.9 (whose band cut already fell back to
//! a 150-face mesh) timed out at 508 s.
//!
//! Data: `kumiko_wrap_exact_band.bin` (cut base), `kumiko_wrap_slot_box_<i>.bin`
//! for the seven boxes whose cut falls back (indices into the captured tool
//! list).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const BOXES: [usize; 7] = [4, 8, 9, 13, 14, 18, 19];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn census(topo: &Topology, solid: SolidId) -> HashMap<&'static str, usize> {
    let mut counts = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        *counts
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    counts
}

#[test]
fn kumiko_wrap_slot_fixture_is_faithful() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_exact_band.bin");
    let c = census(&topo, band);
    assert_eq!(
        (c.get("nurbs"), c.get("cylinder"), c.get("plane")),
        (Some(&126), Some(&5), Some(&7))
    );
    for i in BOXES {
        let b = load(&mut topo, &format!("kumiko_wrap_slot_box_{i}.bin"));
        assert_eq!(census(&topo, b).get("plane"), Some(&6), "box {i}");
    }
}

#[test]
#[ignore = "ready repro: each slot box cut from the exact kumiko band must stay exact (7 fall back today)"]
fn kumiko_wrap_slot_cuts_stay_exact() {
    let mut failures = Vec::new();
    for i in BOXES {
        let mut topo = Topology::new();
        let band = load(&mut topo, "kumiko_wrap_exact_band.bin");
        let b = load(&mut topo, &format!("kumiko_wrap_slot_box_{i}.bin"));
        let before = boolean::mesh_fallback_count();
        let result = boolean::boolean(&mut topo, BooleanOp::Cut, band, b).unwrap();
        if boolean::mesh_fallback_count() != before {
            failures.push(format!(
                "box {i}: mesh fallback ({} faces)",
                census(&topo, result).values().sum::<usize>()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
