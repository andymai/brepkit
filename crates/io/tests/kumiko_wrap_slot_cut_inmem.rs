//! The gridfinity tool's kumiko corner wrap (`slideRailBuilder.test.ts`, "is
//! not carved away by a kumiko wrap either"), captured on brepkit-wasm 4.1.5
//! with the kernel's calls wrapped to save each boolean's operands: the
//! corner band, cut exactly by its helical struts (138 faces, 126 of them
//! NURBS strut-wall pieces), is compound-cut by 19 tilted slot boxes. These
//! are the seven boxes whose cuts meet the struts' grooves and the band's
//! rims; each must stay an exact B-Rep cut. A fallback here hands a 16k to
//! 19k-face planar mesh to every later compound cut in the export.
//!
//! Data: `kumiko_wrap_exact_band.bin` (cut base), `kumiko_wrap_slot_box_<i>.bin`
//! for the seven boxes (indices into the captured tool list).

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

/// Edges used other than twice across the solid's faces (zero for a closed
/// manifold operand).
fn bad_edge_uses(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
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
    assert_eq!(
        bad_edge_uses(&topo, band),
        0,
        "the band is not a closed manifold"
    );
    for i in BOXES {
        let b = load(&mut topo, &format!("kumiko_wrap_slot_box_{i}.bin"));
        assert_eq!(census(&topo, b).get("plane"), Some(&6), "box {i}");
        assert_eq!(
            bad_edge_uses(&topo, b),
            0,
            "box {i} is not a closed manifold"
        );
    }
}

/// Each box cut stays exact. A box face's section ran on past a strut-wall
/// patch's trimmed boundary (8, 9, 13, 14, 18, 19); the band's floor rim, a
/// circle stored as a NURBS edge running against its curve, dropped the
/// box's crossing (4); the plane x cylinder arc never ended at the band's
/// NURBS rim (4, 8, 13, 18); and a groove edge on the outer cylinder was
/// never split where the box's sections end on it (8, 13, 18).
#[test]
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
        } else if bad_edge_uses(&topo, result) != 0 {
            failures.push(format!("box {i}: open or over-shared edges"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
