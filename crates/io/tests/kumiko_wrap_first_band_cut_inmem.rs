//! The gridfinity tool's kumiko corner wrap (`slideRailBuilder.test.ts`, "is
//! not carved away by a kumiko wrap either"), captured on a brepkit-wasm built
//! from main: the first corner band the export compound-cuts by its 19 slot
//! boxes (194 faces, 160 of them NURBS). On that build the compound cut ran
//! 1,698 s and trapped.
//!
//! Data: `kumiko_wrap_first_band.bin` (cut base),
//! `kumiko_wrap_first_band_box_<1..19>.bin` (the captured tool list).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The boxes cut from the band one after another, and the volume after each;
/// each agrees within 0.001 with the previous volume less the exact
/// intersection of the band with the next box, measured at deflection 0.002.
const CHAIN: [usize; 2] = [1, 2];
const CHAIN_VOLUMES: [f64; CHAIN.len()] = [336.335, 334.100];

/// The band less all 19 boxes; at deflection 0.002 it agrees within 0.012
/// with the band less its intersection with the fused boxes.
const COMPOUND_VOLUME: f64 = 267.267;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

/// Edges used other than twice across the solid's faces.
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
fn kumiko_wrap_first_band_fixture_is_faithful() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band.bin");
    let mut census: HashMap<&str, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, band).unwrap() {
        *census
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    assert_eq!(
        ["cylinder", "nurbs", "plane"].map(|k| census.get(k).copied().unwrap_or(0)),
        [14, 160, 20],
        "fixture drifted: {census:?}"
    );
    assert_eq!(bad_edge_uses(&topo, band), 0);
}

/// Box 1 clips a corner off one strut-wall patch, and box 2 then clips the
/// corner where box 1's face, the bore and that patch meet. Each section ends
/// where a smooth boundary curve is split, which the face splitter's walker
/// read as a turn and closed the section on its own reverse; box 2's arc on
/// the bore lies on it only between box 1's elliptical rim and a groove.
#[test]
fn kumiko_wrap_first_band_chain_stays_exact() {
    let mut topo = Topology::new();
    let mut cur = load(&mut topo, "kumiko_wrap_first_band.bin");
    for (i, expected) in CHAIN.into_iter().zip(CHAIN_VOLUMES) {
        let b = load(&mut topo, &format!("kumiko_wrap_first_band_box_{i}.bin"));
        let before = boolean::mesh_fallback_count();
        cur = boolean::boolean(&mut topo, BooleanOp::Cut, cur, b).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "box {i}: mesh fallback"
        );
        assert_eq!(
            bad_edge_uses(&topo, cur),
            0,
            "box {i}: open or over-shared edges"
        );
        let vol = brepkit_operations::measure::oriented_solid_volume(&topo, cur, 0.05).unwrap();
        assert!(
            (vol - expected).abs() <= 0.01,
            "box {i}: volume {vol:.3}, expected {expected:.3}"
        );
    }
}

/// The tool's own call: one `compound_cut` of the band by all 19 boxes stays
/// an exact B-Rep. Box 17 meets the band's radial end plane, whose boundary
/// carries the bore and the strut grooves; the plane x plane section with it
/// must end at that boundary, not run on into the bore.
#[test]
fn kumiko_wrap_first_band_compound_cut_stays_exact() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band.bin");
    let tools: Vec<_> = (1..=19)
        .map(|i| load(&mut topo, &format!("kumiko_wrap_first_band_box_{i}.bin")))
        .collect();
    let before = boolean::mesh_fallback_count();
    let cut = boolean::compound_cut(&mut topo, band, &tools, BooleanOptions::default()).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    assert_eq!(bad_edge_uses(&topo, cut), 0, "open or over-shared edges");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, cut, 0.05).unwrap();
    assert!(
        (vol - COMPOUND_VOLUME).abs() <= 0.01,
        "volume {vol:.3}, expected {COMPOUND_VOLUME:.3}"
    );
}
