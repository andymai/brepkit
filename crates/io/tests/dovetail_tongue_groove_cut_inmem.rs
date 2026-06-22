//! Faithful regression guard for the baseplate doubled-dovetail groove cut.
//!
//! The gridfinity tool's `baseplateGenerator.scenario.dovetail.test.ts`
//! "preferIdenticalPieces produces a watertight STL" (a 4x4 interior tile where
//! EVERY cell boundary on a join edge carries both a tongue and a groove)
//! reported 23 non-manifold STL edges with brepkit and fell back to a slow mesh
//! repair (251 non-manifold edges during winding repair). Capturing the tool's
//! literal kernel operands (via `serializeSolid`, replayed through
//! `arena_io::deserialize_solid`) localized the defect to the FIRST groove `Cut`
//! on the tongue-fused doubled slab — NOT the tongue fuses (which are all clean).
//!
//! The two `.bin` fixtures are the tool's EXACT operands for that cut:
//!   `_slab`   = the 4x4 baseplate slab after its dovetail tongues are fused
//!               (266 faces, watertight). Its z=0 top cap is a single multiply-
//!               connected plane face: 64 outer edges + 16 cell-opening inner
//!               wires (each an 8-edge rounded square).
//!   `_groove` = the female dovetail groove cutter (a 6-face trapezoidal box at
//!               one corner) that pokes through the slab from above.
//!
//! ROOT: the groove cut splits that 16-hole top cap. The holed-plane arrangement
//! split path (`face_splitter::arrangement_regions_from_combined`) wove ALL 16
//! holes into the arrangement even though the corner groove interacts with only
//! one cell. Two bugs compounded:
//!   1. The even-odd "drop a region that fills an original hole (air)" test
//!      probed the OUTER polygon centroid, which for the large holed cap lands
//!      inside one of its OWN cell openings -> the entire material cap was
//!      dropped (free edges along every hole boundary).
//!   2. Chord-fragmenting the 15 untouched arc-bounded cell openings produced
//!      spurious sliver regions that double-emitted hole edges (over-shares).
//!
//! The fix (`face_splitter`): weave only the holes a section actually interacts
//! with; attach the untouched holes whole (preserving exact arc geometry); and
//! run the air-region drop test at a MATERIAL seed (outside the region's own
//! inner-wire holes) rather than the naive outer centroid.
//!
//! This guard asserts the cut is watertight (no free edges), manifold (no
//! over-shared edges), stays analytic (all-planar, no mesh fallback), and is
//! compact.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// Free (used-once) and over-shared (incident to >2 faces) counts keyed by the
/// shared topological `EdgeId` — the authoritative manifold measure (a quantized
/// endpoint key can hide a near-duplicate-vertex non-manifold edge).
fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    let mut by_edge: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *by_edge.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        by_edge.values().filter(|&&c| c == 1).count(),
        by_edge.values().filter(|&&c| c > 2).count(),
    )
}

#[test]
fn dovetail_tongue_groove_cut_is_watertight() {
    let mut topo = Topology::new();
    let slab = load("dovetail_tongue_slab.bin", &mut topo);
    let groove = load("dovetail_tongue_groove.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Cut, slab, groove).unwrap();

    let (free, over) = edge_health(&topo, result);
    let faces = solid_faces(&topo, result).unwrap();
    let curved = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();

    assert_eq!(
        over,
        0,
        "doubled-dovetail groove cut must be manifold (no over-shared edges); \
         got {} faces, {curved} curved, {free} free",
        faces.len()
    );
    assert_eq!(
        free,
        0,
        "doubled-dovetail groove cut must be watertight (no free edges); \
         got {} faces, {curved} curved",
        faces.len()
    );
    // A clean analytic groove cut of a 266-face slab adds only a handful of
    // faces. A mesh fallback explodes to hundreds of planar facets.
    assert!(
        faces.len() < 320,
        "expected a compact analytic result, got {} faces (mesh fallback?)",
        faces.len()
    );
    assert_eq!(
        curved, 0,
        "the slab + groove are all-planar; got {curved} curved faces"
    );
}
