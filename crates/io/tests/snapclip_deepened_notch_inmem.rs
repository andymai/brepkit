//! Faithful guard for the snapClip deepened-notch cut (the stranded-rim case).
//!
//! The cutter is a plain box whose side walls are exactly coincident with an
//! EARLIER cut's pocket walls and whose top floats 0.01 above the old pocket
//! floor. On the two cone (countersink) faces the new wall sections lie on the
//! SAME intersection curves as the old bite's boundary edges, overlapping them
//! on a 0.015 sliver, and the box-top rim section runs almost entirely through
//! the old bite (off-face) with only micro corner-crescent portions on the
//! face. Three conditioning holes broke the split:
//!
//! - sections overhanging the face through an OUTER-wire concavity were never
//!   clipped (the inner-wire air filter only tests holes), so the weave got
//!   off-face rim/tail pieces and emitted the WHOLE face plus disconnected
//!   bite fragments;
//! - registry-presplit section pieces keep their PARENT's pcurve, so their
//!   endpoint UVs evaluate at the parent's endpoints (the B/A ends carried the
//!   rim's B'/A' UVs), disconnecting them from the boundary in UV;
//! - T-junction self-splits minted zero-extent section edges that derailed
//!   the angular walker into degenerate single-edge sub-faces.
//!
//! Fixtures are the tool's EXACT serialized operands (fresh 2026-07-16
//! capture: plate after op-cut-0, first notch-deepening cutter).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_topology::Topology;
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

type Q = (i64, i64, i64);

#[test]
fn deepened_notch_cut_pairs_every_brep_edge() {
    let mut topo = Topology::new();
    let plate = load("snapclip_notch_plate.bin", &mut topo);
    let cutter = load("snapclip_notch_cutter.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, BooleanOp::Cut, plate, cutter).unwrap();

    // Position-quantized B-Rep edge pairing: every edge must be used exactly
    // twice. The stranded-rim failure left the cone faces unsplit (their old
    // rim arcs unpaired) and the new wall sections partnerless.
    let sc = 1.0e5;
    let q = |v: f64| (v * sc).round() as i64;
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for &fid in &solid_faces(&topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = topo.vertex(e.start()).unwrap().point();
                let b = topo.vertex(e.end()).unwrap().point();
                let ka = (q(a.x()), q(a.y()), q(a.z()));
                let kb = (q(b.x()), q(b.y()), q(b.z()));
                let key = if ka <= kb { (ka, kb) } else { (kb, ka) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    let unpaired = occ.values().filter(|&&c| c != 2).count();
    assert_eq!(unpaired, 0, "unpaired position-quantized B-Rep edges");
}
