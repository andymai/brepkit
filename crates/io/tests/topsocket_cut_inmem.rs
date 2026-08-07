//! Stage fixtures from the mixed-detail per-cell half-sockets chain, pinned
//! with the DIRECTED half-edge oracle (tessellate at export tolerance, count
//! half-edges with no opposite twin). That oracle is authoritative for the
//! orientation-mismatch class; the offset-classification "outwardness" audit
//! is NOT — it returns unanimous false positives near concave cylinder
//! corners (call 000's cut result audits "3 inverted faces, 10-0 votes" yet
//! meshes directed-watertight), the classification-probe trap in a new form.
//! Always cross-check any outwardness claim against the directed mesh.
//!
//! Attribution by the directed oracle (2026-08-07 stage capture):
//! - call 000 (cut, this file's `topsocket_cut_*` operands): args 0/0,
//!   result 0 — HEALTHY. The #1401 claim that this cut mints inverted
//!   faces was the false-positive audit; retracted here.
//! - call 001 (cut, `topsocket_chain001_*`): its args ALREADY carry 38 and
//!   78 unmatched half-edges (sum 116); the result carries all 116. The
//!   boolean faithfully preserves the defect, it does not mint it.
//! - calls 002-008: the 116 ride into the body operand of
//!   `mixed_socket_tess_inmem.rs`; the socket-assembly side stays 0.
//!
//! So the mint happens BEFORE call 001, in ops the fuse/cut monkey-patch
//! did not see — the export drives some ops through executeBatch (the known
//! capture gap). The next capture must hook the batch dispatcher too.
//!
//! What is proven about the defective faces: each meshes true to its own
//! effective normal (mesh_orient=1.0 in `fuse_orient`) while DIRECTED
//! pairing fails against neighbours at the quarter-socket rims — adjacent
//! faces' effective orientations genuinely disagree. Which side is wrong
//! needs a sound oracle before any fix.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn directed_unmatched(topo: &Topology, solid: SolidId) -> usize {
    let mesh = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
        topo,
        solid,
        0.01,
        5.0_f64.to_radians(),
    )
    .unwrap();
    let mut half: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            *half.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    half.keys()
        .filter(|&&(x, y)| !half.contains_key(&(y, x)))
        .count()
}

#[test]
fn topsocket_cut_is_directed_watertight() {
    // The chain's first cut is healthy end to end: clean operands, clean
    // result under the directed oracle. Guards the boolean against ever
    // minting orientation mismatches on this configuration.
    let mut topo = Topology::new();
    let base = load("topsocket_cut_base.bin", &mut topo);
    let tool = load("topsocket_cut_tool.bin", &mut topo);
    assert_eq!(directed_unmatched(&topo, base), 0, "base operand");
    assert_eq!(directed_unmatched(&topo, tool), 0, "tool operand");
    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
            .expect("cut must succeed");
    assert_eq!(directed_unmatched(&topo, result), 0, "cut result");
}

#[test]
fn topsocket_chain001_args_carry_the_directed_mismatches() {
    // ACTIVE pin of the true carriers: call 001's args arrive with 38 and
    // 78 unmatched half-edges minted by earlier, uncaptured (batch-driven)
    // ops. A construction-side fix upstream changes these captures' role:
    // re-capture and update or retire this pin when that lands.
    let mut topo = Topology::new();
    let a = load("topsocket_chain001_a.bin", &mut topo);
    assert_eq!(directed_unmatched(&topo, a), 38, "chain001 arg0");
    let b = load("topsocket_chain001_b.bin", &mut topo);
    assert_eq!(directed_unmatched(&topo, b), 78, "chain001 arg1");
}

#[test]
fn topsocket_chain001_cut_preserves_not_mints() {
    // The cut of the two defective args carries exactly the inherited sum:
    // the boolean preserves the mismatches, it does not mint or heal them.
    let mut topo = Topology::new();
    let a = load("topsocket_chain001_a.bin", &mut topo);
    let b = load("topsocket_chain001_b.bin", &mut topo);
    let result = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, a, b)
        .expect("cut must succeed");
    assert_eq!(directed_unmatched(&topo, result), 116, "cut result");
}
