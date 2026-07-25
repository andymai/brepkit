//! Ready-repro: cutting a gridfinity bin wall by one kumiko lattice band
//! leaves 30 free edges, so the analytic result is rejected and the whole
//! kumiko family falls back to the mesh boolean.
//!
//! This one boolean is the root of the kumiko export-integrity family. The
//! chain, measured (see the roadmap's goma entry): the tool's
//! `goma carves a 1x1x6 bin` scenario runs ~850s and trips vitest's per-test
//! timeout, whose abandoned async chain poisons the wasm kernel for every
//! later kumiko scenario — 14 failures from one root. Of that 850s, ~203s is a
//! single `cutAll` of 8 lattice bands, and it costs 203s only because the
//! analytic path is rejected and the mesh fallback runs instead. The analytic
//! path is ~12x faster and keeps all 12 cones and 24 cylinders.
//!
//! What the analytic result gets wrong is small and precise: **30 free edges**,
//! which chain into **4 components whose every vertex has degree exactly 2** —
//! four simple closed outlines, i.e. four missing faces. They all sit in one
//! 0.05mm slab: every vertex is at x=17.00 or x=17.05. The x=17.05 side is this
//! tool's cut plane; the x=17.00 side is NOT a plane but the base's corner
//! cylinder, so it is a plane-vs-cylinder sliver. Each outline mixes lines in
//! the cut plane with one line at x=17.00 and two ellipse arcs bridging the
//! gap, so the missing faces are non-planar patches, not flat slivers.
//!
//! The defect is op-independent (Cut/Fuse both leave ~30 free edges, Intersect
//! aborts assembly), which points at the splitter/section stage rather than
//! classification. It also hits every band: tools 0/2/4/6 all give this exact
//! signature, tools 1/3/5/7 abort with "open growth shell with N faces".
//!
//! Operands captured from the live tool on published 2.128.2; the other seven
//! bands live in `~/.cache/brepkit-parity-captures/2026-07-24/goma-bisect/`
//! and replay via `crates/io/examples/replay_cut_capture.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

#[test]
#[ignore = "ready-repro — kumiko band cut leaves 30 free edges; see doc comment"]
fn goma_wall_band_cut_is_closed() {
    let mut topo = Topology::new();
    let base = load("goma_wall_base.bin", &mut topo);
    let band = load("goma_wall_band.bin", &mut topo);

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, band)
            .expect("analytic cut should not fail outright");

    let faces = solid_faces(&topo, result).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();

    // The analytic surfaces ARE preserved — that is what makes this worth
    // recovering rather than conceding to the mesh fallback.
    let curved = faces
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved >= 30,
        "analytic surfaces should survive the cut, got {curved} curved faces"
    );

    assert_eq!(over, 0, "cut must stay manifold, got {over} over-shared");
    assert_eq!(
        free, 0,
        "cut must be closed; {free} free edges send the whole kumiko family to the mesh fallback"
    );
}
