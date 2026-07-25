//! Regression: cutting a gridfinity bin wall by one kumiko lattice band used
//! to leave 30 free edges, so the analytic result was rejected and the whole
//! kumiko family fell back to the mesh boolean.
//!
//! Root cause (diagnosed in #1223, fixed in #1224): phase FF's AABB in-both
//! pre-filter sampled each raw curve 16× and, for a `Line`, gave up without
//! the adaptive refinement it applies to every other curve type — on the
//! stated reasoning that a straight line cannot be under-sampled at that
//! granularity. But exactness of the LINE says nothing about
//! whether a sample lands in the tiny in-both WINDOW: the section here is a
//! full-height (~20.3mm) cylinder generator whose true in-both span is one
//! ~0.83mm lattice opening, under the filter's ~1.27mm pitch. Whether a band
//! survived was aliasing luck — 12 of the 16 band×cylinder pairs kept their
//! generator and 4 silently lost both. A straight section needs no sampling at
//! all: the predicate is membership in `bb_a ∩ bb_b`, itself an AABB, so the
//! segment is now slab-clipped against it exactly.
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
//! What the analytic result got wrong was small and precise: **30 free edges**,
//! chaining into **4 components whose every vertex has degree exactly 2**. Each
//! was an un-notched span of a corner cylinder's tangent generator at x=17.00,
//! left unpaired because the flat wall at y=−20.750 does have its opening
//! there. The notch — the quarter-cylinder trimmed from θ=90 back to θ=89.24
//! across the tool's 0.05mm `SLAB_OVERLAP` — formed in 5 of 8 z-bands on the
//! outer (r=3.75) cylinder and 7 of 8 on the inner (r=2.55) one.
//!
//! Diagnostic notes worth keeping: the failing bands are NOT distinguished by
//! the bridging ellipse's slope, band width, or band spacing (all three were
//! checked and refuted — the working bay at z 9.291–10.122 slopes the same way
//! as all three failures). And "only 2 of 24 cylinders carry sections" is
//! CORRECT, not short: localizing by the free loops' y/z rather than x puts
//! every outline in the single (+x,−y) corner.
//!
//! Bands 1/3/5/7 still abort with "open growth shell with N faces" — a
//! SEPARATE, pre-existing defect that this fix does not address.
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
