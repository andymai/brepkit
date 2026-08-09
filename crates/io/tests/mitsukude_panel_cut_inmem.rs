//! Cutting ONE mitsukude lattice panel out of a 2x2x6 bin body must stay
//! analytic and watertight. Today the raw GFA cut succeeds in ~200 ms with
//! every cone and cylinder preserved but leaves 8 free edges, so the ops
//! validity gate declares it unusable and takes the mesh fallback — which is
//! where the tool-side "kumiko dividers" scenario spends 82 of its 86 seconds
//! (and picks up non-manifold edges from the fallback's own open output).
//! The perf gap IS this correctness gap.
//!
//! The 8 free edges form one connected loop at the stacking-lip transition
//! band on the east wall (x 38.05 / 38.00 planes joined by an r=0.05 arc band
//! and a 45-degree chamfer, y in [-41.75, -40.05], z in [31.53, 34.8]): a
//! strut pocket crosses the three-surface junction and the sub-face patch
//! spanning it is lost. The 8-panel compound cut fails differently (a
//! 4-face, 0.57 mm^3 open hole-shell fragment aborts assembly), but this
//! single-panel loop is the narrow entry point.
//!
//! Operands captured 2026-08-04 from the tool's boolean traffic (call 036 of
//! the "kumiko dividers perforate the compartment walls" main arm, split to
//! its smallest failing subset: the body and panel 1).

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

fn health(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize, usize) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    let mut curved = 0;
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            curved += 1;
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    (free, over, curved)
}

#[test]
fn mitsukude_operands_are_clean() {
    let mut topo = Topology::new();
    for name in ["mitsukude_bin_body.bin", "mitsukude_panel_tool.bin"] {
        let sid = load(name, &mut topo);
        let (free, over, _) = health(&topo, sid);
        assert_eq!((free, over), (0, 0), "{name} must be closed and manifold");
    }
}

#[test]
fn mitsukude_panel_cut_is_analytic_watertight() {
    // Closed by the sample_plane_cone asymptote-tail extension: the free
    // loop was the taper cone's missing FF section against the prism's
    // grazing x=38.05 end plane, whose entire in-face window fell between
    // two adjacent uniform-u samples of the hyperbola near its asymptote.
    let mut topo = Topology::new();
    let body = load("mitsukude_bin_body.bin", &mut topo);
    let panel = load("mitsukude_panel_tool.bin", &mut topo);

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, body, panel)
            .expect("cut should not abort");

    let (free, over, curved) = health(&topo, result);
    assert!(
        curved > 0,
        "all-planar output is the mesh-fallback tell; the cut must stay analytic"
    );
    assert_eq!(over, 0, "cut must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "cut must be closed, got {free} free edges");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.05).unwrap();
    // Mesh-derived pin: re-calibrated when the developable-face interior
    // grid was removed from the display tessellation (density change only;
    // the pre-fix leak stays 68 units away).
    assert!(
        (vol - 27027.9).abs() < 5.0,
        "cut volume drifted: got {vol:.1}, pinned 27027.9 (the pre-fix leak measured 27095.9 \
         with the pinch band left uncut)"
    );
}
