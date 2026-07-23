//! Captured-operand regression for the gridfinity label-socket tab attach —
//! the fuse of the finished label tab into the 3×1 bin body.
//!
//! Operands captured on 2.127.29 from the warm-cache run of the "3×1 socket
//! auto-quantizes to 3U" scenario: GFA produced a 90-face candidate missing
//! whole faces — 8 free boundary edges — yet every remaining wire was closed
//! and the Euler gate balanced by accident, so `validate_boolean_result`
//! (which only hard-failed unclosed wires and non-manifold edges) accepted
//! the open shell instead of falling back. The same operands built cold
//! differ by float jitter and abort inside assembly ("open hole shell would
//! be dropped"), which is why the scenario passed solo but failed after a
//! 1×1 scenario warmed the cell-socket template cache.
//!
//! With the free-edge hard-fail in the strict gate, both variants reached the
//! watertight mesh fallback. The corner-crescent fixes (boundary-extension
//! salvage in `fill_images_faces` + the woven-spur arrangement trigger in the
//! face splitter) then made the fuse assemble analytically: the tab's square
//! top corners overhang the cavity's rounded corners, and the bin's top ring
//! must re-square there. The volume pin is calibrated by inclusion-exclusion
//! (bin + tab − intersect = 20462.5); the coarse-deflection measurement of
//! the analytic result reads a few mm³ low from chording its cylinder faces.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

#[test]
fn labeltab_attach_fuse_is_watertight() {
    let mut topo = Topology::new();
    let bin = load("labeltab_attach_bin.bin", &mut topo);
    let tab = load("labeltab_attach_tab.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, bin, tab).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
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
    assert_eq!(free, 0, "tab attach must be closed, got {free} free edges");
    assert_eq!(
        over, 0,
        "tab attach must be manifold, got {over} over-shared"
    );

    let curved = faces
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved >= 12,
        "expected an analytic result with the pocket/cavity cylinders, got {curved} curved faces"
    );

    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.005).unwrap();
    // Truth by inclusion-exclusion: 17421.32 + 3046.86 − 5.71 = 20462.5.
    // The band covers coarse-tessellation error on the curved faces; the bad
    // open-shell result measured ~20483 with faces missing and stays outside.
    assert!(
        (vol - 20462.5).abs() < 15.0,
        "fused volume {vol:.2} out of range"
    );
}
