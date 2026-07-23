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
//! With the free-edge hard-fail in the strict gate, both variants reach the
//! watertight fallback. The assertion is watertightness, not analyticity —
//! the GFA assembly root (the dropped hole shell) is tracked separately.

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

    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    // bin 17421.29 + tab 3046.86 minus their overlap; the open-shell result
    // measured 20483.12 with faces missing. Pin near the correct fused value.
    assert!(
        (vol - 20466.5).abs() < 5.0,
        "fused volume {vol:.2} out of range"
    );
}
