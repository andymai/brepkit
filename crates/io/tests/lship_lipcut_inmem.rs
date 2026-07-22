//! Captured-operand regression for the gridfinity custom-shape (3×3 L) lip
//! frustum cut — the shared root of the 22-scenario custom-shape export
//! family.
//!
//! Operands captured from the live tool via `serializeSolid` on 2.127.26
//! (both frustums analytic after the loft arc-segmentation fix). The cut's
//! section loop on the coincident bottom cap used to nest inside its own
//! reversed twin (a zero-area bubble) instead of becoming the ring cap's
//! hole: the first-vertex hole probe sat exactly ON the twin outline and
//! float jitter landed it "inside". The ring cap then never formed, both
//! rims stayed unpaired, and the cut fell to an OPEN mesh fallback that
//! poisoned the whole export chain.

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
fn l_lip_frustum_cut_is_analytic_and_watertight() {
    let mut topo = Topology::new();
    let outer = load("lship_lip_outerfrustum.bin", &mut topo);
    let inner = load("lship_lip_innerfrustum.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
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
    assert_eq!(free, 0, "lip cut must be closed, got {free} free edges");
    assert_eq!(over, 0, "lip cut must be manifold, got {over} over-shared");
    assert!(
        curved >= 30 && faces.len() < 120,
        "analytic result expected, got {curved} curved of {} faces",
        faces.len()
    );
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - 5518.5).abs() < 6.0,
        "lip ring volume out of band: got {vol}"
    );
}
