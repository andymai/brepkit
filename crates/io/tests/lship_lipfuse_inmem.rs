//! Captured-operand regression for the gridfinity custom-shape (3×3 L)
//! body + lip fuse — the last link of the L-family export chain.
//!
//! Operands captured on 2.127.27 (loft and frustum-cut fixes in): the fuse
//! used to reach a 124-face analytic candidate with exactly ONE non-manifold
//! edge — a coincident-ring re-trace at the L's CONCAVE corner woven through
//! the wall wire as an out-and-back spur, plus a two-edge slit face over the
//! same arc — then fall to an OPEN mesh fallback (bd=32 across every L
//! variant). With spur excision the fuse is analytic and watertight.

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
fn l_lip_fuse_is_analytic_and_watertight() {
    let mut topo = Topology::new();
    let body = load("lship_lipfuse_body.bin", &mut topo);
    let top = load("lship_lipfuse_top.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, body, top).unwrap();

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
    assert_eq!(free, 0, "lip fuse must be closed, got {free} free edges");
    assert_eq!(over, 0, "lip fuse must be manifold, got {over} over-shared");
    assert!(
        curved >= 40 && faces.len() < 250,
        "analytic result expected, got {curved} curved of {} faces",
        faces.len()
    );
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - 29716.6).abs() < 20.0,
        "fused volume out of band: got {vol}"
    );
}
