//! Fusing two half-socket pieces of the 3x3 O-shape bin must stay analytic.
//! Today it aborts with "open growth shell with 45 faces", falls to the mesh
//! fallback, and the fallback output poisons the next fuse (whose other
//! operand replays as 1022 all-planar faces), ending in the export's 8
//! non-manifold edges (the `3x3 O-shape + half sockets` export-integrity
//! failure).
//!
//! Both operands are clean 49-face socket pieces: one all-analytic
//! (12 cones + 12 cylinders), the other carrying 12 NURBS faces (the
//! quarter-socket pieces the per-cell dispatch produces). The failure
//! reproduces identically on kernels before and after the 2026-08-04/05
//! engine work; the trigger is the tool's generator changes (the #3223-#3227
//! era) reshaping this configuration.
//!
//! Sibling finding, same capture session: the `2x2 mixed-detail per-cell
//! half sockets` export failure (bnd=259) replays ENTIRELY CLEAN through all
//! nine of its booleans — its leak is post-boolean (tessellation/export or
//! an op class the boolean capture does not hook), the "not every scenario
//! failure is a boolean fallback" class.
//!
//! Operands captured 2026-08-05 via the kernel-test boolean monkey-patch
//! (call 006 of the export chain; call 007 is its downstream collateral).

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
fn oshape_operands_are_clean() {
    let mut topo = Topology::new();
    for name in ["oshape_socket_a.bin", "oshape_socket_b.bin"] {
        let sid = load(name, &mut topo);
        let (free, over, curved) = health(&topo, sid);
        assert_eq!((free, over), (0, 0), "{name} must be closed and manifold");
        assert!(curved > 0, "{name} must keep analytic curved faces");
    }
}

#[test]
#[ignore = "ready repro: the half-socket piece fuse aborts with 'open growth shell with 45 \
            faces', drops to the mesh fallback, and its open output poisons the next fuse, \
            ending in the O-shape export's 8 non-manifold edges"]
fn oshape_socket_fuse_is_analytic_watertight() {
    let mut topo = Topology::new();
    let a = load("oshape_socket_a.bin", &mut topo);
    let b = load("oshape_socket_b.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, a, b)
        .expect("analytic fuse should not abort");

    let (free, over, curved) = health(&topo, result);
    assert!(curved > 0, "all-planar output is the mesh-fallback tell");
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");
}
