//! Fusing the four-socket assembly onto a 2x2 slotted no-lip bin body must
//! stay analytic. Today it aborts with "open hole shell with 45 faces",
//! drops to the mesh fallback, and the fallback's open output carries
//! 107-109 boundary edges into the export (the `2x2 slotted no lip`
//! export-integrity failure).
//!
//! Both operands are clean: the body (F=56, 8 cylinders, watertight) and the
//! socket assembly (F=136, 32 cones + 32 cylinders, watertight). Every other
//! boolean in the export chain replays clean and analytic; this fuse is the
//! sole leak producer. The failure reproduces identically on kernels from
//! before and after the 2026-08-04/05 engine work, so the trigger is the
//! tool's generator changes (the #3223-#3227 era) reshaping this
//! configuration, not an engine regression.
//!
//! Operands captured 2026-08-05 via the kernel-test boolean monkey-patch on
//! the failing export scenario (call 009 of 10).
//!
//! BK_OPEN_SHELL characterization: the aborting 45-face shell has signed
//! volume -51259 and is built from the BODY's own faces (src 10-13, the
//! outer walls and corner cylinders) — the fuse classifies a body-sized
//! chunk as a hole shell, the "no outer shell / misgrouped interior" family
//! rather than a small-fragment drop.

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
fn slotted_operands_are_clean() {
    let mut topo = Topology::new();
    for name in ["slotted_nolip_body.bin", "slotted_socket_assembly.bin"] {
        let sid = load(name, &mut topo);
        let (free, over, curved) = health(&topo, sid);
        assert_eq!((free, over), (0, 0), "{name} must be closed and manifold");
        assert!(curved > 0, "{name} must keep analytic curved faces");
    }
}

#[test]
#[ignore = "ready repro: the socket-assembly fuse aborts with 'open hole shell with 45 faces' \
            and drops to the mesh fallback, whose open output carries 107-109 boundary edges \
            into the slotted no-lip export"]
fn slotted_nolip_socket_fuse_is_analytic_watertight() {
    let mut topo = Topology::new();
    let body = load("slotted_nolip_body.bin", &mut topo);
    let sockets = load("slotted_socket_assembly.bin", &mut topo);

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .expect("analytic fuse should not abort");

    let (free, over, curved) = health(&topo, result);
    assert!(curved > 0, "all-planar output is the mesh-fallback tell");
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");
}
