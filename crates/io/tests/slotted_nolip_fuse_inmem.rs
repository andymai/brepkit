//! Fusing the four-socket assembly onto a 2x2 slotted no-lip bin body must
//! stay analytic. FIXED by the within-rank cross-shell gate in
//! detect_same_domain: the fuse is watertight, keeps the full analytic mix,
//! and its volume equals the operand sum exactly. Historically it aborted
//! with "open hole shell with 45 faces", dropped to the mesh fallback, and
//! the fallback's open output carried 107-109 boundary edges into the
//! export (the `2x2 slotted no lip` export-integrity failure).
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
//!
//! ROOT MEASURED (BK_TRACE + BK_SD + BK_OPEN_SHELL): every free edge of the
//! aborting shell lies at z=21 — the top rim of the body's cavity-wall band.
//! All 191 classified sub-faces were Outside (nothing dropped by
//! classification); the missing face was the body's cavity CEILING Id(31)
//! (32-edge plane at z=21), which detect_same_domain declared a WITHIN-RANK
//! DUPLICATE of the body's exterior top disc Id(9) (8-edge plane, same z=21
//! plane) and dropped before classification. The two faces are coplanar
//! because the no-lip bin has a zero-thickness roof (the cavity ceiling and
//! the exterior top coincide), and the rank-agnostic geometric-containment
//! pass unioned them; the within-rank emission then treated group
//! membership as the #696 residue signature. Extent (edge-set equality)
//! and outward orientation were both tried as discriminants and REFUTED by
//! measurement: the honeycomb's true residue caps and this load-bearing
//! pair produce identical signatures on both (same_outward=Some(true),
//! coextensive=false). The discriminant that holds is SHELL MEMBERSHIP:
//! Id(9) lives in the body's outer shell and Id(31) in its inner (void)
//! shell — residue accumulates within one shell, while a cross-shell
//! coincidence is structural. detect_same_domain now takes the operands'
//! face-to-shell map and skips within-rank dedup for cross-shell pairs.
//!
//! EXPORT-LEVEL NOTE: on the conflict re-cast kernel (the O-shape fix) the
//! export test already passed because the chain's upstream booleans
//! classified differently and stopped feeding this operand pair into the
//! final fuse; the captured operands still reproduced the abort until the
//! coextensivity gate closed the root itself.

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
fn slotted_nolip_socket_fuse_is_analytic_watertight() {
    let mut topo = Topology::new();
    let body = load("slotted_nolip_body.bin", &mut topo);
    let sockets = load("slotted_socket_assembly.bin", &mut topo);
    let vol_body = brepkit_operations::measure::oriented_solid_volume(&topo, body, 0.05).unwrap();
    let vol_sockets =
        brepkit_operations::measure::oriented_solid_volume(&topo, sockets, 0.05).unwrap();

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .expect("analytic fuse should not abort");

    let (free, over, curved) = health(&topo, result);
    assert!(curved > 0, "all-planar output is the mesh-fallback tell");
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");

    // The socket assembly attaches below the body without overlapping it, so
    // the fuse volume must equal the operand sum (measured 135237.533 =
    // 106091.810 + 29145.724).
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - (vol_body + vol_sockets)).abs() < 0.01,
        "fuse volume {vol:.3} must equal the operand sum {:.3}",
        vol_body + vol_sockets
    );
}
