//! The 2x2 mixed-detail per-cell half-sockets bin: every boolean in its
//! export chain replays clean and analytic, and the final fuse here succeeds
//! watertight (free=0, over=0) — yet TESSELLATING that valid B-Rep at export
//! tolerance (0.01 mm / 5 degrees) yields hundreds of mesh boundary edges
//! (511 measured natively; the tool-side export reports 259 after its own
//! welding). A tessellation-parity defect on a clean B-Rep, not a boolean
//! one: the "not every scenario failure is a boolean fallback" class.
//!
//! The per-cell dispatch geometry (three full sockets + three quarter
//! sockets, one 1u block mixed) is what distinguishes this from the sibling
//! socket bins that tessellate clean.
//!
//! Operands captured 2026-08-05 via the kernel-test boolean monkey-patch
//! (call 008, the final fuse of the export chain).

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

fn brep_health(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
    )
}

#[test]
fn mixed_socket_fuse_is_brep_watertight() {
    // The B-Rep side is HEALTHY: this pin holds the boolean blameless so the
    // ignored tessellation repro below cannot be misread as a GFA defect.
    let mut topo = Topology::new();
    let body = load("mixed_socket_body.bin", &mut topo);
    let assembly = load("mixed_socket_assembly.bin", &mut topo);
    let result = brepkit_algo::gfa::boolean(
        &mut topo,
        brepkit_algo::bop::BooleanOp::Fuse,
        body,
        assembly,
    )
    .expect("analytic fuse must succeed");
    assert_eq!(
        brep_health(&topo, result),
        (0, 0),
        "B-Rep must be watertight"
    );
}

#[test]
#[ignore = "ready repro: tessellating the CLEAN fused B-Rep at export tolerance yields ~511 \
            mesh boundary edges — a tessellation-parity defect on the per-cell mixed-socket \
            geometry, not a boolean one"]
fn mixed_socket_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let body = load("mixed_socket_body.bin", &mut topo);
    let assembly = load("mixed_socket_assembly.bin", &mut topo);
    let result = brepkit_algo::gfa::boolean(
        &mut topo,
        brepkit_algo::bop::BooleanOp::Fuse,
        body,
        assembly,
    )
    .expect("analytic fuse must succeed");

    let mesh = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
        &topo,
        result,
        0.01,
        5.0_f64.to_radians(),
    )
    .unwrap();
    let bnd = brepkit_operations::tessellate::boundary_edge_count(&mesh);
    assert_eq!(
        bnd, 0,
        "export-tolerance mesh must be watertight, got {bnd}"
    );
}
