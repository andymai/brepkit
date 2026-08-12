//! READY-REPRO for the coplanar-interface holed-cap fuse family (#1538 tail):
//! the "2x2 with circle insert" bin's pocket has cutDepth equal to the floor
//! depth, so the pocket bottom lands exactly on the body/socket interface
//! plane and the body's interface face carries a circular hole. Fusing the
//! socket assembly across that plane emits ~60 free edges at the interface
//! (raw GFA), and the ops layer pays a mesh fallback whose output feeds the
//! export. Captured tool-side on brepkit-wasm 3.2.24.
//!
//! Same family as `deepcutout_cut_inmem::deep_cutout_socket_fuse_is_watertight`
//! (there the hole is a rectangular through-cut meeting the z=0 interface).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

// Operand notes, measured at capture (brepkit-wasm 3.2.24):
// - `circleinsert_body.bin` already carries 8 same-direction shared edges,
//   minted by its own pocket cut (the same winding-residue family as the
//   deepcutout cut); its free/over census is clean.
// - `circleinsert_sockets.bin` is 4 disjoint feet in one shell (V-E+F = 8,
//   which `validate_solid` mis-reports as an Euler error — it expects one
//   component per shell). Do not "fix" a fixture to satisfy that report.

#[test]
#[ignore = "captured body operand is pre-fix kernel output with baked-in winding errors; re-capture on a released kernel carrying the winding fixes before promoting; see #1538"]
fn circleinsert_socket_fuse_is_watertight() {
    let mut topo = Topology::new();
    let body = load("circleinsert_body.bin", &mut topo);
    let sockets = load("circleinsert_sockets.bin", &mut topo);

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .unwrap();

    let report = brepkit_operations::validate::validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "fuse must validate: {report:?}");
}
