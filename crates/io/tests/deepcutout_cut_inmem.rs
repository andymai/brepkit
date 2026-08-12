//! Cutting a corner-placed deep cutout (a box flush with the recess walls,
//! punching through the fill) out of a solid-mode recessed bin must stay
//! analytic and validate — the #1538 "solid mode with cutout cutDepth >
//! solidSurfaceZ" scenario, captured tool-side on brepkit-wasm 3.2.24.
//!
//! ROOT (FIXED): the cutter's top edges are coincident with the recess
//! ledge's boundary, so their in-common spans become CommonBlock shared
//! edges minted in the PARTNER solid's direction. `expand_edge`
//! (fill_images_faces) assumed every image sub-edge was aligned with its
//! parent and emitted a backwards sub-edge, giving two unclosed wires and 11
//! same-direction shared edges. The ops layer correctly rejected that and
//! paid an all-planar mesh fallback whose volume was wrong by +7, which then
//! poisoned the downstream socket fuse into an open export. Images are now
//! oriented by endpoint chaining.

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

#[test]
fn deep_cutout_cut_is_analytic_and_validates() {
    let mut topo = Topology::new();
    let body = load("deepcutout_body.bin", &mut topo);
    let tool = load("deepcutout_tool.bin", &mut topo);
    let vol_body = brepkit_operations::measure::oriented_solid_volume(&topo, body, 0.05).unwrap();
    let vol_tool = brepkit_operations::measure::oriented_solid_volume(&topo, tool, 0.05).unwrap();

    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        body,
        tool,
    )
    .unwrap();

    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        0,
        "the cut must stay on the exact path"
    );

    // Unclosed wires were the defect; a residue of same-direction shared
    // edges remains in this family (strict pin below, ignored).
    let report = brepkit_operations::validate::validate_solid(&topo, result).unwrap();
    let unclosed: Vec<_> = report
        .issues
        .iter()
        .filter(|i| i.description.contains("not closed"))
        .collect();
    assert!(unclosed.is_empty(), "wires must close: {unclosed:?}");

    // The tool is entirely inside the body, so the cut volume is exact.
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - (vol_body - vol_tool)).abs() < 0.01,
        "cut volume {vol:.3} must equal body minus tool {:.3}",
        vol_body - vol_tool
    );

    let curved = brepkit_topology::explorer::solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(curved > 0, "all-planar output is the mesh-fallback tell");
}

/// READY-REPRO: the cut result must also pass strict validation. Today it
/// carries 9 same-direction shared edges (down from 11 plus two unclosed
/// wires before the expand_edge fix) — the coincident-plane winding residue
/// this family still owes.
#[test]
#[ignore = "cut result carries same-direction shared edges; see #1538"]
fn deep_cutout_cut_validates_strictly() {
    let mut topo = Topology::new();
    let body = load("deepcutout_body.bin", &mut topo);
    let tool = load("deepcutout_tool.bin", &mut topo);
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        body,
        tool,
    )
    .unwrap();
    let report = brepkit_operations::validate::validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "cut result must validate: {report:?}");
}

/// READY-REPRO for the coplanar-interface holed-cap fuse family (#1538 tail):
/// fusing the through-holed cutout body onto the socket assembly meets the
/// sockets' top plane with a bottom face that carries the cutout hole, and
/// raw GFA emits free edges on that interface. Same family as the circle
/// insert case (`circleinsert_interface_fuse_inmem.rs`).
#[test]
#[ignore = "coplanar-interface fuse with a holed interface face emits free edges; see #1538"]
fn deep_cutout_socket_fuse_is_watertight() {
    let mut topo = Topology::new();
    let body = load("deepcutout_result_body.bin", &mut topo);
    let sockets = load("circleinsert_sockets.bin", &mut topo);

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .unwrap();

    let report = brepkit_operations::validate::validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "fuse must validate: {report:?}");
}
