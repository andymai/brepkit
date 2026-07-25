//! Ready-repro: cutting a kumiko corner wedge by one strut drops to the mesh
//! fallback, losing both cylinders — and this is the root of the whole kumiko
//! export-integrity family.
//!
//! The tool carves each lattice band rather than fusing it: it starts from a
//! `wedge` (a `revolve`, so it carries cylindrical corner faces) and runs
//! `cutter = cutAll(cutter, family)` per strut family. Both operands here are
//! small revolve wedges, COAXIAL about the same corner axis — six faces each,
//! two cylinders each, both watertight. The cut should be an easy analytic
//! case. It is not: the result is 60 faces, ALL PLANAR, which is the mesh
//! fallback signature (diagnostic here precisely because the inputs really did
//! have cylinders to lose).
//!
//! Why it matters. Every corner cut in every band takes this path, so every
//! band comes back all-planar; and by the third strut the accumulated result is
//! also OPEN — `free=3`, then `free=2 over=1` at four and five struts. Those
//! open bands are what reach the goma `cutAllBisect`, where four of eight
//! arrive non-watertight (tool1 free=405 over=38, tool3 393/33, tool5 386/36,
//! tool7 428/40) and poison the export. The flat-wall span, whose operands are
//! all planar boxes, replays perfectly clean (F=1146, watertight), which is why
//! the four even bands are fine.
//!
//! So the kumiko family's root is not that the mesh fallback consumes open
//! output (it does — see `mesh_boolean_fallback`, which warns and proceeds).
//! It is that these coaxial wedge cuts fall back AT ALL.
//!
//! WHY IT FALLS BACK, measured: the analytic path does not produce a wrong
//! result, it produces no result. Raw GFA on this exact pair reports
//! `BuilderSolid: 0 growth shells, 1 hole shells` and aborts with
//! "no outer shell found (all shells classified as holes)" — in 0ms. One shell
//! is built and classified INWARD, so nothing is left to be the outer shell.
//! For `Cut(wedge, strut)` the result should be a single outward shell, so the
//! suspect is orientation or the growth-vs-hole decision in `perform_areas`,
//! not the face splitter. Reproduce with
//! `CAPTURE_DIR=<call4> PREFIX=cut RAW=1 TOOL=0 SHELL_LOG=1
//! ./target/release/examples/replay_cut_capture`.
//!
//! Operands captured from the live tool on a local 2.128.5 build via
//! `kumikoCornerCutCapture.test.ts`; full six-call capture in
//! `~/.cache/brepkit-parity-captures/2026-07-25/kumiko-corner/`.

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

fn surface_mix(
    topo: &Topology,
    sid: brepkit_topology::solid::SolidId,
) -> HashMap<&'static str, usize> {
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        *mix.entry(topo.face(fid).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    mix
}

fn edge_uses(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize) {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        let f = topo.face(fid).unwrap();
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
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
fn operands_are_clean_analytic_wedges() {
    // Guard the guard: if the fixtures ever stop being watertight cylinder-
    // bearing wedges, the test below would be measuring the wrong thing. An
    // unvalidated operand already cost this campaign several passes.
    let mut topo = Topology::new();
    for name in ["kumiko_corner_wedge.bin", "kumiko_corner_strut.bin"] {
        let sid = load(name, &mut topo);
        let mix = surface_mix(&topo, sid);
        assert_eq!(
            mix.get("cylinder").copied(),
            Some(2),
            "{name} should carry 2 cylindrical corner faces, got {mix:?}"
        );
        assert_eq!(
            edge_uses(&topo, sid),
            (0, 0),
            "{name} operand must be watertight and manifold"
        );
    }
}

#[test]
#[ignore = "ready-repro — coaxial wedge cut falls back to mesh, losing both cylinders"]
fn kumiko_corner_wedge_cut_stays_analytic() {
    let mut topo = Topology::new();
    let wedge = load("kumiko_corner_wedge.bin", &mut topo);
    let strut = load("kumiko_corner_strut.bin", &mut topo);

    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        wedge,
        strut,
    )
    .expect("corner wedge cut should not fail outright");

    let mix = surface_mix(&topo, result);
    let faces: usize = mix.values().sum();

    // The tell, and the reason this fixture exists: both operands carry
    // cylinders, so an analytic result must keep some. All-planar means the
    // mesh fallback ran.
    assert!(
        mix.get("cylinder").copied().unwrap_or(0) > 0,
        "cut must stay analytic and keep cylindrical corner faces, got {faces} faces {mix:?}"
    );
}
