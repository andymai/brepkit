//! The 2x2 mixed-detail per-cell half-sockets bin: every boolean in its
//! export chain replays clean and analytic, and the final fuse here succeeds
//! watertight (free=0, over=0). FIXED: tessellating that B-Rep at export
//! tolerance is now watertight too (was 511 mesh boundary edges natively,
//! 259 tool-side after welding). A tessellation defect on a clean B-Rep,
//! the "not every scenario failure is a boolean fallback" class.
//!
//! ROOT (measured stage by stage): the fused floor plane at z=5 (32-edge
//! cross-shaped outer wire + one rounded-rect hole) tessellated to ZERO
//! triangles, and the 395 boundary edges were simply its neighbours' rims.
//! Two of its outer CDT constraints (33.5 mm rails) carried last-ULP
//! coordinate noise from boolean vertex welding (0.25000000000001776), and
//! the CDT's flip-based edge recovery stalled at max_iter through the
//! resulting corridor of exactly-degenerate quads — then RETURNED OK
//! without the edge existing. The recorded-but-absent constraint let
//! remove_exterior's flood pour through the gap and erase the whole face
//! region; the hole seed then removed the rest. Fixes: recover_edge now
//! bisects a non-converging constraint with Steiner midpoints (registering
//! sub-constraints), flood_remove_from_point unions the caller's barrier
//! with the CDT's own constraints, run_planar_cdt lifts Steiner vertices
//! instead of dropping their triangles, and boundary Steiner points are
//! spliced into the shared edge sample chains so neighbour faces stay
//! crack-free.
//!
//! WINDING RESIDUAL (open): after the CDT fix the mesh still counts 116
//! HALF-EDGE boundary edges while every undirected edge is two-sided —
//! pairs of triangles traverse a shared rim in the SAME direction. The
//! owners are the corner cylinder bands (builder faces Id(603)/Id(563)/
//! Id(450)/Id(390), ~40 half-edges each) against their NURBS
//! quarter-socket neighbours (Id(589)/Id(510)/Id(615)) at z 19.7-25.3:
//! one side's triangle winding is inverted along the rim. This was always
//! present, hidden inside the original 511 count (395 missing-face edges
//! plus 116 winding).
//!
//! WINDING ROOT WAS FIRST BLAMED UPSTREAM (capture-era measurement): the
//! captured body operand carried 20 same-sense pairs, the assembly zero,
//! and the fuse output 40 — so the defect looked inherited from an earlier
//! op of the export chain.
//!
//! RE-CAPTURE OVERTURNS THAT (2026-08-06, post-orientation-campaign
//! kernel): the full 9-op chain was re-captured with every operand AND
//! intermediate result serialized. ALL are orientation-clean under the
//! strict check — the campaign fixed the construction ops (the fresh body
//! differs from the capture-era one; the assembly is BYTE-IDENTICAL) —
//! yet the final fuse of the two clean operands still emits exactly
//! "20 shared edges have inconsistent face orientations", and the
//! export-tolerance mesh still counts 116 unmatched half-edges (112
//! survive vertex welding into the exported STL; the export tests stay
//! green only because their oracle is UNDIRECTED edge pairing). The
//! winding defect is therefore born INSIDE the GFA fuse for this
//! configuration: the boolean assembler's face-orientation emission on
//! the NURBS-quarter-socket x cylinder-band geometry. Owner faces of the
//! unmatched half-edges (fresh fuse, z 19.7-25.3): cylinder bands
//! Id(603)/Id(563)/Id(451)/Id(391) against NURBS quarter-sockets
//! Id(589)/Id(511)/Id(615). Probes: `crates/io/examples/orient_scan.rs`
//! (per-.bin strict validation) and `fuse_orient.rs` (fuse + per-face
//! half-edge attribution, ~58 ms).
//!
//! The per-cell dispatch geometry (three full sockets + three quarter
//! sockets, one 1u block mixed) is what distinguishes this from the sibling
//! socket bins that tessellate clean.
//!
//! Operands captured 2026-08-05 via the kernel-test boolean monkey-patch
//! (call 008, the final fuse of the export chain).
//!
//! EXPORT-LEVEL NOTE: the export test already passed on the conflict
//! re-cast kernel because the chain's booleans classified differently;
//! this captured B-Rep kept reproducing the 511-edge leak until the CDT
//! recovery fix closed the root itself.

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
fn mixed_socket_body_operand_orientation_defect_is_pinned() {
    // The captured body operand carries the winding root: 20 shared edges
    // traversed same-sense by adjacent faces (the assembly operand has 0,
    // asserted via the same strict option). Validation only sees this with
    // check_orientation on; the default stays off until the construction-op
    // orientation campaign closes. An upstream fix that changes these
    // numbers must update this pin.
    let mut topo = Topology::new();
    let opts = brepkit_operations::validate::ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    let body = load("mixed_socket_body.bin", &mut topo);
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, body, &opts).unwrap();
    assert!(
        report.issues.iter().any(|i| i
            .description
            .contains("20 shared edges have inconsistent face orientations")),
        "body must report its documented 20 same-sense pairs, got {:?}",
        report.issues
    );
    let assembly = load("mixed_socket_assembly.bin", &mut topo);
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, assembly, &opts).unwrap();
    assert!(
        !report
            .issues
            .iter()
            .any(|i| i.description.contains("inconsistent face orientations")),
        "assembly must stay orientation-clean, got {:?}",
        report.issues
    );
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
fn mixed_socket_tessellation_covers_every_face() {
    // ACTIVE guard for the CDT recovery fix: every mesh edge is used by
    // exactly two triangles (undirected), i.e. no face drops out of the
    // tessellation and no T-junction cracks remain. The stricter half-edge
    // watertightness below stays ignored until the winding residual closes.
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
    let mut uses: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let single = uses.values().filter(|&&c| c == 1).count();
    assert_eq!(
        single, 0,
        "every mesh edge must be two-sided, got {single} one-sided"
    );
}

#[test]
fn mixed_socket_fresh_operands_are_orientation_clean() {
    // The 2026-08-06 re-capture: the post-campaign kernel's export chain
    // feeds the final fuse two orientation-CLEAN operands (the capture-era
    // body's 20 same-sense pairs are gone; the assembly is byte-identical
    // to the old capture). Guards the construction-op orientation campaign
    // at this chain's altitude.
    let mut topo = Topology::new();
    let opts = brepkit_operations::validate::ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    let body = load("mixed_socket_body_fresh.bin", &mut topo);
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, body, &opts).unwrap();
    assert!(
        !report
            .issues
            .iter()
            .any(|i| i.description.contains("inconsistent face orientations")),
        "fresh body must be orientation-clean, got {:?}",
        report.issues
    );
    let assembly = load("mixed_socket_assembly.bin", &mut topo);
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, assembly, &opts).unwrap();
    assert!(
        !report
            .issues
            .iter()
            .any(|i| i.description.contains("inconsistent face orientations")),
        "assembly operand must be orientation-clean, got {:?}",
        report.issues
    );
}

#[test]
fn mixed_socket_fresh_fuse_emits_orientation_defect() {
    // ACTIVE pin of the live defect: fusing the two CLEAN fresh operands
    // still reports exactly 20 same-sense pairs, so the inconsistency is
    // born inside the GFA fuse (assembler face-orientation emission), not
    // inherited. A fuse-side fix must flip this pin to assert cleanliness
    // and un-ignore the watertight repro below.
    let mut topo = Topology::new();
    let body = load("mixed_socket_body_fresh.bin", &mut topo);
    let assembly = load("mixed_socket_assembly.bin", &mut topo);
    let result = brepkit_algo::gfa::boolean(
        &mut topo,
        brepkit_algo::bop::BooleanOp::Fuse,
        body,
        assembly,
    )
    .expect("analytic fuse must succeed");
    let opts = brepkit_operations::validate::ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, result, &opts).unwrap();
    assert!(
        report.issues.iter().any(|i| i
            .description
            .contains("20 shared edges have inconsistent face orientations")),
        "fuse output must report its documented 20 same-sense pairs, got {:?}",
        report.issues
    );
}

#[test]
#[ignore = "residual: 116 unmatched half-edges from inverted triangle winding on the \
            NURBS-quarter-socket vs cylinder rims. Re-captured 2026-08-06: the operands \
            are orientation-clean and the GFA fuse itself emits the 20 same-sense pairs \
            (see the header's re-capture note)"]
fn mixed_socket_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let body = load("mixed_socket_body_fresh.bin", &mut topo);
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
