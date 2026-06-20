//! Captured-operand regression + documented residual for the gridfinity "A2"
//! bin: width 2, depth 2, height 4, 2×1 compartments, stacking lip, honeycomb
//! wall pattern, four u-shape wall cutouts.
//!
//! Operands captured from the live gridfinity tool via the `serializeSolid`
//! wasm binding (byte-exact in-memory arena, no geometry re-derivation):
//!   a2hcomb_prewall_body = body box + stacking lip (shell stage output)
//!   a2hcomb_wcut0/1      = the two wall-cutout cut tools
//!   a2hcomb_pcut0..3     = the four honeycomb wall-pattern cut tools
//!
//! The tool's boolean stage runs, in order: first the wall cutouts
//! `Cut(Cut(body, wcut0), wcut1)`, then the honeycomb pattern as one
//! `cutAllBisect` pass `Cut(result, pcut0..3)`.
//!
//! #923 WIN (regression guard). The wall cutouts are now WATERTIGHT and
//! analytic after #923: `Cut(Cut(prewall_body, wcut0), wcut1)` gives 156
//! faces, 52 curved, free=0. Before #923 the second wall-cut dropped a
//! contained back-wall, leaving the body open (the A2 bin mesh-fell-back).
//! `wallcut_step_is_watertight` locks this in for the exact A2 honeycomb-bin
//! operands (a different capture than `gridfinity_wallcut_seq_inmem`, which
//! uses the same tool but no honeycomb).
//!
//! PARTIAL FIX + RESIDUAL (documented). At the raw GFA level, `pcut0` (a
//! full-footprint rounded-rect cutter whose outer wall is COINCIDENT with the
//! body's ±41.75 outer wall) used to make the assembler fail outright with
//! AssemblyFailed "no outer shell found (all shells classified as holes)" — the
//! #899/#911/#923 coincident-wall family. That catastrophic failure (the root
//! of the in-tool hang) is now FIXED: the honeycomb cap (pcut0's z=23 top) is no
//! longer over-covered by an invalid overlapping wire-builder partition, so
//! `Cut(body, pcut0)` returns Ok. In-tool this stops the multi-minute hang —
//! `cutAllBisect` no longer retries a throwing n-way cut by recursively
//! bisecting (re-running the expensive NURBS-faced `pcut1` each time).
//!
//! `honeycomb_cut_no_longer_throws` locks in that pcut0..3 all return Ok. A
//! residual remains: the cuts leave some free edges (an open shell), because the
//! cap's correct partition introduces vertices its NEIGHBOR analytic faces don't
//! yet share (a cross-face partition-consistency gap, not an assembler failure).
//! `honeycomb_cut_residual_documented` pins those residual free-edge counts so a
//! future cross-face-reconciliation fix is forced to update them (toward free=0).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_algo::gfa;
use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// (free edges, over-shared edges) by quantized endpoint position.
fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    let mut occ: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
                *occ.entry(key).or_insert(0) += 1;
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

/// Build the clean wall-cut body the honeycomb pattern cut operates on.
fn build_wallcut_body(topo: &mut Topology) -> SolidId {
    let body = load("a2hcomb_prewall_body.bin", topo);
    let w0 = load("a2hcomb_wcut0.bin", topo);
    let after0 = gfa::boolean(topo, BooleanOp::Cut, body, w0).unwrap();
    let w1 = load("a2hcomb_wcut1.bin", topo);
    gfa::boolean(topo, BooleanOp::Cut, after0, w1).unwrap()
}

#[test]
fn wallcut_step_is_watertight() {
    // #923 regression guard for the A2 honeycomb-bin operands: the wall-cutout
    // sequence the honeycomb pattern then cuts must be watertight + analytic.
    let mut topo = Topology::new();
    let body = build_wallcut_body(&mut topo);
    let (free, over) = edge_health(&topo, body);
    let faces = solid_faces(&topo, body).unwrap().len();
    assert_eq!(
        free, 0,
        "A2 wall-cut body must be watertight (got free={free})"
    );
    assert_eq!(
        over, 0,
        "A2 wall-cut body must be manifold (got over={over})"
    );
    assert_eq!(faces, 156, "A2 wall-cut body face count (got {faces})");
    assert!(
        curved_count(&topo, body) >= 52,
        "A2 wall-cut body preserves the lip cones + corner cylinders"
    );
}

#[test]
fn honeycomb_cut_no_longer_throws() {
    // WIN: the coincident-wall cap over-coverage that made pcut0 fail the
    // assembler (AssemblyFailed "all shells classified as holes" — the in-tool
    // hang root) is fixed. Every honeycomb pattern cut now returns Ok and stays
    // manifold (over-shared edge count 0); the in-tool `cutAllBisect` no longer
    // retries a throwing n-way cut, so the multi-minute hang is gone.
    for tool in [
        "a2hcomb_pcut0.bin",
        "a2hcomb_pcut1.bin",
        "a2hcomb_pcut2.bin",
        "a2hcomb_pcut3.bin",
    ] {
        let mut topo = Topology::new();
        let body = build_wallcut_body(&mut topo);
        let p = load(tool, &mut topo);
        let r = gfa::boolean(&mut topo, BooleanOp::Cut, body, p);
        assert!(
            r.is_ok(),
            "{tool} must not fail the assembler: {:?}",
            r.err()
        );
        let solid = r.unwrap();
        let (_free, over) = edge_health(&topo, solid);
        assert_eq!(over, 0, "{tool} result must stay manifold (over-shared=0)");
    }
}

#[test]
fn honeycomb_cut_pcut0_is_watertight_and_analytic() {
    // WIN: the full-footprint honeycomb cutter `pcut0` (whose outer wall is
    // coincident with the body's ±41.75 outer wall) now produces a WATERTIGHT,
    // analytic result. Previously its z=23 cap dropped one rounded-corner cell
    // (free=8, an open shell) because `dedup_collinear_sections` mistook two
    // ADJACENT PaveBlock fragments of the corner divider wall — meeting end-to-
    // end at a junction whose shared vertex carried ~1e-3 of float noise — for a
    // subset/superset pair and deleted the shorter fragment. The cap's corner
    // cell then had a gap in its boundary and the planar arrangement absorbed it
    // into a giant overlapping region (dropped at assembly), leaving the cell's
    // walls free.
    //
    // The fix requires a real CONTAINMENT (the overlap spans almost the whole
    // shorter segment), not a noisy touching overlap, before dropping a section
    // as a redundant subset — so both adjacent fragments survive and the cell
    // closes. This is the cascade root: a watertight `pcut0` keeps the production
    // `operations::boolean` from rejecting the result and mesh-falling-back to an
    // all-planar body, which then exploded the cost of every subsequent
    // honeycomb cut.
    let mut topo = Topology::new();
    let body = build_wallcut_body(&mut topo);
    let p = load("a2hcomb_pcut0.bin", &mut topo);
    let r = gfa::boolean(&mut topo, BooleanOp::Cut, body, p).unwrap();
    let (free, over) = edge_health(&topo, r);
    assert_eq!(free, 0, "pcut0 must be watertight (got free={free})");
    assert_eq!(over, 0, "pcut0 must be manifold (got over-shared={over})");
    assert!(
        curved_count(&topo, r) > 0,
        "pcut0 must stay analytic (corner cylinders/cones preserved), not mesh fallback"
    );
}

#[test]
fn honeycomb_cut_residual_documented() {
    // DOCUMENTED RESIDUAL: `pcut1..3` (the lip-/NURBS-walled honeycomb cutters)
    // still return Ok but leave some free edges (an open shell) from a SEPARATE,
    // pre-existing cause — free edges spread across many z-levels through the
    // stacking-lip cones and the NURBS honeycomb walls, unrelated to the z=23 cap
    // dedup fixed for `pcut0`. These ceilings pin the current state.
    let residual_free: &[(&str, usize)] = &[
        ("a2hcomb_pcut1.bin", 52),
        ("a2hcomb_pcut2.bin", 35),
        ("a2hcomb_pcut3.bin", 15),
    ];
    for &(tool, expect_free) in residual_free {
        let mut topo = Topology::new();
        let body = build_wallcut_body(&mut topo);
        let p = load(tool, &mut topo);
        let r = gfa::boolean(&mut topo, BooleanOp::Cut, body, p).unwrap();
        let (free, _over) = edge_health(&topo, r);
        // Upper bound, not exact: `edge_health` quantizes raw 3-D coords, so a
        // last-ULP position difference (e.g. a different FPU on a cross-compiled
        // target) could nudge the count. The qualitative invariant (all cuts
        // return Ok + manifold) is asserted by `honeycomb_cut_no_longer_throws`.
        assert!(
            free <= expect_free,
            "RESIDUAL: honeycomb {tool} free-edge count {free} exceeds the documented \
             ceiling {expect_free} (regression in the cap arrangement?)"
        );
    }
}
