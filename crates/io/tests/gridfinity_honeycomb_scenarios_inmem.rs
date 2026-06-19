//! Faithful scenario guards for the two HONEYCOMB feature-stacking cases — the
//! biggest uncovered gridfinity gap (the honeycomb hex-grid FUSE/cut + feature
//! entanglement, vs brepkit's prior CUT-only hex coverage).
//!
//! Operands captured from the live tool via `serializeSolid` (#915):
//!
//!   case 1 — honeycomb + 2×2 compartments + scoop (the 3-way stack):
//!     body, scoop ramp (fuse), 4 honeycomb patternCut groups.
//!   case 2 — wall-cutouts (u-shape, 4 sides + interior) + honeycomb + 2×1
//!     compartments: body, 2 wall-cutout tools, 4 honeycomb patternCut groups.
//!
//! In both, the captured BODY already inherited the upstream stacking-lip-fuse
//! mesh fallback (all-planar; see `gridfinity_lipfuse_dividers_inmem.rs`), so
//! the downstream booleans run on a degraded body. The honeycomb cut groups
//! ARE analytic (hex prisms with cylindrical clip-corner faces), which is the
//! coverage value here: the honeycomb prism geometry round-trips, and these
//! fixtures replay the exact production cut sequence against it.
//!
//! These guards document CURRENT behavior: the production stack collapses to a
//! mesh (or, for case 1, the honeycomb cut on the degraded body fails outright
//! with AssemblyFailed). They will flip once the root lip-fuse fallback is
//! fixed and the body returns analytic.

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

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    faces_per_edge.values().filter(|f| f.len() == 1).count()
}

#[test]
fn honeycomb_comp_scoop_stack_is_currently_a_fallback_scenario() {
    let mut topo = Topology::new();
    let body = load("hc_comp_scoop_inmem_body.bin", &mut topo);
    let scoop = load("hc_comp_scoop_inmem_fuse0.bin", &mut topo);

    // Body inherited the upstream lip-fuse fallback: all-planar.
    assert_eq!(
        curved_count(&topo, body),
        0,
        "case1 body already all-planar (inherited lip-fuse fallback)"
    );

    // The 4 honeycomb patternCut groups DO round-trip analytic (hex prisms with
    // cylindrical clip-corner faces) — the coverage value of this fixture.
    let mut total_hc_curved = 0;
    for i in 0..4 {
        let pc = load(&format!("hc_comp_scoop_inmem_pcut{i}.bin"), &mut topo);
        total_hc_curved += curved_count(&topo, pc);
    }
    assert!(
        total_hc_curved >= 8,
        "honeycomb prism groups must round-trip with cylindrical clip corners, got {total_hc_curved}"
    );

    // Production step 1: fuse the scoop onto the (degraded) body — stays
    // all-planar + non-manifold.
    let after_scoop = gfa::boolean(&mut topo, BooleanOp::Fuse, body, scoop).unwrap();
    assert!(
        free_edge_count(&topo, after_scoop) > 0 && curved_count(&topo, after_scoop) == 0,
        "EXPECTED-FAIL: scoop fuse on the degraded body is non-manifold/all-planar"
    );

    // Production step 2: cut the first honeycomb group from that result. On the
    // degraded all-planar body this currently FAILS outright (the boolean
    // cannot assemble an outer shell). Document it as an error, not a panic.
    let pc0 = load("hc_comp_scoop_inmem_pcut0.bin", &mut topo);
    let cut_result = gfa::boolean(&mut topo, BooleanOp::Cut, after_scoop, pc0);
    assert!(
        cut_result.is_err(),
        "EXPECTED-FAIL guard: honeycomb cut on the degraded body currently errors \
         (AssemblyFailed). When the root lip-fuse fallback is fixed and the body \
         returns analytic, this cut should succeed watertight — flip then."
    );
}

#[test]
fn wallcut_honeycomb_comp_stack_is_currently_a_fallback_scenario() {
    let mut topo = Topology::new();
    let mut cur = load("wallcut_hc_comp_inmem_body.bin", &mut topo);

    // Body inherited the upstream lip-fuse fallback: all-planar.
    assert_eq!(
        curved_count(&topo, cur),
        0,
        "case2 body already all-planar (inherited lip-fuse fallback)"
    );

    // Wall-cutout u-shape tools are analytic (rounded corners).
    let mut wallcut_curved = 0;
    for i in 0..2 {
        let c = load(&format!("wallcut_hc_comp_inmem_cut{i}.bin"), &mut topo);
        wallcut_curved += curved_count(&topo, c);
    }
    assert!(
        wallcut_curved >= 16,
        "wall-cutout tools must round-trip analytic (rounded corners), got {wallcut_curved}"
    );

    // Replay the production cut sequence: 2 wall cutouts, then 4 honeycomb
    // groups. The degraded body starts all-planar; the wall-cutout tools add
    // their own rounded-corner cylinders, but the result is non-manifold (the
    // body's pre-existing free boundary from the upstream fallback persists).
    for i in 0..2 {
        let c = load(&format!("wallcut_hc_comp_inmem_cut{i}.bin"), &mut topo);
        cur = gfa::boolean(&mut topo, BooleanOp::Cut, cur, c).unwrap();
    }
    assert!(
        free_edge_count(&topo, cur) > 0,
        "EXPECTED-FAIL guard: wall-cutout cuts on the degraded compartmented body \
         remain non-manifold (free boundary persists). Re-evaluate once the root \
         lip-fuse fallback is fixed (the body returns analytic)."
    );
}
