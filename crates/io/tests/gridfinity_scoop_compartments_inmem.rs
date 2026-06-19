//! Faithful scenario guard: scoop ramp fuse onto a 2×2-compartmented bin body
//! (gridfinity "2×2 standard + lip + 2×2 compartments + scoop", no honeycomb).
//!
//! Operands captured from the live tool via `serializeSolid` (#915): the bin
//! body (`ctx.solid` after the shell + features stages, forExport) and the
//! pre-fused scoop ramp (one all-planar additive solid per compartment).
//!
//! The production pipeline fuses the scoop ramp onto the body. The captured
//! body is ALREADY a 228-facet all-planar solid: it inherited the mesh fallback
//! from the upstream stacking-lip fuse onto the compartmented shell (the root
//! cause — see `gridfinity_lipfuse_dividers_inmem.rs`). The scoop fuse on that
//! degraded body therefore also stays all-planar and non-manifold.
//!
//! This guard captures the END-TO-END scenario state so that when the root lip
//! fuse is fixed (body becomes analytic again) the scoop fuse here can be
//! re-checked against an analytic body — the fixture stays as the scenario's
//! faithful operand pair regardless. It documents CURRENT behavior.

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
fn scoop_fuse_onto_compartmented_body_is_currently_a_fallback_scenario() {
    let mut topo = Topology::new();
    let body = load("scoop_comp2x2_inmem_body.bin", &mut topo);
    let scoop = load("scoop_comp2x2_inmem_fuse0.bin", &mut topo);

    // The scoop ramp operand itself is a clean (all-planar) additive solid.
    assert_eq!(
        free_edge_count(&topo, scoop),
        0,
        "captured scoop ramp must be a watertight solid"
    );

    // The captured BODY already mesh-fell-back upstream (lip fuse onto the
    // compartmented shell): all-planar, no analytic corners.
    assert_eq!(
        curved_count(&topo, body),
        0,
        "scenario body is already all-planar (inherited the lip-fuse fallback)"
    );

    let result = gfa::boolean(&mut topo, BooleanOp::Fuse, body, scoop).unwrap();
    let free = free_edge_count(&topo, result);
    let curved = curved_count(&topo, result);

    // DOCUMENTED: with the upstream fallback still present, the scoop fuse on
    // the all-planar body stays all-planar and non-manifold. When the root lip
    // fuse is fixed and the body is regenerated analytic, this assertion should
    // be revisited (the scoop fuse must then stay analytic + watertight).
    assert!(
        free > 0 && curved == 0,
        "EXPECTED-FAIL guard: scoop fuse onto the (already-degraded) compartmented \
         body is non-manifold and all-planar (got {free} free, {curved} curved). \
         Re-evaluate once the root lip-fuse fallback is fixed."
    );
}
