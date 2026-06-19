//! Faithful regression guard: stacking-lip fuse onto a COMPARTMENTED body.
//!
//! The dominant correctness gap behind the uncovered gridfinity scenarios
//! (honeycomb + compartments + scoop, wall-cutouts + honeycomb + compartments,
//! scoop + compartment dividers). Operands captured from the live tool via the
//! `serializeSolid` wasm binding (#915).
//!
//! ## What this isolates
//!
//! A 2×2×4 bin with 2×2 compartments routes the dividers through the
//! "multi-cavity cut" shell path: a rounded-corner box is cut by four cavity
//! extrusions, leaving the divider walls as residue. That cut alone is CLEAN
//! and fully analytic (see `gridfinity_cavitycut_inmem.rs`): the captured body
//! here is 38 faces with 8 corner/divider cylinders, zero free edges.
//!
//! The stacking lip is then fused onto that body. THIS fuse is the bug: the
//! interior divider walls poke up to the same z (≈23) as the lip base, and the
//! fuse fails to reconcile the divider-wall top edges against the lip-bottom
//! annulus. The raw GFA fuse leaves those divider tops as free boundary — 14
//! distinct free LINE edges, all at z≈23, tracing the interior 1.2 mm-thick
//! divider cross (x=±0.6, y=±0.6 spanning to the inner wall at ±39.25). The
//! result is an open, non-manifold shell, so the production boolean falls back
//! to a 228-facet all-planar mesh (every one of the 32 analytic surfaces lost).
//!
//! This is distinct from the previously-fixed plain-bin lip fuses
//! (`lipfuse_fixture.rs`, `scoop_fix_inmem.rs` 3×3): those bodies are hollow
//! shells with NO interior dividers. Here the failure is specifically the
//! interior divider-wall-top ↔ lip-bottom reconciliation.
//!
//! This guard documents the CURRENT (broken) behavior: the raw GFA fuse is
//! non-manifold (free edges present) and the production fuse mesh-falls-back
//! (all-planar, no analytic surfaces). When the divider/lip reconciliation is
//! fixed, both assertions flip — update them then.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp as RawOp;
use brepkit_algo::gfa;
use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp as ProdOp, boolean as prod_boolean};
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

/// Free (incident to exactly one face) boundary-edge count, keyed by an
/// orientation-independent quantized endpoint pair counting *distinct faces*.
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
fn lip_fuse_onto_compartmented_body_is_currently_non_manifold() {
    let mut topo = Topology::new();
    let body = load("lipfuse_cavity2x2_inmem_body.bin", &mut topo);
    let lip = load("lipfuse_cavity2x2_inmem_lip.bin", &mut topo);

    // Sanity: both operands are clean analytic solids before the fuse.
    assert_eq!(
        free_edge_count(&topo, body),
        0,
        "captured cavity-cut body must be watertight (the multi-cavity cut is clean)"
    );
    assert!(
        curved_count(&topo, body) >= 8,
        "captured body must keep its 8 corner/divider cylinders"
    );
    assert_eq!(
        free_edge_count(&topo, lip),
        0,
        "captured lip must be a watertight analytic solid"
    );

    let result = gfa::boolean(&mut topo, RawOp::Fuse, body, lip).unwrap();
    let free = free_edge_count(&topo, result);

    // DOCUMENTED BUG: the raw GFA fuse leaves the interior divider-wall top
    // edges unreconciled against the lip-bottom annulus → free boundary.
    // When fixed this must drop to 0; flip the assertion then.
    assert!(
        free > 0,
        "EXPECTED-FAIL guard: lip fuse onto a compartmented body is currently \
         non-manifold ({free} free edges). If this is now 0, the \
         divider-wall/lip reconciliation was fixed — flip this assertion to \
         assert_eq!(free, 0)."
    );
}

#[test]
fn lip_fuse_onto_compartmented_body_production_falls_back_to_mesh() {
    let mut topo = Topology::new();
    let body = load("lipfuse_cavity2x2_inmem_body.bin", &mut topo);
    let lip = load("lipfuse_cavity2x2_inmem_lip.bin", &mut topo);

    let result = prod_boolean(&mut topo, ProdOp::Fuse, body, lip).unwrap();
    let faces = solid_faces(&topo, result).unwrap().len();
    let curved = curved_count(&topo, result);

    // DOCUMENTED BUG: the open shell trips the production mesh fallback, which
    // re-tessellates to a high all-planar facet count (analytic surfaces lost).
    assert!(
        faces > 150 && curved == 0,
        "EXPECTED-FAIL guard: production lip fuse onto a compartmented body \
         currently mesh-falls-back (got {faces} faces, {curved} curved). When \
         the fuse is fixed this becomes a compact analytic result \
         (faces < 120, curved >= 32) — update the assertion then."
    );
}
