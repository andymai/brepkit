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
//! ## Fixed
//!
//! The body-top split now extracts the exposed divider cross. A holed planar
//! cap (the z=23 body top, with the four compartment openings as holes) cut by
//! the lip-footprint sections that bridge across the divider material between
//! those holes is decomposed via even-odd hole nesting on the woven planar
//! arrangement (`integrate_holes_plane` + `arrangement_regions_from_combined`),
//! so the under-lip ring classifies Inside (dropped) and the divider cross
//! classifies Outside (kept). The 14 free divider-wall-top edges are gone; the
//! production fuse is watertight and fully analytic again.
//!
//! The raw GFA fuse still leaves 8 free edges — short stubs (z=23 → z≈23.3) at
//! the four divider-arm/lip-inner-wall junctions, an out-and-back spur in the
//! LIP inner-wall faces that predates and is independent of the divider-cross
//! reconciliation (it is present on the plain shelled-cup lip too) and is
//! healed by the production sew/spur-removal pass.

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

/// Count of free edges lying wholly at z=23 with both endpoints inside the lip
/// footprint (±39.25) — the interior divider-cross top edges (the former
/// 14-free-edge set this fix targets), excluding the lip-wall spur stubs that
/// climb to z≈23.3.
fn divider_cross_free_count(topo: &Topology, solid: SolidId) -> usize {
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
    let q23 = (23.0 * scale).round() as i64;
    let foot = (39.25 * scale).round() as i64;
    faces_per_edge
        .iter()
        .filter(|(key, faces)| {
            faces.len() == 1
                && key.0.2 == q23
                && key.1.2 == q23
                && key.0.0.abs() <= foot
                && key.1.0.abs() <= foot
                && key.0.1.abs() <= foot
                && key.1.1.abs() <= foot
        })
        .count()
}

#[test]
fn lip_fuse_onto_compartmented_body_resolves_divider_cross() {
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

    // The 14 free divider-wall-top LINE edges at z=23 are gone: no free edge
    // remains in the interior divider cross (every edge with both endpoints at
    // z=23 and inside the lip footprint ±39.25 is now shared by two faces).
    assert_eq!(
        divider_cross_free_count(&topo, result),
        0,
        "the interior divider cross at z=23 must be watertight (was 14 free \
         LINE edges before the holed-cap split fix)"
    );

    // The raw GFA still carries the pre-existing LIP-inner-wall out-and-back
    // spurs (8 short z=23→z≈23.3 stubs at the divider-arm junctions), which the
    // production sew/spur-removal pass heals (see the production guard below).
    // This is independent of the divider-cross reconciliation this guard tracks.
    // `<= 8` (not `== 8`) so this survives the separate spur-removal follow-up:
    // the meaningful invariant is that the 14 divider-cross edges are gone and no
    // NEW free edges appear; the 8 pre-existing lip spurs are an upper bound.
    assert!(
        free <= 8,
        "raw GFA must leave at most the 8 pre-existing lip-inner-wall spur stubs \
         (the 14 divider-cross edges are resolved); got {free}"
    );
}

#[test]
fn lip_fuse_onto_compartmented_body_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let body = load("lipfuse_cavity2x2_inmem_body.bin", &mut topo);
    let lip = load("lipfuse_cavity2x2_inmem_lip.bin", &mut topo);

    let result = prod_boolean(&mut topo, ProdOp::Fuse, body, lip).unwrap();
    let faces = solid_faces(&topo, result).unwrap().len();
    let curved = curved_count(&topo, result);
    let free = free_edge_count(&topo, result);

    // The production fuse is now watertight and fully analytic (no mesh
    // fallback): the body's corner/divider cylinders and the lip's cone/cylinder
    // band survive, and the open-shell collapse to a ~228-facet all-planar mesh
    // is gone. The production sew closes the 8 raw lip-inner-wall spur stubs.
    assert_eq!(
        free, 0,
        "production lip fuse must be watertight, got {free} free"
    );
    assert!(
        curved >= 32,
        "production lip fuse must stay analytic (>=32 curved faces), got {curved}"
    );
    assert!(
        faces <= 130,
        "production lip fuse must be a compact analytic result, got {faces} faces"
    );

    // Volume sanity: a stacking-lip fuse onto the compartmented bin body is a
    // solid in the low-2e4 mm^3 range (the mesh fallback collapsed it to ~2.9e3
    // when the shell went open). Guard a sane positive, non-collapsed volume.
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 1.0e-7).unwrap();
    assert!(
        (20_000.0..30_000.0).contains(&vol),
        "production fuse volume {vol} out of the expected ~2.4e4 mm^3 range"
    );
}
