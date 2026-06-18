//! Regression test for the gridfinity scoop-fuse-into-compartmented-bin
//! non-manifold.
//!
//! A faceted concave scoop-ramp prism is fused into a no-lip compartmented bin
//! (a box cut by two cavity prisms with rounded corners). The fuse goes
//! non-manifold three ways:
//!
//! 1. The scoop's sharp staircase pokes into the bin's rounded corner cylinders,
//!    so each tread plane meets a corner cylinder in a sub-millimetre ellipse
//!    arc. Those per-tread arcs must assemble into one continuous split curve so
//!    the cylinder splits into a lower band (engulfed by the scoop → dropped)
//!    and an upper band (faces the open cavity → kept). When the arcs do not
//!    chain, the cylinder stays whole, its interior sample lands inside the
//!    scoop, and it is dropped entirely → free edges.
//! 2. The compartmented bin's cavity walls are exported by STEP — and built by
//!    the tool's rounded-rect extrude — as planar B-splines (NURBS). Their FF
//!    intersections with the scoop's planes never enter the analytic
//!    plane×plane path, so the coincident/abutting wall regions never
//!    same-domain merge. They must be recognised as planes before the fuse.
//! 3. The scoop's bottom face (z=0) is coincident with the bin's bottom cap; the
//!    scoop's side faces cut footprint lines onto the rounded-rect cap, which the
//!    wire builder traces as one self-crossing loop.
//!
//! The fixtures are the tool's literal operands (brepjs sketch+extrude geometry,
//! exported via STEP), fused exactly as the tool fuses them: the compartmented
//! bin is built by cutting `bin_box` with `cavity_0` then `cavity_1` (NO
//! `convert_to_elementary` — the tool keeps the NURBS walls at fuse time), then
//! the scoop ramp is fused into that bin. A native rounded-rect rebuild does not
//! reproduce the exact prism/seam structure, so the regression is guarded with
//! these.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn read_one(name: &str, topo: &mut Topology) -> SolidId {
    let text = std::fs::read_to_string(fixture(name)).unwrap();
    let solids = brepkit_io::step::reader::read_step(&text, topo).unwrap();
    assert_eq!(solids.len(), 1, "expected exactly one solid in {name}");
    solids[0]
}

type QPoint = (i64, i64, i64);
fn qp(p: brepkit_math::vec::Point3) -> QPoint {
    let s = 1.0e5;
    (
        (p.x() * s).round() as i64,
        (p.y() * s).round() as i64,
        (p.z() * s).round() as i64,
    )
}

/// Count how many faces each undirected edge (by quantized endpoints) borders.
fn edge_incidence(topo: &Topology, solid: SolidId) -> HashMap<(QPoint, QPoint), usize> {
    let mut counts: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = qp(topo.vertex(e.start()).unwrap().point());
                let b = qp(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Count how many faces each *topological* edge (by `EdgeId`, per shell)
/// borders. Keying by id — not by quantized geometry — is the stronger
/// manifold witness: it fails when two geometrically coincident but
/// topologically distinct edges leave a real crack (the quantized-endpoint
/// key would merge those into one and hide it). A watertight GFA result shares
/// each non-seam boundary via a single `EdgeId` used by exactly two faces,
/// which is the same invariant the boolean engine enforces internally
/// (`is_closed_manifold`). Walks inner (cavity) shells too — each shell is an
/// independent closed surface, so edges are pooled per shell.
fn topological_edge_use_per_shell(topo: &Topology, solid: SolidId) -> Vec<HashMap<EdgeId, usize>> {
    let s = topo.solid(solid).unwrap();
    std::iter::once(s.outer_shell())
        .chain(s.inner_shells().iter().copied())
        .map(|shell_id| {
            let mut counts: HashMap<EdgeId, usize> = HashMap::new();
            for &fid in topo.shell(shell_id).unwrap().faces() {
                let face = topo.face(fid).unwrap();
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    let wire = topo.wire(wid).unwrap();
                    for oe in wire.edges() {
                        *counts.entry(oe.edge()).or_insert(0) += 1;
                    }
                }
            }
            counts
        })
        .collect()
}

#[test]
fn scoop_fuse_corner_cylinders_are_watertight() {
    let mut topo = Topology::new();
    let bin_box = read_one("scoop_bin_box.step", &mut topo);
    let cavity_0 = read_one("scoop_cavity_0.step", &mut topo);
    let cavity_1 = read_one("scoop_cavity_1.step", &mut topo);
    let scoop = read_one("scoop_scoop_0.step", &mut topo);

    // Build the compartmented bin exactly as the tool does: cut the two cavities
    // from the box. The cavity walls stay planar NURBS (the tool does NOT
    // convert them to planes before the scoop fuse).
    let comp_bin = boolean(&mut topo, BooleanOp::Cut, bin_box, cavity_0).unwrap();
    let comp_bin = boolean(&mut topo, BooleanOp::Cut, comp_bin, cavity_1).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, comp_bin, scoop).unwrap();

    // Watertight (geometric witness): every edge bordered by exactly two faces,
    // keyed by quantized endpoints so distinct ids on a shared curve still match.
    let counts = edge_incidence(&topo, result);
    let free = counts.values().filter(|&&c| c == 1).count();
    let over = counts.values().filter(|&&c| c > 2).count();
    assert_eq!(
        free, 0,
        "scoop fuse result has {free} free edges (non-manifold)"
    );
    assert_eq!(over, 0, "scoop fuse result has {over} over-shared edges");

    // Watertight (topological witness): every boundary edge is one shared
    // `EdgeId` used by exactly two faces within its shell. Stronger than the
    // geometric count — it cannot hide a crack where two coincident-but-distinct
    // edges share a quantized key. This is the invariant the boolean engine
    // itself gates on (`is_closed_manifold`).
    for (shell_idx, counts) in topological_edge_use_per_shell(&topo, result)
        .iter()
        .enumerate()
    {
        let free_ids = counts.values().filter(|&&c| c == 1).count();
        let over_ids = counts.values().filter(|&&c| c > 2).count();
        assert_eq!(
            free_ids, 0,
            "scoop fuse shell {shell_idx} has {free_ids} topologically free edges \
             (distinct EdgeId used by only one face)"
        );
        assert_eq!(
            over_ids, 0,
            "scoop fuse shell {shell_idx} has {over_ids} topologically over-shared edges"
        );
    }

    // No mesh-boolean fallback: the GFA result must survive as analytic
    // geometry. The fallback tessellates everything to planes (hundreds of
    // faces); the analytic fuse keeps the 8 corner cylinders and ~50 planes.
    let faces = solid_faces(&topo, result).unwrap();
    let cylinders = faces
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        cylinders, 8,
        "expected 8 analytic corner cylinders to survive (got {cylinders}); \
         a mesh-boolean fallback would tessellate them away"
    );
    assert!(
        faces.len() < 100,
        "expected a small analytic result, got {} faces (mesh fallback?)",
        faces.len()
    );
}
