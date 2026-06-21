//! Faithful regression guard for the 3×3 scoop+label+lip label-bracket fuse.
//!
//! Operands captured from the live gridfinity tool (the "fuse #13" that fused
//! the analytic body+lip+scoop solid `{cylinder:32, cone:12, plane:105}` with
//! the flat label bracket `{plane:29}`), via the `serializeSolid` wasm binding.
//!
//! The label bracket is a stepped ramp of thin (≈1 mm tall, 123 mm wide)
//! planar strips. Before the EF-containment fix in `pave_filler::phase_ef`,
//! the edge-face containment margin was `max_chord * 0.5`, computed over ALL
//! boundary edges — so a thin strip's long *straight* sides inflated the
//! margin to ≈1.9 mm. Edge-face crossings of the bin's lip-corner step circles
//! (at z = 13.3 / 14.7 / 16.6, up to ≈1.6 mm outside a strip's z-band) were
//! then wrongly accepted as IN-edges, fed to the face splitter as ~20 spurious
//! sections per strip, and realized as ~50 degenerate zero-area "ribbon"
//! sub-faces. Those over-shared their boundary circles (used 4×) and tripped
//! the mesh fallback.
//!
//! The fix bases the sagitta margin only on *curved* boundary edges (a Line has
//! zero sagitta), so an all-Line ramp strip gets a tolerance-scale margin and
//! the off-band crossings are correctly rejected. This guard asserts the
//! defect's signature is gone: the raw GFA fuse must produce NO over-shared
//! edges and NO fully-degenerate (out-and-back) faces.
//!
//! NOTE: this fuse is not yet fully watertight — a separate, deeper
//! bracket-ramp ↔ lip-corner reconciliation gap leaves ~21 free edges (the
//! ramp strips below z=13.3 corner-split against the cavity wall, those above
//! stay whole, so their shared z=13.91 seam is unmatched). That residual is
//! tracked separately; this guard locks in the EF-margin correctness.

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
use brepkit_topology::wire::OrientedEdge;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// Count over-shared boundary edges (an orientation-independent quantized
/// endpoint pair used by more than two faces).
fn over_shared_edges(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    // Track the distinct faces incident to each quantized edge key (not raw
    // wire occurrences): an edge that appears in a single face's outer+inner
    // wire must not read as over-shared. A manifold edge is incident to exactly
    // two faces; >2 distinct faces is the non-manifold over-share we guard against.
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
    faces_per_edge
        .values()
        .filter(|faces| faces.len() > 2)
        .count()
}

/// Count geometrically free edges (a quantized endpoint pair bordered by a
/// single face) — the open-shell witness.
fn free_edges(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut counts: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

/// Count faces whose outer wire is a pure out-and-back spur (collapses to
/// nothing after stripping consecutive same-endpoint opposite-direction
/// pairs) — i.e. degenerate zero-area "ribbon" faces.
fn degenerate_spur_faces(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut count = 0;
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut oes: Vec<(QPoint, QPoint)> = topo
            .wire(face.outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(|oe: &OrientedEdge| {
                let e = topo.edge(oe.edge()).unwrap();
                (
                    q(topo.vertex(e.start()).unwrap().point()),
                    q(topo.vertex(e.end()).unwrap().point()),
                )
            })
            .collect();
        let mut removed = 0;
        loop {
            let n = oes.len();
            if n < 2 {
                break;
            }
            let found = (0..n).find_map(|i| {
                let j = (i + 1) % n;
                (oes[i].0 == oes[j].1 && oes[i].1 == oes[j].0).then_some((i, j))
            });
            match found {
                Some((i, j)) => {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    oes.remove(hi);
                    oes.remove(lo);
                    removed += 2;
                }
                None => break,
            }
        }
        if oes.is_empty() && removed > 0 {
            count += 1;
        }
    }
    count
}

#[test]
fn scooplabel_bracket_fuse_has_no_degenerate_sections() {
    let mut topo = Topology::new();
    let body = load("scooplabel_inmem_body.bin", &mut topo);
    let bracket = load("scooplabel_inmem_bracket.bin", &mut topo);

    let result = gfa::boolean(&mut topo, BooleanOp::Fuse, body, bracket).unwrap();

    let over = over_shared_edges(&topo, result);
    let spurs = degenerate_spur_faces(&topo, result);
    let curved = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();

    assert_eq!(
        over, 0,
        "thin bracket-ramp strips must not over-share lip-corner edges \
         (EF containment margin regressed?); curved={curved}"
    );
    assert_eq!(
        spurs, 0,
        "label-bracket fuse must not emit degenerate out-and-back ribbon faces \
         (spurious EF sections regressed?)"
    );
    // The corner cylinders/cones survive the raw GFA fuse analytically.
    assert!(
        curved >= 36,
        "expected the body's analytic corner surfaces preserved, got {curved} curved"
    );
}

/// TARGET (currently failing): the full scoop+label bracket fuse must be
/// watertight (free edges = 0), analytic (curved ≥ 36), with the lip-foot cone
/// present. This is the 4-part close: Part 1 (#936, bounded oblique plane-cone
/// conic) and Part 3 (co-circular arc split convention, this PR) are landed.
/// Parts 2 (phase_ff cone-hyperbola section weld) and 4 (multi-surface
/// lip-foot z-stack section chaining + z-step ring reconciliation) are not yet
/// implemented, so the lip-foot stack still leaves free edges. Ignored until
/// Parts 2+4 land; remove `#[ignore]` then.
#[test]
#[ignore = "needs Parts 2+4 (cone-hyperbola weld + z-stack section chaining)"]
fn scooplabel_bracket_fuse_is_watertight() {
    let mut topo = Topology::new();
    let body = load("scooplabel_inmem_body.bin", &mut topo);
    let bracket = load("scooplabel_inmem_bracket.bin", &mut topo);

    let result = gfa::boolean(&mut topo, BooleanOp::Fuse, body, bracket).unwrap();

    let free = free_edges(&topo, result);
    let curved = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    let cone_present = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .any(|&f| topo.face(f).unwrap().surface().type_tag() == "cone");

    assert_eq!(free, 0, "scoop+label fuse must be watertight (free edges)");
    assert!(curved >= 36, "analytic surfaces preserved, got {curved}");
    assert!(cone_present, "lip-foot cone must survive the fuse");
}
