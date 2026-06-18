//! Regression test for the gridfinity stacking-lip fuse (the default 2×2 bin).
//!
//! The bin body (a shelled rounded-rect tube) is fused with the stacking-lip
//! ring (a tapered rounded-rect loft, cut hollow) flush at the top rim. Their
//! outer walls are coincident, so the FF section lines on the body wall run
//! ALONG the lip face's boundary edges.
//!
//! The bug: when the section was additionally clipped to the opposing (lip)
//! face's boundary, the flush face re-derived an endpoint from its own geometry
//! with sub-tolerance float noise (e.g. -38 vs -38 + 1.4e-14), shifting the
//! section off the shared corner vertex this face's clip had landed on exactly.
//! The wire then failed to close and the assembler classified every shell as a
//! hole → "no outer shell found" → mesh fallback (≈200 all-planar facets with
//! no curved faces, instead of the 78-face analytic solid).
//!
//! The fixtures are the tool's literal operands (brepjs loft/cut geometry,
//! exported via STEP, round-tripping as exact cylinders/cones/planes). The
//! tapered 5-section lip profile does not reconstruct exactly from a native
//! rebuild, so the regression is guarded with these.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
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

/// Count boundary edges used once (free) and more than twice (over-shared),
/// keyed by quantized orientation-independent endpoint pair.
///
/// Quantized at 1e-6 (not the coarser 1e-5 the wall-cut guard uses): the brepjs
/// lip loft/cut leaves sub-micron (~1e-6) corner arcs at the lip's top rim in
/// BOTH the 2×2 and 2×1 operands — legitimately retained (above the kernel's
/// 1e-7 degenerate threshold) and manifold, but a 1e-5 grid merges two of the
/// 2×1's neighbouring corners into one degenerate self-key and reports a false
/// over-share. A real over-share (a 3-face edge) coincides exactly post-merge,
/// so it still registers at this finer resolution; only the sub-micron input
/// noise is separated.
fn edge_use(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
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
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    let free = counts.values().filter(|&&c| c == 1).count();
    let over = counts.values().filter(|&&c| c > 2).count();
    (free, over)
}

fn curved_face_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count()
}

fn assert_lip_fuse_clean(body_step: &str, lip_step: &str) {
    let mut topo = Topology::new();
    let body = read_one(body_step, &mut topo);
    let lip = read_one(lip_step, &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, body, lip).unwrap();

    let (free, over) = edge_use(&topo, result);
    assert_eq!(
        free, 0,
        "stacking-lip fuse must be watertight (no free edges)"
    );
    assert_eq!(
        over, 0,
        "stacking-lip fuse must be manifold (no over-shared edges)"
    );

    // A clean analytic result is ~78 faces with the rounded corners preserved
    // as cylinders + the tapered lip profile as cones; the mesh-boolean fallback
    // produces ~200 all-planar facets (curved == 0).
    let faces = solid_faces(&topo, result).unwrap().len();
    assert!(
        faces < 120,
        "expected a compact analytic result, got {faces} faces (mesh fallback?)"
    );
    assert!(
        curved_face_count(&topo, result) >= 24,
        "stacking-lip fuse must stay analytic (corner cylinders + lip cones preserved)"
    );
}

#[test]
fn gridfinity_stacking_lip_fuse_is_watertight() {
    assert_lip_fuse_clean("lipfuse_body.step", "lipfuse_lip.step");
}

/// The 2×1 (non-square) bin variant. #899 regressed this case differently from
/// the square 2×2: the FF intersection re-traced the lip's bottom-annulus
/// opening ring (a degenerate hole re-trace) and emitted zero-span arc remnants
/// on the depth walls, so the assembler wove a zero-area annulus and an
/// out-and-back spur → "all shells classified as holes", and the mesh fallback
/// then failed "empty wire", silently dropping the lip. (1×2 is the 2×1 rotated
/// 90°, geometrically identical; one fixture guards both.)
#[test]
fn gridfinity_stacking_lip_fuse_2x1_is_watertight() {
    assert_lip_fuse_clean("lipfuse_2x1_body.step", "lipfuse_2x1_lip.step");
}
