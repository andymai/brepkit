//! Captured-operand READY-REPRO for the custom-shape T lip band cut — the
//! operation that PRODUCES the malformed lip operand behind the T body+lip
//! fuse failure ("assembly failed: no outer shell found").
//!
//! The lip band is `cut(outer prism, inner frustum)`. Both operands are
//! well-formed solids: the outer is a T-outline prism (constant ±62.75 — the
//! capture calls it `lip-outerfrustum`, but its walls are vertical and its
//! corners are cylinders, so the fixture here is named for the geometry), the
//! inner a T-outline frustum widening ±61.55 → ±62.75 so that the two lateral
//! surfaces meet exactly at the top rim. The correct result is a band that
//! tapers to zero thickness at the top, with its z=-2.6 bottom a single RING
//! face (±62.75 outer wire, ±61.55 inner wire).
//!
//! What the cut actually produces is a bottom of TWO nested faces of the SAME
//! orientation — a ±62.75 16-edge disc (the target's bottom, passed through
//! UNSPLIT) alongside a ±61.55 26-edge disc — each with `inners = 0`. The
//! doubled boundary makes ray parity even, so `classify_point` still maps the
//! band correctly, while signed-volume integration double-counts: 20111.8
//! against the true 60735.9 − 54643.9 = 6092.0.
//!
//! The doubled boundary also makes the measured volume TRANSLATION-VARIANT,
//! which is the sharpest available detector for this class: this band
//! (bbox `z[-2.60, 4.40]`) measures 20111.8, while the identical shape after
//! the +15.90 z-translate the downstream fuse is captured with —
//! `lipfuse-top.bin`, same `F=98 curved=48`, same ±62.75 XY bbox, same 7.0
//! height — measures 65641.2. For a well-formed closed boundary the
//! z-dependent terms of the signed-volume integral cancel, so volume cannot
//! depend on position. Unlike a volume-vs-classification disagreement, this
//! check needs no second oracle: translate and re-measure.
//!
//! Two hypotheses are RULED OUT by synthetic controls, both of which are
//! correct today (single ring face, exact volume):
//!   * nested coplanar bottoms alone — box minus a nested box sharing the
//!     bottom plane gives one ring face and volume 3400.000 exactly;
//!   * the zero-thickness pinch alone — cylinder r=10 minus a cone r 9→10 over
//!     the same height gives one ring face and volume 303.687 exactly.
//!
//! The remaining differentiator is the non-convex T outline with arc corners,
//! consistent with the earlier layer map for this family (a concave corner
//! appearing as chord segments on one operand against a true arc on the other).
//!
//! FIXED (classifier depth probe): the band-bottom ring's interior sample sits
//! in the ~1.2mm annulus, coplanar with the tool's bottom cap.
//! `classify_coincident_coplanar` probed by stepping the wedge tip toward the
//! face centroid by fractions of that distance — overshooting the thin band
//! straight into the hole, so no valid probe was found and the unstable ray-cast
//! decided the ring was Inside and dropped it. A thin-band absolute-nudge probe
//! (staying near the tip inside the band) now classifies the ring Outside, so the
//! bottom is a single-covered ring: volume 6090.8, not the doubled 20108.8.
//!
//! Residual (benign): the band bottom tiles as ONE ring PLUS two tiny reflex-
//! corner (T-armpit) pieces — 3 planar faces, not 1 — from redundant coplanar
//! sections the FF-coplanar phase emits at the concave corners. The tiling is
//! exact (areas sum to the band annulus), position-watertight, and
//! translation-invariant, so this is over-segmentation, not the doubled-bottom
//! defect. The single-cover assertions below catch a regression to the doubling.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// Horizontal faces of `solid` lying on the plane `z == target`.
fn faces_on_plane(topo: &Topology, solid: SolidId, target: f64) -> Vec<(SolidFace, usize, usize)> {
    let mut out = Vec::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            continue;
        }
        let wire = topo.wire(face.outer_wire()).unwrap();
        let mut zs = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            for v in [edge.start(), edge.end()] {
                zs.push(topo.vertex(v).unwrap().point().z());
            }
        }
        if zs.is_empty() {
            continue;
        }
        let zmin = zs.iter().copied().fold(f64::MAX, f64::min);
        let zmax = zs.iter().copied().fold(f64::MIN, f64::max);
        if (zmax - zmin).abs() > 1e-9 || (zmin - target).abs() > 1e-6 {
            continue;
        }
        out.push((fid, wire.edges().len(), face.inner_wires().len()));
    }
    out
}

type SolidFace = brepkit_topology::face::FaceId;

#[test]
fn lipband_cut_bottom_is_a_single_ring() {
    let mut topo = Topology::new();
    let outer = load("lipband_outerprism.bin", &mut topo);
    let inner = load("lipband_innerfrustum.bin", &mut topo);

    let v_outer = brepkit_operations::measure::solid_volume(&topo, outer, 0.005).unwrap();
    let v_inner = brepkit_operations::measure::solid_volume(&topo, inner, 0.005).unwrap();

    let band = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    // Single-cover: the band bottom (z == -2.6) must be a ring, so EXACTLY ONE
    // planar face on that plane carries an inner (hole) wire — never two nested
    // hole-less discs (the doubled bottom). Over-segmentation into a ring plus a
    // few reflex-corner pieces is tolerated; a doubling is not.
    let bottom = faces_on_plane(&topo, band, -2.6);
    let with_hole = bottom.iter().filter(|(_, _, inners)| *inners == 1).count();
    let hole_less = bottom.iter().filter(|(_, _, inners)| *inners == 0).count();
    assert_eq!(
        with_hole, 1,
        "band bottom must carry exactly one ring (hole) face, got {with_hole}: {bottom:?}"
    );

    // The bottom tiles the annulus once: total area equals the outer bottom minus
    // the inner bottom. A doubled bottom (the original defect) over-covers the
    // inner disc and this sum jumps by the inner disc's area.
    let area_on_bottom = |solid| -> f64 {
        faces_on_plane(&topo, solid, -2.6)
            .iter()
            .map(|(fid, _, _)| brepkit_operations::measure::face_area(&topo, *fid, 0.01).unwrap())
            .sum()
    };
    let band_bottom_area: f64 = bottom
        .iter()
        .map(|(fid, _, _)| brepkit_operations::measure::face_area(&topo, *fid, 0.01).unwrap())
        .sum();
    let expected_bottom_area = area_on_bottom(outer) - area_on_bottom(inner);
    assert!(
        (band_bottom_area - expected_bottom_area).abs() < expected_bottom_area * 1e-3,
        "band bottom area {band_bottom_area:.3} should equal outer-inner bottom \
         {expected_bottom_area:.3} (hole_less faces there: {hole_less})"
    );

    // Volume follows from the two operands; the doubled bottom over-counts it
    // roughly 3.3x, so the band only has to be tight enough to separate the two
    // regimes.
    let v_band = brepkit_operations::measure::solid_volume(&topo, band, 0.005).unwrap();
    let expected = v_outer - v_inner;
    assert!(
        (v_band - expected).abs() < expected * 5e-3,
        "band volume {v_band:.2} should equal outer {v_outer:.2} - inner {v_inner:.2} = {expected:.2}"
    );
}
