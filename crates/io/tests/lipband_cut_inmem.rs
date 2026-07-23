//! Captured-operand READY-REPRO for the custom-shape T lip band cut — the
//! operation that PRODUCES the malformed lip operand behind the T body+lip
//! fuse failure ("assembly failed: no outer shell found").
//!
//! The lip band is `cut(outer prism, inner frustum)`. Both operands are
//! well-formed solids: the outer is a T-outline prism (constant ±62.75), the
//! inner a T-outline frustum widening ±61.55 → ±62.75 so that the two lateral
//! surfaces meet exactly at the top rim. The correct result is a band that
//! tapers to zero thickness at the top, with its z=-2.6 bottom a single RING
//! face (±62.75 outer wire, ±61.55 inner wire).
//!
//! What the cut actually produces is a bottom of TWO nested faces of the SAME
//! orientation — a ±62.75 16-edge disc (the target's bottom, passed through
//! UNSPLIT) alongside a ±61.55 26-edge disc — each with `inners = 0`. The
//! doubled boundary makes ray parity even, so `classify_point` still maps the
//! band correctly, while signed-volume integration double-counts: 20108.8
//! against the true 60735.9 − 54643.9 = 6092.0.
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
//! Ignored until the split lands. Downstream effect if you need it: the
//! malformed band is `lipfuse-top.bin`, whose two bottom discs both survive
//! the body fuse as the 28 triple-shared z=13.3 interface edges.

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
#[ignore = "ready repro: the lip band cut emits two nested same-orientation \
            bottom discs instead of one ring face"]
fn lipband_cut_bottom_is_a_single_ring() {
    let mut topo = Topology::new();
    let outer = load("lipband_outerfrustum.bin", &mut topo);
    let inner = load("lipband_innerfrustum.bin", &mut topo);

    let v_outer = brepkit_operations::measure::solid_volume(&topo, outer, 0.005).unwrap();
    let v_inner = brepkit_operations::measure::solid_volume(&topo, inner, 0.005).unwrap();

    let band = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    // The band's bottom must be ONE ring face: a single outer wire carrying a
    // single inner (hole) wire. Today it is two nested discs, each hole-less.
    let bottom = faces_on_plane(&topo, band, -2.6);
    assert_eq!(
        bottom.len(),
        1,
        "expected one ring face on the band bottom, got {} faces: {bottom:?}",
        bottom.len()
    );
    assert_eq!(
        bottom[0].2, 1,
        "the band bottom must carry exactly one inner (hole) wire, got {}",
        bottom[0].2
    );

    // Volume follows from the two operands; the doubled bottom over-counts it.
    let v_band = brepkit_operations::measure::solid_volume(&topo, band, 0.005).unwrap();
    let expected = v_outer - v_inner;
    assert!(
        (v_band - expected).abs() < 1.0,
        "band volume {v_band:.2} should equal outer {v_outer:.2} - inner {v_inner:.2} = {expected:.2}"
    );
}
