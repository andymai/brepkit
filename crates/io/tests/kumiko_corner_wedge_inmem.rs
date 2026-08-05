//! Cutting a kumiko corner wedge by one strut must stay analytic AND
//! watertight. Both operands are small coaxial revolve wedges about the same
//! corner axis: six faces each, two cylinders each, watertight and (since the
//! 2026-08-04 re-capture on a post-revolve-fix kernel) outward-oriented.
//!
//! History in brief: the original captures were globally INVERTED wedges (the
//! segmented-revolve winding defect, fixed in `revolve.rs` and verified in
//! `operations/tests/regress_kumiko_corner_wedge.rs`), and the cut fell to the
//! all-planar mesh fallback, which poisoned every kumiko corner band.
//!
//! Current state on the re-captured operands: the cut runs ANALYTIC in 2 ms
//! and keeps both cylinders — but it is WRONG twice over. Point oracles show
//! the strut genuinely OVERLAPS the wedge (B=Inside at (3.15, 0.2, 11.75),
//! which is inside A; the strut protrudes through the wedge's y=0 cap plane),
//! yet the result keeps the FULL wedge volume (285.861, nothing removed) AND
//! drops the y=0 cap, whose region is coplanar with the strut's boundary
//! there (the cap's interior sample lies ON the strut boundary; the
//! coincident-coplanar fast path defers straddlers and ray-cast says Inside).
//! The ignored test below pins the watertight goal state.
//!
//! Re-capture recipe: kernel-test boolean monkey-patch on a 1x1x4 mitsukude
//! corner-wrap generation; the wedge pair is the F=6 two-cylinder cut call
//! (four congruent instances, one per corner).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn surface_mix(
    topo: &Topology,
    sid: brepkit_topology::solid::SolidId,
) -> HashMap<&'static str, usize> {
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        *mix.entry(topo.face(fid).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    mix
}

fn edge_uses(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize) {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        let f = topo.face(fid).unwrap();
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
    )
}

#[test]
fn operands_are_clean_analytic_wedges() {
    // Guard the guard: if the fixtures ever stop being watertight cylinder-
    // bearing wedges, the test below would be measuring the wrong thing. An
    // unvalidated operand already cost this campaign several passes.
    let mut topo = Topology::new();
    for name in ["kumiko_corner_wedge.bin", "kumiko_corner_strut.bin"] {
        let sid = load(name, &mut topo);
        let mix = surface_mix(&topo, sid);
        assert_eq!(
            mix.get("cylinder").copied(),
            Some(2),
            "{name} should carry 2 cylindrical corner faces, got {mix:?}"
        );
        assert_eq!(
            edge_uses(&topo, sid),
            (0, 0),
            "{name} operand must be watertight and manifold"
        );
    }
}

#[test]
fn captured_operands_are_outward_oriented() {
    // Re-captured 2026-08-04 on a post-revolve-fix kernel: both wedges are
    // outward-oriented now, which is what lets the cut below run analytically
    // at all. `oriented_solid_volume` keeps the sign that `solid_volume`
    // discards.
    let mut topo = Topology::new();
    for name in ["kumiko_corner_wedge.bin", "kumiko_corner_strut.bin"] {
        let sid = load(name, &mut topo);
        let signed = brepkit_operations::measure::oriented_solid_volume(&topo, sid, 0.05).unwrap();
        assert!(
            signed > 0.0,
            "{name} should be outward-oriented post-re-capture, got {signed:.3}"
        );
    }
}

#[test]
fn kumiko_corner_wedge_cut_stays_analytic() {
    let mut topo = Topology::new();
    let wedge = load("kumiko_corner_wedge.bin", &mut topo);
    let strut = load("kumiko_corner_strut.bin", &mut topo);

    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        wedge,
        strut,
    )
    .expect("corner wedge cut should not fail outright");

    let mix = surface_mix(&topo, result);
    let faces: usize = mix.values().sum();

    // The tell, and the reason this fixture exists: both operands carry
    // cylinders, so an analytic result must keep some. All-planar means the
    // mesh fallback ran.
    assert!(
        mix.get("cylinder").copied().unwrap_or(0) > 0,
        "cut must stay analytic and keep cylindrical corner faces, got {faces} faces {mix:?}"
    );

    assert_eq!(
        edge_uses(&topo, result),
        (0, 0),
        "cut result must be watertight and manifold"
    );

    // The strut genuinely overlaps the wedge (point-oracle verified), so the
    // cut must REMOVE material. Pin the measured overlap loosely: 285.861
    // down to 247.460 on the fixed splitter chain.
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        vol > 240.0 && vol < 280.0,
        "cut must remove the overlap: got {vol:.3} from the 285.861 wedge"
    );
}
