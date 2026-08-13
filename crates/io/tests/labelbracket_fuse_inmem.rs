//! Captured-operand READY-REPRO for the 2x2 label-bracket fuse (#1510).
//!
//! `fuse(bin body, label bracket)` is the single boolean that owns the tool's
//! label-bracket row: 95ms of the generation's 124ms of boolean time, because
//! raw GFA returns an OPEN shell and `operations::boolean` pays for the mesh
//! fallback (121 all-planar faces where the operands carry 8 cylinders).
//!
//! Both operands are well-formed (free=0, over=0, 0 unmatched directed
//! half-edges each), so this is not a fallback-poisoned capture.
//!
//! Geometry: the bin's cavity wall sits at +-40.550 with r=2.55 corners centred
//! on (+-38, +-38); the bracket is a box x +-40.550, y 28.550..40.550, whose top
//! at z=16.010 pokes 0.01mm above the bin's top at z=16.000 (the consuming
//! tool's deliberate coplanar overlap). The bracket's corners are SQUARE while
//! the cavity's are ROUNDED, so the cavity corner cylinder is exactly TANGENT
//! to the plane y=40.550 at x=+-38.
//!
//! The defect: the splitter builds the right complement sub-face on the
//! bracket's back wall (an inverted-U: the outer rectangle x +-40.550,
//! z 14.800..16.010 minus a notch at x +-38, z 14.800..16.000), but that
//! sub-face STRADDLES the bin's boundary. Its side columns (|x| in 38..40.550,
//! below z=16.000) are inside the bin's material, because the square corner
//! intrudes into the rounded cavity corner; its top strip (z 16.000..16.010) is
//! outside. One classification verdict cannot be right for both, the drop
//! verdict wins, and the whole back wall vanishes — leaving a 10-edge hole.
//!
//! The split that would separate them runs along z=16.000 for |x| in
//! [38, 40.550]. Phase FF DOES compute that section across the full width —
//! `BK_FF_DUMP` reports the Id(18)/Id(23) section spanning x -40.550..40.550 —
//! but only its |x| < 38 portion ends up bounding the sub-face. So the defect
//! is not a missed section; it is that the section's outer pieces are never
//! applied as split edges here. x=+-38 is where the section gets broken, being
//! the tangency point, which points at pave-block attachment rather than at
//! section computation.
//!
//! A second bracket variant (`labelbracket_fingers_bracket.bin`, captured
//! 2026-08-13 against the same bin) ends its eight support ramps in vertical
//! finger strips lying exactly IN the cavity-wall plane y=40.550. Its side
//! wall cuts on the bin's top ring then ride collinearly on the ring's inner
//! boundary over part of their span, and the hole weave's whole-window
//! midpoint probe sat exactly ON the hole polygon — an on-boundary ray-cast
//! whose verdict flipped between the two mirrored walls (+x dropped, -x
//! kept). The dropped wall cut left its lune partner a pendant, the +x
//! corner lune was never split from the ring, and the shell came back open
//! (3 free edges), forcing the mesh fallback — and, through fuse_cluster's
//! fallback bail, a doubled fuse in consumers that try fuseAll first. Fixed
//! by splitting boundary-riding windows at the collinear-overlap ends in
//! `integrate_holes_plane` and dropping the riding piece explicitly.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp as AlgoOp;
use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
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

/// Edges used by exactly one face — a free boundary, i.e. an open shell.
fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(wire) = topo.wire(wid) else { continue };
            for oe in wire.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n == 1).count()
}

/// Both operands are clean going in — guards against a poisoned re-capture.
#[test]
fn labelbracket_fuse_operands_are_well_formed() {
    let mut topo = Topology::new();
    let bin = load("labelbracket_fuse_bin.bin", &mut topo);
    let bracket = load("labelbracket_fuse_bracket.bin", &mut topo);

    assert_eq!(free_edge_count(&topo, bin), 0, "bin operand has free edges");
    assert_eq!(
        free_edge_count(&topo, bracket),
        0,
        "bracket operand has free edges"
    );

    let curved = solid_faces(&topo, bin)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count();
    assert_eq!(curved, 8, "bin operand should carry its 8 cylinders");
}

#[test]
fn labelbracket_fuse_closes_the_back_wall() {
    let mut topo = Topology::new();
    let bin = load("labelbracket_fuse_bin.bin", &mut topo);
    let bracket = load("labelbracket_fuse_bracket.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, AlgoOp::Fuse, bin, bracket)
        .expect("GFA fuse should succeed");

    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "fuse left an open shell: the bracket's y=40.550 back wall was dropped"
    );

    // The back wall must survive above the bracket shelf, up to the 0.01mm
    // overlap band — the face whose absence opens the shell.
    let has_back_wall = solid_faces(&topo, result).unwrap().iter().any(|&fid| {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            return false;
        }
        let mut on_plane = true;
        let mut reaches_top = false;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(wire) = topo.wire(wid) else { continue };
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                for vid in [edge.start(), edge.end()] {
                    let p = topo.vertex(vid).unwrap().point();
                    if (p.y() - 40.550).abs() > 1e-6 {
                        on_plane = false;
                    }
                    if (p.z() - 16.010).abs() < 1e-6 {
                        reaches_top = true;
                    }
                }
            }
        }
        on_plane && reaches_top
    });
    assert!(
        has_back_wall,
        "no face on the y=40.550 plane reaching z=16.010 in the fuse result"
    );
}

#[test]
fn labelbracket_fingers_fuse_is_exact_and_closed() {
    let mut topo = Topology::new();
    let bin = load("labelbracket_fuse_bin.bin", &mut topo);
    let bracket = load("labelbracket_fingers_bracket.bin", &mut topo);

    assert_eq!(free_edge_count(&topo, bin), 0, "bin operand has free edges");
    assert_eq!(
        free_edge_count(&topo, bracket),
        0,
        "bracket operand has free edges"
    );

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Fuse,
        bin,
        bracket,
    )
    .expect("fuse should succeed");
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "fuse degraded to the mesh fallback"
    );

    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "fuse left an open shell: a top-ring corner lune was not split out"
    );

    let curved = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count();
    assert_eq!(curved, 8, "result should keep the bin's 8 cylinders");

    // Analytic union: bin 14102.76 (outer r=3.75 rounded prism minus the
    // r=2.55 cavity) + bracket 1799.652 (all-planar, deflection-exact)
    // minus the two corner lune columns the plate shares with the wall
    // (2 x 1.395 x 1.2 = 3.35) = 15899.06; at 0.01 deflection the cylinder
    // chords under-count ~0.2. Fallback-vs-exact is decided by the counter
    // and free-edge asserts above, not by this band.
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (15898.0..=15900.0).contains(&vol),
        "unexpected fused volume {vol}"
    );
}
