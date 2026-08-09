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
//! [38, 40.550], between the bin's top annulus and the bracket's back wall. It
//! is present for |x| < 38 and absent outside it, and |x|=38 is precisely the
//! tangency point — so the missing section is the one beginning at the graze.
//!
//! Ready-repro: the assertion below is what a fix must satisfy. It is ignored
//! because the fix belongs at the graze/section level (phase FF or the section
//! clip) and every change there has to clear the full face-splitter foil set.

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
#[ignore = "#1510: tangential graze at x=+-38 leaves the bracket's back wall straddling the bin boundary; it is dropped and the shell opens"]
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
