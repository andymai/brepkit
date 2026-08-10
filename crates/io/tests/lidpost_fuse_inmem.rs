//! Captured-operand READY-REPRO for the lid retention-magnet post fuse (#1517).
//!
//! This one boolean is the root of the tool's runaway cluster: the crash the
//! issue records as `Hash table capacity overflow` plus the 14 scenario
//! timeouts. It reproduces in ~5ms here against ~12 minutes in the tool.
//!
//! The chain, measured by tracing every kernel call in the
//! `lidRetentionMagnets` scenario and serialising each boolean's operands:
//!
//! ```text
//! call ~180  fuse(lid, post)  A: F=44  mix=[(cone,4),(cylinder,16),(plane,24)]
//!                             B: F=3   mix=[(cylinder,1),(plane,2)]
//!            raw GFA          F=47 mix=[(cone,4),(cylinder,17),(plane,26)] free=6
//!            -> rejected (euler=-1, 13 boundary edges) -> mesh fallback
//!            -> F=305 mix=[(plane,305)]
//! call ~190+ each further post fuses into the all-planar blob
//!            -> 8852 faces by call 278
//! call  248  fuse(blob, post) never returns
//! ```
//!
//! So the runaway is not a loop with a missing exit: it is one boolean losing
//! its analytic surfaces, and then two dozen more booleans paying compound
//! interest on the debris. Fixing the fuse below removes the blob, and with it
//! both the crash and the timeouts.
//!
//! # The geometry
//!
//! The post is a cylinder of radius 4 centred at (-118.500, 76.500) spanning
//! z=-2.800..-0.700. It sits at a corner of the lid and pokes out through
//! BOTH perpendicular walls, x=-122.000 and y=80.000. The lid carries a
//! horizontal ledge at z=-0.800, so the post's cross-section there is a full
//! circle of radius 4 that the two walls cut into arcs:
//!
//! ```text
//! circle x=-122.000 at y=78.436 and y=74.564
//! circle y= 80.000  at x=-116.564 and x=-120.436
//! ```
//!
//! The six free edges are exactly those arcs plus the post's z=-0.700 rim.
//! Each is used by ONE face where it needs two: the ledge plane keeps three
//! arcs, the post cylinder keeps two, and they are DIFFERENT arcs, so the two
//! surfaces are split against each other inconsistently rather than one of
//! them failing to split at all.
//!
//! # The control pins the discriminant
//!
//! `lidpost_fuse_post_control.bin` is an earlier post from the same scenario,
//! fused into the SAME lid, protruding through ONE wall instead of two. It
//! fuses cleanly: 48 faces, 0 free edges, volume up by the protruding cap. So
//! the defect is the corner specifically, not posts, not protrusion, and not
//! this lid. That is the one variable to hold onto when working the fix.
//!
//! Note `plane_internal_line_loops` logs three "not strictly interior"
//! rejections here at distances 2.8e-14, 8.9e-16 and exactly 0. Those look
//! like a tolerance bug but are not the defect: the sections genuinely touch
//! the boundary, so declining to treat them as an internal loop is correct.

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

fn edge_use_counts(topo: &Topology, solid: SolidId) -> HashMap<EdgeId, usize> {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            // Resolve rather than skip: this count is the evidence that the
            // operands are well-formed, and a wire that fails to resolve would
            // otherwise drop its edges and report a clean zero on broken
            // topology.
            let wire = topo
                .wire(wid)
                .expect("unresolvable wire in captured operand");
            for oe in wire.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    edge_use_counts(topo, solid)
        .values()
        .filter(|&&n| n == 1)
        .count()
}

fn surface_mix(topo: &Topology, solid: SolidId) -> HashMap<&'static str, usize> {
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        *mix.entry(topo.face(fid).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    mix
}

/// Both operands are well-formed, so the defect is in the fuse rather than in
/// a poisoned capture.
#[test]
fn lidpost_operands_are_well_formed() {
    let mut topo = Topology::new();
    let lid = load("lidpost_fuse_lid.bin", &mut topo);
    let post = load("lidpost_fuse_post.bin", &mut topo);

    assert_eq!(free_edge_count(&topo, lid), 0, "lid has free edges");
    assert_eq!(free_edge_count(&topo, post), 0, "post has free edges");

    let lid_mix = surface_mix(&topo, lid);
    assert_eq!(lid_mix.get("cone").copied(), Some(4));
    assert_eq!(lid_mix.get("cylinder").copied(), Some(16));
    assert_eq!(lid_mix.get("plane").copied(), Some(24));

    let post_mix = surface_mix(&topo, post);
    assert_eq!(post_mix.get("cylinder").copied(), Some(1));
    assert_eq!(post_mix.get("plane").copied(), Some(2));
}

/// The defect: GFA keeps the analytic surfaces but leaves the post's arcs
/// single-sided, so the gate rejects the result and the mesh fallback replaces
/// 44 exact faces with hundreds of flat ones.
///
/// Un-ignore when the fuse closes. The assertions below are the acceptance
/// bar, not a description of current behaviour.
#[test]
#[ignore = "#1517: post straddling a lid corner leaves 6 free edges, forcing the mesh fallback"]
fn lidpost_fuse_is_closed_and_analytic() {
    let mut topo = Topology::new();
    let lid = load("lidpost_fuse_lid.bin", &mut topo);
    let post = load("lidpost_fuse_post.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, AlgoOp::Fuse, lid, post).unwrap();

    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "fuse left free edges, which is what forces the mesh fallback"
    );
    let mix = surface_mix(&topo, result);
    assert!(
        mix.get("cylinder").copied().unwrap_or(0) >= 16 && mix.get("cone").copied() == Some(4),
        "fuse lost analytic surfaces: {mix:?}"
    );
}

/// Control: the same lid, an earlier post protruding through ONE wall.
///
/// This is the one-variable comparison for the case above. It fuses cleanly,
/// which rules out posts, protrusion and this lid as the cause and leaves the
/// two-wall corner as the discriminant. If this ever starts failing, the fix
/// under test has broken the ordinary case, not repaired the corner one.
#[test]
fn lidpost_single_wall_control_fuses_closed_and_analytic() {
    let mut topo = Topology::new();
    let lid = load("lidpost_fuse_lid.bin", &mut topo);
    let post = load("lidpost_fuse_post_control.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, AlgoOp::Fuse, lid, post).unwrap();

    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "control fuse left free edges"
    );
    let mix = surface_mix(&topo, result);
    assert_eq!(mix.get("cone").copied(), Some(4));
    assert_eq!(mix.get("cylinder").copied(), Some(18));
    assert_eq!(mix.get("plane").copied(), Some(26));
}
