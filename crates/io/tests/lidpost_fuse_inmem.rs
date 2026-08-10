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
//! Read the lid from its face list (`dump_solid`), not from a mental model of
//! walls and a ceiling. It is a plate spanning the full outer profile from
//! z=-0.800 up to z=0.000, with a pocket the size of the x=+-122 / y=+-80
//! octagon milled up from the bottom at z=-5.793 to a ceiling at z=-0.800.
//! So inside that octagon there is air below z=-0.800 and solid plate above
//! it. Volume 44005.4 confirms it; the pocket-filled reading would be ~205000.
//!
//! The post is a cylinder of radius 4 centred at (-118.500, 76.500) spanning
//! z=-2.800..-0.700, hanging in the pocket at a corner and poking out through
//! BOTH pocket walls, x=-122.000 and y=80.000. Its cross-section at the
//! ceiling is a full circle that the two walls cut at four crossings:
//!
//! ```text
//! 61.045  (-116.564, 80.000)    118.955 (-120.436, 80.000)
//! 151.045 (-122.000, 78.436)    208.955 (-122.000, 74.564)
//! ```
//!
//! Two of the four arcs are exposed (in pocket air): the corner sliver
//! 118.955..151.045 and the big sector 208.955..61.045 through the seam at 0.
//! The other two are buried in the wall material.
//!
//! # What the correct fuse looks like, and where this one diverges
//!
//! The control gives it exactly: 48 faces = the lid's 44, plus the post's
//! cylinder split into its two exposed sectors, the ceiling split in two, and
//! the post's bottom cap. The post's top cap and its z=-0.800..-0.700 band are
//! both buried in the plate and are dropped.
//!
//! Every classification here is right, and `BK_SUBFACE_BOX` shows it: the
//! ceiling splits into three with the post's footprint correctly dropped, and
//! the cylinder's two buried wall sectors are correctly dropped. The defect is
//! the SPLIT. The kept cylinder sub-face spans z=-2.800..-0.700, so it is the
//! two exposed sectors AND the full-circumference band that sits inside the
//! plate, as one piece. That piece is not classification-uniform; its interior
//! sample lands at 150 degrees in the exposed sliver, so the whole thing is
//! kept and drags the buried band along. The six free edges are that band's
//! z=-0.700 rim plus the arcs the two surfaces did not share.
//!
//! `BK_SUBFACE_WIRE` names the mechanism. The cylinder's split at z=-0.800
//! uses the two arcs the FF phase DROPPED as sections (`BK_FF_DUMP`: "drop
//! closed-circle arc ... outside face pair", midpoints at 90 and 180 degrees),
//! and does NOT use the three it kept. The kept arcs go to the ceiling plane
//! alone, which is why they come out free. So the open question is why the
//! cylinder is split against the complement of the section set.
//!
//! The control differs only in where the band ends up. Its seam at 0 degrees
//! falls in a BURIED sector (crossings 28.955, 331.045, 241.045, 298.955), so
//! the band-carrying piece samples at 14.5 degrees inside the wall, classifies
//! Inside, and is dropped with the band. Same wrong split, right answer.
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

/// Control: the same lid, the corresponding post at the opposite corner.
///
/// It carries the SAME wrong split (see the module docs) and still fuses
/// cleanly, because there the mis-attached band rides on a sub-face that
/// classifies Inside and is dropped. Its face mix is the correct answer, so it
/// doubles as the specification for the failing case. Its job here is as a
/// guard: if this starts failing, the fix under test has broken the working
/// case rather than repaired the failing one.
#[test]
fn lidpost_control_post_fuses_closed_and_analytic() {
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
