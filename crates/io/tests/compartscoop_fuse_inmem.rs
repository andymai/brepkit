//! Captured-operand READY-REPRO for the compartments+scoop fuse (#1517).
//!
//! This one boolean is the root of the tool's largest parity cluster: ~45
//! "export mesh is not watertight" failures across unrelated features. The
//! chain was `fuse(scooped bin body, compartment divider assembly)` ->  GFA
//! rejected -> `operations::boolean` pays for the mesh fallback -> the fallback
//! emits an OPEN shell (650 all-planar faces, 4 free edges) -> every downstream
//! op and the export inherit it. The fuse now returns 178 faces,
//! 12 cone / 24 cylinder / 142 plane, 0 free edges.
//!
//! Both operands are well-formed going in, which is what makes this a genuine
//! engine defect rather than a poisoned capture:
//!
//! ```text
//! A: F=114 mix=[(cone,12),(cylinder,24),(plane,78)]  free=0 over=0 vol=18986.437
//! B: F=110 mix=[(plane,110)]                          free=0 over=0 vol=9471.817
//! ```
//!
//! # The root
//!
//! The body's front corner is a cylinder (z=1.200..**13.300**) topped by a
//! cone, and the divider's chorded scoop wall has a facet spanning
//! z=12.879..13.912 — so the facet straddles that junction.
//!
//! A thin planar tread meeting a corner cylinder takes a dedicated path,
//! `trim_ellipse_to_boundary_crossings`, because the in-both arc is a
//! sub-millimetre sliver the generic sampled filters drop. That path split the
//! section at the tread's boundary lines and at the analytic face's SEAM
//! lines, but not at its RIM arcs — it only crossed `EdgeCurve::Line` boundary
//! edges. So nothing split the section at z=13.300 where the cylinder ends,
//! and the single over-long arc reaching to z=13.912 kept its midpoint
//! (z~13.40) inside the extent's 0.121 boundary margin, so the whole thing
//! survived. The tread then bounded the region along one curve and the
//! cylinder along another, 0.687mm apart on the tread's top edge, and the
//! shell came back open.
//!
//! Crossing the rim arcs too splits the section at z=13.300, and the existing
//! midpoint test drops the piece above it with no other change.
//!
//! Bisection from the tool: compartments alone and scoop alone both export
//! watertight; only the combination fails. The sibling cluster (0.4mm walls,
//! exactly 3 free edges) has the same thin-feature shape.

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

/// Guards the capture itself: if a future re-capture picks up a fallback-poisoned
/// operand, this fails before anyone debugs the engine over it.
#[test]
fn compartscoop_operands_are_well_formed() {
    let mut topo = Topology::new();
    let body = load("compartscoop_fuse_body.bin", &mut topo);
    let dividers = load("compartscoop_fuse_dividers.bin", &mut topo);

    assert_eq!(free_edge_count(&topo, body), 0, "body operand is open");
    assert_eq!(
        free_edge_count(&topo, dividers),
        0,
        "divider operand is open"
    );

    let curved = solid_faces(&topo, body)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert_eq!(
        curved, 36,
        "body should carry its 36 curved faces (scoop + corners)"
    );
}

#[test]
fn compartscoop_fuse_is_closed() {
    let mut topo = Topology::new();
    let body = load("compartscoop_fuse_body.bin", &mut topo);
    let dividers = load("compartscoop_fuse_dividers.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, AlgoOp::Fuse, body, dividers)
        .expect("GFA fuse should succeed");

    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "fuse left an open shell; the ops layer then mesh-falls-back and the export inherits the hole"
    );

    let curved = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved > 0,
        "result went all-planar: the scoop's curved faces were lost"
    );
}
