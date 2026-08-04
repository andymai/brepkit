//! Fusing a socket assembly onto a bin body whose cavity has SHARP corners
//! must stay analytic. It used to abort with "open hole shell with 9 faces":
//! `shell` emitted NO inner face for a corner cylinder the thickness
//! swallowed, and the spec assembler closed the corner-wide gap by threading
//! the top ring's inner wire down the cavity verticals and across the floor
//! chamfer diagonals. That wire is edge-paired (free=0, so every health
//! check passed) but geometrically degenerate, so the next fuse's hole-shell
//! grouping correctly refused it. The fix emits the sharp-corner chamfer
//! strip (the collapsed cylinder's wire vertices mapped through the miter
//! positions), and the operands here are re-captured from the fixed shell.
//! Measured end-to-end: exported boundary edges 23/19/17 to 0/0/0 (1x1 wall
//! 3.8/3.9/4.0) and 20/20 to 0/0 (2x2 wall 4/6).
//!
//! Found via the gridfinity floor-pattern `oversized elements` scenario, which
//! is misfiled: its floor pattern is a no-op by contract and the plain bin
//! fails identically. The real trigger is `wallThickness` crossing the bin's
//! corner radius (`BOX_CORNER_RADIUS` = 3.75). Above it the cavity's corner
//! radius `BOX_CORNER_RADIUS - wallThickness` reaches zero and the generator
//! (correctly) emits a sharp-cornered cavity instead of a rounded one, so the
//! body loses its four cavity corner cylinders:
//!
//! | wallThickness | body faces | fuse |
//! |---|---|---|
//! | 3.7 | F=19, 8 cylinders (4 outer + 4 cavity) | succeeds, F=63, free=0 |
//! | 3.8 | F=15, 4 cylinders (outer only) | ABORTS in 13 ms |
//!
//! Exported STL boundary edges go 0 -> 149 at exactly that step, because the
//! abort drops to `mesh_boolean_fallback`, whose output is itself open and is
//! consumed anyway (the long-open fallback-consumption defect). The later
//! export fuses then inherit the damage: replaying the captured chain, op2/3/4
//! are clean socket-to-socket fuses and op5 fails only because its operand is
//! already the poisoned all-planar fallback output (F=310, free=32).
//!
//! Both operands here are CLEAN (free=0 over=0, closed, outward), so this is
//! not the replayed-garbage trap.

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

/// (free, over, curved face count, total faces)
fn health(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize, usize, usize) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    let mut curved = 0;
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            curved += 1;
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
        curved,
        faces.len(),
    )
}

/// Guards the fixture itself: an unvalidated operand has cost this campaign
/// several passes, and a magnitude-only volume check cannot see an inverted
/// shell, so assert the SIGNED volume.
#[test]
fn thickwall_operands_are_clean_and_outward() {
    let mut topo = Topology::new();
    for name in [
        "thickwall_sharp_cavity_body.bin",
        "thickwall_socket_assembly.bin",
    ] {
        let sid = load(name, &mut topo);
        let (free, over, _curved, faces) = health(&topo, sid);
        assert_eq!(free, 0, "{name}: operand must be closed, got {free} free");
        assert_eq!(over, 0, "{name}: operand must be manifold, got {over} over");
        assert!(faces > 0, "{name}: operand has no faces");
        assert!(
            brepkit_operations::measure::oriented_solid_volume(&topo, sid, 0.05).unwrap() > 0.0,
            "{name}: operand must be OUTWARD oriented"
        );
    }
}

/// The body's cavity is sharp-cornered, which is the whole point of the
/// fixture: it carries only the four OUTER corner cylinders, where the 3.7mm
/// control carries eight.
#[test]
fn thickwall_body_cavity_is_sharp_cornered() {
    let mut topo = Topology::new();
    let body = load("thickwall_sharp_cavity_body.bin", &mut topo);
    let (_free, _over, curved, faces) = health(&topo, body);
    assert_eq!(
        faces, 19,
        "expected the 19-face thick-wall body (15 + the 4 cavity chamfer strips)"
    );
    assert_eq!(
        curved, 4,
        "expected only the 4 OUTER corner cylinders; a rounded cavity would add 4 more"
    );
}

#[test]
fn thickwall_sharp_cavity_fuse_is_closed() {
    let mut topo = Topology::new();
    let body = load("thickwall_sharp_cavity_body.bin", &mut topo);
    let sockets = load("thickwall_socket_assembly.bin", &mut topo);

    let body_vol = brepkit_operations::measure::oriented_solid_volume(&topo, body, 0.01).unwrap();

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .expect("analytic fuse should not abort");

    let (free, over, curved, faces) = health(&topo, result);
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");
    assert!(
        curved > 0,
        "all-planar with zero curved faces is the fallback tell: {faces} faces"
    );

    // A union can only add material to the body.
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        vol >= body_vol,
        "a fuse cannot shrink its operand: got {vol} against {body_vol}"
    );
}
