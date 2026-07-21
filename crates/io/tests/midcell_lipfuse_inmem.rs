//! Captured-operand regression for the gridfinity "2×1 lite + mid-cell
//! dividers (cols=3)" lip fuse — the first fallback of that scenario's
//! export chain.
//!
//! Operands captured from the live gridfinity tool via the `serializeSolid`
//! wasm binding (byte-exact in-memory arena):
//!   midcell_lipfuse_body = bin box with divider cavities (boxBuilder output)
//!   midcell_lipfuse_top  = stacking lip, translated to its overlap position
//!
//! The lip's base face is a DOWN-facing annulus (outline + throat hole) that
//! the fuse splits with woven divider sections. The hole-promotion pass
//! sampled the loops through the pcurves, which chord Circle2D corner arcs —
//! the throat's corner points poked past the outline's chord-approximated
//! corners, read as "not nested", and the throat was promoted to its own
//! region instead of staying the annulus's hole. The resulting spurious
//! full disc capped the bin interior, the z=13.3 ring edge went three-way
//! shared, the shell partition split, and the #1146 hole-shell fail-safe
//! sent the fuse to a mesh fallback whose open output (free=121) poisoned
//! every downstream boolean through to the STL export (bd=115).
//!
//! With arc-true via-frame sampling in the promotion pass the fuse is
//! analytic and watertight.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
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

fn edge_use_counts(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let faces = solid_faces(topo, solid).unwrap();
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    (faces.len(), free, over)
}

#[test]
fn midcell_lip_fuse_is_analytic_and_watertight() {
    let mut topo = Topology::new();
    let body = load("midcell_lipfuse_body.bin", &mut topo);
    let top = load("midcell_lipfuse_top.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, body, top).unwrap();
    let (n_faces, free, over) = edge_use_counts(&topo, result);
    assert_eq!(free, 0, "lip fuse must be closed, got {free} free edges");
    assert_eq!(over, 0, "lip fuse must be manifold, got {over} over-shared");

    // The fallback tell: mesh fallback re-emits everything as planes; the
    // analytic path keeps the lip's cylinders and cones.
    let faces = solid_faces(&topo, result).unwrap();
    let curved = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved >= 30 && n_faces < 300,
        "analytic result expected, got {curved} curved of {n_faces} faces"
    );

    // Volume pins the geometry, not just the path: a wrong-but-watertight
    // result (throat filled by the spurious disc, or a dropped region) moves
    // the volume far beyond this band. Reference: analytic fuse of the
    // captured operands (11983.75), watertight at export deflection.
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - 11983.8).abs() < 15.0,
        "fused volume out of band: got {vol}"
    );
}
