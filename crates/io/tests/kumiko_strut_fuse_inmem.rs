//! Captured-operand READY-REPRO for the kumiko corner-window strut fuses
//! (re-captured 2026-08-13 from the tool's kumikoWrapSpike probe after the
//! original fixtures vanished with the parked branch).
//!
//! The corner-wrap cutter is `annular wedge − fuse_all(struts)`. The vertical
//! and horizontal struts are coaxial annular wedges (2 cylinders + 4 planes
//! each) and their pairwise fuse is exact. The two diagonal struts are
//! helix-swept rectangles serialized as ~42-face segmented prisms (all
//! planar, slight dihedral steps), and EVERY fuse involving one degrades to
//! the mesh fallback: the raw GFA result carries 15+ free edges, including
//! ellipse sections where the strut's slanted segment planes cross the
//! wedge's cylinders, scattered across many faces. Via `fuse_all`'s
//! fallback bail the whole strut fuse then errors, which is what
//! kumikoWrapSpike trips over. The prior dig recorded four roots (band
//! rescue, graze scaling, chord-represented NURBS boundaries, reverse-twin
//! misread) before its branch was lost; treat this as a fresh multi-root
//! investigation with these fixtures as the stable repro.
//!
//! The ignored test is the acceptance bar for a fix. It stays ignored until
//! a fix clears the full face-splitter foil set (d4 gridfinity, honeycomb
//! pcut1/pcut3, divider-lip, groove-mouth, junction-disc, cylinder-slot,
//! a1corner).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
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

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n == 1).count()
}

#[test]
fn kumiko_strut_operands_are_well_formed() {
    let mut topo = Topology::new();
    for name in [
        "kumiko_strut_vertical.bin",
        "kumiko_strut_horizontal.bin",
        "kumiko_strut_diag_up.bin",
        "kumiko_strut_diag_down.bin",
    ] {
        let sid = load(name, &mut topo);
        assert_eq!(free_edge_count(&topo, sid), 0, "{name} has free edges");
    }
}

/// The vertical x horizontal wedge fuse is already exact — pin it so a
/// regression in the coaxial-wedge path is caught immediately.
#[test]
fn kumiko_wedge_strut_pair_fuse_is_exact() {
    let mut topo = Topology::new();
    let v = load("kumiko_strut_vertical.bin", &mut topo);
    let h = load("kumiko_strut_horizontal.bin", &mut topo);

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Fuse,
        v,
        h,
    )
    .expect("wedge strut fuse should succeed");
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "wedge strut fuse degraded to the mesh fallback"
    );
    assert_eq!(free_edge_count(&topo, result), 0);
    let cylinders = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() == "cylinder")
        .count();
    assert_eq!(cylinders, 6, "the crossing wedges should keep 6 cylinders");
}

/// READY-REPRO: fusing a wedge strut with the segmented diagonal strut must
/// stay exact. Today the raw GFA result comes back open and the op pays the
/// mesh fallback (the kumiko corner-window frontier).
#[test]
#[ignore = "corner-window frontier: faceted-strut fuses mesh-fall-back (see module doc)"]
fn kumiko_diagonal_strut_fuse_is_exact() {
    let mut topo = Topology::new();
    let v = load("kumiko_strut_vertical.bin", &mut topo);
    let d = load("kumiko_strut_diag_up.bin", &mut topo);

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Fuse,
        v,
        d,
    )
    .expect("diagonal strut fuse should succeed");
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "diagonal strut fuse degraded to the mesh fallback"
    );
    assert_eq!(free_edge_count(&topo, result), 0);
}
