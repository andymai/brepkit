//! Captured-operand READY-REPRO for the kumiko corner-window strut fuses
//! (re-captured 2026-08-13 from the tool's kumikoWrapSpike probe after the
//! original fixtures vanished with the parked branch).
//!
//! The corner-wrap cutter is `annular wedge − fuse_all(struts)`. The vertical
//! and horizontal struts are coaxial annular wedges (2 cylinders + 4 planes
//! each) and their pairwise fuse is exact. The diagonal strut is a
//! helix-swept rectangle; every fuse involving one degrades to the mesh
//! fallback, which `fuse_all`'s bail turns into the hard error
//! kumikoWrapSpike trips over.
//!
//! Two capture generations of the diagonal exist:
//! - `kumiko_strut_diag_up.bin` / `_down.bin`: the pre-twist-guard sweep,
//!   whose side quads were emitted as fitted PLANES with corners up to
//!   ~6e-5 off-plane. Booleans against these dirty operands mis-clip every
//!   seam (kept as tolerance-robustness history; the module keeps only the
//!   operand well-formedness pin for them).
//! - `kumiko_strut_diag_up_ruled.bin`: the honest sweep (twisted quads emit
//!   exact bilinear patches, 40 NURBS + 2 plane faces). With marched chain
//!   ends refined onto the exact patch boundary (the SSI marcher's 0.1%
//!   domain-clamp margin used to leave ~2e-3 junction gaps) and the no-seam
//!   splitter shortcut gated to periodic/sphere faces (a non-periodic
//!   bilinear quad now splits via the generic arrangement), the fuse
//!   assembles exact-typed (64 faces keeping all cylinders and NURBS) but
//!   still leaks ~26 free edges: neighbors disagree about boundary-edge
//!   splits, so seam pieces bound only one side (the free edges have no
//!   coincident twin — see TWIN=1 in replay_pair). The ignored test below
//!   remains the acceptance bar.
//!
//! The prior dig recorded four roots (band rescue, graze scaling,
//! chord-represented NURBS boundaries, reverse-twin misread) before its
//! branch was lost; treat this as a fresh investigation.
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
        "kumiko_strut_diag_up_ruled.bin",
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

/// The wedge's outer cylinder must PARTITION at the strut's marched section
/// chains: below-chain and above-chain pieces kept, the in-strut band cut
/// out (4 cylinder faces in the raw GFA result, 2 per wedge cylinder). The
/// chain ends and the boundary split vertices share one 3D junction but
/// carry UVs from different paths (~1e-6 apart, above the wire graph's
/// exact-tol weld); without the splitter's 3D-identity UV co-registration
/// the pendant pruner eats both chains and the face survives unsplit with
/// its in-strut corner span still in a kept wire.
#[test]
fn kumiko_wedge_outer_cylinder_partitions_at_strut_chains() {
    let mut topo = Topology::new();
    let v = load("kumiko_strut_vertical.bin", &mut topo);
    let d = load("kumiko_strut_diag_up_ruled.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, v, d)
        .expect("raw GFA fuse should assemble");
    let cylinders = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() == "cylinder")
        .count();
    assert!(
        cylinders >= 4,
        "both wedge cylinders must split at the strut chains (got {cylinders} cylinder faces)"
    );
}

/// READY-REPRO: fusing a wedge strut with the honest (ruled-patch) diagonal
/// strut must stay exact. Today the wedge's kept pieces fail to weld to the
/// strut's marched-NURBS section seams and detach as a 5-face open growth
/// shell, so the op pays the mesh fallback (the kumiko corner-window
/// frontier).
#[test]
fn kumiko_diagonal_strut_fuse_is_exact() {
    let mut topo = Topology::new();
    let v = load("kumiko_strut_vertical.bin", &mut topo);
    let d = load("kumiko_strut_diag_up_ruled.bin", &mut topo);

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
