//! Grouped-scoop pinch-shim cut (GH #1445 residual, captured 2026-08-08).
//!
//! The scoop tool's variable-radius corner fillet pinches to zero at the
//! pocket floor: each horn-torus corner patch touches the floor plane
//! tangentially along an arc, and the blend closes that tangent contact
//! with a tiny corner-triangle face coincident with (contained in) the
//! unsplit floor. The same-domain within-rank dedup used to classify those
//! shims as #696 containment residue and drop them, orphaning the tangent
//! arc (torus side) and the stub edges (wall side) into 12 free edges; the
//! validation gate then rejected the GFA result and the cut mesh-fell-back
//! tool-side. The residue gate now keeps a containment-matched duplicate
//! whose edges serve sub-faces outside its SD group (see
//! `detect_same_domain_with_shells`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
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
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                *counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

#[test]
fn operands_are_clean() {
    let mut topo = Topology::new();
    let body = load("gscoop_pinch_body.bin", &mut topo);
    let tool = load("gscoop_pinch_tool.bin", &mut topo);
    assert_eq!(
        free_edge_count(&topo, body),
        0,
        "body operand must be clean"
    );
    assert_eq!(
        free_edge_count(&topo, tool),
        0,
        "tool operand must be clean"
    );
}

#[test]
fn scoop_cut_keeps_pinch_shims_watertight_and_analytic() {
    // The pin requires BOTH watertightness and analytic survival: the mesh
    // fallback also produces a watertight solid, but an all-planar one, so
    // the torus count is what distinguishes a real fix from the fallback.
    let mut topo = Topology::new();
    let body = load("gscoop_pinch_body.bin", &mut topo);
    let tool = load("gscoop_pinch_tool.bin", &mut topo);
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        body,
        tool,
    )
    .unwrap();
    let free = free_edge_count(&topo, result);
    let tori = brepkit_topology::explorer::solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&fid| {
            matches!(
                topo.face(fid).unwrap().surface(),
                brepkit_topology::face::FaceSurface::Torus(_)
            )
        })
        .count();
    assert_eq!(
        free, 0,
        "scoop cut must be watertight, got {free} free edges"
    );
    assert_eq!(
        tori, 8,
        "all 8 horn-torus corner patches must survive analytically, got {tori}"
    );
}
