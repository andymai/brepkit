//! Grouped-scoop cutout from the gridfinity tool (`binGenerator.export.groupedScoop`,
//! "circle + rectangle group with scoop"), captured on brepkit-wasm 3.4.0.
//!
//! The scoop tool is a variable fillet over nine edges with radii 2, 0.6, 0.263
//! and 0.393 on a 12-face body. The corrected blend engine closes the
//! four-stripe mixed-radius junction with an apex fan whose boundary cycle
//! passes one vertex twice (the r=0.263 stripe's cross boundary is a
//! sub-segment of the r=2 stripe's), so the fan mints two radial lines
//! between the same endpoints. The tool is manifold by edge id, but the bin
//! cut's duplicate-edge merge collapses the twins into one four-owner edge,
//! the exact result is rejected, and the mesh fallback exports with open
//! edges. On 3.3.9 the same junction closed with 28 faces and the cut stayed
//! exact.
//!
//! Data: `groupedscoop_fillet_base.bin` + `groupedscoop_fillet_spec.json` (the
//! `filletVariable` call), `groupedscoop_bin_body.bin` (the cut's base).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn edge_ends(topo: &Topology, solid: SolidId) -> Vec<(EdgeId, Point3, Point3)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let eid = oe.edge();
                if !seen.insert(eid) {
                    continue;
                }
                let e = topo.edge(eid).unwrap();
                out.push((
                    eid,
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                ));
            }
        }
    }
    out
}

fn load_case(topo: &mut Topology) -> (SolidId, Vec<(EdgeId, FilletRadiusLaw)>) {
    let base = deserialize_solid(
        &std::fs::read(fixture("groupedscoop_fillet_base.bin")).unwrap(),
        topo,
    )
    .unwrap();
    let specs: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(fixture("groupedscoop_fillet_spec.json")).unwrap(),
    )
    .unwrap();
    let ends = edge_ends(topo, base);
    let mut laws = Vec::new();
    for spec in &specs {
        let v = spec["verts"].as_array().unwrap();
        let f = |i: usize| v[i].as_f64().unwrap();
        let (pa, pb) = (Point3::new(f(0), f(1), f(2)), Point3::new(f(3), f(4), f(5)));
        let r = spec["startRadius"].as_f64().unwrap();
        let (best, dist) = ends
            .iter()
            .map(|(eid, a, b)| {
                let d = ((*a - pa).length() + (*b - pb).length())
                    .min((*a - pb).length() + (*b - pa).length());
                (*eid, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(dist < 1e-6, "spec edge did not match the arena: {dist}");
        laws.push((best, FilletRadiusLaw::Constant(r)));
    }
    (base, laws)
}

/// Pairs of distinct edges whose endpoints coincide (in either order).
fn twin_edge_pairs(topo: &Topology, solid: SolidId) -> Vec<(EdgeId, EdgeId)> {
    let ends = edge_ends(topo, solid);
    let mut twins = Vec::new();
    for (i, (ea, a0, a1)) in ends.iter().enumerate() {
        for (eb, b0, b1) in &ends[i + 1..] {
            let fwd = (*a0 - *b0).length().max((*a1 - *b1).length());
            let rev = (*a0 - *b1).length().max((*a1 - *b0).length());
            if fwd.min(rev) < 1e-9 {
                twins.push((*ea, *eb));
            }
        }
    }
    twins
}

fn edge_use_counts(topo: &Topology, solid: SolidId) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
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
    counts
}

#[test]
fn groupedscoop_fixture_is_faithful() {
    let mut topo = Topology::new();
    let (base, laws) = load_case(&mut topo);
    let faces = brepkit_topology::explorer::solid_faces(&topo, base).unwrap();
    assert_eq!(faces.len(), 12);
    assert_eq!(laws.len(), 9);
    let body = deserialize_solid(
        &std::fs::read(fixture("groupedscoop_bin_body.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    assert_eq!(
        brepkit_topology::explorer::solid_faces(&topo, body)
            .unwrap()
            .len(),
        10
    );
}

#[test]
#[ignore = "ready repro: the junction fan mints twin radials at the four-stripe mixed-radius corner"]
fn groupedscoop_fillet_has_no_twin_edges() {
    let mut topo = Topology::new();
    let (base, laws) = load_case(&mut topo);
    let tool = fillet_variable(&mut topo, base, &laws).unwrap();
    let uses = edge_use_counts(&topo, tool);
    assert_eq!(
        uses.values().filter(|&&c| c != 2).count(),
        0,
        "tool must be closed"
    );
    let twins = twin_edge_pairs(&topo, tool);
    assert!(twins.is_empty(), "twin edges in the scoop tool: {twins:?}");
}

#[test]
#[ignore = "ready repro: the bin cut by the scoop tool must stay exact (3.3.9 did)"]
fn groupedscoop_cut_stays_exact() {
    let mut topo = Topology::new();
    let (base, laws) = load_case(&mut topo);
    let tool = fillet_variable(&mut topo, base, &laws).unwrap();
    let body = deserialize_solid(
        &std::fs::read(fixture("groupedscoop_bin_body.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "the cut took the mesh fallback"
    );
    let uses = edge_use_counts(&topo, result);
    assert_eq!(
        uses.values().filter(|&&c| c != 2).count(),
        0,
        "cut result must be manifold"
    );
}
