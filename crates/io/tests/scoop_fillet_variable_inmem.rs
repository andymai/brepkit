//! Grouped-scoop fillet mint (GH #1445 root, captured 2026-08-08).
//!
//! The tool's grouped-cutout scoop is a `filletVariable` call on the fused
//! rect+circle tool's 9 bottom-outline edges (radii 2 / 0.6 / 0.39 / 0.26 —
//! the junction and short-edge reductions from the adaptive scoop). The
//! fillet emits 6 NURBS walls WITHOUT stitching them to the adjacent faces:
//! the result carries 44 free edges, and every later boolean of the chain
//! faithfully propagates the open boundary into the exported mesh (the
//! issue's 25-63 boundary-edge measurements). The booleans are exonerated —
//! chain replay shows every fuse/cut input-clean except for this operand.
//!
//! Replay tooling: `crates/io/examples/replay_fillet_variable.rs` (matches
//! spec edges by captured endpoint pairs; tool-side midpoint params are not
//! portable).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    let mut counts: HashMap<usize, usize> = HashMap::new();
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

fn load_named_case(
    topo: &mut Topology,
    input: &str,
    spec: &str,
) -> (SolidId, Vec<(brepkit_topology::edge::EdgeId, f64)>) {
    let solid = deserialize_solid(&std::fs::read(fixture(input)).unwrap(), topo).unwrap();
    let specs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(fixture(spec)).unwrap()).unwrap();

    let mut seen: std::collections::HashSet<brepkit_topology::edge::EdgeId> =
        std::collections::HashSet::new();
    let mut edge_ends: Vec<(brepkit_topology::edge::EdgeId, Point3, Point3)> = Vec::new();
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
                edge_ends.push((
                    eid,
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                ));
            }
        }
    }

    let mut picks = Vec::new();
    for spec in &specs {
        let v = spec["verts"].as_array().unwrap();
        let f = |i: usize| v[i].as_f64().unwrap();
        let (pa, pb) = (Point3::new(f(0), f(1), f(2)), Point3::new(f(3), f(4), f(5)));
        let r = spec["startRadius"].as_f64().unwrap();
        let (best, dist) = edge_ends
            .iter()
            .map(|(eid, a, b)| {
                let d = ((*a - pa).length() + (*b - pb).length())
                    .min((*a - pb).length() + (*b - pa).length());
                (*eid, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(dist < 1e-3, "captured edge failed to match: {dist}");
        picks.push((best, r));
    }
    (solid, picks)
}

fn load_case(topo: &mut Topology) -> (SolidId, Vec<(brepkit_topology::edge::EdgeId, f64)>) {
    load_named_case(topo, "gscoop_fillet_input.bin", "gscoop_fillet_spec.json")
}

fn assert_watertight(input: &str, spec: &str) {
    let mut topo = Topology::new();
    let (solid, picks) = load_named_case(&mut topo, input, spec);
    assert_eq!(free_edge_count(&topo, solid), 0, "operand must be clean");
    let edge_laws: Vec<_> = picks
        .into_iter()
        .map(|(e, r)| (e, FilletRadiusLaw::Constant(r)))
        .collect();
    let result = fillet_variable(&mut topo, solid, &edge_laws).unwrap();
    let free = free_edge_count(&topo, result);
    assert_eq!(
        free, 0,
        "fillet must produce a watertight solid, got {free} free edges"
    );
}

/// Depth-step scoop group (captured case 2): members with different cut
/// depths meet at a step, and the junction needs the arc-hypotenuse corner
/// patch plus the 2-edge arc/chord lens fill.
#[test]
fn depthstep_fillet_variable_is_watertight() {
    assert_watertight(
        "gscoop_fillet_depthstep_input.bin",
        "gscoop_fillet_depthstep_spec.json",
    );
}

/// Aggressive near-max radius scoop (captured case 3): stripes pinch to a
/// point at their ends (zero-length cross edges must not be minted) and the
/// corner triangles must carry the terminal-section arc, not its chord.
#[test]
fn aggressive_radius_fillet_variable_is_watertight() {
    assert_watertight(
        "gscoop_fillet_aggressive_input.bin",
        "gscoop_fillet_aggressive_spec.json",
    );
}

#[test]
fn operand_is_clean() {
    let mut topo = Topology::new();
    let (solid, picks) = load_case(&mut topo);
    assert_eq!(free_edge_count(&topo, solid), 0);
    assert_eq!(picks.len(), 9);
}

#[test]
fn scoop_fillet_variable_is_watertight() {
    let mut topo = Topology::new();
    let (solid, picks) = load_case(&mut topo);
    let edge_laws: Vec<_> = picks
        .into_iter()
        .map(|(e, r)| (e, FilletRadiusLaw::Constant(r)))
        .collect();
    let result = fillet_variable(&mut topo, solid, &edge_laws).unwrap();
    let free = free_edge_count(&topo, result);
    assert_eq!(
        free, 0,
        "scoop fillet must produce a watertight solid, got {free} free edges"
    );
}
