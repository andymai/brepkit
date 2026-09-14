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
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceSurface;
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

fn edge_midpoint(topo: &Topology, eid: EdgeId) -> Point3 {
    let e = topo.edge(eid).unwrap();
    let (a, b) = (
        topo.vertex(e.start()).unwrap().point(),
        topo.vertex(e.end()).unwrap().point(),
    );
    let (t0, t1) = e.curve().domain_with_endpoints(a, b);
    e.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b)
}

/// Pairs of distinct edges that coincide geometrically: same endpoints (in
/// either order) and the same midpoint. A line and an arc closing a lens
/// share endpoints but not midpoints and are not twins.
fn twin_edge_pairs(topo: &Topology, solid: SolidId) -> Vec<(EdgeId, EdgeId)> {
    let ends = edge_ends(topo, solid);
    let mut twins = Vec::new();
    for (i, (ea, a0, a1)) in ends.iter().enumerate() {
        for (eb, b0, b1) in &ends[i + 1..] {
            let fwd = (*a0 - *b0).length().max((*a1 - *b1).length());
            let rev = (*a0 - *b1).length().max((*a1 - *b0).length());
            if fwd.min(rev) < 1e-9
                && (edge_midpoint(topo, *ea) - edge_midpoint(topo, *eb)).length() < 1e-9
            {
                twins.push((*ea, *eb));
            }
        }
    }
    twins
}

/// Edge uses keyed by quantized geometry (endpoints and midpoint) instead of
/// arena id, so two ids on one curve count as one edge.
#[allow(clippy::cast_possible_truncation)]
fn positional_edge_uses(topo: &Topology, solid: SolidId) -> HashMap<[i64; 9], usize> {
    let q = |p: Point3| {
        [
            (p.x() * 1e6).round() as i64,
            (p.y() * 1e6).round() as i64,
            (p.z() * 1e6).round() as i64,
        ]
    };
    let mut uses = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (mut a, mut b) = (
                    q(topo.vertex(e.start()).unwrap().point()),
                    q(topo.vertex(e.end()).unwrap().point()),
                );
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let m = q(edge_midpoint(topo, oe.edge()));
                let key = [a[0], a[1], a[2], b[0], b[1], b[2], m[0], m[1], m[2]];
                *uses.entry(key).or_insert(0) += 1;
            }
        }
    }
    uses
}

fn surface_census(topo: &Topology, solid: SolidId) -> HashMap<&'static str, usize> {
    let mut census = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let tag = match topo.face(fid).unwrap().surface() {
            FaceSurface::Plane { .. } => "plane",
            FaceSurface::Cylinder(_) => "cylinder",
            FaceSurface::Cone(_) => "cone",
            FaceSurface::Sphere(_) => "sphere",
            FaceSurface::Torus(_) => "torus",
            FaceSurface::Nurbs(_) => "nurbs",
        };
        *census.entry(tag).or_insert(0) += 1;
    }
    census
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
    let body_volume = solid_volume(&topo, body, 0.01).unwrap();
    let tool_volume = solid_volume(&topo, tool, 0.01).unwrap();
    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "the cut took the mesh fallback"
    );
    // Exactness: the tool's curved surfaces survive as typed faces (a fallback
    // blob is all planes), every edge is used twice by geometry and not only
    // by id, and the volume sits between "tool fully outside" and "tool fully
    // inside".
    let census = surface_census(&topo, result);
    assert!(
        census.get("cylinder").copied().unwrap_or(0) >= 4 && census.contains_key("torus"),
        "cut result lost the scoop's curved faces: {census:?}"
    );
    let by_id = edge_use_counts(&topo, result);
    assert_eq!(
        by_id.values().filter(|&&c| c != 2).count(),
        0,
        "cut result must be manifold by id"
    );
    let by_position = positional_edge_uses(&topo, result);
    assert_eq!(
        by_position.values().filter(|&&c| c != 2).count(),
        0,
        "cut result must be manifold by position"
    );
    let volume = solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        volume <= body_volume + 1e-6 && volume >= body_volume - tool_volume - 1e-6,
        "cut volume {volume} outside [{}, {body_volume}]",
        body_volume - tool_volume
    );
}
