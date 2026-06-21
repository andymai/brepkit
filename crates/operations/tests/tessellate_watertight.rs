//! Mesh-watertightness regression guard for solid tessellation.
//!
//! Each primitive's triangle mesh must be a closed 2-manifold: after welding
//! coincident vertices by position, every undirected edge is shared by exactly
//! two triangles (no boundary or non-manifold edges). This guards the
//! shared-edge vertex pool in `tessellate_solid_*` — an STL export consumer
//! (e.g. a slicer) flags any boundary edge as a hole.

#![allow(clippy::unwrap_used)]
use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::solid::SolidId;
use brepkit_topology::topology::Topology;

/// Returns `(edge_count, non_manifold_or_boundary_count)` after welding
/// coincident vertices by quantized position (1e-6), so the check reflects
/// geometric watertightness rather than index sharing.
fn boundary_edges(positions: &[Point3], indices: &[u32]) -> (usize, usize) {
    let q = 1e6;
    let mut canon: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; positions.len()];
    for (i, p) in positions.iter().enumerate() {
        let key = (
            (p.x() * q).round() as i64,
            (p.y() * q).round() as i64,
            (p.z() * q).round() as i64,
        );
        let next = canon.len() as u32;
        remap[i] = *canon.entry(key).or_insert(next);
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let v = [
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        ];
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let boundary = edges.values().filter(|&&c| c != 2).count();
    (edges.len(), boundary)
}

fn assert_watertight(label: &str, topo: &Topology, solid: SolidId) {
    let mesh = tessellate_solid_with_tolerance(topo, solid, 0.01, 0.1).unwrap();
    assert!(!mesh.indices.is_empty(), "{label}: empty mesh");
    let (edges, boundary) = boundary_edges(&mesh.positions, &mesh.indices);
    assert_eq!(
        boundary,
        0,
        "{label}: tessellation not watertight — {boundary}/{edges} edges with incidence != 2 \
         ({} verts, {} tris)",
        mesh.positions.len(),
        mesh.indices.len() / 3,
    );
}

#[test]
fn box_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 8.0, 6.0).unwrap();
    assert_watertight("box", &topo, solid);
}

#[test]
fn cylinder_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 4.0, 10.0).unwrap();
    assert_watertight("cylinder", &topo, solid);
}

#[test]
fn cone_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 5.0, 2.0, 9.0).unwrap();
    assert_watertight("cone", &topo, solid);
}

#[test]
fn sphere_tessellation_is_watertight() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 5.0, 24).unwrap();
    assert_watertight("sphere", &topo, solid);
}
