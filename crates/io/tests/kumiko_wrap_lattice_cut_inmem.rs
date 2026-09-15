//! The first mesh fallback of the gridfinity tool's kumiko corner wrap export
//! (`slideRailBuilder.test.ts`, "is not carved away by a kumiko wrap either"),
//! captured on brepkit-wasm 3.4.0: the flat wall's planar strut lattice
//! (62 planes) compound-cut by four helical corner struts (39 to 64 NURBS
//! walls plus two or three planar faces each). The fuse ladder's arrangements end with "open growth
//! shell with 6 faces would be dropped" and 77 non-manifold edges, the op
//! falls back to a 647-face planar blob, and every later compound cut in the
//! export consumes such blobs (the 8-tool cut then takes 226 s in the tool).
//! Natively this cut is 9.5 s, 5.3 s of it in the edge-face phase.
//!
//! Data: `kumiko_wrap_lattice.bin` (base), `kumiko_wrap_strut_{0..3}.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOptions, compound_cut};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
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

fn load_case(topo: &mut Topology) -> (SolidId, Vec<SolidId>) {
    let base = load(topo, "kumiko_wrap_lattice.bin");
    let tools = (0..4)
        .map(|i| load(topo, &format!("kumiko_wrap_strut_{i}.bin")))
        .collect();
    (base, tools)
}

#[test]
fn kumiko_wrap_lattice_fixture_is_faithful() {
    let mut topo = Topology::new();
    let (base, tools) = load_case(&mut topo);
    assert_eq!(surface_census(&topo, base).get("plane"), Some(&62));
    for &tool in &tools {
        let census = surface_census(&topo, tool);
        assert!(
            census.get("nurbs").copied().unwrap_or(0) >= 39,
            "strut lost its NURBS walls: {census:?}"
        );
        assert!(
            census.get("plane").copied().unwrap_or(0) >= 2,
            "strut caps: {census:?}"
        );
    }
    for solid in std::iter::once(base).chain(tools) {
        let uses = edge_use_counts(&topo, solid);
        assert_eq!(
            uses.values().filter(|&&c| c != 2).count(),
            0,
            "operand is not manifold"
        );
    }
}

#[test]
#[ignore = "ready repro: the flat lattice compound-cut by four helical struts must stay exact (assembly drops an open growth shell; 77 non-manifold edges)"]
fn kumiko_wrap_lattice_compound_cut_stays_exact() {
    let mut topo = Topology::new();
    let (base, tools) = load_case(&mut topo);
    let before = boolean::mesh_fallback_count();
    let result = compound_cut(&mut topo, base, &tools, BooleanOptions::default()).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "the compound cut took the mesh fallback"
    );
    let census = surface_census(&topo, result);
    assert!(
        census.get("nurbs").copied().unwrap_or(0) > 0,
        "the struts' NURBS walls did not survive: {census:?}"
    );
    let uses = edge_use_counts(&topo, result);
    assert_eq!(
        uses.values().filter(|&&c| c != 2).count(),
        0,
        "result must be manifold"
    );
}
