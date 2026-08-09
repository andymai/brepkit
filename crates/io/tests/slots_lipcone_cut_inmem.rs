//! Slots wall-pattern lip-cone cut (GH #1446/#1445 root, captured 2026-08-08).
//!
//! The tool's generator carves the `slots` wall pattern as one compoundCut of
//! 60 rounded-corner slot prisms (6 planes + 8 cylinders each) from the 3x3x5
//! bin body (42 planes + 32 cylinders + 12 lip cones). Every SINGLE slot cut
//! already emits exactly 6 free boundary edges — a closed hexagonal loop at
//! the slot corner where the tool's rounded corner crosses the bin's lip
//! cone (the loop carries an ellipse edge, a plane-cone section). One cone
//! sub-face piece is dropped. In the full 60-tool compoundCut this becomes
//! 540 free edges, validation rejects the GFA result, and the 14 s mesh
//! fallback emits the 7766-face all-planar solid the tool measured (2x
//! triangles, 56x slower than the reference kernel).
//!
//! Operands captured from the raw kernel via the monkey-patch recipe
//! (compoundCut tools arrive as a Uint32Array — Array.isArray misses them).

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
    let body = load("slots_lip_body.bin", &mut topo);
    let tool = load("slots_lip_tool.bin", &mut topo);
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
fn single_slot_cut_is_watertight_and_analytic() {
    // The pin requires BOTH watertightness and analytic survival: the mesh
    // fallback also produces a watertight solid, but an all-planar one, so
    // the cone count is what distinguishes a real fix from the fallback.
    let mut topo = Topology::new();
    let body = load("slots_lip_body.bin", &mut topo);
    let tool = load("slots_lip_tool.bin", &mut topo);
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        body,
        tool,
    )
    .unwrap();
    let free = free_edge_count(&topo, result);
    let cones = brepkit_topology::explorer::solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&fid| {
            matches!(
                topo.face(fid).unwrap().surface(),
                brepkit_topology::face::FaceSurface::Cone(_)
            )
        })
        .count();
    assert_eq!(
        free, 0,
        "single-slot cut must be watertight, got {free} free edges"
    );
    assert!(
        cones >= 12,
        "the lip cones must survive analytically (mesh fallback would flatten them), got {cones}"
    );

    // Density pin: the display tessellation of developable faces is
    // tolerance-driven (no interior grid, no curvature floor). The floored
    // grid measured 4100 triangles at 0.01 mm / 5 deg; tolerance-driven is
    // 2324. Guards against the floor silently returning to this path.
    let mesh = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
        &topo,
        result,
        0.01,
        5.0_f64.to_radians(),
    )
    .unwrap();
    assert_eq!(
        brepkit_operations::tessellate::boundary_edge_count(&mesh),
        0,
        "single-slot cut mesh must stay watertight"
    );
    let tris = mesh.indices.len() / 3;
    assert!(
        tris < 3000,
        "single-slot cut tessellation densified past the tolerance-driven budget: {tris} triangles"
    );
}
