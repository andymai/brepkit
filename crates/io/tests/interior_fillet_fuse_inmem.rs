//! The gridfinity tool's interior fillet fused into its bin
//! (`binGenerator.export.interiorFillet.test.ts`), captured on a brepkit-wasm
//! build that rounds the compartment air's floor rim.
//!
//! The fillet material is the compartment outline grown into the walls and
//! floor less the rounded air. At each of the pocket's four outer corners
//! its fillet torus is a spindle (major 0.1, minor 2.45) whose outer equator
//! touches the bin's corner wall, a cylinder of radius 2.55 on the torus's
//! axis, along one circle at z = 4.7. Traced by the marcher, that circle
//! came apart into dozens of pieces, the wall was split there into pieces
//! the fuse could not classify, and the fuse fell back to a mesh.
//!
//! Two more captures from `binGenerator.export.interiorFilletScoops.test.ts`
//! fuse the plain fillet into bins whose scoops or wall cutout change the
//! pocket's corners: there the bin's corner wall is only a strip of the
//! cylinder the material's air wall lies on, so the air wall must be split
//! along the strip's edges to be matched with it.
//!
//! Data: `interior_fillet_bin.bin` (the bin), `interior_fillet_material.bin`
//! (the fillet material), and the `interior_fillet_scoops_*` and
//! `interior_fillet_cutout_*` pairs.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn edge_uses(topo: &Topology, solid: SolidId) -> HashMap<usize, usize> {
    let mut uses = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    uses
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let fallbacks = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, op, a, b).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "{op:?} fell back"
    );
    assert!(
        edge_uses(topo, result).values().all(|&n| n == 2),
        "{op:?} left free or over-shared edges"
    );
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{op:?}: mesh not watertight");
    result
}

/// Fuses the fillet material into its bin and cuts it from the bin, both
/// exactly, and checks they read the same overlap: what the fuse leaves out
/// of the material is the part inside the bin, which the cut takes away.
fn assert_fuses_exactly(bin_data: &str, material_data: &str) {
    let mut topo = Topology::new();
    let bin = load(&mut topo, bin_data);
    let material = load(&mut topo, material_data);
    let fused = exact(&mut topo, BooleanOp::Fuse, bin, material);
    let cut = exact(&mut topo, BooleanOp::Cut, bin, material);

    let volume = |topo: &Topology, s: SolidId| solid_volume(topo, s, 0.001).unwrap();
    let (v_bin, v_material) = (volume(&topo, bin), volume(&topo, material));
    let through_fuse = v_bin + v_material - volume(&topo, fused);
    let through_cut = v_bin - volume(&topo, cut);
    assert!(
        (through_fuse - through_cut).abs() < 1e-6 * v_bin,
        "overlap {through_fuse} through the fuse, {through_cut} through the cut"
    );
    // The overlap both readings agree on, pinned so that a classification
    // error they share also fails.
    assert!(
        (through_fuse - 12_957.832_253_1).abs() < 1e-6 * v_bin,
        "overlap {through_fuse}"
    );
}

#[test]
fn interior_fillet_material_fuses_into_its_bin_exactly() {
    assert_fuses_exactly("interior_fillet_bin.bin", "interior_fillet_material.bin");
}

/// Scoops on two adjacent walls: at the corners beside them the bin's wall
/// is a strip topped by a lip cone's arc that stops short of the air wall's
/// quarter, beside a plane whose sloped edge is a NURBS curve stored from
/// its end vertex back.
#[test]
fn interior_fillet_material_fuses_into_a_scooped_bin_exactly() {
    assert_fuses_exactly(
        "interior_fillet_scoops_bin.bin",
        "interior_fillet_scoops_material.bin",
    );
}

/// A wall cutout beside a scoop.
#[test]
fn interior_fillet_material_fuses_into_a_bin_with_a_wall_cutout_exactly() {
    assert_fuses_exactly(
        "interior_fillet_cutout_bin.bin",
        "interior_fillet_cutout_material.bin",
    );
}
