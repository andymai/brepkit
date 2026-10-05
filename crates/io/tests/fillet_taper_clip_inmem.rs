//! The gridfinity tool's interior fillet material clipped to a bin's tapered
//! envelope (`binGenerator.export.interiorFillet.test.ts`, "a tapered bottom
//! band keeps the fillet inside the leaning wall"), captured on a
//! brepkit-wasm build that rounds the compartment air's floor rim.
//!
//! The material is a rounded box (half sizes 44.03 by 23.03, corners r 3.03,
//! z 1.25 to 22.55) less its pocket (half sizes 43.55 by 22.55, corners r
//! 2.55, floor at z 2.25), whose floor rim is rounded r 2.45: cylinders along
//! the runs and spindle tori (major 0.1) at the corners. The envelope's walls
//! lean in below z 6, to half sizes 41.03 by 20.03 at z 0, over corners that
//! are oblique extrusions of a quarter circle; above z 6 it is the material's
//! outer box with the same corner cylinders. Below z 5 the rounded pocket
//! reaches past the leaning walls, so their fuse encloses a thin ring of the
//! pocket's air.
//!
//! Data: `fillet_taper_clip_material.bin` and `fillet_taper_clip_envelope.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let fallbacks = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, op, a, b).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "{op:?} fell back"
    );
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert!(
        uses.values().all(|&n| n == 2),
        "{op:?} left free or over-shared edges"
    );
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{op:?}: mesh not watertight"
    );
    result
}

/// Integrated numerically (to 1e-3) over horizontal sections, each a pair of
/// concentric rounded rectangles.
const MATERIAL: f64 = 6_927.028;
const ENVELOPE: f64 = 99_873.947;
const COMMON: f64 = 5_657.297;

#[test]
fn a_fillet_material_and_a_tapered_envelope_keep_their_closed_forms() {
    let mut topo = Topology::new();
    let material = load(&mut topo, "fillet_taper_clip_material.bin");
    let envelope = load(&mut topo, "fillet_taper_clip_envelope.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, material, envelope);
    let cut = exact(&mut topo, BooleanOp::Cut, material, envelope);
    let fused = exact(&mut topo, BooleanOp::Fuse, material, envelope);

    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    for (label, got, want) in [
        ("intersect", volume(common), COMMON),
        ("cut", volume(cut), MATERIAL - COMMON),
        ("fuse", volume(fused), MATERIAL + ENVELOPE - COMMON),
    ] {
        assert!(
            (got - want).abs() <= 2e-4 * MATERIAL,
            "{label} volume {got}, expected {want}"
        );
    }
}
