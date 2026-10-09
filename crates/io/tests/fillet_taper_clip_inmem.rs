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
//! A second material, `scoop_beside_taper_material.bin`, comes from "a scoop
//! beside tapered side walls keeps the plain fillet" in the tool's
//! `binGenerator.export.interiorFilletScoops.test.ts`, against the envelope in
//! `taper_clip_envelope.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
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

fn slab(topo: &mut Topology, y0: f64, y1: f64) -> SolidId {
    let slab = make_box(topo, 100.0, y1 - y0, 40.0).unwrap();
    transform_solid(topo, slab, &Mat4::translation(-50.0, y0, -5.0)).unwrap();
    slab
}

/// The envelope against a material whose pocket ramps up a front scoop and
/// whose side walls take the envelope's lean in their own floor rim. The
/// corner tori meet the leaning walls' planes only past those walls' ends,
/// at the tori's own meridians, and the scoop's rim, fitted to within 1.4e-6
/// of its vertex on a leaning wall, meets that wall's plane there.
///
/// Away from the scoop every horizontal section is a rounded rectangle: the
/// outer box (corner r 3.03 about (+-41, +-17)), the envelope (the same
/// corners shifted to 38 + z/2 below z 6) and the pocket's air (corner
/// 0.1 + w about the same centres, w the floor rim's reach at z). Integrated
/// over those, the envelope trims 234.306 from the material between y -12
/// and 12 and 80.83 beyond y 12.
#[test]
fn a_scooped_fillet_material_clips_to_its_tapered_envelope_exactly() {
    let mut topo = Topology::new();
    let material = load(&mut topo, "scoop_beside_taper_material.bin");
    let envelope = load(&mut topo, "taper_clip_envelope.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, material, envelope);
    let trimmed = |topo: &mut Topology, y0: f64, y1: f64| {
        let (a, b) = (slab(topo, y0, y1), slab(topo, y0, y1));
        let whole = exact(topo, BooleanOp::Intersect, material, a);
        let clipped = exact(topo, BooleanOp::Intersect, common, b);
        (whole, clipped)
    };
    // Planes and cylinders only: both integrate exactly.
    let (whole, clipped) = trimmed(&mut topo, -12.0, 12.0);
    let run =
        solid_volume(&topo, whole, 0.001).unwrap() - solid_volume(&topo, clipped, 0.001).unwrap();
    assert!(
        (run - 234.306).abs() < 1e-2,
        "straight runs trimmed by {run}"
    );
    let (whole, clipped) = trimmed(&mut topo, 12.0, 25.0);
    let corners = oriented_solid_volume(&topo, whole, 0.001).unwrap()
        - oriented_solid_volume(&topo, clipped, 0.001).unwrap();
    assert!(
        (corners - 80.83).abs() < 0.1,
        "corners trimmed by {corners}"
    );
}
