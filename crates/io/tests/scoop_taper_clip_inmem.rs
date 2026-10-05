//! A scoop clipped to a bin's tapered envelope, from the gridfinity tool's
//! interior fillet (`binGenerator.export.interiorFilletScoops.test.ts`, "a
//! scoop beside tapered side walls keeps the plain fillet").
//!
//! The scoop is a prism along x over [-44.03, 44.03] whose profile is a
//! floor at z 1.25, a lip, a cubic scoop up to the back wall at y -18.15,
//! the back wall, a top at z 23.25 and the outer wall at y -20.03. The
//! envelope is a box over the bin's footprint (x within 44.03, y within
//! 20.03, z 0 to 25.25) with r 3.03 vertical corners, whose x walls lean
//! in below z 6 to x 41.03 at z 0 (the corners there are oblique
//! extrusions of a quarter circle). The scoop's end profiles lie in the
//! envelope's upright x walls, and each passes 0.05 above the corner where
//! the wall, its leaning foot and the corner cylinder meet, touching the
//! cylinder there along its ruling.
//!
//! The clip fell back to a mesh, which every later boolean of the test's
//! fillet then took as an operand.
//!
//! Data: `taper_clip_scoop.bin` and `taper_clip_envelope.bin`.

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

/// Integrated numerically (to 1e-3): the scoop's profile area times its
/// length, the envelope's horizontal sections up its height, and the
/// scoop's sections times the envelope's x extent at each (y, z).
const SCOOP: f64 = 5_248.357;
const ENVELOPE: f64 = 88_153.937;
const COMMON: f64 = 5_078.933;

#[test]
fn a_scoop_clipped_by_a_tapered_envelope_keeps_its_closed_form() {
    let mut topo = Topology::new();
    let scoop = load(&mut topo, "taper_clip_scoop.bin");
    let envelope = load(&mut topo, "taper_clip_envelope.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, scoop, envelope);
    let cut = exact(&mut topo, BooleanOp::Cut, scoop, envelope);
    let fused = exact(&mut topo, BooleanOp::Fuse, scoop, envelope);

    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    for (label, got, want) in [
        ("intersect", volume(common), COMMON),
        ("cut", volume(cut), SCOOP - COMMON),
        ("fuse", volume(fused), SCOOP + ENVELOPE - COMMON),
    ] {
        assert!(
            (got - want).abs() <= 2e-4 * SCOOP,
            "{label} volume {got}, expected {want}"
        );
    }
}
