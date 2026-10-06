//! A slotted bin's body fused onto its base, from the gridfinity tool
//! (`binGenerator.export.lipWall.test.ts`, "3x4 (inverse of #1379) slotted +
//! stacking lip preview"): the body arrives as a 1082-face planar mesh whose
//! rounded corners are polygons inscribed in the base's rim, standing on that
//! rim at z 4.75.
//!
//! Each body wall's bottom edge is a chord of a corner's rim arc with both
//! ends on it. The base's corner cones take no section, so they are rebuilt
//! from their edge images and the CommonBlock edges, which are keyed by
//! endpoint pair alone: a cone that took the chords for its rim left each rim
//! arc on the base top's sliver alone and each chord on three faces. The
//! cones keep their arcs only while each body edge's crossing at its own end
//! vertex lands short of it and leaves a sliver pave.
//!
//! Data: `slotted_lip_body.bin` and `slotted_lip_base.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::oriented_solid_volume;
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

#[test]
fn a_mesh_body_on_its_base_rim_fuses_exactly() {
    let mut topo = Topology::new();
    let body = load(&mut topo, "slotted_lip_body.bin");
    let base = load(&mut topo, "slotted_lip_base.bin");
    let parts = oriented_solid_volume(&topo, body, 0.001).unwrap()
        + oriented_solid_volume(&topo, base, 0.001).unwrap();

    let fallbacks = boolean::mesh_fallback_count();
    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, body, base).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "the fuse fell back"
    );

    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, fused).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert!(
        uses.values().all(|&n| n == 2),
        "the fuse left free or over-shared edges"
    );
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(&topo, fused, 0.01).unwrap()),
        "mesh not watertight"
    );

    // The body only rests on the base, so the fuse holds both whole.
    let vol = oriented_solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (vol - parts).abs() / parts < 1e-6,
        "fused volume {vol:.3} != body + base {parts:.3}"
    );
}
