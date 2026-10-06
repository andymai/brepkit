//! A caption glyph engraved into a label plate, from the gridfinity tool
//! (`labelPlateBuilder.test.ts`, two-line captions): the glyph's walls are
//! extruded from fitted font curves, and one rounded edge is a cylinder face
//! spanning 8 degrees whose rulings stand 1.3e-6 and 9.7e-6 off the
//! cylinder its arcs were fitted to.
//!
//! The plate's top plane cuts that cylinder in a circle whose in-face arc is
//! shorter than one of the circle's sample steps, so only its exact crossings
//! with the wall's boundary keep it. Those crossings are the rulings, which
//! the exact circle-segment test rejected at 1e-7, so the arc was dropped,
//! the glyph's outline on the plate top stayed open, and the cut fell back.
//! In wasm the next cut ran on the mesh for 786 s and trapped.
//!
//! Data: `glyph_ruling_plate.bin` and `glyph_ruling_glyph.bin`.

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

fn every_edge_twice(topo: &Topology, solid: SolidId) -> bool {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    uses.values().all(|&n| n == 2)
}

#[test]
fn a_glyph_engraved_into_a_label_plate_cuts_exactly() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "glyph_ruling_plate.bin");
    let glyph = load(&mut topo, "glyph_ruling_glyph.bin");
    let plate_vol = oriented_solid_volume(&topo, plate, 0.001).unwrap();

    let fallbacks = boolean::mesh_fallback_count();
    let cut = boolean::boolean(&mut topo, BooleanOp::Cut, plate, glyph).unwrap();
    let common = boolean::boolean(&mut topo, BooleanOp::Intersect, plate, glyph).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "a boolean fell back"
    );

    for (solid, name) in [(cut, "cut"), (common, "intersect")] {
        assert!(
            every_edge_twice(&topo, solid),
            "the {name} left free or over-shared edges"
        );
        let report = validate_solid(&topo, solid).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, solid, 0.01).unwrap()),
            "{name} mesh not watertight"
        );
    }

    // The glyph stands 0.01 above the plate top, so the two pieces make up
    // the plate and the common part is under the glyph's own volume.
    let cut_vol = oriented_solid_volume(&topo, cut, 0.001).unwrap();
    let common_vol = oriented_solid_volume(&topo, common, 0.001).unwrap();
    let glyph_vol = oriented_solid_volume(&topo, glyph, 0.001).unwrap();
    assert!(
        (cut_vol + common_vol - plate_vol).abs() / plate_vol < 1e-6,
        "cut {cut_vol:.4} + common {common_vol:.4} != plate {plate_vol:.4}"
    );
    assert!(
        common_vol > 0.9 * glyph_vol && common_vol < glyph_vol,
        "common {common_vol:.4} against glyph {glyph_vol:.4}"
    );
}
