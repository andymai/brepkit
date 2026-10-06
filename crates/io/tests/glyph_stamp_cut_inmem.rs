//! A text stamp cut into the underside of a fit-test card, from the gridfinity
//! tool (`fitTestSlice.scenario.test.ts`): the glyph is a mirrored "e", a
//! prism standing from 0.01 below the card's bottom face to 0.4 inside it.
//!
//! The card's bottom face splits into the glyph's ring around its counter
//! and the counter itself. The ring's sample point was the midpoint between
//! the outline's first sample and the nearest counter sample, and from the
//! foot of the "e"'s mouth that midpoint lies in the mouth, off the ring:
//! the ring read as outside the glyph, stayed on the card's floor over the
//! pocket, and its outline edges went to three faces. The cut fell back to
//! a mesh, and every later cut on the card ran on that mesh (tens of seconds
//! each in wasm) until one trapped.
//!
//! Data: `glyph_stamp_card.bin` and `glyph_stamp_glyph.bin`.

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
fn a_glyph_stamped_into_a_card_cuts_exactly() {
    let mut topo = Topology::new();
    let card = load(&mut topo, "glyph_stamp_card.bin");
    let glyph = load(&mut topo, "glyph_stamp_glyph.bin");
    let card_vol = oriented_solid_volume(&topo, card, 0.001).unwrap();

    let fallbacks = boolean::mesh_fallback_count();
    let cut = boolean::boolean(&mut topo, BooleanOp::Cut, card, glyph).unwrap();
    let common = boolean::boolean(&mut topo, BooleanOp::Intersect, card, glyph).unwrap();
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

    // The glyph's 0.01 below the card is outside it, so the two pieces make
    // up the card and the common part is under the glyph's own volume.
    let cut_vol = oriented_solid_volume(&topo, cut, 0.001).unwrap();
    let common_vol = oriented_solid_volume(&topo, common, 0.001).unwrap();
    let glyph_vol = oriented_solid_volume(&topo, glyph, 0.001).unwrap();
    assert!(
        (cut_vol + common_vol - card_vol).abs() / card_vol < 1e-6,
        "cut {cut_vol:.4} + common {common_vol:.4} != card {card_vol:.4}"
    );
    assert!(
        common_vol > 0.9 * glyph_vol && common_vol < glyph_vol,
        "common {common_vol:.4} against glyph {glyph_vol:.4}"
    );
}
