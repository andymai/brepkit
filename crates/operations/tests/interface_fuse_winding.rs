//! #1538 coplanar-interface family: cutting a through-hole out of a plate and
//! fusing the plate onto a block across the shared plane must produce strictly
//! valid winding (the free/over census cannot see same-direction shared
//! edges; only `validate_solid`'s orientation check can).
//!
//! Three winding emitters have failed here, each invisible to watertightness:
//! - the internal-loops splitter normalized disc/hole loops with a signed
//!   area taken in the surface's own parameterization, which inverts the
//!   verdict on a down-facing plane (fixed: local-frame areas);
//! - `merge_duplicate_edges` never flipped closed edges, silently reversing
//!   winding when two coincident circles parameterize opposite ways (fixed:
//!   tangent comparison at the shared point — parameter-frame evaluation is
//!   NOT valid for closed curves, whose domains anchor at their own
//!   reference directions);
//! - `rebuild_face_with_cb_edges` degenerated to `forward=true` for a closed
//!   rim replaced by its CommonBlock circle (fixed: same comparison).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn plate_and_block(topo: &mut Topology) -> (SolidId, SolidId) {
    let plate = brepkit_operations::primitives::make_box(topo, 80.0, 80.0, 5.0).unwrap();
    brepkit_operations::transform::transform_solid(topo, plate, &Mat4::translation(0.0, 0.0, 5.0))
        .unwrap();
    let block = brepkit_operations::primitives::make_box(topo, 80.0, 80.0, 5.0).unwrap();
    (plate, block)
}

fn assert_strictly_valid(topo: &Topology, sid: SolidId, label: &str) {
    let report = brepkit_operations::validate::validate_solid(topo, sid).unwrap();
    assert!(report.is_valid(), "{label} must validate: {report:?}");
}

#[test]
fn rect_through_hole_cut_and_interface_fuse_have_valid_winding() {
    let mut topo = Topology::new();
    let (plate, block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 7.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(30.0, 30.0, 4.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "rect through-hole cut");

    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_strictly_valid(&topo, fused, "rect interface fuse");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, fused, 0.05).unwrap();
    assert!((vol - 62000.0).abs() < 0.5, "fuse volume {vol:.3}");
}

#[test]
fn circle_through_hole_cut_and_interface_fuse_have_valid_winding() {
    let mut topo = Topology::new();
    let (plate, block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 7.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(40.0, 40.0, 4.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "circle through-hole cut");

    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_strictly_valid(&topo, fused, "circle interface fuse");
}

/// A cutter whose bottom cap is COINCIDENT with the plate's bottom plane
/// (the circle-insert cutDepth == floor configuration). The kept band's
/// original rim merges with the section circle parameterized the other way;
/// the direction map must come from tangents at the shared point, not from
/// parameter-frame evaluation (a closed circle's domain anchors at its own
/// reference direction).
#[test]
fn coincident_cap_pocket_cut_has_valid_winding() {
    let mut topo = Topology::new();
    let (plate, _block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 6.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(40.0, 40.0, 5.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "coincident-cap pocket cut");
}
