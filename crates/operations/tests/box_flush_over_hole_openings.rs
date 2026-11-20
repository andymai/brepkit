//! A box fused flush onto a drilled plate's face, its footprint over the
//! holes' openings: five holes open inside the footprint, and the footprint's
//! top and bottom edges run through the centres of two more, so each of those
//! openings is cut in half by a box edge. The footprint's loop runs through
//! the holes' rims, so it is no internal loop of the face; the box edge's
//! chord across an opening and the opening's half rim are two edges, each of
//! its own two faces. The fuse is exact, valid and at its closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const R: f64 = 1.7;
/// The holes' centres (y, z), through the plate along x.
const HOLES: [(f64, f64); 7] = [
    (-8.66, -5.0),
    (0.0, -10.0),
    (8.66, -5.0),
    (-8.66, 5.0),
    (0.0, 0.0),
    (8.66, 5.0),
    (0.0, 10.0),
];

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

#[test]
fn a_box_flush_over_hole_openings_fuses_exactly() {
    let mut topo = Topology::new();
    // The plate: 6 thick in x, its face at x = 6; the box's footprint is
    // y in [-10, 10], z in [-10, 10].
    let mut plate = placed_box(&mut topo, [0.0, -20.0, -20.0], [6.0, 20.0, 20.0]);
    for (y, z) in HOLES {
        let drill = make_cylinder(&mut topo, R, 8.0).unwrap();
        transform_solid(
            &mut topo,
            drill,
            &(Mat4::translation(-1.0, y, z) * Mat4::rotation_y(FRAC_PI_2)),
        )
        .unwrap();
        plate = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
    }
    let block = placed_box(&mut topo, [6.0, -10.0, -10.0], [36.0, 10.0, 10.0]);

    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let want = 6.0 * 40.0 * 40.0 - 7.0 * PI * R * R * 6.0 + 30.0 * 20.0 * 20.0;
    let volume = solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}
