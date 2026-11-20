//! A box fused flat onto a drilled plate, overhanging it all round: the
//! box's bottom face takes the plate's outline and its holes' rims. The
//! plate's top and the box bottom inside the outline lie face to face and
//! go; the rest of the box bottom is a ring round the outline and a disc
//! over each hole, capping it. The discs were carved into the ring, which
//! kept the rims as holes beside them. The fuse is exact, valid and at its
//! closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const HOLE_R: f64 = 1.7;
const HOLES: [(f64, f64); 4] = [(0.0, -13.5), (-11.691, 6.75), (0.0, 0.0), (11.691, 6.75)];

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

#[test]
fn a_box_fused_flat_onto_a_drilled_plate_caps_its_holes() {
    let mut topo = Topology::new();
    let mut plate = placed_box(&mut topo, [-18.0, -18.0, 0.0], [18.0, 18.0, 3.0]);
    for (x, y) in HOLES {
        let drill = make_cylinder(&mut topo, HOLE_R, 5.0).unwrap();
        transform_solid(&mut topo, drill, &Mat4::translation(x, y, -1.0)).unwrap();
        plate = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
    }
    let block = placed_box(&mut topo, [-57.5, -51.0, 3.0], [57.5, 51.0, 118.0]);

    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let want = 36.0 * 36.0 * 3.0 - 4.0 * PI * HOLE_R * HOLE_R * 3.0 + 115.0 * 102.0 * 115.0;
    let volume = solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}
