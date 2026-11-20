//! A pin through a plate and across the side of a block standing over it:
//! the plate cuts the pin's wall along two whole circles, which start and
//! end on the wall's seam line, and the block's side cuts a patch out of the
//! wall above them. The band of the pin's wall between the two circles (the
//! wall of the pin's hole through the plate), bounded only by circles and
//! the seam, was dropped: the cut was left open and fell back to the mesh
//! boolean. Read from STEP (where a closed circle starts at its vertex on
//! the seam), with the block on either side of the pin's axis and in every
//! pose of the pin's seam (so the patch also spans it), the cut is exact,
//! valid and at its closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const R: f64 = 1.7;
/// The pin's axis, 1.09 from the block's side face at x = -10.
const PIN: (f64, f64) = (-11.09, -394.99);
const BLOCK_HEIGHT: f64 = 771.0 - 738.2;

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

/// `solid` written to STEP and read back into a fresh arena.
fn through_step(topo: &Topology, solid: SolidId, into: &mut Topology) -> SolidId {
    read_step(&write_step(topo, &[solid]).unwrap(), into).unwrap()[0]
}

/// The plate, a block over it whose side face is x = -10 (`behind`: the
/// block reaches back over the pin's axis to x = -30, else it stands in
/// front of the axis, to x = 10), and a column joining them away from the
/// pin, as one solid.
fn part(topo: &mut Topology, behind: bool) -> SolidId {
    let (lo, hi, col) = if behind {
        (-30.0, -10.0, -30.0)
    } else {
        (-10.0, 10.0, 0.0)
    };
    let plate = placed_box(topo, [-60.0, -420.0, 707.2], [60.0, -370.0, 713.2]);
    let column = placed_box(topo, [col, -400.0, 713.2], [col + 10.0, -390.0, 738.2]);
    let block = placed_box(topo, [lo, -400.0, 738.2], [hi, -390.0, 771.0]);
    let base = boolean(topo, BooleanOp::Fuse, plate, column).unwrap();
    boolean(topo, BooleanOp::Fuse, base, block).unwrap()
}

/// The pin's share of the part: its whole disc through the plate, and in
/// the block the part of its disc on the block's side of the face.
fn removed(behind: bool) -> f64 {
    let d = -10.0 - PIN.0;
    let in_front = R * R * (d / R).acos() - d * (R * R - d * d).sqrt();
    let disc = PI * R * R;
    let in_block = if behind { disc - in_front } else { in_front };
    disc * 6.0 + in_block * BLOCK_HEIGHT
}

#[test]
fn a_pin_through_a_plate_and_a_wall_over_it_cuts_exactly() {
    for behind in [true, false] {
        for seam in [0.0_f64, 90.0, 180.0, 270.0] {
            let label = format!(
                "block {}, seam {seam}",
                if behind { "behind" } else { "in front" }
            );
            let mut built = Topology::new();
            let part = part(&mut built, behind);
            let pin = make_cylinder(&mut built, R, 444.0).unwrap();
            transform_solid(
                &mut built,
                pin,
                &(Mat4::translation(PIN.0, PIN.1, 706.2) * Mat4::rotation_z(seam.to_radians())),
            )
            .unwrap();
            let mut topo = Topology::new();
            let (part, pin) = (
                through_step(&built, part, &mut topo),
                through_step(&built, pin, &mut topo),
            );
            let want = solid_volume(&topo, part, 0.001).unwrap() - removed(behind);
            let before = mesh_fallback_count();
            let cut = boolean(&mut topo, BooleanOp::Cut, part, pin).unwrap();
            assert_eq!(mesh_fallback_count(), before, "{label}: mesh fallback");
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let volume = solid_volume(&topo, cut, 0.001).unwrap();
            assert!(
                (volume - want).abs() < 1e-6 * want,
                "{label}: volume {volume}, want {want}"
            );
        }
    }
}
