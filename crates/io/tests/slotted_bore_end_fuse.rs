//! A box fused over the end of a bore that a slot cut from below opens
//! along most of its length: the bore's wall is a cylinder whose wire
//! carries the slot's outline across the wall's seam (the seam runs along
//! the slot's floor), and the box's side cuts the wall's arc just past the
//! slot's end. Laid out right, the wall's face trace also returned the slot
//! itself (a pocket of the outside, bounded by the wall's own boundary run
//! backwards) as a piece of the wall, which no neighbour shared. Read from
//! STEP with the bore's seam on the slot's floor, the fuse is exact, valid
//! and at its closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

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

const R: f64 = 1.45;
/// The bore's axis, along y.
const AXIS: (f64, f64) = (-20.568, 483.595);
/// The slot: x across the bore, up to just under the bore's axis, and
/// along y short of both ends.
const SLOT: ([f64; 2], f64, [f64; 2]) = ([-22.5, -18.5], 483.5, [-48.5, 48.0]);

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

fn through_step(topo: &Topology, solid: SolidId, into: &mut Topology) -> SolidId {
    read_step(&write_step(topo, &[solid]).unwrap(), into).unwrap()[0]
}

/// The bore's disc below `z`, as an area: the slot takes it along its run.
fn disc_below(z: f64) -> f64 {
    let h = z - (AXIS.1 - R);
    let d = R - h;
    R * R * (d / R).acos() - d * (R * R - d * d).sqrt()
}

/// Fuses the block over the slotted bore's end with the bore's seam turned
/// by `turn` degrees about its axis (270: on the slot's floor) and checks
/// the result.
fn fuse_with_seam_turned(turn: f64) {
    {
        let mut built = Topology::new();
        let plate = placed_box(&mut built, [-40.0, -51.0, 470.0], [40.0, 51.0, 500.0]);
        let bore = make_cylinder(&mut built, R, 102.0).unwrap();
        transform_solid(
            &mut built,
            bore,
            &(Mat4::translation(AXIS.0, 51.0, AXIS.1)
                * Mat4::rotation_x(FRAC_PI_2)
                * Mat4::rotation_z(turn.to_radians())),
        )
        .unwrap();
        let bored = boolean(&mut built, BooleanOp::Cut, plate, bore).unwrap();
        let slot = placed_box(
            &mut built,
            [SLOT.0[0], SLOT.2[0], 460.0],
            [SLOT.0[1], SLOT.2[1], SLOT.1],
        );
        let part = boolean(&mut built, BooleanOp::Cut, bored, slot).unwrap();
        let block = placed_box(&mut built, [-30.0, -51.0, 441.72], [30.0, -48.0, 501.72]);

        let mut topo = Topology::new();
        let (part, block) = (
            through_step(&built, part, &mut topo),
            through_step(&built, block, &mut topo),
        );
        // The plate less the bore and the slot below it, and the block, less
        // what of the plate the block covers: its 3 of the plate's length,
        // less the bore there and the slot's last 0.5.
        let slot_section = (SLOT.0[1] - SLOT.0[0]) * (SLOT.1 - 470.0) - disc_below(SLOT.1);
        let part_volume =
            80.0 * 102.0 * 30.0 - PI * R * R * 102.0 - slot_section * (SLOT.2[1] - SLOT.2[0]);
        let shared = 60.0 * 3.0 * 30.0 - PI * R * R * 3.0 - slot_section * 0.5;
        let want = part_volume + 60.0 * 3.0 * 60.0 - shared;

        let before = mesh_fallback_count();
        let fused = boolean(&mut topo, BooleanOp::Fuse, part, block).unwrap();
        assert_eq!(mesh_fallback_count(), before, "turn {turn}: mesh fallback");
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "turn {turn}: {:?}", report.issues);
        let volume = solid_volume(&topo, fused, 0.001).unwrap();
        assert!(
            (volume - want).abs() < 1e-6 * want,
            "turn {turn}: volume {volume}, want {want}"
        );
    }
}

#[test]
fn a_box_over_a_slotted_bores_end_fuses_exactly() {
    fuse_with_seam_turned(270.0);
}

/// The bore's seam off the slot's floor, on its walls or over the bore: the
/// wall's outline keeps the slot as a hole and the fuse still falls back.
#[test]
#[ignore = "falls back to the mesh boolean with the seam off the slot's floor"]
fn a_box_over_a_slotted_bores_end_fuses_exactly_with_the_seam_elsewhere() {
    for turn in [0.0, 90.0, 180.0] {
        fuse_with_seam_turned(turn);
    }
}
