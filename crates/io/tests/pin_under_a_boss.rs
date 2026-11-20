//! A pin cut along a plate's underside, just above it, where a square boss
//! stands under the plate with a round pocket sunk through it. The
//! underside is a plane face with the boss's square foot as its hole, and
//! the pin's wall meets it in two lines that cross that hole: inside the
//! square the pin runs through the boss, where the underside is no face.
//! Trimmed only to the underside's outline, the lines ran on through the
//! hole and split the pin's wall where nothing met it. Read from STEP, as
//! a file gives them, the cut is exact, valid and at its integrated volume
//! in every turn of the pin's seam.
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

const PIN_R: f64 = 1.7;
/// The pin's axis, along y: across the boss, 0.106 above the underside.
const PIN_X: f64 = 21.25;
const PIN_Z: f64 = 0.106;
const POCKET_R: f64 = 30.5;

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

/// The plate (150 x 209 x 10 above z = 0), the boss under it (84 x 84 x 1)
/// and the pocket (radius 30.5, from the boss's foot 5 into the plate).
fn part(topo: &mut Topology) -> SolidId {
    let plate = placed_box(topo, [-75.0, -105.0, 0.0], [75.0, 104.0, 10.0]);
    let boss = placed_box(topo, [-42.0, -42.0, -1.0], [42.0, 42.0, 0.0]);
    let bossed = boolean(topo, BooleanOp::Fuse, plate, boss).unwrap();
    let pocket = make_cylinder(topo, POCKET_R, 6.0).unwrap();
    transform_solid(topo, pocket, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    boolean(topo, BooleanOp::Cut, bossed, pocket).unwrap()
}

fn part_volume() -> f64 {
    150.0 * 209.0 * 10.0 + 84.0 * 84.0 - PI * POCKET_R * POCKET_R * 6.0
}

/// What the pin takes from the part: its disc above the underside along the
/// plate past the boss, and above the boss's foot along the boss outside the
/// pocket (the pocket holds the whole pin's height).
fn removed_volume() -> f64 {
    let n = 20_000;
    let mut removed = 0.0;
    for i in 0..n {
        let x = PIN_X - PIN_R + (f64::from(i) + 0.5) * 2.0 * PIN_R / f64::from(n);
        let dx = 2.0 * PIN_R / f64::from(n);
        let h = (PIN_R * PIN_R - (x - PIN_X).powi(2)).sqrt();
        let (lo, hi) = (PIN_Z - h, PIN_Z + h);
        let above = |floor: f64| (hi - lo.max(floor)).max(0.0);
        let boss_run = 84.0 - 2.0 * (POCKET_R * POCKET_R - x * x).sqrt();
        removed += dx * (above(0.0) * (62.0 + 63.0) + above(-1.0) * boss_run);
    }
    removed
}

fn through_step(topo: &Topology, solid: SolidId, into: &mut Topology) -> SolidId {
    read_step(&write_step(topo, &[solid]).unwrap(), into).unwrap()[0]
}

/// Cuts the pin, its seam turned by `turn` degrees about its axis from the
/// plate's far side, and checks the result.
fn cut_with_seam_turned(turn: f64) {
    let mut built = Topology::new();
    let part = part(&mut built);
    let pin = make_cylinder(&mut built, PIN_R, 221.0).unwrap();
    transform_solid(
        &mut built,
        pin,
        &(Mat4::translation(PIN_X, 111.0, PIN_Z)
            * Mat4::rotation_x(FRAC_PI_2)
            * Mat4::rotation_z(turn.to_radians())),
    )
    .unwrap();
    let mut topo = Topology::new();
    let (part, pin) = (
        through_step(&built, part, &mut topo),
        through_step(&built, pin, &mut topo),
    );

    let before = mesh_fallback_count();
    let cut = boolean(&mut topo, BooleanOp::Cut, part, pin).unwrap();
    assert_eq!(mesh_fallback_count(), before, "turn {turn}: mesh fallback");
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "turn {turn}: {:?}", report.issues);
    let want = part_volume() - removed_volume();
    let volume = solid_volume(&topo, cut, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "turn {turn}: volume {volume}, want {want}"
    );
}

#[test]
fn a_pin_across_a_bosss_square_foot_cuts_exactly() {
    for turn in [0.0, 90.0, 180.0, 270.0] {
        cut_with_seam_turned(turn);
    }
}
