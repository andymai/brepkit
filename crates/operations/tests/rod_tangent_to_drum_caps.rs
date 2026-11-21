//! A rod standing on a drum, as wide as the drum is long.
//!
//! The drum is a cylinder of radius 8 along `x`, from `x = -6` to `6`; the
//! rod a cylinder of radius 6 along `z`, from `z = 4` (inside the drum) to
//! `20`. The rod's wall is tangent to both of the drum's end caps, along
//! the lines `x = ±6, y = 0`, and the two walls meet in one closed curve
//! that touches both of the drum's rims, at `(±6, 0, 8)`.
//!
//! Two things sent it to the mesh fallback when the rod's seam lies on a
//! tangent line, as `make_cylinder` puts it unturned. The drum's end plane
//! and the rod's floor plane meet in a line that touches the floor's disc
//! at one point, and the two roots of that tangency, a rounding apart, read
//! as a run of the line inside the disc, so the line was kept across the
//! whole cap. And the walls' closed section, started on the rod's seam,
//! starts on the drum's rim, so the drum's wall did not take it as a hole.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const DRUM_R: f64 = 8.0;
const DRUM_HALF: f64 = 6.0;
const ROD_R: f64 = 6.0;
const ROD_FOOT: f64 = 4.0;
const ROD_LEN: f64 = 16.0;

/// The volume the rod and the drum share: over the rod's disc, from its
/// foot up to the drum's wall.
fn shared() -> f64 {
    const STEPS: usize = 20_000;
    let f = |y: f64| {
        2.0 * (ROD_R * ROD_R - y * y).max(0.0).sqrt()
            * ((DRUM_R * DRUM_R - y * y).sqrt() - ROD_FOOT)
    };
    #[allow(clippy::cast_precision_loss)]
    let h = 2.0 * ROD_R / STEPS as f64;
    let mut sum = f(-ROD_R) + f(ROD_R);
    for k in 1..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let y = -ROD_R + h * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(y);
    }
    sum * h / 3.0
}

fn run(op: BooleanOp, turn: f64) {
    let mut topo = Topology::new();
    let drum = make_cylinder(&mut topo, DRUM_R, 2.0 * DRUM_HALF).unwrap();
    transform_solid(&mut topo, drum, &Mat4::rotation_y(FRAC_PI_2)).unwrap();
    transform_solid(&mut topo, drum, &Mat4::translation(-DRUM_HALF, 0.0, 0.0)).unwrap();
    let rod = make_cylinder(&mut topo, ROD_R, ROD_LEN).unwrap();
    transform_solid(&mut topo, rod, &Mat4::rotation_z(turn)).unwrap();
    transform_solid(&mut topo, rod, &Mat4::translation(0.0, 0.0, ROD_FOOT)).unwrap();

    let result = boolean(&mut topo, op, drum, rod).unwrap();
    // The exact result has the walls and caps (5 faces for the fuse and
    // the cut, 3 for the intersection); the mesh fallback has dozens.
    let faces = solid_faces(&topo, result).unwrap().len();
    assert!(
        faces <= 5,
        "{op:?} turn {turn}: {faces} faces, not the exact walls and caps"
    );
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{op:?} turn {turn}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "{op:?} turn {turn}: mesh not watertight"
    );

    let drum_v = PI * DRUM_R * DRUM_R * 2.0 * DRUM_HALF;
    let rod_v = PI * ROD_R * ROD_R * ROD_LEN;
    let want = match op {
        BooleanOp::Fuse => drum_v + rod_v - shared(),
        BooleanOp::Cut => drum_v - shared(),
        BooleanOp::Intersect => shared(),
    };
    let got = solid_volume(&topo, result, 1e-3).unwrap();
    assert!(
        (got - want).abs() <= 1e-4 * want,
        "{op:?} turn {turn}: volume {got}, want {want}"
    );
}

#[test]
fn a_rod_tangent_to_a_drum_s_caps_stays_exact() {
    for op in [BooleanOp::Fuse, BooleanOp::Cut, BooleanOp::Intersect] {
        for turn in [0.0, 1.0, FRAC_PI_2, PI] {
            run(op, turn);
        }
    }
}
