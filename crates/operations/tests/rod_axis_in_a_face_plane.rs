//! A rod along `y` whose axis lies in the plane of a face of the solid it cuts
//! and crosses that face's edge, as a hinge lid's knuckle clearance bores do
//! on its pocket ceiling. Every result is exact, valid and measures its closed
//! form, upright, turned and mirrored, whichever way the rod's seam points.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The area of a disc of radius `r` centred at `x = cx` lying where `x > 0`.
fn disc_past_zero(r: f64, cx: f64) -> f64 {
    let d = (-cx / r).clamp(-1.0, 1.0);
    r * r * (d.acos() - d * (1.0 - d * d).sqrt())
}

/// A rod of radius `r` along `y` from `y0` to `y1`, its axis through
/// `(cx, ., cz)`, turned `spin` about that axis.
fn rod(topo: &mut Topology, r: f64, cx: f64, cz: f64, y0: f64, y1: f64, spin: f64) -> SolidId {
    let rod = make_cylinder(topo, r, y1 - y0).unwrap();
    let place =
        Mat4::translation(cx, y0, cz) * Mat4::rotation_x(-FRAC_PI_2) * Mat4::rotation_z(spin);
    transform_solid(topo, rod, &place).unwrap();
    rod
}

/// The poses each scene is checked in: upright, turned about two axes, and
/// mirrored through `x = 0`, each with the rod turned about its own axis to
/// eighths of a turn.
fn poses() -> [Mat4; 3] {
    [
        Mat4::identity(),
        Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4),
        Mat4::scale(-1.0, 1.0, 1.0),
    ]
}

fn spins(pose: Mat4) -> impl Iterator<Item = (Mat4, f64)> {
    (0..8).map(move |k| (pose, f64::from(k) * std::f64::consts::FRAC_PI_4))
}

/// Cuts `tool` from `base` in `pose`, requiring an exact valid result of
/// volume `expected`.
fn cut_is_exact(topo: &mut Topology, base: SolidId, tool: SolidId, pose: &Mat4, expected: f64) {
    transform_solid(topo, base, pose).unwrap();
    transform_solid(topo, tool, pose).unwrap();
    let before = mesh_fallback_count();
    let cut = boolean(topo, BooleanOp::Cut, base, tool).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let vol = solid_volume(topo, cut, 0.001).unwrap();
    assert!(
        (vol - expected).abs() < 1e-6 * expected,
        "volume {vol}, expected {expected}"
    );
}

/// A plate whose top face has a rounded corner, cut by a rod whose axis lies
/// in the top face's plane and crosses its edge. Each rod cap meets the top
/// face in a line through the cap's centre, inside both faces for a unit of
/// its length: shorter than the step at which the section filter sampled the
/// line across both faces' boxes, and with an arc on each outline the exact
/// clip declined, so the line was dropped and the top face never split.
#[test]
fn a_rod_whose_axis_lies_in_a_rounded_plates_top_is_exact() {
    let (r, cx) = (2.0, -1.0);
    let notch = std::f64::consts::PI * 9.0 / 4.0 * 2.0;
    let removed = 4.0 * disc_past_zero(r, cx) / 2.0;
    for (pose, spin) in poses().into_iter().flat_map(spins) {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 20.0, 20.0, 2.0).unwrap();
        transform_solid(&mut topo, plate, &Mat4::translation(0.0, 0.0, -2.0)).unwrap();
        let corner = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        transform_solid(&mut topo, corner, &Mat4::translation(20.0, 20.0, -3.0)).unwrap();
        let base = boolean(&mut topo, BooleanOp::Cut, plate, corner).unwrap();
        let tool = rod(&mut topo, r, cx, 0.0, 4.0, 8.0, spin);
        cut_is_exact(&mut topo, base, tool, &pose, 800.0 - notch - removed);
    }
}
