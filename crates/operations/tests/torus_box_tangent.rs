//! A box whose face touches a torus at one point on its outer equator: the
//! torus piece inside the box is sampled at that point, on the box, and its
//! other samples have to stay on the piece (whose loops run across the
//! tube's seam) for it to read as inside. And a box whose face touches the
//! tube's inner side, where the face meets the torus in a figure-eight.

#![allow(clippy::unwrap_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// Booleans of the torus (major radius 3, minor 1) with a box spanning
/// x from `x0` to `x1`, y from -1 to 1 and z from -2 to 2.
fn run(op: BooleanOp, x0: f64, x1: f64) -> (Vec<&'static str>, f64) {
    let mut topo = Topology::default();
    let torus = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();
    let block = make_box(&mut topo, x1 - x0, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(x0, -1.0, -2.0)).unwrap();
    let result = boolean(&mut topo, op, torus, block).unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    let kinds = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect();
    (kinds, solid_volume(&topo, result, 0.002).unwrap())
}

#[test]
fn a_box_touching_the_outer_equator_cuts_exactly() {
    // The box reaching past x = 4 holds the same piece of the torus, so the
    // touching box's results differ from it only by the box's own volume.
    let past = 0.1 * 2.0 * 4.0;
    for (op, extra) in [
        (BooleanOp::Intersect, 0.0),
        (BooleanOp::Cut, 0.0),
        (BooleanOp::Fuse, -past),
    ] {
        let (kinds, touching) = run(op, 2.5, 4.0);
        let (_, beyond) = run(op, 2.5, 4.1);
        assert!(
            kinds.contains(&"torus") && kinds.len() < 20,
            "{op:?}: {} faces {kinds:?}",
            kinds.len()
        );
        assert!(
            (touching - (beyond + extra)).abs() < 1e-6 * beyond,
            "{op:?}: volume {touching} against {}",
            beyond + extra
        );
    }
}

/// The torus's volume past x = 2 within |y| <= 1. At a distance `rho` from
/// the axis the tube stands `2 sqrt(1 - (rho - 3)^2)` tall, and the box holds
/// the angles about the axis up to the lesser of `acos(2 / rho)` and
/// `asin(1 / rho)`, which meet at `rho = sqrt(5)`. With `rho = 3 + sin(phi)`
/// the integrand is smooth on either side of that.
fn past_the_inner_face() -> f64 {
    let f = |phi: f64| {
        let rho = 3.0 + phi.sin();
        let angle = (2.0 / rho).min(1.0).acos().min((1.0 / rho).min(1.0).asin());
        4.0 * phi.cos().powi(2) * angle * rho
    };
    let simpson = |a: f64, b: f64| {
        let n = 20_000;
        let h = (b - a) / f64::from(n);
        let inner: f64 = (1..n)
            .map(|i| f64::from(2 + 2 * (i % 2)) * f(h.mul_add(f64::from(i), a)))
            .sum();
        (f(a) + f(b) + inner) * h / 3.0
    };
    let kink = (5.0_f64.sqrt() - 3.0).asin();
    simpson(-FRAC_PI_2, kink) + simpson(kink, FRAC_PI_2)
}

#[test]
fn a_box_face_on_the_inner_equator_cuts_exactly() {
    // The face x = 2 = R - r touches the tube's inner side at (2, 0, 0), a
    // saddle, and meets the torus in a figure-eight crossing itself there.
    // The box to x = 4 touches the outer equator too.
    let shared = past_the_inner_face();
    let torus = 2.0 * PI * PI * 3.0;
    for x1 in [4.0, 4.5] {
        let block = (x1 - 2.0) * 2.0 * 4.0;
        for (op, want) in [
            (BooleanOp::Intersect, shared),
            (BooleanOp::Fuse, torus + block - shared),
        ] {
            let (kinds, got) = run(op, 2.0, x1);
            assert!(
                kinds.contains(&"torus") && kinds.len() < 20,
                "{op:?} to x = {x1}: {} faces {kinds:?}",
                kinds.len()
            );
            assert!(
                (got - want).abs() < 1e-6 * want,
                "{op:?} to x = {x1}: volume {got} against {want}"
            );
        }
    }
}
