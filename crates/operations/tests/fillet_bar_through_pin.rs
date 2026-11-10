//! A round along a bar that runs through a wider pin, ending on the pin's
//! wall.
//!
//! The bar, `2 L` long, `2 W` wide and `H` high, runs along `x` through a
//! pin of radius `R > W` standing up `z` at the origin, and is fused with
//! it. An edge where the bar's top meets its side `y = W` runs from the
//! bar's end to the pin's wall, which is neither square to it nor flat:
//! the round runs on to the wall and notches it.
//!
//! The notch is checked against the wall's other boundaries, seen along the
//! round's axis. The bar's opening on the far side of the pin lies straight
//! behind the corner along that axis, `2 sqrt(R^2 - W^2)` away, and so
//! seen along it reaches into the notch; it is not in the notch's way.
//!
//! The round takes off, along the edge, the corner's strip outside the
//! round, out to the pin's wall: at a height `s` above the round's axis
//! side, `y = W - r + s`, the strip is `r - sqrt(r^2 - s^2)` deep and runs
//! from the wall, `x = sqrt(R^2 - y^2)`, out to the bar's end, `x = L`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const L: f64 = 20.0;
const W: f64 = 3.0;
const H: f64 = 4.0;
const R: f64 = 5.0;

/// The bar fused through the pin, the pin turned `turn` about its axis
/// (which moves its seam).
fn bar_through_pin(topo: &mut Topology, turn: f64) -> SolidId {
    let bar = make_box(topo, 2.0 * L, 2.0 * W, H).unwrap();
    transform_solid(topo, bar, &Mat4::translation(-L, -W, 0.0)).unwrap();
    let pin = make_cylinder(topo, R, H + 4.0).unwrap();
    transform_solid(
        topo,
        pin,
        &(Mat4::translation(0.0, 0.0, -2.0) * Mat4::rotation_z(turn)),
    )
    .unwrap();
    boolean(topo, BooleanOp::Fuse, bar, pin).unwrap()
}

/// The edge along the top of the bar's side `y = W` on the `+x` arm.
fn arm_edge(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            matches!(edge.curve(), EdgeCurve::Line)
                && [p, q]
                    .iter()
                    .all(|p| (p.y() - W).abs() < 1e-9 && (p.z() - H).abs() < 1e-9 && p.x() > 0.0)
        })
        .expect("the arm's edge")
}

/// The volume the round of radius `r` takes off.
fn removed(r: f64) -> f64 {
    // Simpson's rule in `u`, with `s = r sin u`, where the integrand is
    // smooth.
    const N: usize = 2000;
    let f = |u: f64| {
        let s = r * u.sin();
        let depth = r * (1.0 - u.cos());
        let y = W - r + s;
        depth * (L - R.mul_add(R, -y * y).sqrt()) * r * u.cos()
    };
    let h = FRAC_PI_2 / N as f64;
    let mut sum = f(0.0) + f(FRAC_PI_2);
    for i in 1..N {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

#[test]
fn a_round_ending_on_a_pin_the_bar_runs_through_is_built() {
    for turn in [FRAC_PI_2, 2.0] {
        for r in [1.0, 2.0] {
            let mut topo = Topology::new();
            let solid = bar_through_pin(&mut topo, turn);
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let edge = arm_edge(&topo, solid);
            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            assert!(
                result.failed.is_empty(),
                "turn {turn}, r = {r}: {:?}",
                result.failed
            );
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(
                report.is_valid(),
                "turn {turn}, r = {r}: {:?}",
                report.issues
            );
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "turn {turn}, r = {r}: mesh open");
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            let expected = removed(r);
            assert!(
                (lost - expected).abs() <= 1e-6 * before,
                "turn {turn}, r = {r}: removed {lost}, the strip out to the pin {expected}"
            );
        }
    }
}
