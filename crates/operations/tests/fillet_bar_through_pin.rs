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
//! A pin only a little wider than the bar puts that opening closer behind
//! the corner than the round is wide, and a seam at `2` rad splits the
//! opening's top edge where it lies on the notch's side seen along the
//! axis. The result must not change when the whole part is turned or
//! mirrored.
//!
//! The round takes off, along the edge, the corner's strip outside the
//! round, out to the pin's wall: at a height `s` above the round's axis
//! side, `y = W - r + s`, the strip is `r - sqrt(r^2 - s^2)` deep and runs
//! from the wall, `x = sqrt(R^2 - y^2)`, out to the bar's end, `x = L`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Vec3;
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

/// The bar fused through a pin of radius `pin`, the pin turned `turn` about
/// its axis (which moves its seam).
fn bar_through_pin(topo: &mut Topology, pin: f64, turn: f64) -> SolidId {
    let bar = make_box(topo, 2.0 * L, 2.0 * W, H).unwrap();
    transform_solid(topo, bar, &Mat4::translation(-L, -W, 0.0)).unwrap();
    let pin = make_cylinder(topo, pin, H + 4.0).unwrap();
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

/// The volume the round of radius `r` takes off, out to a pin of radius
/// `pin`.
fn removed(r: f64, pin: f64) -> f64 {
    // Simpson's rule in `u`, with `s = r sin u`, where the integrand is
    // smooth.
    const N: usize = 2000;
    let f = |u: f64| {
        let s = r * u.sin();
        let depth = r * (1.0 - u.cos());
        let y = W - r + s;
        depth * (L - pin.mul_add(pin, -y * y).sqrt()) * r * u.cos()
    };
    let h = FRAC_PI_2 / N as f64;
    let mut sum = f(0.0) + f(FRAC_PI_2);
    for i in 1..N {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// A mirror through a plane of no special direction.
fn mirror() -> Mat4 {
    let n = Vec3::new(0.3, -0.5, 0.81).normalize().unwrap();
    let n = [n.x(), n.y(), n.z()];
    let through = 2.0 * (1.0 * n[0] + 2.0 * n[1] - 0.5 * n[2]);
    let mut m = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = if i == j { 1.0 } else { 0.0 } - 2.0 * n[i] * n[j];
        }
        m[i][3] = through * n[i];
    }
    m[3][3] = 1.0;
    Mat4(m)
}

#[test]
fn a_round_ending_on_a_pin_the_bar_runs_through_is_built() {
    let poses = [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(3.1, -1.7, 2.3)
                * Mat4::rotation_z(1.1)
                * Mat4::rotation_y(0.4)
                * Mat4::rotation_x(0.7),
        ),
        ("mirrored", mirror()),
    ];
    for pin in [5.0, 3.2] {
        for turn in [FRAC_PI_2, 2.0] {
            for r in [1.0, 2.0] {
                for (pose, matrix) in &poses {
                    let case = format!("pin {pin}, turn {turn}, r = {r}, {pose}");
                    let mut topo = Topology::new();
                    let solid = bar_through_pin(&mut topo, pin, turn);
                    let edge = arm_edge(&topo, solid);
                    transform_solid(&mut topo, solid, matrix).unwrap();
                    let before = solid_volume(&topo, solid, 1e-3).unwrap();
                    let result = fillet_v2(&mut topo, solid, &[edge], r)
                        .map_err(|e| format!("{case}: {e}"))
                        .unwrap();
                    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
                    let report = validate_solid(&topo, result.solid).unwrap();
                    assert!(report.is_valid(), "{case}: {:?}", report.issues);
                    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
                    assert!(is_watertight(&mesh), "{case}: mesh open");
                    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
                    let expected = removed(r, pin);
                    assert!(
                        (lost - expected).abs() <= 1e-6 * before,
                        "{case}: removed {lost}, the strip out to the pin {expected}"
                    );
                }
            }
        }
    }
}
