//! A thin ring (radii 4 and 0.1) and a box 2 wide across it, whose floor
//! is a plane through the ring's centre turned a little about the box's
//! long axis. The floor meets the tube in two loops a tube's diameter (0.2)
//! apart, and each of the floor's long edges crosses both loops. A crossing
//! lies on one loop only: taken on the other loop too, it split that loop
//! where it does not run, and the boolean fell back to a mesh.
//!
//! A half turn about the box's long axis maps the ring onto itself, the
//! box's slab onto itself and the two sides of the floor onto each other,
//! so the box holds half of the ring's part inside the slab.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

const MAJOR: f64 = 4.0;
const MINOR: f64 = 0.1;
const HALF_WIDTH: f64 = 1.0;

/// The ring's part inside the slab `|x| <= HALF_WIDTH`: at the angle `phi`
/// round the tube, the ring's radius is `rho = MAJOR + MINOR sin(phi)`, the
/// slab holds `2 (pi - 2 acos(HALF_WIDTH / rho))` of that circle, and the
/// tube is `2 MINOR cos(phi)` high.
fn ring_in_slab() -> f64 {
    let n = 20_000;
    let h = PI / f64::from(n);
    let f = |phi: f64| {
        let rho = MINOR.mul_add(phi.sin(), MAJOR);
        let turn = 2.0 * 2.0f64.mul_add(-(HALF_WIDTH / rho).acos(), PI);
        rho * turn * 2.0 * MINOR * MINOR * phi.cos() * phi.cos()
    };
    let a = -0.5 * PI;
    let mut sum = f(a) + f(a + PI);
    for i in 1..n {
        sum += f(h.mul_add(f64::from(i), a)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn check(tilt: f64, op: BooleanOp, pose: &Mat4, label: &str) {
    let label = format!("{label}, tilt {tilt}, {op:?}");
    let mut topo = Topology::new();
    let ring = make_torus(&mut topo, MAJOR, MINOR, 32).unwrap();
    let lid = make_box(&mut topo, 2.0 * HALF_WIDTH, 20.0, 10.0).unwrap();
    transform_solid(
        &mut topo,
        lid,
        &(Mat4::rotation_x(tilt) * Mat4::translation(-HALF_WIDTH, -10.0, 0.0)),
    )
    .unwrap();
    for solid in [ring, lid] {
        transform_solid(&mut topo, solid, pose).unwrap();
    }
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let piece = boolean_with_options(&mut topo, op, ring, lid, exact)
        .unwrap_or_else(|e| panic!("{label}: {e}"));

    let report = validate_solid(&topo, piece).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, piece, 0.001).unwrap()),
        "{label}: mesh not watertight"
    );
    let in_box = 0.5 * ring_in_slab();
    let want = match op {
        BooleanOp::Intersect => in_box,
        _ => 2.0 * PI * PI * MAJOR * MINOR * MINOR - in_box,
    };
    let volume = solid_volume(&topo, piece, 0.0001).unwrap();
    assert!(
        (volume - want).abs() <= 1e-6 * want,
        "{label}: volume {volume}, want {want}"
    );
}

#[test]
fn a_tilted_floor_through_a_thin_rings_centre_takes_half_its_slab() {
    let turned =
        Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    for tilt in [0.0, 0.005, 0.01, 0.02] {
        for op in [BooleanOp::Cut, BooleanOp::Intersect] {
            for (pose, name) in [(Mat4::identity(), "upright"), (turned, "turned")] {
                check(tilt, op, &pose, name);
            }
        }
    }
}
