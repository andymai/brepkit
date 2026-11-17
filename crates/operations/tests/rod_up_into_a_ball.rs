//! A rod standing below a turned ball and ending inside it, off its axis:
//! the end cap crosses the ball, so the rod's wall meets the ball in an arch
//! with both ends on the cap's rim, and the arch crosses the wall's seam. In
//! every pose each operation is exact, valid and watertight, and holds the
//! volume integrated over the rod's cross-section.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const CENTRE: (f64, f64) = (-0.104_759_314_427_086_86, -0.441_570_733_074_779_4);
const RADIUS: f64 = 0.295_773_971_926_665_66;
const BOTTOM: f64 = -2.0;
const TOP: f64 = -0.844_130_743_216_372_9;

/// The volume the rod and the unit ball share: over the rod's cross-section
/// in polar steps about its axis, the height from the ball's floor up to
/// the rod's top.
fn shared() -> f64 {
    let n = 1000;
    let (dr, dt) = (RADIUS / f64::from(n), TAU / f64::from(4 * n));
    let mut total = 0.0;
    for i in 0..n {
        let r = dr * (f64::from(i) + 0.5);
        for j in 0..4 * n {
            let (sin, cos) = (dt * (f64::from(j) + 0.5)).sin_cos();
            let (x, y) = (r.mul_add(cos, CENTRE.0), r.mul_add(sin, CENTRE.1));
            let floor = -(1.0 - x.mul_add(x, y * y)).max(0.0).sqrt();
            total += (TOP - floor).max(0.0) * r * dr * dt;
        }
    }
    total
}

#[test]
fn a_rod_ending_inside_a_ball_is_exact() {
    let ball = 4.0 * PI / 3.0;
    let rod = PI * RADIUS * RADIUS * (TOP - BOTTOM);
    let both = shared();
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
    let poses = [
        ("upright", Mat4::identity()),
        ("turned", turn),
        ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ("turned and mirrored", Mat4::scale(1.0, -1.0, 1.0) * turn),
        ("rolled", Mat4::rotation_z(2.0)),
    ];
    for (pose, m) in poses {
        for (op, truth) in [
            (BooleanOp::Cut, ball - both),
            (BooleanOp::Intersect, both),
            (BooleanOp::Fuse, ball + rod - both),
        ] {
            let label = format!("{pose} {op:?}");
            let mut topo = Topology::new();
            let a = make_sphere(&mut topo, 1.0, 32).unwrap();
            let spin = Mat4::rotation_z(4.404_457_174_719_163)
                * Mat4::rotation_y(2.492_139_368_054_709_7)
                * Mat4::rotation_x(1.142_226_672_810_132_2);
            transform_solid(&mut topo, a, &spin).unwrap();
            let b = make_cylinder(&mut topo, RADIUS, TOP - BOTTOM).unwrap();
            let place = Mat4::translation(CENTRE.0, CENTRE.1, BOTTOM)
                * Mat4::rotation_z(1.931_530_959_468_46);
            transform_solid(&mut topo, b, &place).unwrap();
            transform_solid(&mut topo, a, &m).unwrap();
            transform_solid(&mut topo, b, &m).unwrap();
            let opts = BooleanOptions {
                mesh_fallback: false,
                ..BooleanOptions::default()
            };
            let result = boolean_with_options(&mut topo, op, a, b, opts);
            let result = result.expect(&label);
            let tags: Vec<&str> = solid_faces(&topo, result)
                .unwrap()
                .iter()
                .map(|&f| topo.face(f).unwrap().surface().type_tag())
                .collect();
            assert!(
                tags.len() <= 6 && tags.contains(&"sphere") && tags.contains(&"cylinder"),
                "{label}: surfaces {tags:?}"
            );
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.0005).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}
