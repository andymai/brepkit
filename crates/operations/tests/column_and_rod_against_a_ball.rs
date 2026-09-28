//! A square column fused with a thin rod beside it, against a ball wider
//! than the column: the ball bulges through the column's four sides and the
//! rod runs through the ball's side. Upright and with the ball turned, each
//! operation is exact, valid and watertight, and holds the integrated volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const R: f64 = 3.0;

/// The volume a rod of radius `r` along `z` through `(x, y)` shares with the
/// ball: the ball's chord through each point of the rod's disc, integrated
/// over the disc.
fn rod_in_ball(x: f64, y: f64, r: f64) -> f64 {
    let n = 400;
    let (ds, dt) = (r / f64::from(n), 2.0 * PI / f64::from(n));
    let mut total = 0.0;
    for i in 0..n {
        let s = (f64::from(i) + 0.5) * ds;
        for j in 0..n {
            let (sin, cos) = ((f64::from(j) + 0.5) * dt).sin_cos();
            let (px, py) = (s.mul_add(cos, x), s.mul_add(sin, y));
            let q = px.mul_add(-px, py.mul_add(-py, R * R));
            if q > 0.0 {
                total += 2.0 * q.sqrt() * s;
            }
        }
    }
    total * ds * dt
}

#[test]
fn a_column_and_rod_against_a_ball_are_exact() {
    let ball = 4.0 / 3.0 * PI * R.powi(3);
    // The column's sides stand 2.5 from the ball's centre, each cutting off
    // a cap 0.5 high.
    let caps = 4.0 * PI * 0.25 * (3.0f64.mul_add(R, -0.5)) / 3.0;
    for (r, (x, y), turned) in [(0.1, (2.75, 0.0), true), (0.15, (-2.7, -0.9), false)] {
        let tool = 250.0 + PI * r * r * 10.0;
        let shared = ball - caps + rod_in_ball(x, y, r);
        for (op, ball_first, truth) in [
            (BooleanOp::Cut, false, tool - shared),
            (BooleanOp::Cut, true, ball - shared),
            (BooleanOp::Intersect, false, shared),
        ] {
            let label =
                format!("rod {r} at ({x}, {y}), turned {turned}, {op:?} ball first {ball_first}");
            let mut topo = Topology::new();
            let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
            transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
            let rod = make_cylinder(&mut topo, r, 10.0).unwrap();
            transform_solid(&mut topo, rod, &Mat4::translation(x, y, -5.0)).unwrap();
            let tool = boolean(&mut topo, BooleanOp::Fuse, column, rod).unwrap();
            let sphere = make_sphere(&mut topo, R, 32).unwrap();
            if turned {
                let turn = Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3);
                transform_solid(&mut topo, sphere, &turn).unwrap();
            }
            let (a, b) = if ball_first {
                (sphere, tool)
            } else {
                (tool, sphere)
            };
            let result = boolean(&mut topo, op, a, b).unwrap();
            let faces = solid_faces(&topo, result).unwrap();
            assert!(faces.len() <= 14, "{label}: {} faces", faces.len());
            let tags: Vec<&str> = faces
                .iter()
                .map(|&f| topo.face(f).unwrap().surface().type_tag())
                .collect();
            assert!(
                tags.contains(&"sphere") && tags.contains(&"cylinder") && !tags.contains(&"nurbs"),
                "{label}: surfaces {tags:?}"
            );
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(!mesh.indices.is_empty(), "{label}: no triangles");
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-4,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}
