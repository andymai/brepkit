//! A ball of radius 3 against a block whose faces `y = 0` and `z = 1` cut
//! the ball: the block holds half of the ball's cap above `z = 1`. The
//! section of the face `y = 0` on the ball runs through the ball's pole,
//! along two meridians a half turn apart, and one of them is the ball's
//! seam. The piece's loop in the ball's `(u, v)` passes the pole there, and
//! the nearest way round between the two meridians was a matter of the last
//! bit: in a turned pose the loop was read round the other half of the cap,
//! the piece and the rest of the ball each took a point of the other, and
//! the boolean fell back to a mesh.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn poses() -> [(&'static str, Mat4); 6] {
    [
        ("id", Mat4::identity()),
        ("rz0.5", Mat4::rotation_z(0.5)),
        ("rx0.35", Mat4::rotation_x(0.35)),
        ("ry0.3rz1", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mx", Mat4::scale(-1.0, 1.0, 1.0)),
        (
            "moved",
            Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
        ),
    ]
}

/// `(op, block first, faces, volume, a point inside, a point outside)`.
type Row = (BooleanOp, bool, usize, f64, [f64; 3], [f64; 3]);

fn rows() -> [Row; 3] {
    // Half of the cap of height 2 of a ball of radius 3.
    let within = 14.0 * PI / 3.0;
    [
        (
            BooleanOp::Cut,
            false,
            3,
            36.0 * PI - within,
            [0.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
        ),
        (
            BooleanOp::Intersect,
            false,
            3,
            within,
            [0.0, 1.0, 2.0],
            [0.0, -1.0, 2.0],
        ),
        (
            BooleanOp::Cut,
            true,
            7,
            200.0 - within,
            [4.0, 4.0, 4.0],
            [0.0, 1.0, 2.0],
        ),
    ]
}

#[test]
fn a_block_takes_half_a_cap_of_a_ball_in_each_pose() {
    // The block goes to its place in one move, and in two: the two ways
    // differ in the last bits of the block's coordinates.
    for ((pose, m), one_move) in poses().into_iter().flat_map(|p| [(p, true), (p, false)]) {
        for (op, swap, faces, volume, inside, outside) in rows() {
            let label = format!("{pose} {op:?} swap={swap} one_move={one_move}");
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
            transform_solid(&mut topo, ball, &m).unwrap();
            let block = make_box(&mut topo, 10.0, 5.0, 4.0).unwrap();
            let lift = Mat4::translation(-5.0, 0.0, 1.0);
            if one_move {
                transform_solid(&mut topo, block, &(m * lift)).unwrap();
            } else {
                transform_solid(&mut topo, block, &lift).unwrap();
                transform_solid(&mut topo, block, &m).unwrap();
            }
            let (x, y) = if swap { (block, ball) } else { (ball, block) };
            let opts = BooleanOptions {
                mesh_fallback: false,
                ..BooleanOptions::default()
            };
            let result = boolean_with_options(&mut topo, op, x, y, opts)
                .unwrap_or_else(|e| panic!("{label}: {e}"));

            assert_eq!(
                solid_faces(&topo, result).unwrap().len(),
                faces,
                "{label}: faces"
            );
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: valid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: watertight");
            let got = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (got - volume).abs() < 1e-9 * volume,
                "{label}: volume {got} against {volume}"
            );
            let at = |p: [f64; 3]| m.mul_point(Point3::new(p[0], p[1], p[2]));
            assert_eq!(
                classify_point(&topo, result, at(inside), 0.01, 1e-7).unwrap(),
                PointClassification::Inside,
                "{label}: inside point"
            );
            assert_eq!(
                classify_point(&topo, result, at(outside), 0.01, 1e-7).unwrap(),
                PointClassification::Outside,
                "{label}: outside point"
            );
        }
    }
}
