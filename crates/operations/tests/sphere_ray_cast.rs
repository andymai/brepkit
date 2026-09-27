//! The engine's ray cast reads a sphere face that planes do not bound (the
//! upper hemisphere less a quarter, left by a box corner at the ball's
//! centre) against its own wires, the ball's equator chords standing for the
//! great-circle arcs they project to.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_algo::FaceClass;
use brepkit_algo::classifier::{RayCastGeoms, classify_ray_cast_cached, ray_parity_cached};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// Points this close to a surface are not checked.
const NEAR: f64 = 0.02;

/// The ball of radius 3 less, and within, the box from `corner` to 10 past
/// it on every axis, turned, tilted and mirrored: every point of a grid over
/// the ball more than [`NEAR`] from both surfaces reads right, and so does
/// each axis ray from it that the vote does not discount, but for a ray
/// through the ball's equator plane between its chords and its circle, where
/// a plane face on the equator ends at the chords (the roadmap's
/// `make_sphere` quirk row).
#[test]
fn a_ball_less_a_box_corner_reads_by_its_wires() {
    let poses = [
        ("upright", Mat4::identity()),
        ("turned", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    for corner in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 0.5]] {
        for (op, name) in [(BooleanOp::Cut, "less"), (BooleanOp::Intersect, "within")] {
            for (pose_name, pose) in poses {
                let name = format!("ball {name} the box at {corner:?}, {pose_name}");
                let mut topo = Topology::new();
                let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
                let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                let at = Mat4::translation(corner[0], corner[1], corner[2]);
                transform_solid(&mut topo, block, &(pose * at)).unwrap();
                transform_solid(&mut topo, ball, &pose).unwrap();
                let result = boolean(&mut topo, op, ball, block).unwrap();
                assert!(
                    solid_faces(&topo, result).unwrap().iter().any(|&f| topo
                        .face(f)
                        .unwrap()
                        .surface()
                        .type_tag()
                        == "sphere"),
                    "{name}: fell back to a mesh"
                );
                reads_right(&name, &topo, result, pose, corner, op);
            }
        }
    }
}

fn reads_right(
    name: &str,
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    pose: Mat4,
    corner: [f64; 3],
    op: BooleanOp,
) {
    let geoms = RayCastGeoms::new(topo, solid).unwrap();
    let up =
        pose.mul_point(Point3::new(0.0, 0.0, 1.0)) - pose.mul_point(Point3::new(0.0, 0.0, 0.0));
    let sagitta = 3.0 * (std::f64::consts::PI / 32.0).cos();
    let past_the_chords = |p: Point3, dir: Vec3| {
        let along = dir.dot(up);
        let t = -Vec3::new(p.x(), p.y(), p.z()).dot(up) / along;
        along.abs() > 1e-12 && t > 0.0 && {
            let q = p + dir * t;
            let rho = Vec3::new(q.x(), q.y(), q.z()).length();
            rho > sagitta - NEAR && rho < 3.0 + NEAR
        }
    };
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut wrong = Vec::new();
    for i in 0..17 {
        for j in 0..17 {
            for k in 0..17 {
                let at = |n: i32| 7.0f64.mul_add(f64::from(n) / 16.0, -3.5);
                let local = Point3::new(at(i), at(j), at(k));
                let ball = Vec3::new(local.x(), local.y(), local.z()).length() - 3.0;
                let c = [local.x(), local.y(), local.z()];
                let block = (0..3).fold(f64::NEG_INFINITY, |d, a| d.max(corner[a] - c[a]));
                if ball.abs() < NEAR || block.abs() < NEAR {
                    continue;
                }
                let expected = if op == BooleanOp::Cut {
                    ball < 0.0 && block > 0.0
                } else {
                    ball < 0.0 && block < 0.0
                };
                let p = pose.mul_point(local);
                let got = classify_ray_cast_cached(&geoms, p).unwrap() == FaceClass::Inside;
                let ray_misreads = axes.iter().any(|&dir| {
                    let (odd, discounted) = ray_parity_cached(&geoms, p, dir);
                    !discounted && !past_the_chords(p, dir) && odd != expected
                });
                if got != expected || ray_misreads {
                    wrong.push(local);
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{name}: {} misread, first {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(4)]
    );
}
