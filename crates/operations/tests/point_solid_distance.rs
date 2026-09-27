//! Point-to-solid distance measures a face as trimmed, not its whole
//! surface, and every shell of the solid, its cavities' included.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::distance::point_to_solid_distance;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::shell_op::shell;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::{SolidId, Topology};

type Make = fn(&mut Topology) -> SolidId;

/// A named solid, a point, and its true distance from the solid.
type Case = (&'static str, Make, (f64, f64, f64), f64);

/// `make_sphere(3, 32)` less the slab `1 < z < 2`.
fn ball_less_slab(topo: &mut Topology) -> SolidId {
    let ball = make_sphere(topo, 3.0, 32).unwrap();
    let slab = make_box(topo, 10.0, 10.0, 1.0).unwrap();
    transform_solid(topo, slab, &Mat4::translation(-5.0, -5.0, 1.0)).unwrap();
    boolean(topo, BooleanOp::Cut, ball, slab).unwrap()
}

/// Upright, turned and moved, and mirrored.
fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// Points whose nearest point on a face's whole surface lies off the face
/// (above a cylinder's rim, over a cone's small end, on the part of a
/// sphere a slab took, behind a half cylinder's flat), inside a solid near
/// its cap, and in a hollow ball's cavity: each distance, by both public
/// distances, matches the one to the solid as it is within `1e-6`.
#[test]
fn distance_reads_faces_as_trimmed() {
    let cases: [Case; 7] = [
        (
            "cylinder, above its rim",
            |t| make_cylinder(t, 5.0, 10.0).unwrap(),
            (-10.0, 0.0, 20.0),
            125.0_f64.sqrt(),
        ),
        (
            "cylinder, beside it",
            |t| make_cylinder(t, 5.0, 10.0).unwrap(),
            (-10.0, 0.0, 5.0),
            5.0,
        ),
        (
            "cylinder, inside near its top",
            |t| make_cylinder(t, 5.0, 10.0).unwrap(),
            (0.0, 0.0, 9.0),
            1.0,
        ),
        (
            "frustum, over its small end",
            |t| make_cone(t, 5.0, 2.0, 10.0).unwrap(),
            (0.0, 0.0, 13.0),
            3.0,
        ),
        (
            "ball less a slab, on the sphere the slab took",
            ball_less_slab,
            (6.75_f64.sqrt(), 0.0, 1.5),
            0.5,
        ),
        (
            "hollow ball, in its cavity",
            |t| {
                let ball = make_sphere(t, 5.0, 32).unwrap();
                shell(t, ball, 1.0, &[]).unwrap()
            },
            (0.0, 0.0, 1.0),
            3.0,
        ),
        (
            "half cylinder, behind its flat",
            |t| {
                let cylinder = make_cylinder(t, 5.0, 10.0).unwrap();
                let half = make_box(t, 20.0, 20.0, 20.0).unwrap();
                transform_solid(t, half, &Mat4::translation(0.0, -10.0, -5.0)).unwrap();
                boolean(t, BooleanOp::Cut, cylinder, half).unwrap()
            },
            (8.0, 0.0, 5.0),
            8.0,
        ),
    ];
    for (name, make, (x, y, z), truth) in cases {
        for (pose, place) in poses() {
            let label = format!("{name}, {pose}");
            let mut topo = Topology::new();
            let solid = make(&mut topo);
            transform_solid(&mut topo, solid, &place).unwrap();
            let p = place.mul_point(Point3::new(x, y, z));
            let ops = point_to_solid_distance(&topo, p, solid).unwrap().distance;
            let check = brepkit_check::distance::point_to_solid(&topo, p, solid)
                .unwrap()
                .distance;
            for (which, got) in [("operations", ops), ("check", check)] {
                assert!(
                    (got - truth).abs() < 1e-6 * truth.max(1.0),
                    "{label}: {which} reads {got}, truth {truth}"
                );
            }
        }
    }
}

/// A point on the part of the ball's sphere the slab took lies half a unit
/// from the solid: it reads outside, not on the boundary.
#[test]
fn a_point_on_a_removed_surface_is_not_on_the_boundary() {
    for (pose, place) in poses() {
        let mut topo = Topology::new();
        let solid = ball_less_slab(&mut topo);
        transform_solid(&mut topo, solid, &place).unwrap();
        let p = place.mul_point(Point3::new(6.75_f64.sqrt(), 0.0, 1.5));
        let class = classify_point(&topo, solid, p, 0.01, 1e-7).unwrap();
        assert_eq!(class, PointClassification::Outside, "{pose}");
    }
}
