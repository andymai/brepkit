//! Point classification near a round rim: a ray's hit on a plane face is
//! read against the face's own lines and arcs, so a point just inside a
//! rim, between where chords of it would fall, reads inside.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::TAU;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

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

/// Both public classifiers read `p` inside when `inside`, outside when not.
fn assert_reads(
    topo: &Topology,
    solid: brepkit_topology::SolidId,
    p: Point3,
    inside: bool,
    label: &str,
) {
    let ops = classify_point(topo, solid, p, 0.01, 1e-7).unwrap();
    let want = if inside {
        PointClassification::Inside
    } else {
        PointClassification::Outside
    };
    assert_eq!(ops, want, "{label}: operations reads {ops:?}");
    let check = brepkit_check::classify::classify_point(
        topo,
        solid,
        p,
        &brepkit_check::classify::ClassifyOptions::default(),
    )
    .unwrap();
    let want = if inside {
        brepkit_check::classify::PointClassification::Inside
    } else {
        brepkit_check::classify::PointClassification::Outside
    };
    assert_eq!(check, want, "{label}: check reads {check:?}");
}

/// A unit cylinder 1 tall, upright, turned and mirrored: points 0.002 and
/// 0.0005 inside its wall, midway between where 32 chords of its rims would
/// fall, at five heights near and between its caps, read inside; points
/// 0.0005 outside read outside.
#[test]
fn points_just_inside_a_cylinders_rim_read_inside() {
    for (pose, place) in poses() {
        let mut topo = Topology::new();
        let cylinder = make_cylinder(&mut topo, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, cylinder, &place).unwrap();
        for (r, inside) in [(0.998_f64, true), (0.9995, true), (1.0005, false)] {
            for k in 0..32 {
                let t = (f64::from(k) + 0.5) * TAU / 32.0;
                for z in [0.02, 0.05, 0.5, 0.95, 0.98] {
                    let p = place.mul_point(Point3::new(r * t.cos(), r * t.sin(), z));
                    assert_reads(
                        &topo,
                        cylinder,
                        p,
                        inside,
                        &format!("{pose} r {r} t {t} z {z}"),
                    );
                }
            }
        }
    }
}

/// `make_sphere(3, 32)` less the slab `1 < z < 2`, upright, turned and
/// mirrored: every point of a grid inside the ball below and above the slab
/// reads inside, though rays from them cross the slab's discs close to
/// their rims.
#[test]
fn a_ball_less_a_slab_reads_inside_up_to_its_discs() {
    for (pose, place) in poses() {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
        let slab = make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
        transform_solid(&mut topo, slab, &Mat4::translation(-5.0, -5.0, 1.0)).unwrap();
        transform_solid(&mut topo, ball, &place).unwrap();
        transform_solid(&mut topo, slab, &place).unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, ball, slab).unwrap();
        for i in -12..=12 {
            for j in -12..=12 {
                for z in [0.5_f64, 0.9, 2.1, 2.5] {
                    let (x, y) = (f64::from(i) * 0.24, f64::from(j) * 0.24);
                    if x.mul_add(x, y.mul_add(y, z * z)) > 2.95 * 2.95 {
                        continue;
                    }
                    let p = place.mul_point(Point3::new(x, y, z));
                    assert_reads(&topo, cut, p, true, &format!("{pose} ({x}, {y}, {z})"));
                }
            }
        }
    }
}
