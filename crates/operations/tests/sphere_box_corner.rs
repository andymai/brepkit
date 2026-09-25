//! A ball less a box whose corner pokes into it keeps a three-sided pocket:
//! the hemisphere the corner reaches gets a hole bounded by three arcs, which
//! must wind against the hemisphere's outer wire. Each piece is an exact,
//! valid, watertight solid whose volume matches the corner's integral, above
//! or below the equator and with both solids turned. A turned ball keeps its
//! exact cuts by a plane and a rod too.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

const RADIUS: f64 = 3.0;

/// The ball's piece past `x = a`, `y = b` and `z = c`: across `y` the height
/// `sqrt(R² - x² - y²) - c` integrates in closed form, leaving a Simpson
/// integral in `x`, substituted `x = end - s²` where the piece pinches off.
fn corner_piece(a: f64, b: f64, c: f64) -> f64 {
    let x_end = (RADIUS * RADIUS - b * b - c * c).sqrt();
    let across = |x: f64| {
        let c2 = RADIUS.mul_add(RADIUS, -(x * x));
        let y_end = (c2 - c * c).max(0.0).sqrt();
        let g = |y: f64| {
            0.5 * y.mul_add(
                y.mul_add(-y, c2).max(0.0).sqrt(),
                c2 * (y / c2.sqrt()).asin(),
            ) - c * y
        };
        g(y_end) - g(b)
    };
    let f = |s: f64| across(s.mul_add(-s, x_end)) * 2.0 * s;
    let (n, span) = (800_u32, (x_end - a).sqrt());
    let step = span / f64::from(n);
    let mut sum = f(0.0) + f(span);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step * f64::from(k));
    }
    sum * step / 3.0
}

#[test]
fn ball_less_a_box_corner() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    // A box of side 10 with its near corner at (a, b, c) reaching into the
    // ball, below the equator when `flip` mirrors it through z = 0, and the
    // whole scene turned by `turn`.
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4);
    for (corner, flip, turned) in [
        ((1.0, 1.0, 1.0), false, false),
        ((1.0, 1.2, 0.8), false, false),
        ((0.5, 0.5, 0.5), false, false),
        ((1.5, 0.2, 0.3), false, false),
        ((1.0, 1.2, 0.8), true, false),
        ((1.0, 1.2, 0.8), false, true),
    ] {
        let (a, b, c) = corner;
        let piece = corner_piece(a, b, c);
        let z0 = if flip { -c - 10.0 } else { c };
        let place = |m: Mat4| if turned { turn * m } else { m };
        for (op, truth) in [
            (BooleanOp::Cut, ball - piece),
            (BooleanOp::Intersect, piece),
        ] {
            let label = format!("{corner:?} flip {flip} turned {turned} {op:?}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, block, &place(Mat4::translation(a, b, z0))).unwrap();
            if turned {
                transform_solid(&mut topo, sphere, &turn).unwrap();
            }
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, op, sphere, block).unwrap();
            assert_eq!(
                mesh_fallback_count(),
                before,
                "{label}: fell back to a mesh"
            );
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            // Just inside the corner, and on the ball's far side.
            let z_in = if flip { -c - 0.05 } else { c + 0.05 };
            let lift = |p: Point3| if turned { turn.mul_point(p) } else { p };
            let at = |p: Point3| {
                classify_point(&topo, result, lift(p), &ClassifyOptions::default()).unwrap()
            };
            let (pocket, far) = if op == BooleanOp::Cut {
                (PointClassification::Outside, PointClassification::Inside)
            } else {
                (PointClassification::Inside, PointClassification::Outside)
            };
            assert_eq!(
                at(Point3::new(a + 0.05, b + 0.05, z_in)),
                pocket,
                "{label}: corner"
            );
            assert_eq!(at(Point3::new(-2.0, -1.0, 0.3)), far, "{label}: far side");
        }
    }
}

/// The ball above `z = 1` (a cap `h = 2` high, `pi h² (3R - h) / 3`) and the
/// ball less a rod of radius 0.5 along `z` through `(1, 0.5)` (the chord
/// through the ball over the rod's disc, by Simpson in polar coordinates),
/// with ball and tool turned together: the pieces stay exact and keep their
/// volumes however the ball's axis points.
#[test]
fn turned_ball_keeps_exact_cuts() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let cap = PI * 4.0 * (3.0 * RADIUS - 2.0) / 3.0;
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let column = simpson(200, 0.0, 0.5, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), 1.0), r.mul_add(th.sin(), 0.5));
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    for (name, turn) in [
        ("about x", Mat4::rotation_x(0.4)),
        ("about y", Mat4::rotation_y(1.1)),
        (
            "about z then x",
            Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4),
        ),
    ] {
        for (tool, op, truth) in [
            ("cap", BooleanOp::Intersect, cap),
            ("rod", BooleanOp::Cut, ball - column),
        ] {
            let label = format!("{tool} turned {name}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let cutter = if tool == "cap" {
                let b = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
                transform_solid(&mut topo, b, &(turn * Mat4::translation(-10.0, -10.0, 1.0)))
                    .unwrap();
                b
            } else {
                let c = make_cylinder(&mut topo, 0.5, 10.0).unwrap();
                transform_solid(&mut topo, c, &(turn * Mat4::translation(1.0, 0.5, -5.0))).unwrap();
                c
            };
            transform_solid(&mut topo, sphere, &turn).unwrap();
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, op, sphere, cutter).unwrap();
            assert_eq!(
                mesh_fallback_count(),
                before,
                "{label}: fell back to a mesh"
            );
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}
