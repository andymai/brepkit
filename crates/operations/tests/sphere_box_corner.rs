//! A ball less a box whose corner pokes into it keeps a three-sided pocket:
//! the hemisphere the corner reaches gets a hole bounded by three arcs, which
//! must wind against the hemisphere's outer wire. Each piece is an exact,
//! valid, watertight solid whose volume matches the corner's integral, above
//! or below the equator. The box shortcut's octant feeds a second boolean.
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
    // ball, below the equator when `flip` mirrors it through z = 0.
    for (corner, flip) in [
        ((1.0, 1.0, 1.0), false),
        ((1.0, 1.2, 0.8), false),
        ((0.5, 0.5, 0.5), false),
        ((1.5, 0.2, 0.3), false),
        ((1.0, 1.2, 0.8), true),
    ] {
        let (a, b, c) = corner;
        let piece = corner_piece(a, b, c);
        let z0 = if flip { -c - 10.0 } else { c };
        for (op, truth) in [
            (BooleanOp::Cut, ball - piece),
            (BooleanOp::Intersect, piece),
        ] {
            let label = format!("{corner:?} flip {flip} {op:?}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, block, &Mat4::translation(a, b, z0)).unwrap();
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
            let at =
                |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
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

/// The ball's octant from the box shortcut feeds a second boolean: less a rod
/// of radius 0.4 along `z` through `(1, 1)` (the column over the rod's disc,
/// by Simpson in polar coordinates) it stays exact below the equator, and
/// less a box corner at `(1, 1, ±1)` it loses the corner's piece (within 2%,
/// a mesh allowed) rather than ignoring the tool.
#[test]
fn box_octant_feeds_a_second_boolean() {
    let octant = PI * RADIUS.powi(3) / 6.0;
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let column = simpson(200, 0.0, 0.4, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), 1.0), r.mul_add(th.sin(), 1.0));
            RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    for lower in [false, true] {
        let octant_of = |topo: &mut Topology| {
            let sphere = make_sphere(topo, RADIUS, 32).unwrap();
            let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
            let z0 = if lower { -10.0 } else { 0.0 };
            transform_solid(topo, block, &Mat4::translation(0.0, 0.0, z0)).unwrap();
            boolean(topo, BooleanOp::Intersect, sphere, block).unwrap()
        };
        if lower {
            let mut topo = Topology::new();
            let piece = octant_of(&mut topo);
            let rod = make_cylinder(&mut topo, 0.4, 20.0).unwrap();
            transform_solid(&mut topo, rod, &Mat4::translation(1.0, 1.0, -10.0)).unwrap();
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, BooleanOp::Cut, piece, rod).unwrap();
            assert_eq!(mesh_fallback_count(), before, "rod: fell back to a mesh");
            let (volume, truth) = (solid_volume(&topo, result, 0.01).unwrap(), octant - column);
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "rod: volume {volume}, truth {truth}"
            );
        }
        let mut topo = Topology::new();
        let piece = octant_of(&mut topo);
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let z0 = if lower { -11.0 } else { 1.0 };
        transform_solid(&mut topo, block, &Mat4::translation(1.0, 1.0, z0)).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, piece, block).unwrap();
        let (volume, truth) = (
            solid_volume(&topo, result, 0.01).unwrap(),
            octant - corner_piece(1.0, 1.0, 1.0),
        );
        assert!(
            (volume - truth).abs() < 2e-2 * truth,
            "corner lower {lower}: volume {volume}, truth {truth}"
        );
    }
}
