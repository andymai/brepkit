//! The check crate's properties read trimmed cylinder, cone and sphere faces
//! exactly, along their wires' own curves: slanted rims, a sphere face
//! whose rim dips between wall arcs, hyperbola edges, and a rod's bore
//! through a cap, upright, turned and mirrored.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::properties::{PropertiesOptions, center_of_mass, solid_area, solid_volume};
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        ("turned", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

fn placed(topo: &mut Topology, solid: SolidId, pose: Mat4, at: Mat4) -> SolidId {
    transform_solid(topo, solid, &(pose * at)).unwrap();
    solid
}

fn volume(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, &PropertiesOptions::default()).unwrap()
}

/// `make_cylinder(5, 10)` less a slab tilted 0.4 about x whose lower face
/// passes through `(0, 0, 7)`: the wall runs up to an ellipse over its seam,
/// so the solid is `7 * 25π` and the wall `70π`.
#[test]
fn a_cylinder_cut_on_a_slant_reads_exactly() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let cylinder = make_cylinder(&mut topo, 5.0, 10.0).unwrap();
        placed(&mut topo, cylinder, pose, Mat4::identity());
        let slab = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let tilt = Mat4::translation(0.0, 0.0, 7.0)
            * Mat4::rotation_x(0.4)
            * Mat4::translation(-10.0, -10.0, 0.0);
        placed(&mut topo, slab, pose, tilt);
        let result = boolean(&mut topo, BooleanOp::Cut, cylinder, slab).unwrap();
        let (got, truth) = (volume(&topo, result), 7.0 * 25.0 * PI);
        assert!(
            (got - truth).abs() < 1e-9 * truth,
            "{name}: volume {got}, truth {truth}"
        );
        let area = solid_area(&topo, result, &PropertiesOptions::default()).unwrap();
        let slant = 25.0 * PI / 0.4_f64.cos();
        let truth = 25.0 * PI + 70.0 * PI + slant;
        assert!(
            (area - truth).abs() < 1e-9 * truth,
            "{name}: area {area}, truth {truth}"
        );
    }
}

/// The ball of radius 3 within the column `|x|, |y| < 2.5`: its sphere
/// faces' rims dip to the equator between the wall arcs, and the solid is
/// the ball less four caps of height 0.5, its centre of mass the ball's.
#[test]
fn a_ball_within_a_column_reads_exactly() {
    let caps = 4.0 * PI * 0.25 * 0.5f64.mul_add(-1.0, 9.0) / 3.0;
    let truth = 4.0 / 3.0 * PI * 27.0 - caps;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
        placed(&mut topo, ball, pose, Mat4::identity());
        let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
        placed(&mut topo, column, pose, Mat4::translation(-2.5, -2.5, -5.0));
        let result = boolean(&mut topo, BooleanOp::Intersect, ball, column).unwrap();
        let got = volume(&topo, result);
        assert!(
            (got - truth).abs() < 1e-9 * truth,
            "{name}: volume {got}, truth {truth}"
        );
        let c = center_of_mass(&topo, result, &PropertiesOptions::default()).unwrap();
        assert!(
            c.x().abs().max(c.y().abs()).max(c.z().abs()) < 1e-9,
            "{name}: centre of mass {c:?}"
        );
    }
}

/// The pointed `make_cone(5, 0, 10)` less the half-space `x > 1`: its wall
/// is bounded by hyperbola edges (NURBS curves, integrated span by span) and
/// runs through the apex, and the solid is the cone less the integral of the
/// segment the half-space takes off each section.
#[test]
fn a_cone_less_a_half_space_reads_exactly() {
    let segment = |r: f64| {
        if r <= 1.0 {
            0.0
        } else {
            r * r * (1.0 / r).acos() - (r * r - 1.0).sqrt()
        }
    };
    let (n, top) = (20_000_u32, 8.0);
    let step = top / f64::from(n);
    let at = |k: u32| segment(0.5f64.mul_add(-step * f64::from(k), 5.0));
    let mut removed = at(0) + at(n);
    for k in 1..n {
        removed += if k % 2 == 1 { 4.0 } else { 2.0 } * at(k);
    }
    let truth = 250.0 * PI / 3.0 - removed * step / 3.0;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let cone = make_cone(&mut topo, 5.0, 0.0, 10.0).unwrap();
        placed(&mut topo, cone, pose, Mat4::identity());
        let block = make_box(&mut topo, 5.0, 12.0, 12.0).unwrap();
        placed(&mut topo, block, pose, Mat4::translation(1.0, -6.0, -1.0));
        let result = boolean(&mut topo, BooleanOp::Cut, cone, block).unwrap();
        let got = volume(&topo, result);
        assert!(
            (got - truth).abs() < 1e-9 * truth,
            "{name}: volume {got}, truth {truth}"
        );
    }
}

/// The ball of radius 3 less the column `|x|, |y| < 2.5` fused with a rod of
/// radius 0.1 through `(2.75, 0)`: the rod bores a cap, its rim one closed
/// NURBS edge, and takes off its column of the ball (by Simpson in polar
/// coordinates, good to about 1e-9).
#[test]
fn a_bored_cap_reads_its_bore() {
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let bore = simpson(200, 0.0, 0.1, &|q: f64| {
        q * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (q.mul_add(th.cos(), 2.75), q * th.sin());
            2.0 * (9.0 - x * x - y * y).max(0.0).sqrt()
        })
    });
    let caps = 4.0 * PI * 0.25 * 0.5f64.mul_add(-1.0, 9.0) / 3.0;
    let truth = caps - bore;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
        placed(&mut topo, ball, pose, Mat4::identity());
        let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
        placed(&mut topo, column, pose, Mat4::translation(-2.5, -2.5, -5.0));
        let rod = make_cylinder(&mut topo, 0.1, 10.0).unwrap();
        placed(&mut topo, rod, pose, Mat4::translation(2.75, 0.0, -5.0));
        let tool = boolean(&mut topo, BooleanOp::Fuse, column, rod).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, ball, tool).unwrap();
        let got = volume(&topo, result);
        assert!(
            (got - truth).abs() < 1e-8 * truth,
            "{name}: volume {got}, truth {truth}"
        );
    }
}
