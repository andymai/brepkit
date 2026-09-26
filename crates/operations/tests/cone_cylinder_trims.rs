//! The boolean engine's ray cast reads a cone or cylinder face whose edges
//! are not all rulings and axis circles (a cone cut by planes that miss its
//! apex, a cylinder cut on a slant) against the face's own wires.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_algo::FaceClass;
use brepkit_algo::classifier::classify_ray_cast;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Points this close to a surface are not checked.
const NEAR: f64 = 0.02;

/// Signed distance to `make_cone(5, top, 10)`, negative inside (the slant
/// side's distance measured square to it).
fn cone(p: Point3, top: f64) -> f64 {
    let slope = (5.0 - top) / 10.0;
    let rho = p.x().hypot(p.y());
    let side = (rho - slope.mul_add(-p.z(), 5.0)) / slope.hypot(1.0);
    side.max(-p.z()).max(p.z() - 10.0)
}

/// Signed distance to `make_cylinder(5, 10)`, negative inside.
fn cylinder(p: Point3) -> f64 {
    (p.x().hypot(p.y()) - 5.0).max(-p.z()).max(p.z() - 10.0)
}

/// Signed distance to the box from `lo` to `hi`, negative inside.
fn in_box(p: Point3, lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let c = [p.x(), p.y(), p.z()];
    (0..3).fold(f64::NEG_INFINITY, |d, k| {
        d.max(lo[k] - c[k]).max(c[k] - hi[k])
    })
}

/// A box from `lo` to `hi`.
fn boxed(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

/// The slab 20 by 20 by 10 tilted 0.4 about x with its lower face through
/// `(0, 0, 7)`, and its placement.
fn slab(topo: &mut Topology) -> (SolidId, Mat4) {
    let b = make_box(topo, 20.0, 20.0, 10.0).unwrap();
    let pose = Mat4::translation(0.0, 0.0, 7.0)
        * Mat4::rotation_x(0.4)
        * Mat4::translation(-10.0, -10.0, 0.0);
    transform_solid(topo, b, &pose).unwrap();
    (b, pose)
}

/// Every point of a grid over `[-6, 6]^2 x [-1, 11]` more than [`NEAR`] from
/// both operands' surfaces reads as `inside` says (from the operands' signed
/// distances).
fn reads_right(
    name: &str,
    topo: &Topology,
    solid: SolidId,
    distances: &dyn Fn(Point3) -> (f64, f64),
    inside: fn(f64, f64) -> bool,
) {
    let mut wrong = Vec::new();
    for i in 0..21 {
        for j in 0..21 {
            for k in 0..21 {
                let at = |n: i32, lo: f64| 12.0f64.mul_add(f64::from(n) / 20.0, lo);
                let p = Point3::new(at(i, -6.0), at(j, -6.0), at(k, -1.0));
                let (a, b) = distances(p);
                if a.abs() < NEAR || b.abs() < NEAR {
                    continue;
                }
                let got = classify_ray_cast(topo, solid, p).unwrap() == FaceClass::Inside;
                if got != inside(a, b) {
                    wrong.push(p);
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

fn intersect(a: f64, b: f64) -> bool {
    a < 0.0 && b < 0.0
}

fn cut(a: f64, b: f64) -> bool {
    a < 0.0 && b > 0.0
}

/// A cone within a box its base overhangs: four hyperbolas below the top rim.
#[test]
fn a_cone_within_a_box_reads_by_its_wires() {
    let mut topo = Topology::new();
    let base = make_cone(&mut topo, 5.0, 2.0, 10.0).unwrap();
    let (lo, hi) = ([-3.0, -3.0, -1.0], [3.0, 3.0, 11.0]);
    let tool = boxed(&mut topo, lo, hi);
    let r = boolean(&mut topo, BooleanOp::Intersect, base, tool).unwrap();
    let distances = |p: Point3| (cone(p, 2.0), in_box(p, lo, hi));
    reads_right("cone within a box", &topo, r, &distances, intersect);
}

/// A cone less the half-space `x > 1`: two hyperbolas from base to top, or
/// to the apex's side of a pointed cone.
#[test]
fn a_cone_less_a_half_space_reads_by_its_wires() {
    for top in [2.0, 0.0] {
        let mut topo = Topology::new();
        let base = make_cone(&mut topo, 5.0, top, 10.0).unwrap();
        let (lo, hi) = ([1.0, -6.0, -1.0], [6.0, 6.0, 11.0]);
        let tool = boxed(&mut topo, lo, hi);
        let r = boolean(&mut topo, BooleanOp::Cut, base, tool).unwrap();
        let distances = |p: Point3| (cone(p, top), in_box(p, lo, hi));
        reads_right(
            &format!("cone to radius {top} less x > 1"),
            &topo,
            r,
            &distances,
            cut,
        );
    }
}

/// A cone and a cylinder cut on a slant: an ellipse above the base rim.
#[test]
fn a_cone_and_a_cylinder_cut_on_a_slant_read_by_their_wires() {
    for (name, is_cone) in [("cone", true), ("cylinder", false)] {
        let mut topo = Topology::new();
        let base = if is_cone {
            make_cone(&mut topo, 5.0, 2.0, 10.0).unwrap()
        } else {
            make_cylinder(&mut topo, 5.0, 10.0).unwrap()
        };
        let (tool, pose) = slab(&mut topo);
        let r = boolean(&mut topo, BooleanOp::Cut, base, tool).unwrap();
        let back = pose.inverse().unwrap();
        let distances = |p: Point3| {
            let own = if is_cone { cone(p, 2.0) } else { cylinder(p) };
            (own, in_box(back.mul_point(p), [0.0; 3], [20.0, 20.0, 10.0]))
        };
        reads_right(
            &format!("{name} less a slant slab"),
            &topo,
            r,
            &distances,
            cut,
        );
    }
}
