//! A round wider than the rim it starts from: a tube's inner top rim
//! rounded at a radius past the wall's thickness.
//!
//! The tube runs from `RI` to `RO` about `z`, `H` high. A round of radius
//! `r` along its inner top rim would rest its ball on the top plane `r`
//! from the rim, past the outer rim, so the top ring is swallowed: the ball
//! rests on the inner wall and pivots on the outer rim. In a meridian plane
//! its centre stands `RI + r` from the axis, `sqrt(r^2 - (RI + r - RO)^2)`
//! below the top, and the round runs from its contact on the inner wall to
//! the outer rim, where it meets the outer wall in place of the ring.
//!
//! The round takes off, at each height `z` between its contact and the top,
//! the annulus from `RI` out to the ball's circle.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

const RI: f64 = 3.0;
const RO: f64 = 5.0;
const H: f64 = 10.0;

/// The tube, placed by `pose`.
fn tube(topo: &mut Topology, pose: &Mat4) -> SolidId {
    let outer = make_cylinder(topo, RO, H).unwrap();
    let inner = make_cylinder(topo, RI, H + 2.0).unwrap();
    transform_solid(topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(topo, BooleanOp::Cut, outer, inner).unwrap();
    transform_solid(topo, tube, pose).unwrap();
    tube
}

/// The tube's inner top rim, placed by `pose`.
fn inner_top_rim(topo: &Topology, solid: SolidId, pose: &Mat4) -> EdgeId {
    let top = pose.mul_point(Point3::new(0.0, 0.0, H));
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c)
                if (c.radius() - RI).abs() < 1e-9 && (c.center() - top).length() < 1e-9)
        })
        .expect("the inner top rim")
}

/// The volume the round of radius `r` takes off.
fn removed(r: f64) -> f64 {
    const N: usize = 2000;
    let centre = RI + r;
    let depth = (r * r - (centre - RO).powi(2)).sqrt();
    // Simpson's rule in the height above the ball's centre, `s`.
    let f = |s: f64| PI * ((centre - (r * r - s * s).sqrt()).powi(2) - RI * RI);
    let h = depth / N as f64;
    let mut sum = f(0.0) + f(depth);
    for i in 1..N {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

#[test]
fn a_round_wider_than_a_tube_wall_pivots_on_the_outer_rim() {
    let poses = [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
        ),
    ];
    for (name, pose) in &poses {
        for r in [2.5, 3.0] {
            let mut topo = Topology::new();
            let solid = tube(&mut topo, pose);
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let rim = inner_top_rim(&topo, solid, pose);
            let result = fillet_v2(&mut topo, solid, &[rim], r).unwrap();
            assert!(
                result.failed.is_empty(),
                "{name}, r = {r}: {:?}",
                result.failed
            );
            let faces = solid_faces(&topo, result.solid).unwrap().len();
            assert_eq!(
                faces, 4,
                "{name}, r = {r}: the ring is swallowed by the round"
            );
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{name}, r = {r}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{name}, r = {r}: mesh open");
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            let expected = removed(r);
            assert!(
                (lost - expected).abs() <= 1e-3 * expected,
                "{name}, r = {r}: removed {lost}, the pivoting ball takes off {expected}"
            );
        }
    }
}

/// The volume a round of radius `r` takes off between a rim of radius
/// `rim` and the circle of radius `far` it pivots on, the ball's centre
/// `centre` from the axis, its contact on the wall `depth` from the
/// rim's plane: at each height the annulus from the rim out to the ball's
/// circle (or in from it, where `far < rim`).
fn pivoted(r: f64, rim: f64, centre: f64, depth: f64) -> f64 {
    const N: usize = 2000;
    let sign: f64 = if centre > rim { 1.0 } else { -1.0 };
    let f = |s: f64| {
        let ball = sign.mul_add(-(r * r - s * s).sqrt(), centre);
        PI * (ball * ball - rim * rim).abs()
    };
    let h = depth / N as f64;
    let mut sum = f(0.0) + f(depth);
    for i in 1..N {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// A disc thinner than the round along its top rim: the ball rests on the
/// top and pivots on the bottom rim, the wall goes, and the round runs from
/// the top to the bottom rim.
#[test]
fn a_round_past_a_disc_s_thickness_pivots_on_the_bottom_rim() {
    const R: f64 = 4.32;
    const T: f64 = 0.5;
    for r in [0.8, 1.0] {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, R, T).unwrap();
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let rim = solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(c)
                    if (c.center().z() - T).abs() < 1e-9)
            })
            .expect("the top rim");
        let result = fillet_v2(&mut topo, solid, &[rim], r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let faces = solid_faces(&topo, result.solid).unwrap().len();
        assert_eq!(faces, 3, "r = {r}: the wall is swallowed by the round");
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh open");
        let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        // In a meridian the ball's centre stands `r` below the top and
        // `r` from the bottom rim; each height of the disc loses the
        // annulus out from the ball's circle to the rim.
        let reach = (r * r - (r - T).powi(2)).sqrt();
        let centre = R - reach;
        let expected = {
            const N: usize = 2000;
            let f = |z: f64| {
                let ball = centre + (r * r - (z - (T - r)).powi(2)).sqrt();
                PI * (R * R - ball * ball)
            };
            let h = T / N as f64;
            let mut sum = f(0.0) + f(T);
            for i in 1..N {
                sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            sum * h / 3.0
        };
        assert!(
            (lost - expected).abs() <= 1e-3 * expected,
            "r = {r}: removed {lost}, the pivoting ball takes off {expected}"
        );
    }
}

/// A counterbore whose ledge is narrower than the round along its inner
/// rim: the ball rests on the narrow bore's wall and pivots on the wide
/// bore's rim, and the ledge goes.
#[test]
fn a_round_past_a_counterbore_ledge_pivots_on_the_wide_bore() {
    const NARROW: f64 = 3.0;
    const WIDE: f64 = 3.4;
    const LEDGE: f64 = 6.0;
    for r in [0.5, 0.6] {
        let mut topo = Topology::new();
        let block = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let narrow = make_cylinder(&mut topo, NARROW, 12.0).unwrap();
        transform_solid(&mut topo, narrow, &Mat4::translation(10.0, 10.0, -1.0)).unwrap();
        let wide = make_cylinder(&mut topo, WIDE, 6.0).unwrap();
        transform_solid(&mut topo, wide, &Mat4::translation(10.0, 10.0, LEDGE)).unwrap();
        let bored = boolean(&mut topo, BooleanOp::Cut, block, narrow).unwrap();
        let solid = boolean(&mut topo, BooleanOp::Cut, bored, wide).unwrap();
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let faces_before = solid_faces(&topo, solid).unwrap().len();
        let rim = solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(c)
                    if (c.radius() - NARROW).abs() < 1e-9 && (c.center().z() - LEDGE).abs() < 1e-9)
            })
            .expect("the ledge's inner rim");
        let result = fillet_v2(&mut topo, solid, &[rim], r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let faces = solid_faces(&topo, result.solid).unwrap().len();
        assert_eq!(
            faces, faces_before,
            "r = {r}: the ledge gives way to the round"
        );
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh open");
        let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        let centre = NARROW + r;
        let depth = (r * r - (centre - WIDE).powi(2)).sqrt();
        let expected = pivoted(r, NARROW, centre, depth);
        assert!(
            (lost - expected).abs() <= 1e-3 * expected,
            "r = {r}: removed {lost}, the pivoting ball takes off {expected}"
        );
    }
}

/// The counterbore above cut to a quarter by two planes through its axis:
/// the ledge's inner rim is an arc ending on the cut faces, and the ledge
/// a quarter ring whose ends lie in the round's end sections. The ball
/// still rests on the narrow bore's wall and pivots on the wide bore's
/// rim, the ledge goes, and the round ends on the cut faces in its end
/// sections.
#[test]
fn a_round_past_a_cut_counterbore_ledge_pivots_on_the_wide_bore() {
    const NARROW: f64 = 3.0;
    const WIDE: f64 = 3.4;
    const LEDGE: f64 = 6.0;
    for r in [0.5, 0.6] {
        let mut topo = Topology::new();
        let block = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let narrow = make_cylinder(&mut topo, NARROW, 12.0).unwrap();
        transform_solid(&mut topo, narrow, &Mat4::translation(10.0, 10.0, -1.0)).unwrap();
        let wide = make_cylinder(&mut topo, WIDE, 6.0).unwrap();
        transform_solid(&mut topo, wide, &Mat4::translation(10.0, 10.0, LEDGE)).unwrap();
        // All but the quarter with x and y past 10 cut away, through the
        // bores' seams, so each rim kept is one arc.
        let front = brepkit_operations::primitives::make_box(&mut topo, 22.0, 11.0, 12.0).unwrap();
        transform_solid(&mut topo, front, &Mat4::translation(-1.0, -1.0, -1.0)).unwrap();
        let side = brepkit_operations::primitives::make_box(&mut topo, 11.0, 22.0, 12.0).unwrap();
        transform_solid(&mut topo, side, &Mat4::translation(-1.0, -1.0, -1.0)).unwrap();
        let bored = boolean(&mut topo, BooleanOp::Cut, block, narrow).unwrap();
        let bored = boolean(&mut topo, BooleanOp::Cut, bored, wide).unwrap();
        let bored = boolean(&mut topo, BooleanOp::Cut, bored, front).unwrap();
        let solid = boolean(&mut topo, BooleanOp::Cut, bored, side).unwrap();
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let faces_before = solid_faces(&topo, solid).unwrap().len();
        let rim: Vec<EdgeId> = solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(c)
                    if (c.radius() - NARROW).abs() < 1e-9 && (c.center().z() - LEDGE).abs() < 1e-9)
            })
            .collect();
        assert_eq!(rim.len(), 1, "the ledge's inner rim, whole");
        let result = fillet_v2(&mut topo, solid, &rim, r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let faces = solid_faces(&topo, result.solid).unwrap().len();
        assert_eq!(
            faces, faces_before,
            "r = {r}: the ledge gives way to the round"
        );
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh open");
        let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        let centre = NARROW + r;
        let depth = (r * r - (centre - WIDE).powi(2)).sqrt();
        let expected = 0.25 * pivoted(r, NARROW, centre, depth);
        assert!(
            (lost - expected).abs() <= 1e-3 * expected,
            "r = {r}: removed {lost}, the pivoting ball takes off {expected}"
        );
    }
}
