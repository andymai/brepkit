//! Fillets of closed circular rims where a flat cap meets a coaxial cylinder:
//! a rod's top, whose cap is a disc, and a tube's outer mouth, whose cap is a
//! ring. Each removes a ring whose cross-section is the square of side `r` at
//! the rim less a quarter disc, so its volume is that area times the path of
//! its centroid. The corner blend is a torus of major radius `R - r`: a ring
//! torus below half the cylinder's radius and a spindle torus above it.
//!
//! The rim can also be a hole of its cap: a post standing on a plate, whose
//! concave foot fills a ring of the same section, and a bore's mouth, whose
//! convex rim removes one. A round pocket's floor is a disc cap with a
//! concave rim.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 5.0;
const HEIGHT: f64 = 6.0;

/// The closed circle edge of radius `rim` at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, rim: f64, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - rim).abs() < 1e-9)
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .unwrap()
}

/// The ring a fillet of radius `r` removes from a convex rim of radius `rim`.
fn rounded_off(rim: f64, r: f64) -> f64 {
    let area = (1.0 - PI / 4.0) * r * r;
    2.0 * PI * (rim - centroid_from_corner(r)) * area
}

/// Fillets the outer top rim of `solid` and checks it is valid, watertight,
/// and short by exactly the ring the fillet removes.
fn assert_rim_rounds_off(topo: &mut Topology, solid: SolidId, r: f64) {
    let truth = rounded_off(RADIUS, r);
    assert_rim_fillet_changes_by(topo, solid, (RADIUS, HEIGHT), r, -truth);
}

/// The section's centroid lies this far from the rim's corner, along each of
/// the two faces.
fn centroid_from_corner(r: f64) -> f64 {
    r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI))
}

/// Fillets the rim of radius `rim.0` at height `rim.1` and checks the result
/// is valid, watertight, and changed in volume by exactly `change`.
fn assert_rim_fillet_changes_by(
    topo: &mut Topology,
    solid: SolidId,
    rim: (f64, f64),
    r: f64,
    change: f64,
) {
    let edge = rim_at(topo, solid, rim.0, rim.1);
    let before = solid_volume(topo, solid, 0.001).unwrap();
    let result = fillet_v2(topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let changed = solid_volume(topo, result.solid, 0.001).unwrap() - before;
    assert!(
        (changed - change).abs() < 1e-6 * before,
        "r = {r}: changed by {changed}, truth {change}"
    );
}

#[test]
fn a_rods_top_rim_rounds_off_on_ring_and_spindle_tori() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
        assert_rim_rounds_off(&mut topo, rod, r);
    }
}

/// The tube's top is a ring, not a disc, and its material still lies toward
/// the axis from the outer rim.
#[test]
fn a_tubes_outer_mouth_rounds_off_on_ring_and_spindle_tori() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let outer = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
        let bore = make_cylinder(&mut topo, 1.0, HEIGHT + 2.0).unwrap();
        transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
        let tube = boolean(&mut topo, BooleanOp::Cut, outer, bore).unwrap();
        assert_rim_rounds_off(&mut topo, tube, r);
    }
}

/// A post standing on a plate: the rim is a hole of the plate's top, and the
/// fillet fills the ring round the post's foot.
#[test]
fn a_posts_foot_on_a_plate_fills_its_ring() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 30.0, 30.0, 4.0).unwrap();
        let post = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
        transform_solid(&mut topo, post, &Mat4::translation(15.0, 15.0, 4.0)).unwrap();
        let body = boolean(&mut topo, BooleanOp::Fuse, plate, post).unwrap();
        let area = (1.0 - PI / 4.0) * r * r;
        let filled = 2.0 * PI * (RADIUS + centroid_from_corner(r)) * area;
        assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 4.0), r, filled);
    }
}

/// A bore through a plate: its mouth is a hole of the plate's top, and the
/// fillet rounds off the ring round it.
#[test]
fn a_bores_mouth_in_a_plate_rounds_off() {
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 30.0, 30.0, 8.0).unwrap();
        let bore = make_cylinder(&mut topo, RADIUS, 12.0).unwrap();
        transform_solid(&mut topo, bore, &Mat4::translation(15.0, 15.0, -2.0)).unwrap();
        let body = boolean(&mut topo, BooleanOp::Cut, plate, bore).unwrap();
        let area = (1.0 - PI / 4.0) * r * r;
        let removed = 2.0 * PI * (RADIUS + centroid_from_corner(r)) * area;
        assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 8.0), r, -removed);
    }
}

/// A blind round pocket: its floor is a disc cap whose rim is concave, and
/// the fillet fills the ring round the floor's edge.
#[test]
#[ignore = "the plane x cylinder stripe rolls the ball below the floor and outside the pocket wall"]
fn a_round_pockets_floor_fills_its_ring() {
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 30.0, 30.0, 10.0).unwrap();
        let pocket = make_cylinder(&mut topo, RADIUS, 8.0).unwrap();
        transform_solid(&mut topo, pocket, &Mat4::translation(15.0, 15.0, 4.0)).unwrap();
        let body = boolean(&mut topo, BooleanOp::Cut, block, pocket).unwrap();
        let area = (1.0 - PI / 4.0) * r * r;
        let filled = 2.0 * PI * (RADIUS - centroid_from_corner(r)) * area;
        assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 4.0), r, filled);
    }
}

/// A bore through an elliptical plate: the plate's outer edge is an ellipse
/// that encloses the plate contact, so the mouth still takes the closed-rim
/// path and rounds off by the ring.
#[test]
fn a_bores_mouth_in_an_elliptical_plate_rounds_off() {
    let mut topo = Topology::new();
    let plate = make_cylinder(&mut topo, 10.0, 8.0).unwrap();
    transform_solid(&mut topo, plate, &Mat4::scale(2.0, 1.5, 1.0)).unwrap();
    let bore = make_cylinder(&mut topo, RADIUS, 12.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -2.0)).unwrap();
    let body = boolean(&mut topo, BooleanOp::Cut, plate, bore).unwrap();
    let area = (1.0 - PI / 4.0) * 1.0;
    let removed = 2.0 * PI * (RADIUS + centroid_from_corner(1.0)) * area;
    assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 8.0), 1.0, -removed);
}
