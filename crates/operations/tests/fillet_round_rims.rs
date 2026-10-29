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
//! concave rim. A cone's rims meet their caps at other angles: a frustum's
//! two rims, a tapered post's foot on a plate, and a tapered hole's mouth.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
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

/// A bore's mouth fillet wider than its plate is thick would put the wall
/// contact past the plate's far face: it is refused, never a solid that
/// crosses itself.
#[test]
fn a_bores_mouth_fillet_wider_than_its_plate_is_refused() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 30.0, 30.0, 2.0).unwrap();
    let bore = make_cylinder(&mut topo, RADIUS, 6.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(15.0, 15.0, -2.0)).unwrap();
    let body = boolean(&mut topo, BooleanOp::Cut, plate, bore).unwrap();
    let edge = rim_at(&topo, body, RADIUS, 2.0);
    let Ok(result) = fillet_v2(&mut topo, body, &[edge], 3.0) else {
        return;
    };
    assert!(
        !result.failed.is_empty() || result.solid == body,
        "a fillet wider than the plate must not build"
    );
}

/// A radius 5 post standing 8 tall on a 16 x 16 x 4 plate, its top cut by
/// the plane rising `slope` along x through the point 3 above its foot.
fn post_with_a_sloped_top(topo: &mut Topology, slope: f64) -> SolidId {
    let plate = make_box(topo, 16.0, 16.0, 4.0).unwrap();
    let post = make_cylinder(topo, RADIUS, 8.0).unwrap();
    transform_solid(topo, post, &Mat4::translation(8.0, 8.0, 4.0)).unwrap();
    let body = boolean(topo, BooleanOp::Fuse, plate, post).unwrap();
    let above = make_box(topo, 30.0, 30.0, 20.0).unwrap();
    let tilt = Mat4::translation(8.0, 8.0, 7.0)
        * Mat4::rotation_y(-slope.atan())
        * Mat4::translation(-15.0, -15.0, 0.0);
    transform_solid(topo, above, &tilt).unwrap();
    boolean(topo, BooleanOp::Cut, body, above).unwrap()
}

/// The highest point of a post's sloped top clears its foot fillet's wall
/// contact, but the lowest does not: the contact circle would cross the top.
#[test]
fn a_posts_foot_fillet_reaching_past_its_sloped_top_is_refused() {
    let mut topo = Topology::new();
    let body = post_with_a_sloped_top(&mut topo, 0.3);
    let edge = rim_at(&topo, body, RADIUS, 4.0);
    let Ok(result) = fillet_v2(&mut topo, body, &[edge], 2.0) else {
        return;
    };
    assert!(
        !result.failed.is_empty() || result.solid == body,
        "a contact circle crossing the top must not build"
    );
}

/// A sloped top whose lowest point clears the contact keeps the ring exact.
#[test]
fn a_posts_foot_under_a_sloped_top_fills_its_ring() {
    let mut topo = Topology::new();
    let body = post_with_a_sloped_top(&mut topo, 0.3);
    let r = 1.0;
    let area = (1.0 - PI / 4.0) * r * r;
    let filled = 2.0 * PI * (RADIUS + centroid_from_corner(r)) * area;
    assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 4.0), r, filled);
}

/// A bore cut by a cylinder standing either way up leaves its rim circle
/// turning either way about the bore's axis; the mouth rounds off the same.
#[test]
fn a_bores_mouth_rounds_off_whichever_way_its_rim_turns() {
    for flip in [false, true] {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 30.0, 30.0, 8.0).unwrap();
        let bore = make_cylinder(&mut topo, RADIUS, 12.0).unwrap();
        let placed = if flip {
            Mat4::translation(15.0, 15.0, 10.0) * Mat4::rotation_x(std::f64::consts::PI)
        } else {
            Mat4::translation(15.0, 15.0, -2.0)
        };
        transform_solid(&mut topo, bore, &placed).unwrap();
        let body = boolean(&mut topo, BooleanOp::Cut, plate, bore).unwrap();
        let area = 1.0 - PI / 4.0;
        let removed = 2.0 * PI * (RADIUS + centroid_from_corner(1.0)) * area;
        assert_rim_fillet_changes_by(&mut topo, body, (RADIUS, 8.0), 1.0, -removed);
    }
}

/// The ring a fillet of radius `r` fills or removes at a rim of radius `rim`,
/// from the plate's and the wall's unit rays leaving the corner in the
/// (radial, axial) half-plane: the kite between the corner, both contacts and
/// the ball's centre, less the ball's sector, turned about the axis (Pappus).
fn ring_volume(rim: f64, plate: (f64, f64), wall: (f64, f64), r: f64) -> f64 {
    let theta = plate.0.mul_add(wall.0, plate.1 * wall.1).acos();
    let reach = r / (theta / 2.0).tan();
    let to_centre = r / (theta / 2.0).sin();
    let (bx, by) = (plate.0 + wall.0, plate.1 + wall.1);
    let len = bx.hypot(by);
    let corner = (rim, 0.0);
    let centre = (rim + bx / len * to_centre, by / len * to_centre);
    let on_plate = (rim + plate.0 * reach, plate.1 * reach);
    let on_wall = (rim + wall.0 * reach, wall.1 * reach);
    let moment = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        let area = (b.0 - a.0)
            .mul_add(c.1 - a.1, -(c.0 - a.0) * (b.1 - a.1))
            .abs()
            / 2.0;
        area * (a.0 + b.0 + c.0) / 3.0
    };
    let kite = moment(corner, on_plate, centre) + moment(corner, centre, on_wall);
    let phi = PI - theta;
    let sector_rho =
        centre.0 + (corner.0 - centre.0) / to_centre * 4.0 * r * (phi / 2.0).sin() / (3.0 * phi);
    2.0 * PI * (kite - 0.5 * r * r * phi * sector_rho)
}

fn unit(x: f64, y: f64) -> (f64, f64) {
    let len = x.hypot(y);
    (x / len, y / len)
}

/// A frustum's rims are convex: the bottom one leans in at the base, the top
/// one opens past a right angle at the top.
#[test]
fn a_frustums_rims_round_off() {
    for r in [0.3, 0.6] {
        let mut topo = Topology::new();
        let frustum = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
        let removed = ring_volume(3.0, (-1.0, 0.0), unit(-3.0, 6.0), r);
        assert_rim_fillet_changes_by(&mut topo, frustum, (3.0, 0.0), r, -removed);
    }
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
    let removed = ring_volume(1.0, (-1.0, 0.0), unit(2.0, 4.0), 0.3);
    assert_rim_fillet_changes_by(&mut topo, frustum, (1.0, 4.0), 0.3, -removed);
}

/// A tapered post on a plate, narrowing or widening as it rises: its foot is
/// a hole of the plate's top, and the fillet fills the ring round it.
#[test]
fn a_tapered_posts_foot_on_a_plate_fills_its_ring() {
    for (bottom, top, r) in [(5.0, 3.0, 1.0), (5.0, 3.0, 2.0), (3.0, 5.0, 1.0)] {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 30.0, 30.0, 4.0).unwrap();
        let post = make_cone(&mut topo, bottom, top, 6.0).unwrap();
        transform_solid(&mut topo, post, &Mat4::translation(15.0, 15.0, 4.0)).unwrap();
        let body = boolean(&mut topo, BooleanOp::Fuse, plate, post).unwrap();
        let wall = unit(top - bottom, 6.0);
        let filled = ring_volume(bottom, (1.0, 0.0), wall, r);
        assert_rim_fillet_changes_by(&mut topo, body, (bottom, 4.0), r, filled);
    }
}

/// A tapered hole through a plate, wider at the top: its mouth is a hole of
/// the plate's top, and the fillet rounds off the ring round it.
#[test]
fn a_tapered_holes_mouth_rounds_off() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 30.0, 30.0, 8.0).unwrap();
    let hole = make_cone(&mut topo, 2.0, 5.0, 8.0).unwrap();
    transform_solid(&mut topo, hole, &Mat4::translation(15.0, 15.0, 0.0)).unwrap();
    let body = boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    let removed = ring_volume(5.0, (1.0, 0.0), unit(-3.0, 8.0), 1.0);
    assert_rim_fillet_changes_by(&mut topo, body, (5.0, 8.0), 1.0, -removed);
}

/// Fillets `rim` and checks it is refused, or the result is valid and
/// watertight: never a band whose wall contact overruns the wall.
fn assert_refused_or_sound(topo: &mut Topology, solid: SolidId, rim: (f64, f64), r: f64) {
    let edge = rim_at(topo, solid, rim.0, rim.1);
    let Ok(result) = fillet_v2(topo, solid, &[edge], r) else {
        return;
    };
    if !result.failed.is_empty() || result.solid == solid {
        return;
    }
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
}

/// A wall shorter than the fillet reaches: the contact would land past the
/// wall's far rim.
#[test]
fn a_rim_fillet_reaching_past_a_short_wall_is_refused() {
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 2.99, 0.2).unwrap();
    assert_refused_or_sound(&mut topo, frustum, (2.99, 0.2), 0.3);

    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, RADIUS, 0.2).unwrap();
    assert_refused_or_sound(&mut topo, disc, (RADIUS, 0.2), 0.3);

    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 30.0, 30.0, 4.0).unwrap();
    let post = make_cone(&mut topo, 5.0, 4.99, 0.2).unwrap();
    transform_solid(&mut topo, post, &Mat4::translation(15.0, 15.0, 4.0)).unwrap();
    let body = boolean(&mut topo, BooleanOp::Fuse, plate, post).unwrap();
    assert_refused_or_sound(&mut topo, body, (5.0, 4.0), 0.3);
}
