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

use brepkit_blend::BlendError;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::OperationsError;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::FaceSurface;
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
    assert_edges_lie_on_their_faces(topo, result.solid, r);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let changed = solid_volume(topo, result.solid, 0.001).unwrap() - before;
    assert!(
        (changed - change).abs() < 1e-6 * before,
        "r = {r}: changed by {changed}, truth {change}"
    );
}

/// Every edge of `solid` lies on each face that uses it, sampled at three
/// points inside it: the torus band's seam is its meridian arc, not the
/// chord across the round.
fn assert_edges_lie_on_their_faces(topo: &Topology, solid: SolidId, r: f64) {
    let faces_of = brepkit_topology::explorer::edge_to_face_map(topo, solid).unwrap();
    for edge_id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        for face_id in &faces_of[&edge_id.index()] {
            let surface = topo.face(*face_id).unwrap().surface();
            for f in [0.25, 0.5, 0.75] {
                let p = edge
                    .curve()
                    .evaluate_with_endpoints((t1 - t0).mul_add(f, t0), a, b);
                let off = if let FaceSurface::Plane { normal, d } = surface {
                    (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                } else {
                    let (u, v) = surface.project_point(p).unwrap();
                    (surface.evaluate(u, v).unwrap() - p).length()
                };
                assert!(
                    off < 1e-6,
                    "r = {r}: a {} edge stands {off} off its {} face",
                    edge.curve().type_tag(),
                    surface.type_tag()
                );
            }
        }
    }
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

/// Fillets all of `rims` in a single call.
fn fillet_rims(
    topo: &mut Topology,
    solid: SolidId,
    rims: &[(f64, f64)],
    r: f64,
) -> Option<SolidId> {
    let edges: Vec<EdgeId> = rims
        .iter()
        .map(|&(rim, z)| rim_at(topo, solid, rim, z))
        .collect();
    let result = fillet_v2(topo, solid, &edges, r).ok()?;
    (result.failed.is_empty() && result.solid != solid).then_some(result.solid)
}

/// Each rim's contact fits the wall alone, but the two land past each other
/// and their bands would overlap.
#[test]
fn both_rims_of_a_wall_too_short_for_their_two_fillets_are_refused() {
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 2.99, 0.5).unwrap();
    assert!(fillet_rims(&mut topo, frustum, &[(3.0, 0.0), (2.99, 0.5)], 0.3).is_none());

    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, RADIUS, 0.5).unwrap();
    assert!(fillet_rims(&mut topo, disc, &[(RADIUS, 0.0), (RADIUS, 0.5)], 0.3).is_none());

    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 30.0, 30.0, 0.5).unwrap();
    let bore = make_cylinder(&mut topo, RADIUS, 0.5).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(15.0, 15.0, 0.0)).unwrap();
    let body = boolean(&mut topo, BooleanOp::Cut, plate, bore).unwrap();
    assert!(fillet_rims(&mut topo, body, &[(RADIUS, 0.0), (RADIUS, 0.5)], 0.3).is_none());
}

/// A wall tall enough for both fillets keeps a strip between their contacts.
#[test]
fn both_rims_of_a_wall_round_off_in_one_call() {
    let r = 0.3;
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 2.99, 1.0).unwrap();
    let before = solid_volume(&topo, frustum, 0.001).unwrap();
    let removed = ring_volume(3.0, (-1.0, 0.0), unit(-0.01, 1.0), r)
        + ring_volume(2.99, (-1.0, 0.0), unit(0.01, 1.0), r);
    let rounded = fillet_rims(&mut topo, frustum, &[(3.0, 0.0), (2.99, 1.0)], r).unwrap();
    assert_sound_and_changed_by(&topo, rounded, before, -removed);

    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, RADIUS, 1.0).unwrap();
    let before = solid_volume(&topo, disc, 0.001).unwrap();
    let rounded = fillet_rims(&mut topo, disc, &[(RADIUS, 0.0), (RADIUS, 1.0)], r).unwrap();
    assert_sound_and_changed_by(&topo, rounded, before, -2.0 * rounded_off(RADIUS, r));
}

/// Both rims of a frustum rebuild every face it has, so no input face is
/// left to anchor the orientation: the result must point outward wherever
/// the part sits.
#[test]
fn both_rims_of_a_frustum_point_outward_wherever_it_sits() {
    let r = 0.3;
    for dz in [0.0, -100.0] {
        let mut topo = Topology::new();
        let frustum = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
        transform_solid(&mut topo, frustum, &Mat4::translation(0.0, 0.0, dz)).unwrap();
        let before = solid_volume(&topo, frustum, 0.001).unwrap();
        let removed = ring_volume(3.0, (-1.0, 0.0), unit(-3.0, 6.0), r)
            + ring_volume(1.0, (-1.0, 0.0), unit(2.0, 4.0), r);
        let rounded = fillet_rims(&mut topo, frustum, &[(3.0, dz), (1.0, 4.0 + dz)], r)
            .unwrap_or_else(|| unreachable!("dz = {dz}: both rims must round off"));
        assert_sound_and_changed_by(&topo, rounded, before, -removed);
        let oriented = oriented_solid_volume(&topo, rounded, 0.001).unwrap();
        assert!(
            oriented > 0.0,
            "dz = {dz}: the shell points inward ({oriented})"
        );
    }
}

/// A tube's outer and inner rims share its two caps, and a wall 0.5 tall
/// leaves no room for the two contacts of either wall at r 0.3.
#[test]
fn rims_sharing_their_caps_on_walls_too_short_for_two_fillets_are_refused() {
    let mut topo = Topology::new();
    let outer = make_cylinder(&mut topo, RADIUS, 0.5).unwrap();
    let inner = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let rims = [(RADIUS, 0.0), (RADIUS, 0.5), (1.0, 0.0), (1.0, 0.5)];
    assert!(fillet_rims(&mut topo, tube, &rims, 0.3).is_none());
}

fn assert_sound_and_changed_by(topo: &Topology, solid: SolidId, before: f64, change: f64) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    let changed = solid_volume(topo, solid, 0.001).unwrap() - before;
    assert!(
        (changed - change).abs() < 1e-6 * before,
        "changed by {changed}, truth {change}"
    );
}

/// A rod drilled through at `offset` from its axis by a hole of radius
/// `hole`, its top rim then rounded by `r`: the rounded solid, or why the
/// round was not built.
fn round_a_drilled_rods_top(
    offset: f64,
    hole: f64,
    r: f64,
) -> Result<(Topology, SolidId, f64), String> {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let pin = make_cylinder(&mut topo, hole, HEIGHT + 2.0).unwrap();
    transform_solid(&mut topo, pin, &Mat4::translation(offset, 0.0, -1.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, rod, pin).unwrap();
    let before = solid_volume(&topo, drilled, 0.001).unwrap();
    let edge = rim_at(&topo, drilled, RADIUS, HEIGHT);
    match fillet_v2(&mut topo, drilled, &[edge], r) {
        Err(e) => Err(e.to_string()),
        Ok(result) if !result.failed.is_empty() => Err(format!("{:?}", result.failed)),
        Ok(result) => Ok((topo, result.solid, before)),
    }
}

/// A hole on the top under the round, crossing the circle its contact runs
/// along or lying between that circle and the rim: the band would cut
/// through the hole, which the fillet does not build, so it is refused,
/// not built as a top whose outline crosses or leaves out its hole.
#[test]
fn a_rods_top_rim_round_over_a_hole_is_refused() {
    for (offset, hole) in [(3.5, 0.8), (4.2, 0.3)] {
        assert!(
            round_a_drilled_rods_top(offset, hole, 2.0).is_err(),
            "a round over the hole at {offset} was built"
        );
    }
}

/// The same hole wholly inside the contact's circle keeps the round, and the
/// result is sound: short of the drilled rod by the rim's strip.
#[test]
fn a_rods_top_rim_rounds_off_beside_a_hole() {
    let (topo, solid, before) = round_a_drilled_rods_top(1.5, 0.8, 2.0).unwrap();
    assert_sound_and_changed_by(&topo, solid, before, -rounded_off(RADIUS, 2.0));
}

/// A rod 10 tall, its top cut by the plane rising 0.3 along x through the
/// point 7 above its axis's foot, so its top rim is an ellipse (a walked
/// round), and cross-drilled along y at height `z` by a bore of radius
/// `bore` (none at 0), which leaves two holes in its wall where the rim is
/// at height 7.
fn sloped_rod(topo: &mut Topology, z: f64, bore: f64) -> SolidId {
    let rod = make_cylinder(topo, RADIUS, 10.0).unwrap();
    let above = make_box(topo, 30.0, 30.0, 20.0).unwrap();
    let tilt = Mat4::translation(0.0, 0.0, 7.0)
        * Mat4::rotation_y(-0.3_f64.atan())
        * Mat4::translation(-15.0, -15.0, 0.0);
    transform_solid(topo, above, &tilt).unwrap();
    let rod = boolean(topo, BooleanOp::Cut, rod, above).unwrap();
    if bore == 0.0 {
        return rod;
    }
    let pin = make_cylinder(topo, bore, 3.0 * RADIUS).unwrap();
    let across = Mat4::translation(0.0, 1.5 * RADIUS, z) * Mat4::rotation_x(PI / 2.0);
    transform_solid(topo, pin, &across).unwrap();
    boolean(topo, BooleanOp::Cut, rod, pin).unwrap()
}

/// The sloped rod's top rim.
fn sloped_rim(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
        .unwrap()
}

/// Rounds the sloped rod's top rim by `r`, cross-drilled as `sloped_rod`
/// says: the rounded solid and the volume before, or why it was not built.
fn round_a_sloped_rods_top(z: f64, bore: f64, r: f64) -> Result<(Topology, SolidId, f64), String> {
    let mut topo = Topology::new();
    let rod = sloped_rod(&mut topo, z, bore);
    let before = solid_volume(&topo, rod, 0.001).unwrap();
    let edge = sloped_rim(&topo, rod);
    match fillet_v2(&mut topo, rod, &[edge], r) {
        Err(e) => Err(e.to_string()),
        Ok(result) if !result.failed.is_empty() => Err(format!("{:?}", result.failed)),
        Ok(result) => Ok((topo, result.solid, before)),
    }
}

/// Whether rounding the sloped rod's top rim by `r`, cross-drilled as
/// `sloped_rod` says and the whole part shrunk `tenths` times by a tenth,
/// is refused for its trim: the wall's outline and a hole of it would
/// cross.
fn sloped_rods_round_is_refused_for_its_trim(z: f64, bore: f64, r: f64, tenths: i32) -> bool {
    let mut topo = Topology::new();
    let rod = sloped_rod(&mut topo, z, bore);
    // `transform_solid` reads a determinant below its tolerance as
    // degenerate, so the part shrinks a tenth at a time.
    for _ in 0..tenths {
        transform_solid(&mut topo, rod, &Mat4::scale(0.1, 0.1, 0.1)).unwrap();
    }
    let edge = sloped_rim(&topo, rod);
    matches!(
        fillet_v2(&mut topo, rod, &[edge], r * 0.1_f64.powi(tenths)),
        Err(OperationsError::Blend(BlendError::TrimmingFailure { .. }))
    )
}

/// A bore through the wall reaching above where the round's contact runs
/// on the wall (about r below the rim where the bore meets it): the contact
/// runs over the bore's holes, which the fillet does not cut, so it is
/// refused, not built as a wall whose outline crosses its holes (valid by
/// edge count, but no wall at all there). A bore wholly between the contact
/// and the rim lies under the band the round takes off, and is refused too:
/// the wall kept its holes outside its trimmed outline. Each holds for the
/// part a thousand times smaller, as one drawn in metres.
#[test]
fn a_rods_top_rim_round_over_a_bore_in_its_wall_is_refused() {
    for (z, bore) in [(5.2, 0.5), (4.7, 0.4), (6.0, 0.3)] {
        for tenths in [0, 3] {
            assert!(
                sloped_rods_round_is_refused_for_its_trim(z, bore, 2.0, tenths),
                "the round over the bore at {z}, shrunk {tenths} tenths, was not refused for its trim"
            );
        }
    }
}

/// The same bore wholly below the contact keeps the round, and the result
/// is sound: the round takes off what it takes off the undrilled rod.
#[test]
fn a_rods_top_rim_rounds_off_above_a_bore_in_its_wall() {
    let (plain, plain_solid, plain_before) = round_a_sloped_rods_top(0.0, 0.0, 2.0).unwrap();
    let removed = plain_before - solid_volume(&plain, plain_solid, 0.001).unwrap();
    let (topo, solid, before) = round_a_sloped_rods_top(3.0, 0.5, 2.0).unwrap();
    assert_sound_and_changed_by(&topo, solid, before, -removed);
}
