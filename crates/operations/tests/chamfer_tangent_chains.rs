//! Chamfers along tangent chains of lines and arcs on one flat cap: an
//! obround prism's bottom edges and an obround pocket's floor and mouth.
//!
//! A chamfer of distance `d` sweeps a right triangle of area `d^2 / 2` along
//! the chain, so it removes (or, on a concave edge, adds) that area times the
//! length of its centroid's path: `L` along a line of length `L`, and
//! `theta * (R -+ d / 3)` around an arc of radius `R`. A chain that stops at a
//! tangent neighbour ends in a flat triangle across the edge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const LENGTH: f64 = 40.0;
const RADIUS: f64 = 8.0;
const HEIGHT: f64 = 10.0;

/// The obround's outline area: a 40 x 16 rectangle and a disc of radius 8.
const AREA: f64 = LENGTH * 2.0 * RADIUS + PI * RADIUS * RADIUS;

/// One half-turn arc's sweep at chamfer distance 1 and centroid radius
/// `8 -+ 1 / 3`.
fn half_turn(centroid_radius: f64) -> f64 {
    0.5 * PI * centroid_radius
}

/// An obround prism, `z` from `z0` up by `height`, whose ends are single half
/// circles about `(0, 0)` and `(40, 0)`, its profile wound either way.
fn obround(topo: &mut Topology, clockwise: bool, z0: f64, height: f64) -> SolidId {
    let p = |t: &mut Topology, x, y| t.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7));
    let a = p(topo, 0.0, -RADIUS);
    let b = p(topo, 0.0, RADIUS);
    let c = p(topo, LENGTH, RADIUS);
    let d = p(topo, LENGTH, -RADIUS);
    let circle = |cx: f64, turn: f64| {
        EdgeCurve::Circle(
            Circle3D::new(Point3::new(cx, 0.0, z0), Vec3::new(0.0, 0.0, turn), RADIUS).unwrap(),
        )
    };
    let sides = if clockwise {
        vec![
            (a, b, circle(0.0, -1.0)),
            (b, c, EdgeCurve::Line),
            (c, d, circle(LENGTH, -1.0)),
            (d, a, EdgeCurve::Line),
        ]
    } else {
        vec![
            (b, a, circle(0.0, 1.0)),
            (a, d, EdgeCurve::Line),
            (d, c, circle(LENGTH, 1.0)),
            (c, b, EdgeCurve::Line),
        ]
    };
    let edges: Vec<_> = sides
        .into_iter()
        .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true))
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 0.0, if clockwise { -1.0 } else { 1.0 });
    let plane = FaceSurface::Plane {
        normal,
        d: normal.dot(Vec3::new(0.0, 0.0, z0)),
    };
    let face = topo.add_face(Face::new(wire, Vec::new(), plane));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap()
}

/// The edges of `solid` whose both ends lie at height `z` and pass `keep`.
fn edges_at(
    topo: &Topology,
    solid: SolidId,
    z: f64,
    keep: impl Fn(Point3, Point3) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let a = topo.vertex(edge.start()).unwrap().point();
            let b = topo.vertex(edge.end()).unwrap().point();
            (a.z() - z).abs() < 1e-9 && (b.z() - z).abs() < 1e-9 && keep(a, b)
        })
        .collect()
}

fn is_line(topo: &Topology, e: EdgeId) -> bool {
    matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Line)
}

/// Upright, turned about one axis and about all three, and mirrored.
fn poses() -> Vec<(&'static str, Mat4)> {
    vec![
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        ("turned", Mat4::rotation_z(0.7)),
        (
            "tilted",
            Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// Valid, watertight, oriented outward, and `expected` in volume.
fn assert_exact(topo: &Topology, result: SolidId, expected: f64, what: &str) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - expected).abs() < 1e-6,
        "{what}: volume {exact}, expected {expected}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let meshed = oriented_solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected).abs() < 1e-3 * expected,
        "{what}: meshed volume {meshed}, expected {expected}"
    );
}

/// How many of the faces of `solid` are not planes, and how many are.
fn census(topo: &Topology, solid: SolidId) -> (usize, usize) {
    let faces = solid_faces(topo, solid).unwrap();
    let curved = faces
        .iter()
        .filter(|&&f| !matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
        .count();
    (curved, faces.len() - curved)
}

/// Whether `p`, placed by `pose`, reads as inside `solid`.
fn inside(topo: &Topology, solid: SolidId, pose: &Mat4, p: Point3) -> bool {
    let at = pose.mul_point(p);
    classify_point(topo, solid, at, &ClassifyOptions::default()).unwrap()
        == PointClassification::Inside
}

/// One straight bottom edge, ending at a tangent arc at either end: a prism of
/// the triangle comes off, flat at both ends.
#[test]
fn one_straight_edge_takes_off_its_prism() {
    let plain = AREA * HEIGHT;
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = obround(&mut topo, clockwise, 0.0, HEIGHT);
            let edge = edges_at(&topo, solid, 0.0, |a, b| a.y() > 0.0 && b.y() > 0.0)[0];
            assert!(is_line(&topo, edge));
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &[edge], 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain - 20.0, &what);
            assert_eq!(census(&topo, result), (2, 7), "{what}");
            // In the prism, over the kept floor, and just past its flat end.
            assert!(
                !inside(&topo, result, &pose, Point3::new(20.0, 7.8, 0.1)),
                "{what}"
            );
            assert!(
                inside(&topo, result, &pose, Point3::new(20.0, 7.2, 0.5)),
                "{what}"
            );
            assert!(
                inside(&topo, result, &pose, Point3::new(40.5, 7.8, 0.1)),
                "{what}"
            );
        }
    }
}

/// The two straight bottom edges, two separate chains on one cap.
#[test]
fn two_straight_edges_are_two_chains() {
    let plain = AREA * HEIGHT;
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = obround(&mut topo, clockwise, 0.0, HEIGHT);
            let edges: Vec<_> = edges_at(&topo, solid, 0.0, |_, _| true)
                .into_iter()
                .filter(|&e| is_line(&topo, e))
                .collect();
            assert_eq!(edges.len(), 2);
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &edges, 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain - 40.0, &what);
            assert_eq!(census(&topo, result), (2, 10), "{what}");
        }
    }
}

/// One half-circle bottom edge alone: a cone band between two flat ends.
#[test]
fn one_arc_takes_off_its_half_ring() {
    let plain = AREA * HEIGHT;
    let arc = half_turn(RADIUS - 1.0 / 3.0);
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = obround(&mut topo, clockwise, 0.0, HEIGHT);
            let edge = edges_at(&topo, solid, 0.0, |a, b| a.x() > 1.0 && b.x() > 1.0)
                .into_iter()
                .find(|&e| !is_line(&topo, e))
                .unwrap();
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &[edge], 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain - arc, &what);
            assert_eq!(census(&topo, result), (3, 6), "{what}");
            assert!(
                !inside(&topo, result, &pose, Point3::new(47.8, 0.0, 0.1)),
                "{what}"
            );
            assert!(
                inside(&topo, result, &pose, Point3::new(47.2, 0.0, 0.5)),
                "{what}"
            );
        }
    }
}

/// The whole bottom loop closes on itself, so nothing ends in a flat triangle:
/// two planes and two cone bands, and one floor, wall set and top as before.
#[test]
fn the_bottom_loop_closes_on_itself() {
    let plain = AREA * HEIGHT;
    let removed = 40.0 + 2.0 * half_turn(RADIUS - 1.0 / 3.0);
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = obround(&mut topo, clockwise, 0.0, HEIGHT);
            let edges = edges_at(&topo, solid, 0.0, |_, _| true);
            assert_eq!(edges.len(), 4);
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &edges, 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain - removed, &what);
            assert_eq!(census(&topo, result), (4, 6), "{what}");
        }
    }
}

/// Both rims: the walls and their rulings shorten from both ends.
#[test]
fn both_rims_shorten_the_walls_from_both_ends() {
    let plain = AREA * HEIGHT;
    let removed = 2.0 * (40.0 + 2.0 * half_turn(RADIUS - 1.0 / 3.0));
    for clockwise in [true, false] {
        let mut topo = Topology::new();
        let solid = obround(&mut topo, clockwise, 0.0, HEIGHT);
        let mut edges = edges_at(&topo, solid, 0.0, |_, _| true);
        edges.extend(edges_at(&topo, solid, HEIGHT, |_, _| true));

        let result = chamfer(&mut topo, solid, &edges, 1.0).unwrap();

        assert_exact(
            &topo,
            result,
            plain - removed,
            &format!("clockwise {clockwise}"),
        );
        assert_eq!(census(&topo, result), (6, 8));
    }
}

/// A block holding an obround pocket 10 deep, its floor at `z = 15`.
fn pocketed_block(topo: &mut Topology, clockwise: bool) -> (SolidId, f64) {
    let block = make_box(topo, 100.0, 60.0, 25.0).unwrap();
    transform_solid(topo, block, &Mat4::translation(-30.0, -30.0, 0.0)).unwrap();
    let tool = obround(topo, clockwise, 15.0, 15.0);
    let result = boolean(topo, BooleanOp::Cut, block, tool).unwrap();
    (result, 100.0 * 60.0 * 25.0 - AREA * 10.0)
}

fn over_the_pocket(a: Point3, b: Point3) -> bool {
    let near = |p: Point3| p.x().abs() < 60.0 && p.x() > -20.0 && p.y().abs() < 20.0;
    near(a) && near(b)
}

/// A concave floor loop fills its ring, which lies inside the pocket, so its
/// centroid runs at `8 - 1 / 3` about each arc's axis.
#[test]
fn a_pockets_floor_loop_adds_its_ring() {
    let added = 40.0 + 2.0 * half_turn(RADIUS - 1.0 / 3.0);
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (solid, plain) = pocketed_block(&mut topo, clockwise);
            let edges = edges_at(&topo, solid, 15.0, over_the_pocket);
            assert_eq!(edges.len(), 4);
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &edges, 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain + added, &what);
            assert!(
                inside(&topo, result, &pose, Point3::new(20.0, 7.8, 15.1)),
                "{what}"
            );
            assert!(
                !inside(&topo, result, &pose, Point3::new(20.0, 6.5, 15.5)),
                "{what}"
            );
        }
    }
}

/// A mouth loop is a hole of the block's top: its cap moves out, away from the
/// pocket, so the centroid runs at `8 + 1 / 3`.
#[test]
fn a_pockets_mouth_loop_is_a_hole_of_its_cap() {
    let removed = 40.0 + 2.0 * half_turn(RADIUS + 1.0 / 3.0);
    for clockwise in [true, false] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (solid, plain) = pocketed_block(&mut topo, clockwise);
            // The boolean's mouth rim can be split further at seam vertices.
            let edges = edges_at(&topo, solid, 25.0, over_the_pocket);
            assert!(edges.len() >= 4);
            transform_solid(&mut topo, solid, &pose).unwrap();

            let result = chamfer(&mut topo, solid, &edges, 1.0).unwrap();

            let what = format!("{name} clockwise {clockwise}");
            assert_exact(&topo, result, plain - removed, &what);
            assert!(
                !inside(&topo, result, &pose, Point3::new(20.0, 8.2, 24.9)),
                "{what}"
            );
            assert!(
                inside(&topo, result, &pose, Point3::new(20.0, 9.5, 24.5)),
                "{what}"
            );
        }
    }
}

/// A distance that would swallow an arc's cap or a wall is refused rather than
/// built inside out.
#[test]
fn a_distance_past_the_arc_or_the_wall_is_refused() {
    let mut topo = Topology::new();
    let solid = obround(&mut topo, false, 0.0, HEIGHT);
    let edges = edges_at(&topo, solid, 0.0, |_, _| true);
    assert!(chamfer(&mut topo, solid, &edges, RADIUS + 1.0).is_err());
    assert!(chamfer(&mut topo, solid, &edges, HEIGHT + 1.0).is_err());
}

/// A selection no exact path takes must not come back with the solid's arcs
/// flattened to chords: the planar rebuild reads only a face's corners.
#[test]
fn the_planar_rebuild_refuses_curved_edges() {
    let mut topo = Topology::new();
    // A D: a half disc about the origin closed by a straight back wall, whose
    // back edge ends at a sharp corner with the arc's wall.
    let a = topo.add_vertex(Vertex::new(Point3::new(0.0, -5.0, 0.0), 1e-7));
    let b = topo.add_vertex(Vertex::new(Point3::new(0.0, 5.0, 0.0), 1e-7));
    let arc = EdgeCurve::Circle(
        Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap(),
    );
    let edges = vec![
        OrientedEdge::new(topo.add_edge(Edge::new(a, b, arc)), true),
        OrientedEdge::new(topo.add_edge(Edge::new(b, a, EdgeCurve::Line)), true),
    ];
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    let face = topo.add_face(Face::new(wire, Vec::new(), plane));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
    // The vertical edge at the corner between the wall and the arc's wall.
    let corner = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let p = topo.vertex(edge.start()).unwrap().point();
            let q = topo.vertex(edge.end()).unwrap().point();
            (p.z() - q.z()).abs() > 1.0 && p.y() > 0.0
        })
        .unwrap();

    let refused = chamfer(&mut topo, solid, &[corner], 0.5);

    assert!(
        refused.is_err(),
        "a chamfer through the planar rebuild read {:?}",
        refused.map(|s| solid_volume(&topo, s, 0.001))
    );
}

/// A hole the rebuild can carry goes through it: a drilled plate's far edge
/// comes off without closing the hole over.
#[test]
fn the_planar_rebuild_keeps_a_hole_it_does_not_reach() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
    let inner = make_box(&mut topo, 4.0, 4.0, 8.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(8.0, 8.0, -2.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, plate, inner).unwrap();
    let edge = edges_at(&topo, drilled, 4.0, |a, b| a.y() < 1e-9 && b.y() < 1e-9)[0];

    let result = chamfer(&mut topo, drilled, &[edge], 0.5).unwrap();

    assert_exact(
        &topo,
        result,
        20.0 * 20.0 * 4.0 - 4.0 * 4.0 * 4.0 - 0.5 * 0.5 * 0.5 * 20.0,
        "drilled plate",
    );
}

/// One the chamfer does reach is refused rather than closed over.
#[test]
fn the_planar_rebuild_refuses_a_hole_it_reaches() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
    let inner = make_box(&mut topo, 4.0, 4.0, 8.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(8.0, 8.0, -2.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, plate, inner).unwrap();
    let rim: Vec<_> = edges_at(&topo, drilled, 4.0, |a, b| {
        let held = |p: Point3| p.x() > 1.0 && p.x() < 19.0 && p.y() > 1.0 && p.y() < 19.0;
        held(a) && held(b)
    });
    assert_eq!(rim.len(), 4);

    assert!(chamfer(&mut topo, drilled, &rim, 0.5).is_err());
}

/// A hole within a bevel's set-back of its edge, though the bevel shares no
/// edge or vertex with it: the rebuild would leave the hole's walls crossing
/// the new face.
#[test]
fn the_planar_rebuild_refuses_a_hole_inside_the_bevel() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
    let inner = make_box(&mut topo, 4.0, 0.5, 8.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(8.0, 0.25, -2.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, plate, inner).unwrap();
    let edge = edges_at(&topo, drilled, 4.0, |a, b| a.y() < 1e-9 && b.y() < 1e-9)[0];

    assert!(chamfer(&mut topo, drilled, &[edge], 1.0).is_err());
}

/// A straight wall opened just above the rim the chain would raise past the
/// opening: refused, not built with the opening under its new rim.
#[test]
fn a_chain_refuses_a_wall_opening_inside_the_bevel() {
    let mut topo = Topology::new();
    let solid = obround(&mut topo, false, 0.0, HEIGHT);
    let edge = edges_at(&topo, solid, 0.0, |a, b| a.y() > 0.0 && b.y() > 0.0)[0];
    assert!(is_line(&topo, edge));
    let window = make_box(&mut topo, 4.0, 4.0, 0.4).unwrap();
    transform_solid(&mut topo, window, &Mat4::translation(18.0, 6.0, 0.3)).unwrap();
    let opened = boolean(&mut topo, BooleanOp::Cut, solid, window).unwrap();
    let edge = edges_at(&topo, opened, 0.0, |a, b| {
        a.y() > 0.0 && b.y() > 0.0 && (a.x() - b.x()).abs() > 1.0
    })[0];

    assert!(chamfer(&mut topo, opened, &[edge], 1.0).is_err());
}

/// A shaft through the cap inside the strip the chain would take.
#[test]
fn a_chain_refuses_a_cap_hole_inside_the_bevel() {
    let mut topo = Topology::new();
    let solid = obround(&mut topo, false, 0.0, HEIGHT);
    let shaft = make_box(&mut topo, 4.0, 1.0, HEIGHT + 4.0).unwrap();
    transform_solid(&mut topo, shaft, &Mat4::translation(18.0, 6.5, -2.0)).unwrap();
    let opened = boolean(&mut topo, BooleanOp::Cut, solid, shaft).unwrap();
    let edge = edges_at(&topo, opened, 0.0, |a, b| {
        a.y() > 7.5 && b.y() > 7.5 && (a.x() - b.x()).abs() > 1.0
    })[0];

    assert!(chamfer(&mut topo, opened, &[edge], 1.0).is_err());
}

/// Two chains on one cap whose strips overlap: the second finds the first's
/// new edge in its way.
#[test]
fn two_chains_must_clear_each_other() {
    let mut topo = Topology::new();
    let solid = obround(&mut topo, false, 0.0, HEIGHT);
    let edges: Vec<_> = edges_at(&topo, solid, 0.0, |_, _| true)
        .into_iter()
        .filter(|&e| is_line(&topo, e))
        .collect();
    // The straight sides stand 16 apart, so at 8.5 their strips overlap by 1.
    assert!(chamfer(&mut topo, solid, &edges, 8.5).is_err());
}

/// A chain keeps a cavity, and the cavity's faces keep their trims.
#[test]
fn a_chain_keeps_a_cavity_and_its_pcurves() {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};
    use brepkit_topology::pcurve::PCurve;

    let mut topo = Topology::new();
    let solid = obround(&mut topo, false, 0.0, HEIGHT);
    let edges = edges_at(&topo, solid, 0.0, |_, _| true);
    let core = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    transform_solid(&mut topo, core, &Mat4::translation(18.0, -2.0, 3.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, solid, core).unwrap();
    let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
    let face = topo.shell(cavity).unwrap().faces()[0];
    let rim = brepkit_topology::explorer::face_edges(&topo, face).unwrap()[0];
    let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
    topo.pcurves_mut()
        .set(rim, face, PCurve::new(Curve2D::Line(line), 0.0, 1.0));
    // The boolean's cut can split the rim; it is the same four-sided loop.
    let loop_edges = edges_at(&topo, hollow, 0.0, |_, _| true);
    assert!(loop_edges.len() >= edges.len());

    let result = chamfer(&mut topo, hollow, &loop_edges, 1.0).unwrap();

    let removed = 40.0 + 2.0 * half_turn(RADIUS - 1.0 / 3.0);
    assert_exact(&topo, result, AREA * HEIGHT - 64.0 - removed, "hollow");
    let kept = topo.solid(result).unwrap().inner_shells()[0];
    assert_ne!(kept, cavity);
    let copied_face = topo.shell(kept).unwrap().faces()[0];
    assert_eq!(topo.pcurves().pcurves_for_face(copied_face).len(), 1);
}
