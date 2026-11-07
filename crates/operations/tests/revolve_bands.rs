//! A revolve builds one face per profile edge, as the primitives do, not one
//! per quarter turn. A half disc turned whole about its diameter is the ball
//! `make_sphere` builds, so booleans against it stay exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn at(topo: &mut Topology, p: Point3) -> brepkit_topology::vertex::VertexId {
    topo.add_vertex(Vertex::new(p, 1e-7))
}

/// The half disc of radius `r` about `center`, its diameter along the unit
/// `axis` and its arc bulging along the unit `side`, across it. The arc can be
/// cut at its middle, and the wire can run the other way round.
fn half_disc(
    topo: &mut Topology,
    center: Point3,
    axis: Vec3,
    side: Vec3,
    r: f64,
    split: bool,
    backward: bool,
) -> FaceId {
    let normal = axis.cross(side);
    let top = at(topo, center + axis * r);
    let bottom = at(topo, center - axis * r);
    let circle = || EdgeCurve::Circle(Circle3D::new(center, normal, r).unwrap());
    let mut edges = Vec::new();
    if split {
        let middle = at(topo, center + side * r);
        edges.push(topo.add_edge(Edge::new(top, middle, circle())));
        edges.push(topo.add_edge(Edge::new(middle, bottom, circle())));
    } else {
        edges.push(topo.add_edge(Edge::new(top, bottom, circle())));
    }
    edges.push(topo.add_edge(Edge::new(bottom, top, EdgeCurve::Line)));
    let oriented: Vec<_> = if backward {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    } else {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    };
    let wire = topo.add_wire(Wire::new(oriented, true).unwrap());
    let plane_normal = if backward { -normal } else { normal };
    let d = plane_normal.dot(Vec3::new(center.x(), center.y(), center.z()));
    topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: plane_normal,
            d,
        },
    ))
}

/// A polygon face in the plane its corners lie in.
fn polygon(topo: &mut Topology, corners: &[Point3]) -> FaceId {
    let vs: Vec<_> = corners.iter().map(|&p| at(topo, p)).collect();
    let edges: Vec<_> = (0..vs.len())
        .map(|i| {
            OrientedEdge::new(
                topo.add_edge(Edge::new(vs[i], vs[(i + 1) % vs.len()], EdgeCurve::Line)),
                true,
            )
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = (corners[1] - corners[0])
        .cross(corners[2] - corners[0])
        .normalize()
        .unwrap();
    let c = corners[0];
    let d = normal.dot(Vec3::new(c.x(), c.y(), c.z()));
    topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal, d },
    ))
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut faces = BTreeMap::new();
    for face in solid_faces(topo, solid).unwrap() {
        *faces
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    faces
}

fn expected(faces: &[(&'static str, usize)]) -> BTreeMap<&'static str, usize> {
    faces.iter().copied().collect()
}

fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, tolerance: f64, what: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let measured = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < tolerance,
        "{what}: volume {measured}, expected {volume}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
}

/// Axes and sides to turn a half disc about: upright, either side, tilted.
fn frames() -> Vec<(Vec3, Vec3)> {
    let (x, y, z) = (
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let tilted = Vec3::new(1.0, 2.0, 2.0).normalize().unwrap();
    let across = tilted.cross(x).normalize().unwrap();
    vec![(z, x), (z, -y), (-z, x), (x, y), (tilted, across)]
}

/// Turned whole, every such profile is the primitive's two hemispheres.
#[test]
fn a_half_disc_turned_whole_is_the_ball() {
    let center = Point3::new(3.0, -4.0, 5.0);
    let r = 15.0;
    let ball = 4.0 / 3.0 * PI * r * r * r;
    for (axis, side) in frames() {
        for split in [false, true] {
            for backward in [false, true] {
                let mut topo = Topology::new();
                let face = half_disc(&mut topo, center, axis, side, r, split, backward);

                let solid = revolve(&mut topo, face, center, axis, 2.0 * PI).unwrap();

                let what = format!("axis {axis:?} split {split} backward {backward}");
                assert_eq!(census(&topo, solid), expected(&[("sphere", 2)]), "{what}");
                assert_sound(&topo, solid, ball, 1e-6, &what);
            }
        }
    }
}

/// The issue's cut: exact and matching the primitive's, where four bands fell
/// back to a mesh of 1448 faces.
#[test]
fn a_box_less_the_turned_ball_matches_the_primitive() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let (axis, side) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));

    let mut topo = Topology::new();
    let face = half_disc(&mut topo, center, axis, side, 15.0, false, false);
    let turned = revolve(&mut topo, face, center, axis, 2.0 * PI).unwrap();
    let block = make_box(&mut topo, 50.0, 50.0, 20.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-25.0, -25.0, -20.0)).unwrap();
    let cut = boolean(&mut topo, BooleanOp::Cut, block, turned).unwrap();

    let block = make_box(&mut topo, 50.0, 50.0, 20.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-25.0, -25.0, -20.0)).unwrap();
    let primitive = make_sphere(&mut topo, 15.0, 32).unwrap();
    let plain = boolean(&mut topo, BooleanOp::Cut, block, primitive).unwrap();

    assert_eq!(census(&topo, cut), expected(&[("plane", 6), ("sphere", 1)]));
    assert_eq!(census(&topo, cut), census(&topo, plain));
    let volume = solid_volume(&topo, plain, 0.001).unwrap();
    assert_sound(&topo, cut, volume, 1e-6, "box less the turned ball");
}

/// A half disc whose arc is not centred on the axis is no ball: it keeps the
/// general path.
#[test]
fn a_disc_off_its_centre_is_no_ball() {
    let mut topo = Topology::new();
    let (axis, side) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
    let face = half_disc(
        &mut topo,
        Point3::new(5.0, 0.0, 0.0),
        axis,
        side,
        2.0,
        false,
        false,
    );

    let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), axis, 2.0 * PI).unwrap();

    // Pappus: the half disc's area around its centroid's circle, 4 r / (3 pi)
    // past its diameter.
    let area = PI * 2.0 * 2.0 / 2.0;
    let centroid = 5.0 + 4.0 * 2.0 / (3.0 * PI);
    assert_sound(
        &topo,
        solid,
        2.0 * PI * centroid * area,
        1e-6,
        "off-centre half disc",
    );
    assert!(!census(&topo, solid).contains_key("sphere"));
}

/// A square turned three quarters: two walls, its two rings, and two ends, one
/// face each however many quarter turns they span.
#[test]
fn a_part_turn_builds_one_face_per_profile_edge() {
    let axis_point = Point3::new(0.0, 0.0, 0.0);
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let corners = [
        Point3::new(5.0, 0.0, -4.0),
        Point3::new(13.0, 0.0, -4.0),
        Point3::new(13.0, 0.0, 4.0),
        Point3::new(5.0, 0.0, 4.0),
    ];
    for angle in [PI, 1.5 * PI, 1.9 * PI] {
        let mut topo = Topology::new();
        let face = polygon(&mut topo, &corners);

        let solid = revolve(&mut topo, face, axis_point, axis, angle).unwrap();

        let what = format!("angle {angle}");
        assert_eq!(
            census(&topo, solid),
            expected(&[("cylinder", 2), ("plane", 4)]),
            "{what}"
        );
        assert_sound(&topo, solid, 64.0 * 9.0 * angle, 1e-6, &what);
    }
}

/// A band a boolean can then cut across: one face per edge stays exact.
#[test]
fn a_part_turn_cuts_exactly() {
    let corners = [
        Point3::new(5.0, 0.0, -4.0),
        Point3::new(13.0, 0.0, -4.0),
        Point3::new(13.0, 0.0, 4.0),
        Point3::new(5.0, 0.0, 4.0),
    ];
    let mut topo = Topology::new();
    let face = polygon(&mut topo, &corners);
    let solid = revolve(
        &mut topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        1.5 * PI,
    )
    .unwrap();
    // Everything above z = 1.
    let lid = make_box(&mut topo, 60.0, 60.0, 10.0).unwrap();
    transform_solid(&mut topo, lid, &Mat4::translation(-30.0, -30.0, 1.0)).unwrap();

    let cut = boolean(&mut topo, BooleanOp::Cut, solid, lid).unwrap();

    // The wall's two cylinders survive, where a mesh fallback leaves only planes.
    assert_eq!(census(&topo, cut).get("cylinder"), Some(&2));
    assert_sound(&topo, cut, 8.0 * 5.0 * 9.0 * 1.5 * PI, 1e-6, "cut");
}

/// A half disc turned part way stays in its quarter-turn bands: merged past
/// half a turn, a ball face around its poles meshes open. Those bands are exact
/// and watertight.
#[test]
fn a_half_disc_part_way_is_sound() {
    let center = Point3::new(0.0, 0.0, 0.0);
    let (axis, side) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
    for angle in [PI / 3.0, PI, 1.5 * PI] {
        let mut topo = Topology::new();
        let face = half_disc(&mut topo, center, axis, side, 15.0, false, false);

        let solid = revolve(&mut topo, face, center, axis, angle).unwrap();

        let volume = 4.0 / 3.0 * PI * 15.0 * 15.0 * 15.0 * angle / (2.0 * PI);
        assert_sound(&topo, solid, volume, 1e-6, &format!("angle {angle}"));
    }
}

/// A hole sweeps a passage however either wire is wound. Wound with the outer
/// wire it used to sweep as material, reading 1130.97 against 678.58 at a
/// quarter turn.
#[test]
fn a_holed_profile_sweeps_its_passage_however_it_is_wound() {
    let p = |x: f64, z: f64| Point3::new(x, 0.0, z);
    let loop_of = |topo: &mut Topology, corners: &[Point3]| {
        let vs: Vec<_> = corners.iter().map(|&c| at(topo, c)).collect();
        let edges: Vec<_> = (0..vs.len())
            .map(|i| {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(vs[i], vs[(i + 1) % vs.len()], EdgeCurve::Line)),
                    true,
                )
            })
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    };
    for outer_ccw in [true, false] {
        for hole_ccw in [true, false] {
            for angle in [PI / 2.0, PI, 1.5 * PI, 2.0 * PI] {
                let mut topo = Topology::new();
                let mut outer = vec![p(5.0, -4.0), p(13.0, -4.0), p(13.0, 4.0), p(5.0, 4.0)];
                if !outer_ccw {
                    outer.reverse();
                }
                let mut hole = vec![p(7.0, -2.0), p(11.0, -2.0), p(11.0, 2.0), p(7.0, 2.0)];
                if !hole_ccw {
                    hole.reverse();
                }
                let outer = loop_of(&mut topo, &outer);
                let hole = loop_of(&mut topo, &hole);
                let face = topo.add_face(Face::new(
                    outer,
                    vec![hole],
                    FaceSurface::Plane {
                        normal: Vec3::new(0.0, -1.0, 0.0),
                        d: 0.0,
                    },
                ));

                let solid = revolve(
                    &mut topo,
                    face,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    angle,
                )
                .unwrap();

                assert_sound(
                    &topo,
                    solid,
                    (64.0 - 16.0) * 9.0 * angle,
                    1e-6,
                    &format!("outer ccw {outer_ccw} hole ccw {hole_ccw} angle {angle}"),
                );
            }
        }
    }
}

/// A square with a square hole turned part way: the hole's four sides sweep
/// their own faces, one each, inside the outer ones.
#[test]
fn a_part_turn_of_a_holed_profile_keeps_its_hole() {
    let hole_corners = [
        Point3::new(7.0, 0.0, -2.0),
        Point3::new(7.0, 0.0, 2.0),
        Point3::new(11.0, 0.0, 2.0),
        Point3::new(11.0, 0.0, -2.0),
    ];
    let mut topo = Topology::new();
    let outer = polygon(
        &mut topo,
        &[
            Point3::new(5.0, 0.0, -4.0),
            Point3::new(13.0, 0.0, -4.0),
            Point3::new(13.0, 0.0, 4.0),
            Point3::new(5.0, 0.0, 4.0),
        ],
    );
    let vs: Vec<_> = hole_corners.iter().map(|&p| at(&mut topo, p)).collect();
    let edges: Vec<_> = (0..4)
        .map(|i| {
            OrientedEdge::new(
                topo.add_edge(Edge::new(vs[i], vs[(i + 1) % 4], EdgeCurve::Line)),
                true,
            )
        })
        .collect();
    let hole = topo.add_wire(Wire::new(edges, true).unwrap());
    topo.face_mut(outer).unwrap().inner_wires_mut().push(hole);

    let angle = 1.5 * PI;
    let solid = revolve(
        &mut topo,
        outer,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        angle,
    )
    .unwrap();

    assert_sound(&topo, solid, (64.0 - 16.0) * 9.0 * angle, 1e-6, "holed");
    assert_eq!(
        census(&topo, solid),
        expected(&[("cylinder", 4), ("plane", 6)]),
        "two end rings, and the outer and hole rings top and bottom"
    );
}

/// A rectangle with one side a half circle, part way: the side's torus band
/// merges with itself, and the volume holds.
#[test]
fn a_part_turn_merges_a_torus_band() {
    let mut topo = Topology::new();
    let a = at(&mut topo, Point3::new(1.0, 0.0, 0.0));
    let b = at(&mut topo, Point3::new(2.0, 0.0, 0.0));
    let c = at(&mut topo, Point3::new(2.0, 0.0, 1.0));
    let d = at(&mut topo, Point3::new(1.0, 0.0, 1.0));
    let side = EdgeCurve::Circle(
        Circle3D::new(Point3::new(2.0, 0.0, 0.5), Vec3::new(0.0, -1.0, 0.0), 0.5).unwrap(),
    );
    let edges = vec![
        OrientedEdge::new(topo.add_edge(Edge::new(a, b, EdgeCurve::Line)), true),
        OrientedEdge::new(topo.add_edge(Edge::new(b, c, side)), true),
        OrientedEdge::new(topo.add_edge(Edge::new(c, d, EdgeCurve::Line)), true),
        OrientedEdge::new(topo.add_edge(Edge::new(d, a, EdgeCurve::Line)), true),
    ];
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: 0.0,
        },
    ));

    let angle = 1.5 * PI;
    let solid = revolve(
        &mut topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        angle,
    )
    .unwrap();

    // Pappus: the rectangle's moment about the axis plus the half disc's.
    let half = PI * 0.25 / 2.0;
    let moment = 1.5 + half * (2.0 + 4.0 * 0.5 / (3.0 * PI));
    assert_sound(&topo, solid, moment * angle, 1e-6, "torus band");
    assert_eq!(
        census(&topo, solid),
        expected(&[("cylinder", 1), ("plane", 4), ("torus", 1)])
    );
}

/// Winding is read from the profile, so it holds wherever the axis's origin was
/// placed along the axis. Far along, a hole's chart read as degenerate and kept
/// its winding, which swept it as material.
#[test]
fn a_hole_reads_the_same_with_the_axis_origin_far_along() {
    let p = |x: f64, z: f64| Point3::new(x, 0.0, z);
    let loop_of = |topo: &mut Topology, corners: &[Point3]| {
        let vs: Vec<_> = corners.iter().map(|&c| at(topo, c)).collect();
        let edges: Vec<_> = (0..vs.len())
            .map(|i| {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(vs[i], vs[(i + 1) % vs.len()], EdgeCurve::Line)),
                    true,
                )
            })
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    };
    for outer_ccw in [true, false] {
        for hole_ccw in [true, false] {
            let mut topo = Topology::new();
            let mut outer = vec![p(5.0, -4.0), p(13.0, -4.0), p(13.0, 4.0), p(5.0, 4.0)];
            if !outer_ccw {
                outer.reverse();
            }
            let mut hole = vec![p(7.0, -2.0), p(11.0, -2.0), p(11.0, 2.0), p(7.0, 2.0)];
            if !hole_ccw {
                hole.reverse();
            }
            let outer = loop_of(&mut topo, &outer);
            let hole = loop_of(&mut topo, &hole);
            let face = topo.add_face(Face::new(
                outer,
                vec![hole],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, -1.0, 0.0),
                    d: 0.0,
                },
            ));

            let angle = 1.5 * PI;
            let solid = revolve(
                &mut topo,
                face,
                Point3::new(0.0, 0.0, -200_000.0),
                Vec3::new(0.0, 0.0, 1.0),
                angle,
            )
            .unwrap();

            // Rounding at a 200,000 origin; sweeping the hole as material reads
            // hundreds off.
            assert_sound(
                &topo,
                solid,
                (64.0 - 16.0) * 9.0 * angle,
                1e-4,
                &format!("far origin, outer ccw {outer_ccw} hole ccw {hole_ccw}"),
            );
        }
    }
}

/// A profile whose plane misses the axis: a straight side sweeps a hyperboloid,
/// not a cone. Each point `(x, z)` of the plane at `y = 4` turns on a circle of
/// radius `r = sqrt(x^2 + 16)`, and `dr = x / r dx`, so the volume is the turn
/// times the region's moment about `x = 0`, as if the plane held the axis.
#[test]
fn a_profile_off_the_axis_sweeps_its_true_surface() {
    let p = |x: f64, z: f64| Point3::new(x, 4.0, z);
    for (angle, label) in [(PI, "half turn"), (2.0 * PI, "whole turn")] {
        let mut topo = Topology::new();
        // A triangle with a slanted side, its area 8 and centroid at x = 13 / 3.
        let face = polygon(&mut topo, &[p(3.0, -2.0), p(7.0, 2.0), p(3.0, 2.0)]);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            angle,
        )
        .unwrap();

        let moment = 8.0 * 13.0 / 3.0;
        assert_sound(&topo, solid, moment * angle, 1e-2, label);
    }
}

/// A profile's far side square to the axis and its plane missing the axis: the
/// flat edge sweeps its plane, and only the slanted ones sweep hyperboloids.
#[test]
fn a_flat_edge_off_the_axis_sweeps_its_plane() {
    let mut topo = Topology::new();
    let face = polygon(
        &mut topo,
        &[
            Point3::new(3.0, 4.0, -1.0),
            Point3::new(7.0, 4.0, -1.0),
            Point3::new(7.0, 4.0, 1.0),
            Point3::new(3.0, 4.0, 1.0),
        ],
    );
    let origin = Point3::new(0.0, 0.0, 0.0);
    let solid = revolve(&mut topo, face, origin, Vec3::new(0.0, 0.0, 1.0), PI).unwrap();
    let faces = census(&topo, solid);
    assert_eq!(faces.get("plane"), Some(&4), "{faces:?}");
    assert!(validate_solid(&topo, solid).unwrap().is_valid());
    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh));
}

/// An arc centred on the axis that stops short of it clears the axis, so its
/// bands merge like any other: the circle it lies on reaches the axis, it
/// does not.
#[test]
fn an_arc_stopping_short_of_the_axis_merges() {
    let mut topo = Topology::new();
    let normal = Vec3::new(0.0, -1.0, 0.0);
    let center = Point3::new(0.0, 0.0, 0.0);
    let r: f64 = 2.0;
    let side = r * r - 1.0;
    let (low, high) = (
        at(&mut topo, Point3::new(side.sqrt(), 0.0, -1.0)),
        at(&mut topo, Point3::new(side.sqrt(), 0.0, 1.0)),
    );
    let circle = EdgeCurve::Circle(Circle3D::new(center, normal, r).unwrap());
    let arc = topo.add_edge(Edge::new(low, high, circle));
    let chord = topo.add_edge(Edge::new(high, low, EdgeCurve::Line));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)],
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    let solid = revolve(&mut topo, face, center, Vec3::new(0.0, 0.0, 1.0), 1.5 * PI).unwrap();
    assert_eq!(solid_faces(&topo, solid).unwrap().len(), 4);
    assert!(validate_solid(&topo, solid).unwrap().is_valid());
    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh));
}

/// A profile all but square to the axis: its caps stand 2e-4 off the flat band
/// they meet at an edge, too near for `unify_faces` to tell them apart, and
/// the bands stay where they are rather than take a cap into their plane.
#[test]
fn a_cap_all_but_flat_on_its_band_stays_its_own_face() {
    let mut topo = Topology::new();
    let tilt = 1e-4;
    let face = polygon(
        &mut topo,
        &[
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(13.0, 0.0, 0.0),
            Point3::new(13.0, 4.0, 4.0 * tilt),
            Point3::new(5.0, 4.0, 4.0 * tilt),
        ],
    );
    let origin = Point3::new(0.0, 0.0, 0.0);
    let solid = revolve(&mut topo, face, origin, Vec3::new(0.0, 0.0, 1.0), PI).unwrap();
    let caps = solid_faces(&topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| {
            matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, .. }
                if normal.dot(Vec3::new(0.0, 0.0, 1.0)).abs() < 1.0 - 1e-12)
        })
        .count();
    assert_eq!(caps, 2);
}
