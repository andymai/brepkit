//! A solid of NURBS faces cut by a tool with a flat face in a plane that
//! holds edges of those faces.
//!
//! The blank is a prism on a symmetric aerofoil section: an upper and a lower
//! NURBS face meeting along the leading and trailing edges, both in the plane
//! `y = 0`, and a flat cap at each end. A tool whose bottom face lies in that
//! plane leaves the lower half. The plane meets the two NURBS faces along
//! their own edges, where the face-face phase drops the section as being the
//! edge, so the tool's face is split along the edges themselves.
//!
//! Each cap takes its section along the chord, from the vertex at the nose
//! to the vertex at the trailing edge, where its two splines meet.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::loft::{loft, loft_smooth};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const CHORD: f64 = 40.0;
const SPAN: f64 = 60.0;

/// Half the section's thickness at `x` along its unit chord.
fn half_thickness(x: f64) -> f64 {
    0.425
        * (0.2969 * x.sqrt() - 0.126 * x - 0.3516 * x * x + 0.2843 * x.powi(3) - 0.1036 * x.powi(4))
}

/// An aerofoil section in the plane `x = at`, its chord along `z` from 2 to
/// 42 on `y = 0`: an upper and a lower spline, the lower `lower` times as
/// thick as the upper.
fn section(topo: &mut Topology, at: f64, lower: f64) -> FaceId {
    let n = 12;
    let point = |x: f64, side: f64| {
        Point3::new(
            at,
            side * half_thickness(x) * CHORD,
            (x - 0.5).mul_add(CHORD, 22.0),
        )
    };
    let xs: Vec<f64> = (0..=n)
        .map(|i| 0.5 * (1.0 - (std::f64::consts::PI * f64::from(i) / f64::from(n)).cos()))
        .collect();
    let upper: Vec<Point3> = xs.iter().map(|&x| point(x, 1.0)).collect();
    let lower: Vec<Point3> = xs.iter().rev().map(|&x| point(x, -lower)).collect();
    let le = topo.add_vertex(Vertex::new(upper[0], 1e-7));
    let te = topo.add_vertex(Vertex::new(upper[n as usize], 1e-7));
    let up = brepkit_math::nurbs::fitting::interpolate(&upper, 3).unwrap();
    let lo = brepkit_math::nurbs::fitting::interpolate(&lower, 3).unwrap();
    let e_up = topo.add_edge(Edge::new(le, te, EdgeCurve::NurbsCurve(up)));
    let e_lo = topo.add_edge(Edge::new(te, le, EdgeCurve::NurbsCurve(lo)));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(e_up, true), OrientedEdge::new(e_lo, true)],
            true,
        )
        .unwrap(),
    );
    topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(1.0, 0.0, 0.0),
            d: at,
        },
    ))
}

fn prism(topo: &mut Topology, lower: f64) -> SolidId {
    let sections = [
        section(topo, 0.0, lower),
        section(topo, SPAN / 2.0, lower),
        section(topo, SPAN, lower),
    ];
    loft_smooth(topo, &sections).unwrap()
}

/// The area of half the section, from a fine sum.
fn half_area() -> f64 {
    let n = 20_000;
    (0..n)
        .map(|i| half_thickness((f64::from(i) + 0.5) / f64::from(n)) / f64::from(n))
        .sum::<f64>()
        * CHORD
        * CHORD
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

/// `blank` less `tool`, both moved by `pose`, is the prism's lower part: its
/// lower NURBS face, the cut face in `y = 0`, and the lower part of each cap.
/// `lower` is the thickness of that part against the upper one.
fn leaves_the_lower_half(
    topo: &mut Topology,
    blank: SolidId,
    tool: SolidId,
    lower: f64,
    (name, pose): &(&str, Mat4),
) {
    transform_solid(topo, blank, pose).unwrap();
    transform_solid(topo, tool, pose).unwrap();
    let before = boolean::mesh_fallback_count();
    let half = boolean::boolean(topo, BooleanOp::Cut, blank, tool).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "mesh fallback, {name}"
    );
    let report = validate_solid(topo, half).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        census(topo, half),
        BTreeMap::from([("nurbs", 1), ("plane", 3)])
    );
    for x in [5.0, 30.0, 55.0] {
        for z in [8.0, 22.0, 36.0] {
            let class = |y: f64| {
                let p = pose.mul_point(Point3::new(x, y, z));
                classify_point(topo, half, p, 0.01, 1e-7).unwrap()
            };
            assert_eq!(
                class(-0.5 * lower),
                PointClassification::Inside,
                "below at {x}, {z}"
            );
            assert_eq!(
                class(0.5),
                PointClassification::Outside,
                "above at {x}, {z}"
            );
        }
    }
    let expected = half_area() * lower * SPAN;
    let volume = solid_volume(topo, half, 0.005).unwrap();
    assert!(
        (volume - expected).abs() < 2e-3 * expected,
        "volume {volume} should be {expected}"
    );
}

/// The poses each cut runs in: as built, turned off every axis, moved far
/// from the origin, and mirrored.
fn poses() -> [(&'static str, Mat4); 4] {
    [
        ("as built", Mat4::identity()),
        ("turned", Mat4::rotation_x(0.41) * Mat4::rotation_z(1.13)),
        (
            "shifted",
            Mat4::translation(1000.0, -7.3, 0.123) * Mat4::rotation_y(0.3),
        ),
        ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
    ]
}

/// A box on the chord plane, across a prism whose lower part is `lower`
/// times as thick as its upper part.
fn box_on_the_chord_plane(lower: f64) {
    for pose in poses() {
        let mut topo = Topology::new();
        let blank = prism(&mut topo, lower);
        let tool = make_box(&mut topo, 80.0, 50.0, 70.0).unwrap();
        transform_solid(&mut topo, tool, &Mat4::translation(-10.0, 0.0, -10.0)).unwrap();
        leaves_the_lower_half(&mut topo, blank, tool, lower, &pose);
    }
}

#[test]
fn a_box_on_the_chord_plane_leaves_the_lower_half() {
    // The box's bottom face also takes a section from each cap, so it has
    // sections of its own beside the two edges that lie in it.
    box_on_the_chord_plane(1.0);
}

#[test]
fn a_box_on_the_chord_plane_of_a_cambered_section_leaves_the_lower_part() {
    box_on_the_chord_plane(0.5);
}

/// A lens in the plane `y`, from (-40, 22) to (100, 22) in `(x, z)` and
/// back, each edge a quadratic spline bowed 60 off the chord.
fn lens(topo: &mut Topology, y: f64) -> FaceId {
    let a = topo.add_vertex(Vertex::new(Point3::new(-40.0, y, 22.0), 1e-7));
    let b = topo.add_vertex(Vertex::new(Point3::new(100.0, y, 22.0), 1e-7));
    let arc = |from: f64, to: f64, side: f64| {
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(from, y, 22.0),
                Point3::new(30.0, y, side.mul_add(120.0, 22.0)),
                Point3::new(to, y, 22.0),
            ],
            vec![1.0; 3],
        )
        .unwrap()
    };
    let out = topo.add_edge(Edge::new(
        a,
        b,
        EdgeCurve::NurbsCurve(arc(-40.0, 100.0, -1.0)),
    ));
    let back = topo.add_edge(Edge::new(
        b,
        a,
        EdgeCurve::NurbsCurve(arc(100.0, -40.0, 1.0)),
    ));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(out, true), OrientedEdge::new(back, true)],
            true,
        )
        .unwrap(),
    );
    topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: y,
        },
    ))
}

#[test]
fn a_tool_whose_flat_face_has_spline_edges_leaves_the_lower_half() {
    // The tool's bottom face is bounded by two splines, so its region is
    // read on sampled loops.
    for pose in poses() {
        let mut topo = Topology::new();
        let blank = prism(&mut topo, 1.0);
        let profiles = [lens(&mut topo, 0.0), lens(&mut topo, 50.0)];
        let tool = loft(&mut topo, &profiles).unwrap();
        leaves_the_lower_half(&mut topo, blank, tool, 1.0, &pose);
    }
}

#[test]
fn a_box_tilted_about_the_nose_leaves_the_part_under_its_face() {
    // The box's bottom face holds the leading edge alone. Its section on each
    // cap starts at the nose vertex and ends on the lower spline.
    let slope = 0.05_f64;
    let tilt = Mat4::translation(0.0, 0.0, 2.0)
        * Mat4::rotation_x(slope.atan())
        * Mat4::translation(0.0, 0.0, -2.0);
    // The cut face is y = -slope (z - 2).
    let n = 20_000;
    let area = (0..n)
        .map(|i| {
            let x = (f64::from(i) + 0.5) / f64::from(n);
            let h = half_thickness(x);
            ((-slope * x).min(h) + h).max(0.0) / f64::from(n)
        })
        .sum::<f64>()
        * CHORD
        * CHORD;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let blank = prism(&mut topo, 1.0);
        let tool = make_box(&mut topo, 80.0, 50.0, 70.0).unwrap();
        transform_solid(
            &mut topo,
            tool,
            &(tilt * Mat4::translation(-10.0, 0.0, -10.0)),
        )
        .unwrap();
        transform_solid(&mut topo, blank, &pose).unwrap();
        transform_solid(&mut topo, tool, &pose).unwrap();
        let before = boolean::mesh_fallback_count();
        let part = boolean::boolean(&mut topo, BooleanOp::Cut, blank, tool).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "mesh fallback, {name}"
        );
        let report = validate_solid(&topo, part).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(
            census(&topo, part),
            BTreeMap::from([("nurbs", 1), ("plane", 3)])
        );
        for x in [5.0, 30.0, 55.0] {
            for z in [8.0, 14.0, 20.0] {
                let class = |off: f64| {
                    let p = pose.mul_point(Point3::new(x, -slope * (z - 2.0) + off, z));
                    classify_point(&topo, part, p, 0.01, 1e-7).unwrap()
                };
                assert_eq!(
                    class(-0.1),
                    PointClassification::Inside,
                    "below at {x}, {z}"
                );
                assert_eq!(
                    class(0.1),
                    PointClassification::Outside,
                    "above at {x}, {z}"
                );
            }
        }
        let expected = area * SPAN;
        let volume = solid_volume(&topo, part, 0.005).unwrap();
        assert!(
            (volume - expected).abs() < 2e-3 * expected,
            "volume {volume} should be {expected}"
        );
    }
}
