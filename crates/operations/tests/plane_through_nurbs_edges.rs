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
//! The tool's face does split. The cut still fails on the caps, whose section
//! along the chord is not trimmed where their two splines meet (#2404), so
//! both tests wait on that.

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

/// A symmetric aerofoil section in the plane `x = at`, its chord along `z`
/// from 2 to 42 on `y = 0`: an upper and a lower spline.
fn section(topo: &mut Topology, at: f64) -> FaceId {
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
    let lower: Vec<Point3> = xs.iter().rev().map(|&x| point(x, -1.0)).collect();
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

fn prism(topo: &mut Topology) -> SolidId {
    let sections = [
        section(topo, 0.0),
        section(topo, SPAN / 2.0),
        section(topo, SPAN),
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

/// `blank` less `tool` is the prism's lower half: its lower NURBS face, the
/// cut face in `y = 0`, and half of each cap.
fn leaves_the_lower_half(topo: &mut Topology, blank: SolidId, tool: SolidId) {
    let before = boolean::mesh_fallback_count();
    let half = boolean::boolean(topo, BooleanOp::Cut, blank, tool).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(topo, half).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        census(topo, half),
        BTreeMap::from([("nurbs", 1), ("plane", 3)])
    );
    for x in [5.0, 30.0, 55.0] {
        for z in [8.0, 22.0, 36.0] {
            let class =
                |y: f64| classify_point(topo, half, Point3::new(x, y, z), 0.01, 1e-7).unwrap();
            assert_eq!(
                class(-0.5),
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
    let expected = half_area() * SPAN;
    let volume = solid_volume(topo, half, 0.005).unwrap();
    assert!(
        (volume - expected).abs() < 2e-3 * expected,
        "volume {volume} should be {expected}"
    );
}

#[test]
#[ignore = "the section on each cap is not trimmed at the nose: #2404"]
fn a_box_on_the_chord_plane_leaves_the_lower_half() {
    // The box's bottom face also takes a section from each cap, so it has
    // sections of its own beside the two edges that lie in it.
    let mut topo = Topology::new();
    let blank = prism(&mut topo);
    let tool = make_box(&mut topo, 80.0, 50.0, 70.0).unwrap();
    transform_solid(&mut topo, tool, &Mat4::translation(-10.0, 0.0, -10.0)).unwrap();
    leaves_the_lower_half(&mut topo, blank, tool);
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
#[ignore = "the section on each cap is not trimmed at the nose: #2404"]
fn a_tool_whose_flat_face_has_spline_edges_leaves_the_lower_half() {
    // The tool's bottom face is bounded by two splines, so its region is
    // read on sampled loops.
    let mut topo = Topology::new();
    let blank = prism(&mut topo);
    let profiles = [lens(&mut topo, 0.0), lens(&mut topo, 50.0)];
    let tool = loft(&mut topo, &profiles).unwrap();
    leaves_the_lower_half(&mut topo, blank, tool);
}
