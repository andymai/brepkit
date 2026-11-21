//! Spline faces that run to a point, measured face by face.
//!
//! - A disc of radius 0.8 lofted to a point off its axis, 0.5 and 2 above
//!   it: the side is a ruled spline to the point, alone and on a box.
//! - A square lofted to a point off its middle.
//! - A cubic B-spline from a point on the z axis, revolved a whole turn.
//!
//! The mesh of a face like this meets the point in one vertex or a few,
//! each with one `(u, v)` of the whole side of the domain that the point
//! stands for. Each solid measures its closed form (Cavalieri, or the
//! revolved profile's `π x² dz`), upright, moved, and turned about an
//! oblique axis.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::loft::loft_to_point;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::revolve::revolve;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_circle_edge, make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn poses() -> [(&'static str, Mat4); 4] {
    let turn = Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3);
    [
        ("upright", Mat4::identity()),
        ("moved", Mat4::translation(20.0, 0.0, 0.0)),
        ("turned", turn),
        (
            "turned and moved",
            Mat4::translation(20.0, -7.0, 5.0) * turn,
        ),
    ]
}

/// The disc of radius 0.8 about `(5, 5, 10)` in the plane `z = 10`, lofted
/// to the point `(5, 5, 10) + apex`, and its volume.
fn cone(topo: &mut Topology, apex: Vec3) -> (SolidId, f64) {
    let centre = Point3::new(5.0, 5.0, 10.0);
    let rim = make_circle_edge(topo, centre, Vec3::new(0.0, 0.0, 1.0), 0.8, 1e-7).unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    let base = make_planar_face_from_wire(topo, wire).unwrap();
    let cone = loft_to_point(topo, base, centre + apex).unwrap();
    (cone, PI * 0.64 * apex.z() / 3.0)
}

/// The square of side 1.6 about `(5, 5, 10)` lofted to `(5.3, 5.2, 10.5)`.
fn pyramid(topo: &mut Topology) -> (SolidId, f64) {
    let corners = [
        Point3::new(4.2, 4.2, 10.0),
        Point3::new(5.8, 4.2, 10.0),
        Point3::new(5.8, 5.8, 10.0),
        Point3::new(4.2, 5.8, 10.0),
    ];
    let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
    let base = make_planar_face_from_wire(topo, wire).unwrap();
    let pyramid = loft_to_point(topo, base, Point3::new(5.3, 5.2, 10.5)).unwrap();
    (pyramid, 2.56 * 0.5 / 3.0)
}

/// The solid fused on the top of the box `[0, 10]^3`.
fn on_box(topo: &mut Topology, (solid, volume): (SolidId, f64)) -> (SolidId, f64) {
    let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    (
        boolean(topo, BooleanOp::Fuse, block, solid).unwrap(),
        1000.0 + volume,
    )
}

/// The profile between the z axis and a cubic B-spline from the origin to
/// `(1, 0, 1)`, closed along `z = 1`, revolved a whole turn about the axis.
/// Its volume is `π x² dz` along the spline, by Gauss on each knot span.
fn revolved(topo: &mut Topology) -> (SolidId, f64) {
    let through = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.15),
        Point3::new(0.85, 0.0, 0.5),
        Point3::new(1.0, 0.0, 1.0),
    ];
    let spline = brepkit_math::nurbs::fitting::interpolate(&through, 3).unwrap();
    let foot = topo.add_vertex(Vertex::new(through[0], 1e-7));
    let rim = topo.add_vertex(Vertex::new(through[3], 1e-7));
    let top = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), 1e-7));
    let edges = [
        topo.add_edge(Edge::new(foot, rim, EdgeCurve::NurbsCurve(spline.clone()))),
        topo.add_edge(Edge::new(rim, top, EdgeCurve::Line)),
        topo.add_edge(Edge::new(top, foot, EdgeCurve::Line)),
    ];
    let wire = topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let profile = make_planar_face_from_wire(topo, wire).unwrap();
    let solid = revolve(
        topo,
        profile,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        TAU,
    )
    .unwrap();
    let (t0, t1) = spline.domain();
    let mut ends = vec![t0];
    ends.extend(spline.knots().iter().copied().filter(|&k| k > t0 && k < t1));
    ends.push(t1);
    ends.dedup();
    let volume: f64 = ends
        .windows(2)
        .map(|span| {
            let (half, mid) = (0.5 * (span[1] - span[0]), 0.5 * (span[1] + span[0]));
            gauss_legendre_points(20)
                .iter()
                .map(|g| {
                    let t = half.mul_add(g.x, mid);
                    let x = spline.evaluate(t).x();
                    g.w * half * PI * x * x * spline.derivatives(t, 1)[1].z()
                })
                .sum::<f64>()
        })
        .sum();
    (solid, volume)
}

fn measures(build: &dyn Fn(&mut Topology) -> (SolidId, f64), within: f64, name: &str) {
    for (pose, m) in poses() {
        let mut topo = Topology::new();
        let (solid, truth) = build(&mut topo);
        transform_solid(&mut topo, solid, &m).unwrap();
        let volume = solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < within * truth,
            "{name}, {pose}: volume {volume}, truth {truth}"
        );
    }
}

#[test]
fn a_disc_lofted_to_a_point_off_its_axis_measures_its_cone() {
    for apex in [Vec3::new(0.3, 0.2, 0.5), Vec3::new(0.5, 0.0, 2.0)] {
        let name = format!("apex {apex:?}");
        measures(&|t| cone(t, apex), 1e-4, &name);
        measures(
            &|t| {
                let c = cone(t, apex);
                on_box(t, c)
            },
            1e-4,
            &format!("{name} on a box"),
        );
    }
}

#[test]
fn a_square_lofted_to_a_point_measures_its_pyramid() {
    measures(&pyramid, 1e-9, "pyramid");
    measures(
        &|t| {
            let p = pyramid(t);
            on_box(t, p)
        },
        1e-9,
        "pyramid on a box",
    );
}

#[test]
fn a_spline_from_the_axis_revolved_measures_its_solid() {
    measures(&revolved, 1e-9, "revolved");
}
