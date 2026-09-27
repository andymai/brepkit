//! Revolving a profile with a half-circle side about an axis it clears, part
//! way or a full turn: the side sweeps a torus band, and the solid measures
//! its Pappus volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The rectangle over `1 < x < 2`, `0 < z < 1` in `y = 0`, one side a half
/// circle bulging out of it: the `x = 2` side away from the axis, or the
/// `x = 1` side toward it. The wire runs counterclockwise about `-y`, or the
/// other way round (each edge then used backward). Returns the face and its
/// Pappus volume per radian.
fn profile(topo: &mut Topology, toward_axis: bool, backward: bool) -> (FaceId, f64) {
    let v = |topo: &mut Topology, x: f64, z: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, 0.0, z), 1e-7))
    };
    let corners = [
        v(topo, 1.0, 0.0),
        v(topo, 2.0, 0.0),
        v(topo, 2.0, 1.0),
        v(topo, 1.0, 1.0),
    ];
    // The half circle on side `k` (from corner k to k + 1), counterclockwise
    // about `-y` like the wire, so it bulges out of the rectangle.
    let (side, center) = if toward_axis {
        (3, Point3::new(1.0, 0.0, 0.5))
    } else {
        (1, Point3::new(2.0, 0.0, 0.5))
    };
    let mut edges = Vec::new();
    for k in 0..4 {
        let curve = if k == side {
            EdgeCurve::Circle(Circle3D::new(center, Vec3::new(0.0, -1.0, 0.0), 0.5).unwrap())
        } else {
            EdgeCurve::Line
        };
        edges.push(topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], curve)));
    }
    let oriented: Vec<OrientedEdge> = if backward {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    } else {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    };
    let wire = topo.add_wire(Wire::new(oriented, true).unwrap());
    let normal = if backward { 1.0 } else { -1.0 };
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, normal, 0.0),
            d: 0.0,
        },
    ));
    let half = PI / 8.0;
    let offset = 2.0 / (3.0 * PI);
    let half_x = if toward_axis {
        1.0 - offset
    } else {
        2.0 + offset
    };
    (face, half.mul_add(half_x, 1.5))
}

/// Revolved 90, 180, 270 or 300 degrees, or a full turn, about z: each solid
/// is valid, meshes watertight and measures its Pappus volume within `1e-9`,
/// the side bulging away from the axis or toward it, the profile wound either
/// way. A partial turn's later rings carry the profile's own curves turned
/// about the axis (a half circle's chord in their place bounded the end cap
/// and the next band by the secant), and each torus band takes its side from
/// the arc's midpoint, where the chord's centre sits on the tube's core.
#[test]
fn a_half_circle_side_revolves_to_its_pappus_volume() {
    for toward_axis in [false, true] {
        for backward in [false, true] {
            for degrees in [90.0_f64, 180.0, 270.0, 300.0, 360.0] {
                let label = format!(
                    "{} the axis, {}, {degrees} degrees",
                    if toward_axis { "toward" } else { "away from" },
                    if backward {
                        "wound backward"
                    } else {
                        "wound forward"
                    }
                );
                let mut topo = Topology::new();
                let (face, per_radian) = profile(&mut topo, toward_axis, backward);
                let solid = revolve(
                    &mut topo,
                    face,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    degrees.to_radians(),
                )
                .unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let truth = per_radian * degrees.to_radians();
                let volume = solid_volume(&topo, solid, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
            }
        }
    }
}
