//! Rounds along a line that runs on round an arc, tangent only to a file's
//! precision.
//!
//! A slot's outline: a straight side, a half circle, a straight side, and a
//! square end. The first side is turned by a few 1e-8 rad, as a file's
//! digits leave a tangent junction, so it meets the arc with that kink. The
//! two stripes then end their contacts on the top face `radius * kink` apart
//! (2e-7 at r = 5). Read as two vertices, the two stripes' cross-sections
//! were not shared, and a zero-area patch closed the gap: one face too many,
//! and its mesh edges over-used.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The arc's radius.
const B: f64 = 18.5;
/// The straight sides' length.
const X: f64 = 23.5;
/// The slot's height.
const H: f64 = 37.0;

/// The slot's outline in z = 0, extruded to `H`: (0, -B) to (X, -B), a half
/// circle about (X, 0) to (X, B), back to (0, B), and the square end. The
/// first side starts `kink * X` lower, so it meets the arc `kink` off
/// tangent. With `reversed`, the arc is stored the other way round.
fn slot(topo: &mut Topology, kink: f64, reversed: bool) -> SolidId {
    let points = [(0.0, -B - kink * X), (X, -B), (X, B), (0.0, B)];
    let vertices: Vec<_> = points
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let axis = if reversed { -1.0 } else { 1.0 };
    let arc = Circle3D::new(Point3::new(X, 0.0, 0.0), Vec3::new(0.0, 0.0, axis), B).unwrap();
    let edges: Vec<_> = (0..4)
        .map(|i| {
            if i == 1 {
                let curve = EdgeCurve::Circle(arc.clone());
                if reversed {
                    let edge = topo.add_edge(Edge::new(vertices[2], vertices[1], curve));
                    return OrientedEdge::new(edge, false);
                }
                let edge = topo.add_edge(Edge::new(vertices[1], vertices[2], curve));
                return OrientedEdge::new(edge, true);
            }
            let edge = topo.add_edge(Edge::new(
                vertices[i],
                vertices[(i + 1) % 4],
                EdgeCurve::Line,
            ));
            OrientedEdge::new(edge, true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), H).unwrap()
}

/// Rounds the top edges `pick` selects at `radius` and checks the result:
/// `faces` faces, valid, watertight, and `removed` taken away.
fn assert_rounds(
    case: &str,
    kink: f64,
    reversed: bool,
    radius: f64,
    pick: impl Fn(Point3, Point3, &EdgeCurve) -> bool,
    faces: usize,
    removed: f64,
) {
    let mut topo = Topology::new();
    let solid = slot(&mut topo, kink, reversed);
    let edges: Vec<_> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            (start.z() - H).abs() < 1e-9
                && (end.z() - H).abs() < 1e-9
                && pick(start, end, edge.curve())
        })
        .collect();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, radius).unwrap();
    assert!(
        result.failed.is_empty(),
        "{case}: failed {:?}",
        result.failed
    );
    let count = solid_faces(&topo, result.solid).unwrap().len();
    assert_eq!(count, faces, "{case}: faces");
    assert!(
        validate_solid(&topo, result.solid).unwrap().is_valid(),
        "{case}: invalid"
    );
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    let taken = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    assert!(
        (taken - removed).abs() <= 1e-6 * removed,
        "{case}: removed {taken}, closed form {removed}"
    );
}

/// The volume a round of `radius` takes from a square edge: per unit length,
/// and the inset of that cross-section's centroid from the edge.
fn section(radius: f64) -> (f64, f64) {
    let area = (1.0 - PI / 4.0) * radius * radius;
    let inset = radius.powi(3) * (5.0 / 6.0 - PI / 4.0) / area;
    (area, inset)
}

#[test]
fn a_round_runs_on_round_an_arc_kinked_by_a_files_digits() {
    let first_side = |start: Point3, end: Point3, curve: &EdgeCurve| {
        matches!(curve, EdgeCurve::Circle(_)) || start.y() < 0.0 && end.y() < 0.0
    };
    let both_sides = |start: Point3, end: Point3, _: &EdgeCurve| start.x() > 1e-9 || end.x() > 1e-9;
    for kink in [0.0, 4e-8] {
        for reversed in [false, true] {
            for radius in [1.0, 5.0] {
                let (area, inset) = section(radius);
                let arc = PI * (B - inset);
                let case = format!("kink {kink}, reversed {reversed}, r {radius}");
                assert_rounds(
                    &format!("{case}, arc and one side"),
                    kink,
                    reversed,
                    radius,
                    first_side,
                    9,
                    area * (X + arc),
                );
                assert_rounds(
                    &format!("{case}, arc and both sides"),
                    kink,
                    reversed,
                    radius,
                    both_sides,
                    9,
                    area * (2.0 * X + arc),
                );
            }
        }
    }
}
