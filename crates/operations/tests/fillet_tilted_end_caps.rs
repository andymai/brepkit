//! A round of a straight edge whose ends meet tilted end faces.
//!
//! The round's sections stand square to the edge, so where an end face is
//! tilted the end section stands off it. The round now runs on (or back) to
//! the end face: each contact to where it meets the end face's plane, on
//! the end face's corner edges, and the band ends on the cylinder's section
//! by that plane, an ellipse the end face is notched along. It had ended in
//! the section plane, closed by a runout patch, and the support face then
//! ran out along the patch's edge and back, which its mesh skipped: open.
//!
//! The round takes off the corner's strip, of area `r^2 (1 - pi / 4)`,
//! between the two end planes. The distance between them runs linearly
//! across the strip, so the volume is the strip's area times that distance
//! at the strip's centroid, `r (10 - 3 pi) / (12 - 3 pi)` from the corner.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const L: f64 = 20.0;
const W: f64 = 8.0;
const H: f64 = 6.0;

/// A prism: the polygon `points` (in a plane through them with normal
/// `normal`, wound either way) extruded `distance` along `normal`.
fn prism(topo: &mut Topology, points: &[Point3], normal: Vec3, distance: f64) -> SolidId {
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    for i in 1..points.len() - 1 {
        area += (points[i] - points[0]).cross(points[i + 1] - points[0]);
    }
    let mut points = points.to_vec();
    if area.dot(normal) < 0.0 {
        points.reverse();
    }
    let ids: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], EdgeCurve::Line);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let d = normal.dot(Vec3::new(points[0].x(), points[0].y(), points[0].z()));
    let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }));
    extrude(topo, face, normal, distance).unwrap()
}

/// The edge of `solid` from `a` to `b` (either way round).
fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            (near(p, a) && near(q, b)) || (near(p, b) && near(q, a))
        })
        .expect("the edge")
}

/// Rounds `edge` of `solid` by `r`, and checks the result is valid,
/// watertight, has the round and the notched ends' ellipses, and has lost
/// `removed`.
fn assert_rounds(mut topo: Topology, solid: SolidId, edge: EdgeId, r: f64, removed: f64) {
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces_before = solid_faces(&topo, solid).unwrap().len();
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    assert_eq!(
        solid_faces(&topo, result.solid).unwrap().len(),
        faces_before + 1,
        "r = {r}: the round alone is added"
    );
    let ellipses = solid_edges(&topo, result.solid)
        .unwrap()
        .into_iter()
        .filter(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
        .count();
    assert_eq!(ellipses, 2, "r = {r}: each end notched along an ellipse");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    assert!(
        (lost - removed).abs() <= 1e-6 * before,
        "r = {r}: lost {lost}, the strip {removed}"
    );
}

/// The corner strip's area and its centroid's distance from the corner.
fn strip(r: f64) -> (f64, f64) {
    (
        r * r * (1.0 - PI / 4.0),
        r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI),
    )
}

/// A prism over a trapezoid in the `xy` plane, its ends leaning in `y`: the
/// left end runs from `(0, 0)` to `(-left, W)`, the right from `(L, 0)` to
/// `(L + right, W)`. The edge rounded is its top front edge, along `x` at
/// `y = 0`, `z = H`.
#[test]
fn ends_leaning_across_the_round() {
    for (left, right) in [(3.0, 2.0), (-2.0, 3.0), (-3.0, -1.5)] {
        for r in [1.0, 2.5] {
            let mut topo = Topology::new();
            let solid = prism(
                &mut topo,
                &[
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(L, 0.0, 0.0),
                    Point3::new(L + right, W, 0.0),
                    Point3::new(-left, W, 0.0),
                ],
                Vec3::new(0.0, 0.0, 1.0),
                H,
            );
            let edge = edge_between(
                &topo,
                solid,
                Point3::new(0.0, 0.0, H),
                Point3::new(L, 0.0, H),
            );
            let (area, offset) = strip(r);
            let length = L + (left + right) * offset / W;
            assert_rounds(topo, solid, edge, r, area * length);
        }
    }
}

/// A prism over a trapezoid in the `xz` plane, its ends leaning in `z`: the
/// left end runs from `(0, 0)` to `(-left, H)`, the right from `(L, 0)` to
/// `(L + right, H)`. The edge rounded is its top front edge, along `x` at
/// `y = 0`, `z = H`, between the leaning ends' tops.
#[test]
fn ends_leaning_along_the_round() {
    for (left, right) in [(3.0, 2.0), (-2.0, 3.0), (-3.0, -1.5)] {
        for r in [1.0, 2.5] {
            let mut topo = Topology::new();
            let solid = prism(
                &mut topo,
                &[
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(L, 0.0, 0.0),
                    Point3::new(L + right, 0.0, H),
                    Point3::new(-left, 0.0, H),
                ],
                Vec3::new(0.0, 1.0, 0.0),
                W,
            );
            let edge = edge_between(
                &topo,
                solid,
                Point3::new(-left, 0.0, H),
                Point3::new(L + right, 0.0, H),
            );
            let (area, offset) = strip(r);
            let length = L + (left + right) * (H - offset) / H;
            assert_rounds(topo, solid, edge, r, area * length);
        }
    }
}
