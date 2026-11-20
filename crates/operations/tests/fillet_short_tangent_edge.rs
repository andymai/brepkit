//! A face's outline rounded where it runs on round a short edge tangent to
//! its neighbour and then turns a convex corner.
//!
//! A prism whose top outline runs along a straight side, on round a short
//! B-spline leaving it tangent, then turns up to the far side. Its whole top
//! outline is rounded. The rounds of the straight side and of the side past
//! the corner meet as the two rounds of one corner, crossing a radius's
//! setback from it. A radius past the short edge's length puts that crossing
//! beyond the short edge, on the straight side's contact. The trim ended
//! each contact on its own neighbour's, so the top face came out a loop that
//! crossed itself, a lobe of it walked backwards: valid and watertight, but
//! not the part. Such a round is now refused; smaller ones still build.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The prism's height.
const H: f64 = 20.0;

/// The prism: along `y = 0` to `x = 40`, on round a quadratic B-spline
/// leaving the side tangent and ending `(2, 0.1)` further on, then up to
/// `(60, 30)` and back along `y = 30`; extruded `H` up `z`.
fn prism(topo: &mut Topology) -> SolidId {
    let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let corners = [
        p(0.0, 0.0),
        p(40.0, 0.0),
        p(42.0, 0.1),
        p(60.0, 30.0),
        p(0.0, 30.0),
    ];
    let ids: Vec<_> = corners
        .iter()
        .map(|&q| topo.add_vertex(Vertex::new(q, 1e-7)))
        .collect();
    let spline = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![p(40.0, 0.0), p(41.0, 0.0), p(42.0, 0.1)],
        vec![1.0; 3],
    )
    .unwrap();
    let curves = [
        EdgeCurve::Line,
        EdgeCurve::NurbsCurve(spline),
        EdgeCurve::Line,
        EdgeCurve::Line,
        EdgeCurve::Line,
    ];
    let edges: Vec<_> = curves
        .into_iter()
        .enumerate()
        .map(|(i, curve)| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The edges of the prism's top face.
fn top_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - H).abs() < 1e-9)
        })
        .collect()
}

/// Whether the outer loop of any face of `solid` facing up or down crosses
/// itself, read from samples along its edges.
fn has_crossed_cap(topo: &Topology, solid: SolidId) -> bool {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .any(|face_id| {
            let face = topo.face(face_id).unwrap();
            let FaceSurface::Plane { normal, .. } = face.surface() else {
                return false;
            };
            if normal.z().abs() < 0.9 {
                return false;
            }
            let mut loop_points = Vec::new();
            for oriented in topo.wire(face.outer_wire()).unwrap().edges() {
                let edge = topo.edge(oriented.edge()).unwrap();
                let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                for k in 0..32 {
                    let f = f64::from(k) / 32.0;
                    let f = if oriented.is_forward() { f } else { 1.0 - f };
                    let q = edge
                        .curve()
                        .evaluate_with_endpoints(t0 + (t1 - t0) * f, a, b);
                    loop_points.push((q.x(), q.y()));
                }
            }
            let n = loop_points.len();
            let orient = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
                (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
            };
            (0..n).any(|i| {
                (i + 2..n).any(|j| {
                    if (j + 1) % n == i {
                        return false;
                    }
                    let (a, b) = (loop_points[i], loop_points[(i + 1) % n]);
                    let (c, d) = (loop_points[j], loop_points[(j + 1) % n]);
                    orient(a, b, c) * orient(a, b, d) < 0.0
                        && orient(c, d, a) * orient(c, d, b) < 0.0
                })
            })
        })
}

#[test]
fn rounds_within_the_short_edge_build() {
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let solid = prism(&mut topo);
        let edges = top_edges(&topo, solid);
        let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: the mesh is open");
        assert!(
            !has_crossed_cap(&topo, result.solid),
            "r = {r}: the top crosses itself"
        );
    }
}

#[test]
fn a_round_crossing_past_the_short_edge_is_not_built_crossed() {
    for r in [4.0, 5.0, 6.0, 8.0] {
        let mut topo = Topology::new();
        let solid = prism(&mut topo);
        let edges = top_edges(&topo, solid);
        if let Ok(result) = fillet_v2(&mut topo, solid, &edges, r) {
            assert!(
                !has_crossed_cap(&topo, result.solid),
                "r = {r}: the top face's loop crosses itself"
            );
        }
    }
}
