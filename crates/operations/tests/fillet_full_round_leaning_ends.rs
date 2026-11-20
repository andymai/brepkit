//! A slab's two short sides rounded on both rims at half the slab's width,
//! each a full round, where the long sides lean off square to them.
//!
//! Each round runs on to the long sides at its ends. Leaning about 24
//! degrees, a long side at the round's depth stands as far along the edge
//! as the edge is long, so one end of each contact runs back nearly its
//! whole length and the other runs on outward as far. The run-on read each
//! terminal against the contact as it stood: taken first, the end running
//! back found the cap a few percent from the far end, short of the far
//! end's own run-on, and was refused; that round ended in its section plane
//! and the result was left open. A terminal refused for a short run is now
//! tried again once the others have run on.
//!
//! Across the slab's thickness `2 r`, the rounds take off, at a side's
//! depth `n` below it, `h(y) = r - sqrt(r^2 - y^2)` of material; along the
//! side the long sides stand `w(n) = w0 + k n` apart, so a side loses
//! `w0 r^2 (2 - pi / 2) + k r^3 (10 / 3 - pi) / 2`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
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

const R: f64 = 150.0;

/// The quadrilateral `points` in the `xz` plane at `y = -R`, extruded `2 R`
/// along `y`.
fn slab(topo: &mut Topology, points: &[(f64, f64)]) -> SolidId {
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let mut points: Vec<Point3> = points.iter().map(|&(x, z)| Point3::new(x, -R, z)).collect();
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    for i in 1..points.len() - 1 {
        area += (points[i] - points[0]).cross(points[i + 1] - points[0]);
    }
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
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: -R },
    ));
    extrude(topo, face, normal, 2.0 * R).unwrap()
}

/// The edge of `solid` from `a` to `b` (either way round).
fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let near = |p: Point3, q: Point3| (p - q).length() < 1e-6;
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

/// What rounding the side from `a` to `b` takes off, the slab's outline
/// running on from `b` to `c` and from `a` back to `d`.
fn removed(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> f64 {
    let sub = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0, p.1 - q.1);
    let cross = |p: (f64, f64), q: (f64, f64)| p.0 * q.1 - p.1 * q.0;
    let side = sub(b, a);
    let w0 = side.0.hypot(side.1);
    let along = (side.0 / w0, side.1 / w0);
    // Inward, towards c and d.
    let inward = if cross(along, sub(c, a)) < 0.0 {
        (along.1, -along.0)
    } else {
        (-along.1, along.0)
    };
    // How far along the side each long side moves per unit of depth.
    let shift = |from: (f64, f64), to: (f64, f64)| {
        let run = sub(to, from);
        let depth = run.0 * inward.0 + run.1 * inward.1;
        (run.0 * along.0 + run.1 * along.1) / depth
    };
    let k = shift(b, c) - shift(a, d);
    w0 * R * R * (2.0 - PI / 2.0) + k * R.powi(3) * (10.0 / 3.0 - PI) / 2.0
}

#[test]
fn full_rounds_between_leaning_long_sides() {
    // A long thin quadrilateral whose long sides lean about 24 degrees off
    // square to its short ones, and its mirror image.
    let quad = [
        (160.128, 717.856),
        (214.492, 747.242),
        (476.509, 422.841),
        (445.793, 364.176),
    ];
    let mirrored = quad.map(|(x, z)| (-x, z));
    let poses = [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
    ];
    for (name, [a, b, c, d]) in [("quad", quad), ("mirrored", mirrored)] {
        for (turn, pose) in poses.iter().enumerate() {
            let case = format!("{name}, pose {turn}");
            let mut topo = Topology::new();
            let solid = slab(&mut topo, &[a, b, c, d]);
            let at = |(x, z): (f64, f64), y: f64| Point3::new(x, y, z);
            let picks: Vec<(Point3, Point3)> = [(a, b), (c, d)]
                .into_iter()
                .flat_map(|(p, q)| [-R, R].map(|y| (at(p, y), at(q, y))))
                .collect();
            transform_solid(&mut topo, solid, pose).unwrap();
            let edges: Vec<EdgeId> = picks
                .iter()
                .map(|&(p, q)| edge_between(&topo, solid, pose.mul_point(p), pose.mul_point(q)))
                .collect();
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let result = fillet_v2(&mut topo, solid, &edges, R).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 1.0).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            let faces = solid_faces(&topo, result.solid).unwrap();
            let cylinders = faces
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .count();
            // The four rounds; the short sides go.
            assert_eq!(cylinders, 4, "{case}: the rounds");
            assert_eq!(faces.len(), 8, "{case}: the short sides go");
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            let expected = removed(a, b, c, d) + removed(c, d, a, b);
            assert!(
                (lost - expected).abs() <= 1e-6 * before,
                "{case}: lost {lost}, the closed form {expected}"
            );
        }
    }
}
