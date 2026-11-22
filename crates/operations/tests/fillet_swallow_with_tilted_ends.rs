//! A round wider than the narrow face beside it, whose ends meet tilted end
//! faces.
//!
//! The part is a thin strip, a fan blade's tip in miniature: its plan is a
//! parallelogram `W` wide and `L` long whose two long sides are B-splines
//! (straight, so the walls are spline surfaces) and whose two short sides
//! lean by `S` the same way, extruded `H` up `z`. The top edge of one long
//! wall is rounded with a radius wider than the strip's top: the ball rests
//! on that wall and pivots on the far wall's top edge, the band meets the
//! far wall there, and the top face goes. At each end the band meets a
//! short side, a plane tilted to its end section.
//!
//! The narrow face was swallowed only where its end edges lay in the
//! round's end sections. Now the far edge is carried on past its ends so the
//! end sections still find their pivot, the band is carried on past its
//! ends and cut back to the end faces, its pivoted contact ending on the far
//! edge's corner, and each end face is notched along it.
//!
//! The volume taken off: every section square to the strip loses the same
//! piece, the corner of the `W` by `H` rectangle outside the ball's arc from
//! its contact on the wall up to the far wall's top corner, and the two end
//! faces lean the same way, so the piece runs `L` along the strip.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
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

const W: f64 = 0.5;
const L: f64 = 20.0;
const S: f64 = 0.4;
const H: f64 = 6.0;

fn point(x: f64, y: f64) -> Point3 {
    Point3::new(x, y, 0.0)
}

/// A straight B-spline from `a` to `b`.
fn straight_spline(a: Point3, b: Point3) -> EdgeCurve {
    let middle = Point3::new(
        0.5 * (a.x() + b.x()),
        0.5 * (a.y() + b.y()),
        0.5 * (a.z() + b.z()),
    );
    EdgeCurve::NurbsCurve(
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![a, middle, b],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap(),
    )
}

/// The strip.
fn strip(topo: &mut Topology) -> SolidId {
    let corners = [point(0.0, 0.0), point(L, 0.0), point(L + S, W), point(S, W)];
    let vertices = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let edges = [
        Edge::new(
            vertices[0],
            vertices[1],
            straight_spline(corners[0], corners[1]),
        ),
        Edge::new(vertices[1], vertices[2], EdgeCurve::Line),
        Edge::new(
            vertices[2],
            vertices[3],
            straight_spline(corners[2], corners[3]),
        ),
        Edge::new(vertices[3], vertices[0], EdgeCurve::Line),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The top edge of the wall at `y = 0`.
fn wall_top(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                (p.z() - H).abs() < 1e-9 && p.y().abs() < 1e-9
            })
        })
        .expect("the wall's top edge")
}

/// The section's piece the round takes off: the ball's centre is `r` in
/// from the wall, its arc rises from the wall to the far wall's top corner
/// `(W, H)`, and the piece is the rectangle's corner above that arc.
fn piece_area(r: f64) -> f64 {
    const STEPS: usize = 2000;
    let rise = (r * r - (r - W) * (r - W)).sqrt();
    let arc = |y: f64| (r * r - (r - y) * (r - y)).max(0.0).sqrt();
    #[allow(clippy::cast_precision_loss)]
    let step = W / STEPS as f64;
    let mut sum = arc(0.0) + arc(W);
    for k in 1..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let y = step * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * arc(y);
    }
    W * rise - sum * step / 3.0
}

fn round_and_check(r: f64) {
    let mut topo = Topology::new();
    let solid = strip(&mut topo);
    let edge = wall_top(&topo, solid);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces_before = solid_faces(&topo, solid).unwrap().len();

    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let faces = solid_faces(&topo, result.solid).unwrap().len();
    assert_eq!(
        faces, faces_before,
        "r = {r}: the band replaces the swallowed top face"
    );

    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let want = piece_area(r) * L;
    assert!(
        (lost - want).abs() <= 1e-4 * want,
        "r = {r}: lost {lost}, the round's sweep {want}"
    );
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
}

#[test]
fn a_round_swallows_a_narrow_face_between_tilted_end_faces() {
    for r in [0.8, 1.0, 1.25, 1.5, 2.0] {
        round_and_check(r);
    }
}

/// The mesh of the round's solid closes for each radius, deflection and
/// pose. The end face's mesh covers the chords between the notch arc's
/// samples. The band's mesh must not cover them too: its triangles at the
/// notch take points on its surface.
#[test]
fn the_round_s_mesh_closes_over_its_notched_ends() {
    let poses = [
        ("upright", Mat4::identity()),
        ("turned", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    for r in [0.8, 1.0, 1.1, 1.2, 1.25] {
        for (name, pose) in &poses {
            let mut topo = Topology::new();
            let solid = strip(&mut topo);
            let edge = wall_top(&topo, solid);
            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            transform_solid(&mut topo, result.solid, pose).unwrap();
            for deflection in [0.01, 0.002] {
                let mesh = tessellate_solid(&topo, result.solid, deflection).unwrap();
                assert!(
                    is_watertight(&mesh),
                    "r = {r}, {name}: mesh not watertight at {deflection}"
                );
            }
        }
    }
}
