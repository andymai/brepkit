//! A fillet on a body whose two blocks meet along one edge alone.
//!
//! Two boxes stand corner to corner, sharing one vertical edge, in one
//! shell: that edge has four faces. Rounding an edge of one box far from
//! it leaves the shared edge as it was, with its four faces. The fillet's
//! incidence audit took every edge it had not built to have two uses, so
//! it read the untouched edge as broken and refused the round. An edge the
//! fillet leaves alone now keeps the uses it had in the input.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::OrientedEdge;

const SIZE: f64 = 10.0;

fn at(topo: &Topology, v: VertexId) -> Point3 {
    topo.vertex(v).unwrap().point()
}

fn near(p: Point3, q: Point3) -> bool {
    (p - q).length() < 1e-9
}

/// The vertical edge of `solid` at `(x, y)`.
fn vertical(topo: &Topology, solid: SolidId, x: f64, y: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = at(topo, v);
                (p.x() - x).abs() < 1e-9 && (p.y() - y).abs() < 1e-9
            })
        })
        .expect("the vertical edge")
}

/// Two boxes of side `SIZE`, the second moved `(SIZE, SIZE)`, sharing the
/// vertical edge at `(SIZE, SIZE)`, as one solid of one shell.
fn corner_to_corner(topo: &mut Topology) -> SolidId {
    let a = make_box(topo, SIZE, SIZE, SIZE).unwrap();
    let b = make_box(topo, SIZE, SIZE, SIZE).unwrap();
    transform_solid(topo, b, &Mat4::translation(SIZE, SIZE, 0.0)).unwrap();

    let a_edge = vertical(topo, a, SIZE, SIZE);
    let b_edge = vertical(topo, b, SIZE, SIZE);
    let (a_start, a_end) = {
        let edge = topo.edge(a_edge).unwrap();
        (edge.start(), edge.end())
    };
    let shared = [a_start, a_end].map(|v| (v, at(topo, v)));

    // The second box's edges take the first's vertices at the shared edge.
    for e in solid_edges(topo, b).unwrap() {
        let (start, end) = {
            let edge = topo.edge(e).unwrap();
            (at(topo, edge.start()), at(topo, edge.end()))
        };
        for &(v, p) in &shared {
            if near(start, p) {
                topo.edge_mut(e).unwrap().set_start(v);
            }
            if near(end, p) {
                topo.edge_mut(e).unwrap().set_end(v);
            }
        }
    }
    // And the first's shared edge in place of their own.
    let b_faces = solid_faces(topo, b).unwrap();
    for &f in &b_faces {
        let wire = topo.face(f).unwrap().outer_wire();
        for oriented in topo.wire_mut(wire).unwrap().edges_mut() {
            if oriented.edge() == b_edge {
                *oriented = OrientedEdge::new(a_edge, oriented.is_forward());
            }
        }
    }

    let mut faces = solid_faces(topo, a).unwrap();
    faces.extend(b_faces);
    let shell = topo.add_shell(Shell::new(faces).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

#[test]
fn a_round_beside_an_edge_of_four_faces_builds() {
    let mut topo = Topology::new();
    let solid = corner_to_corner(&mut topo);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    assert!(
        (before - 2.0 * SIZE.powi(3)).abs() < 1e-6,
        "volume {before}"
    );

    // The first box's vertical edge at the origin, far from the shared one.
    let r = 2.0;
    let edge = vertical(&topo, solid, 0.0, 0.0);
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);

    let faces = solid_faces(&topo, result.solid).unwrap().len();
    assert_eq!(faces, 13, "the round added, the rest as they were");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let want = (4.0 - PI) * r * r / 4.0 * SIZE;
    assert!(
        (lost - want).abs() < 1e-3 * want,
        "lost {lost}, the round takes off {want}"
    );
}
