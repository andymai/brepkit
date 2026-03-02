//! Test helper functions for building common topological shapes.
//!
//! Gated behind the `test-utils` feature so production builds don't include them.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use brepkit_math::vec::{Point3, Vec3};

use crate::edge::{Edge, EdgeCurve};
use crate::face::{Face, FaceId, FaceSurface};
use crate::shell::Shell;
use crate::solid::{Solid, SolidId};
use crate::topology::Topology;
use crate::vertex::Vertex;
use crate::wire::{OrientedEdge, Wire};

/// Tolerance used by test helpers.
const TOL: f64 = 1e-7;

/// Creates a unit square face on the XY plane (z=0).
///
/// Vertices: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
/// Normal: +Z
///
/// # Panics
///
/// Panics if wire creation fails (should never happen with valid input).
#[must_use]
pub fn make_unit_square_face(topo: &mut Topology) -> FaceId {
    let v0 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v1 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v2 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
    let v3 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));

    let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )
    .expect("test wire creation should not fail");

    let wid = topo.wires.alloc(wire);

    topo.faces.alloc(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

/// Creates a unit triangle face on the XY plane (z=0).
///
/// Vertices: (0,0,0), (1,0,0), (0,1,0)
/// Normal: +Z
///
/// # Panics
///
/// Panics if wire creation fails.
#[must_use]
pub fn make_unit_triangle_face(topo: &mut Topology) -> FaceId {
    let v0 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v1 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v2 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));

    let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
        ],
        true,
    )
    .expect("test wire creation should not fail");

    let wid = topo.wires.alloc(wire);

    topo.faces.alloc(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

/// Helper: create a rectangular face from 4 vertex IDs on a given plane.
fn make_quad_face(
    topo: &mut Topology,
    v0: crate::vertex::VertexId,
    v1: crate::vertex::VertexId,
    v2: crate::vertex::VertexId,
    v3: crate::vertex::VertexId,
    normal: Vec3,
    d: f64,
) -> FaceId {
    let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )
    .expect("test wire creation should not fail");

    let wid = topo.wires.alloc(wire);
    topo.faces
        .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
}

/// Creates a unit cube solid with vertices at (0,0,0) to (1,1,1).
///
/// The cube has 8 vertices, 24 edges (4 per face, not shared — simplified
/// construction for testing), 6 faces, 1 shell, and 1 solid.
///
/// # Panics
///
/// Panics if any topology construction fails.
#[must_use]
pub fn make_unit_cube(topo: &mut Topology) -> SolidId {
    // 8 vertices of the unit cube.
    let v000 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v100 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v110 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
    let v010 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));
    let v001 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, 1.0), TOL));
    let v101 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 0.0, 1.0), TOL));
    let v111 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(1.0, 1.0, 1.0), TOL));
    let v011 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 1.0, 1.0), TOL));

    // 6 faces (each has its own edges — simplified for testing).
    // Bottom (z=0): v000, v100, v110, v010  normal -Z
    let bottom = make_quad_face(topo, v000, v010, v110, v100, Vec3::new(0.0, 0.0, -1.0), 0.0);
    // Top (z=1): v001, v101, v111, v011  normal +Z
    let top = make_quad_face(topo, v001, v101, v111, v011, Vec3::new(0.0, 0.0, 1.0), 1.0);
    // Front (y=0): v000, v100, v101, v001  normal -Y
    let front = make_quad_face(topo, v000, v100, v101, v001, Vec3::new(0.0, -1.0, 0.0), 0.0);
    // Back (y=1): v010, v110, v111, v011  normal +Y
    let back = make_quad_face(topo, v010, v110, v111, v011, Vec3::new(0.0, 1.0, 0.0), 1.0);
    // Left (x=0): v000, v010, v011, v001  normal -X
    let left = make_quad_face(topo, v000, v010, v011, v001, Vec3::new(-1.0, 0.0, 0.0), 0.0);
    // Right (x=1): v100, v110, v111, v101  normal +X
    let right = make_quad_face(topo, v100, v110, v111, v101, Vec3::new(1.0, 0.0, 0.0), 1.0);

    let shell = Shell::new(vec![bottom, top, front, back, left, right])
        .expect("test shell creation should not fail");
    let shell_id = topo.shells.alloc(shell);

    topo.solids.alloc(Solid::new(shell_id, vec![]))
}
