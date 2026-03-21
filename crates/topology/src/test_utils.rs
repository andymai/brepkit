//! Test helper functions for building common topological shapes.
//!
//! Gated behind the `test-utils` feature so production builds don't include them.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use brepkit_math::vec::{Point3, Vec3};

use crate::edge::{Edge, EdgeCurve, EdgeId};
use crate::face::{Face, FaceId, FaceSurface};
use crate::shell::Shell;
use crate::solid::{Solid, SolidId};
use crate::topology::Topology;
use crate::vertex::{Vertex, VertexId};
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
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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

    let wid = topo.add_wire(wire);

    topo.add_face(Face::new(
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
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
        ],
        true,
    )
    .expect("test wire creation should not fail");

    let wid = topo.add_wire(wire);

    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

/// Creates a CW-wound unit square face on the XY plane (z=0).
///
/// Same geometry as [`make_unit_square_face`] but with reversed winding:
/// vertices traverse (0,0)→(0,1)→(1,1)→(1,0) (clockwise from +Z).
/// The stored normal is -Z (consistent with the CW winding via Newell's method).
///
/// This simulates the common case where external callers (e.g. brepjs)
/// provide CW-wound profiles.
///
/// # Panics
///
/// Panics if wire creation fails.
#[must_use]
pub fn make_cw_unit_square_face(topo: &mut Topology) -> FaceId {
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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

    let wid = topo.add_wire(wire);

    // Newell normal of CW winding → -Z
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ))
}

/// Creates an offset unit cube solid with vertices at `(ox,oy,oz)` to
/// `(ox+1, oy+1, oz+1)`, using manifold shared edges.
///
/// Every edge is referenced by exactly two faces with opposing orientations,
/// making the shell a valid 2-manifold.
///
/// # Panics
///
/// Panics if any topology construction fails.
#[must_use]
#[allow(clippy::similar_names)]
pub fn make_unit_cube_manifold_at(topo: &mut Topology, ox: f64, oy: f64, oz: f64) -> SolidId {
    // 8 vertices
    let v: [VertexId; 8] = [
        topo.add_vertex(Vertex::new(Point3::new(ox, oy, oz), TOL)), // 0: 000
        topo.add_vertex(Vertex::new(Point3::new(ox + 1.0, oy, oz), TOL)), // 1: 100
        topo.add_vertex(Vertex::new(Point3::new(ox + 1.0, oy + 1.0, oz), TOL)), // 2: 110
        topo.add_vertex(Vertex::new(Point3::new(ox, oy + 1.0, oz), TOL)), // 3: 010
        topo.add_vertex(Vertex::new(Point3::new(ox, oy, oz + 1.0), TOL)), // 4: 001
        topo.add_vertex(Vertex::new(Point3::new(ox + 1.0, oy, oz + 1.0), TOL)), // 5: 101
        topo.add_vertex(Vertex::new(Point3::new(ox + 1.0, oy + 1.0, oz + 1.0), TOL)), // 6: 111
        topo.add_vertex(Vertex::new(Point3::new(ox, oy + 1.0, oz + 1.0), TOL)), // 7: 011
    ];

    // 12 edges of a cube (each shared by exactly 2 faces).
    // Bottom ring (z=oz)
    let e_b0 = topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Line)); // 0→1
    let e_b1 = topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Line)); // 1→2
    let e_b2 = topo.add_edge(Edge::new(v[2], v[3], EdgeCurve::Line)); // 2→3
    let e_b3 = topo.add_edge(Edge::new(v[3], v[0], EdgeCurve::Line)); // 3→0
    // Top ring (z=oz+1)
    let e_t0 = topo.add_edge(Edge::new(v[4], v[5], EdgeCurve::Line)); // 4→5
    let e_t1 = topo.add_edge(Edge::new(v[5], v[6], EdgeCurve::Line)); // 5→6
    let e_t2 = topo.add_edge(Edge::new(v[6], v[7], EdgeCurve::Line)); // 6→7
    let e_t3 = topo.add_edge(Edge::new(v[7], v[4], EdgeCurve::Line)); // 7→4
    // Verticals
    let e_v0 = topo.add_edge(Edge::new(v[0], v[4], EdgeCurve::Line)); // 0→4
    let e_v1 = topo.add_edge(Edge::new(v[1], v[5], EdgeCurve::Line)); // 1→5
    let e_v2 = topo.add_edge(Edge::new(v[2], v[6], EdgeCurve::Line)); // 2→6
    let e_v3 = topo.add_edge(Edge::new(v[3], v[7], EdgeCurve::Line)); // 3→7

    // Helper to build a face from 4 oriented edges.
    let mk = |topo: &mut Topology, edges: [(EdgeId, bool); 4], normal: Vec3, d: f64| -> FaceId {
        let wire = Wire::new(
            edges
                .iter()
                .map(|&(eid, fwd)| OrientedEdge::new(eid, fwd))
                .collect(),
            true,
        )
        .expect("test wire");
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
    };

    // Each face is CCW when viewed from outside (along its outward normal).
    // Each of the 12 edges must appear once forward and once reversed.
    //
    // Edge definitions (forward direction):
    //   b0: 0→1, b1: 1→2, b2: 2→3, b3: 3→0
    //   t0: 4→5, t1: 5→6, t2: 6→7, t3: 7→4
    //   v0: 0→4, v1: 1→5, v2: 2→6, v3: 3→7
    //
    // Edge usage per face (F=forward, R=reversed):
    //   Bottom (-Z): b0R b3R b2R b1R  → traversal: 1→0→3→2→1
    //   Top    (+Z): t0F t1F t2F t3F  → traversal: 4→5→6→7→4
    //   Front  (-Y): b0F v1F t0R v0R  → traversal: 0→1→5→4→0
    //   Back   (+Y): b2F v3F t2R v2R  → traversal: 2→3→7→6→2
    //   Left   (-X): b3F v0F t3R v3R  → traversal: 3→0→4→7→3
    //   Right  (+X): b1F v2F t1R v1R  → traversal: 1→2→6→5→1

    let bottom = mk(
        topo,
        [(e_b0, false), (e_b3, false), (e_b2, false), (e_b1, false)],
        Vec3::new(0.0, 0.0, -1.0),
        -oz,
    );

    let top = mk(
        topo,
        [(e_t0, true), (e_t1, true), (e_t2, true), (e_t3, true)],
        Vec3::new(0.0, 0.0, 1.0),
        oz + 1.0,
    );

    let front = mk(
        topo,
        [(e_b0, true), (e_v1, true), (e_t0, false), (e_v0, false)],
        Vec3::new(0.0, -1.0, 0.0),
        -oy,
    );

    let back = mk(
        topo,
        [(e_b2, true), (e_v3, true), (e_t2, false), (e_v2, false)],
        Vec3::new(0.0, 1.0, 0.0),
        oy + 1.0,
    );

    let left = mk(
        topo,
        [(e_b3, true), (e_v0, true), (e_t3, false), (e_v3, false)],
        Vec3::new(-1.0, 0.0, 0.0),
        -ox,
    );

    let right = mk(
        topo,
        [(e_b1, true), (e_v2, true), (e_t1, false), (e_v1, false)],
        Vec3::new(1.0, 0.0, 0.0),
        ox + 1.0,
    );

    let shell = Shell::new(vec![bottom, top, front, back, left, right]).expect("test shell");
    let shell_id = topo.add_shell(shell);
    topo.add_solid(Solid::new(shell_id, vec![]))
}

/// Creates a unit cube solid at the origin with manifold shared edges.
///
/// Equivalent to `make_unit_cube_manifold_at(topo, 0.0, 0.0, 0.0)`.
///
/// # Panics
///
/// Panics if any topology construction fails.
#[must_use]
pub fn make_unit_cube_manifold(topo: &mut Topology) -> SolidId {
    make_unit_cube_manifold_at(topo, 0.0, 0.0, 0.0)
}

/// Helper: create a rectangular face from 4 vertex IDs on a given plane.
fn make_quad_face(
    topo: &mut Topology,
    v0: VertexId,
    v1: VertexId,
    v2: VertexId,
    v3: VertexId,
    normal: Vec3,
    d: f64,
) -> FaceId {
    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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

    let wid = topo.add_wire(wire);
    topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
}

/// Creates a unit cube solid with vertices at (0,0,0) to (1,1,1).
///
/// **Non-manifold**: The cube has 8 vertices, 24 edges (4 per face, not shared
/// — simplified construction for testing), 6 faces, 1 shell, and 1 solid.
/// For a manifold cube with shared edges, use [`make_unit_cube_manifold`].
///
/// # Panics
///
/// Panics if any topology construction fails.
#[must_use]
pub fn make_unit_cube_non_manifold(topo: &mut Topology) -> SolidId {
    // 8 vertices of the unit cube.
    let v000 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
    let v100 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
    let v110 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
    let v010 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));
    let v001 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), TOL));
    let v101 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), TOL));
    let v111 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 1.0), TOL));
    let v011 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 1.0), TOL));

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
    let shell_id = topo.add_shell(shell);

    topo.add_solid(Solid::new(shell_id, vec![]))
}
