//! Tests for tessellation.

#![allow(clippy::unwrap_used, deprecated)]

use brepkit_math::det_hash::{DetHashMap, DetHashSet};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::nurbs::tessellate_nurbs;
use super::*;

#[test]
fn tessellate_square() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    let mesh = tessellate(&topo, face, 0.1).unwrap();

    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(mesh.normals.len(), 4);
    assert_eq!(mesh.indices.len(), 6);
}

#[test]
fn tessellate_triangle() {
    let mut topo = Topology::new();
    let face = make_unit_triangle_face(&mut topo);

    let mesh = tessellate(&topo, face, 0.1).unwrap();

    assert_eq!(mesh.positions.len(), 3);
    assert_eq!(mesh.normals.len(), 3);
    assert_eq!(mesh.indices.len(), 3);
}

/// Tessellate a simple bilinear NURBS surface (a flat quad as NURBS).
#[test]
fn tessellate_nurbs_surface() {
    let mut topo = Topology::new();

    let surface = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();

    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));

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
    .unwrap();
    let wid = topo.add_wire(wire);

    let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

    let mesh = tessellate(&topo, face, 0.25).unwrap();

    assert_eq!(mesh.positions.len(), 25);
    assert_eq!(mesh.normals.len(), 25);
    assert_eq!(mesh.indices.len(), 96);

    for pos in &mesh.positions {
        assert!(pos.x() >= -1e-10 && pos.x() <= 1.0 + 1e-10);
        assert!(pos.y() >= -1e-10 && pos.y() <= 1.0 + 1e-10);
        assert!((pos.z()).abs() < 1e-10);
    }
}

#[test]
fn tessellate_l_shape_nonconvex() {
    let mut topo = Topology::new();

    let points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(2.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(1.0, 2.0, 0.0),
        Point3::new(0.0, 2.0, 0.0),
    ];

    let verts: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();

    let n = verts.len();
    let edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.add_edge(Edge::new(verts[i], verts[next], EdgeCurve::Line))
        })
        .collect();

    let wire = Wire::new(
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);

    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let mesh = tessellate(&topo, face, 0.1).unwrap();

    assert_eq!(mesh.positions.len(), 6, "should have 6 vertices");
    assert_eq!(
        mesh.indices.len(),
        12,
        "L-shape should have 4 triangles (12 indices)"
    );

    let mut total_area = 0.0;
    for t in 0..mesh.indices.len() / 3 {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let a = mesh.positions[i1] - mesh.positions[i0];
        let b = mesh.positions[i2] - mesh.positions[i0];
        total_area += 0.5 * a.cross(b).length();
    }
    assert!(
        (total_area - 3.0).abs() < 0.01,
        "L-shape area should be ~3.0, got {total_area}"
    );
}

#[test]
fn tessellate_flat_surface_few_triangles() {
    let surface = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();

    let mesh = tessellate_nurbs(&surface, 0.1, 0.0).mesh;

    assert_eq!(
        mesh.indices.len() / 3,
        32,
        "flat surface should have exactly 32 triangles, got {}",
        mesh.indices.len() / 3
    );
}

#[test]
fn tessellate_curved_surface_more_at_curves() {
    let mut cps = Vec::new();
    let mut ws = Vec::new();
    for i in 0..4 {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        for j in 0..4 {
            #[allow(clippy::cast_precision_loss)]
            let z = ((i + j) as f64 * 0.8).sin() * 2.0;
            #[allow(clippy::cast_precision_loss)]
            row.push(Point3::new(j as f64, i as f64, z));
            wrow.push(1.0);
        }
        cps.push(row);
        ws.push(wrow);
    }
    let curved = NurbsSurface::new(
        3,
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        cps,
        ws,
    )
    .unwrap();

    let flat = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();

    let deflection = 0.05;
    let flat_mesh = tessellate_nurbs(&flat, deflection, 0.0).mesh;
    let curved_mesh = tessellate_nurbs(&curved, deflection, 0.0).mesh;

    let flat_tris = flat_mesh.indices.len() / 3;
    let curved_tris = curved_mesh.indices.len() / 3;

    assert!(
        curved_tris > flat_tris,
        "curved surface should have more triangles ({curved_tris}) than flat ({flat_tris})"
    );
}

// -- Watertight tessellation tests --

#[test]
fn tessellate_solid_box_watertight() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let tri_count = mesh.indices.len() / 3;
    assert_eq!(
        tri_count, 12,
        "box should have 12 triangles, got {tri_count}"
    );

    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "box mesh should be watertight (0 boundary edges), got {boundary}"
    );
    assert!(is_watertight(&mesh), "box mesh should be watertight");
}

#[test]
fn tessellate_plain_cylinder_watertight() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();
    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "plain cylinder should be watertight (0 boundary edges), got {boundary}"
    );
}

/// Regression for issue #696: dovetail-style fuse where a small tongue protrudes
/// into two adjacent slabs. The downstream consumer (gridfinity-layout-tool)
/// adds a TONGUE_PROTRUSION specifically to avoid coplanar fuse residue, but
/// brepkit's pipeline produced non-manifold tessellation output. This
/// minimal case exercises the same topological pattern.
#[test]
fn tessellate_dovetail_fuse_manifold_issue_696() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let slab_a = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    let slab_b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, slab_b, &Mat4::translation(10.0, 0.0, 0.0))
        .unwrap();
    // Tongue from x=8 to x=12 — 2mm protrusion into each slab. Centered on
    // the y axis at y=4..6, full slab height z=0..1.
    let tongue = crate::primitives::make_box(&mut topo, 4.0, 2.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, tongue, &Mat4::translation(8.0, 4.0, 0.0))
        .unwrap();

    let ab = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, slab_a, slab_b)
        .unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, ab, tongue).unwrap();

    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let nm = non_manifold_edge_count(&mesh);
    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        nm, 0,
        "dovetail fuse should produce a 2-manifold mesh (0 non-manifold edges), got {nm}"
    );
    assert_eq!(
        boundary, 0,
        "dovetail fuse should produce a watertight mesh (0 boundary edges), got {boundary}"
    );
}

/// Extension of #696 repro: multi-tile chain (3 slabs, 2 tongues) plus a hollow
/// cut. Approximates the lightweight-floor + multi-join-edge pattern from the
/// failing 4x4 / 5x4 dovetail baseplates.
#[test]
fn tessellate_dovetail_multi_tile_hollow_issue_696() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let slab_a = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    let slab_b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, slab_b, &Mat4::translation(10.0, 0.0, 0.0))
        .unwrap();
    let slab_c = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, slab_c, &Mat4::translation(20.0, 0.0, 0.0))
        .unwrap();

    let tongue_ab = crate::primitives::make_box(&mut topo, 4.0, 2.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, tongue_ab, &Mat4::translation(8.0, 4.0, 0.0))
        .unwrap();
    let tongue_bc = crate::primitives::make_box(&mut topo, 4.0, 2.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, tongue_bc, &Mat4::translation(18.0, 4.0, 0.0))
        .unwrap();

    let ab = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, slab_a, slab_b)
        .unwrap();
    let ab2 =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, ab, tongue_ab).unwrap();
    let abc =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, ab2, slab_c).unwrap();
    let fused = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, abc, tongue_bc)
        .unwrap();

    // Hollow out the floor: cut a thin interior pocket.
    let pocket = crate::primitives::make_box(&mut topo, 28.0, 8.0, 0.6).unwrap();
    crate::transform::transform_solid(&mut topo, pocket, &Mat4::translation(1.0, 1.0, 0.2))
        .unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, fused, pocket).unwrap();

    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let nm = non_manifold_edge_count(&mesh);
    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        nm, 0,
        "multi-tile dovetail+hollow should be 2-manifold (0 non-manifold edges), got {nm}"
    );
    assert_eq!(
        boundary, 0,
        "multi-tile dovetail+hollow should be watertight (0 boundary edges), got {boundary}"
    );
}

/// Trapezoidal tongue (real dovetail profile — narrow at tip, wider at base)
/// joining two slabs. The trapezoid creates 45-degree edges where the tongue
/// meets the slabs, which is where coplanar fuse residue tends to appear.
#[test]
fn tessellate_dovetail_trapezoidal_tongue_issue_696() {
    use brepkit_math::mat::Mat4;
    use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};

    let mut topo = Topology::new();
    let slab_a = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    let slab_b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, slab_b, &Mat4::translation(10.0, 0.0, 0.0))
        .unwrap();

    // Trapezoidal tongue extruded in +Z. Wide bases at x=8 and x=12 (each
    // 2mm inside its slab); narrow waist at x=9.8 / x=10.2. CCW order so
    // the face normal points up.
    let pts = vec![
        Point3::new(8.0, 4.0, 0.0),
        Point3::new(9.8, 4.8, 0.0),
        Point3::new(10.2, 4.8, 0.0),
        Point3::new(12.0, 4.0, 0.0),
        Point3::new(12.0, 6.0, 0.0),
        Point3::new(10.2, 5.2, 0.0),
        Point3::new(9.8, 5.2, 0.0),
        Point3::new(8.0, 6.0, 0.0),
    ];
    let wire_id = make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
    let face_id = make_face_from_wire(&mut topo, wire_id).unwrap();
    let tongue =
        crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let ab = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, slab_a, slab_b)
        .unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, ab, tongue).unwrap();

    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let nm = non_manifold_edge_count(&mesh);
    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        nm, 0,
        "trapezoidal-tongue fuse should produce a 2-manifold mesh, got {nm} non-manifold edges"
    );
    assert_eq!(
        boundary, 0,
        "trapezoidal-tongue fuse should produce a watertight mesh, got {boundary} boundary edges"
    );
}

/// Direct unit tests for `dedupe_coincident_triangles` — the synthetic
/// dovetail tests above don't reproduce the upstream symptom and so leave the
/// new Phase-7 pass untested by itself.
#[test]
fn dedupe_cancels_opposing_winding_pair() {
    let mut mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
        indices: vec![0, 1, 2, 0, 2, 1],
    };
    let mut tri_faces = vec![0_u32; mesh.indices.len() / 3];
    super::mesh_ops::dedupe_coincident_triangles(&mut mesh, Some(&mut tri_faces));
    assert_eq!(
        mesh.indices.len(),
        0,
        "opposing-winding triangle pair should cancel"
    );
    assert_eq!(
        mesh.positions.len(),
        0,
        "unreferenced positions should be dropped after cancel"
    );
}

#[test]
fn dedupe_collapses_same_winding_duplicate() {
    let mut mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
        indices: vec![0, 1, 2, 0, 1, 2],
    };
    let mut tri_faces = vec![0_u32; mesh.indices.len() / 3];
    super::mesh_ops::dedupe_coincident_triangles(&mut mesh, Some(&mut tri_faces));
    assert_eq!(
        mesh.indices.len(),
        3,
        "same-winding duplicate should collapse to one triangle"
    );
    assert_eq!(mesh.positions.len(), 3, "all 3 vertices still referenced");
}

#[test]
fn dedupe_matches_position_coincidence_not_index() {
    // Two triangles at the same positions but with distinct vertex IDs —
    // the case where boundary-vertex welding didn't catch them. Same
    // winding, so dedup keeps one.
    let p = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let mut mesh = TriangleMesh {
        positions: vec![p[0], p[1], p[2], p[0], p[1], p[2]],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
        indices: vec![0, 1, 2, 3, 4, 5],
    };
    let mut tri_faces = vec![0_u32; mesh.indices.len() / 3];
    super::mesh_ops::dedupe_coincident_triangles(&mut mesh, Some(&mut tri_faces));
    assert_eq!(
        mesh.indices.len(),
        3,
        "position-coincident triangle should collapse even with distinct IDs"
    );
    assert_eq!(
        mesh.positions.len(),
        3,
        "duplicate positions should be compacted"
    );
}

#[test]
fn dedupe_preserves_thin_plate_geometry() {
    // 1e-4mm-thick plate: front face (z=0) and back face (z=1e-4) tessellate
    // to disjoint triangle pairs that share x/y. The quantization grid must
    // be tight enough to keep them distinct.
    let mut mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1e-4),
            Point3::new(1.0, 0.0, 1e-4),
            Point3::new(0.0, 1.0, 1e-4),
        ],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
        indices: vec![0, 1, 2, 3, 5, 4],
    };
    let mut tri_faces = vec![0_u32; mesh.indices.len() / 3];
    super::mesh_ops::dedupe_coincident_triangles(&mut mesh, Some(&mut tri_faces));
    assert_eq!(
        mesh.indices.len(),
        6,
        "1e-4mm-apart triangles should NOT collapse"
    );
}

#[test]
fn tessellate_boolean_cut_cylinder_watertight() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    let box_s = crate::primitives::make_box(&mut topo, 3.0, 3.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, box_s, &Mat4::translation(-1.5, -1.5, 1.5))
        .unwrap();

    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, cyl, box_s).unwrap();

    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let boundary = boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "boolean cut cylinder should be watertight (0 boundary edges), got {boundary}"
    );
}

/// Issue #696: a cylindrical hole drilled through a box must tessellate
/// watertight across radii and deflections.
///
/// Previously a drilled-hole cylinder lateral face took the snap path, which
/// tessellated the cylinder independently and reconciled its rim vertices to
/// the shared edge pool by 1e-6 proximity. At radius/deflection combinations
/// where the independent rim sampling and the shared-edge sampling diverged by
/// one segment (e.g. r=3.25, deflection=0.05), the rim vertices landed at
/// different angles, failed the snap, and became near-coincident duplicates —
/// cracking the mesh (up to 252 boundary edges). The fix tessellates such bands
/// directly from the shared rim vertices (`tessellate_revolution_band_shared`),
/// making them watertight by construction.
#[test]
fn tessellate_drilled_hole_watertight_across_radii() {
    use brepkit_math::mat::Mat4;

    for &r in &[2.5_f64, 3.0, 3.25, 3.5, 4.0, 5.0] {
        for &defl in &[0.05_f64, 0.1] {
            let mut topo = Topology::new();
            let box_s = crate::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
            let cyl = crate::primitives::make_cylinder(&mut topo, r, 20.0).unwrap();
            crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(10.0, 10.0, -5.0))
                .unwrap();
            let result =
                crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, box_s, cyl)
                    .unwrap();
            let mesh = tessellate_solid(&topo, result, defl).unwrap();
            let boundary = boundary_edge_count(&mesh);
            let nm = non_manifold_edge_count(&mesh);
            assert_eq!(
                (boundary, nm),
                (0, 0),
                "drilled hole r={r} defl={defl} must be watertight, got bd={boundary} nm={nm}"
            );
        }
    }
}

/// Issue #696 end-to-end: a gridfinity-style tile (pocketed slab + four magnet
/// holes drilled through the floor into the pocket cavity) must tessellate
/// watertight. This is the multi-feature scenario the consumer hit; the magnet
/// cylinders are drilled holes that exercise the shared-rim band path.
#[test]
fn tessellate_gridfinity_magnet_tile_watertight() {
    use crate::boolean::{BooleanOp, boolean};
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let slab = crate::primitives::make_box(&mut topo, 42.0, 42.0, 8.0).unwrap();
    crate::transform::transform_solid(&mut topo, slab, &Mat4::translation(0.0, 0.0, -8.0)).unwrap();
    let pocket = crate::primitives::make_box(&mut topo, 35.0, 35.0, 6.5).unwrap();
    crate::transform::transform_solid(&mut topo, pocket, &Mat4::translation(3.5, 3.5, -6.0))
        .unwrap();
    let mut tile = boolean(&mut topo, BooleanOp::Cut, slab, pocket).unwrap();
    for (cx, cy) in [(7.0, 7.0), (35.0, 7.0), (7.0, 35.0), (35.0, 35.0)] {
        let cyl = crate::primitives::make_cylinder(&mut topo, 3.25, 4.0).unwrap();
        crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(cx, cy, -8.5))
            .unwrap();
        tile = boolean(&mut topo, BooleanOp::Cut, tile, cyl).unwrap();
    }
    for &defl in &[0.05_f64, 0.1] {
        let mesh = tessellate_solid(&topo, tile, defl).unwrap();
        assert_eq!(
            (boundary_edge_count(&mesh), non_manifold_edge_count(&mesh)),
            (0, 0),
            "magnet tile must tessellate watertight at deflection {defl}"
        );
    }
}

#[test]
fn tessellate_boolean_cut_cone_watertight() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let cone = crate::primitives::make_cone(&mut topo, 1.5, 0.5, 4.0).unwrap();
    let box_s = crate::primitives::make_box(&mut topo, 4.0, 4.0, 1.0).unwrap();
    crate::transform::transform_solid(&mut topo, box_s, &Mat4::translation(-2.0, -2.0, 1.5))
        .unwrap();

    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, cone, box_s).unwrap();

    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let boundary = position_based_boundary_count(&mesh);
    assert_eq!(
        boundary, 0,
        "boolean cut cone should be watertight (0 position-based boundary edges), got {boundary}"
    );
}

/// A cylinder bored through a sphere yields two spherical latitude bands (each a
/// `Sphere` face whose only inner wire is the tunnel rim, bounded by the equator)
/// plus an inner cylinder wall. Those bands degenerate in UV (each constant-v
/// latitude projects to a zero-area segment), so the CDT path used to fill the
/// removed polar cap — skinning over the tunnel mouth and inflating the mesh area
/// to ~648 vs the analytic band area ~587.67. The structured latitude-band path
/// must instead leave the mouth open: area must approach the analytic value from
/// below (inscribed), the mesh must be watertight, and no vertex may reach the
/// sphere pole (|z| stays at the rim's z, ~5.196, well below the radius 6).
#[test]
fn bored_sphere_band_area_and_watertight() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let sphere = crate::primitives::make_sphere(&mut topo, 6.0, 24).unwrap();
    let cyl = crate::primitives::make_cylinder(&mut topo, 3.0, 30.0).unwrap();
    crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(0.0, 0.0, -15.0)).unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, sphere, cyl).unwrap();

    // Analytic surface area: two spherical zones from the equator (z=0) to the
    // rim (z=±sqrt(R²−r²)) plus the inner cylinder wall.
    let r_sphere = 6.0_f64;
    let z_rim = (r_sphere * r_sphere - 9.0).sqrt();
    let zone = 2.0 * std::f64::consts::PI * r_sphere * z_rim; // one band
    let cyl_wall = 2.0 * std::f64::consts::PI * 3.0 * (2.0 * z_rim);
    let analytic = 2.0 * zone + cyl_wall;

    // Area converges to the analytic value from below as deflection tightens.
    let mut prev_area = 0.0_f64;
    for &defl in &[0.1_f64, 0.02, 0.01] {
        let mesh = tessellate_solid(&topo, result, defl).unwrap();

        let mut area = 0.0;
        for t in 0..mesh.indices.len() / 3 {
            let a = mesh.positions[mesh.indices[t * 3 + 1] as usize]
                - mesh.positions[mesh.indices[t * 3] as usize];
            let b = mesh.positions[mesh.indices[t * 3 + 2] as usize]
                - mesh.positions[mesh.indices[t * 3] as usize];
            area += 0.5 * a.cross(b).length();
        }

        assert!(
            is_watertight(&mesh),
            "bored sphere must tessellate watertight at deflection {defl}: bd={} nm={}",
            boundary_edge_count(&mesh),
            non_manifold_edge_count(&mesh)
        );

        // The polar cap must be open: no vertex may approach the sphere pole.
        let max_abs_z = mesh
            .positions
            .iter()
            .map(|p| p.z().abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_z < z_rim + 1e-6,
            "tunnel mouth is filled: a vertex reached |z|={max_abs_z} (rim is {z_rim}, pole would be {r_sphere})"
        );

        // Inscribed band: area is below analytic and within deflection-scaled
        // tolerance, and never below the previous (coarser) deflection.
        assert!(
            area <= analytic + 1.0,
            "mesh area {area} exceeds analytic band area {analytic} (cap fill?) at deflection {defl}"
        );
        assert!(
            area >= prev_area,
            "area should not regress as deflection tightens: {area} < {prev_area}"
        );
        prev_area = area;
    }

    // At the default display deflection the area is close to analytic (and far
    // from the ~648 cap-filled value the CDT path produced).
    let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
    let mut area = 0.0;
    for t in 0..mesh.indices.len() / 3 {
        let a = mesh.positions[mesh.indices[t * 3 + 1] as usize]
            - mesh.positions[mesh.indices[t * 3] as usize];
        let b = mesh.positions[mesh.indices[t * 3 + 2] as usize]
            - mesh.positions[mesh.indices[t * 3] as usize];
        area += 0.5 * a.cross(b).length();
    }
    assert!(
        (area - analytic).abs() < 5.0,
        "default-deflection bored-sphere area {area} should be ~{analytic} (band), not ~648 (cap-filled)"
    );

    // The per-face mesher behind the exports must keep every band, with its
    // tunnel mouth open, too.
    for fid in brepkit_topology::explorer::solid_faces(&topo, result).unwrap() {
        let face_mesh = tessellate(&topo, fid, 0.1).unwrap();
        assert!(
            !face_mesh.indices.is_empty(),
            "face {fid:?} meshes to nothing"
        );
        let reach = face_mesh
            .positions
            .iter()
            .map(|p| p.z().abs())
            .fold(0.0_f64, f64::max);
        assert!(
            reach < z_rim + 1e-6,
            "face {fid:?} fills its tunnel mouth: |z| reaches {reach}"
        );
    }
}

/// A box ∩ centered-sphere produces two annular sphere "collar" patches whose
/// outer wire varies in v (a scalloped great-circle/equator floor) plus a
/// latitude-cap hole. This is the varying-v generalization of the bored-sphere
/// band: the collar must tessellate watertight (the CDT path leaves 98+ free
/// edges) and its area must match the analytic box∩sphere boundary.
#[test]
fn box_centered_sphere_collar_tessellates_watertight() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sp = crate::primitives::make_sphere(&mut topo, 6.0, 24).unwrap();
    crate::transform::transform_solid(&mut topo, sp, &Mat4::translation(5.0, 5.0, 5.0)).unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Intersect, bx, sp).unwrap();

    // Analytic boundary area: 6 plane discs (radius² = R²−5² = 11) + the sphere
    // minus 6 spherical caps (each cap zone area = 2πRh, h=1).
    let r: f64 = 6.0;
    let disc_area = 6.0 * std::f64::consts::PI * 11.0;
    let cap_zone = 2.0 * std::f64::consts::PI * r * 1.0;
    let sphere_patch = 4.0 * std::f64::consts::PI * r * r - 6.0 * cap_zone;
    let analytic_area = disc_area + sphere_patch;

    for &defl in &[0.05_f64, 0.005] {
        let mesh = tessellate_solid(&topo, result, defl).unwrap();
        assert!(
            is_watertight(&mesh),
            "box∩sphere collar must tessellate watertight at deflection {defl}: bd={} nm={}",
            boundary_edge_count(&mesh),
            non_manifold_edge_count(&mesh)
        );
        let mut area = 0.0;
        for t in 0..mesh.indices.len() / 3 {
            let a = mesh.positions[mesh.indices[t * 3 + 1] as usize]
                - mesh.positions[mesh.indices[t * 3] as usize];
            let b = mesh.positions[mesh.indices[t * 3 + 2] as usize]
                - mesh.positions[mesh.indices[t * 3] as usize];
            area += 0.5 * a.cross(b).length();
        }
        // Inscribed mesh area is below analytic; within ~3% at these deflections.
        assert!(
            area <= analytic_area + 1.0 && area > analytic_area * 0.97,
            "collar mesh area {area} should be ~{analytic_area} (no cap-fill) at deflection {defl}"
        );
    }
}

#[test]
fn torus_box_notch_band_tessellates_watertight() {
    use brepkit_math::mat::Mat4;

    let mut topo = Topology::new();
    let tor = crate::primitives::make_torus(&mut topo, 10.0, 3.0, 32).unwrap();
    let bx = crate::primitives::make_box(&mut topo, 8.0, 8.0, 8.0).unwrap();
    crate::transform::transform_solid(&mut topo, bx, &Mat4::translation(6.0, -4.0, -4.0)).unwrap();
    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, tor, bx).unwrap();

    // The kept toroidal notch band must tessellate watertight (shared seam
    // vertices with the notch walls — no cracks), STABLE across deflections.
    for &defl in &[0.1_f64, 0.05, 0.02] {
        let mesh = tessellate_solid(&topo, result, defl).unwrap();
        assert!(
            is_watertight(&mesh),
            "torus−box notch band must tessellate watertight at deflection {defl}: bd={} nm={}",
            boundary_edge_count(&mesh),
            non_manifold_edge_count(&mesh)
        );
    }
}

#[test]
fn tessellate_solid_box_correct_area() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let mut total_area = 0.0;
    for t in 0..mesh.indices.len() / 3 {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let a = mesh.positions[i1] - mesh.positions[i0];
        let b = mesh.positions[i2] - mesh.positions[i0];
        total_area += 0.5 * a.cross(b).length();
    }
    assert!(
        (total_area - 52.0).abs() < 0.1,
        "box surface area should be ~52.0, got {total_area}"
    );
}

#[test]
fn tessellate_solid_box_shared_vertices() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    assert_eq!(
        mesh.positions.len(),
        8,
        "unit box should have exactly 8 shared vertices, got {}",
        mesh.positions.len()
    );
}

#[test]
fn tessellate_solid_cylinder_shared_topology() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let edge_map = brepkit_topology::explorer::edge_to_face_map(&topo, solid).unwrap();
    let shared_count = edge_map.values().filter(|faces| faces.len() >= 2).count();
    assert!(
        shared_count >= 2,
        "cylinder should have at least 2 shared edges, got {shared_count}"
    );

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();
    assert!(mesh.indices.len() >= 3, "cylinder should have triangles");
    assert!(!mesh.positions.is_empty(), "cylinder should have vertices");
}

#[test]
fn tessellate_solid_sphere_produces_mesh() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    assert!(mesh.indices.len() >= 3, "sphere should have triangles");
    assert!(!mesh.positions.is_empty(), "sphere should have vertices");
}

#[test]
fn is_watertight_basic() {
    let mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 4],
        indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
    };
    assert!(is_watertight(&mesh));
    assert_eq!(boundary_edge_count(&mesh), 0);
}

#[test]
fn is_watertight_open_mesh() {
    let mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ],
        normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
        indices: vec![0, 1, 2],
    };
    assert!(!is_watertight(&mesh));
    assert_eq!(boundary_edge_count(&mesh), 3);
}

#[test]
fn tessellate_solid_normals_unit_length() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    for (i, n) in mesh.normals.iter().enumerate() {
        let len = n.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "normal {i} should be unit length, got {len}"
        );
    }
}

// -- Curvature-adaptive tessellation tests --

#[test]
fn curvature_adaptive_refines_high_curvature() {
    let mut cps = Vec::new();
    let mut ws = Vec::new();
    for i in 0..4 {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        for j in 0..4 {
            #[allow(clippy::cast_precision_loss)]
            let x = (j as f64) / 3.0;
            #[allow(clippy::cast_precision_loss)]
            let y = (i as f64) / 3.0;
            let z = 2.0 * (1.0 - (x - 0.5).powi(2) - (y - 0.5).powi(2));
            #[allow(clippy::cast_precision_loss)]
            row.push(Point3::new(j as f64, i as f64, z));
            wrow.push(1.0);
        }
        cps.push(row);
        ws.push(wrow);
    }
    let dome = NurbsSurface::new(
        3,
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        cps,
        ws,
    )
    .unwrap();

    let fine_mesh = tessellate_nurbs(&dome, 0.01, 0.0).mesh;
    let coarse_mesh = tessellate_nurbs(&dome, 0.5, 0.0).mesh;

    assert!(
        fine_mesh.indices.len() / 3 > coarse_mesh.indices.len() / 3,
        "finer deflection should produce more triangles: fine={}, coarse={}",
        fine_mesh.indices.len() / 3,
        coarse_mesh.indices.len() / 3
    );
}

#[test]
fn curvature_adaptive_midpoint_sag_check() {
    let mut cps = Vec::new();
    let mut ws = Vec::new();
    for i in 0..4 {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        for j in 0..4 {
            #[allow(clippy::cast_precision_loss)]
            let z = ((i + j) as f64 * 0.5).sin() * 1.5;
            #[allow(clippy::cast_precision_loss)]
            row.push(Point3::new(j as f64, i as f64, z));
            wrow.push(1.0);
        }
        cps.push(row);
        ws.push(wrow);
    }
    let surface = NurbsSurface::new(
        3,
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        cps,
        ws,
    )
    .unwrap();

    let deflection = 0.05;
    let mesh = tessellate_nurbs(&surface, deflection, 0.0).mesh;

    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count > 32,
        "curved surface should have more than base 32 triangles, got {tri_count}"
    );

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let a = mesh.positions[i1] - mesh.positions[i0];
        let b = mesh.positions[i2] - mesh.positions[i0];
        let area = 0.5 * a.cross(b).length();
        assert!(area > 0.0, "triangle {t} has zero area");
    }
}

#[test]
fn sample_solid_edges_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();

    let edge_lines = sample_solid_edges(&topo, solid, 0.1).unwrap();

    assert_eq!(edge_lines.offsets.len(), 12, "box should have 12 edges");
    assert_eq!(
        edge_lines.positions.len(),
        24,
        "12 line edges x 2 points = 24 points"
    );
}

#[test]
fn sample_solid_edges_cylinder() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 3.0).unwrap();

    let edge_lines = sample_solid_edges(&topo, solid, 0.1).unwrap();
    assert_eq!(
        edge_lines.offsets.len(),
        2,
        "filtered cylinder should have 2 circle edges, got {}",
        edge_lines.offsets.len()
    );
    assert!(
        edge_lines.positions.len() > 10,
        "cylinder edges should have many sample points, got {}",
        edge_lines.positions.len()
    );

    let all_edges = sample_solid_edges_filtered(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        false,
    )
    .unwrap();
    assert!(
        all_edges.offsets.len() >= 3,
        "unfiltered cylinder should have at least 3 edges, got {}",
        all_edges.offsets.len()
    );
}

#[test]
fn sample_solid_edges_angular_tolerance_densifies_curves() {
    // A tighter angular tolerance must refine curved edges (the cylinder's two
    // circle rims) even with the linear deflection held fixed. Regression guard:
    // meshEdges() previously hardcoded DEFAULT_ANGULAR_TOL, so the caller's
    // angular tolerance had no effect (brepkit#952).
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 3.0).unwrap();

    // Loose linear deflection so the angular criterion governs sample density.
    let deflection = 1.0;
    let coarse = sample_solid_edges_filtered(&topo, solid, deflection, 0.5, false).unwrap();
    let fine = sample_solid_edges_filtered(&topo, solid, deflection, 0.05, false).unwrap();

    assert!(
        fine.positions.len() > coarse.positions.len(),
        "finer angular tolerance must add edge samples: coarse={}, fine={}",
        coarse.positions.len(),
        fine.positions.len()
    );
}

#[test]
fn sample_solid_edges_boolean_filters_coplanar() {
    // Fuse two boxes flush along x=10 with the second narrower in y (0..6 vs 0..10).
    // The shared x=10 strip (y 0..6) becomes internal, but the top (z=10), bottom
    // (z=0), and front (y=0) faces of the two boxes stay as coplanar adjacent
    // fragments when unify_faces is off (make_box puts y=0 at the front). The seams
    // between those three same-plane fragment pairs are exactly the smooth edges
    // sample_solid_edges should drop.
    use brepkit_math::mat::Mat4;
    let mut topo = Topology::new();
    let a = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = crate::primitives::make_box(&mut topo, 10.0, 6.0, 10.0).unwrap();
    crate::transform::transform_solid(&mut topo, b, &Mat4::translation(10.0, 0.0, 0.0)).unwrap();
    let opts = crate::boolean::BooleanOptions {
        unify_faces: false,
        ..Default::default()
    };
    let fused = crate::boolean::boolean_with_options(
        &mut topo,
        crate::boolean::BooleanOp::Fuse,
        a,
        b,
        opts,
    )
    .unwrap();

    let filtered = sample_solid_edges(&topo, fused, 0.1).unwrap();
    let all = sample_solid_edges_filtered(
        &topo,
        fused,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        false,
    )
    .unwrap();

    // Exactly the three coplanar seams (top, bottom, front) must be dropped — a bare
    // `filtered < unfiltered` would still pass if the boolean output drifted to a
    // single removed seam, defeating the point of the test.
    assert_eq!(
        filtered.offsets.len() + 3,
        all.offsets.len(),
        "exactly 3 coplanar seams should be filtered: filtered={}, unfiltered={}",
        filtered.offsets.len(),
        all.offsets.len()
    );
}

#[test]
fn tessellate_solid_filleted_box_nurbs_boundary() {
    let mut topo = Topology::new();
    let bx = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let edges = {
        let s = topo.solid(bx).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let face_id = sh.faces()[0];
        let face = topo.face(face_id).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        vec![wire.edges()[0].edge()]
    };
    let filleted = crate::fillet::fillet_rolling_ball(&mut topo, bx, &edges, 0.5).unwrap();
    let mesh = tessellate_solid(&topo, filleted, 0.1).unwrap();

    assert!(
        mesh.indices.len() >= 3,
        "filleted box should have triangles"
    );
    assert!(
        !mesh.positions.is_empty(),
        "filleted box should have vertices"
    );

    let boundary = boundary_edge_count(&mesh);
    assert!(
        boundary < mesh.indices.len() / 3,
        "filleted box should have few boundary edges, got {boundary}"
    );
}

// -- P3: Tessellation Quality tests --

#[test]
fn test_no_degenerate_triangles() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let tri_count = mesh.indices.len() / 3;
    assert!(tri_count > 0, "sphere should produce triangles");

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let a = mesh.positions[i1] - mesh.positions[i0];
        let b = mesh.positions[i2] - mesh.positions[i0];
        let area = 0.5 * a.cross(b).length();
        assert!(area > 0.0, "triangle {t} is degenerate (area = {area})");
    }
}

#[test]
fn test_min_angle_above_threshold() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let tri_count = mesh.indices.len() / 3;
    assert!(tri_count > 0, "cylinder should produce triangles");

    let min_angle_threshold = 0.0175;

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let edges_arr = [(p1 - p0, p2 - p0), (p0 - p1, p2 - p1), (p0 - p2, p1 - p2)];

        for (j, (ea, eb)) in edges_arr.iter().enumerate() {
            let len_a = ea.length();
            let len_b = eb.length();
            if len_a < 1e-15 || len_b < 1e-15 {
                continue;
            }
            let cos_angle = ea.dot(*eb) / (len_a * len_b);
            let angle = cos_angle.clamp(-1.0, 1.0).acos();
            assert!(
                angle > min_angle_threshold,
                "triangle {t} vertex {j} has angle {:.4} rad ({:.2} deg), below threshold",
                angle,
                angle.to_degrees()
            );
        }
    }
}

#[test]
fn test_max_sag_within_deflection() {
    let radius = 1.0;
    let deflection = 0.05;
    let mut topo = Topology::new();
    let solid = crate::primitives::make_sphere(&mut topo, radius, 16).unwrap();
    let mesh = tessellate_solid(&topo, solid, deflection).unwrap();

    let tri_count = mesh.indices.len() / 3;
    assert!(tri_count > 0);

    let mut max_sag = 0.0_f64;
    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let centroid = Point3::new(
            (mesh.positions[i0].x() + mesh.positions[i1].x() + mesh.positions[i2].x()) / 3.0,
            (mesh.positions[i0].y() + mesh.positions[i1].y() + mesh.positions[i2].y()) / 3.0,
            (mesh.positions[i0].z() + mesh.positions[i1].z() + mesh.positions[i2].z()) / 3.0,
        );
        let dist_from_origin =
            (centroid.x().powi(2) + centroid.y().powi(2) + centroid.z().powi(2)).sqrt();
        let sag = (dist_from_origin - radius).abs();
        max_sag = max_sag.max(sag);
    }

    assert!(
        max_sag < 2.0 * deflection,
        "max sag {max_sag} exceeds 2*deflection ({})",
        2.0 * deflection
    );
}

#[test]
fn test_watertight_solid_mesh() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
    let snap_pt = |p: Point3| -> (i64, i64, i64) { (snap(p.x()), snap(p.y()), snap(p.z())) };

    let mut pos_map: DetHashMap<(i64, i64, i64), usize> = DetHashMap::default();
    let mut next_id = 0_usize;
    let canonical: Vec<usize> = mesh
        .positions
        .iter()
        .map(|&p| {
            let key = snap_pt(p);
            *pos_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    let tri_count = mesh.indices.len() / 3;
    let mut half_edges: DetHashSet<(usize, usize)> = DetHashSet::default();
    for t in 0..tri_count {
        let a = canonical[mesh.indices[t * 3] as usize];
        let b = canonical[mesh.indices[t * 3 + 1] as usize];
        let c = canonical[mesh.indices[t * 3 + 2] as usize];
        half_edges.insert((a, b));
        half_edges.insert((b, c));
        half_edges.insert((c, a));
    }

    let boundary_count = half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count();
    assert_eq!(
        boundary_count, 0,
        "box mesh should be watertight (0 boundary edges), got {boundary_count}"
    );
}

#[test]
fn test_consistent_winding() {
    let dx = 2.0;
    let dy = 3.0;
    let dz = 4.0;
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, dx, dy, dz).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let mut signed_vol = 0.0;
    let tri_count = mesh.indices.len() / 3;
    for t in 0..tri_count {
        let v0 = mesh.positions[mesh.indices[t * 3] as usize];
        let v1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
        let v2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
        let a = Vec3::new(v0.x(), v0.y(), v0.z());
        let b = Vec3::new(v1.x(), v1.y(), v1.z());
        let c = Vec3::new(v2.x(), v2.y(), v2.z());
        signed_vol += a.dot(b.cross(c));
    }
    signed_vol /= 6.0;

    assert!(
        signed_vol > 0.0,
        "signed volume should be positive (outward normals), got {signed_vol}"
    );

    let expected_vol = dx * dy * dz;
    let rel_err = (signed_vol - expected_vol).abs() / expected_vol;
    assert!(
        rel_err < 0.01,
        "signed volume {signed_vol} differs from expected {expected_vol} by {:.2}%",
        rel_err * 100.0
    );
}

#[test]
fn test_vertex_on_surface_sphere() {
    let radius = 2.0;
    let mut topo = Topology::new();
    let solid = crate::primitives::make_sphere(&mut topo, radius, 16).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    for (i, p) in mesh.positions.iter().enumerate() {
        let dist = (p.x().powi(2) + p.y().powi(2) + p.z().powi(2)).sqrt();
        assert!(
            (dist - radius).abs() < 1e-6,
            "vertex {i} at dist {dist} from origin, expected {radius}"
        );
    }
}

#[test]
fn test_no_t_junctions_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
    let unique: brepkit_math::det_hash::DetHashSet<(i64, i64, i64)> = mesh
        .positions
        .iter()
        .map(|p| (snap(p.x()), snap(p.y()), snap(p.z())))
        .collect();

    assert_eq!(
        unique.len(),
        8,
        "unit box should have 8 unique vertices (no T-junctions), got {}",
        unique.len()
    );
}

#[test]
fn test_circle_deflection_scaling() {
    let mut topo = Topology::new();
    let small = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
    let large = crate::primitives::make_cylinder(&mut topo, 10.0, 2.0).unwrap();

    let deflection = 0.1;
    let mesh_small = tessellate_solid(&topo, small, deflection).unwrap();
    let mesh_large = tessellate_solid(&topo, large, deflection).unwrap();

    let tri_small = mesh_small.indices.len() / 3;
    let tri_large = mesh_large.indices.len() / 3;

    assert!(
        tri_large > tri_small,
        "larger cylinder should have more triangles ({tri_large}) than smaller ({tri_small})"
    );
}

#[test]
fn test_tessellate_boolean_result_watertight() {
    let mut topo = Topology::new();
    let a = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    let b = crate::primitives::make_box(&mut topo, 1.5, 1.5, 1.5).unwrap();
    crate::transform::transform_solid(
        &mut topo,
        b,
        &brepkit_math::mat::Mat4::translation(0.5, 0.5, 0.5),
    )
    .unwrap();

    let cut = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

    let mesh = tessellate_solid(&topo, cut, 0.1).unwrap();

    let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
    let snap_pt = |p: Point3| -> (i64, i64, i64) { (snap(p.x()), snap(p.y()), snap(p.z())) };

    let mut pos_map: DetHashMap<(i64, i64, i64), usize> = DetHashMap::default();
    let mut next_id = 0_usize;
    let canonical: Vec<usize> = mesh
        .positions
        .iter()
        .map(|&p| {
            let key = snap_pt(p);
            *pos_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    let tri_count = mesh.indices.len() / 3;
    let mut half_edges: DetHashSet<(usize, usize)> = DetHashSet::default();
    for t in 0..tri_count {
        let ca = canonical[mesh.indices[t * 3] as usize];
        let cb = canonical[mesh.indices[t * 3 + 1] as usize];
        let cc = canonical[mesh.indices[t * 3 + 2] as usize];
        half_edges.insert((ca, cb));
        half_edges.insert((cb, cc));
        half_edges.insert((cc, ca));
    }

    let boundary_count = half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count();
    assert_eq!(
        boundary_count, 0,
        "boolean cut result should be watertight (0 boundary edges), got {boundary_count}"
    );
}

// -- Winding tests --

/// Helper: compute raw signed volume WITHOUT abs(), to detect winding issues.
fn signed_volume_raw(mesh: &TriangleMesh) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;
    let mut total = 0.0;
    for t in 0..tri_count {
        let v0 = pos[idx[t * 3] as usize];
        let v1 = pos[idx[t * 3 + 1] as usize];
        let v2 = pos[idx[t * 3 + 2] as usize];
        let a = Vec3::new(v0.x(), v0.y(), v0.z());
        let b = Vec3::new(v1.x(), v1.y(), v1.z());
        let c = Vec3::new(v2.x(), v2.y(), v2.z());
        total += a.dot(b.cross(c));
    }
    total / 6.0
}

#[test]
fn reversed_sphere_face_tessellation_correct_winding() {
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;

    let mut topo = Topology::new();
    let sphere = crate::primitives::make_sphere(&mut topo, 3.0, 32).unwrap();

    let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
    crate::transform::transform_solid(&mut topo, sphere, &mat).unwrap();

    let mesh_normal = tessellate_solid(&topo, sphere, 0.05).unwrap();
    let vol_normal = signed_volume_raw(&mesh_normal);

    let solid_data = topo.solid(sphere).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let face_copies: Vec<_> = shell
        .faces()
        .iter()
        .map(|&fid| {
            let face = topo.face(fid).unwrap();
            (
                face.outer_wire(),
                face.inner_wires().to_vec(),
                face.surface().clone(),
            )
        })
        .collect();

    let mut rev_face_ids = Vec::new();
    for (outer_wire, inner_wires, surface) in face_copies {
        let new_face = Face::new_reversed(outer_wire, inner_wires, surface);
        rev_face_ids.push(topo.add_face(new_face));
    }
    let rev_shell = Shell::new(rev_face_ids).unwrap();
    let rev_shell_id = topo.add_shell(rev_shell);
    let rev_solid = topo.add_solid(Solid::new(rev_shell_id, vec![]));

    let mesh_reversed = tessellate_solid(&topo, rev_solid, 0.05).unwrap();
    let vol_reversed = signed_volume_raw(&mesh_reversed);

    assert!(
        vol_normal > 0.0,
        "normal sphere signed volume should be positive, got {vol_normal}"
    );
    assert!(
        vol_reversed < 0.0,
        "reversed sphere signed volume should be negative, got {vol_reversed} \
         (this fails if tessellate_nonplanar_snap double-flips)"
    );
    assert!(
        (vol_normal + vol_reversed).abs() < 1.0,
        "normal + reversed should cancel to ~0, got {}",
        vol_normal + vol_reversed
    );
}

#[test]
fn boolean_cut_result_has_positive_signed_volume() {
    let mut topo = Topology::new();
    let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sp = crate::primitives::make_sphere(&mut topo, 3.0, 32).unwrap();
    let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
    crate::transform::transform_solid(&mut topo, sp, &mat).unwrap();

    let cut_result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, bx, sp).unwrap();

    let mesh = tessellate_solid(&topo, cut_result, 0.05).unwrap();
    let vol = signed_volume_raw(&mesh);

    assert!(
        vol > 0.0,
        "boolean cut result should have positive signed volume, got {vol}"
    );

    let expected_approx = 1000.0 - (4.0 / 3.0) * std::f64::consts::PI * 27.0;
    let rel_err = (vol - expected_approx).abs() / expected_approx;
    assert!(
        rel_err < 0.15,
        "volume {vol} too far from expected ~{expected_approx:.1} (rel error {rel_err:.3})"
    );
}

#[test]
fn per_face_tessellation_matches_face_normal() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    for &fid in shell.faces() {
        let mesh = tessellate(&topo, fid, 0.1).unwrap();
        let face = topo.face(fid).unwrap();
        if let FaceSurface::Plane { normal, .. } = face.surface()
            && mesh.indices.len() >= 3
        {
            let i0 = mesh.indices[0] as usize;
            let i1 = mesh.indices[1] as usize;
            let i2 = mesh.indices[2] as usize;
            let a = mesh.positions[i1] - mesh.positions[i0];
            let b = mesh.positions[i2] - mesh.positions[i0];
            let tri_normal = a.cross(b);
            let dot = tri_normal.dot(*normal);
            assert!(
                dot > 0.0,
                "Face normal {:?} disagrees with tri normal {:?} (dot={dot})",
                normal,
                tri_normal
            );
        }
    }
}

#[test]
fn tessellate_box_with_hole_from_boolean() {
    let mut topo = Topology::new();
    let base = crate::primitives::make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
    let hole = crate::primitives::make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    crate::transform::transform_solid(
        &mut topo,
        hole,
        &brepkit_math::mat::Mat4::translation(5.0, 5.0, -1.0),
    )
    .unwrap();

    let cut =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, base, hole).unwrap();

    let mesh = tessellate_solid(&topo, cut, 0.5).unwrap();
    assert!(!mesh.positions.is_empty(), "should produce vertices");
    assert!(!mesh.indices.is_empty(), "should produce triangles");
}

#[test]
fn tessellate_thin_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1000.0, 1.0, 1.0).unwrap();

    let mesh = tessellate_solid(&topo, solid, 1.0).unwrap();
    assert!(!mesh.positions.is_empty(), "should produce vertices");
    assert!(!mesh.indices.is_empty(), "should produce triangles");
}

#[test]
fn tessellate_small_torus_reasonable_count() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_torus(&mut topo, 5.0, 0.1, 32).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count > 100,
        "torus should produce enough triangles: got {tri_count}"
    );
    assert!(
        tri_count < 10_000,
        "small torus should not over-tessellate: got {tri_count} triangles (expected <10000)"
    );
}

// -- Gridfinity tessellation reproducers (#259) --

#[test]
fn fillet_box_triangle_count() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let box_mesh = tessellate_solid(&topo, solid, 0.1).unwrap();
    let box_tris = box_mesh.indices.len() / 3;

    let edges = brepkit_topology::explorer::solid_edges(&topo, solid).unwrap();
    let filleted = crate::fillet::fillet_rolling_ball(&mut topo, solid, &edges[..1], 1.0);
    if let Ok(filleted_id) = filleted {
        let fillet_mesh = tessellate_solid(&topo, filleted_id, 0.1).unwrap();
        let fillet_tris = fillet_mesh.indices.len() / 3;
        let ratio = fillet_tris as f64 / box_tris as f64;
        assert!(
            ratio < 10.0,
            "fillet should not over-tessellate: box={box_tris}, fillet={fillet_tris}, ratio={ratio:.1}x (issue #259)"
        );
    }
}

#[test]
fn fillet_small_radius_tessellation() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
    let edges = brepkit_topology::explorer::solid_edges(&topo, solid).unwrap();
    let filleted = crate::fillet::fillet_rolling_ball(&mut topo, solid, &edges[..1], 0.5);
    if let Ok(filleted_id) = filleted {
        let mesh = tessellate_solid(&topo, filleted_id, 0.1).unwrap();
        let tri_count = mesh.indices.len() / 3;
        assert!(
            tri_count < 50_000,
            "small-radius fillet should not over-tessellate: got {tri_count} triangles (issue #259)"
        );
    }
}

#[test]
fn torus_tessellation_count() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_torus(&mut topo, 5.0, 0.1, 32).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count < 10_000,
        "torus tessellation should be bounded: got {tri_count} triangles (issue #259)"
    );
}

/// Count distinct angular bands around a cylinder's circumference by
/// projecting lateral-face vertices to their angle about the z axis.
fn distinct_angular_bands(mesh: &TriangleMesh, radius: f64) -> usize {
    let mut bins: DetHashSet<i64> = DetHashSet::default();
    for p in &mesh.positions {
        let rr = (p.x() * p.x() + p.y() * p.y()).sqrt();
        if (rr - radius).abs() > radius * 0.05 {
            continue;
        }
        let ang = p.y().atan2(p.x());
        // 0.01 rad bins -- finer than any per-segment angle of interest.
        bins.insert((ang / 0.01).round() as i64);
    }
    bins.len()
}

#[test]
fn cylinder_small_radius_respects_angular_tolerance() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 0.5, 2.0).unwrap();

    let mesh = tessellate_solid_with_tolerance(&topo, solid, 0.1, 0.35).unwrap();
    let bands = distinct_angular_bands(&mesh, 0.5);
    let expected = (std::f64::consts::TAU / 0.35).ceil() as usize;
    assert!(
        bands >= expected,
        "small-radius cylinder should have >= {expected} angular bands, got {bands}"
    );
}

#[test]
fn torus_minor_arc_min_segments() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_torus(&mut topo, 5.0, 0.4, 32).unwrap();

    let alpha = 0.35;
    let mesh = tessellate_solid_with_tolerance(&topo, solid, 0.1, alpha).unwrap();

    // Count distinct minor-circle latitudes by binning distance from the
    // tube center circle (radius R) -- a proxy for the v direction density.
    let r_major = 5.0;
    let mut bins: DetHashSet<i64> = DetHashSet::default();
    for p in &mesh.positions {
        let rho = (p.x() * p.x() + p.y() * p.y()).sqrt();
        let dr = rho - r_major;
        bins.insert(((dr).atan2(p.z()) / 0.01).round() as i64);
    }
    let expected = (std::f64::consts::TAU / alpha).ceil() as usize;
    assert!(
        bins.len() >= expected,
        "torus minor circle should have >= {expected} bands, got {}",
        bins.len()
    );
}

#[test]
fn angular_tolerance_monotonic() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 0.5, 2.0).unwrap();

    let coarse = tessellate_solid_with_tolerance(&topo, solid, 0.1, 0.5).unwrap();
    let fine = tessellate_solid_with_tolerance(&topo, solid, 0.1, 0.2).unwrap();
    assert!(
        fine.indices.len() >= coarse.indices.len(),
        "tighter angular tol must not reduce triangles: fine={} coarse={}",
        fine.indices.len(),
        coarse.indices.len()
    );
}

#[test]
fn coarse_curvature_unchanged() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 100.0, 5.0).unwrap();

    // theta_lin << alpha here, so the angular cap is inactive and the output
    // must match the legacy linear-only path (alpha disabled => 0.0).
    let with_alpha = tessellate_solid_with_tolerance(&topo, solid, 0.01, 0.5).unwrap();
    let linear_only = tessellate_solid_with_tolerance(&topo, solid, 0.01, 0.0).unwrap();
    assert_eq!(
        with_alpha.indices.len(),
        linear_only.indices.len(),
        "large-radius geometry must be backward compatible"
    );
}

#[test]
fn small_radius_cylinder_watertight_with_angular_tol() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 0.5, 2.0).unwrap();
    let mesh = tessellate_solid_with_tolerance(&topo, solid, 0.1, 0.2).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "small-radius cylinder must stay watertight with angular tol"
    );
}

#[test]
fn fillet_cylinder_triangle_count() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 5.0, 10.0).unwrap();
    let edges = brepkit_topology::explorer::solid_edges(&topo, solid).unwrap();
    let filleted = crate::fillet::fillet_rolling_ball(&mut topo, solid, &edges[..1], 0.5);
    if let Ok(filleted_id) = filleted {
        let mesh = tessellate_solid(&topo, filleted_id, 0.1).unwrap();
        let tri_count = mesh.indices.len() / 3;
        assert!(
            tri_count < 50_000,
            "fillet on cylinder should not over-tessellate: got {tri_count} triangles (issue #259)"
        );
    }
}

/// Build a solid by extruding a closed ellipse (`semi_major`, `semi_minor`) in
/// the XY plane by `height` along +Z. The boundary is a single closed
/// `Ellipse` edge, matching what `sketchEllipse(a, b).extrude(h)` produces.
fn extrude_ellipse(
    topo: &mut Topology,
    semi_major: f64,
    semi_minor: f64,
    height: f64,
) -> brepkit_topology::solid::SolidId {
    let center = Point3::new(0.0, 0.0, 0.0);
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let ellipse =
        brepkit_math::curves::Ellipse3D::new(center, normal, semi_major, semi_minor).unwrap();

    // A single closed edge (start == end) at the major-axis vertex (t = 0).
    let seam = ellipse.evaluate(0.0);
    let vid = topo.add_vertex(Vertex::new(seam, 1e-7));
    let edge = topo.add_edge(Edge::new(vid, vid, EdgeCurve::Ellipse(ellipse)));
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));

    crate::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap()
}

#[test]
fn eccentric_ellipse_extrude_volume_matches_analytic() {
    use std::f64::consts::PI;

    // Regression guard for the #717 ellipse tessellation density drop.
    // sketchEllipse(5, 2).extrude(10) must mesh densely enough that the
    // tessellation-derived volume matches the analytic pi*a*b*h within the
    // brepjs parity tolerance (toBeCloseTo(vol, 0) => |err| < 0.5 absolute).
    let cases = [
        (5.0_f64, 2.0_f64, 10.0_f64),
        (10.0, 1.0, 4.0),
        (8.0, 3.0, 2.0),
    ];
    for (a, b, h) in cases {
        let mut topo = Topology::new();
        let solid = extrude_ellipse(&mut topo, a, b, h);
        // brepjs measureVolume uses DEFAULT_DEFLECTION = 0.01.
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let analytic = PI * a * b * h;
        assert!(
            (vol - analytic).abs() < 0.5,
            "ellipse({a},{b}).extrude({h}): mesh volume {vol:.4} vs analytic {analytic:.4} \
             (err {:.4}); eccentric ellipse wall under-tessellated",
            (vol - analytic).abs()
        );
    }
}

#[test]
fn ellipse_wall_facet_count_is_curvature_appropriate() {
    // The elliptical wall must carry enough facets to resolve its curvature
    // at the default deflection. For ellipse(5, 2) at deflection 0.01 a
    // curvature-faithful sampler needs ~200 segments around the loop; assert a
    // conservative floor so a future density regression is caught directly at
    // the tessellation layer (not only via the volume check).
    let mut topo = Topology::new();
    let solid = extrude_ellipse(&mut topo, 5.0, 2.0, 10.0);
    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    let n_pos = mesh.positions.len();
    assert!(
        n_pos >= 200,
        "ellipse(5,2) wall under-tessellated: only {n_pos} mesh vertices at deflection 0.01"
    );
}

#[test]
fn circle_and_degenerate_ellipse_do_not_over_tessellate() {
    // The fix must not blow up density on near-circular or circular inputs.
    // A near-circular ellipse(5, 5) extrude should produce a similar facet
    // count to a true circle of the same radius, and stay well bounded.
    let mut topo_e = Topology::new();
    let solid_e = extrude_ellipse(&mut topo_e, 5.0, 5.0, 10.0);
    let mesh_e = tessellate_solid(&topo_e, solid_e, 0.01).unwrap();
    let n_e = mesh_e.positions.len();

    let mut topo_c = Topology::new();
    let solid_c = crate::primitives::make_cylinder(&mut topo_c, 5.0, 10.0).unwrap();
    let mesh_c = tessellate_solid(&topo_c, solid_c, 0.01).unwrap();
    let n_c = mesh_c.positions.len();

    assert!(
        n_e < 4 * n_c.max(1),
        "near-circular ellipse over-tessellates: {n_e} verts vs cylinder {n_c}"
    );
    assert!(
        n_e < 5_000,
        "near-circular ellipse(5,5) over-tessellates: {n_e} mesh vertices"
    );
}

// -- Grouped solid tessellation (wasm export path) --

/// Count boundary and non-manifold edges with vertices unified by quantized
/// (1e-4) position keys -- the same equivalence an STL export induces.
fn quantized_edge_defects(mesh: &TriangleMesh) -> (usize, usize) {
    const EXPORT_GRID: f64 = 1e-4;

    let mut pos_to_canonical: DetHashMap<(i64, i64, i64), u32> = DetHashMap::default();
    let mut canonical_ids: Vec<u32> = Vec::with_capacity(mesh.positions.len());
    for pos in &mesh.positions {
        let key = point_merge_key(*pos, EXPORT_GRID);
        let next = pos_to_canonical.len() as u32;
        canonical_ids.push(*pos_to_canonical.entry(key).or_insert(next));
    }

    let mut edge_count: DetHashMap<(u32, u32), (u32, u32)> = DetHashMap::default();
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = canonical_ids[tri[0] as usize];
        let i1 = canonical_ids[tri[1] as usize];
        let i2 = canonical_ids[tri[2] as usize];
        if i0 == i1 || i1 == i2 || i2 == i0 {
            continue;
        }
        for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
            let entry = edge_count.entry((a.min(b), a.max(b))).or_default();
            if a < b {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }

    let boundary = edge_count
        .values()
        .filter(|&&(f, r)| f + r == 1 || (f + r == 2 && (f == 0 || r == 0)))
        .count();
    let non_manifold = edge_count.values().filter(|&&(f, r)| f + r > 2).count();
    (boundary, non_manifold)
}

/// Box(21^3) cut by a through-cylinder(r=3.75) at (6,6): the canonical
/// boolean-result solid that the wasm `tessellateSolidGrouped` path exported
/// with T-junctions.
fn make_box_with_through_hole(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    use brepkit_math::mat::Mat4;
    let bx = crate::primitives::make_box(topo, 21.0, 21.0, 21.0).unwrap();
    let cyl = crate::primitives::make_cylinder(topo, 3.75, 30.0).unwrap();
    crate::transform::transform_solid(topo, cyl, &Mat4::translation(6.0, 6.0, -5.0)).unwrap();
    crate::boolean::boolean(topo, crate::boolean::BooleanOp::Cut, bx, cyl).unwrap()
}

/// Regression for the wasm `tessellateSolidGrouped` export path: the previous
/// implementation merged standalone per-face tessellations, whose mismatched
/// boundary vertices produced 156 boundary edges on this solid even under
/// STL-export (1e-4) vertex quantization. The grouped output must now match
/// the watertight shared-edge-pool invariant.
#[test]
fn grouped_tessellation_watertight_box_cut_cylinder() {
    let mut topo = Topology::new();
    let solid = make_box_with_through_hole(&mut topo);

    // The watertight ungrouped path is the reference: it must pass.
    let watertight = tessellate_solid_with_tolerance(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
    .unwrap();
    let (wb, wn) = quantized_edge_defects(&watertight);
    assert_eq!(
        wb, 0,
        "watertight path must have 0 boundary edges, got {wb}"
    );
    assert_eq!(
        wn, 0,
        "watertight path must have 0 non-manifold edges, got {wn}"
    );

    let (mesh, _offsets) = tessellate_solid_grouped_with_tolerance(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
    .unwrap();
    let (gb, gn) = quantized_edge_defects(&mesh);
    assert_eq!(
        gb, 0,
        "grouped tessellation must have 0 boundary edges, got {gb}"
    );
    assert_eq!(
        gn, 0,
        "grouped tessellation must have 0 non-manifold edges, got {gn}"
    );

    // Grouped output is a triangle-order permutation of the ungrouped mesh.
    assert_eq!(mesh.indices.len(), watertight.indices.len());
    assert_eq!(mesh.positions.len(), watertight.positions.len());
}

#[test]
fn grouped_tessellation_offsets_invariants() {
    let mut topo = Topology::new();
    let solid = make_box_with_through_hole(&mut topo);
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

    let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
    .unwrap();

    assert_eq!(
        offsets.len(),
        faces.len() + 1,
        "one offset per face plus sentinel (brepjs maps faceHash positionally)"
    );
    assert_eq!(offsets[0], 0);
    assert_eq!(
        *offsets.last().unwrap() as usize,
        mesh.indices.len(),
        "sentinel must equal indices.len()"
    );
    for w in offsets.windows(2) {
        assert!(w[0] <= w[1], "offsets must be monotonic");
        assert_eq!((w[1] - w[0]) % 3, 0, "group sizes must be whole triangles");
    }
}

/// Group alignment check: every triangle in face i's group must lie on face
/// i's surface. A silent offset misalignment (e.g. triangle deletion without
/// filtering the attribution array) would put cylinder triangles in plane
/// groups and fail here.
#[test]
fn grouped_tessellation_triangles_lie_on_their_face() {
    let mut topo = Topology::new();
    let solid = make_box_with_through_hole(&mut topo);
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

    let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
    .unwrap();

    let mut nonempty = 0;
    for (fi, &face_id) in faces.iter().enumerate() {
        let surf = topo.face(face_id).unwrap().surface().clone();
        let group = &mesh.indices[offsets[fi] as usize..offsets[fi + 1] as usize];
        if !group.is_empty() {
            nonempty += 1;
        }
        for &vid in group {
            let p = mesh.positions[vid as usize];
            let dist = match &surf {
                FaceSurface::Plane { normal, d } => {
                    (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                }
                FaceSurface::Cylinder(cyl) => {
                    let to_p = p - cyl.origin();
                    let axial = to_p.dot(cyl.axis());
                    ((to_p - cyl.axis() * axial).length() - cyl.radius()).abs()
                }
                _ => 0.0,
            };
            assert!(
                dist < 1e-6,
                "face {fi} group contains a vertex {dist:.2e} off its surface"
            );
        }
    }
    assert!(
        nonempty >= faces.len() - 1,
        "expected nearly all groups nonempty, got {nonempty}/{}",
        faces.len()
    );
}

/// Build a closed prism solid whose top/bottom caps are the given planar
/// polygon (shared boundary edges with the side walls). Used to exercise
/// watertight meshing of a self-intersecting (pinched) planar cap.
fn build_prism(
    topo: &mut Topology,
    poly2d: &[(f64, f64)],
    z0: f64,
    z1: f64,
) -> brepkit_topology::solid::SolidId {
    build_prism_with_holed_wall(topo, poly2d, z0, z1, None)
}

/// [`build_prism`] with a small rectangular hole through the wall over side
/// `holed` (from vertex `holed` to the next), if given.
fn build_prism_with_holed_wall(
    topo: &mut Topology,
    poly2d: &[(f64, f64)],
    z0: f64,
    z1: f64,
    holed: Option<usize>,
) -> brepkit_topology::solid::SolidId {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let n = poly2d.len();
    let tol = 1e-7;
    let top_v: Vec<_> = poly2d
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, z1), tol)))
        .collect();
    let bot_v: Vec<_> = poly2d
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, z0), tol)))
        .collect();
    let top_e: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(top_v[i], top_v[(i + 1) % n], EdgeCurve::Line)))
        .collect();
    let bot_e: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(bot_v[i], bot_v[(i + 1) % n], EdgeCurve::Line)))
        .collect();
    let vert_e: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(bot_v[i], top_v[i], EdgeCurve::Line)))
        .collect();

    let top_wire = topo.add_wire(
        Wire::new(
            top_e.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let bot_wire = topo.add_wire(
        Wire::new(
            bot_e.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let top_face = topo.add_face(Face::new(
        top_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z1,
        },
    ));
    let bot_face = topo.add_face(Face::new(
        bot_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -z0,
        },
    ));

    let mut faces = vec![top_face, bot_face];
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = poly2d[i];
        let (xj, yj) = poly2d[j];
        let (dx, dy) = (xj - xi, yj - yi);
        // Horizontal normal perpendicular to the wall (dir x z-axis).
        let nrm = Vec3::new(dy, -dx, 0.0);
        let len = (nrm.x() * nrm.x() + nrm.y() * nrm.y()).sqrt().max(1e-12);
        let normal = Vec3::new(nrm.x() / len, nrm.y() / len, 0.0);
        let d = normal.x() * xi + normal.y() * yi;
        let quad = Wire::new(
            vec![
                OrientedEdge::new(bot_e[i], true),
                OrientedEdge::new(vert_e[j], true),
                OrientedEdge::new(top_e[i], false),
                OrientedEdge::new(vert_e[i], false),
            ],
            true,
        )
        .unwrap();
        let quad_wire = topo.add_wire(quad);
        let mut holes = Vec::new();
        if holed == Some(i) {
            let mut at = |t: f64, z: f64| {
                let p = Point3::new(xi + t * dx, yi + t * dy, z0 + z * (z1 - z0));
                topo.add_vertex(Vertex::new(p, tol))
            };
            let corners = [at(0.3, 0.4), at(0.3, 0.6), at(0.7, 0.6), at(0.7, 0.4)];
            let edges: Vec<OrientedEdge> = (0..4)
                .map(|k| {
                    let e = Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line);
                    OrientedEdge::new(topo.add_edge(e), true)
                })
                .collect();
            holes.push(topo.add_wire(Wire::new(edges, true).unwrap()));
        }
        faces.push(topo.add_face(Face::new(
            quad_wire,
            holes,
            FaceSurface::Plane { normal, d },
        )));
    }

    let shell = topo.add_shell(Shell::new(faces).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// Regression: a boolean occasionally emits a planar face whose outer wire
/// pinches through zero width — two boundary arcs overlap by a few hundred
/// microns, so the projected boundary polygon self-intersects. CDT recovers the
/// crossing constraints with Steiner vertices; dropping their triangles left a
/// hole in the ledge (nonzero boundary edges). These two polygons are the exact
/// self-intersecting ledge boundaries captured from the gridfinity "2x2 with
/// stadium slot" cut at deflection 0.1 (28 pts) and 0.05 (34 pts). The prism
/// meshing must stay watertight at both, with each cap covering the ledge's
/// own area: a fan from one corner covers it twice over. The two walls whose
/// bases cross pass through each other along the vertical line over the
/// crossing, and both mesh it, so that line is the one edge more than two
/// triangles may share.
#[test]
#[allow(clippy::unreadable_literal)]
fn pinched_ledge_prism_is_watertight() {
    // defl 0.1 capture (28 pts).
    let poly_28: Vec<(f64, f64)> = vec![
        (-34.56, -40.55),
        (-16.54, -40.550000000000004),
        (-15.738116168931496, -40.49608307905734),
        (-14.950668099841263, -40.3353029453993),
        (-14.191831677633331, -40.070554012967634),
        (-13.475267711451144, -39.7066023743365),
        (-12.813876008545115, -39.25000000003244),
        (-15.240000000000073, -39.25000000000008),
        (-35.8599999999999, -39.25000000000011),
        (-38.0, -39.25000000000011),
        (-38.38627124296872, -39.18882064536907),
        (-38.73473156536566, -39.011271242968796),
        (-39.01127124296875, -38.73473156536567),
        (-39.188820645369, -38.386271242968725),
        (-39.25000000000001, -38.0),
        (-39.25000000000001, -35.85999999999999),
        (-39.250000000000014, -33.240000000000016),
        (-39.250000000000014, -30.813876008504252),
        (-39.706602374315146, -31.475267711415285),
        (-40.07055401295537, -32.19183167760453),
        (-40.335302945393764, -32.950668099821144),
        (-40.49608307905595, -33.73811616892115),
        (-40.55, -34.54),
        (-40.55, -34.56),
        (-40.256828532607976, -36.411011796305935),
        (-39.40601179630594, -38.080833661231914),
        (-38.08083366123192, -39.40601179630594),
        (-36.411011796305935, -40.256828532607976),
    ];
    // defl 0.05 capture (34 pts).
    let poly_34: Vec<(f64, f64)> = vec![
        (-34.56, -40.55),
        (-16.54, -40.550000000000004),
        (-15.966381445824009, -40.52247111140214),
        (-15.398035372875258, -40.4401374805693),
        (-14.8401857997605, -40.30375588658364),
        (-14.297960265295309, -40.114579896626324),
        (-13.776342698138523, -39.87434834366997),
        (-13.280127606436032, -39.58526934379438),
        (-12.813876008545115, -39.25000000003244),
        (-15.240000000000073, -39.25000000000008),
        (-35.8599999999999, -39.25000000000011),
        (-38.0, -39.25000000000011),
        (-38.38627124296872, -39.18882064536907),
        (-38.73473156536566, -39.011271242968796),
        (-39.01127124296875, -38.73473156536567),
        (-39.188820645369, -38.386271242968725),
        (-39.25000000000001, -38.0),
        (-39.25000000000001, -35.85999999999999),
        (-39.250000000000014, -33.240000000000016),
        (-39.250000000000014, -30.813876008504252),
        (-39.58526934377004, -31.280127606398516),
        (-39.87434834365278, -31.776342698105463),
        (-40.11457989661517, -32.29796026526766),
        (-40.30375588657729, -32.84018579973906),
        (-40.440137480566456, -33.398035372860626),
        (-40.52247111140143, -33.96638144581659),
        (-40.55, -34.54),
        (-40.55, -34.56),
        (-40.399818193969125, -35.892900394398325),
        (-39.9568035187355, -37.15896359731418),
        (-39.243170579983506, -38.294703913133816),
        (-38.294703913133816, -39.2431705799835),
        (-37.15896359731418, -39.9568035187355),
        (-35.892900394398325, -40.399818193969125),
    ];

    for (label, poly) in [("28pt/defl0.1", &poly_28), ("34pt/defl0.05", &poly_34)] {
        let mut topo = Topology::new();
        let solid = build_prism(&mut topo, poly, 0.0, 4.0);
        let outline_area = polygon_area(poly).abs();
        for defl in [0.1, 0.05] {
            let mesh = tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                boundary_edge_count(&mesh),
                0,
                "{label} defl={defl}: pinched ledge left boundary edges (crack)"
            );
            for z in [0.0, 4.0] {
                let cap = level_area(&mesh, z);
                assert!(
                    (cap - outline_area).abs() < 0.01 * outline_area,
                    "{label} defl={defl}: cap at z={z} covers {cap}, the ledge {outline_area}"
                );
            }
            let mut uses: DetHashMap<(u32, u32), u32> = DetHashMap::default();
            for t in mesh.indices.chunks_exact(3) {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *uses.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            for (&(a, b), _) in uses.iter().filter(|&(_, &n)| n > 2) {
                let (pa, pb) = (mesh.positions[a as usize], mesh.positions[b as usize]);
                assert!(
                    (pa.x() - pb.x()).abs() < 1e-9 && (pa.y() - pb.y()).abs() < 1e-9,
                    "{label} defl={defl}: a non-manifold edge off the walls' crossing"
                );
            }
        }
    }
}

/// A wall with a hole meshes as a planar face with holes, in a batch of its
/// own; when the cap beside it crosses itself on that wall's side, the wall
/// must still take the crossing vertices or the two crack apart there. The
/// ledge's side 25 crosses two others.
#[test]
#[allow(clippy::unreadable_literal)]
fn a_holed_wall_beside_a_crossing_cap_takes_its_crossing_vertices() {
    let poly: Vec<(f64, f64)> = vec![
        (-34.56, -40.55),
        (-16.54, -40.550000000000004),
        (-15.738116168931496, -40.49608307905734),
        (-14.950668099841263, -40.3353029453993),
        (-14.191831677633331, -40.070554012967634),
        (-13.475267711451144, -39.7066023743365),
        (-12.813876008545115, -39.25000000003244),
        (-15.240000000000073, -39.25000000000008),
        (-35.8599999999999, -39.25000000000011),
        (-38.0, -39.25000000000011),
        (-38.38627124296872, -39.18882064536907),
        (-38.73473156536566, -39.011271242968796),
        (-39.01127124296875, -38.73473156536567),
        (-39.188820645369, -38.386271242968725),
        (-39.25000000000001, -38.0),
        (-39.25000000000001, -35.85999999999999),
        (-39.250000000000014, -33.240000000000016),
        (-39.250000000000014, -30.813876008504252),
        (-39.706602374315146, -31.475267711415285),
        (-40.07055401295537, -32.19183167760453),
        (-40.335302945393764, -32.950668099821144),
        (-40.49608307905595, -33.73811616892115),
        (-40.55, -34.54),
        (-40.55, -34.56),
        (-40.256828532607976, -36.411011796305935),
        (-39.40601179630594, -38.080833661231914),
        (-38.08083366123192, -39.40601179630594),
        (-36.411011796305935, -40.256828532607976),
    ];
    let mut topo = Topology::new();
    let solid = build_prism_with_holed_wall(&mut topo, &poly, 0.0, 4.0, Some(25));
    // The hole has no faces behind it, so its own rim is open; nothing else
    // may be.
    for defl in [0.1, 0.05] {
        let mesh = tessellate_solid(&topo, solid, defl).unwrap();
        let mut half: DetHashSet<(u32, u32)> = DetHashSet::default();
        for t in mesh.indices.chunks_exact(3) {
            half.extend([(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]);
        }
        for &(a, b) in &half {
            if half.contains(&(b, a)) {
                continue;
            }
            let (za, zb) = (
                mesh.positions[a as usize].z(),
                mesh.positions[b as usize].z(),
            );
            assert!(
                (1.5..=2.5).contains(&za) && (1.5..=2.5).contains(&zb),
                "defl={defl}: an open edge at z = {za}..{zb} cracks the holed wall from the cap"
            );
        }
    }
}

/// Signed area of a closed polygon in the xy-plane.
fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    (0..n)
        .map(|i| {
            let ((x0, y0), (x1, y1)) = (poly[i], poly[(i + 1) % n]);
            x0 * y1 - x1 * y0
        })
        .sum::<f64>()
        / 2.0
}

/// Total area of the triangles lying in the plane `z`.
fn level_area(mesh: &TriangleMesh, z: f64) -> f64 {
    mesh.indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]))
        .filter(|p| p.iter().all(|q| (q.z() - z).abs() < 1e-9))
        .map(|[a, b, c]| 0.5 * (b - a).cross(c - a).length())
        .sum()
}

/// A 40 x 30 rectangle whose base carries a trochoid that turns up into it
/// and back across itself, sampled about 0.09 apart.
fn curled_outline() -> Vec<(f64, f64)> {
    use std::f64::consts::TAU;
    let (a, r, steps) = (1.0, 3.0, 220);
    let mut poly = vec![(0.0, 0.0)];
    for k in 0..=steps {
        let t = TAU * f64::from(k) / f64::from(steps);
        poly.push((10.0 + a * t + r * t.sin(), r * (1.0 - t.cos())));
    }
    poly.extend([(40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]);
    poly
}

/// A rim that curls through itself the way it runs (a freeform wall folding
/// over near its foot) winds twice round the inside of the curl: a notch out
/// of the cap, which must cover only what the rim winds round once. Where it
/// does, the cap and the wall both run the curl's sides the same way round,
/// so they are no crack, and the curl's samples, closer than twice the
/// deflection, must not be welded into one point.
#[test]
fn a_cap_whose_rim_curls_through_itself_leaves_the_curl_out() {
    let poly = curled_outline();

    // The curl is the loop between the two sides that cross.
    let n = poly.len();
    let cross = |p: (f64, f64), q: (f64, f64), o: (f64, f64)| {
        (q.0 - p.0) * (o.1 - p.1) - (q.1 - p.1) * (o.0 - p.0)
    };
    let (i, j) = (0..n)
        .flat_map(|i| (i + 2..n).map(move |j| (i, j)))
        .find(|&(i, j)| {
            let (p, q, s, t) = (poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]);
            cross(p, q, s) * cross(p, q, t) < 0.0 && cross(s, t, p) * cross(s, t, q) < 0.0
        })
        .unwrap();
    let (p, q, s, t) = (poly[i], poly[i + 1], poly[j], poly[j + 1]);
    let f = cross(s, t, p) / (cross(s, t, p) - cross(s, t, q));
    let crossing = (p.0 + f * (q.0 - p.0), p.1 + f * (q.1 - p.1));
    let mut curl = vec![crossing];
    curl.extend_from_slice(&poly[i + 1..=j]);
    let once_round = polygon_area(&poly) - 2.0 * polygon_area(&curl);
    let walls: f64 = (0..n)
        .map(|k| {
            let ((x0, y0), (x1, y1)) = (poly[k], poly[(k + 1) % n]);
            (x1 - x0).hypot(y1 - y0)
        })
        .sum::<f64>()
        * 5.0;

    let mut topo = Topology::new();
    let solid = build_prism(&mut topo, &poly, 0.0, 5.0);
    for defl in [0.1, 0.05] {
        let mesh = tessellate_solid(&topo, solid, defl).unwrap();
        for z in [0.0, 5.0] {
            let cap = level_area(&mesh, z);
            assert!(
                (cap - once_round).abs() < 1e-6 * once_round,
                "defl={defl}: cap at z={z} covers {cap}, the rim winds once round {once_round}"
            );
        }
        let total: f64 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]))
            .map(|[a, b, c]| 0.5 * (b - a).cross(c - a).length())
            .sum();
        let side = total - level_area(&mesh, 0.0) - level_area(&mesh, 5.0);
        assert!(
            (side - walls).abs() < 1e-6 * walls,
            "defl={defl}: walls mesh {side} of {walls}"
        );
        let base = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap()[1];
        let alone = level_area(&tessellate(&topo, base, defl).unwrap(), 0.0);
        assert!(
            (alone - once_round).abs() < 1e-6 * once_round,
            "defl={defl}: the base meshed alone covers {alone} of {once_round}"
        );
    }
}

/// A developable band whose ruling edge carries vertices near both rims (a
/// pin's tip line sectioned by the knuckle's step plane 0.4 mm inside each
/// end) must still be meshed with every triangle inside one rim step of arc.
/// A boundary-only Delaunay fans flat triangles from such a vertex across
/// the whole 70 degree span, and every mesh consumer (export, volume) then
/// under-counts the bore by that span's sagitta.
///
/// Ready repro, ignored until the display/export mesher seeds developable
/// bands: an interior row alone still leaves fans (the Delaunay of a band
/// whose length dwarfs its arc is anisotropic), and refining fat triangles
/// by their centroids diverges on full-turn walls. See the roadmap row.
#[test]
#[ignore = "ready repro: the export mesher fans a band whose ruling carries vertices"]
fn band_with_split_ruling_keeps_triangles_on_the_arc() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;

    let mut topo = Topology::new();
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), axis, 1.0).unwrap();
    let u1 = 70.0_f64.to_radians();
    let (c, s) = (u1.cos(), u1.sin());
    let pts = [
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(c, s, 0.0),
        Point3::new(c, s, 0.4),
        Point3::new(c, s, 9.6),
        Point3::new(c, s, 10.0),
        Point3::new(1.0, 0.0, 10.0),
    ];
    let v: Vec<_> = pts
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let rim =
        |z: f64| EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, z), axis, 1.0).unwrap());
    let bottom = topo.add_edge(Edge::new(v[0], v[1], rim(0.0)));
    let r1 = topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Line));
    let r2 = topo.add_edge(Edge::new(v[2], v[3], EdgeCurve::Line));
    let r3 = topo.add_edge(Edge::new(v[3], v[4], EdgeCurve::Line));
    let top = topo.add_edge(Edge::new(v[5], v[4], rim(10.0)));
    let seam = topo.add_edge(Edge::new(v[5], v[0], EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(bottom, true),
            OrientedEdge::new(r1, true),
            OrientedEdge::new(r2, true),
            OrientedEdge::new(r3, true),
            OrientedEdge::new(top, false),
            OrientedEdge::new(seam, true),
        ],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));

    let face_data = topo.face(face).unwrap().clone();
    let edge_global_indices: DetHashMap<usize, Vec<u32>> = DetHashMap::default();
    let mut merged = TriangleMesh::default();
    let mut point_to_global: DetHashMap<(i64, i64, i64), u32> = DetHashMap::default();
    super::nonplanar::tessellate_nonplanar_cdt(
        &topo,
        face,
        &face_data,
        0.01,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        false,
        &edge_global_indices,
        &mut merged,
        &mut point_to_global,
    )
    .unwrap();

    // Every vertex on the surface, every triangle centroid within a few
    // chord sagittas of it (a 5 degree step sags 0.001; a 70 degree fan
    // triangle's centroid sits 0.18 inside).
    for p in &merged.positions {
        assert!(
            (p.x().hypot(p.y()) - 1.0).abs() < 1e-6,
            "vertex off the cylinder: {p:?}"
        );
    }
    let mut worst = (0.0_f64, String::new());
    for t in merged.indices.chunks_exact(3) {
        let q = |k: usize| merged.positions[t[k] as usize];
        let (a, b, c) = (q(0), q(1), q(2));
        let cx = (a.x() + b.x() + c.x()) / 3.0;
        let cy = (a.y() + b.y() + c.y()) / 3.0;
        let sag = 1.0 - cx.hypot(cy);
        if sag > worst.0 {
            let ang = |p: Point3| p.y().atan2(p.x()).to_degrees();
            worst = (
                sag,
                format!(
                    "({:.2}deg,z{:.2}) ({:.2}deg,z{:.2}) ({:.2}deg,z{:.2})",
                    ang(a),
                    a.z(),
                    ang(b),
                    b.z(),
                    ang(c),
                    c.z()
                ),
            );
        }
    }
    // The bar: a triangle across ONE rim step (two vertices at one sample,
    // the third at the next) has its centroid this far inside; a fan across
    // the 70 degree arc sits 0.18 in.
    let mut rim: Vec<f64> = merged
        .positions
        .iter()
        .filter(|p| p.z().abs() < 1e-9)
        .map(|p| p.y().atan2(p.x()))
        .collect();
    rim.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let step = rim.windows(2).map(|w| w[1] - w[0]).fold(0.0_f64, f64::max);
    let one_step = 1.0 - ((2.0 + step.cos()).powi(2) + step.sin().powi(2)).sqrt() / 3.0;
    assert!(
        worst.0 < 1.5 * one_step,
        "a triangle centroid sits {:.4} inside the cylinder at {} (one rim step allows {:.4}): the band was fanned across its arc",
        worst.0,
        worst.1,
        one_step
    );
}

/// #1487: constraint recovery can mint Steiner vertices the caller's
/// `cdt_to_global` never tracked. Pre-#1478 the interior-grid resize masked
/// the gap; with the curvature floor off, developable faces insert no
/// interior points and the vote loop indexed past `final_global_ids`.
/// A bowtie boundary in UV forces the constrained-crossing split, the
/// deterministic way to mint an untracked Steiner vertex.
#[test]
fn cdt_covers_steiner_vertices_from_constraint_recovery() {
    use brepkit_math::surfaces::CylindricalSurface;

    let mut topo = Topology::new();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let u1 = 1.0_f64;
    let pts = [
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(u1.cos(), u1.sin(), 1.0),
        Point3::new(u1.cos(), u1.sin(), 0.0),
        Point3::new(1.0, 0.0, 1.0),
    ];
    let verts: Vec<_> = pts
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..4)
        .map(|i| topo.add_edge(Edge::new(verts[i], verts[(i + 1) % 4], EdgeCurve::Line)))
        .collect();
    let wire = Wire::new(
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));

    let face_data = topo.face(face).unwrap().clone();
    let edge_global_indices: DetHashMap<usize, Vec<u32>> = DetHashMap::default();
    let mut merged = TriangleMesh::default();
    let mut point_to_global: DetHashMap<(i64, i64, i64), u32> = DetHashMap::default();

    let result = super::nonplanar::tessellate_nonplanar_cdt(
        &topo,
        face,
        &face_data,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        false,
        &edge_global_indices,
        &mut merged,
        &mut point_to_global,
    );
    assert!(result.is_ok(), "CDT tessellation failed: {result:?}");

    let n = merged.positions.len() as u32;
    for &idx in &merged.indices {
        assert!(idx < n, "triangle index {idx} out of bounds (len {n})");
    }
}

/// A right triangle with a vertex on the axis, revolved a quarter turn: the
/// cone face meshed on its own covers its quarter (`π √5 / 4`) however the
/// solid is turned or mirrored.
#[test]
fn a_revolved_cone_quarter_meshes_its_own_span() {
    use brepkit_math::mat::Mat4;

    let truth = std::f64::consts::PI * 5.0_f64.sqrt() / 4.0;
    for pose in 0..4 {
        let mut topo = Topology::new();
        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 2.0),
        ];
        let ids: Vec<_> = corners
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let edges = (0..3)
            .map(|i| {
                let e = topo.add_edge(Edge::new(ids[i], ids[(i + 1) % 3], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: 0.0,
        };
        let profile = topo.add_face(Face::new(wire, vec![], plane));
        let origin = Point3::new(0.0, 0.0, 0.0);
        let mut solid = crate::revolve::revolve(
            &mut topo,
            profile,
            origin,
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();
        match pose {
            1 => crate::transform::transform_solid(
                &mut topo,
                solid,
                &Mat4::rotation_x(std::f64::consts::PI),
            )
            .unwrap(),
            2 => crate::transform::transform_solid(
                &mut topo,
                solid,
                &Mat4::rotation_y(std::f64::consts::FRAC_PI_2),
            )
            .unwrap(),
            3 => {
                solid = crate::mirror::mirror(
                    &mut topo,
                    solid,
                    Point3::new(0.1, 0.0, 0.0),
                    Vec3::new(1.0, 0.2, 0.0),
                )
                .unwrap();
            }
            _ => {}
        }
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        let cone = faces
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
            .unwrap();
        let mesh = tessellate(&topo, cone, 0.002).unwrap();
        let area: f64 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum();
        assert!(
            (area - truth).abs() < 2e-3 * truth,
            "pose {pose}: cone meshed alone has area {area}, truth {truth}"
        );
    }
}

/// The same solid meshes in the same vertex and triangle order, and with the
/// same normals bit for bit, whatever the arena held before it: a caller that
/// fingerprints or sums a mesh in order (a feature cache comparing two builds)
/// must read two builds alike. The solids are turned off the axes so their
/// shared vertices' normals are sums whose rounding depends on the order.
#[test]
fn a_solid_meshes_in_the_same_order_after_any_history() {
    type Bits = (Vec<[u64; 3]>, Vec<[u64; 3]>, Vec<u32>, Vec<u32>);
    let bits = |p: &[Point3], n: &[Vec3]| {
        (
            p.iter()
                .map(|q| [q.x().to_bits(), q.y().to_bits(), q.z().to_bits()])
                .collect::<Vec<_>>(),
            n.iter()
                .map(|v| [v.x().to_bits(), v.y().to_bits(), v.z().to_bits()])
                .collect::<Vec<_>>(),
        )
    };
    let turn = brepkit_math::mat::Mat4::rotation_x(0.37) * brepkit_math::mat::Mat4::rotation_z(1.1);
    let mesh_after = |prior: usize, cylinder: bool| -> Bits {
        let mut topo = Topology::new();
        for _ in 0..prior {
            crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        }
        let solid = if cylinder {
            crate::primitives::make_cylinder(&mut topo, 2.0, 5.0).unwrap()
        } else {
            crate::primitives::make_box(&mut topo, 10.0, 6.0, 4.0).unwrap()
        };
        crate::transform::transform_solid(&mut topo, solid, &turn).unwrap();
        let (mesh, offsets) =
            tessellate_solid_grouped_with_tolerance(&topo, solid, 0.1, 0.5).unwrap();
        let (p, n) = bits(&mesh.positions, &mesh.normals);
        (p, n, mesh.indices, offsets)
    };
    for cylinder in [false, true] {
        let first = mesh_after(0, cylinder);
        for prior in 1..4 {
            assert!(
                mesh_after(prior, cylinder) == first,
                "after {prior} earlier boxes, cylinder = {cylinder}"
            );
        }
    }
}

/// The nine control points and weights of a rational quadratic full circle
/// of unit radius in the xy plane, starting at angle `start` and running
/// counter-clockwise.
fn unit_circle_polygon(start: f64) -> Vec<(f64, f64, f64)> {
    (0..9)
        .map(|k| {
            let angle = start + f64::from(k) * std::f64::consts::FRAC_PI_4;
            let (scale, weight) = if k % 2 == 0 {
                (1.0, 1.0)
            } else {
                (std::f64::consts::SQRT_2, std::f64::consts::FRAC_1_SQRT_2)
            };
            (scale * angle.cos(), scale * angle.sin(), weight)
        })
        .collect()
}

/// The knots of [`unit_circle_polygon`] over `[lo, hi]`.
fn circle_knots(lo: f64, hi: f64) -> Vec<f64> {
    let q = |k: f64| lo + (hi - lo) * k / 4.0;
    vec![
        q(0.0),
        q(0.0),
        q(0.0),
        q(1.0),
        q(1.0),
        q(2.0),
        q(2.0),
        q(3.0),
        q(3.0),
        q(4.0),
        q(4.0),
        q(4.0),
    ]
}

/// A rational NURBS surface revolving the `(radius, z, weight)` profile
/// (its degree read off its knots) about the z axis, the turn running over
/// `turn` along u from +x, or, with `turn_along_v`, along v from -x (the
/// profile then along u), as files carry surfaces of revolution.
fn revolved_nurbs(
    profile: &[(f64, f64, f64)],
    profile_knots: Vec<f64>,
    turn_along_v: bool,
    turn: (f64, f64),
) -> NurbsSurface {
    use std::f64::consts::PI;
    let profile_degree = profile_knots.len() - profile.len() - 1;
    let start = if turn_along_v { PI } else { 0.0 };
    let turn_knots = circle_knots(turn.0, turn.1);
    let circle = unit_circle_polygon(start);
    let point = |&(cx, cy, cw): &(f64, f64, f64), &(r, z, pw): &(f64, f64, f64)| {
        (Point3::new(r * cx, r * cy, z), cw * pw)
    };
    let grid: Vec<Vec<(Point3, f64)>> = if turn_along_v {
        profile
            .iter()
            .map(|p| circle.iter().map(|c| point(c, p)).collect())
            .collect()
    } else {
        circle
            .iter()
            .map(|c| profile.iter().map(|p| point(c, p)).collect())
            .collect()
    };
    let (knots_u, knots_v, degree_u, degree_v) = if turn_along_v {
        (profile_knots, turn_knots, profile_degree, 2)
    } else {
        (turn_knots, profile_knots, 2, profile_degree)
    };
    NurbsSurface::new(
        degree_u,
        degree_v,
        knots_u,
        knots_v,
        grid.iter()
            .map(|row| row.iter().map(|&(p, _)| p).collect())
            .collect(),
        grid.iter()
            .map(|row| row.iter().map(|&(_, w)| w).collect())
            .collect(),
    )
    .unwrap()
}

/// A closed rational NURBS circle of radius `r` about the z axis at height
/// `z`, starting on +x.
fn nurbs_circle_edge(topo: &mut Topology, r: f64, z: f64) -> brepkit_topology::edge::EdgeId {
    let polygon = unit_circle_polygon(0.0);
    let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
        2,
        circle_knots(0.0, 1.0),
        polygon
            .iter()
            .map(|&(x, y, _)| Point3::new(r * x, r * y, z))
            .collect(),
        polygon.iter().map(|&(_, _, w)| w).collect(),
    )
    .unwrap();
    let vertex = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, z), 1e-7));
    topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::NurbsCurve(curve)))
}

/// Every triangle of `faces`'s groups as its three corners.
fn face_group_triangles(mesh: &TriangleMesh, offsets: &[u32], face: usize) -> Vec<[Point3; 3]> {
    mesh.indices[offsets[face] as usize..offsets[face + 1] as usize]
        .chunks_exact(3)
        .map(|t| [0, 1, 2].map(|k| mesh.positions[t[k] as usize]))
        .collect()
}

/// The largest distance from `surface_distance` of any corner or centroid of
/// `triangles`, and their total area.
fn deviation_and_area(
    triangles: &[[Point3; 3]],
    surface_distance: &dyn Fn(Point3) -> f64,
) -> (f64, f64) {
    let mut worst: f64 = 0.0;
    let mut area = 0.0;
    for &[a, b, c] in triangles {
        let centroid = Point3::new(
            (a.x() + b.x() + c.x()) / 3.0,
            (a.y() + b.y() + c.y()) / 3.0,
            (a.z() + b.z() + c.z()) / 3.0,
        );
        for p in [a, b, c, centroid] {
            worst = worst.max(surface_distance(p));
        }
        area += 0.5 * (b - a).cross(c - a).length();
    }
    (worst, area)
}

/// The `(radius, z, weight)` profile of a hemisphere of radius `r`.
fn hemisphere_profile(r: f64) -> [(f64, f64, f64); 3] {
    [
        (r, 0.0, 1.0),
        (r, r, std::f64::consts::FRAC_1_SQRT_2),
        (0.0, r, 1.0),
    ]
}

/// A dome over a planar floor: the NURBS surface revolving `profile` (from
/// the rim at z = 0 up to the axis) about z, its face bounded by the rim
/// alone, or, with `seam`, by the rim and the meridian run up to the pole
/// and back, as files carry it, the pole's vertex `seam` sideways off it.
fn nurbs_dome(
    profile: &[(f64, f64, f64)],
    knots: Vec<f64>,
    seam: Option<f64>,
) -> (Topology, brepkit_topology::solid::SolidId) {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let mut topo = Topology::new();
    let rim = nurbs_circle_edge(&mut topo, profile[0].0, 0.0);
    let mut dome_edges = vec![OrientedEdge::new(rim, true)];
    if let Some(off_pole) = seam {
        let rim_vertex = topo.edge(rim).unwrap().start();
        let top = profile[profile.len() - 1];
        let pole = topo.add_vertex(Vertex::new(Point3::new(0.0, off_pole, top.1), 1e-7));
        let meridian = brepkit_math::nurbs::curve::NurbsCurve::new(
            knots.len() - profile.len() - 1,
            knots.clone(),
            profile
                .iter()
                .map(|&(r, z, _)| Point3::new(r, 0.0, z))
                .collect(),
            profile.iter().map(|&(_, _, w)| w).collect(),
        )
        .unwrap();
        let seam = topo.add_edge(Edge::new(rim_vertex, pole, EdgeCurve::NurbsCurve(meridian)));
        dome_edges.extend([
            OrientedEdge::new(seam, true),
            OrientedEdge::new(seam, false),
        ]);
    }
    let dome = revolved_nurbs(profile, knots, false, (0.0, 1.0));
    let dome_wire = topo.add_wire(Wire::new(dome_edges, true).unwrap());
    let base_wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, false)], true).unwrap());
    let dome_face = topo.add_face(Face::new(dome_wire, vec![], FaceSurface::Nurbs(dome)));
    let base_face = topo.add_face(Face::new(
        base_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![base_face, dome_face]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));
    (topo, solid)
}

/// A rational NURBS hemisphere over a planar disc, its pole a collapsed v
/// edge with no boundary on it. Its pole closure runs a virtual seam up a
/// meridian; unsampled, the CDT spanned that meridian with one chord and
/// left two fins through the dome's inside, 3.9 off the sphere.
#[test]
fn a_nurbs_dome_meshes_its_pole_closure_on_the_sphere() {
    let r = 20.0;
    let (topo, solid) = nurbs_dome(
        &hemisphere_profile(r),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        None,
    );
    let truth = 2.0 * std::f64::consts::PI * r * r;
    for deflection in [0.5, 0.1] {
        let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
            &topo,
            solid,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "deflection {deflection}");
        let (worst, area) = deviation_and_area(&face_group_triangles(&mesh, &offsets, 1), &|p| {
            ((p - Point3::new(0.0, 0.0, 0.0)).length() - r).abs()
        });
        assert!(
            worst <= deflection,
            "deflection {deflection}: dome mesh {worst} off the sphere"
        );
        assert!(
            (area - truth).abs() <= 0.01 * truth,
            "deflection {deflection}: dome mesh area {area}, truth {truth}"
        );
    }
}

/// A NURBS hemisphere written with a seam, its face bounded by the rim and a
/// meridian run up to the pole and back, the pole's vertex where it falls
/// and one a file put 1e-5 off it. The seam's two copies meet at the pole,
/// so they are one run of seam samples; given one u, the run folded onto a
/// line and the face fell to the snap mesher, which meshes the surface's
/// whole domain and welds it to the rim by proximity.
#[test]
fn a_seamed_nurbs_dome_meshes_through_its_pole() {
    let r = 20.0;
    let area = 2.0 * std::f64::consts::PI * r * r;
    let volume = 2.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    for off_pole in [0.0, 1e-5] {
        let (topo, solid) = nurbs_dome(
            &hemisphere_profile(r),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            Some(off_pole),
        );
        for deflection in [0.5, 0.1] {
            let case = format!("pole vertex {off_pole} off, deflection {deflection}");
            let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
                &topo,
                solid,
                deflection,
                brepkit_math::chord::DEFAULT_ANGULAR_TOL,
            )
            .unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{case}");
            let triangles = face_group_triangles(&mesh, &offsets, 1);
            let (worst, meshed) = deviation_and_area(&triangles, &|p| {
                ((p - Point3::new(0.0, 0.0, 0.0)).length() - r).abs()
            });
            assert!(
                worst <= deflection,
                "{case}: dome mesh {worst} off the sphere"
            );
            assert!(
                (meshed - area).abs() <= 0.01 * area,
                "{case}: dome mesh area {meshed}, truth {area}"
            );
            let enclosed = signed_volume_raw(&mesh);
            assert!(
                (enclosed - volume).abs() <= 0.02 * volume,
                "{case}: mesh volume {enclosed}, truth {volume}"
            );
        }
    }
}

/// A dense polyline along the planar cubic Bezier `(radius, z)` profile.
fn cubic_profile_polyline(profile: [(f64, f64); 4]) -> Vec<(f64, f64)> {
    const SAMPLES: u32 = 4000;
    (0..=SAMPLES)
        .map(|k| {
            let t = f64::from(k) / f64::from(SAMPLES);
            let s = 1.0 - t;
            let b = [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t];
            (0..4).fold((0.0, 0.0), |(r, z), i| {
                (r + b[i] * profile[i].0, z + b[i] * profile[i].1)
            })
        })
        .collect()
}

/// The distance from `p` to the surface revolving the profile `polyline`
/// about the z axis: the nearest of a coarse pass over its vertices, then
/// every segment beside it.
fn distance_to_revolved_polyline(polyline: &[(f64, f64)], p: Point3) -> f64 {
    const STRIDE: usize = 20;
    let q = (p.x().hypot(p.y()), p.z());
    let to_segment = |a: (f64, f64), b: (f64, f64)| {
        let (dr, dz) = (b.0 - a.0, b.1 - a.1);
        let len2 = dr.mul_add(dr, dz * dz);
        let f = if len2 > 0.0 {
            ((q.0 - a.0).mul_add(dr, (q.1 - a.1) * dz) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (q.0 - f.mul_add(dr, a.0)).hypot(q.1 - f.mul_add(dz, a.1))
    };
    let near = (0..polyline.len())
        .step_by(STRIDE)
        .min_by(|&i, &j| {
            let d = |k: usize| (polyline[k].0 - q.0).hypot(polyline[k].1 - q.1);
            d(i).total_cmp(&d(j))
        })
        .unwrap_or(0);
    let lo = near.saturating_sub(2 * STRIDE);
    let hi = (near + 2 * STRIDE).min(polyline.len() - 1);
    (lo..hi)
        .map(|k| to_segment(polyline[k], polyline[k + 1]))
        .fold(f64::INFINITY, f64::min)
}

/// A NURBS dome whose cubic profile is inflected so that its middle lies on
/// the chord from rim to pole while its quarter points stand 0.63 off it.
/// Read only at its middle, the pole closure's meridian looked straight,
/// took no samples, and the triangles beside it cut through the dome.
#[test]
fn a_nurbs_dome_samples_an_inflected_pole_closure() {
    let profile = [(4.0, 0.0), (4.0, 6.0), (0.0, 4.0), (0.0, 10.0)];
    let weighted: Vec<(f64, f64, f64)> = profile.iter().map(|&(r, z)| (r, z, 1.0)).collect();
    let (topo, solid) = nurbs_dome(
        &weighted,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        None,
    );
    let deflection = 0.01;
    let (mesh, offsets) =
        tessellate_solid_grouped_with_tolerance(&topo, solid, deflection, 0.5).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0);
    let polyline = cubic_profile_polyline(profile);
    let (worst, _) = deviation_and_area(&face_group_triangles(&mesh, &offsets, 1), &|p| {
        distance_to_revolved_polyline(&polyline, p)
    });
    assert!(worst <= deflection, "dome mesh {worst} off its surface");
}

/// A rational NURBS half torus between two rim circles, joined by the mesher
/// along a virtual seam across the tube. Spaced like the rims (four times the
/// tube's radius), the seam's samples cut chords half a millimetre inside
/// the tube at any deflection.
#[test]
fn a_nurbs_torus_band_meshes_its_joining_seam_within_deflection() {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use std::f64::consts::FRAC_1_SQRT_2;
    let (major, minor) = (20.0, 5.0);
    let mut topo = Topology::new();
    let outer = nurbs_circle_edge(&mut topo, major + minor, 0.0);
    let inner = nurbs_circle_edge(&mut topo, major - minor, 0.0);
    let band = revolved_nurbs(
        &[
            (major + minor, 0.0, 1.0),
            (major + minor, minor, FRAC_1_SQRT_2),
            (major, minor, 1.0),
            (major - minor, minor, FRAC_1_SQRT_2),
            (major - minor, 0.0, 1.0),
        ],
        vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
        false,
        (0.0, 1.0),
    );
    let wire = |topo: &mut Topology, edge, forward| {
        topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, forward)], true).unwrap())
    };
    let band_outer = wire(&mut topo, outer, true);
    let band_inner = wire(&mut topo, inner, false);
    let base_outer = wire(&mut topo, outer, false);
    let base_inner = wire(&mut topo, inner, true);
    let band_face = topo.add_face(Face::new(
        band_outer,
        vec![band_inner],
        FaceSurface::Nurbs(band),
    ));
    let base_face = topo.add_face(Face::new(
        base_outer,
        vec![base_inner],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![base_face, band_face]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));
    let truth = 2.0 * std::f64::consts::PI.powi(2) * major * minor;
    for deflection in [0.5, 0.1, 0.05] {
        let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
            &topo,
            solid,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "deflection {deflection}");
        let (worst, area) = deviation_and_area(&face_group_triangles(&mesh, &offsets, 1), &|p| {
            let ring = p.x().hypot(p.y()) - major;
            (ring.hypot(p.z()) - minor).abs()
        });
        assert!(
            worst <= deflection,
            "deflection {deflection}: band mesh {worst} off the torus"
        );
        assert!(
            (area - truth).abs() <= 0.01 * truth,
            "deflection {deflection}: band mesh area {area}, truth {truth}"
        );
    }
}

/// The radii of [`fillet_post`]'s post, fillet and top.
const FILLET_POST_RADII: (f64, f64, f64) = (0.3, 0.25, 0.05);
/// The heights of [`fillet_post`]'s base, shoulder and top.
const FILLET_POST_HEIGHTS: (f64, f64, f64) = (1.7, 2.75, 3.0);

/// A post rounded over by a revolved NURBS fillet whose profile runs along u
/// and whose turn closes along v over `turn`, seamed by its profile arc, up
/// to a small flat top: an inch-unit part's pin.
fn fillet_post(turn: (f64, f64)) -> (Topology, brepkit_topology::solid::SolidId) {
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};
    let (post, fillet, top) = FILLET_POST_RADII;
    let (z_base, z_shoulder, z_top) = FILLET_POST_HEIGHTS;
    let mut topo = Topology::new();
    let vertex = |topo: &mut Topology, x: f64, z: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, 0.0, z), 1e-7))
    };
    let (v_base, v_shoulder, v_top) = (
        vertex(&mut topo, -post, z_base),
        vertex(&mut topo, -post, z_shoulder),
        vertex(&mut topo, -top, z_top),
    );
    // The rims' parametric origins lie off the seam, so their samples
    // straddle its vertices by half a step.
    let rim = |z: f64, r: f64, normal_z: f64| {
        Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, z),
            Vec3::new(0.0, 0.0, normal_z),
            r,
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap()
    };
    let e_base = topo.add_edge(Edge::new(
        v_base,
        v_base,
        EdgeCurve::Circle(rim(z_base, post, -1.0)),
    ));
    let e_shoulder = topo.add_edge(Edge::new(
        v_shoulder,
        v_shoulder,
        EdgeCurve::Circle(rim(z_shoulder, post, 1.0)),
    ));
    let e_top = topo.add_edge(Edge::new(
        v_top,
        v_top,
        EdgeCurve::Circle(rim(z_top, top, -1.0)),
    ));
    let e_wall_seam = topo.add_edge(Edge::new(v_shoulder, v_base, EdgeCurve::Line));
    let e_fillet_seam = topo.add_edge(Edge::new(
        v_top,
        v_shoulder,
        EdgeCurve::Circle(
            Circle3D::new(
                Point3::new(-top, 0.0, z_shoulder),
                Vec3::new(0.0, -1.0, 0.0),
                fillet,
            )
            .unwrap(),
        ),
    ));
    let mut wire = |edges: &[(brepkit_topology::edge::EdgeId, bool)]| {
        topo.add_wire(
            Wire::new(
                edges
                    .iter()
                    .map(|&(e, forward)| OrientedEdge::new(e, forward))
                    .collect(),
                true,
            )
            .unwrap(),
        )
    };
    let wall_wire = wire(&[
        (e_shoulder, false),
        (e_wall_seam, true),
        (e_base, false),
        (e_wall_seam, false),
    ]);
    let fillet_wire = wire(&[
        (e_fillet_seam, true),
        (e_shoulder, false),
        (e_fillet_seam, false),
        (e_top, false),
    ]);
    let top_wire = wire(&[(e_top, false)]);
    let base_wire = wire(&[(e_base, true)]);
    let fillet_surface = revolved_nurbs(
        &[
            (post, z_shoulder, 1.0),
            (post, z_top, FRAC_1_SQRT_2),
            (top, z_top, 1.0),
        ],
        vec![0.0, 0.0, 0.0, FRAC_PI_2, FRAC_PI_2, FRAC_PI_2],
        true,
        turn,
    );
    let wall = topo.add_face(Face::new(
        wall_wire,
        vec![],
        FaceSurface::Cylinder(
            brepkit_math::surfaces::CylindricalSurface::new(
                Point3::new(0.0, 0.0, z_base),
                Vec3::new(0.0, 0.0, 1.0),
                post,
            )
            .unwrap(),
        ),
    ));
    let fillet_face = topo.add_face(Face::new_reversed(
        fillet_wire,
        vec![],
        FaceSurface::Nurbs(fillet_surface),
    ));
    let top_face = topo.add_face(Face::new(
        top_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z_top,
        },
    ));
    let base_face = topo.add_face(Face::new(
        base_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -z_base,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![wall, fillet_face, top_face, base_face]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));
    (topo, solid)
}

/// [`fillet_post`] with its fillet's turn over `[-pi, pi]`, as files write
/// it, and over `[0, 1]`, whose period is not a full turn's. The CDT read
/// the seam as running along v, folded the fillet over itself and left most
/// of it, and the top disc the weld then collapsed, unmeshed.
#[test]
fn a_fillet_closed_along_v_meshes_with_its_top() {
    use std::f64::consts::{FRAC_PI_2, PI};
    let (post, fillet, top) = FILLET_POST_RADII;
    let (z_base, z_shoulder, _) = FILLET_POST_HEIGHTS;
    // Pappus over the profile: the wall's cylinder, the top's disc swept
    // down to the shoulder, and the quarter disc the fillet bounds.
    let quarter = PI * fillet * fillet / 4.0;
    let volume = PI * post * post * (z_shoulder - z_base)
        + PI * top * top * fillet
        + 2.0 * PI * (top + 4.0 * fillet / (3.0 * PI)) * quarter;
    let fillet_area = 2.0 * PI * fillet * (top * FRAC_PI_2 + fillet);
    let top_area = PI * top * top;
    for turn in [(-PI, PI), (0.0, 1.0)] {
        let (topo, solid) = fillet_post(turn);
        for deflection in [0.5, 0.05, 0.0125] {
            let case = format!("turn {turn:?}, deflection {deflection}");
            let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
                &topo,
                solid,
                deflection,
                brepkit_math::chord::DEFAULT_ANGULAR_TOL,
            )
            .unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{case}");
            let (worst, area) =
                deviation_and_area(&face_group_triangles(&mesh, &offsets, 1), &|p| {
                    let ring = p.x().hypot(p.y()) - top;
                    (ring.hypot(p.z() - z_shoulder) - fillet).abs()
                });
            assert!(
                worst <= deflection.min(0.01),
                "{case}: fillet mesh {worst} off the surface"
            );
            assert!(
                area >= 0.98 * fillet_area && area <= fillet_area,
                "{case}: fillet mesh area {area}, truth {fillet_area}"
            );
            let (_, cap) = deviation_and_area(&face_group_triangles(&mesh, &offsets, 2), &|_| 0.0);
            assert!(
                cap >= 0.97 * top_area && cap <= top_area,
                "{case}: top mesh area {cap}, truth {top_area}"
            );
            let meshed = signed_volume_raw(&mesh);
            assert!(
                (meshed - volume).abs() <= 0.03 * volume,
                "{case}: mesh volume {meshed}, truth {volume}"
            );
        }
    }
}

/// A hemisphere written with a seam: its one face bounded by the equator
/// and a meridian run up to the pole and back, as files carry sphere caps.
/// The seam's two copies meet at the pole, so they are one run of seam
/// samples; given one u, the run folded onto a line, the cap bounded nothing
/// in (u, v) and the face fell back to a grid over the other hemisphere.
#[test]
fn a_seamed_sphere_cap_meshes_its_own_hemisphere() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let r = 5.0;
    let origin = Point3::new(0.0, 0.0, 0.0);
    let (x, y, z) = (
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let mut topo = Topology::new();
    let on_equator = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, 0.0), 1e-7));
    let pole = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, r), 1e-7));
    let equator = topo.add_edge(Edge::new(
        on_equator,
        on_equator,
        EdgeCurve::Circle(Circle3D::new_with_ref(origin, z, r, x).unwrap()),
    ));
    let seam = topo.add_edge(Edge::new(
        on_equator,
        pole,
        EdgeCurve::Circle(Circle3D::with_axes(origin, -y, r, x, z).unwrap()),
    ));
    let cap_wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(equator, true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .unwrap(),
    );
    let floor_wire =
        topo.add_wire(Wire::new(vec![OrientedEdge::new(equator, false)], true).unwrap());
    let sphere = SphericalSurface::with_axis_and_ref_dir(origin, r, z, x).unwrap();
    let cap = topo.add_face(Face::new(cap_wire, vec![], FaceSurface::Sphere(sphere)));
    let floor = topo.add_face(Face::new(
        floor_wire,
        vec![],
        FaceSurface::Plane { normal: -z, d: 0.0 },
    ));
    let shell = topo.add_shell(Shell::new(vec![floor, cap]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));
    let area = 2.0 * std::f64::consts::PI * r * r;
    let volume = 2.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    for deflection in [0.1, 0.01] {
        let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
            &topo,
            solid,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "deflection {deflection}");
        let triangles = face_group_triangles(&mesh, &offsets, 1);
        let (worst, meshed) =
            deviation_and_area(&triangles, &|p| ((p - origin).length() - r).abs());
        assert!(
            triangles.iter().flatten().all(|p| p.z() >= -1e-9),
            "deflection {deflection}: the cap reaches below its equator"
        );
        assert!(
            worst <= deflection,
            "deflection {deflection}: cap mesh {worst} off the sphere"
        );
        assert!(
            (meshed - area).abs() <= 0.01 * area,
            "deflection {deflection}: cap mesh area {meshed}, truth {area}"
        );
        let enclosed = signed_volume_raw(&mesh);
        assert!(
            (enclosed - volume).abs() <= 0.01 * volume,
            "deflection {deflection}: mesh volume {enclosed}, truth {volume}"
        );
    }
}

/// A cylinder whose wall is a rational NURBS surface of revolution with its
/// straight profile along u and its turn closing along v, bounded by two rim
/// circles and no seam: once with the surface spanning exactly the wall, and
/// once running past both rims. The mesher read the rims as winding along v,
/// which it never joins; it left the first open and covered the second's
/// whole surface.
#[test]
fn a_nurbs_wall_closed_along_v_meshes_between_its_rims() {
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let (r, height) = (4.0, 10.0);
    let z = Vec3::new(0.0, 0.0, 1.0);
    for (z_lo, z_hi) in [(0.0, height), (-2.0, height + 2.0)] {
        let circle = unit_circle_polygon(0.0);
        let row = |level: f64| -> Vec<(Point3, f64)> {
            circle
                .iter()
                .map(|&(cx, cy, w)| (Point3::new(r * cx, r * cy, level), w))
                .collect()
        };
        let grid = [row(z_lo), row(z_hi)];
        let wall_surface = NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            circle_knots(0.0, 1.0),
            grid.iter()
                .map(|row| row.iter().map(|&(p, _)| p).collect())
                .collect(),
            grid.iter()
                .map(|row| row.iter().map(|&(_, w)| w).collect())
                .collect(),
        )
        .unwrap();
        let mut topo = Topology::new();
        let mut rim = |level: f64| {
            let vertex = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, level), 1e-7));
            let circle = Circle3D::new_with_ref(
                Point3::new(0.0, 0.0, level),
                z,
                r,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap();
            topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Circle(circle)))
        };
        let (bottom, top) = (rim(0.0), rim(height));
        let mut wire = |edge, forward| {
            topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, forward)], true).unwrap())
        };
        let (wall_outer, wall_inner, floor_wire, roof_wire) = (
            wire(bottom, true),
            wire(top, false),
            wire(bottom, false),
            wire(top, true),
        );
        let wall = topo.add_face(Face::new_reversed(
            wall_outer,
            vec![wall_inner],
            FaceSurface::Nurbs(wall_surface),
        ));
        let floor = topo.add_face(Face::new(
            floor_wire,
            vec![],
            FaceSurface::Plane { normal: -z, d: 0.0 },
        ));
        let roof = topo.add_face(Face::new(
            roof_wire,
            vec![],
            FaceSurface::Plane {
                normal: z,
                d: height,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![floor, wall, roof]).unwrap());
        let solid = topo.add_solid(Solid::new(shell, vec![]));
        let area = 2.0 * std::f64::consts::PI * r * height;
        let volume = std::f64::consts::PI * r * r * height;
        for deflection in [0.1, 0.01] {
            let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
                &topo,
                solid,
                deflection,
                brepkit_math::chord::DEFAULT_ANGULAR_TOL,
            )
            .unwrap();
            let case = format!("profile {z_lo}..{z_hi}, deflection {deflection}");
            assert_eq!(boundary_edge_count(&mesh), 0, "{case}");
            let triangles = face_group_triangles(&mesh, &offsets, 1);
            let (worst, meshed) =
                deviation_and_area(&triangles, &|p| (p.x().hypot(p.y()) - r).abs());
            assert!(
                triangles
                    .iter()
                    .flatten()
                    .all(|p| p.z() >= -1e-9 && p.z() <= height + 1e-9),
                "{case}: the wall runs past its rims"
            );
            assert!(
                worst <= deflection,
                "{case}: wall mesh {worst} off the cylinder"
            );
            assert!(
                (meshed - area).abs() <= 0.02 * area,
                "{case}: wall mesh area {meshed}, truth {area}"
            );
            let enclosed = signed_volume_raw(&mesh);
            assert!(
                (enclosed - volume).abs() <= 0.03 * volume,
                "{case}: mesh volume {enclosed}, truth {volume}"
            );
        }
    }
}

/// Two triangular prisms stacked into one solid that keeps both copies of
/// the triangle they share, as a boolean can leave them, plus a sliver face
/// whose wire runs one edge there and back. The coincident pair of opposite
/// sense cancels to nothing by design and is no hole; the sliver bounds
/// nothing and meshes to nothing, which the mesher reports.
#[test]
fn only_a_face_left_genuinely_empty_is_reported_unmeshed() {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let mut topo = Topology::new();
    let triangle = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
    let lower = build_prism(&mut topo, &triangle, 0.0, 1.0);
    let upper = build_prism(&mut topo, &triangle, 1.0, 2.0);
    let mut faces = brepkit_topology::explorer::solid_faces(&topo, lower).unwrap();
    faces.extend(brepkit_topology::explorer::solid_faces(&topo, upper).unwrap());
    let shared: Vec<_> = faces
        .iter()
        .copied()
        .filter(|&f| {
            matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                if normal.x().abs() < 1e-12 && normal.y().abs() < 1e-12 && (d.abs() - 1.0).abs() < 1e-12)
        })
        .collect();
    assert_eq!(shared.len(), 2, "the prisms share one triangle, kept twice");
    let (a, b) = (
        topo.add_vertex(Vertex::new(Point3::new(3.0, 0.0, 0.0), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(4.0, 0.0, 0.0), 1e-7)),
    );
    let there = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
    let sliver_wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(there, true),
                OrientedEdge::new(there, false),
            ],
            true,
        )
        .unwrap(),
    );
    let sliver = topo.add_face(Face::new(
        sliver_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    faces.push(sliver);
    let shell = topo.add_shell(Shell::new(faces).unwrap());
    let solid = topo.add_solid(Solid::new(shell, vec![]));
    let core = super::solid::tessellate_solid_core(
        &topo,
        solid,
        0.1,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        true,
        false,
    )
    .unwrap();
    let tri_faces = core.tri_faces.unwrap();
    let all = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
    for &face in &shared {
        let index = all.iter().position(|&f| f == face).unwrap();
        assert!(
            !tri_faces.iter().any(|&f| f as usize == index),
            "the coincident pair cancels in the mesh"
        );
    }
    assert_eq!(core.unmeshed, vec![sliver]);
}
