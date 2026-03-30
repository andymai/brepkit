//! Tests for tessellation.

#![allow(clippy::unwrap_used, deprecated)]

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

    let mesh = tessellate_nurbs(&surface, 0.1).mesh;

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
    let flat_mesh = tessellate_nurbs(&flat, deflection).mesh;
    let curved_mesh = tessellate_nurbs(&curved, deflection).mesh;

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

#[test]
#[ignore = "GFA pipeline limitation -- old boolean pipeline removed"]
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

    let fine_mesh = tessellate_nurbs(&dome, 0.01).mesh;
    let coarse_mesh = tessellate_nurbs(&dome, 0.5).mesh;

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
    let mesh = tessellate_nurbs(&surface, deflection).mesh;

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

    let all_edges = sample_solid_edges_filtered(&topo, solid, 0.1, false).unwrap();
    assert!(
        all_edges.offsets.len() >= 3,
        "unfiltered cylinder should have at least 3 edges, got {}",
        all_edges.offsets.len()
    );
}

#[test]
#[ignore = "GFA pipeline limitation -- old boolean pipeline removed"]
fn sample_solid_edges_boolean_filters_coplanar() {
    let mut topo = Topology::new();
    let big = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let small = crate::primitives::make_box(&mut topo, 3.0, 3.0, 15.0).unwrap();
    let opts = crate::boolean::BooleanOptions {
        unify_faces: false,
        ..Default::default()
    };
    let cut = crate::boolean::boolean_with_options(
        &mut topo,
        crate::boolean::BooleanOp::Cut,
        big,
        small,
        opts,
    )
    .unwrap();

    let filtered = sample_solid_edges(&topo, cut, 0.1).unwrap();
    let all = sample_solid_edges_filtered(&topo, cut, 0.1, false).unwrap();

    assert!(
        filtered.offsets.len() < all.offsets.len(),
        "filtered ({}) should be fewer than unfiltered ({})",
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
    use std::collections::{HashMap, HashSet};

    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();
    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
    let snap_pt = |p: Point3| -> (i64, i64, i64) { (snap(p.x()), snap(p.y()), snap(p.z())) };

    let mut pos_map: HashMap<(i64, i64, i64), usize> = HashMap::new();
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
    let mut half_edges: HashSet<(usize, usize)> = HashSet::new();
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
    let unique: std::collections::HashSet<(i64, i64, i64)> = mesh
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
#[ignore = "GFA pipeline limitation -- old boolean pipeline removed"]
fn test_tessellate_boolean_result_watertight() {
    use std::collections::{HashMap, HashSet};

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

    let mut pos_map: HashMap<(i64, i64, i64), usize> = HashMap::new();
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
    let mut half_edges: HashSet<(usize, usize)> = HashSet::new();
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
        if let FaceSurface::Plane { normal, .. } = face.surface() {
            if mesh.indices.len() >= 3 {
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
