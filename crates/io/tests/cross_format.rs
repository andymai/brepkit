//! Cross-format IO round-trip tests.
//!
//! Verifies that geometric properties (bounding box, triangle count,
//! face count, volume) are preserved through serialization/deserialization.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::vec::Point3;
use brepkit_operations::measure;
use brepkit_operations::primitives;
use brepkit_topology::Topology;
use brepkit_topology::explorer;

// ── Helpers ────────────────────────────────────────────────────────────

/// Compute the axis-aligned bounding box of a `TriangleMesh`.
fn mesh_aabb(mesh: &brepkit_operations::tessellate::TriangleMesh) -> (Point3, Point3) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in &mesh.positions {
        let coords = [p.x(), p.y(), p.z()];
        for i in 0..3 {
            if coords[i] < min[i] {
                min[i] = coords[i];
            }
            if coords[i] > max[i] {
                max[i] = coords[i];
            }
        }
    }
    (
        Point3::new(min[0], min[1], min[2]),
        Point3::new(max[0], max[1], max[2]),
    )
}

/// Assert two points are approximately equal (within `tol`).
fn assert_point_approx(a: Point3, b: Point3, tol: f64, label: &str) {
    assert!(
        (a.x() - b.x()).abs() < tol && (a.y() - b.y()).abs() < tol && (a.z() - b.z()).abs() < tol,
        "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4}), tol={tol}",
        b.x(),
        b.y(),
        b.z(),
        a.x(),
        a.y(),
        a.z(),
    );
}

const DEFLECTION: f64 = 0.01;

// ── STL round-trip ─────────────────────────────────────────────────────

#[test]
fn stl_binary_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    assert!(!bytes.is_empty(), "STL binary output should not be empty");

    let mesh = brepkit_io::stl::read_stl(&bytes).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count >= 12,
        "box should have at least 12 triangles, got {tri_count}"
    );

    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "STL box min");
    assert_point_approx(max, Point3::new(2.0, 3.0, 4.0), 1e-6, "STL box max");
}

#[test]
fn stl_ascii_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();

    let bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Ascii,
    )
    .unwrap();
    assert!(!bytes.is_empty(), "STL ASCII output should not be empty");

    let mesh = brepkit_io::stl::read_stl(&bytes).unwrap();
    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "STL ASCII box min");
    assert_point_approx(max, Point3::new(1.0, 2.0, 3.0), 1e-6, "STL ASCII box max");
}

#[test]
fn stl_roundtrip_cylinder_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_cylinder(&mut topo, 1.0, 5.0).unwrap();

    let bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let mesh = brepkit_io::stl::read_stl(&bytes).unwrap();

    let (min, max) = mesh_aabb(&mesh);
    // Cylinder radius=1, height=5: AABB approximately [-1,-1,0] to [1,1,5].
    // Tessellation may be slightly inside the true circle.
    assert_point_approx(min, Point3::new(-1.0, -1.0, 0.0), 0.05, "STL cylinder min");
    assert_point_approx(max, Point3::new(1.0, 1.0, 5.0), 0.05, "STL cylinder max");
}

#[test]
fn stl_roundtrip_preserves_triangle_count() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Write and read back twice — triangle count should be stable.
    let bytes1 = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let mesh1 = brepkit_io::stl::read_stl(&bytes1).unwrap();

    let bytes2 = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let mesh2 = brepkit_io::stl::read_stl(&bytes2).unwrap();

    assert_eq!(
        mesh1.indices.len(),
        mesh2.indices.len(),
        "STL triangle count should be deterministic"
    );
}

// ── OBJ round-trip ─────────────────────────────────────────────────────

#[test]
fn obj_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 3.0, 2.0, 1.0).unwrap();

    let obj_str = brepkit_io::obj::write_obj(&topo, &[solid], DEFLECTION).unwrap();
    assert!(!obj_str.is_empty(), "OBJ output should not be empty");

    let mesh = brepkit_io::obj::read_obj(&obj_str).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count >= 12,
        "box should have at least 12 triangles, got {tri_count}"
    );

    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "OBJ box min");
    assert_point_approx(max, Point3::new(3.0, 2.0, 1.0), 1e-6, "OBJ box max");
}

#[test]
fn obj_roundtrip_cylinder_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_cylinder(&mut topo, 2.0, 3.0).unwrap();

    let obj_str = brepkit_io::obj::write_obj(&topo, &[solid], DEFLECTION).unwrap();
    let mesh = brepkit_io::obj::read_obj(&obj_str).unwrap();

    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(-2.0, -2.0, 0.0), 0.1, "OBJ cylinder min");
    assert_point_approx(max, Point3::new(2.0, 2.0, 3.0), 0.1, "OBJ cylinder max");
}

// ── PLY round-trip ─────────────────────────────────────────────────────

#[test]
fn ply_ascii_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 4.0, 5.0, 6.0).unwrap();

    let bytes = brepkit_io::ply::write_ply(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::ply::writer::PlyFormat::Ascii,
    )
    .unwrap();
    assert!(!bytes.is_empty(), "PLY ASCII output should not be empty");

    let mesh = brepkit_io::ply::read_ply(&bytes).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count >= 12,
        "PLY box should have at least 12 triangles, got {tri_count}"
    );

    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "PLY box min");
    assert_point_approx(max, Point3::new(4.0, 5.0, 6.0), 1e-6, "PLY box max");
}

#[test]
fn ply_binary_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let bytes = brepkit_io::ply::write_ply(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::ply::writer::PlyFormat::BinaryLittleEndian,
    )
    .unwrap();

    let mesh = brepkit_io::ply::read_ply(&bytes).unwrap();
    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "PLY binary box min");
    assert_point_approx(max, Point3::new(1.0, 1.0, 1.0), 1e-6, "PLY binary box max");
}

// ── glTF (GLB) round-trip ──────────────────────────────────────────────

#[test]
fn glb_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let bytes = brepkit_io::gltf::write_glb(&topo, &[solid], DEFLECTION).unwrap();
    assert!(!bytes.is_empty(), "GLB output should not be empty");

    let mesh = brepkit_io::gltf::read_glb(&bytes).unwrap();
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count >= 12,
        "GLB box should have at least 12 triangles, got {tri_count}"
    );

    let (min, max) = mesh_aabb(&mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "GLB box min");
    assert_point_approx(max, Point3::new(2.0, 2.0, 2.0), 1e-6, "GLB box max");
}

// ── 3MF round-trip ─────────────────────────────────────────────────────

#[test]
fn threemf_roundtrip_box_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.5, 2.5, 3.5).unwrap();

    let bytes = brepkit_io::threemf::write_threemf(&topo, &[solid], DEFLECTION).unwrap();
    assert!(!bytes.is_empty(), "3MF output should not be empty");

    let meshes = brepkit_io::threemf::read_threemf(&bytes).unwrap();
    assert!(!meshes.is_empty(), "3MF should contain at least one mesh");

    let mesh = &meshes[0];
    let tri_count = mesh.indices.len() / 3;
    assert!(
        tri_count >= 12,
        "3MF box should have at least 12 triangles, got {tri_count}"
    );

    let (min, max) = mesh_aabb(mesh);
    assert_point_approx(min, Point3::new(0.0, 0.0, 0.0), 1e-6, "3MF box min");
    assert_point_approx(max, Point3::new(1.5, 2.5, 3.5), 1e-6, "3MF box max");
}

// ── STEP round-trip ────────────────────────────────────────────────────

#[test]
fn step_roundtrip_box_face_count() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    assert!(
        step_str.contains("ISO-10303-21;"),
        "STEP output should have ISO header"
    );

    let mut read_topo = Topology::new();
    let solids = brepkit_io::step::reader::read_step(&step_str, &mut read_topo).unwrap();
    assert_eq!(solids.len(), 1, "should read back exactly one solid");

    let read_solid = read_topo.solid(solids[0]).unwrap();
    let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
    assert_eq!(
        shell.faces().len(),
        6,
        "box should have 6 faces after STEP round-trip"
    );
}

#[test]
fn step_roundtrip_box_volume() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let vol_before = measure::solid_volume(&topo, solid, DEFLECTION).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();

    let mut read_topo = Topology::new();
    let solids = brepkit_io::step::reader::read_step(&step_str, &mut read_topo).unwrap();

    let vol_after = measure::solid_volume(&read_topo, solids[0], DEFLECTION).unwrap();

    let rel_error = (vol_before - vol_after).abs() / vol_before;
    assert!(
        rel_error < 1e-6,
        "STEP round-trip volume: before={vol_before}, after={vol_after}, rel_error={rel_error}"
    );
}

#[test]
fn step_roundtrip_multiple_solids() {
    let mut topo = Topology::new();
    let s1 = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let s2 = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[s1, s2]).unwrap();

    let mut read_topo = Topology::new();
    let solids = brepkit_io::step::reader::read_step(&step_str, &mut read_topo).unwrap();
    assert_eq!(solids.len(), 2, "should read back two solids from STEP");
}

// ── IGES round-trip ────────────────────────────────────────────────────

#[test]
fn iges_roundtrip_box_face_count() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();

    let iges_str = brepkit_io::iges::writer::write_iges(&topo, &[solid]).unwrap();
    assert!(!iges_str.is_empty(), "IGES output should not be empty");

    let mut read_topo = Topology::new();
    let solids = brepkit_io::iges::reader::read_iges(&iges_str, &mut read_topo).unwrap();
    assert_eq!(solids.len(), 1, "should read back exactly one solid");

    let read_solid = read_topo.solid(solids[0]).unwrap();
    let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
    assert_eq!(
        shell.faces().len(),
        6,
        "box should have 6 faces after IGES round-trip"
    );
}

// ── STL mesh import round-trip ─────────────────────────────────────────

#[test]
fn stl_import_roundtrip_preserves_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Export to STL
    let bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();

    // Read back as mesh
    let mesh = brepkit_io::stl::read_stl(&bytes).unwrap();
    let (mesh_min, mesh_max) = mesh_aabb(&mesh);

    // Import mesh into B-Rep topology
    let mut import_topo = Topology::new();
    let imported = brepkit_io::stl::import_mesh(&mut import_topo, &mesh, 1e-6).unwrap();

    // Re-export the imported solid to STL and compare AABB
    let bytes2 = brepkit_io::stl::write_stl(
        &import_topo,
        &[imported],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let mesh2 = brepkit_io::stl::read_stl(&bytes2).unwrap();
    let (mesh2_min, mesh2_max) = mesh_aabb(&mesh2);

    assert_point_approx(mesh_min, mesh2_min, 1e-6, "import round-trip min");
    assert_point_approx(mesh_max, mesh2_max, 1e-6, "import round-trip max");
}

// ── Cross-format consistency ───────────────────────────────────────────

#[test]
fn mesh_formats_agree_on_triangle_count() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let stl_bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let stl_mesh = brepkit_io::stl::read_stl(&stl_bytes).unwrap();

    let obj_str = brepkit_io::obj::write_obj(&topo, &[solid], DEFLECTION).unwrap();
    let obj_mesh = brepkit_io::obj::read_obj(&obj_str).unwrap();

    let ply_bytes = brepkit_io::ply::write_ply(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::ply::writer::PlyFormat::Ascii,
    )
    .unwrap();
    let ply_mesh = brepkit_io::ply::read_ply(&ply_bytes).unwrap();

    let glb_bytes = brepkit_io::gltf::write_glb(&topo, &[solid], DEFLECTION).unwrap();
    let glb_mesh = brepkit_io::gltf::read_glb(&glb_bytes).unwrap();

    let stl_tris = stl_mesh.indices.len() / 3;
    let obj_tris = obj_mesh.indices.len() / 3;
    let ply_tris = ply_mesh.indices.len() / 3;
    let glb_tris = glb_mesh.indices.len() / 3;

    // All mesh formats should produce the same triangle count for the same input.
    assert_eq!(stl_tris, obj_tris, "STL vs OBJ triangle count mismatch");
    assert_eq!(stl_tris, ply_tris, "STL vs PLY triangle count mismatch");
    assert_eq!(stl_tris, glb_tris, "STL vs GLB triangle count mismatch");
}

#[test]
fn mesh_formats_agree_on_aabb() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 5.0, 7.0, 3.0).unwrap();

    let stl_bytes = brepkit_io::stl::write_stl(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::stl::writer::StlFormat::Binary,
    )
    .unwrap();
    let stl_mesh = brepkit_io::stl::read_stl(&stl_bytes).unwrap();
    let (stl_min, stl_max) = mesh_aabb(&stl_mesh);

    let obj_str = brepkit_io::obj::write_obj(&topo, &[solid], DEFLECTION).unwrap();
    let obj_mesh = brepkit_io::obj::read_obj(&obj_str).unwrap();
    let (obj_min, obj_max) = mesh_aabb(&obj_mesh);

    let ply_bytes = brepkit_io::ply::write_ply(
        &topo,
        &[solid],
        DEFLECTION,
        brepkit_io::ply::writer::PlyFormat::Ascii,
    )
    .unwrap();
    let ply_mesh = brepkit_io::ply::read_ply(&ply_bytes).unwrap();
    let (ply_min, ply_max) = mesh_aabb(&ply_mesh);

    assert_point_approx(stl_min, obj_min, 1e-6, "STL vs OBJ min");
    assert_point_approx(stl_max, obj_max, 1e-6, "STL vs OBJ max");
    assert_point_approx(stl_min, ply_min, 1e-6, "STL vs PLY min");
    assert_point_approx(stl_max, ply_max, 1e-6, "STL vs PLY max");
}

// ── STEP round-trip (geometry preservation) ──────────────────────────

#[test]
fn step_roundtrip_preserves_vertex_positions() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let orig_verts: Vec<_> = explorer::solid_vertices(&topo, solid)
        .unwrap()
        .iter()
        .map(|vid| topo.vertex(*vid).unwrap().point())
        .collect();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let reimport_verts: Vec<_> = explorer::solid_vertices(&topo2, solids2[0])
        .unwrap()
        .iter()
        .map(|vid| topo2.vertex(*vid).unwrap().point())
        .collect();

    assert_eq!(
        orig_verts.len(),
        reimport_verts.len(),
        "vertex count should match"
    );

    for orig in &orig_verts {
        let has_match = reimport_verts.iter().any(|reimp| {
            let dx = orig.x() - reimp.x();
            let dy = orig.y() - reimp.y();
            let dz = orig.z() - reimp.z();
            (dx * dx + dy * dy + dz * dz).sqrt() < 1e-6
        });
        assert!(
            has_match,
            "vertex ({:.3}, {:.3}, {:.3}) not found in reimport",
            orig.x(),
            orig.y(),
            orig.z()
        );
    }
}

#[test]
fn step_roundtrip_cylinder_surface_type() {
    let mut topo = Topology::new();
    let solid = primitives::make_cylinder(&mut topo, 1.5, 3.0).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let faces = explorer::solid_faces(&topo2, solids2[0]).unwrap();
    let has_cylinder = faces.iter().any(|fid| {
        let face = topo2.face(*fid).unwrap();
        matches!(
            face.surface(),
            brepkit_topology::face::FaceSurface::Cylinder(_)
        )
    });
    assert!(
        has_cylinder,
        "cylinder surface type should survive STEP round-trip"
    );
}

#[test]
fn step_roundtrip_nurbs_edge_survives() {
    let mut topo = Topology::new();
    let solid = primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let edges_before = explorer::solid_edges(&topo, solid).unwrap();
    let circle_count_before = edges_before
        .iter()
        .filter(|eid| {
            let edge = topo.edge(**eid).unwrap();
            matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
        })
        .count();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let edges_after = explorer::solid_edges(&topo2, solids2[0]).unwrap();
    let circle_count_after = edges_after
        .iter()
        .filter(|eid| {
            let edge = topo2.edge(**eid).unwrap();
            matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
        })
        .count();

    assert_eq!(
        circle_count_before, circle_count_after,
        "circle edges should survive STEP round-trip: before={circle_count_before}, after={circle_count_after}"
    );
}

#[test]
fn step_output_has_valid_syntax() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();

    assert!(
        step_str.contains("ISO-10303-21"),
        "should contain ISO header"
    );
    assert!(
        step_str.contains("HEADER;"),
        "should contain HEADER section"
    );
    assert!(step_str.contains("DATA;"), "should contain DATA section");
    assert!(
        step_str.contains("END-ISO-10303-21;"),
        "should contain END marker"
    );

    for line in step_str.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') && trimmed.contains('(') {
            let opens = trimmed.chars().filter(|c| *c == '(').count();
            let closes = trimmed.chars().filter(|c| *c == ')').count();
            assert_eq!(
                opens, closes,
                "unbalanced parentheses in STEP entity: {trimmed}"
            );
        }
    }
}

#[test]
fn step_roundtrip_center_of_mass() {
    let mut topo = Topology::new();
    let solid = primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let com_before = measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let com_after = measure::solid_center_of_mass(&topo2, solids2[0], 0.1).unwrap();

    let dx = com_before.x() - com_after.x();
    let dy = com_before.y() - com_after.y();
    let dz = com_before.z() - com_after.z();
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(
        dist < 1e-6,
        "CoM should survive STEP round-trip: before=({:.3},{:.3},{:.3}), after=({:.3},{:.3},{:.3}), dist={dist:.2e}",
        com_before.x(),
        com_before.y(),
        com_before.z(),
        com_after.x(),
        com_after.y(),
        com_after.z()
    );
}

#[test]
fn step_roundtrip_sphere() {
    let mut topo = Topology::new();
    let solid = primitives::make_sphere(&mut topo, 2.0, 32).unwrap();

    let vol_before = measure::solid_volume(&topo, solid, 0.05).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let vol_after = measure::solid_volume(&topo2, solids2[0], 0.05).unwrap();
    let rel_error = (vol_before - vol_after).abs() / vol_before;
    assert!(
        rel_error < 0.01,
        "sphere volume should survive STEP round-trip: before={vol_before:.3}, after={vol_after:.3} (error: {:.1}%)",
        rel_error * 100.0
    );
}

#[test]
fn step_roundtrip_cone() {
    let mut topo = Topology::new();
    let solid = primitives::make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[solid]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let faces_before = explorer::solid_faces(&topo, solid).unwrap().len();
    let faces_after = explorer::solid_faces(&topo2, solids2[0]).unwrap().len();
    assert_eq!(
        faces_before, faces_after,
        "cone face count should survive STEP round-trip"
    );
}

#[test]
#[ignore = "bug: STEP round-trip loses ~32% volume on boolean results with curved faces"]
fn step_roundtrip_boolean_result_volume() {
    let mut topo = Topology::new();
    let base = primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let cyl = primitives::make_cylinder(&mut topo, 1.0, 6.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        cyl,
        &brepkit_math::mat::Mat4::translation(2.0, 2.0, -1.0),
    )
    .unwrap();

    let cut = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        base,
        cyl,
    )
    .unwrap();

    let vol_before = measure::solid_volume(&topo, cut, 0.1).unwrap();

    let step_str = brepkit_io::step::write_step(&topo, &[cut]).unwrap();
    let mut topo2 = Topology::new();
    let solids2 = brepkit_io::step::reader::read_step(&step_str, &mut topo2).unwrap();

    let vol_after = measure::solid_volume(&topo2, solids2[0], 0.1).unwrap();
    let rel_error = (vol_before - vol_after).abs() / vol_before;
    assert!(
        rel_error < 0.05,
        "boolean result volume should survive STEP: before={vol_before:.3}, after={vol_after:.3} (error: {:.1}%)",
        rel_error * 100.0
    );
}
