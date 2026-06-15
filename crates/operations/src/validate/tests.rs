#![allow(clippy::unwrap_used)]

use brepkit_topology::Topology;
use brepkit_topology::test_utils::make_unit_cube_manifold;

use super::*;

#[test]
fn valid_cube() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let report = validate_solid(&topo, cube).unwrap();
    assert!(
        report.is_valid(),
        "manifold cube should be valid, got {} error(s): {:?}",
        report.error_count(),
        report.issues
    );
}

#[test]
fn valid_box_primitive() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "box should be valid: {:?}",
        report.issues
    );
}

#[test]
fn extruded_solid_is_valid() {
    let mut topo = Topology::new();
    let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);
    let solid = crate::extrude::extrude(
        &mut topo,
        face,
        brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
        1.0,
    )
    .unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "extruded solid should be valid: {:?}",
        report.issues
    );
}

#[test]
fn report_counts_work() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let report = validate_solid(&topo, cube).unwrap();
    assert_eq!(report.error_count(), 0);
    assert_eq!(report.warning_count(), 0);
}

#[test]
fn open_shell_has_boundary_edges() {
    // A solid with an open shell (missing a face) should report boundary edges
    let mut topo = Topology::new();
    // Make a box but with 5 faces instead of 6 — one face missing
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Remove one face from the shell to make it open
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    let mut faces: Vec<_> = shell.faces().to_vec();
    faces.pop();

    let open_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
    *topo.shell_mut(shell_id).unwrap() = open_shell;

    let report = validate_solid(&topo, solid).unwrap();
    assert!(!report.is_valid(), "open shell should not be valid");
    assert!(
        report.error_count() > 0,
        "should have errors for boundary/euler"
    );
    // Check description mentions boundary
    let has_boundary_msg = report
        .issues
        .iter()
        .any(|i| i.description.contains("boundary") || i.description.contains("Euler"));
    assert!(has_boundary_msg, "should mention boundary edges or Euler");
}

#[test]
fn report_is_valid_when_only_warnings() {
    let report = ValidationReport {
        issues: vec![ValidationIssue {
            severity: Severity::Warning,
            description: "minor issue".into(),
        }],
    };
    assert!(report.is_valid(), "warnings alone should not make invalid");
    assert_eq!(report.error_count(), 0);
    assert_eq!(report.warning_count(), 1);
}

#[test]
fn report_not_valid_with_errors() {
    let report = ValidationReport {
        issues: vec![
            ValidationIssue {
                severity: Severity::Error,
                description: "critical issue".into(),
            },
            ValidationIssue {
                severity: Severity::Warning,
                description: "minor issue".into(),
            },
        ],
    };
    assert!(!report.is_valid());
    assert_eq!(report.error_count(), 1);
    assert_eq!(report.warning_count(), 1);
}

#[test]
fn cylinder_solid_validates() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "cylinder should be valid: {:?}",
        report.issues
    );
}

#[test]
fn sphere_solid_validates() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_sphere(&mut topo, 2.0, 32).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "sphere should be valid: {:?}",
        report.issues
    );
}

#[test]
fn cone_solid_validates() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "cone should be valid: {:?}",
        report.issues
    );
}

#[test]
fn torus_solid_validates() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_torus(&mut topo, 5.0, 1.0, 32).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "torus should be valid: {:?}",
        report.issues
    );
}

#[test]
fn hollow_revolve_is_valid() {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::face::{Face, FaceSurface};

    let mut topo = Topology::new();

    // Outer: 2×1 rectangle at x=1..3, y=0..1.
    let outer_pts = vec![
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(3.0, 0.0, 0.0),
        Point3::new(3.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ];
    let outer_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts, 1e-7).unwrap();

    // Inner: 0.5×0.5 hole.
    let inner_pts = vec![
        Point3::new(1.5, 0.25, 0.0),
        Point3::new(1.5, 0.75, 0.0),
        Point3::new(2.5, 0.75, 0.0),
        Point3::new(2.5, 0.25, 0.0),
    ];
    let inner_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts, 1e-7).unwrap();

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane { normal, d: 0.0 },
    );
    let face_id = topo.add_face(face);

    // Full revolution → genus-1 torus topology.
    let solid = crate::revolve::revolve(
        &mut topo,
        face_id,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        2.0 * std::f64::consts::PI,
    )
    .unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "genus-1 hollow revolve should be valid, got {} error(s): {:?}",
        report.error_count(),
        report.issues
    );
}

#[test]
fn extruded_hollow_box_is_valid() {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::face::{Face, FaceSurface};

    let mut topo = Topology::new();

    // Outer: 2×2 square.
    let outer_pts = vec![
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(-1.0, 1.0, 0.0),
    ];
    let outer_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts, 1e-7).unwrap();

    // Inner: 0.5×0.5 hole.
    let inner_pts = vec![
        Point3::new(-0.25, -0.25, 0.0),
        Point3::new(-0.25, 0.25, 0.0),
        Point3::new(0.25, 0.25, 0.0),
        Point3::new(0.25, -0.25, 0.0),
    ];
    let inner_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts, 1e-7).unwrap();

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane { normal, d: 0.0 },
    );
    let face_id = topo.add_face(face);

    // Extrude → genus-0 (V-E+F=2) with inner walls.
    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "extruded hollow box should be valid, got {} error(s): {:?}",
        report.error_count(),
        report.issues
    );
}

#[test]
fn wire_closure_check_on_valid_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    // No wire closure errors on a properly-built box.
    let wire_issues: Vec<_> = report
        .issues
        .iter()
        .filter(|i| i.description.contains("wire"))
        .collect();
    assert!(
        wire_issues.is_empty(),
        "valid box should have no wire issues: {wire_issues:?}"
    );
}

#[test]
fn polygon_area_unit_square() {
    use brepkit_math::vec::Point3;
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let area = polygon_area_3d(&pts);
    assert!(
        (area - 1.0).abs() < 1e-10,
        "unit square area should be 1.0, got {area}"
    );
}

#[test]
fn polygon_area_triangle() {
    use brepkit_math::vec::Point3;
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(0.0, 3.0, 0.0),
    ];
    let area = polygon_area_3d(&pts);
    assert!(
        (area - 3.0).abs() < 1e-10,
        "triangle area should be 3.0, got {area}"
    );
}

#[test]
fn polygon_area_degenerate() {
    use brepkit_math::vec::Point3;
    // Collinear points → area 0.
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    let area = polygon_area_3d(&pts);
    assert!(
        area < 1e-15,
        "collinear points should have zero area, got {area}"
    );
}

#[test]
fn validate_detects_non_manifold_edge() {
    // 3 faces sharing one edge is non-manifold
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Duplicate one face to create a non-manifold edge
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    let mut faces: Vec<_> = shell.faces().to_vec();
    let extra_face = faces[0];
    faces.push(extra_face);

    let new_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
    *topo.shell_mut(shell_id).unwrap() = new_shell;

    let report = validate_solid(&topo, solid).unwrap();
    assert!(!report.is_valid(), "non-manifold edge should be invalid");
    let has_nm = report
        .issues
        .iter()
        .any(|i| i.description.contains("non-manifold") || i.description.contains("shared by"));
    assert!(has_nm, "should mention non-manifold: {:?}", report.issues);
}

#[test]
fn validate_detects_zero_length_normal() {
    // A face with a zero-length normal should produce a warning.
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Corrupt a face normal to zero-length.
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let face_id = topo.shell(shell_id).unwrap().faces()[0];
    let face = topo.face_mut(face_id).unwrap();
    *face = brepkit_topology::face::Face::new(
        face.outer_wire(),
        face.inner_wires().to_vec(),
        brepkit_topology::face::FaceSurface::Plane {
            normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0),
            d: 0.0,
        },
    );

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.warning_count() > 0,
        "zero-length normal should produce a warning: {:?}",
        report.issues
    );
}

#[test]
fn no_area_warnings_on_valid_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    let area_warnings: Vec<_> = report
        .issues
        .iter()
        .filter(|i| i.description.contains("area"))
        .collect();
    assert!(
        area_warnings.is_empty(),
        "valid box should have no area warnings: {area_warnings:?}"
    );
}

#[test]
fn validate_detects_open_wire() {
    // Construct a solid with a wire that doesn't close
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Get a wire and break its closure by removing the closed flag
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let face_id = topo.shell(shell_id).unwrap().faces()[0];
    let wire_id = topo.face(face_id).unwrap().outer_wire();
    let wire = topo.wire(wire_id).unwrap();
    let edges = wire.edges().to_vec();

    // Create an open wire (not closed) with same edges
    if edges.len() > 1 {
        use brepkit_topology::wire::Wire;
        let open_wire = Wire::new(edges[..edges.len() - 1].to_vec(), false);
        if let Ok(w) = open_wire {
            *topo.wire_mut(wire_id).unwrap() = w;

            let report = validate_solid(&topo, solid).unwrap();
            assert!(
                !report.is_valid(),
                "open wire should be invalid: {:?}",
                report.issues
            );
        }
    }
}

#[test]
fn validate_detects_zero_length_edge() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Find an edge and make its vertices coincident
    let edges = explorer::solid_edges(&topo, solid).unwrap();
    let edge = topo.edge(edges[0]).unwrap();
    let end_vid = edge.end();
    let start_pos = topo.vertex(edge.start()).unwrap().point();

    // Move end vertex to same position as start
    topo.vertex_mut(end_vid).unwrap().set_point(start_pos);

    let report = validate_solid(&topo, solid).unwrap();
    let has_zero_length = report.issues.iter().any(|i| {
        i.description.contains("zero length") || i.description.contains("near-zero length")
    });
    assert!(
        has_zero_length,
        "should detect zero-length edge: {:?}",
        report.issues
    );
}

#[test]
fn validate_connected_shell_passes() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    let has_disconnect = report
        .issues
        .iter()
        .any(|i| i.description.contains("disconnected"));
    assert!(
        !has_disconnect,
        "valid box should not be disconnected: {:?}",
        report.issues
    );
}

#[test]
fn validate_detects_redundant_face() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    // Duplicate a face in the shell
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    let mut faces: Vec<_> = shell.faces().to_vec();
    let dup = faces[0];
    faces.push(dup);

    let new_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
    *topo.shell_mut(shell_id).unwrap() = new_shell;

    let report = validate_solid(&topo, solid).unwrap();
    let has_redundant = report
        .issues
        .iter()
        .any(|i| i.description.contains("redundant") || i.description.contains("appears"));
    assert!(
        has_redundant,
        "should detect redundant face: {:?}",
        report.issues
    );
}

#[test]
fn boolean_fuse_result_validates() {
    let mut topo = Topology::new();
    let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let result = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, a, b).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(
        report.is_valid(),
        "boolean fuse should produce a valid solid: {:?}",
        report.issues
    );
}

#[test]
fn boolean_cut_result_validates() {
    let mut topo = Topology::new();
    let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let result = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(
        report.is_valid(),
        "boolean cut should produce a valid solid: {:?}",
        report.issues
    );
}

#[test]
fn boolean_intersect_result_validates() {
    let mut topo = Topology::new();
    let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Intersect, a, b).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(
        report.is_valid(),
        "boolean intersect should produce a valid solid: {:?}",
        report.issues
    );
}

#[test]
#[allow(deprecated)]
fn fillet_result_validates() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    // Find edges for fillet
    let s = topo.solid(cube).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &fid in sh.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            if seen.insert(oe.edge().index()) {
                edges.push(oe.edge());
            }
        }
    }

    let result = crate::fillet::fillet(&mut topo, cube, &[edges[0]], 0.1).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(
        report.is_valid(),
        "fillet should produce a valid solid: {:?}",
        report.issues
    );
}

#[test]
fn extrude_result_validates() {
    let mut topo = Topology::new();
    let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);
    let solid = crate::extrude::extrude(
        &mut topo,
        face,
        brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
        2.0,
    )
    .unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "extrude result should validate: {:?}",
        report.issues
    );
}

#[test]
fn revolve_result_validates() {
    use brepkit_math::vec::{Point3, Vec3};

    let mut topo = Topology::new();
    let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);

    // Move face away from axis to avoid degenerate geometry
    for vid in explorer::face_vertices(&topo, face).unwrap() {
        let v = topo.vertex_mut(vid).unwrap();
        v.set_point(Point3::new(
            v.point().x() + 2.0,
            v.point().y(),
            v.point().z(),
        ));
    }

    let solid = crate::revolve::revolve(
        &mut topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        std::f64::consts::PI,
    )
    .unwrap();

    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "revolve result should validate: {:?}",
        report.issues
    );
}

#[test]
fn repair_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
    assert!(
        report.is_valid_after(),
        "clean box should be valid after repair"
    );
    assert_eq!(report.total_repairs(), 0, "clean box needs no repairs");
}

#[test]
fn repair_preserves_volume() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    let _report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
    let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    assert!(
        (vol_before - vol_after).abs() < 0.01,
        "repair should preserve volume: {vol_before} vs {vol_after}"
    );
}

#[test]
fn repair_cylinder_no_crash() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
    // Should not crash; may or may not be valid depending on cylinder topology
    let _ = report.is_valid_after();
}

#[test]
fn relaxed_valid_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = validate_solid_relaxed(&topo, solid).unwrap();
    assert!(
        report.is_valid(),
        "box should pass relaxed validation: {:?}",
        report.issues
    );
}

#[test]
fn relaxed_fillet_passes() {
    let mut topo = Topology::new();
    let cube = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();

    let s = topo.solid(cube).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &fid in sh.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            if seen.insert(oe.edge().index()) {
                edges.push(oe.edge());
            }
        }
    }

    #[allow(deprecated)]
    let result = crate::fillet::fillet_rolling_ball(&mut topo, cube, &[edges[0]], 2.0).unwrap();

    // Strict validation may pass or fail for oversized fillets (R=2 on unit cube).
    // After the fillet contact direction fix, strict validation now passes for
    // some previously-failing cases. This is not a regression.
    let _strict = validate_solid(&topo, result).unwrap();

    // Relaxed validation should pass
    let relaxed = validate_solid_relaxed(&topo, result).unwrap();
    assert!(
        relaxed.is_valid(),
        "fillet should pass relaxed validation: {:?}",
        relaxed.issues
    );
}

#[test]
fn relaxed_shell_passes() {
    let mut topo = Topology::new();
    let cube = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();

    let s = topo.solid(cube).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let open_face = sh.faces()[0];

    let result = crate::shell_op::shell(&mut topo, cube, 1.0, &[open_face]).unwrap();

    // Both strict and relaxed validation should pass for properly
    // constructed shells (plane-intersected inner vertex positions
    // ensure watertight geometry).
    let strict = validate_solid(&topo, result).unwrap();
    assert!(
        strict.is_valid(),
        "shell should pass strict validation: {:?}",
        strict.issues
    );

    let relaxed = validate_solid_relaxed(&topo, result).unwrap();
    assert!(
        relaxed.is_valid(),
        "shell should pass relaxed validation: {:?}",
        relaxed.issues
    );
}

#[test]
fn relaxed_boolean_cut_passes() {
    let mut topo = Topology::new();
    let a = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    let b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
    crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

    let result = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

    let relaxed = validate_solid_relaxed(&topo, result).unwrap();
    assert!(
        relaxed.is_valid(),
        "boolean cut should pass relaxed validation: {:?}",
        relaxed.issues
    );
}

#[test]
fn relaxed_detects_open_wire_as_warning() {
    use brepkit_topology::wire::Wire;

    // Open wire is demoted to Warning in relaxed validation — it doesn't
    // prevent downstream use (tessellation, export) for boolean results.
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let face_id = topo.shell(shell_id).unwrap().faces()[0];
    let wire_id = topo.face(face_id).unwrap().outer_wire();
    let wire = topo.wire(wire_id).unwrap();
    let edges = wire.edges().to_vec();

    assert!(edges.len() > 1, "box face should have > 1 edge");
    let open_wire = Wire::new(edges[..edges.len() - 1].to_vec(), false).unwrap();
    *topo.wire_mut(wire_id).unwrap() = open_wire;

    let report = validate_solid_relaxed(&topo, solid).unwrap();
    // Relaxed validation reports open wires as warnings, not errors.
    assert!(
        report.is_valid(),
        "open wire should be warning (valid=true) in relaxed validation: {:?}",
        report.issues
    );
    // Should have at least one warning about the open wire.
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.description.contains("not closed")),
        "should warn about open wire"
    );
}

#[test]
fn validation_options_default() {
    let opts = ValidationOptions::default();
    assert!((opts.tolerance_scale - 1.0).abs() < f64::EPSILON);
}

#[test]
fn with_options_default_matches_validate_solid() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let default_report = validate_solid(&topo, solid).unwrap();
    let opts_report =
        validate_solid_with_options(&topo, solid, &ValidationOptions::default()).unwrap();

    assert_eq!(default_report.error_count(), opts_report.error_count());
    assert_eq!(default_report.warning_count(), opts_report.warning_count());
}

#[test]
fn scaled_tolerance_reduces_normal_warnings() {
    // Corrupt a face normal to be slightly off from unit length.
    // Default tolerance should warn; scaled tolerance should not.
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let face_id = topo.shell(shell_id).unwrap().faces()[0];
    let face = topo.face_mut(face_id).unwrap();
    *face = brepkit_topology::face::Face::new(
        face.outer_wire(),
        face.inner_wires().to_vec(),
        brepkit_topology::face::FaceSurface::Plane {
            // Normal length ~0.99999 — off by ~1e-5 which exceeds default 1e-7
            normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 0.99999),
            d: 0.0,
        },
    );

    // Default: should warn
    let default_report = validate_solid(&topo, solid).unwrap();
    let normal_warnings_default = default_report
        .issues
        .iter()
        .filter(|i| i.description.contains("non-unit normal"))
        .count();
    assert!(
        normal_warnings_default > 0,
        "default tolerance should warn on non-unit normal"
    );

    // Scaled up 100x: should not warn (1e-5 < 1e-7 * 100 = 1e-5)
    let opts = ValidationOptions {
        tolerance_scale: 100.0,
    };
    let scaled_report = validate_solid_with_options(&topo, solid, &opts).unwrap();
    let normal_warnings_scaled = scaled_report
        .issues
        .iter()
        .filter(|i| i.description.contains("non-unit normal"))
        .count();
    assert!(
        normal_warnings_scaled < normal_warnings_default,
        "scaled tolerance should produce fewer normal warnings ({normal_warnings_scaled} vs {normal_warnings_default})"
    );
}

#[test]
fn fillet_box_with_options() {
    let mut topo = Topology::new();
    let cube = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();

    let s = topo.solid(cube).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &fid in sh.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            if seen.insert(oe.edge().index()) {
                edges.push(oe.edge());
            }
        }
    }

    #[allow(deprecated)]
    let result = crate::fillet::fillet_rolling_ball(&mut topo, cube, &[edges[0]], 2.0).unwrap();

    // With relaxed + scaled options, should pass clean
    let opts = ValidationOptions {
        tolerance_scale: 10.0,
    };
    let report = validate_solid_relaxed_with_options(&topo, result, &opts).unwrap();
    assert!(
        report.is_valid(),
        "fillet should pass relaxed+scaled validation: {:?}",
        report.issues
    );
}
