//! Comprehensive solid validation.
//!
//! Equivalent to `BRepCheck_Analyzer` in `OpenCascade`. Performs
//! structural and geometric validation on solids.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::TopologyError;
use brepkit_topology::explorer;
use brepkit_topology::solid::SolidId;

/// A validation issue found in a solid.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// Human-readable description.
    pub description: String,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The solid is invalid and may cause downstream failures.
    Error,
    /// The solid has a potential problem but may still be usable.
    Warning,
}

/// Result of validating a solid.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether the solid passed all validation checks (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Count of error-severity issues.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .count()
    }

    /// Count of warning-severity issues.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .count()
    }
}

/// Validate a solid, returning a report of all issues found.
///
/// Checks performed:
/// Returns `true` if every edge in the face is a straight line.
fn face_all_edges_straight(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
) -> Result<bool, TopologyError> {
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// 1. **Euler-Poincaré**: V - E + F = 2(1 - g) for genus-g closed solid
/// 2. **Manifold edges**: each edge shared by exactly 2 faces
/// 3. **Boundary edges**: no edge shared by only 1 face (open shell)
/// 4. **Degenerate faces**: each face has at least 3 vertices
/// 5. **Face normal consistency**: normals should be non-zero
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_solid(
    topo: &Topology,
    solid: SolidId,
) -> Result<ValidationReport, crate::OperationsError> {
    let mut issues = Vec::new();
    let tol = Tolerance::new();

    // 1. Entity counts and Euler characteristic.
    let (f, e, v) = explorer::solid_entity_counts(topo, solid)?;

    // Euler-Poincaré formula: V - E + F = 2(1 - g) for a closed orientable
    // surface of genus g. Genus-0 (sphere-like) → 2, genus-1 (torus) → 0,
    // genus-2 (double torus) → -2, etc.
    //
    // Rather than computing genus from inner wires (which is fragile),
    // we check that the Euler characteristic is consistent: it must be an
    // even integer ≤ 2, giving a non-negative integer genus.
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    let genus_times_2 = 2 - euler;
    if genus_times_2 < 0 || genus_times_2 % 2 != 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!(
                "Euler characteristic V-E+F = {euler} is invalid \
                 (expected even value ≤ 2 for closed solid, got V={v}, E={e}, F={f})"
            ),
        });
    }

    // 2 & 3. Edge sharing: each edge should be in exactly 2 faces.
    let edge_map = explorer::edge_to_face_map(topo, solid)?;
    let mut boundary_edges = 0;
    let mut non_manifold_edges = 0;

    for (&edge_idx, faces) in &edge_map {
        match faces.len() {
            0 => {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!("edge {edge_idx} is not referenced by any face"),
                });
            }
            1 => {
                boundary_edges += 1;
            }
            2 => {} // correct
            n => {
                non_manifold_edges += 1;
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "edge {edge_idx} is shared by {n} faces (non-manifold, expected 2)"
                    ),
                });
            }
        }
    }

    if boundary_edges > 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!("{boundary_edges} boundary edge(s) found (shell is not closed)"),
        });
    }

    if non_manifold_edges > 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!("{non_manifold_edges} non-manifold edge(s) found"),
        });
    }

    // 4. Degenerate faces.
    //
    // Only faces on a planar surface bounded entirely by straight edges
    // require ≥3 unique vertices. Faces with curved edges (Circle,
    // Ellipse, NURBS) or non-planar surfaces (Cylinder, Sphere, Torus,
    // etc.) can validly have fewer vertices because the surface/edge
    // geometry defines the boundary shape.
    let faces = explorer::solid_faces(topo, solid)?;
    for fid in &faces {
        let face_data = topo.face(*fid)?;
        let is_planar = matches!(
            face_data.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        );

        if is_planar && face_all_edges_straight(topo, face_data)? {
            let face_verts = explorer::face_vertices(topo, *fid)?;
            if face_verts.len() < 3 {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "face {} has only {} vertices (need at least 3)",
                        fid.index(),
                        face_verts.len()
                    ),
                });
            }
        }
    }

    // 5. Face normal consistency.
    for fid in &faces {
        let face = topo.face(*fid)?;
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            let len = normal.length();
            if !tol.approx_eq(len, 1.0) {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has non-unit normal (length = {len})",
                        fid.index()
                    ),
                });
            }
        }
    }

    // 6. Wire closure: every wire must form a closed loop.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if let Err(_e) = brepkit_topology::validation::validate_wire_closed(wire, &topo.edges) {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "wire {} on face {} is not closed",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    // 7. Degenerate face area: faces with near-zero polygon area are likely slivers.
    //
    // Only meaningful for faces bounded entirely by straight edges.
    // The polygon area formula uses vertex positions, which is
    // meaningless when edges are curved (e.g. a cylinder cap has
    // 1 vertex → zero polygon area despite being a valid disc).
    let area_tol_sq = tol.linear * tol.linear;
    for fid in &faces {
        let face = topo.face(*fid)?;

        // Skip non-planar faces and faces with curved edges — the polygon
        // area formula is only meaningful for planar faces with straight edges.
        if !matches!(
            face.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        ) {
            continue;
        }
        if !face_all_edges_straight(topo, face)? {
            continue;
        }

        let wire = topo.wire(face.outer_wire())?;

        // Collect wire vertex positions.
        let mut positions = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            positions.push(topo.vertex(vid)?.point());
        }

        // Compute face area using the shoelace formula generalized to 3D
        // (sum of cross products of consecutive edge vectors from centroid).
        if positions.len() >= 3 {
            let area = polygon_area_3d(&positions);
            if area < area_tol_sq {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has near-zero area ({area:.2e} < {area_tol_sq:.2e})",
                        fid.index()
                    ),
                });
            }
        }
    }

    Ok(ValidationReport { issues })
}

/// Compute the area of a 3D polygon using the cross-product method.
///
/// For a planar polygon with vertices `p0, p1, ..., pN`, the area is
/// half the magnitude of the sum of cross products `(p[i] - p[0]) × (p[i+1] - p[0])`.
fn polygon_area_3d(positions: &[brepkit_math::vec::Point3]) -> f64 {
    use brepkit_math::vec::Vec3;

    if positions.len() < 3 {
        return 0.0;
    }

    let p0 = positions[0];
    let mut sum = Vec3::new(0.0, 0.0, 0.0);

    for i in 1..positions.len() - 1 {
        let a = positions[i] - p0;
        let b = positions[i + 1] - p0;
        sum = sum + a.cross(b);
    }

    sum.length() * 0.5
}

#[cfg(test)]
mod tests {
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
        faces.pop(); // Remove last face

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
            brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts).unwrap();

        // Inner: 0.5×0.5 hole.
        let inner_pts = vec![
            Point3::new(1.5, 0.25, 0.0),
            Point3::new(1.5, 0.75, 0.0),
            Point3::new(2.5, 0.75, 0.0),
            Point3::new(2.5, 0.25, 0.0),
        ];
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d: 0.0 },
        );
        let face_id = topo.faces.alloc(face);

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
            brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts).unwrap();

        // Inner: 0.5×0.5 hole.
        let inner_pts = vec![
            Point3::new(-0.25, -0.25, 0.0),
            Point3::new(-0.25, 0.25, 0.0),
            Point3::new(0.25, 0.25, 0.0),
            Point3::new(0.25, -0.25, 0.0),
        ];
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d: 0.0 },
        );
        let face_id = topo.faces.alloc(face);

        // Extrude → genus-0 (V-E+F=2) with inner walls.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "extruded hollow box should be valid, got {} error(s): {:?}",
            report.error_count(),
            report.issues
        );
    }

    // ── Wire closure validation ──────────────────────

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

    // ── Degenerate face area ─────────────────────────

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

    // ── repair_solid ─────────────────────────────────

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
}
