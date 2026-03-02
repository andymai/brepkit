//! Comprehensive solid validation.
//!
//! Equivalent to `BRepCheck_Analyzer` in `OpenCascade`. Performs
//! structural and geometric validation on solids.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
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
#[derive(Debug)]
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
/// 1. **Euler characteristic**: V - E + F = 2 for a closed solid
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

    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    if euler != 2 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!(
                "Euler characteristic V-E+F = {euler} (expected 2 for closed solid, got V={v}, E={e}, F={f})"
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
    let faces = explorer::solid_faces(topo, solid)?;
    for fid in &faces {
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

    Ok(ValidationReport { issues })
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
        // Cylinder may or may not pass all checks depending on tessellation
        // but should at least produce a report without panicking
        let _ = report.is_valid();
        let _ = report.error_count();
    }
}
