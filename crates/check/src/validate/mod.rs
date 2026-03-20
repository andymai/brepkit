//! Hierarchical shape validation.

pub mod checks;
pub(crate) mod edge;
pub(crate) mod face;
pub(crate) mod shell;
pub(crate) mod solid;
pub(crate) mod vertex;
pub(crate) mod wire;

pub use checks::{CheckId, EntityRef, Severity, ValidationIssue, ValidationReport};

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Options controlling which checks run.
#[derive(Debug, Clone)]
pub struct ValidateOptions {
    /// Geometric tolerance scale factor (default 1.0).
    pub tolerance_scale: f64,
    /// Checks to skip.
    pub disabled_checks: HashSet<CheckId>,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            tolerance_scale: 1.0,
            disabled_checks: HashSet::new(),
        }
    }
}

/// Validate a solid (full check suite).
///
/// Runs solid-level checks, then shell and wire checks on each shell.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_solid(
    topo: &Topology,
    solid_id: SolidId,
    options: &ValidateOptions,
) -> Result<ValidationReport, CheckError> {
    let mut report = ValidationReport::default();
    let solid_data = topo.solid(solid_id)?;

    // Solid-level checks
    if !options
        .disabled_checks
        .contains(&CheckId::SolidEulerCharacteristic)
    {
        report.issues.extend(solid::check_euler(topo, solid_id)?);
    }
    if !options
        .disabled_checks
        .contains(&CheckId::SolidDuplicateFaces)
    {
        report
            .issues
            .extend(solid::check_duplicate_faces(topo, solid_id)?);
    }

    // Shell checks on outer shell + inner shells
    let shells: Vec<_> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();
    for &sid in &shells {
        report
            .issues
            .extend(validate_shell_checks(topo, sid, options)?);
    }

    Ok(report)
}

/// Validate a single shell.
///
/// Runs shell-level checks and wire checks for each face.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_shell(
    topo: &Topology,
    shell_id: ShellId,
    options: &ValidateOptions,
) -> Result<ValidationReport, CheckError> {
    let mut report = ValidationReport::default();
    report
        .issues
        .extend(validate_shell_checks(topo, shell_id, options)?);
    Ok(report)
}

/// Internal: run shell + wire checks on a shell.
fn validate_shell_checks(
    topo: &Topology,
    shell_id: ShellId,
    options: &ValidateOptions,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let mut issues = Vec::new();

    // Shell checks
    if !options.disabled_checks.contains(&CheckId::ShellEmpty) {
        issues.extend(shell::check_shell_empty(topo, shell_id)?);
    }
    if !options.disabled_checks.contains(&CheckId::ShellConnected) {
        issues.extend(shell::check_shell_connected(topo, shell_id)?);
    }
    if !options.disabled_checks.contains(&CheckId::ShellClosed) {
        issues.extend(shell::check_shell_closed(topo, shell_id)?);
    }

    // Wire checks for each face's outer wire
    let shell = topo.shell(shell_id)?;
    let mut checked_wires = HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wid = face.outer_wire();
        if checked_wires.insert(wid) {
            if !options.disabled_checks.contains(&CheckId::WireEmpty) {
                issues.extend(wire::check_wire_empty(topo, wid)?);
            }
            if !options.disabled_checks.contains(&CheckId::WireNotConnected) {
                issues.extend(wire::check_wire_connected(topo, wid)?);
            }
            if !options.disabled_checks.contains(&CheckId::WireClosure3D) {
                issues.extend(wire::check_wire_closure(topo, wid)?);
            }
            if !options
                .disabled_checks
                .contains(&CheckId::WireRedundantEdge)
            {
                issues.extend(wire::check_wire_redundant(topo, wid)?);
            }
            if !options
                .disabled_checks
                .contains(&CheckId::WireSelfIntersection)
            {
                issues.extend(wire::check_wire_self_intersection(topo, wid)?);
            }
        }
    }

    // Face checks
    let mut checked_faces = HashSet::new();
    for &fid in shell.faces() {
        if checked_faces.insert(fid) {
            if !options.disabled_checks.contains(&CheckId::FaceNoSurface) {
                issues.extend(face::check_face_has_surface(topo, fid)?);
            }
            if !options
                .disabled_checks
                .contains(&CheckId::FaceOrientationConsistency)
            {
                issues.extend(face::check_face_orientation(topo, fid)?);
            }
        }
    }

    // Edge and vertex checks
    let mut checked_edges = HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let outer_wire = topo.wire(face.outer_wire())?;
        for oe in outer_wire.edges() {
            let eid = oe.edge();
            if checked_edges.insert(eid) {
                if !options.disabled_checks.contains(&CheckId::EdgeRangeValid) {
                    issues.extend(edge::check_edge_range(
                        topo,
                        eid,
                        options.tolerance_scale * 1e-7,
                    )?);
                }
                if !options.disabled_checks.contains(&CheckId::EdgeDegenerate) {
                    issues.extend(edge::check_edge_degenerate(
                        topo,
                        eid,
                        options.tolerance_scale * 1e-7,
                    )?);
                }
                // Vertex-on-curve checks
                if !options.disabled_checks.contains(&CheckId::VertexOnCurve) {
                    let edge_data = topo.edge(eid)?;
                    issues.extend(vertex::check_vertex_on_curve(
                        topo,
                        edge_data.start(),
                        eid,
                        options.tolerance_scale * 1e-4,
                    )?);
                    if edge_data.start() != edge_data.end() {
                        issues.extend(vertex::check_vertex_on_curve(
                            topo,
                            edge_data.end(),
                            eid,
                            options.tolerance_scale * 1e-4,
                        )?);
                    }
                }
            }
        }
    }

    // SameParameter: check edge's 3D curve vs PCurve on each adjacent face.
    if !options
        .disabled_checks
        .contains(&CheckId::EdgeSameParameter)
    {
        let mut sp_checked = HashSet::new();
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            let outer_wire = topo.wire(face.outer_wire())?;
            for oe in outer_wire.edges() {
                let eid = oe.edge();
                if sp_checked.insert((eid, fid)) {
                    issues.extend(edge::check_edge_same_parameter(
                        topo,
                        eid,
                        fid,
                        options.tolerance_scale * 1e-4,
                    )?);
                }
            }
        }
    }

    Ok(issues)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn valid_box_no_issues() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let opts = ValidateOptions::default();
        let report = validate_solid(&topo, cube, &opts).unwrap();
        assert!(
            report.is_valid(),
            "unit cube should have no errors, got: {:?}",
            report.issues
        );
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn edge_range_valid_for_box() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let opts = ValidateOptions::default();
        let report = validate_solid(&topo, cube, &opts).unwrap();
        let range_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.check == CheckId::EdgeRangeValid)
            .collect();
        assert!(
            range_issues.is_empty(),
            "box edges should have valid ranges, got: {range_issues:?}"
        );
    }

    #[test]
    fn valid_box_detailed() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let opts = ValidateOptions::default();
        let report = validate_solid(&topo, cube, &opts).unwrap();
        assert_eq!(report.error_count(), 0, "errors: {:?}", report.issues);
        assert_eq!(report.warning_count(), 0, "warnings: {:?}", report.issues);
    }

    #[test]
    fn euler_characteristic_correct() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let opts = ValidateOptions::default();
        let report = validate_solid(&topo, cube, &opts).unwrap();
        // Cube: V=8, E=12, F=6 → V-E+F = 2
        let euler_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.check == CheckId::SolidEulerCharacteristic)
            .collect();
        assert!(
            euler_issues.is_empty(),
            "cube Euler characteristic should be 2, got issues: {euler_issues:?}"
        );
    }
}
