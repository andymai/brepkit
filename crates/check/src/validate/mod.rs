//! Hierarchical shape validation.

pub mod checks;
pub mod curved_overlap;
pub(crate) mod edge;
pub(crate) mod face;
pub mod face_overlap;
pub mod face_wires;
pub mod shell;
pub(crate) mod solid;
pub(crate) mod vertex;
pub(crate) mod wire;

pub use checks::{CheckId, EntityRef, Severity, ValidationIssue, ValidationReport};
pub use edge::check_edge_same_parameter;
pub use vertex::check_vertex_on_curve;

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// An error for each curve, surface or pcurve of `faces` that is not well
/// formed.
///
/// It reads each face's surface, each edge of its wires and each pcurve of
/// such an edge on the face (see `FaceSurface::is_well_formed`,
/// `EdgeCurve::is_well_formed` and `Curve2D::is_well_formed`). A plane has
/// no parameters to read, and the validators measure its normal themselves,
/// so a plane is not reported here.
///
/// The other checks read curves and surfaces, and one that is malformed can
/// read past its arrays or give points that are not finite. A solid or
/// shell that holds any is reported on these errors alone.
///
/// # Errors
///
/// Returns an error if a face, wire or edge is not in `topo`.
pub fn check_geometry_well_formed(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<ValidationIssue>, CheckError> {
    let issue = |check, entity, description: String| ValidationIssue {
        check,
        severity: Severity::Error,
        entity,
        description,
        deviation: None,
    };
    let mut issues = Vec::new();
    let mut seen = HashSet::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        let surface = face.surface();
        if !surface.is_planar() && !surface.is_well_formed() {
            issues.push(issue(
                CheckId::FaceNoSurface,
                EntityRef::Face(fid),
                "the face's surface is malformed, so it is not read".into(),
            ));
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let eid = oe.edge();
                if seen.insert(eid) && !topo.edge(eid)?.curve().is_well_formed() {
                    issues.push(issue(
                        CheckId::EdgeNoCurve3D,
                        EntityRef::Edge(eid),
                        "the edge's curve is malformed, so it is not read".into(),
                    ));
                }
                if let Some(pc) = topo.pcurves().get(eid, fid)
                    && !(pc.t_start().is_finite()
                        && pc.t_end().is_finite()
                        && pc.curve().is_well_formed())
                {
                    issues.push(issue(
                        CheckId::EdgeSameParameter,
                        EntityRef::Edge(eid),
                        format!(
                            "PCurve is malformed: its span or its curve's data cannot be \
                             evaluated, on face {}",
                            fid.index()
                        ),
                    ));
                }
            }
        }
    }
    Ok(issues)
}

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
/// Runs solid-level checks, then shell and wire checks on each shell. A
/// solid with a malformed curve, surface or pcurve is reported on those
/// alone (see [`check_geometry_well_formed`]).
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_solid(
    topo: &Topology,
    solid_id: SolidId,
    options: &ValidateOptions,
) -> Result<ValidationReport, CheckError> {
    let solid_data = topo.solid(solid_id)?;
    let faces = brepkit_topology::explorer::solid_faces(topo, solid_id)?;
    let mut report = ValidationReport {
        issues: check_geometry_well_formed(topo, &faces)?,
    };
    if !report.issues.is_empty() {
        return Ok(report);
    }

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
    if !options
        .disabled_checks
        .contains(&CheckId::SolidFacesOverlap)
    {
        report.issues.extend(face_overlap::check_faces_overlap(
            topo,
            solid_id,
            options.tolerance_scale * 1e-6,
        )?);
        report
            .issues
            .extend(curved_overlap::check_curved_faces_overlap(
                topo,
                solid_id,
                options.tolerance_scale * 1e-6,
            )?);
    }

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
/// Runs shell-level checks and wire checks for each face. A shell with a
/// malformed curve, surface or pcurve is reported on those alone (see
/// [`check_geometry_well_formed`]).
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_shell(
    topo: &Topology,
    shell_id: ShellId,
    options: &ValidateOptions,
) -> Result<ValidationReport, CheckError> {
    let mut report = ValidationReport {
        issues: check_geometry_well_formed(topo, topo.shell(shell_id)?.faces())?,
    };
    if !report.issues.is_empty() {
        return Ok(report);
    }
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

    if !options.disabled_checks.contains(&CheckId::ShellEmpty) {
        issues.extend(shell::check_shell_empty(topo, shell_id)?);
    }
    if !options.disabled_checks.contains(&CheckId::ShellConnected) {
        issues.extend(shell::check_shell_connected(topo, shell_id)?);
    }
    if !options.disabled_checks.contains(&CheckId::ShellClosed) {
        issues.extend(shell::check_shell_closed(topo, shell_id)?);
    }
    if !options
        .disabled_checks
        .contains(&CheckId::ShellOrientationConsistent)
    {
        issues.extend(shell::check_shell_orientation(topo, shell_id)?);
    }

    let shell = topo.shell(shell_id)?;
    let mut checked_wires = HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let mut wire_ids = vec![face.outer_wire()];
        wire_ids.extend(face.inner_wires().iter().copied());
        for wid in wire_ids {
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
            }
        }
    }

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
            if !options
                .disabled_checks
                .contains(&CheckId::FaceWiresIntersect)
            {
                issues.extend(face_wires::check_face_wires_apart(
                    topo,
                    fid,
                    options.tolerance_scale * 1e-6,
                )?);
            }
            if !options.disabled_checks.contains(&CheckId::FaceWireTurns) {
                issues.extend(face_wires::check_face_wire_turns(topo, fid)?);
            }
            if !options
                .disabled_checks
                .contains(&CheckId::WireSelfIntersection)
            {
                issues.extend(face_wires::check_face_wire_crossings(
                    topo,
                    fid,
                    options.tolerance_scale * 1e-6,
                )?);
            }
        }
    }

    let mut checked_edges = HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let mut wire_ids = vec![face.outer_wire()];
        wire_ids.extend(face.inner_wires().iter().copied());
        for wid in wire_ids {
            let wire_data = topo.wire(wid)?;
            for oe in wire_data.edges() {
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
                    if !options.disabled_checks.contains(&CheckId::VertexOnSurface) {
                        let edge_data = topo.edge(eid)?;
                        issues.extend(vertex::check_vertex_on_surface(
                            topo,
                            edge_data.start(),
                            fid,
                            options.tolerance_scale * 1e-4,
                        )?);
                        if edge_data.start() != edge_data.end() {
                            issues.extend(vertex::check_vertex_on_surface(
                                topo,
                                edge_data.end(),
                                fid,
                                options.tolerance_scale * 1e-4,
                            )?);
                        }
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
            let mut wire_ids = vec![face.outer_wire()];
            wire_ids.extend(face.inner_wires().iter().copied());
            for wid in wire_ids {
                let wire_data = topo.wire(wid)?;
                for oe in wire_data.edges() {
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
