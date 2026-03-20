//! Solid validation checks.

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::explorer;
use brepkit_topology::solid::SolidId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check Euler-Poincare formula: V - E + F = 2 for genus-0 closed manifold.
#[allow(clippy::cast_possible_wrap)]
pub fn check_euler(topo: &Topology, solid_id: SolidId) -> Result<Vec<ValidationIssue>, CheckError> {
    let (faces, edges, vertices) = explorer::solid_entity_counts(topo, solid_id)?;
    let euler = vertices as i64 - edges as i64 + faces as i64;
    if euler != 2 {
        return Ok(vec![ValidationIssue {
            check: CheckId::SolidEulerCharacteristic,
            severity: Severity::Warning,
            entity: EntityRef::Solid(solid_id),
            description: format!("Euler characteristic V-E+F = {euler} (expected 2 for genus-0)"),
            deviation: Some((euler - 2).unsigned_abs() as f64),
        }]);
    }
    Ok(vec![])
}

/// Check that no face ID appears in multiple shells.
pub fn check_duplicate_faces(
    topo: &Topology,
    solid_id: SolidId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let solid = topo.solid(solid_id)?;
    let mut seen = HashSet::new();
    let mut issues = Vec::new();

    let all_shells =
        std::iter::once(solid.outer_shell()).chain(solid.inner_shells().iter().copied());

    for sid in all_shells {
        let shell = topo.shell(sid)?;
        for &fid in shell.faces() {
            if !seen.insert(fid) {
                issues.push(ValidationIssue {
                    check: CheckId::SolidDuplicateFaces,
                    severity: Severity::Error,
                    entity: EntityRef::Solid(solid_id),
                    description: "face appears in multiple shells".into(),
                    deviation: None,
                });
                break;
            }
        }
    }
    Ok(issues)
}
