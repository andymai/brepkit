//! Shell validation checks.

use std::collections::{HashMap, HashSet, VecDeque};

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::shell::ShellId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check shell is not empty.
pub fn check_shell_empty(
    topo: &Topology,
    shell_id: ShellId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let shell = topo.shell(shell_id)?;
    if shell.faces().is_empty() {
        return Ok(vec![ValidationIssue {
            check: CheckId::ShellEmpty,
            severity: Severity::Error,
            entity: EntityRef::Shell(shell_id),
            description: "shell contains no faces".into(),
            deviation: None,
        }]);
    }
    Ok(vec![])
}

/// Collect all edge IDs from a face (outer wire + inner wires).
fn face_edge_ids(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
) -> Result<Vec<EdgeId>, crate::CheckError> {
    let face = topo.face(face_id)?;
    let mut eids = Vec::new();
    let wire = topo.wire(face.outer_wire())?;
    for oe in wire.edges() {
        eids.push(oe.edge());
    }
    for &iw in face.inner_wires() {
        let inner_wire = topo.wire(iw)?;
        for oe in inner_wire.edges() {
            eids.push(oe.edge());
        }
    }
    Ok(eids)
}

/// Check shell connectivity: all faces connected via shared edges (BFS).
#[allow(clippy::too_many_lines)]
pub fn check_shell_connected(
    topo: &Topology,
    shell_id: ShellId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let shell = topo.shell(shell_id)?;
    let faces = shell.faces();
    if faces.len() <= 1 {
        return Ok(vec![]);
    }

    // Build edge -> face-index adjacency
    let mut edge_to_faces: HashMap<EdgeId, Vec<usize>> = HashMap::new();
    for (fi, &fid) in faces.iter().enumerate() {
        for eid in face_edge_ids(topo, fid)? {
            edge_to_faces.entry(eid).or_default().push(fi);
        }
    }

    // BFS from face 0
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(0usize);
    queue.push_back(0usize);

    while let Some(fi) = queue.pop_front() {
        let fid = faces[fi];
        for eid in face_edge_ids(topo, fid)? {
            if let Some(neighbors) = edge_to_faces.get(&eid) {
                for &nfi in neighbors {
                    if visited.insert(nfi) {
                        queue.push_back(nfi);
                    }
                }
            }
        }
    }

    if visited.len() < faces.len() {
        return Ok(vec![ValidationIssue {
            check: CheckId::ShellConnected,
            severity: Severity::Error,
            entity: EntityRef::Shell(shell_id),
            description: format!(
                "shell has {} connected components ({} of {} faces reached)",
                faces.len() - visited.len() + 1,
                visited.len(),
                faces.len()
            ),
            deviation: None,
        }]);
    }
    Ok(vec![])
}

/// Check that shell face orientations are consistent: for each edge shared
/// by two faces, it should be used once FORWARD and once REVERSED.
pub fn check_shell_orientation(
    topo: &Topology,
    shell_id: ShellId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let shell = topo.shell(shell_id)?;

    // Map each edge to its (face_index, effective_forward) pairs
    let mut edge_uses: HashMap<EdgeId, Vec<(usize, bool)>> = HashMap::new();

    for (fi, &fid) in shell.faces().iter().enumerate() {
        let face = topo.face(fid)?;
        // Account for face reversal
        let face_reversed = face.is_reversed();
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let effective_forward = oe.is_forward() != face_reversed;
            edge_uses
                .entry(oe.edge())
                .or_default()
                .push((fi, effective_forward));
        }
        for &iw in face.inner_wires() {
            if let Ok(inner_wire) = topo.wire(iw) {
                for oe in inner_wire.edges() {
                    let effective_forward = oe.is_forward() != face_reversed;
                    edge_uses
                        .entry(oe.edge())
                        .or_default()
                        .push((fi, effective_forward));
                }
            }
        }
    }

    let mut misoriented = 0usize;

    for uses in edge_uses.values() {
        if uses.len() == 2 {
            // Shared edge: should be used once forward and once reversed
            if uses[0].1 == uses[1].1 {
                misoriented += 1;
            }
        }
    }

    if misoriented > 0 {
        return Ok(vec![ValidationIssue {
            check: CheckId::ShellOrientationConsistent,
            severity: Severity::Error,
            entity: EntityRef::Shell(shell_id),
            description: format!("{misoriented} shared edges have inconsistent face orientations"),
            deviation: Some(misoriented as f64),
        }]);
    }

    Ok(vec![])
}

/// Check shell closure: every edge shared by exactly 2 faces.
pub fn check_shell_closed(
    topo: &Topology,
    shell_id: ShellId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let shell = topo.shell(shell_id)?;
    let mut edge_count: HashMap<EdgeId, usize> = HashMap::new();

    for &fid in shell.faces() {
        for eid in face_edge_ids(topo, fid)? {
            *edge_count.entry(eid).or_default() += 1;
        }
    }

    let boundary = edge_count.values().filter(|&&c| c == 1).count();
    let nonmanifold = edge_count.values().filter(|&&c| c > 2).count();

    if boundary == 0 && nonmanifold == 0 {
        return Ok(vec![]);
    }

    let mut issues = Vec::new();
    if boundary > 0 {
        issues.push(ValidationIssue {
            check: CheckId::ShellClosed,
            severity: Severity::Error,
            entity: EntityRef::Shell(shell_id),
            description: format!("shell has {boundary} free (boundary) edges"),
            deviation: Some(boundary as f64),
        });
    }
    if nonmanifold > 0 {
        issues.push(ValidationIssue {
            check: CheckId::ShellClosed,
            severity: Severity::Error,
            entity: EntityRef::Shell(shell_id),
            description: format!("shell has {nonmanifold} non-manifold edges (shared by >2 faces)"),
            deviation: Some(nonmanifold as f64),
        });
    }
    Ok(issues)
}
