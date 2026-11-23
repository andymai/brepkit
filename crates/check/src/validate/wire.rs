//! Wire validation checks.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::wire::WireId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check that a wire is not empty.
pub fn check_wire_empty(
    topo: &Topology,
    wire_id: WireId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let wire = topo.wire(wire_id)?;
    if wire.edges().is_empty() {
        return Ok(vec![ValidationIssue {
            check: CheckId::WireEmpty,
            severity: Severity::Error,
            entity: EntityRef::Wire(wire_id),
            description: "wire contains no edges".into(),
            deviation: None,
        }]);
    }
    Ok(vec![])
}

/// Check that consecutive edges share vertices.
pub fn check_wire_connected(
    topo: &Topology,
    wire_id: WireId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let wire = topo.wire(wire_id)?;
    let edges = wire.edges();
    if edges.len() < 2 {
        return Ok(vec![]);
    }

    let mut issues = Vec::new();
    for i in 0..edges.len() - 1 {
        let edge_a = topo.edge(edges[i].edge())?;
        let edge_b = topo.edge(edges[i + 1].edge())?;
        let end_a = edges[i].oriented_end(edge_a);
        let start_b = edges[i + 1].oriented_start(edge_b);
        if end_a != start_b {
            issues.push(ValidationIssue {
                check: CheckId::WireNotConnected,
                severity: Severity::Error,
                entity: EntityRef::Wire(wire_id),
                description: format!("edges {} and {} not connected", i, i + 1),
                deviation: None,
            });
        }
    }
    Ok(issues)
}

/// Check 3D wire closure (last edge end == first edge start).
pub fn check_wire_closure(
    topo: &Topology,
    wire_id: WireId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let wire = topo.wire(wire_id)?;
    if !wire.is_closed() {
        return Ok(vec![]);
    }
    let edges = wire.edges();
    if edges.is_empty() {
        return Ok(vec![]);
    }

    let first_edge = topo.edge(edges[0].edge())?;
    let last_edge = topo.edge(edges[edges.len() - 1].edge())?;
    let first_start = edges[0].oriented_start(first_edge);
    let last_end = edges[edges.len() - 1].oriented_end(last_edge);

    if first_start != last_end {
        return Ok(vec![ValidationIssue {
            check: CheckId::WireClosure3D,
            severity: Severity::Error,
            entity: EntityRef::Wire(wire_id),
            description: "wire not closed: last edge end != first edge start".into(),
            deviation: None,
        }]);
    }
    Ok(vec![])
}

/// Check for edges appearing 3+ times in same wire.
pub fn check_wire_redundant(
    topo: &Topology,
    wire_id: WireId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let wire = topo.wire(wire_id)?;
    let mut counts: HashMap<_, usize> = HashMap::new();
    for oe in wire.edges() {
        *counts.entry(oe.edge()).or_default() += 1;
    }
    let mut issues = Vec::new();
    for (eid, count) in counts {
        if count >= 3 {
            issues.push(ValidationIssue {
                check: CheckId::WireRedundantEdge,
                severity: Severity::Error,
                entity: EntityRef::Edge(eid),
                description: format!("edge appears {count} times in wire"),
                deviation: None,
            });
        }
    }
    Ok(issues)
}
