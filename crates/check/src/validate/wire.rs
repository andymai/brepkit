//! Wire validation checks.

use std::collections::HashMap;

use brepkit_math::vec::Point3;
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

/// Check for wire self-intersection by sampling edges and testing for crossings.
///
/// Samples each edge at 8 points and checks for segment-segment crossings
/// between non-adjacent edge pairs.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn check_wire_self_intersection(
    topo: &Topology,
    wire_id: WireId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let wire = topo.wire(wire_id)?;
    let edges = wire.edges();
    if edges.len() < 3 {
        return Ok(vec![]); // Need at least 3 edges for self-intersection
    }

    // Sample each edge into polyline segments.
    let samples_per_edge = 8usize;
    let mut edge_segments: Vec<Vec<Point3>> = Vec::new();

    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let p0 = topo.vertex(edge.start())?.point();
        let p1 = topo.vertex(edge.end())?.point();

        match edge.curve() {
            brepkit_topology::edge::EdgeCurve::Line => {
                edge_segments.push(vec![p0, p1]);
            }
            brepkit_topology::edge::EdgeCurve::Circle(c) => {
                let mut pts = Vec::with_capacity(samples_per_edge + 1);
                for k in 0..=samples_per_edge {
                    let t = std::f64::consts::TAU * (k as f64) / (samples_per_edge as f64);
                    pts.push(c.evaluate(t));
                }
                if !oe.is_forward() {
                    pts.reverse();
                }
                edge_segments.push(pts);
            }
            brepkit_topology::edge::EdgeCurve::Ellipse(e) => {
                let mut pts = Vec::with_capacity(samples_per_edge + 1);
                for k in 0..=samples_per_edge {
                    let t = std::f64::consts::TAU * (k as f64) / (samples_per_edge as f64);
                    pts.push(e.evaluate(t));
                }
                if !oe.is_forward() {
                    pts.reverse();
                }
                edge_segments.push(pts);
            }
            brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
                let (t0, t1) = nc.domain();
                let mut pts = Vec::with_capacity(samples_per_edge + 1);
                for k in 0..=samples_per_edge {
                    let t = t0 + (t1 - t0) * (k as f64) / (samples_per_edge as f64);
                    pts.push(nc.evaluate(t));
                }
                if !oe.is_forward() {
                    pts.reverse();
                }
                edge_segments.push(pts);
            }
        }
    }

    // Check for segment-segment intersections between non-adjacent edges.
    let n_edges = edge_segments.len();
    for i in 0..n_edges {
        for j in (i + 2)..n_edges {
            // Skip adjacent edges (first and last are also adjacent in a closed wire).
            if j == n_edges - 1 && i == 0 {
                continue;
            }

            for si in 0..edge_segments[i].len().saturating_sub(1) {
                let a0 = edge_segments[i][si];
                let a1 = edge_segments[i][si + 1];
                for sj in 0..edge_segments[j].len().saturating_sub(1) {
                    let b0 = edge_segments[j][sj];
                    let b1 = edge_segments[j][sj + 1];

                    let (dist, _, _) =
                        crate::distance::edge::segment_segment_distance(a0, a1, b0, b1);
                    if dist < 1e-6 {
                        return Ok(vec![ValidationIssue {
                            check: CheckId::WireSelfIntersection,
                            severity: Severity::Error,
                            entity: EntityRef::Wire(wire_id),
                            description: format!(
                                "wire self-intersection between edges {i} and {j}"
                            ),
                            deviation: Some(dist),
                        }]);
                    }
                }
            }
        }
    }

    Ok(vec![])
}
