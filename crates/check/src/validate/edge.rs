//! Edge geometric validation checks.

use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check that an edge's parameter range is valid (non-degenerate).
pub fn check_edge_range(
    topo: &Topology,
    edge_id: EdgeId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let edge = topo.edge(edge_id)?;
    match edge.curve() {
        EdgeCurve::Line => Ok(vec![]), // Line geometry defined by vertices
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Ok(vec![]), // Full curves, always valid
        EdgeCurve::NurbsCurve(nc) => {
            let (t0, t1) = nc.domain();
            if (t1 - t0).abs() < tolerance {
                return Ok(vec![ValidationIssue {
                    check: CheckId::EdgeRangeValid,
                    severity: Severity::Error,
                    entity: EntityRef::Edge(edge_id),
                    description: format!("edge NURBS domain [{t0}, {t1}] has zero extent"),
                    deviation: Some((t1 - t0).abs()),
                }]);
            }
            Ok(vec![])
        }
    }
}

/// Check if an edge is degenerate (start == end and near-zero length).
#[allow(clippy::too_many_lines)]
pub fn check_edge_degenerate(
    topo: &Topology,
    edge_id: EdgeId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let edge = topo.edge(edge_id)?;
    if edge.start() != edge.end() {
        return Ok(vec![]);
    }

    // Closed edges (full circles) are not degenerate
    match edge.curve() {
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => return Ok(vec![]),
        EdgeCurve::Line => {
            let p0 = topo.vertex(edge.start())?.point();
            let p1 = topo.vertex(edge.end())?.point();
            let len = (p1 - p0).length();
            if len < tolerance {
                return Ok(vec![ValidationIssue {
                    check: CheckId::EdgeDegenerate,
                    severity: Severity::Warning,
                    entity: EntityRef::Edge(edge_id),
                    description: format!("degenerate line edge: length {len:.2e}"),
                    deviation: Some(len),
                }]);
            }
        }
        EdgeCurve::NurbsCurve(nc) => {
            // Sample curve length
            let (t0, t1) = nc.domain();
            let n_samples = 10;
            let mut length = 0.0;
            let mut prev = nc.evaluate(t0);
            #[allow(clippy::cast_precision_loss)]
            for i in 1..=n_samples {
                let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
                let curr = nc.evaluate(t);
                length += (curr - prev).length();
                prev = curr;
            }
            if length < tolerance {
                return Ok(vec![ValidationIssue {
                    check: CheckId::EdgeDegenerate,
                    severity: Severity::Warning,
                    entity: EntityRef::Edge(edge_id),
                    description: format!("degenerate NURBS edge: length {length:.2e}"),
                    deviation: Some(length),
                }]);
            }
        }
    }
    Ok(vec![])
}
