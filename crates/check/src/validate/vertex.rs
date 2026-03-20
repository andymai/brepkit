//! Vertex geometric validation checks.

use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::vertex::VertexId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check that a vertex lies on its edge's 3D curve within tolerance.
///
/// Evaluates the edge curve at the domain start/end and measures
/// the distance to the vertex position.
pub fn check_vertex_on_curve(
    topo: &Topology,
    vertex_id: VertexId,
    edge_id: EdgeId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let vertex = topo.vertex(vertex_id)?;
    let edge = topo.edge(edge_id)?;
    let pos = vertex.point();

    // For Line edges, the vertices ARE the geometry — always consistent.
    let deviation = match edge.curve() {
        EdgeCurve::Line => return Ok(vec![]),
        EdgeCurve::Circle(c) => {
            let t_closest = c.project(pos);
            (pos - c.evaluate(t_closest)).length()
        }
        EdgeCurve::Ellipse(e) => {
            let t_closest = e.project(pos);
            (pos - e.evaluate(t_closest)).length()
        }
        EdgeCurve::NurbsCurve(nc) => {
            let (t0, t1) = nc.domain();
            let d_start = (pos - nc.evaluate(t0)).length();
            let d_end = (pos - nc.evaluate(t1)).length();
            d_start.min(d_end)
        }
    };

    if deviation > tolerance {
        return Ok(vec![ValidationIssue {
            check: CheckId::VertexOnCurve,
            severity: Severity::Warning,
            entity: EntityRef::Vertex(vertex_id),
            description: format!(
                "vertex deviates {deviation:.2e} from edge curve (tolerance {tolerance:.2e})"
            ),
            deviation: Some(deviation),
        }]);
    }
    Ok(vec![])
}
