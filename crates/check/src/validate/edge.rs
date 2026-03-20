//! Edge geometric validation checks.

use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;

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

/// Check that an edge's 3D curve matches its PCurve(surface) within tolerance.
///
/// Samples N points along the edge, evaluates both the 3D curve and the
/// PCurve projected through the surface, and measures the maximum deviation.
#[allow(clippy::cast_precision_loss)]
pub fn check_edge_same_parameter(
    topo: &Topology,
    edge_id: EdgeId,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    // Get the PCurve for this edge on this face.
    let pcurve = match topo.pcurves().get(edge_id, face_id) {
        Some(pc) => pc,
        None => return Ok(vec![]), // No PCurve registered — can't check
    };

    let edge = topo.edge(edge_id)?;
    let face = topo.face(face_id)?;

    // Skip planes — they have no UV parameterization for evaluate().
    if matches!(
        face.surface(),
        brepkit_topology::face::FaceSurface::Plane { .. }
    ) {
        return Ok(vec![]);
    }

    // Sample N points along the PCurve parameter range.
    let n_samples = 10;
    let mut max_deviation = 0.0f64;

    let start_pt = topo.vertex(edge.start())?.point();
    let end_pt = topo.vertex(edge.end())?.point();
    let (t_start, t_end) = edge.curve().domain_with_endpoints(start_pt, end_pt);

    for i in 0..=n_samples {
        let t_norm = i as f64 / n_samples as f64;
        let t_pcurve = pcurve.t_start() + (pcurve.t_end() - pcurve.t_start()) * t_norm;

        // Evaluate PCurve to get (u, v) on surface.
        let uv = pcurve.evaluate(t_pcurve);

        // Evaluate surface at (u, v) to get 3D point from PCurve path.
        let pcurve_3d = match face.surface().evaluate(uv.x(), uv.y()) {
            Some(pt) => pt,
            None => continue, // Should not happen for non-Plane surfaces
        };

        // Evaluate 3D curve at same parameter in native domain.
        let t_curve = t_start + (t_end - t_start) * t_norm;
        let curve_3d = edge
            .curve()
            .evaluate_with_endpoints(t_curve, start_pt, end_pt);

        let deviation = (pcurve_3d - curve_3d).length();
        max_deviation = max_deviation.max(deviation);
    }

    if max_deviation > tolerance {
        return Ok(vec![ValidationIssue {
            check: CheckId::EdgeSameParameter,
            severity: Severity::Error,
            entity: EntityRef::Edge(edge_id),
            description: format!(
                "3D curve deviates {max_deviation:.2e} from PCurve(surface) (tolerance {tolerance:.2e})"
            ),
            deviation: Some(max_deviation),
        }]);
    }

    Ok(vec![])
}
