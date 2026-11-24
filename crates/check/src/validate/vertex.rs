//! Vertex geometric validation checks.

use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::VertexId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check that a vertex lies where its edge's 3D curve passes, within
/// tolerance.
///
/// The curve is read at the end of the edge's span (see
/// [`EdgeCurve::domain_with_endpoints`]) that the vertex bounds, as every
/// reader of the edge reads it, so a vertex at the wrong end of a B-spline
/// or past the end of its span is found. A closed edge's one vertex, or a
/// vertex that bounds neither end, need only lie on the curve: a closed
/// circle is read from its frame's origin, wherever its vertex lies. A
/// curve that is not well formed (see [`EdgeCurve::is_well_formed`]) is
/// reported, not read.
///
/// # Errors
///
/// Returns an error if the edge or a vertex is not in `topo`.
pub fn check_vertex_on_curve(
    topo: &Topology,
    vertex_id: VertexId,
    edge_id: EdgeId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let edge = topo.edge(edge_id)?;
    let curve = edge.curve();
    // For Line edges, the vertices ARE the geometry — always consistent.
    if matches!(curve, EdgeCurve::Line) {
        return Ok(vec![]);
    }
    if !curve.is_well_formed() {
        return Ok(vec![ValidationIssue {
            check: CheckId::VertexOnCurve,
            severity: Severity::Warning,
            entity: EntityRef::Edge(edge_id),
            description: "the edge's curve is malformed, so it is not read".into(),
            deviation: None,
        }]);
    }
    let pos = topo.vertex(vertex_id)?.point();
    let (start, end) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let at_end = if edge.start() == edge.end() {
        None
    } else if vertex_id == edge.start() {
        Some(true)
    } else if vertex_id == edge.end() {
        Some(false)
    } else {
        None
    };
    let deviation = if let Some(first) = at_end {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        let t = if first { t0 } else { t1 };
        (pos - curve.evaluate_with_endpoints(t, start, end)).length()
    } else {
        distance_to_curve(curve, pos)
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

/// How far `pos` lies from the nearest point of `curve`, a circle, an
/// ellipse or a B-spline over its whole domain.
fn distance_to_curve(curve: &EdgeCurve, pos: brepkit_math::vec::Point3) -> f64 {
    match curve {
        EdgeCurve::Line => 0.0,
        EdgeCurve::Circle(c) => (pos - c.evaluate(c.project(pos))).length(),
        EdgeCurve::Ellipse(e) => (pos - e.evaluate(e.project(pos))).length(),
        EdgeCurve::NurbsCurve(nc) => {
            let (t0, t1) = nc.domain();
            let ends = (pos - nc.evaluate(t0))
                .length()
                .min((pos - nc.evaluate(t1)).length());
            brepkit_math::nurbs::projection::project_point_to_curve(nc, pos, 1e-10)
                .map_or(ends, |p| p.distance.min(ends))
        }
    }
}

/// Check that a vertex lies on a face's surface within tolerance.
///
/// Projects the vertex onto the surface and measures deviation.
pub fn check_vertex_on_surface(
    topo: &Topology,
    vertex_id: VertexId,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let vertex = topo.vertex(vertex_id)?;
    let pos = vertex.point();
    let face = topo.face(face_id)?;

    let deviation = match face.surface() {
        FaceSurface::Plane { normal, d } => {
            let pv = brepkit_math::vec::Vec3::new(pos.x(), pos.y(), pos.z());
            (normal.dot(pv) - d).abs()
        }
        FaceSurface::Cylinder(s) => {
            let (u, v) = s.project_point(pos);
            (pos - s.evaluate(u, v)).length()
        }
        FaceSurface::Cone(s) => {
            let (u, v) = s.project_point(pos);
            (pos - s.evaluate(u, v)).length()
        }
        FaceSurface::Sphere(s) => {
            let (u, v) = s.project_point(pos);
            (pos - s.evaluate(u, v)).length()
        }
        FaceSurface::Torus(s) => {
            let (u, v) = s.project_point(pos);
            (pos - s.evaluate(u, v)).length()
        }
        FaceSurface::Nurbs(s) => {
            match brepkit_math::nurbs::projection::project_point_to_surface(s, pos, tolerance) {
                Ok(proj) => proj.distance,
                Err(_) => return Ok(vec![]),
            }
        }
    };

    if deviation > tolerance {
        return Ok(vec![ValidationIssue {
            check: CheckId::VertexOnSurface,
            severity: Severity::Warning,
            entity: EntityRef::Vertex(vertex_id),
            description: format!(
                "vertex deviates {deviation:.2e} from face surface (tolerance {tolerance:.2e})"
            ),
            deviation: Some(deviation),
        }]);
    }
    Ok(vec![])
}
