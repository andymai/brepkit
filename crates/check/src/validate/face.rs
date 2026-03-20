//! Face geometric validation checks.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// Check that a face has a valid surface (always true in current model,
/// but validates the face can be resolved).
pub fn check_face_has_surface(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let _face = topo.face(face_id)?;
    // In the current model, FaceSurface is always present (it's a required field).
    // This check validates the face entity exists and is resolvable.
    Ok(vec![])
}

/// Check face orientation consistency: face normal should be consistent
/// with the outer wire winding direction.
///
/// Uses Newell's method on the outer wire polygon to determine winding,
/// then compares with the face surface normal at the polygon centroid.
pub fn check_face_orientation(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(vec![]); // Can't determine winding for degenerate polygon
    }

    // Compute polygon normal via Newell's method
    let wire_normal = newell_normal(&polygon);
    if wire_normal.length() < 1e-15 {
        return Ok(vec![]); // Degenerate polygon
    }

    // Get surface normal at polygon centroid
    let face = topo.face(face_id)?;
    let centroid = polygon_centroid(&polygon);

    let surface_normal = if let Some((u, v)) = face.surface().project_point(centroid) {
        let n = face.surface().normal(u, v);
        if face.is_reversed() { -n } else { n }
    } else {
        // Plane: use stored normal directly
        let n = face.surface().normal(0.0, 0.0);
        if face.is_reversed() { -n } else { n }
    };

    // Check if normals agree (dot product > 0 means same direction)
    let dot = wire_normal.dot(surface_normal);
    if dot < -0.1 {
        // Allow some tolerance for curved surfaces
        return Ok(vec![ValidationIssue {
            check: CheckId::FaceOrientationConsistency,
            severity: Severity::Warning,
            entity: EntityRef::Face(face_id),
            description: format!("face normal inconsistent with wire winding (dot={dot:.3})"),
            deviation: Some(dot.abs()),
        }]);
    }

    Ok(vec![])
}

/// Compute polygon normal via Newell's method.
fn newell_normal(verts: &[Point3]) -> Vec3 {
    crate::util::polygon_normal(verts)
}

/// Compute polygon centroid.
fn polygon_centroid(verts: &[Point3]) -> Point3 {
    let n = verts.len() as f64;
    let sx: f64 = verts.iter().map(|v| v.x()).sum();
    let sy: f64 = verts.iter().map(|v| v.y()).sum();
    let sz: f64 = verts.iter().map(|v| v.z()).sum();
    Point3::new(sx / n, sy / n, sz / n)
}
