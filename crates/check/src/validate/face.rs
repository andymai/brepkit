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

/// Check face orientation consistency: the outer wire runs counter-clockwise
/// about its surface's normal, on a reversed face too (the flag turns the
/// face, not its wire).
///
/// Uses Newell's method on the outer wire polygon to determine winding,
/// then compares with the surface normal at the polygon centroid.
pub fn check_face_orientation(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(vec![]); // Can't determine winding for degenerate polygon
    }

    // A loop whose projected area vanishes against its size (a full band's
    // two opposite rims and doubled seam) has no winding to compare.
    let area_vector = newell_vector(&polygon);
    let extent = polygon
        .iter()
        .map(|p| (*p - polygon[0]).length())
        .fold(0.0_f64, f64::max);
    if area_vector.length() <= 1e-9 * extent * extent {
        return Ok(vec![]);
    }
    let wire_normal = area_vector * (1.0 / area_vector.length());

    let face = topo.face(face_id)?;
    let centroid = polygon_centroid(&polygon);

    let surface_normal = if let Some((u, v)) = face.surface().project_point(centroid) {
        face.surface().normal(u, v)
    } else {
        // Plane: use stored normal directly
        face.surface().normal(0.0, 0.0)
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

/// Twice the polygon's vector area (Newell's method).
fn newell_vector(verts: &[Point3]) -> Vec3 {
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    for (a, b) in verts.iter().zip(verts.iter().cycle().skip(1)) {
        sum += Vec3::new(
            (a.y() - b.y()) * (a.z() + b.z()),
            (a.z() - b.z()) * (a.x() + b.x()),
            (a.x() - b.x()) * (a.y() + b.y()),
        );
    }
    sum
}

/// Compute polygon centroid.
fn polygon_centroid(verts: &[Point3]) -> Point3 {
    let n = verts.len() as f64;
    let sx: f64 = verts.iter().map(|v| v.x()).sum();
    let sy: f64 = verts.iter().map(|v| v.y()).sum();
    let sz: f64 = verts.iter().map(|v| v.z()).sum();
    Point3::new(sx / n, sy / n, sz / n)
}
