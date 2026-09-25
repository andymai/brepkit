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
    // A loop on a sphere bounds a region on either side of it, and its wire
    // alone says which: there is no normal to hold it against (a cap bigger
    // than a hemisphere has its boundary's centroid over the far pole).
    if matches!(
        topo.face(face_id)?.surface(),
        brepkit_topology::face::FaceSurface::Sphere(_)
    ) {
        return Ok(vec![]);
    }
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

/// Twice the polygon's vector area (Newell's method), taken about its first
/// point so a loop far from the origin keeps no absolute-coordinate residue.
fn newell_vector(verts: &[Point3]) -> Vec3 {
    let Some(&origin) = verts.first() else {
        return Vec3::new(0.0, 0.0, 0.0);
    };
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let local: Vec<Vec3> = verts.iter().map(|p| *p - origin).collect();
    for (a, b) in local.iter().zip(local.iter().cycle().skip(1)) {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A cap bigger than a hemisphere, bounded by one circle, has its
    /// boundary's centroid over the far pole: its wire alone gives its side,
    /// so neither orientation warns.
    #[test]
    fn a_cap_past_its_hemisphere_raises_no_warning() {
        for reversed in [false, true] {
            let mut topo = Topology::default();
            let circle = Circle3D::new(
                Point3::new(0.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                3.0_f64.sqrt(),
            )
            .unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let surface = FaceSurface::Sphere(
                SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap(),
            );
            let face = if reversed {
                topo.add_face(Face::new_reversed(wire, vec![], surface))
            } else {
                topo.add_face(Face::new(wire, vec![], surface))
            };
            assert!(
                check_face_orientation(&topo, face).unwrap().is_empty(),
                "reversed {reversed}: warned"
            );
        }
    }
}
