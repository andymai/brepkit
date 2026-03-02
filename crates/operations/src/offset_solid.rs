//! Full 3D solid offset (parallel shell).
//!
//! Offsets all faces of a solid by a uniform distance along their normals.
//! Equivalent to `BRepOffsetAPI_MakeOffsetShape` in `OpenCascade`.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;
use crate::boolean::face_vertices;
use crate::dot_normal_point;

/// Offset all faces of a solid by a uniform distance.
///
/// Positive distance offsets outward (solid grows), negative inward (solid shrinks).
/// Currently supports planar faces only — NURBS faces require surface offset
/// with self-intersection removal.
///
/// # Errors
///
/// Returns an error if the solid contains non-planar faces or the offset
/// distance causes the solid to collapse.
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    let tol = Tolerance::new();

    if tol.approx_eq(distance, 0.0) {
        return Err(OperationsError::InvalidInput {
            reason: "offset distance must be non-zero".into(),
        });
    }

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut offset_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let (normal, _d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                return Err(OperationsError::InvalidInput {
                    reason: "solid offset currently only supports planar faces".into(),
                });
            }
        };

        let verts = face_vertices(topo, fid)?;

        // Offset each vertex along the face normal
        let offset_verts: Vec<Point3> = verts.iter().map(|&v| v + normal * distance).collect();

        let new_d = dot_normal_point(normal, offset_verts[0]);
        offset_faces.push((offset_verts, normal, new_d));
    }

    crate::boolean::assemble_solid(topo, &offset_faces, tol)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn offset_box_outward() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let original_vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.5).unwrap();
        let offset_vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        assert!(
            offset_vol > original_vol,
            "outward offset should increase volume: {offset_vol} > {original_vol}"
        );
    }

    #[test]
    fn offset_box_inward() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();

        let original_vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        let offset = offset_solid(&mut topo, solid, -0.5).unwrap();
        let offset_vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        assert!(
            offset_vol < original_vol,
            "inward offset should decrease volume: {offset_vol} < {original_vol}"
        );
    }

    #[test]
    fn offset_zero_error() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        assert!(offset_solid(&mut topo, solid, 0.0).is_err());
    }

    #[test]
    fn offset_preserves_face_count() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.3).unwrap();

        let shell = topo
            .shell(topo.solid(offset).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            6,
            "offset box should still have 6 faces"
        );
    }
}
