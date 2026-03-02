//! Thicken a face into a solid by offsetting along its normal.
//!
//! A convenience operation that extrudes a face along its own normal
//! direction. Equivalent to a specialized form of
//! `BRepOffsetAPI_MakeOffsetShape` in `OpenCascade`.

use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::extrude::extrude;

/// Thicken a planar face into a solid by extruding along its normal.
///
/// Positive `thickness` extrudes in the face normal direction;
/// negative extrudes in the opposite direction.
///
/// # Errors
///
/// Returns an error if `thickness` is zero, the face is not planar,
/// or the extrusion fails.
pub fn thicken(
    topo: &mut Topology,
    face: FaceId,
    thickness: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = brepkit_math::tolerance::Tolerance::new();

    if tol.approx_eq(thickness, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "thickness must be non-zero".into(),
        });
    }

    let face_data = topo.face(face)?;
    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Nurbs(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "thicken of NURBS faces is not supported".into(),
            });
        }
    };

    let (direction, distance) = if thickness > 0.0 {
        (normal, thickness)
    } else {
        (-normal, -thickness)
    };

    extrude(topo, face, direction, distance)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    #[test]
    fn thicken_positive() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = thicken(&mut topo, face, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 2.0),
            "1×1 face thickened by 2 should have volume ~2.0, got {vol}"
        );
    }

    #[test]
    fn thicken_negative() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = thicken(&mut topo, face, -1.5).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.5),
            "1×1 face thickened by -1.5 should have volume ~1.5, got {vol}"
        );
    }

    #[test]
    fn thicken_zero_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        assert!(thicken(&mut topo, face, 0.0).is_err());
    }
}
