//! Mirror operation: reflect a solid across a plane.
//!
//! Creates a new copy of the solid reflected across the specified plane.

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::transform::transform_solid;

/// Mirror a solid across a plane defined by a point and normal.
///
/// Creates a new solid that is a reflection of the input solid across
/// the given plane. The original solid is not modified.
///
/// # Algorithm
///
/// Constructs a Householder reflection matrix:
///   `M = I - 2 * n * nᵀ` (for reflection through the origin)
/// with appropriate translation for a plane not passing through the origin.
///
/// Note: mirroring flips the handedness, so face normals are reversed.
/// The transform operation handles this via the inverse-transpose normal
/// transformation.
///
/// # Errors
///
/// Returns an error if the plane normal is zero-length or the solid is invalid.
pub fn mirror(
    topo: &mut Topology,
    solid: SolidId,
    plane_point: Point3,
    plane_normal: Vec3,
) -> Result<SolidId, crate::OperationsError> {
    let normal = plane_normal.normalize()?;

    // First, copy the solid (we need a new independent copy).
    let new_solid = crate::copy::copy_solid(topo, solid)?;

    // Build the reflection matrix.
    // For a plane through point P with normal n:
    //   Reflect(x) = x - 2 * ((x - P) · n) * n
    // This is equivalent to: translate(-P), reflect through origin, translate(P).
    //
    // The reflection matrix through the origin with normal n is:
    //   | 1-2nx²   -2nxny  -2nxnz  0 |
    //   | -2nxny  1-2ny²   -2nynz  0 |
    //   | -2nxnz  -2nynz  1-2nz²   0 |
    //   |   0        0        0     1 |
    let nx = normal.x();
    let ny = normal.y();
    let nz = normal.z();

    // d = P · n (signed distance from origin to plane)
    let d = nx.mul_add(
        plane_point.x(),
        ny.mul_add(plane_point.y(), nz * plane_point.z()),
    );

    // Build the full reflection matrix including translation.
    // T(P) * Reflect * T(-P) = Reflect + 2d * n column
    let mat = Mat4([
        [
            (2.0 * nx).mul_add(-nx, 1.0),
            -2.0 * nx * ny,
            -2.0 * nx * nz,
            2.0 * d * nx,
        ],
        [
            -2.0 * nx * ny,
            (2.0 * ny).mul_add(-ny, 1.0),
            -2.0 * ny * nz,
            2.0 * d * ny,
        ],
        [
            -2.0 * nx * nz,
            -2.0 * ny * nz,
            (2.0 * nz).mul_add(-nz, 1.0),
            2.0 * d * nz,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    transform_solid(topo, new_solid, &mat)?;

    Ok(new_solid)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn mirror_box_across_yz_plane() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Mirror across YZ plane (x=0).
        let mirrored = mirror(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        // The mirrored solid should have the same volume.
        let vol_orig = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let vol_mirror = crate::measure::solid_volume(&topo, mirrored, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol_orig, vol_mirror),
            "mirrored volume should match: {vol_orig} vs {vol_mirror}"
        );
    }

    #[test]
    fn mirror_preserves_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let mirrored = mirror(
            &mut topo,
            solid,
            Point3::new(5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let vol = crate::measure::solid_volume(&topo, mirrored, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 24.0),
            "mirrored box should have volume ~24.0, got {vol}"
        );
    }

    #[test]
    fn mirror_creates_new_solid() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mirrored = mirror(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        // Should be a different solid ID.
        assert_ne!(
            solid.index(),
            mirrored.index(),
            "mirror should create a new solid"
        );
    }

    #[test]
    fn mirror_zero_normal_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let result = mirror(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        );
        assert!(result.is_err());
    }
}
