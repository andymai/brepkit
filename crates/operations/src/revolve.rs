//! Revolution of a profile around an axis.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::{face::FaceId, solid::SolidId};

/// Revolve a face around an axis to produce a solid.
///
/// # Errors
///
/// Returns an error if the angle is out of range or the axis is zero-length.
pub fn revolve(
    _face: FaceId,
    _axis_origin: Point3,
    _axis_direction: Vec3,
    _angle_radians: f64,
) -> Result<SolidId, crate::OperationsError> {
    todo!("revolve not yet implemented")
}
