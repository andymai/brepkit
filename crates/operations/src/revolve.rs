//! Revolution of a profile around an axis.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Revolve a face around an axis to produce a solid.
///
/// # Errors
///
/// Returns an error if the angle is out of range or the axis is zero-length.
pub fn revolve(
    _topo: &mut Topology,
    _face: FaceId,
    _axis_origin: Point3,
    _axis_direction: Vec3,
    _angle_radians: f64,
) -> Result<SolidId, crate::OperationsError> {
    Err(crate::OperationsError::InvalidInput {
        reason: "revolve not yet implemented".into(),
    })
}
