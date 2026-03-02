//! Linear extrusion of faces/wires along a direction vector.

use brepkit_math::vec::Vec3;
use brepkit_topology::{face::FaceId, solid::SolidId};

/// Extrude a face along a direction to produce a solid.
///
/// # Errors
///
/// Returns an error if the face is invalid or the direction is zero-length.
pub fn extrude(
    _face: FaceId,
    _direction: Vec3,
    _distance: f64,
) -> Result<SolidId, crate::OperationsError> {
    todo!("extrude not yet implemented")
}
