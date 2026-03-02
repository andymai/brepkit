//! Affine transforms applied to topological shapes.

use brepkit_math::mat::Mat4;
use brepkit_topology::solid::SolidId;

/// Apply an affine transform to a solid.
///
/// # Errors
///
/// Returns an error if the transform is degenerate (zero determinant).
pub fn transform_solid(_solid: SolidId, _matrix: &Mat4) -> Result<SolidId, crate::OperationsError> {
    todo!("transform not yet implemented")
}
