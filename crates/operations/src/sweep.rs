//! Path sweep: sweep a profile along a curve.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Sweep a face along a path curve to produce a solid.
///
/// # Errors
///
/// Returns an error if the path or profile is invalid.
pub fn sweep(
    _topo: &mut Topology,
    _profile: FaceId,
    _path: &NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    Err(crate::OperationsError::InvalidInput {
        reason: "sweep not yet implemented".into(),
    })
}
