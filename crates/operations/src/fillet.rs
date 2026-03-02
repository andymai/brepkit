//! Edge filleting (rounding edges with a constant or variable radius).

use brepkit_topology::{edge::EdgeId, solid::SolidId};

/// Fillet one or more edges of a solid.
///
/// # Errors
///
/// Returns an error if any edge is not part of the solid.
pub fn fillet(
    _solid: SolidId,
    _edges: &[EdgeId],
    _radius: f64,
) -> Result<SolidId, crate::OperationsError> {
    todo!("fillet not yet implemented")
}
