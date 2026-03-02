//! Boolean operations on solids: fuse, cut, and intersect.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The type of boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union of two solids.
    Fuse,
    /// Subtraction: first minus second.
    Cut,
    /// Intersection: common volume.
    Intersect,
}

/// Perform a boolean operation on two solids.
///
/// # Errors
///
/// Returns an error if the operation fails (e.g., non-manifold result).
pub fn boolean(
    _topo: &mut Topology,
    _op: BooleanOp,
    _a: SolidId,
    _b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    Err(crate::OperationsError::InvalidInput {
        reason: "boolean operations not yet implemented".into(),
    })
}
