//! V2 offset operations delegating to brepkit-offset.

use brepkit_offset::{JointType, OffsetError, OffsetOptions};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Map an `OffsetError` to the most appropriate `OperationsError` variant,
/// preserving structured error information where possible.
fn map_offset_error(e: OffsetError) -> OperationsError {
    match e {
        OffsetError::Topology(t) => OperationsError::Topology(t),
        OffsetError::Math(m) => OperationsError::Math(m),
        OffsetError::Algo(a) => OperationsError::Algo(a),
        other => OperationsError::InvalidInput {
            reason: format!("{other}"),
        },
    }
}

/// Offset all faces of a solid (V2 pipeline).
///
/// # Errors
///
/// Returns an error if the offset fails.
pub fn offset_solid_v2(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    brepkit_offset::offset_solid(topo, solid, distance, OffsetOptions::default())
        .map_err(map_offset_error)
}

/// Shell (hollow solid) operation (V2 pipeline).
///
/// # Errors
///
/// Returns an error if the offset fails.
pub fn shell_v2(
    topo: &mut Topology,
    solid: SolidId,
    thickness: f64,
    exclude: &[FaceId],
) -> Result<SolidId, OperationsError> {
    brepkit_offset::thick_solid(topo, solid, thickness, exclude, OffsetOptions::default())
        .map_err(map_offset_error)
}

/// Offset with arc joints (V2 pipeline).
///
/// # Errors
///
/// Returns an error if the offset fails.
pub fn offset_solid_arc_v2(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    let options = OffsetOptions {
        joint: JointType::Arc,
        ..Default::default()
    };
    brepkit_offset::offset_solid(topo, solid, distance, options).map_err(map_offset_error)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use brepkit_topology::Topology;

    #[test]
    fn offset_v2_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let result = offset_solid_v2(&mut topo, solid, 0.5).unwrap();
        let shell = topo
            .shell(topo.solid(result).unwrap().outer_shell())
            .unwrap();
        assert_eq!(shell.faces().len(), 6);
    }
}
