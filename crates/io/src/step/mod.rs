//! STEP (ISO 10303) data exchange.

mod assembly;
pub mod reader;
mod revolution;
pub mod writer;

pub use reader::read_step;
pub use writer::write_step;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_faces, solid_vertices};
use brepkit_topology::solid::SolidId;

use crate::IoError;

/// The `FILE_DESCRIPTION` brepkit writes.
const EXPORT_DESCRIPTION: &str = "brepkit STEP export";

/// Marks a brepkit export whose face bounds follow ISO 10303-42 (a bound runs
/// about the face's normal). Earlier exports wrote a reversed face's bounds
/// about its surface's normal.
const ISO_FACE_BOUNDS: &str = "face bounds per ISO 10303-42";

/// Marks a brepkit export whose cones carry ISO 10303-42 semi-angles, from the
/// axis, placed at their apex. Earlier exports wrote the angle from the plane
/// across the axis.
const ISO_CONE_ANGLES: &str = "cone semi-angles per ISO 10303-42";

/// Gives each vertex of `solids` the linear tolerance, in the millimetres
/// the solids are placed in, raised where one of its edges' curves ends off
/// it. The reader builds each vertex with the linear tolerance in the
/// file's unit, and a placement scales the tolerance with the unit.
fn settle_tolerances(topo: &mut Topology, solids: &[SolidId]) -> Result<(), IoError> {
    let linear = Tolerance::new().linear;
    for &solid in solids {
        for vertex in solid_vertices(topo, solid)? {
            topo.vertex_mut(vertex)?.set_tolerance(linear);
        }
        let faces = solid_faces(topo, solid)?;
        brepkit_operations::heal::cover_curve_end_gaps(topo, &faces)?;
    }
    Ok(())
}
