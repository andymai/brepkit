//! STEP (ISO 10303) data exchange.

pub mod reader;
pub mod writer;

pub use reader::read_step;
pub use writer::write_step;

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
