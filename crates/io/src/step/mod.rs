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
