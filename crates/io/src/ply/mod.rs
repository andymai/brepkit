//! PLY (Polygon File Format / Stanford Triangle Format) I/O.
//!
//! Supports ASCII and binary little-endian variants for triangle meshes.

pub mod reader;
pub mod writer;

pub use reader::read_ply;
pub use writer::write_ply;
