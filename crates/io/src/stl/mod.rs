//! STL (stereolithography) import and export.
//!
//! Supports both binary and ASCII STL formats.
//! STL is the most common format for 3D printing.

pub mod import;
pub mod reader;
pub mod writer;

pub use import::import_mesh;
pub use reader::read_stl;
pub use writer::write_stl;
