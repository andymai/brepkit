//! IGES (Initial Graphics Exchange Specification) import and export.
//!
//! Supports basic B-Rep geometry entities: lines, NURBS curves,
//! planes, and NURBS surfaces.

pub mod reader;
pub mod writer;

pub use reader::read_iges;
pub use writer::write_iges;
