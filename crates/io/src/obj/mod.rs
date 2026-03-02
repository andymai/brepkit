//! OBJ (Wavefront) mesh import and export.
//!
//! OBJ is the simplest widely-supported 3D mesh format.
//! Supports vertex positions, normals, and triangle/polygon faces.

pub mod reader;
pub mod writer;

pub use reader::read_obj;
pub use writer::write_obj;
