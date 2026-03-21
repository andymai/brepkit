//! glTF 2.0 binary (.glb) import and export.
//!
//! Imports and exports tessellated B-Rep geometry as glTF binary files
//! suitable for web viewers, game engines, and real-time 3D applications.

pub mod reader;
pub mod writer;

pub use reader::{read_glb, read_glb_solid};
pub use writer::write_glb;
