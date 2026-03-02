//! glTF 2.0 binary (.glb) export.
//!
//! Exports tessellated B-Rep geometry as a glTF binary file suitable
//! for web viewers, game engines, and real-time 3D applications.

pub mod writer;

pub use writer::write_glb;
