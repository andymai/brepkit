//! Domain-aligned binding modules.
//!
//! Each module adds `#[wasm_bindgen] impl BrepKernel { ... }` methods
//! for a specific domain. wasm-bindgen supports multiple impl blocks
//! across files.

pub mod assembly;
pub mod batch;
pub mod booleans;
pub mod checkpoint;
pub mod heal;
#[cfg(feature = "io")]
pub mod io;
pub mod measure;
pub mod nurbs;
pub mod operations;
pub mod polygon2d;
pub mod primitives;
pub mod query;
pub mod shapes;
pub mod sketch;
pub mod tessellate;
pub mod transforms;
