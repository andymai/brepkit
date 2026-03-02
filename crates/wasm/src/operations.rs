//! JS-facing modeling operations via `wasm-bindgen`.
//!
//! Individual operations (extrude, transform, tessellate) are currently
//! exposed as methods on [`super::kernel::BrepKernel`].
//!
//! This module is reserved for future standalone operation bindings
//! (e.g. boolean operations) that may not fit the kernel pattern.
