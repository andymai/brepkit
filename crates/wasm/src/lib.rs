//! # brepkit-wasm
//!
//! WebAssembly bindings for brepkit via `wasm-bindgen`.
//!
//! This is layer L3, the public API surface for JavaScript/TypeScript consumers.
//!
//! The primary entry point is [`kernel::BrepKernel`], which owns all modeling
//! state and exposes shape creation, operations, and tessellation to JS.

pub mod error;
pub mod io;
pub mod kernel;
pub mod operations;
pub mod shapes;
