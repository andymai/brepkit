//! # brepkit-wasm
//!
//! WebAssembly bindings for brepkit via `wasm-bindgen`.
//!
//! This is layer L3, the public API surface for JavaScript/TypeScript consumers.
//!
//! The primary entry point is [`kernel::BrepKernel`], which owns all modeling
//! state and exposes shape creation, operations, and tessellation to JS.

mod bindings;
pub mod error;
mod handles;
mod helpers;
pub mod kernel;
pub mod shapes;
mod state;
mod types;
