# brepkit

A modern, pure-Rust B-Rep CAD kernel compiled to WebAssembly.

## Overview

brepkit is a boundary representation (B-Rep) solid modeling kernel designed as a
modern alternative to legacy C++ CAD kernels. It provides NURBS-native geometry,
boolean operations, filleting, and data exchange — all in memory-safe Rust with
first-class WASM support.

## Architecture

The kernel is organized as a layered Cargo workspace:

| Layer | Crate | Purpose |
|-------|-------|---------|
| L0 | `brepkit-math` | Vectors, matrices, NURBS, geometric predicates |
| L1 | `brepkit-topology` | B-Rep data structures (vertex, edge, face, solid) |
| L2 | `brepkit-operations` | Boolean ops, fillets, extrusion, tessellation |
| L2 | `brepkit-io` | STEP and 3MF import/export |
| L3 | `brepkit-wasm` | WebAssembly bindings via wasm-bindgen |

## Getting Started

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Build WASM bindings
cargo build -p brepkit-wasm --target wasm32-unknown-unknown
```

## License

Apache-2.0
