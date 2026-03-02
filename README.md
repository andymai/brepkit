# brepkit

The B-Rep modeling engine behind [brepjs](https://github.com/andymai/brepjs), written in Rust and compiled to WebAssembly.

## Overview

brepkit is the computational backend that powers brepjs. It handles the heavy
lifting — NURBS geometry, boolean operations, filleting, tessellation, and data
exchange — in memory-safe Rust with first-class WASM support. brepjs provides
the developer-facing TypeScript API; brepkit provides the engine underneath.

## Architecture

brepkit is organized as a layered Cargo workspace:

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
