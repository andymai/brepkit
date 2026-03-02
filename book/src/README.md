# brepkit

The B-Rep modeling engine behind [brepjs](https://github.com/andymai/brepjs),
written in Rust and compiled to WebAssembly.

brepkit is the computational backend that powers brepjs. It handles NURBS
geometry, boolean operations, filleting, tessellation, and data exchange — in
memory-safe Rust with first-class WASM support.

## Why brepkit?

- **Pure Rust** — no C/C++ dependencies, no complex build systems
- **WASM-first** — designed for browser and Node.js environments
- **Memory-safe** — no undefined behavior, no use-after-free
- **Layered architecture** — clean separation of math, topology, operations, and I/O
- **Modern tooling** — strict linting, property-based testing, comprehensive CI
