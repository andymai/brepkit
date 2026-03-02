# brepkit

A modern, pure-Rust B-Rep CAD kernel compiled to WebAssembly.

brepkit provides boundary representation solid modeling with NURBS-native
geometry, boolean operations, filleting, and data exchange — all in
memory-safe Rust with first-class WASM support.

## Why brepkit?

- **Pure Rust** — no C/C++ dependencies, no complex build systems
- **WASM-first** — designed for browser and Node.js environments
- **Memory-safe** — no undefined behavior, no use-after-free
- **Layered architecture** — clean separation of math, topology, operations, and I/O
- **Modern tooling** — strict linting, property-based testing, comprehensive CI
