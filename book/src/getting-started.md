# Getting Started

## Prerequisites

- [Rust](https://rustup.rs/) (stable, edition 2024)
- [wasm-bindgen CLI](https://rustwasm.github.io/wasm-bindgen/) for WASM builds
- [Node.js](https://nodejs.org/) 20+ for TypeScript bindings

## Building

```bash
# Clone the repository
git clone https://github.com/andymai/brepkit.git
cd brepkit

# Build all Rust crates
cargo build --workspace

# Run tests
cargo test --workspace

# Build WASM target
cargo build -p brepkit-wasm --target wasm32-unknown-unknown
```

## Using from TypeScript

```bash
cd bindings/ts
npm install
npm run build
```

```typescript
import { initBrepkit } from '@brepkit/wasm';

await initBrepkit();
```

## Development

```bash
# Install development tooling
npm install          # Husky hooks, commitlint
cargo install cargo-deny cargo-llvm-cov  # CI tools

# Format and lint
cargo fmt --all
cargo clippy --all-targets

# Check crate boundaries
./scripts/check-boundaries.sh
```
