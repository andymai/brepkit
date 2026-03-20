# brepkit

Solid modeling kernel for Rust and WebAssembly.

[![CI](https://github.com/andymai/brepkit/actions/workflows/ci.yml/badge.svg)](https://github.com/andymai/brepkit/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/brepkit-wasm)](https://www.npmjs.com/package/brepkit-wasm)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)

**[Architecture](#architecture)** · **[Performance](#performance)** · **[Getting Started](#getting-started)** · **[Contributing](./CONTRIBUTING.md)**

```rust
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::boolean::{boolean, BooleanOp};
use brepkit_operations::measure::solid_volume;
use brepkit_io::step::writer::write_step;
use brepkit_topology::Topology;

let mut topo = Topology::new();

// Create a block with a cylindrical hole
let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
let hole = make_cylinder(&mut topo, 5.0, 15.0)?;
let drilled = boolean(&mut topo, BooleanOp::Cut, block, hole)?;

// Measure and export
let vol = solid_volume(&topo, drilled, 0.1)?;
let step = write_step(&topo, &[drilled])?;
```

## Why build a CAD kernel?

Building a solid modeling kernel from scratch is one of those problems measured in developer-careers, not sprints. Parasolid, ACIS, and OpenCascade represent decades of accumulated effort — surface-surface intersection, robust boolean classification, exact geometric predicates, degenerate case handling. Each is a research domain in its own right.

brepkit grew out of building [gridfinitylayouttool.com](https://gridfinitylayouttool.com). I needed parametric CAD in the browser and the existing options were either proprietary or compiled-from-C++ with bridge overhead. OpenCascade works — [brepjs](https://github.com/andymai/brepjs) proves that — but performance ceilings and an 11 MB WASM bundle kept getting in the way.

brepkit is the answer to a simple question: what if the kernel was built for WebAssembly from day one? Rust's type system, no `unsafe`, no `unwrap`, no `panic` — all operations return `Result`. The WASM bundle is 1.8 MB.

## Status

Core operations work and are fast: primitives, booleans, fillets, chamfers, extrude, revolve, sweep, loft, pipe, STEP I/O. The math layer — NURBS evaluation, exact geometric predicates, all analytic surface intersections — is mature. Booleans preserve analytic surfaces (cylinders stay cylinders) and handle compound operations efficiently.

Still maturing: torus booleans fall back to the tessellated path, revolve/sweep/loft/pipe require planar profiles (extrude handles NURBS), and IGES round-trips lose analytic surface types.

For production use today, [brepjs](https://github.com/andymai/brepjs) with the OpenCascade kernel is the stable path. brepkit is the faster replacement in progress — the kernel abstraction layer in brepjs means switching is a one-line change.

## Features

**Modeling** — box, cylinder, cone, sphere, torus plus extrude, revolve, sweep, loft, pipe, helical sweep

**Booleans** — union, cut, intersect with analytic surface preservation

**Modifiers** — fillet (constant + variable radius), chamfer, shell, draft, offset, thicken, mirror, pattern

**Sectioning** — cross-section curves, split by plane or surface

**Measurement** — bounding box, area, volume, center of mass, distance, point classification (in/on/out)

**Geometry** — NURBS evaluation, surface-surface intersection, curve fitting (LSPIA), point projection, self-intersection detection

**Tessellation** — parallel CDT for planar faces, snap tessellation for analytic surfaces

**Repair** — healing, defeaturing, sewing, validation, face filling (Coons patch)

**Feature Recognition** — automatic detection of holes, pockets, slots, bosses, and ribs

**Sketching** — 2D constraint solver for sketch-driven modeling

**Assemblies** — hierarchical product structure with transforms, flattening, BOM

**Evolution** — face provenance tracking through booleans and modeling operations

## Architecture

Layered Cargo workspace. Each layer depends only on layers below it, with one exception: `brepkit-io` also uses `brepkit-operations` for tessellation during mesh export. Boundaries are enforced by `scripts/check-boundaries.sh` and checked in CI.

| Layer | Crate | What it does |
|-------|-------|-------------|
| L0 | `brepkit-math` | Points, vectors, matrices, NURBS curves/surfaces, geometric predicates, CDT, convex hull |
| L1 | `brepkit-topology` | Arena-allocated B-Rep: vertex, edge, wire, face, shell, solid. Half-edge adjacency graph |
| L2 | `brepkit-operations` | Booleans, fillet, chamfer, extrude, revolve, sweep, loft, shell, measure |
| L2 | `brepkit-io` | Import/export for 7 formats (2 B-Rep, 5 mesh). Uses operations for tessellation during mesh export |
| L3 | `brepkit-wasm` | JavaScript API via wasm-bindgen |

## Performance

Median times from the [brepjs benchmark suite](https://github.com/andymai/brepjs/tree/main/benchmarks) (5 iterations, Node.js, Linux x86_64). WASM is single-threaded; native benchmarks use criterion with rayon.

| Operation | brepkit (WASM) | OCCT (WASM) | Speedup | brepkit (native) |
|-----------|---------------|-------------|---------|-----------------|
| fuse(box, box) | 5.7 ms | 83.7 ms | 15x | 336 µs |
| cut(box, cylinder) | 4.2 ms | 123.8 ms | 29x | 221 µs |
| intersect(box, sphere) | 31.9 ms | 107.1 ms | 3.4x | 2.4 ms |
| box + chamfer | 0.1 ms | 7.8 ms | 78x | 55 µs |
| box + fillet | 0.3 ms | 8.1 ms | 27x | 75 µs |
| multi-boolean (16 holes) | 1.7 ms | 52.0 ms | 31x | 1.2 ms |
| mesh sphere (tol=0.01) | 20.0 ms | 61.3 ms | 3.1x | 1.8 ms |
| exportSTEP (×10) | 0.9 ms | 19.2 ms | 21x | — |

Booleans preserve analytic surfaces — cylinders and planes stay as exact geometry, keeping face counts low (72 vs ~7,000 for a 9-step compound boolean).

> Native benchmarks: `cargo bench -p brepkit-operations`. WASM comparison: `brepjs/benchmarks/`.

## Data Exchange

| Format | Type | Import | Export |
|--------|------|--------|--------|
| STEP | B-Rep | ✓ | ✓ |
| IGES | B-Rep | ✓ | ✓* |
| STL | Mesh | ✓ | ✓ |
| 3MF | Mesh | ✓ | ✓ |
| OBJ | Mesh | ✓ | ✓ |
| PLY | Mesh | ✓ | ✓ |
| glTF | Mesh | ✓ | ✓ |

STEP preserves exact geometry on round-trip. *IGES export converts analytic surfaces to NURBS. Mesh formats export tessellated triangles.

## Getting Started

### As a Rust dependency

```toml
[dependencies]
brepkit-math = { git = "https://github.com/andymai/brepkit" }
brepkit-topology = { git = "https://github.com/andymai/brepkit" }
brepkit-operations = { git = "https://github.com/andymai/brepkit" }
brepkit-io = { git = "https://github.com/andymai/brepkit" }        # optional
```

### As a WASM package

```bash
npm install brepkit-wasm
```

```js
import init, { BrepKernel } from "brepkit-wasm";

await init();
const kernel = new BrepKernel();
const solid = kernel.makeBox(10, 20, 30);
```

For a higher-level TypeScript API, see [brepjs](https://github.com/andymai/brepjs).

### Building from source

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all

# WASM (full)
cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release

# WASM (smaller, no IO)
cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release --no-default-features
```

## Projects Using brepkit

- [brepjs](https://github.com/andymai/brepjs) - CAD modeling for JavaScript
- [Gridfinity Layout Tool](https://github.com/andymai/gridfinity-layout-tool) - Web-based Gridfinity storage layout generator

[Open a PR](https://github.com/andymai/brepkit/pulls) to add your project.

## License

Licensed under either of

- [Apache License, Version 2.0](./LICENSE-APACHE)
- [MIT License](./LICENSE-MIT)

at your option.
