<div align="center">

# brepkit

Solid modeling kernel for Rust and WebAssembly.

[![CI](https://github.com/andymai/brepkit/actions/workflows/ci.yml/badge.svg)](https://github.com/andymai/brepkit/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/brepkit-wasm)](https://www.npmjs.com/package/brepkit-wasm)
[![Last release](https://img.shields.io/github/release-date/andymai/brepkit?label=last%20release)](https://github.com/andymai/brepkit/releases)
[![Commit activity](https://img.shields.io/github/commit-activity/m/andymai/brepkit?label=commits%2Fmonth)](https://github.com/andymai/brepkit/commits/main)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/) [![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

**[Architecture](#architecture)** · **[Performance](#performance)** · **[Getting Started](#getting-started)** · **[Contributing](./CONTRIBUTING.md)**

</div>

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

brepkit is a from-scratch B-Rep kernel in Rust targeting WebAssembly. `unsafe` is forbidden, `unwrap` and `panic` are denied by default. Every public operation returns `Result`.

The project grew out of building [gridfinitylayouttool.com](https://gridfinitylayouttool.com), where the existing options for parametric CAD in the browser were proprietary or compiled from C++.

## Status

brepkit is in active development. Core modeling works. Some areas are still maturing.

| Category                | Feature                                                        | Status  |
| ----------------------- | -------------------------------------------------------------- | ------- |
| **Primitives**          | Box, cylinder, cone, sphere, torus                             | Stable  |
| **Booleans**            | Union, cut, intersect (plane, cylinder, cone, sphere, NURBS)   | Stable  |
| **Booleans**            | Torus surface booleans                                         | Planned |
| **Modifiers**           | Fillet (constant + variable radius), chamfer, shell, draft     | Stable  |
| **Modifiers**           | Offset face, offset solid, thicken, mirror, pattern            | Stable  |
| **Sweeps**              | Extrude (planar + NURBS profiles)                              | Stable  |
| **Sweeps**              | Revolve, sweep, loft, pipe (planar profiles only)              | Stable  |
| **Sweeps**              | Helical sweep                                                  | Stable  |
| **Sweeps**              | Non-planar profiles for revolve, sweep, loft, pipe             | Planned |
| **Sectioning**          | Cross-section curves, split by plane or surface                | Stable  |
| **Measurement**         | BBox, area, volume, center of mass, inertia tensor             | Stable  |
| **Measurement**         | Point-to-solid, solid-to-solid distance, point classification  | Stable  |
| **Geometry**            | NURBS evaluation, derivatives, knot ops, fitting, projection   | Stable  |
| **Geometry**            | Analytic surface intersections (plane, cylinder, cone, sphere) | Stable  |
| **Geometry**            | Curve-curve intersection (Bezier clipping)                     | Stable  |
| **Tessellation**        | Adaptive deflection, CDT, analytic surface optimization        | Stable  |
| **Repair**              | Shape healing (30+ wire/face/shell fixes), sewing, validation  | Stable  |
| **I/O**                 | STEP AP203 import/export (geometry-preserving round-trip)      | Stable  |
| **I/O**                 | STL, 3MF, OBJ, PLY, glTF import/export                         | Stable  |
| **I/O**                 | IGES import/export                                             | Beta    |
| **Sketching**           | 2D constraint solver                                           | Stable  |
| **Feature Recognition** | Holes, pockets, chamfers, fillets, patterns                    | Beta    |
| **Assemblies**          | Hierarchical structure, transforms, BOM                        | Beta    |
| **Evolution**           | Face provenance tracking through operations                    | Beta    |
| **Defeaturing**         | Remove specified faces/features from solid                     | Beta    |

## Scope

To set expectations, this project deliberately does not:

- **Render scenes or manage viewports** — brepkit produces geometry and tessellated meshes; rendering (camera, lighting, shading) is left to the caller (e.g. Three.js, wgpu).
- **Perform toolpath planning or slicing** — export STEP, STL, or 3MF and pass the output to a CAM tool or slicer (Bambu Studio, PrusaSlicer, etc.).
- **Support mesh-based modeling** — the kernel operates on exact B-Rep geometry; subdivision surfaces, polygon meshes, and voxel representations are out of scope.
- **Provide a GUI or interactive editor** — brepkit is a library; building a UI around it (like [gridfinitylayouttool.com](https://gridfinitylayouttool.com)) is the application's responsibility.
- **Simulate physics or perform FEA** — measurement (volume, CoM, inertia) is included, but stress analysis, collision detection, and dynamics are not.

## Roadmap

Broad directions, no dates.

- **Boolean completeness** — extend the unified General Fuse pipeline to all surface types including torus
- **Sweep generalization** — non-planar profile support for revolve, sweep, loft, and pipe
- **Parallel tessellation in WASM** — native builds already parallelize per-face meshing; bring it to the (single-threaded) WASM target via threads
- **Assembly metadata** — colors, layers, materials, and PMI for richer data exchange
- **Expanded I/O** — lossless IGES round-trips and broader STEP entity coverage
- **Documentation** — API reference, tutorials, and architectural guides

## Architecture

Layered Cargo workspace. Crates depend only on the same or lower layers. Boundaries are enforced by CI.

| Layer | Crate                | What it does                                                                                       |
| ----- | -------------------- | -------------------------------------------------------------------------------------------------- |
| L0    | `brepkit-math`       | Points, vectors, matrices, NURBS curves/surfaces, geometric predicates, CDT, convex hull           |
| L1    | `brepkit-geometry`   | Curve sampling (uniform, deflection, arc-length, curvature), extrema, analytic-to-NURBS conversion |
| L1    | `brepkit-topology`   | Arena-allocated B-Rep: vertex, edge, wire, face, shell, solid. Edge-to-face / face-neighbor adjacency index |
| L2    | `brepkit-algo`       | General Fuse Algorithm (GFA) boolean engine: pave filler, face classification, solid assembly      |
| L2    | `brepkit-blend`      | Walking-based fillet and chamfer with constant, variable, and custom radius laws                   |
| L2    | `brepkit-heal`       | Shape healing: 30+ fixes, analysis, sewing, tolerance management, configurable pipeline            |
| L2    | `brepkit-check`      | Point classification, validation (19 checks), properties (volume/area/CoM), distance queries       |
| L2    | `brepkit-offset`     | Solid offset and thickening via global face-face intersection                                      |
| L2    | `brepkit-sketch`     | 2D parametric constraint solver (GCS) with a DogLeg trust-region solver                            |
| L3    | `brepkit-operations` | Booleans, fillet, chamfer, extrude, revolve, sweep, loft, shell, offset, measure, tessellation     |
| L3    | `brepkit-io`         | Import/export: STEP, IGES, STL, 3MF, OBJ, PLY, glTF                                                |
| L4    | `brepkit-wasm`       | JavaScript API via wasm-bindgen with batch execution and checkpoint/restore                        |

## Performance

Median times from the [brepjs benchmark suite](https://github.com/andymai/brepjs/tree/main/benchmarks) (5 iterations, Node.js, Linux x86_64). WASM is single-threaded; native benchmarks use criterion.

| Operation                | brepkit (WASM) | OCCT (WASM) | Speedup | brepkit (native) |
| ------------------------ | -------------- | ----------- | ------- | ---------------- |
| fuse(box, box)           | 0.5 ms         | 43.3 ms     | 87x     | 122 µs           |
| cut(box, cylinder)       | 59.6 ms        | 72.0 ms     | 1.2x    | 24.1 ms          |
| intersect(box, sphere)   | 0.3 ms         | 62.6 ms     | 209x    | 104 µs           |
| box + chamfer            | 0.1 ms         | 5.6 ms      | 56x     | 44 µs            |
| box + fillet             | 0.3 ms         | 6.3 ms      | 21x     | 73 µs            |
| multi-boolean (16 holes) | 7.0 ms         | 31.2 ms     | 4.5x    | 4.1 ms           |
| mesh sphere (tol=0.01)   | 33.4 ms        | 49.7 ms     | 1.5x    | 1.5 ms           |
| exportSTEP (×10)         | 1.1 ms         | 18.6 ms     | 17x     | —                |

Booleans preserve analytic surfaces, keeping face counts low (72 vs ~7,000 for a 9-step compound boolean).

> OCCT comparison uses [occt-wasm](https://www.npmjs.com/package/occt-wasm), an OpenCASCADE build compiled to WebAssembly. Both kernels run single-threaded in Node.js. Boolean and `exportSTEP` rows are timed as batches of ten operations. Native benchmarks: `cargo bench -p brepkit-operations --bench cad_operations`. Full benchmark source: [brepjs/benchmarks](https://github.com/andymai/brepjs/tree/main/benchmarks). Measured 2026-06-23.

## Data Exchange

| Format     | Type  | Import | Export |
| ---------- | ----- | ------ | ------ |
| STEP AP203 | B-Rep | ✓      | ✓      |
| IGES       | B-Rep | ✓      | ✓\*    |
| STL        | Mesh  | ✓      | ✓      |
| 3MF        | Mesh  | ✓      | ✓      |
| OBJ        | Mesh  | ✓      | ✓      |
| PLY        | Mesh  | ✓      | ✓      |
| glTF       | Mesh  | ✓      | ✓      |

STEP preserves exact geometry on round-trip, including analytic surfaces (plane, cylinder, cone, sphere, torus). \*IGES export currently writes planar and NURBS surfaces; analytic surfaces (cylinder, cone, sphere, torus) are not yet exported, though their edges are. Mesh formats export tessellated triangles.

## Getting Started

### As a Rust dependency

Not yet published to crates.io. Use git dependencies for now:

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
import { BrepKernel } from 'brepkit-wasm';

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

# Generate API docs
cargo doc --workspace --no-deps --open
```

## Projects Using brepkit

- [brepjs](https://github.com/andymai/brepjs) — CAD modeling for JavaScript
- [Gridfinity Layout Tool](https://github.com/andymai/gridfinity-layout-tool) — Web-based Gridfinity storage layout generator

[Open a PR](https://github.com/andymai/brepkit/pulls) to add your project.

## License

Licensed under either of

- [Apache License, Version 2.0](./LICENSE-APACHE)
- [MIT License](./LICENSE-MIT)

at your option.
