# brepkit

Solid modeling kernel for Rust and WebAssembly.

[![CI](https://github.com/andymai/brepkit/actions/workflows/ci.yml/badge.svg)](https://github.com/andymai/brepkit/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/brepkit-wasm)](https://www.npmjs.com/package/brepkit-wasm)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)

**[Architecture](#architecture)** · **[Performance](#performance)** · **[Examples](#common-patterns)** · **[Contributing](./CONTRIBUTING.md)**

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

## Overview

Solids are boundary representations (B-Rep): faces, edges, and vertices with exact NURBS geometry, not triangle meshes. Booleans, fillets, and STEP export operate on exact curves and surfaces.

- No `unsafe`, no `unwrap`, no `panic`. All operations return `Result`
- Compiles to WebAssembly (browser/Node.js) and runs natively with rayon parallelism
- 1,200+ tests (proptest, golden files, integration), 60% minimum coverage enforced in CI

## Architecture

Layered Cargo workspace. Each layer depends only on layers below it. Boundaries are enforced by `scripts/check-boundaries.sh` and checked in CI.

| Layer | Crate | What it does |
|-------|-------|-------------|
| L0 | `brepkit-math` | Points, vectors, matrices, NURBS curves/surfaces, geometric predicates, CDT, convex hull |
| L1 | `brepkit-topology` | Arena-allocated B-Rep: vertex, edge, wire, face, shell, solid. Half-edge adjacency graph |
| L2 | `brepkit-operations` | Booleans, fillet, chamfer, extrude, revolve, sweep, loft, shell, measure (40 modules) |
| L2 | `brepkit-io` | Import/export for 7 formats (2 B-Rep, 5 mesh). Uses operations for tessellation during mesh export |
| L3 | `brepkit-wasm` | JavaScript API via wasm-bindgen |

## Features

**Primitives:** box, cylinder, cone, sphere, torus

**Booleans:** union, cut, intersect with analytic surface preservation. Mesh-based (co-refinement) and NURBS-based variants

**Shape modifications:** extrude, revolve, sweep, loft, pipe, helical sweep, chamfer, fillet (constant + variable radius), shell, draft, offset, thicken, mirror, pattern

**Sectioning:** cross-section curves, split by plane/surface

**Measurement:** bounding box, area, volume, center of mass, distance, point classification (in/on/out)

**Geometry:** NURBS evaluation, surface-surface intersection, curve fitting (LSPIA), point projection, self-intersection detection

**Tessellation:** parallel CDT for planar faces, snap tessellation for analytic surfaces

**Repair:** healing, defeaturing, sewing, validation, face filling (Coons patch)

**Sketching:** 2D constraint solver for sketch-driven modeling

**Assemblies:** hierarchical product structure with transforms, flattening, BOM

**Evolution:** face provenance tracking through booleans and modeling operations

## Data Exchange

| Format | Type | Import | Export |
|--------|------|--------|--------|
| STEP | B-Rep | ✓ | ✓ |
| IGES | B-Rep | ✓ | ✓ |
| STL | Mesh | ✓ | ✓ |
| 3MF | Mesh | ✓ | ✓ |
| OBJ | Mesh | ✓ | ✓ |
| PLY | Mesh | ✓ | ✓ |
| glTF | Mesh | ✓ | ✓ |

B-Rep formats preserve exact geometry. Mesh formats export tessellated triangles.

## Performance

Compound boolean staircase (9 sequential union/cut operations):

| Benchmark | brepkit (WASM) | OCCT (WASM) | brepkit (native) |
|-----------|---------------|-------------|------------------|
| 9-step boolean staircase | 281 ms | 3,800 ms | 40 ms |
| Result face count | 72 | ~7,000 | 72 |

The lower face count comes from analytic surface preservation: booleans keep cylinders and planes as exact surfaces instead of tessellating to triangles.

- **Tessellation** (64-hole plate): 29 ms with parallel CDT

> See `crates/operations/benches/` for reproduction. Native benchmarks use rayon; WASM is single-threaded.

## Common Patterns

### Measure a solid

```rust
use brepkit_operations::measure::{solid_volume, solid_bounding_box};

let vol = solid_volume(&topo, result, 0.1)?;
let bbox = solid_bounding_box(&topo, result)?;
println!("Volume: {vol:.2} mm³, bounds: {bbox:?}");
```

### Export to STEP

```rust
use brepkit_io::step::writer::write_step;

let step_string = write_step(&topo, &[result])?;
std::fs::write("output.step", step_string)?;
```

### Import from STEP

```rust
use brepkit_io::step::reader::read_step;

let step_data = std::fs::read_to_string("input.step")?;
let solids = read_step(&step_data, &mut topo)?;
```

### Error handling

All operations return `Result`. Errors are typed per crate:

```rust
use brepkit_operations::OperationsError;

match boolean(&mut topo, BooleanOp::Fuse, a, b) {
    Ok(fused) => { /* use the result */ }
    Err(OperationsError::InvalidInput { reason }) => eprintln!("Bad input: {reason}"),
    Err(e) => eprintln!("Operation failed: {e}"),
}
```

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

Using brepkit? [Open a PR](https://github.com/andymai/brepkit/pulls) to add your project.

## License

[AGPL-3.0-only](./LICENSE)
