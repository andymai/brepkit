<div align="center">

# brepkit

Solid modeling kernel for Rust and WebAssembly.

[![CI](https://github.com/andymai/brepkit/actions/workflows/ci.yml/badge.svg)](https://github.com/andymai/brepkit/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/brepkit-wasm)](https://www.npmjs.com/package/brepkit-wasm)
[![Last release](https://img.shields.io/github/release-date/andymai/brepkit?label=last%20release)](https://github.com/andymai/brepkit/releases)
[![Commit activity](https://img.shields.io/github/commit-activity/m/andymai/brepkit?label=commits%2Fmonth)](https://github.com/andymai/brepkit/commits/main)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org/) [![unsafe denied](https://img.shields.io/badge/unsafe-denied-success.svg)](#why-a-cad-kernel)

**[Architecture](#architecture)** · **[Performance](#performance)** · **[Getting Started](#getting-started)** · **[Known Limitations](#known-limitations)** · **[Contributing](./CONTRIBUTING.md)**

</div>

One exact-geometry engine, from Rust and from JavaScript. Cut a solid, measure it, export it.

```rust
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::boolean::{boolean, BooleanOp};
use brepkit_operations::measure::solid_volume;
use brepkit_io::step::write_step;
use brepkit_topology::Topology;

let mut topo = Topology::new();

// Primitives are anchored at the origin, so this cylinder rounds off the
// block's corner. Use `transform_solid` to place it somewhere else.
let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
let cutter = make_cylinder(&mut topo, 5.0, 15.0)?;
let notched = boolean(&mut topo, BooleanOp::Cut, block, cutter)?;

// Measure and export
let vol = solid_volume(&topo, notched, 0.1)?;
let step = write_step(&topo, &[notched])?;
```

```js
import { BrepKernel } from 'brepkit-wasm';

const kernel = new BrepKernel();

// Primitives are anchored at the origin, so this cylinder rounds off the
// block's corner. Use `transformSolid` to place it somewhere else.
const block = kernel.makeBox(30, 20, 10);
const cutter = kernel.makeCylinder(5, 15);
const notched = kernel.cut(block, cutter);

// Measure and export
const vol = kernel.volume(notched, 0.1);
const step = kernel.exportStep(notched); // Uint8Array
```

## Why a CAD kernel?

brepkit is a B-Rep solid modeling kernel written from scratch in Rust. It targets WebAssembly, so the same kernel runs in the browser and on the desktop. `unsafe` is denied by lint, as are `unwrap` and `panic`. Every public operation returns a `Result`.

It grew out of building [gridfinitylayouttool.com](https://gridfinitylayouttool.com), where the options for parametric CAD in the browser were proprietary or compiled from large C++ codebases.

The geometry is exact. Booleans run on analytic and NURBS surfaces and keep those surfaces through the operation, so a cylinder stays a cylinder instead of becoming a bag of triangles. That keeps face counts low and round-trips lossless.

## Status

brepkit is in active development. Core modeling is solid. Each feature below is marked stable, beta, planned, or experimental, and [Known Limitations](#known-limitations) covers the gaps.

| Category                | Feature                                                                      | Status       |
| ----------------------- | ---------------------------------------------------------------------------- | ------------ |
| **Primitives**          | Box, cylinder, cone, sphere, torus, ellipsoid                                | Stable       |
| **Primitives**          | Convex hull, Minkowski sum (convex inputs)                                   | Stable       |
| **Booleans**            | Union, cut, intersect on plane, cylinder, cone, sphere, NURBS                | Stable       |
| **Booleans**            | Batch fuse-all (disjoint-aware union)                                        | Stable       |
| **Booleans**            | Torus booleans (box ± torus, coaxial torus)                                  | Beta         |
| **Modifiers**           | Fillet (constant + variable radius), chamfer (walking engine)                | Stable       |
| **Modifiers**           | Shell (hollow solid)                                                         | Stable       |
| **Modifiers**           | Offset face, offset solid, thicken, mirror, pattern                          | Stable       |
| **Modifiers**           | Draft (planar faces)                                                         | Beta         |
| **Sweeps**              | Extrude (planar + NURBS profiles)                                            | Stable       |
| **Sweeps**              | Revolve, sweep, loft, pipe (planar profiles)                                 | Stable       |
| **Sweeps**              | Helical sweep                                                                | Stable       |
| **Sweeps**              | Non-planar profiles for loft, sweep, pipe, revolve                           | Beta         |
| **Construction**        | Coons-patch face fill, sew, untrim                                           | Stable       |
| **Sectioning**          | Cross-section faces, split by plane                                          | Stable       |
| **Measurement**         | Bounding box, area, volume, center of mass                                   | Stable       |
| **Measurement**         | Point-to-solid, solid-to-solid distance, point classification                | Stable       |
| **Drawing**             | Hidden-line edge projection                                                  | Stable       |
| **Geometry**            | NURBS evaluation, derivatives, knot ops, fitting, projection                 | Stable       |
| **Geometry**            | Analytic intersections (plane × cylinder, cone, sphere exact; torus sampled) | Stable       |
| **Geometry**            | Surface-surface intersection (analytic + marching)                           | Stable       |
| **Geometry**            | Curve-curve intersection (Bezier clipping)                                   | Stable       |
| **Tessellation**        | Adaptive deflection, CDT, analytic-surface optimization                      | Stable       |
| **Repair**              | Shape healing (wire, face, shell fixes), sewing, validation                  | Stable       |
| **I/O**                 | STEP import/export (analytic-preserving round-trip)                          | Stable       |
| **I/O**                 | STL, 3MF, OBJ, PLY, glTF (`.glb`) import/export                              | Stable       |
| **I/O**                 | IGES import/export                                                           | Experimental |
| **Sketching**           | 2D constraint solver (DogLeg)                                                | Stable       |
| **Feature Recognition** | Holes, pockets, chamfers, fillets                                            | Beta         |
| **Assemblies**          | Hierarchy, transforms, bill of materials                                     | Beta         |
| **Evolution**           | Face provenance through booleans                                             | Beta         |
| **Defeaturing**         | Remove planar faces                                                          | Beta         |
| **Rendering**           | Offscreen wgpu render to image plus face-id buffer (`brepkit-render`)        | Experimental |

## Known Limitations

A few areas are still maturing. Worth knowing before you build on them:

- **Boolean fallback.** Most booleans run on an exact path that preserves analytic and NURBS surfaces. Hard configurations fall back to a mesh-based boolean: coincident-face contact, coaxial analytic surfaces, razor-thin geometry, or very high face counts. The fallback returns a usable, non-degenerate solid, but it tessellates the curved faces and is not guaranteed watertight.
- **Torus booleans.** Box-with-torus and coaxial-torus cases work and give correct volumes. General torus-to-torus and torus-with-other-surface intersections have known gaps and may fall back to meshing.
- **Non-planar profiles.** Loft, sweep, and pipe accept profiles with non-planar surfaces, and close non-planar section boundaries with bilinear caps for four-sided rings (boundaries with more than four edges, or holes on a non-planar section, are not yet supported). Revolve accepts non-planar profile surfaces; a full revolution takes any boundary, but a partial revolution still requires a planar boundary for its caps. The smooth, scaled/guided, and multi-section sweep variants accept non-planar profiles too; only the miter-corner variant still requires planar profiles (its bisector-plane joint faces would otherwise be non-planar).
- **IGES is experimental.** Export writes planar and NURBS surfaces but skips analytic surfaces and approximates circular and elliptical edges as polylines. Import reconstructs planar placeholder faces only. Use STEP for B-Rep exchange.
- **Inertia tensor.** Volume, area, bounding box, and center of mass are computed for any solid. A full inertia tensor exists only as closed-form formulas for analytic primitives and is not exposed through the modeling or WASM API.
- **Beta subsystems.** Feature recognition, assemblies, evolution tracking, and defeaturing work but are still maturing. Defeaturing handles planar faces only.

## Scope

brepkit deliberately does not:

- **Bundle a viewport into the kernel.** The core emits exact geometry and tessellated meshes; camera, lighting, and shading belong to the caller (Three.js and the like). The optional `brepkit-render` crate provides offscreen wgpu rendering with a face-id buffer, for tests and headless verification, and is not required by any core operation.
- **Plan toolpaths or slice.** Export STEP, STL, or 3MF and pass the output to a CAM tool or slicer.
- **Model with meshes.** The kernel operates on exact B-Rep geometry. Subdivision surfaces, polygon meshes, and voxels are out of scope.
- **Provide a GUI.** brepkit is a library. Building a UI around it, like [gridfinitylayouttool.com](https://gridfinitylayouttool.com), is the application's job.
- **Simulate physics.** Measurement (volume, area, center of mass) is included. Stress analysis, collision detection, and dynamics are not.

## Architecture

Layered Cargo workspace. Each crate depends only on the same or lower layers, and CI enforces the boundaries.

| Layer | Crate                | What it does                                                                                        |
| ----- | -------------------- | --------------------------------------------------------------------------------------------------- |
| L0    | `brepkit-math`       | Points, vectors, matrices, NURBS curves and surfaces, geometric predicates, CDT, convex hull        |
| L1    | `brepkit-geometry`   | Curve sampling (uniform, deflection, arc-length, curvature), extrema, analytic-to-NURBS conversion  |
| L1    | `brepkit-topology`   | Arena-allocated B-Rep: vertex, edge, wire, face, shell, solid, with an edge-to-face adjacency index |
| L2    | `brepkit-algo`       | General Fuse boolean engine: pave filler, face classification, solid assembly                       |
| L2    | `brepkit-blend`      | Walking-based fillet and chamfer with constant, variable, and custom radius laws                    |
| L2    | `brepkit-heal`       | Shape healing: analysis, fixing, upgrading, sewing, tolerance management, configurable pipeline     |
| L2    | `brepkit-check`      | Point classification, validation, properties (volume, area, center of mass), distance               |
| L2    | `brepkit-offset`     | Solid offset and thickening via global face-face intersection                                       |
| L2    | `brepkit-sketch`     | 2D parametric constraint solver (GCS) using a DogLeg trust-region method                            |
| L3    | `brepkit-operations` | Booleans, fillet, chamfer, extrude, revolve, sweep, loft, shell, offset, measure, tessellation      |
| L3    | `brepkit-io`         | Import and export: STEP, IGES, STL, 3MF, OBJ, PLY, glTF                                             |
| L4    | `brepkit-wasm`       | JavaScript API via wasm-bindgen, with batch execution and checkpoint/restore                        |
| L4    | `brepkit-render`     | Offscreen wgpu rendering to a color image plus a face-id buffer. Optional, nothing depends on it    |

## Performance

Median times from the [brepjs benchmark suite](https://github.com/andymai/brepjs/tree/main/benchmarks) (5 iterations, Node.js, Linux x86_64). WASM is single-threaded. Native benchmarks use criterion.

| Operation                | brepkit (WASM) | OCCT (WASM) | Speedup | brepkit (native) |
| ------------------------ | -------------- | ----------- | ------- | ---------------- |
| fuse(box, box) (×10)     | 0.5 ms         | 43.7 ms     | 87x     | 122 µs           |
| cut(box, cylinder) (×10) | 28.3 ms        | 64.3 ms     | 2.3x    | 9.3 ms           |
| box + chamfer            | 0.2 ms         | 5.4 ms      | 27x     | 46 µs            |
| box + fillet             | 0.3 ms         | 6.2 ms      | 21x     | 127 µs           |
| multi-boolean (16 holes) | 4.7 ms         | 30.1 ms     | 6.4x    | 2.8 ms           |
| mesh sphere (tol=0.01)   | 7.1 ms         | 51.9 ms     | 7.3x    | 6.0 ms           |
| exportSTEP (×10)         | 0.9 ms         | 14.3 ms     | 16x     | n/a              |

Every quoted row is output-verified across both kernels before timing is compared: fuse, chamfer, and sphere volumes match exactly; cut, fillet, and multi-boolean volumes agree within 0.004%. The sphere mesh densities are comparable at equal tolerance (9,800 triangles vs 10,176). The `intersect(box, sphere)` row is excluded: brepkit currently keeps the wrong sphere region for that configuration (an open, pinned defect), so its ~200x timing would not be a like-for-like comparison.

Booleans preserve analytic surfaces, so face counts stay low across chained operations. A nine-step compound boolean settles at 72 faces while a mesh-based approach would reach roughly 7,000. The same holds for blends: a straight edge filleted between two planar faces keeps an exact cylindrical wall rather than a NURBS approximation of one.

> The OCCT comparison uses [occt-wasm](https://www.npmjs.com/package/occt-wasm), an OpenCASCADE build compiled to WebAssembly. Both kernels run single-threaded in Node.js. Boolean and `exportSTEP` rows are timed as batches of ten operations. WASM figures are medians of `kernel-comparison.bench.test.ts` (5 iterations) against a local `cargo xtask wasm-build` package, hash-verified at the require path. Native figures: `cargo bench -p brepkit-operations --bench cad_operations`, except the mesh-sphere row, which is measured at the same parameters as the WASM row (`tessellate_solid_with_tolerance`, deflection 0.01, angular 0.1 rad) via `crates/operations/examples/perf_probe.rs` — the criterion suite's sphere case meshes per-face and is not comparable. Full benchmark source: [brepjs/benchmarks](https://github.com/andymai/brepjs/tree/main/benchmarks). Measured 2026-08-06 on brepkit main (post-2.129.8, with the display-sphere tessellation fix).

## Data Exchange

| Format        | Type  | Import  | Export |
| ------------- | ----- | ------- | ------ |
| STEP          | B-Rep | ✓       | ✓      |
| STL           | Mesh  | ✓       | ✓      |
| 3MF           | Mesh  | ✓       | ✓      |
| OBJ           | Mesh  | ✓       | ✓      |
| PLY           | Mesh  | ✓\*     | ✓      |
| glTF (`.glb`) | Mesh  | ✓       | ✓      |
| IGES          | B-Rep | preview | lossy  |

STEP preserves exact geometry on round-trip. Analytic surfaces (plane, cylinder, cone, sphere, torus) are written as native STEP surface entities rather than tessellated, and they read back to the same surface types. NURBS surfaces are preserved too, as are line, circle, ellipse, and NURBS edges.

Mesh formats export tessellated triangles. glTF is binary `.glb`, with no materials or scene graph. IGES is experimental, as described in [Known Limitations](#known-limitations).

\* PLY import is available in the Rust crate but is not yet exposed in the WASM API.

## Getting Started

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

### As a Rust dependency

Not yet published to crates.io. Use git dependencies for now:

```toml
[dependencies]
brepkit-math = { git = "https://github.com/andymai/brepkit" }
brepkit-topology = { git = "https://github.com/andymai/brepkit" }
brepkit-operations = { git = "https://github.com/andymai/brepkit" }
brepkit-io = { git = "https://github.com/andymai/brepkit" }        # optional
```

### Building from source

Requires Rust 1.88 or newer.

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all

# WASM (with I/O)
cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release

# WASM (smaller, no I/O)
cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release --no-default-features

# API docs
cargo doc --workspace --no-deps --open
```

## Roadmap

Broad directions, no dates.

- **Boolean robustness.** Harden torus and mixed-surface booleans, and shrink the set of inputs that fall back to meshing.
- **Sweep generalization.** Extend non-planar profile support to the miter-corner sweep, to section boundaries with more than four edges, and to partial revolutions with non-planar boundaries.
- **Parallel tessellation in WASM.** Native builds already parallelize per-face meshing. Bring it to the WASM target via threads.
- **Assembly metadata.** Colors, layers, materials, and PMI for richer data exchange.
- **Lossless IGES.** Real B-Rep import and analytic-surface export.
- **Documentation.** API reference, tutorials, and architectural guides.

## Projects Using brepkit

- [brepjs](https://github.com/andymai/brepjs), CAD modeling for JavaScript.
- [Gridfinity Layout Tool](https://github.com/andymai/gridfinity-layout-tool), a web-based Gridfinity storage layout generator.

[Open a PR](https://github.com/andymai/brepkit/pulls) to add your project.

## License

Licensed under either of

- [Apache License, Version 2.0](./LICENSE-APACHE)
- [MIT License](./LICENSE-MIT)

at your option.
