# Performance Optimization Research for brepkit

Research into academic papers, industry publications, and practical techniques for
optimizing a B-Rep CAD modeling engine written in Rust, compiled to WebAssembly.

Compiled: 2026-03-02

---

## Table of Contents

1. [Spatial Indexing for CAD](#1-spatial-indexing-for-cad)
2. [Parallel and SIMD Geometry Processing](#2-parallel-and-simd-geometry-processing)
3. [Memory-Efficient B-Rep and Cache-Friendly Layouts](#3-memory-efficient-b-rep-and-cache-friendly-layouts)
4. [Tessellation Algorithms](#4-tessellation-algorithms)
5. [WebAssembly Performance](#5-webassembly-performance)
6. [Acceleration Structures for Boolean Operations](#6-acceleration-structures-for-boolean-operations)
7. [Arena Allocation and Data-Oriented Design](#7-arena-allocation-and-data-oriented-design)
8. [Incremental Computation](#8-incremental-computation)
9. [Recommendations for brepkit](#9-recommendations-for-brepkit)

---

## 1. Spatial Indexing for CAD

### 1.1 BVH Survey

**"A Survey on Bounding Volume Hierarchies for Ray Tracing"**
- Authors: Daniel Meister, Shinji Ogaki, Carsten Benthin, Michael J. Doyle, Michael Guthe, Jiri Bittner
- Year: 2021
- Published: Computer Graphics Forum (Eurographics State of the Art Report)
- URL: https://meistdan.github.io/publications/bvh_star/paper.pdf

Key contributions: Comprehensive survey of BVH construction, traversal, and optimization
techniques. Covers SAH (Surface Area Heuristic) construction, spatial splits, wide BVH
(BVH4/BVH8), TLAS/BLAS two-level hierarchies, and parallel construction. Though focused on
ray tracing, the traversal and construction algorithms are directly applicable to any spatial
query problem including proximity queries, intersection tests, and collision detection in
CAD kernels.

**Applicability to brepkit:** The current codebase performs surface-surface intersection
via grid sampling and Newton refinement. A BVH over NURBS control-point bounding boxes
could prune impossible intersection pairs before expensive numerical work begins. The
SAH-based construction gives near-optimal tree quality for non-uniform geometry distributions
typical in CAD models.

### 1.2 BVH with Osculating Toroidal Patches for SSI

**"Surface-Surface-Intersection Computation Using a Bounding Volume Hierarchy with Osculating Toroidal Patches in the Leaf Nodes"**
- Authors: Y. Park, S.H. Son, M.S. Kim, G. Elber
- Year: 2020
- Published: Computer-Aided Design, vol. 127, article 102866
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0010448520300592

Key contributions: Instead of subdividing NURBS surfaces into flat triangles for
intersection, this method uses osculating toroidal patches as BVH leaf nodes. The toroidal
patches provide higher-order approximation of the surface, which dramatically reduces the
number of leaf-leaf intersection tests needed -- especially for near-tangential
intersections where subdivision methods struggle with exponential pair counts. The torus
patches also make bounding-box computation of surface normals and point projection simpler.

**Applicability to brepkit:** This is directly relevant to `intersection.rs` (the
NURBS-NURBS SSI module). The current alternating-projection + marching approach could be
augmented with a BVH pre-filter using tight bounding volumes around NURBS patches, reducing
the sampling grid resolution needed for initial seed finding.

### 1.3 Embree: CPU-Optimized BVH Framework

**"Embree: A Kernel Framework for Efficient CPU Ray Tracing"**
- Authors: Ingo Wald, Sven Woop, Carsten Benthin, Gregory S. Johnson, Manfred Ernst
- Year: 2014
- Published: ACM Transactions on Graphics (SIGGRAPH 2014)
- URL: https://cseweb.ucsd.edu/~ravir/274/15/papers/a143-wald.pdf

Key contributions: Production-quality BVH implementation using wide BVH nodes (BVH4 for
SSE/AVX, BVH8 for AVX-512). Key techniques:
- **SAH with binned construction**: O(N log N) build using binned cost evaluation instead
  of exhaustive sweep; about 350ms for complex scenes vs. seconds for full sweep
- **Wide nodes**: BVH4 tests 4 child bounding boxes simultaneously with a single SIMD
  instruction; reduces tree depth and traversal steps
- **Packet traversal**: Tests multiple rays against BVH nodes simultaneously
- **Ordered traversal**: Visits children in front-to-back order for early termination
- **TLAS/BLAS**: Two-level acceleration for instanced/animated geometry

**Applicability to brepkit:** The algorithmic patterns (SAH binned build, wide-node
traversal) apply to any spatial query acceleration, not just ray tracing. A BVH4 using
wasm-simd128 (4x f32 lanes) could accelerate face-face overlap tests in boolean operations,
proximity queries for fillet/chamfer radius computation, and point-in-solid classification.

### 1.4 Jacco Bikker's BVH Tutorial Series

**"How to Build a BVH" (Parts 1-9)**
- Author: Jacco Bikker (Utrecht University)
- Year: 2022
- URL: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

Key contributions: Practical, implementation-focused tutorial covering:
- Part 1: Basic data structure and naive build
- Part 2: SAH-based construction with cost model
- Part 3: Binned BVH building for fast construction
- Part 5: TLAS/BLAS two-level hierarchy
- Part 7: Consolidation and optimization passes
- Part 9: GPU adaptation

Written in "sane C++" with full source code. An excellent reference for implementing
a first BVH from scratch.

**Applicability to brepkit:** Ideal step-by-step guide for adding a BVH to the math
or operations crate. The binned SAH approach from Part 3 is the sweet spot of build
quality vs. construction speed for interactive CAD.

### 1.5 R*-tree for Dynamic Spatial Indexing

**R-tree and R*-tree**
- Original: A. Guttman, "R-trees: A Dynamic Index Structure for Spatial Searching", 1984
- R*-variant: N. Beckmann et al., 1990
- Rust crate: `rstar` (https://github.com/georust/rstar)

Key properties: Unlike BVH (which is typically rebuilt from scratch), R*-trees support
efficient incremental insertion and deletion. This makes them suitable for dynamic CAD
editing sessions where entities are added/removed frequently.

The Rust `rstar` crate provides a well-tested R*-tree implementation. The `geo-index`
crate (https://github.com/kylebarron/geo-index) provides a packed, immutable variant
that is ~2x faster to construct and ~33% faster to query than `rstar` when the index
does not need updates.

**Applicability to brepkit:** An R*-tree maintained alongside the topology arena could
accelerate edge/face proximity queries, snap-to operations, and interference detection
during boolean operations. For read-heavy use cases (e.g., tessellation, export), a
packed immutable index would be faster.

---

## 2. Parallel and SIMD Geometry Processing

### 2.1 GPU NURBS Evaluation

**"Direct Evaluation of NURBS Curves and Surfaces on the GPU"**
- Authors: Adarsh Krishnamurthy, Ravi Khardekar, Sara McMains
- Year: 2007 (SPM), 2009 (CAD journal, optimized version)
- Published: ACM Symposium on Solid and Physical Modeling; Computer-Aided Design vol. 41
- URL: https://mcmains.me.berkeley.edu/pubs/SPM07KrishnamurthyKhardMcMains.pdf
- Optimized: https://mcmains.me.berkeley.edu/pubs/CAD09KrishKhardMcMains.pdf

Key contributions: Direct NURBS evaluation on GPU without converting to Bezier patches
first. Stores control points and knot vectors as GPU textures. Achieved 40x+ speedup
over CPU for evaluation and voxelization. Supports arbitrary degree NURBS. The optimized
2009 version improves memory access patterns and supports larger models.

**"Performing Efficient NURBS Modeling Operations on the GPU"**
- Authors: Krishnamurthy, McMains, et al.
- Year: 2009
- Published: IEEE TVCG
- URL: https://mcmains.me.berkeley.edu/pubs/TVCG09krishnamurthyMcMainsEtAl.pdf

Extends to modeling operations (trim, intersect, evaluate) entirely on GPU.

**Applicability to brepkit:** While WebAssembly does not have GPU access natively,
the key insight -- batch evaluation of many parameter-space samples simultaneously --
applies to SIMD. A single NURBS surface evaluation requires repeated basis function
computation that is highly vectorizable. WebGPU compute shaders could also be used
from JavaScript to accelerate batch NURBS evaluation.

### 2.2 SIMD NURBS Ray Tracing

**"Interactive Ray Tracing of NURBS Surfaces by Using SIMD Instructions and the GPU in Parallel"**
- Author: Oliver Abert
- Year: 2005
- Published: Semantic Scholar

Key contributions: Transforms the recursive Cox-de Boor basis function evaluation into
a SIMD-friendly iterative form, avoiding recursion and reducing instruction count. Uses
SSE to evaluate 4 basis functions simultaneously.

**Applicability to brepkit:** The Cox-de Boor transformation technique is directly
applicable. brepkit's NURBS evaluation currently uses the standard recursive form. An
iterative, SIMD-friendly formulation using wasm-simd128 (4 x f32) could evaluate 4
parameter values simultaneously.

### 2.3 Matrix-Based NURBS Evaluation

**"Parallel Inverse Evaluation of NURBS Surfaces Based on Matrix Representation"**
- Authors: Bao, Liu, Zou
- Year: 2023
- URL: https://github.com/Qiang-Zou/GPU-NURBS (code)

Key contributions: Transforms B-spline evaluation from recursive basis function
computation into matrix multiplication. This reformulation is inherently parallel
and shows ~100x speedup over traditional methods. Matrix operations map well to
SIMD and GPU compute.

**Applicability to brepkit:** Matrix-based NURBS evaluation would be a significant
refactor but offers substantial speedup. The matrix form also benefits from optimized
BLAS-like operations. For WASM, even without full BLAS, the regular memory access
pattern of matrix multiply is more cache-friendly than recursive basis evaluation.

### 2.4 WebAssembly SIMD in Rust

**"Authoring a SIMD Enhanced Wasm Library with Rust"**
- Author: Nick Babcock
- URL: https://nickb.dev/blog/authoring-a-simd-enhanced-wasm-library-with-rust/

Practical guide for using wasm-simd128 from Rust:
- Enable with `RUSTFLAGS="-C target-feature=+simd128"` or in `.cargo/config.toml`
- Use `core::arch::wasm32` intrinsics (e.g., `f32x4_add`, `f32x4_mul`)
- Conditional compilation: `#[cfg(target_feature = "simd128")]`
- All modern browsers support wasm-simd128 (Chrome 91+, Firefox 89+, Safari 16.4+)
- Cannot detect SIMD support at runtime in WASM; must compile separate SIMD/scalar builds
  or use a single SIMD build (safe given browser support levels in 2025+)

**V8 WebAssembly SIMD documentation:**
- URL: https://v8.dev/features/simd
- Reports up to 30% performance boost for parallel workloads

**Applicability to brepkit:** The math crate's `Vec3`, `Point3`, `Mat4` operations are
prime candidates for SIMD. A `f32x4` can hold (x, y, z, w) for homogeneous coordinates.
Batch NURBS basis function evaluation, tessellation vertex transformation, and
bounding-box computation could all benefit.

---

## 3. Memory-Efficient B-Rep and Cache-Friendly Layouts

### 3.1 Data-Oriented Design

**"Data-Oriented Design and C++"**
- Author: Mike Acton (Engine Director, Insomniac Games)
- Year: CppCon 2014
- Slides: https://neil3d.github.io/assets/img/ecs/DOD-Cpp.pdf
- URL: https://isocpp.org/blog/2015/01/cppcon-2014-data-oriented-design-and-c-mike-acton

Core principles:
1. The purpose of software is to transform data; understand data to understand performance
2. A cache miss (~200 cycles) is far more expensive than a sqrt (~15 cycles)
3. Design for the common case: arrays of homogeneous data, not trees of heterogeneous objects
4. "There is never one of anything" -- design for batch processing, not single-item access
5. Reported 6.8-10x speedups from cache-friendly data layout transformations alone

**"Data-Oriented Design" (Book)**
- Author: Richard Fabian
- URL: https://www.dataorienteddesign.com/site.php

Comprehensive treatment of DOD principles with examples across game engines and
simulation software.

**Applicability to brepkit:** The topology arena already uses a data-oriented layout
(arrays of entities indexed by typed handles). The next step is to evaluate whether
hot fields (e.g., geometry pointers, adjacency links) are separated from cold fields
(e.g., metadata, user attributes) to maximize cache utilization during traversal-heavy
operations like tessellation and boolean evaluation.

### 3.2 Compact Half-Edge Structures

**Geometry-Central: Halfedge Mesh Internals**
- URL: https://geometry-central.net/surface/surface_mesh/internals/

Key design decisions:
- All elements stored in flat, contiguous buffers (std::vector-like)
- Index-based references instead of pointers (enables serialization, cache friendliness)
- Implicitly-stored half-edges (each edge implies two half-edges by index arithmetic)
- Separate attribute arrays associated with elements by index

**CGAL Halfedge Data Structures**
- URL: https://doc.cgal.org/latest/HalfedgeDS/index.html

CGAL's HalfedgeDS uses a configurable design: items (vertex, halfedge, face) can be
stored in vectors or lists. The vector-based storage is compact and cache-friendly for
static meshes; list-based supports efficient insertion/deletion for dynamic editing.

**OpenMesh**
- Authors: M. Botsch, S. Steinberg, S. Bischoff, L. Kobbelt
- URL: https://www.researchgate.net/publication/2534339

Provides "smart handles" (typed indices with methods) over array-based storage.
Uses traits-based customization of stored attributes.

**Applicability to brepkit:** The current arena design stores each topology entity type
in its own Vec, which is already close to optimal. Potential improvements:
- **Structure-of-Arrays (SoA)** within entity types: separate vecs for adjacency links
  vs. geometry data, so traversal-only operations skip geometry loading
- **Implicit half-edge pairs**: store half-edges such that edge N has half-edges at
  indices 2N and 2N+1, eliminating the twin pointer
- **Compact adjacency encoding**: if vertex degree is bounded, store adjacency inline
  (no heap allocation per vertex)

### 3.3 NeuroNURBS: Compact NURBS Representations

**"NeuroNURBS: Learning Efficient Surface Representations for 3D Solids"**
- Year: 2024
- URL: https://arxiv.org/html/2411.10848v1

Achieves 79.9% memory reduction for storing 3D solids by learning compressed NURBS
representations. While the ML approach itself may not be applicable, the analysis of
NURBS memory consumption is informative: control point arrays, knot vectors, and weight
arrays dominate memory for complex models.

**Applicability to brepkit:** For models with many similar surfaces (e.g., arrays of
identical holes), a flyweight pattern sharing control-point data with per-instance
transforms could significantly reduce memory.

---

## 4. Tessellation Algorithms

### 4.1 Adaptive Tessellation of Trimmed NURBS

**"Adaptive Tessellation for Trimmed NURBS Surface"**
- URL: https://www.academia.edu/802844/Adaptive_Tessellation_for_Trimmed_NURBS_Surface

Uses flatness-test-based adaptive subdivision in parameter space, with finer
tessellation near trim curves and high-curvature regions. Avoids the over-tessellation
of uniform grids and the under-tessellation of curvature-blind approaches.

**"Efficient Trimmed NURBS Tessellation"**
- URL: https://www.researchgate.net/publication/221546526_Efficient_trimmed_NURBS_tessellation

Generates view-dependent LODs on the fly from the NURBS representation, without
pre-tessellation. The adaptive method tessellates in the parametric domain, then
projects to 3D, handling trim curves as constraints in the 2D triangulation.

**Applicability to brepkit:** The current tessellation likely uses a fixed grid in
parameter space. An adaptive approach guided by:
- Surface curvature (more triangles in high-curvature regions)
- Edge length in screen space (view-dependent LOD)
- Proximity to trim curves (finer tessellation at boundaries)
could reduce triangle count by 3-10x for typical CAD models while maintaining visual
quality.

### 4.2 View-Dependent Adaptive Tessellation

**"View-dependent Adaptive Tessellation of Spline Surfaces"**
- Authors: Jatin Chhugani, Subodh Kumar
- Year: 2001 (IIT Delhi)
- URL: https://www.cse.iitd.ac.in/~subodh/research/papers/adaptess.pdf

Key contributions: Uses a hierarchical organization of pre-computed domain samples
and maintains a running Delaunay triangulation of the active subset each frame. Only
recomputes tessellation for regions where the view has changed significantly.

**"Adaptive NURBS Tessellation on GPU"**
- Published: Springer
- URL: https://link.springer.com/content/pdf/10.1007/978-981-287-134-3_5.pdf

GPU-based approach: tessellation interval estimation followed by conversion to rational
Bezier patches and gap-free tessellation. Uses hardware tessellation units where
available.

**Applicability to brepkit:** For the WASM target, view-dependent LOD is particularly
valuable because:
1. Reduces the JS/WASM data transfer for mesh updates
2. Allows coarse tessellation during interaction (rotate/zoom), refined on idle
3. Multi-level tessellation can be streamed incrementally to avoid frame drops

### 4.3 GPU-Based Trimming and Tessellation

**"GPU-based Trimming and Tessellation of NURBS and T-Spline Surfaces"**
- Published: ACM Transactions on Graphics
- URL: https://dl.acm.org/doi/10.1145/1073204.1073305

Performs trimming and tessellation entirely on the GPU, avoiding CPU-GPU data transfer
bottlenecks. The algorithm handles T-junctions and gap-free stitching between adjacent
patches at different tessellation levels.

---

## 5. WebAssembly Performance

### 5.1 Performance Characteristics

**"Understanding the Performance of WebAssembly Applications"**
- URL: https://weihang-wang.github.io/papers/imc21.pdf
- Also: https://benchmarkingwasm.github.io/BenchmarkingWebAssembly/

Key findings:
- WASM achieves ~95% of native performance for compute-intensive tasks in modern browsers
- Firefox executes WASM faster than Chrome (0.61x ratio) on desktop
- Memory consumption is significantly higher than native: +24MB for large inputs, +74MB
  for extra-large inputs (due to linear memory model)
- `-O2` optimization level is the best balance of speed vs. code size
- `-Ofast` is faster but may produce larger binaries that hurt load time
- Memory64 causes 10-15% slowdown for memory-heavy workloads vs. 32-bit

### 5.2 wasm-bindgen Overhead

**"Rust + WebAssembly Performance: Pure JS vs. wasm-bindgen vs. Raw WASM with SIMD"**
- URL: https://medium.com/@oemaxwell/rust-webassembly-performance-javascript-vs-wasm-bindgen-vs-raw-wasm-with-simd-687b1dc8127b
- Also: https://dev.to/bence_rcz_fe471c168707c1/rust-webassembly-performance-javascript-vs-wasm-bindgen-vs-raw-wasm-with-simd-4pco

Key findings:
- wasm-bindgen: 3-5x faster than pure JavaScript
- Raw WASM exports: 8-10x faster than pure JavaScript
- The gap between wasm-bindgen and raw WASM comes from data marshalling (copying arrays
  between JS and WASM memory) and the generated glue code
- For compute-heavy functions with small inputs (e.g., boolean on two solids), the
  overhead is negligible
- For large data transfers (e.g., mesh vertices for tessellation), the copy overhead
  matters

**"We built a faster WASM-bindgen (2.5x)"**
- URL: https://news.ycombinator.com/item?id=45664341

Reports on optimizations to wasm-bindgen for high-frequency JS-Rust calls, achieving
2.5x improvement in binding overhead.

**Applicability to brepkit:** The WASM layer should:
1. Minimize JS/WASM boundary crossings by batching operations
2. Return mesh data as SharedArrayBuffer or typed array views into WASM memory
   (avoiding copy) where possible
3. Use `wasm-bindgen`'s `#[wasm_bindgen(js_name = ...)]` for ergonomic API while
   keeping hot paths as raw exports
4. Consider SIMD builds as the default (browser support is universal in 2025+)

### 5.3 WASM-Specific Optimization Techniques

**"WebAssembly Performance Optimization: From Bytecode to Blazing Speed"**
- URL: https://nerdleveltech.com/webassembly-performance-optimization-from-bytecode-to-blazing-speed

Key techniques:
- **Tiered compilation**: Browsers use baseline (fast startup) then optimizing (peak perf)
  JIT. Design code to be JIT-friendly (avoid indirect calls, use concrete types)
- **Memory pooling**: Pre-allocate large linear memory blocks and sub-allocate. Avoid
  frequent `memory.grow` calls which are expensive
- **Code size matters**: Smaller WASM binary = faster download + faster compilation.
  Use `wasm-opt -Oz` and `lto = true` in Cargo.toml
- **Avoid allocator thrashing**: Use arena/bump allocation instead of per-object
  malloc/free. brepkit's arena pattern is already optimal here

**Benchmarking WebAssembly for Embedded Systems**
- Published: ACM TACO, 2025
- URL: https://dl.acm.org/doi/10.1145/3736169

Reports that JIT modes in WASM runtimes perform comparably to native execution, with
SIMD support providing up to 30% additional speedup.

---

## 6. Acceleration Structures for Boolean Operations

### 6.1 Interactive and Robust Mesh Booleans

**"Interactive and Robust Mesh Booleans"**
- Authors: Gianmarco Cherchi, Fabio Pellacini, Marco Attene, Marco Livesu
- Year: 2022
- Published: ACM Transactions on Graphics (SIGGRAPH Asia 2022)
- URL: https://dl.acm.org/doi/10.1145/3550454.3555460
- Code: https://github.com/gcherchi/InteractiveAndRobustMeshBooleans

Key contributions:
- Interactive frame rates for booleans on meshes up to 200K triangles
- **Filtered exact arithmetic**: Attempts floating-point first, falls back to exact
  rationals only when needed. This cascaded approach avoids the overhead of full exact
  arithmetic (which is 10-100x slower) while maintaining robustness
- Uses a spatial hash grid for rapid triangle-triangle overlap detection
- Classifies mesh regions via graph flood-fill after intersection computation

**Applicability to brepkit:** The filtered-exact-arithmetic approach is directly
applicable. brepkit's current boolean implementation could use:
1. A spatial hash or BVH for face-pair filtering before intersection
2. Floating-point predicates with exact-arithmetic fallback for robustness
3. Region classification via topological flood-fill

### 6.2 Hybrid Boolean Operations

**"Boolean Operation for CAD Models Using a Hybrid Representation"**
- Published: ACM Transactions on Graphics, vol. 44(4), 2025
- URL: https://dl.acm.org/doi/10.1145/3730908

Key contributions: Addresses common issues in B-Rep booleans: low efficiency, missing
results, and non-watertightness. Uses a hybrid mesh/B-Rep representation where:
- Intersection computation uses the mesh representation (fast, robust)
- Result reconstruction uses the B-Rep representation (exact geometry)
This mirrors brepkit's existing "tessellate-then-clip" approach for NURBS booleans
but provides a more principled framework.

### 6.3 EMBER: Exact Mesh Booleans

**"EMBER: Exact Mesh Booleans via Efficient & Robust Local Arrangements"**
- Year: 2022
- Published: ACM SIGGRAPH 2022
- URL: https://dl.acm.org/doi/abs/10.1145/3528223.3530181

Computes exact mesh arrangements using local intersection computations, avoiding
global sweep-line complexity. Demonstrates that locality-aware algorithms with
spatial acceleration outperform global approaches by orders of magnitude.

---

## 7. Arena Allocation and Data-Oriented Design

### 7.1 Catherine West's RustConf 2018 Keynote

**"Using Rust for Game Development"**
- Author: Catherine West (kyren)
- Year: RustConf 2018
- Blog: https://kyren.github.io/2018/09/14/rustconf-talk.html

Key insights for Rust data-oriented design:
- Inheritance hierarchies fight the borrow checker; flat, arena-allocated data wins
- ECS (Entity Component System) is the natural Rust-friendly architecture
- Generational indices solve the ABA problem (dangling references after deletion)
- Separate storage per component type enables cache-friendly iteration
- "If you find yourself with &mut references to multiple parts of a data structure
  simultaneously, you probably need an arena"

**Applicability to brepkit:** The topology arena with typed `Id<T>` handles is
already following this pattern. The "snapshot then allocate" pattern noted in MEMORY.md
is the correct ECS-style solution to borrow-checker conflicts during mutation.

### 7.2 Generational Arena Benchmarks

**Generational Arena Bench**
- URL: https://github.com/mooman219/generational_arena_bench

Benchmarks multiple Rust arena crates:

| Crate | Key Size | Insert | Lookup | Notes |
|-------|----------|--------|--------|-------|
| `slotmap` | 8 bytes | Fast | Fast | Secondary maps, compact iteration |
| `thunderdome` | 8 bytes | Fast | Fast | NonZero optimization for Option |
| `generational-arena` | 16 bytes | Medium | Medium | Zero unsafe code |

Key observations:
- `slotmap` uses a "hop" representation for faster iteration with holes
- `thunderdome` is minimal and fast, with 8-byte keys via NonZero
- All three are significantly faster than HashMap<Id, Entity> for typical CAD patterns
  (sequential access, infrequent deletion)

**Applicability to brepkit:** If iteration performance over sparse arenas becomes a
bottleneck (e.g., iterating all faces after many deletions), `slotmap`'s hop-based
compaction or periodic arena defragmentation could help.

### 7.3 Arena Allocation in Rust

**"Arenas in Rust"**
- Author: Manish Goregaokar
- Year: 2021
- URL: https://manishearth.github.io/blog/2021/03/15/arenas-in-rust/

Covers typed arenas, bump allocators, and region-based memory management. Key points:
- Bump allocation is the fastest allocator (pointer increment)
- Arena-allocated objects have uniform lifetime, simplifying borrow checking
- For CAD: the model is the arena; operations create new entities in the arena

**"Guide to Using Arenas in Rust"**
- URL: https://blog.logrocket.com/guide-using-arenas-rust/

Practical patterns for arena usage including scoped arenas for temporary computations
and the relationship between arena lifetime and reference validity.

---

## 8. Incremental Computation

### 8.1 Salsa: Incremental Computation Framework

**Salsa**
- Authors: Niko Matsakis et al.
- URL: https://github.com/salsa-rs/salsa
- Docs: https://docs.rs/salsa/latest/salsa/

A Rust framework for on-demand, incrementalized computation. Used by rust-analyzer
and the Rust compiler. Core concepts:
- Define computations as **queries** (functions K -> V)
- Queries can be **inputs** (user-set) or **derived** (computed from other queries)
- Results are memoized; when inputs change, Salsa determines the minimal set of
  derived queries that need recomputation
- Uses a dependency graph to track which queries depend on which inputs

**"Durable Incrementality"**
- Author: rust-analyzer team
- Year: 2023
- URL: https://rust-analyzer.github.io/blog/2023/07/24/durable-incrementality.html

Describes persistent incrementality across process restarts, relevant for CAD where
models are saved and reopened.

**Applicability to brepkit:** For parametric CAD (future feature), Salsa's model maps
directly to the CAD dependency graph:
- **Inputs**: sketch dimensions, feature parameters
- **Derived queries**: feature geometry, boolean results, tessellation
- When a user changes one dimension, only the dependent features recompute

This would be a major architectural addition but would enable interactive parametric
modeling where changing one parameter does not rebuild the entire model.

### 8.2 Incremental Computation Theory

**"Incremental Computation via Partial Evaluation"**
- Published: Yale CS Technical Report
- URL: https://cpsc.yale.edu/sites/default/files/files/tr889.pdf

Theoretical foundation for incremental computation via specializing programs to their
static inputs, leaving only dynamic-input-dependent computation for runtime.

**"Incremental Computation: What Is the Essence?"**
- Year: 2024
- Published: ACM SIGPLAN PEPM 2024
- URL: https://dl.acm.org/doi/10.1145/3635800.3637447

Survey of incremental computation approaches, distinguishing between:
- **Change propagation**: re-execute only what depends on changed inputs
- **Memoization**: cache results and look up by input
- **Self-adjusting computation**: automatically track and replay dependencies

### 8.3 CAD-Specific Dependency Management

**"An Improved Explicit Reference Modeling Methodology for Parametric Design"**
- Year: 2023
- Published: Computer-Aided Design
- URL: https://dl.acm.org/doi/10.1016/j.cad.2023.103541

Addresses dependency management in parametric CAD, identifying that acyclic digraphs
(DAGs) are the right abstraction for feature dependencies, with explicit references
enabling better traceability and consistency maintenance.

---

## 9. Recommendations for brepkit

Organized by estimated impact and implementation effort.

### High Impact, Moderate Effort

1. **Add a BVH to the math crate** -- A simple AABB-based BVH with SAH construction
   would accelerate boolean operations (face-pair filtering), point-in-solid
   classification, and surface intersection seed finding. Use Jacco Bikker's tutorial
   as implementation guide. The `bvh` Rust crate (https://crates.io/crates/bvh) provides
   a ready-made SAH BVH with rayon-parallel build if bringing in a dependency is acceptable.

2. **Enable wasm-simd128 for the math crate** -- Add `target-feature = "+simd128"` to
   the WASM build config. Implement SIMD paths for Vec3/Point3/Mat4 operations using
   `core::arch::wasm32` intrinsics with scalar fallback. This benefits every operation
   that touches geometry.

3. **Adaptive tessellation** -- Replace fixed-grid parameter-space tessellation with
   curvature-adaptive subdivision. Use the flatness-test approach: recursively subdivide
   parameter-space quads until the mid-edge deviation from the true surface is below a
   threshold (the linear deflection tolerance). This can reduce triangle counts by 5-10x
   for typical CAD models with mixed flat/curved regions.

### High Impact, Higher Effort

4. **Filtered exact arithmetic for booleans** -- Implement the cascaded approach from
   Cherchi et al.: fast floating-point predicates with exact-rational fallback. This
   would make boolean operations robust against degenerate configurations without
   sacrificing performance in the common case.

5. **Salsa-based incremental computation** -- For parametric CAD features, adopt the
   Salsa framework to track dependencies between modeling operations. This is a large
   architectural change but enables interactive parametric editing.

### Moderate Impact, Low Effort

6. **SoA layout for hot topology paths** -- Profile tessellation and boolean operations
   to identify which entity fields are accessed together. Split entity structs into
   hot (adjacency) and cold (metadata) arrays to improve cache utilization during
   traversal.

7. **Batch WASM API** -- Add batch operation endpoints to the WASM layer that accept
   multiple operations in a single call, reducing JS/WASM boundary crossing overhead.
   Return mesh data as views into WASM linear memory rather than copied TypedArrays.

8. **Build-time optimization** -- Enable `lto = true` and `opt-level = 2` in the
   release WASM profile. Run `wasm-opt -O2` on the output. Measure code size vs.
   performance tradeoffs with `wasm-opt -Oz`.

### Lower Priority / Future Work

9. **Matrix-based NURBS evaluation** -- Reformulate NURBS basis evaluation as matrix
   operations per Bao/Liu/Zou 2023. Significant refactor but potentially 100x speedup
   for batch evaluation.

10. **Two-level BVH (TLAS/BLAS)** -- For assembly support, use a two-level hierarchy
    where each part has its own BLAS and the assembly has a TLAS. Instance transforms
    only update the TLAS, not rebuild per-part BVHs.

11. **Arena compaction** -- After heavy editing (many entity deletions), compact the
    arena to eliminate holes and restore cache-sequential access. Requires updating all
    outstanding IDs.

---

## Rust Ecosystem Crates of Interest

| Crate | Purpose | URL |
|-------|---------|-----|
| `bvh` | SAH BVH with rayon parallel build | https://crates.io/crates/bvh |
| `rstar` | R*-tree spatial index | https://crates.io/crates/rstar |
| `geo-index` | Packed immutable spatial index | https://crates.io/crates/geo-index |
| `slotmap` | Fast arena with hop-based iteration | https://crates.io/crates/slotmap |
| `thunderdome` | Minimal generational arena | https://crates.io/crates/thunderdome |
| `salsa` | Incremental computation framework | https://crates.io/crates/salsa |
| `ultraviolet` | SIMD-first math library | https://crates.io/crates/ultraviolet |
| `glam` | Fast math with optional SIMD | https://crates.io/crates/glam |
| `wasm-opt` | WASM binary optimizer | https://crates.io/crates/wasm-opt |

---

## Confidence Assessment

- **BVH for spatial queries**: HIGH confidence this would improve boolean/intersection
  performance. Well-studied, many implementations available, directly applicable.
- **WASM SIMD**: HIGH confidence for math operations. Universal browser support,
  straightforward to implement for Vec3/Mat4.
- **Adaptive tessellation**: HIGH confidence for triangle count reduction. Standard
  technique in CAD, clear implementation path.
- **Filtered exact arithmetic**: MEDIUM confidence. Would improve robustness but
  implementation complexity is significant. May want to use an existing library.
- **Salsa for incremental computation**: MEDIUM confidence for applicability. Excellent
  framework but requires careful design of the query graph for CAD operations.
- **Matrix NURBS evaluation**: LOW-MEDIUM confidence. Promising research but limited
  production validation outside GPU contexts.

---

## Sources

### Papers
- [BVH Survey (Meister et al. 2021)](https://meistdan.github.io/publications/bvh_star/paper.pdf)
- [SSI with Osculating Toroidal BVH (Park et al. 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010448520300592)
- [Embree (Wald et al. 2014)](https://cseweb.ucsd.edu/~ravir/274/15/papers/a143-wald.pdf)
- [GPU NURBS Evaluation (Krishnamurthy et al. 2007)](https://mcmains.me.berkeley.edu/pubs/SPM07KrishnamurthyKhardMcMains.pdf)
- [Optimized GPU NURBS (Krishnamurthy et al. 2009)](https://mcmains.me.berkeley.edu/pubs/CAD09KrishKhardMcMains.pdf)
- [GPU NURBS Modeling Ops (Krishnamurthy et al. 2009)](https://mcmains.me.berkeley.edu/pubs/TVCG09krishnamurthyMcMainsEtAl.pdf)
- [SIMD NURBS Ray Tracing (Abert 2005)](https://www.semanticscholar.org/paper/Interactive-Ray-Tracing-of-NURBS-Surfaces-by-Using-Abert/3ce6f07332dc88ae71d234dbd8d17d76d0020061)
- [Matrix NURBS Evaluation (Bao et al. 2023)](https://github.com/Qiang-Zou/GPU-NURBS)
- [Interactive Mesh Booleans (Cherchi et al. 2022)](https://dl.acm.org/doi/10.1145/3550454.3555460)
- [Hybrid Boolean Ops (ACM TOG 2025)](https://dl.acm.org/doi/10.1145/3730908)
- [EMBER Exact Mesh Booleans (SIGGRAPH 2022)](https://dl.acm.org/doi/abs/10.1145/3528223.3530181)
- [WASM Performance Study (IMC 2021)](https://weihang-wang.github.io/papers/imc21.pdf)
- [WASM Embedded Benchmarks (ACM TACO 2025)](https://dl.acm.org/doi/10.1145/3736169)
- [Adaptive Tessellation Trimmed NURBS](https://www.academia.edu/802844/Adaptive_Tessellation_for_Trimmed_NURBS_Surface)
- [View-Dependent Adaptive Tessellation (Chhugani & Kumar)](https://www.cse.iitd.ac.in/~subodh/research/papers/adaptess.pdf)
- [Adaptive NURBS Tessellation on GPU (Springer)](https://link.springer.com/content/pdf/10.1007/978-981-287-134-3_5.pdf)
- [Incremental Computation via Partial Evaluation (Yale)](https://cpsc.yale.edu/sites/default/files/files/tr889.pdf)
- [Incremental Computation Essence (PEPM 2024)](https://dl.acm.org/doi/10.1145/3635800.3637447)
- [Parametric CAD Dependencies (CAD 2023)](https://dl.acm.org/doi/10.1016/j.cad.2023.103541)
- [NeuroNURBS (2024)](https://arxiv.org/html/2411.10848v1)

### Tutorials and Blog Posts
- [Jacco Bikker BVH Tutorial Series](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
- [Mike Acton DOD Talk](https://isocpp.org/blog/2015/01/cppcon-2014-data-oriented-design-and-c-mike-acton)
- [Catherine West RustConf 2018](https://kyren.github.io/2018/09/14/rustconf-talk.html)
- [WASM SIMD in Rust](https://nickb.dev/blog/authoring-a-simd-enhanced-wasm-library-with-rust/)
- [Rust WASM Performance Comparison](https://medium.com/@oemaxwell/rust-webassembly-performance-javascript-vs-wasm-bindgen-vs-raw-wasm-with-simd-687b1dc8127b)
- [V8 WASM SIMD](https://v8.dev/features/simd)
- [Arenas in Rust](https://manishearth.github.io/blog/2021/03/15/arenas-in-rust/)
- [Geometry-Central Half-Edge Internals](https://geometry-central.net/surface/surface_mesh/internals/)
- [Durable Incrementality (rust-analyzer)](https://rust-analyzer.github.io/blog/2023/07/24/durable-incrementality.html)

### Rust Crates and Tools
- [rstar R*-tree](https://github.com/georust/rstar)
- [bvh crate](https://github.com/svenstaro/bvh)
- [geo-index](https://github.com/kylebarron/geo-index)
- [Salsa](https://github.com/salsa-rs/salsa)
- [generational-arena](https://github.com/fitzgen/generational-arena)
- [Generational Arena Benchmarks](https://github.com/mooman219/generational_arena_bench)
- [OpenCASCADE Parallel Meshing](https://dev.opencascade.org/doc/overview/html/occt_user_guides__mesh.html)
- [OCCT Parallel Development Guide](https://opencascade.blogspot.com/2009/06/developing-parallel-applications-with.html)
