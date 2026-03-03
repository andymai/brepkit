# brepkit — Product Roadmap (Q1–Q3 2026)

> **Vision**: The browser-native B-Rep geometry SDK. Exact solid modeling
> in any web app — no server, no legacy C++ dependencies.
>
> **Dogfood**: [gridfinity-layout-tool](https://github.com/andymai/gridfinity-layout-tool) —
> parametric gridfinity bin generator. Migrating from `brepjs-opencascade` to
> [brepjs](https://github.com/andymai/brepjs) + [brepkit-wasm](https://github.com/andymai/brepkit/tree/main/crates/wasm).
>
> **License**: Free non-commercial. Commercial license required for products.

---

## Current State (March 2026)

| Metric         | Value                                            |
|----------------|--------------------------------------------------|
| Tests          | 807+ passing                                     |
| WASM bindings  | ~145                                             |
| CAD operations | ~25 + helpers                                    |
| I/O formats    | 7 (STEP, IGES, STL, 3MF, OBJ, PLY, glTF)        |
| PRs merged     | 38                                               |
| WASM binary    | 1.9 MB raw / 602 KB gzip'd (release, no wasm-opt)|
| Public release | [brepjs](https://github.com/andymai/brepjs) on npm (stable API); brepkit not yet published |

### Known Gaps

| Gap                                    | Severity | Phase |
|----------------------------------------|----------|-------|
| Adapter drops sweep options (`withContact`) | Critical | P1.1  |
| Adapter drops loft options (`ruled: true`) | Critical | P1.1  |
| Adapter drops boolean options (`commonFace`) | Critical | P1.1 |
| OpenCascade → brepkit migration        | Critical | P1.2  |
| Fillet vertex blending (3-edge corners)| High     | P2.2  |
| Boolean coplanar face handling         | Medium   | P2.1  |
| `meshEdges()` returns stub (no edge viz) | Medium | P1.1  |
| NURBS tessellation (no boundary conformance) | Medium | —  |
| WASM bundle size (602 KB over target)  | Medium   | P3.2  |
| STL per-face tessellation (non-manifold) | Medium | P1.3 |
| No 2D grid pattern                     | Low      | P1.4  |

### Release Strategy

- **Cadence**: Continuous semver via npm
- **Versioning**: brepjs semver; brepkit crates version-locked internally
- **Distribution**: npm (primary), crates.io deferred
- **Docs**: TypeDoc API reference + curated examples

---

## Phase 1: Migration & Quick Wins (March–April 2026)

Get gridfinity-layout-tool running on brepkit with watertight export and a performance baseline. Fillet corners deferred — bins ship with chamfers initially.

### Dependencies

```
P1.1 (adapter audit) → P1.2 (migration) → P1.5 (integration tests)
P1.3 (STL fix) ─────────────────────────────→ P1.5
P1.4 (grid pattern) ────────────────────────→ P1.5
P1.6 (perf baseline) ── parallel with all ──→
```

### P1.1 — brepjs Adapter Audit

The adapter silently drops unsupported options. `sweepPipeShell` ignores `withContact`, `frenet`, `correction` — stacking lip geometry will be wrong.

- [ ] Audit every adapter method for silently ignored options (grep for `_options`, unused params)
- [ ] Catalog pass-through vs. dropped options
- [ ] Fix critical adapter gaps used by gridfinity-layout-tool:
  - `sweepPipeShell` — `withContact` (stacking lip), `correction` (profile alignment)
  - `loftAdvanced` — `ruled: true` (socket tapers, baseplate pockets, connector chamfers — 5 call sites)
  - `fuse`/`fuseAll` — `optimisation: 'commonFace'` (6 call sites — won't break geometry but leaves redundant faces)
  - `meshEdges` — currently returns stub (edge visualization missing)
  - `fillet` — verify edge selection via `iterShapes` + `getBounds`
  - `shell` — verify open-top-box case (remove top face, offset inward)
  - `edgeFinder`/`faceFinder` — verify `getBounds`, face normals
- [ ] Add adapter integration tests for each fixed method

### P1.2 — OpenCascade → brepkit Migration

6 non-test files reference OpenCascade directly; generator files import from brepjs. Migration is primarily swapping WASM init.

- [ ] Replace `brepjs-opencascade` WASM init with brepkit WASM init in `wasmInstantiator.ts`
- [ ] Update `wasmPreload.ts` and `wasmCapabilities.ts` (remove SharedArrayBuffer/pthread logic)
- [ ] Remove `brepjs-opencascade` from `package.json`
- [ ] Run all generator tests against brepkit backend
- [ ] Visual diff: same bin spec → compare meshes for regressions

### P1.3 — STL Watertight Export

- [ ] Switch STL writer to `tessellate_solid()` (shared-edge tessellator)
- [ ] Maintain backward compat for ASCII/binary modes
- [ ] Test: box STL → 0 boundary edges in MeshLab

### P1.4 — Grid Pattern (2D)

- [ ] Add `grid_pattern(solid, dir_x, dir_y, spacing_x, spacing_y, count_x, count_y)`
- [ ] Return `Compound` of positioned copies
- [ ] WASM binding: `gridPattern()`

### P1.5 — Gridfinity Integration Testing

Validates full pipeline: brepjs API → brepkit-wasm → STL/3MF. Depends on P1.1 and P1.2.

- [ ] Integration test: 2×1×3u bin with stacking lip
- [ ] Integration test: 3×3 baseplate with magnet holes
- [ ] Verify printability: import STL into PrusaSlicer, check for manifold errors
- [ ] Measure boolean pass rate on gridfinity-specific operations

### P1.6 — Performance Baseline

Can run in parallel with other Phase 1 work.

- [ ] Benchmark suite: primitive creation, boolean cut, fillet, shell, tessellation, pattern
- [ ] Gridfinity benchmarks: 1×1 bin gen, 3×3 baseplate gen
- [ ] In-browser measurement via `performance.now()` around brepjs calls
- [ ] Record WASM load time, peak memory, wall-clock per operation

### Phase 1 Exit Criteria

| Metric                                   | Target                                          |
|------------------------------------------|--------------------------------------------------|
| gridfinity-layout-tool runs on brepkit   | No `brepjs-opencascade` in `package.json`        |
| Generator tests pass                     | All existing tests green on brepkit backend      |
| Stacking lip geometry correct            | Visual match with OpenCascade output             |
| STL export watertight                    | 0 boundary edges on test solids                  |
| Performance baseline documented          | Numbers recorded for all key operations          |

---

## Phase 2: Robustness (May–June 2026)

Harden booleans and add fillet corner blending. Pick **one** of P2.2 or P2.3 — both are multi-week research efforts. P2.4 is a stretch goal.

### Dependencies

```
P2.1 (boolean campaign) → P2.4 (healing, stretch)
P2.2 (vertex blending) ─or─ P2.3 (non-planar fillet)  ← pick one
```

### P2.1 — Boolean Reliability Campaign

The try-and-fallback design silently degrades to tessellated approximation.

- [ ] Build 100-case stress-test suite (categories: coplanar, tangent, thin-wall, near-miss)
- [ ] Add telemetry: log which path was taken (analytic / NURBS / tessellated fallback)
- [ ] Fix coplanar face classification (CoplanarSame/Opposite dropping fragments)
- [ ] Make tessellated fallback deflection configurable (currently hardcoded at 1.0)
- [ ] Explicit error on degenerate result (empty solid, isolated face)
- [ ] Target: <5% fallback-to-tessellation rate on the test suite

### P2.2 — Fillet Vertex Blending (Convex Corners)

Scoped to convex 3-edge corners only (box-like shapes). Concave corners and n-edge vertices deferred.

- [ ] Spherical vertex blend patches at convex 3-edge junctions
- [ ] Watertight stitching between blend patch and adjacent fillet surfaces
- [ ] Test: fillet all 12 edges of a box → 0 boundary edges
- [ ] Stretch: concave 3-edge corners if convex ships early

### P2.3 — Non-Planar Fillet *(alternative if vertex blending is blocked)*

Both `fillet()` and `fillet_rolling_ball()` reject non-planar faces.

- [ ] Rolling-ball algorithm for cylinder–plane junctions
- [ ] Cylinder–cylinder junctions (common at bin corners)
- [ ] Test: fillet edges of a cylinder → smooth blend between cap and wall

### P2.4 — Healing & Validation Hardening *(stretch goal)*

- [ ] `repair_solid()` convenience function chaining healing passes
- [ ] Auto-heal small gaps (< tolerance) in wire closure
- [ ] Detect and warn on degenerate faces (area < tolerance²)
- [ ] Opt-in `validate_solid()` pass (behind feature flag to avoid perf overhead)

### Phase 2 Exit Criteria

| Metric                          | Target                                        |
|---------------------------------|-----------------------------------------------|
| Boolean pass rate               | > 95% on 100-case stress suite                |
| Fillet box edges                | 0 boundary edges (if P2.2 completed)          |
| Fillet cylinder edges           | Smooth cap-to-wall blend (if P2.3 completed)  |
| No silent boolean degradation   | Telemetry logs path for every boolean call    |

---

## Phase 3: Performance & Polish (July–August 2026)

Optimize for interactive web apps and prepare for public launch.

### Dependencies

```
P3.1 (perf) depends on P1.6 (baseline)
P3.2 (bundle) ── parallel with P3.1
P3.3 (docs) ──→ P3.4 (npm publish)
```

### P3.1 — Performance Optimization

- [ ] Profile WASM execution against Phase 1 baseline
- [ ] Optimize hot paths identified in baseline
- [ ] Target: 1×1×1 gridfinity bin < 100ms in browser
- [ ] Target: 4×4 baseplate with magnet holes < 500ms

### P3.2 — WASM Bundle Optimization

Current: 1.9 MB raw / 602 KB gzip'd.

- [ ] `wasm-opt -O3 --strip-debug` for release builds
- [ ] Dead code elimination via `cargo features` gating (e.g., IGES behind feature flag)
- [ ] Evaluate core + io bundle split
- [ ] Target: < 400 KB gzip'd for core-only bundle

### P3.3 — API Reference & Examples

- [ ] TypeDoc API reference for [brepjs](https://github.com/andymai/brepjs)
- [ ] 10 curated examples:
  - Basic box with fillet
  - Gridfinity bin (showcase)
  - Boolean operations (union, cut, intersect)
  - Sweep along a path
  - STEP import → modify → STL export
  - Pattern (linear, circular, grid)
  - Measurement (volume, area, center of mass)
  - Transform and mirror
  - Multi-solid assembly
  - Custom profile extrusion
- [ ] Static site with copy-paste code snippets (live WASM playground deferred)
- [ ] README badges (npm version, bundle size, license)

### P3.4 — npm Package Preparation

- [ ] Finalize dual-license text + LICENSE file
- [ ] `package.json` metadata (description, keywords, repository)
- [ ] CHANGELOG.md with semver history
- [ ] Publish v1.0.0-beta.1 to npm

### Phase 3 Exit Criteria

| Metric            | Target                                  |
|-------------------|-----------------------------------------|
| Bin generation    | < 100ms for 1×1 bin (vs. Phase 1 baseline) |
| WASM bundle size  | < 400 KB gzip'd (core-only)             |
| npm published     | v1.0.0-beta.x with TypeDoc reference    |
| Examples live     | 10 examples on static site              |

---

## Out of Scope (This Cycle)

| Item                                      | Reason                                              |
|-------------------------------------------|------------------------------------------------------|
| Mesh-only workflows (sculpting, subdiv)   | Not B-Rep; different audience                        |
| Desktop/native distribution (crates.io)   | Focus on npm/WASM                                    |
| Simulation (FEA/CFD)                      | Different product category                           |
| Parametric history / feature tree         | Can be built at brepjs layer later                   |
| General NURBS-to-NURBS fillet             | Convex corners + cylinder junctions cover gridfinity |
| BREP/SAT import                           | STEP covers interop needs                            |
| Concave vertex blending                   | After convex case proves out                         |

---

## Success Criteria (September 2026)

| Metric                        | Target                                             |
|-------------------------------|----------------------------------------------------|
| gridfinity-layout-tool on brepkit | Live, generating printable bins (no OpenCascade) |
| Boolean pass rate             | > 95% on 100-case stress suite                     |
| npm published                 | v1.0.0-beta.x with docs                           |
| First external user           | >= 1 developer building with brepjs                |
| WASM bundle size              | < 400 KB gzip'd (core-only)                       |
| Bin generation perf           | < 100ms for 1×1 bin in browser                    |
| Gridfinity operation coverage | All 14 operations in appendix at Ready status      |

---

## Risk Register

| Risk                                          | Likelihood | Impact   | Mitigation                                                           |
|-----------------------------------------------|------------|----------|----------------------------------------------------------------------|
| Stacking lip sweep (`withContact`) wrong geometry | High   | Critical | P1.1 audit; implement `withContact` in brepkit if needed             |
| Ruled lofts produce wrong surface (smooth not ruled) | High | High | P1.1 audit; pass `ruled` through adapter to brepkit `loft()` |
| `shell` fails on gridfinity box geometry      | Medium     | Critical | Test shell(open-top box) early in P1.1; boolean-cut fallback         |
| Fillet vertex blending harder than expected    | High       | High     | Scope to convex 3-edge only; ship with chamfers until ready          |
| Adapter drops more options than cataloged      | High       | High     | Systematic P1.1 audit with per-method integration tests              |
| Boolean coplanar handling causes regressions   | Medium     | High     | Comprehensive test suite before changing                             |
| WASM bundle still >500 KB after optimization   | Medium     | Medium   | Feature-gate I/O modules; offer core-only bundle                     |
| Non-commercial license deters adoption         | Low        | Medium   | Clear license FAQ; fast response on commercial inquiries             |
| Solo capacity bottleneck                       | High       | Medium   | AI-assisted dev; prioritize ruthlessly; P2.2 vs P2.3 is pick-one    |

---

## Appendix: Gridfinity Operation Map

Operations required by [gridfinity-layout-tool](https://github.com/andymai/gridfinity-layout-tool):

| Gridfinity Feature             | brepkit Operation                        | Status       | Notes                                        |
|--------------------------------|------------------------------------------|--------------|----------------------------------------------|
| Bin body (rounded box)         | `make_box` + `fillet`                    | Partial | Vertex blending missing (P2.2)               |
| Stacking lip profile           | `sweepSketch` with `withContact`         | At Risk | Adapter drops `withContact` (P1.1)           |
| Shell (hollow box)             | `shell` (remove top face, offset inward) | Unverified | Open-top-box case untested                   |
| Base foot (chamfered step)     | `chamfer` or `extrude` + `boolean cut`   | Ready   | —                                            |
| Magnet pockets                 | `make_cylinder` + `boolean cut`          | Ready   | —                                            |
| Screw holes                    | `make_cylinder` + `boolean cut`          | Ready   | —                                            |
| Internal dividers              | `make_box` + `boolean cut`               | Ready   | —                                            |
| Label slot (45 deg cut)        | `extrude` + `boolean cut`                | Ready   | —                                            |
| Scoop (cylindrical cut)        | `make_cylinder` + `boolean cut`          | Ready   | —                                            |
| Baseplate grid                 | `grid_pattern` (P1.4)                    | Planned | Currently requires two `linear_pattern` calls|
| STL export                     | `write_stl`                              | Fix in P1.3 | Switching to shared-edge tessellation        |
| Socket/pocket ruled lofts      | `loftWith({ ruled: true })`              | At Risk | Adapter drops `ruled` option (P1.1)          |
| Edge visualization             | `meshEdges()`                            | Broken  | Adapter returns stub (P1.1)                  |
| 3MF export                     | `write_3mf`                              | Ready   | Watertight                                   |
