# brepkit — Product Roadmap (Q1–Q3 2026)

> **Vision**: The browser-native B-Rep geometry SDK. Exact solid modeling
> in any web app — no server, no legacy C++ dependencies.
>
> **Dogfood**: [gridfinity-layout-tool](https://github.com/andymai/gridfinity-layout-tool) —
> parametric gridfinity bin generator. Migrating from `brepjs-opencascade` to
> [brepjs](https://github.com/andymai/brepjs) + [brepkit-wasm](https://github.com/andymai/brepkit/tree/main/crates/wasm).
>
> **License**: Apache-2.0

---

## Current State (March 2026)

| Metric         | Value                                            |
|----------------|--------------------------------------------------|
| Tests          | 892 passing                                      |
| WASM bindings  | ~145                                             |
| CAD operations | ~25 + helpers                                    |
| I/O formats    | 7 (STEP, IGES, STL, 3MF, OBJ, PLY, glTF)        |
| PRs merged     | 48                                               |
| WASM binary    | Full: 1.3 MB / 516 KB gzip'd · Core-only: 953 KB / 369 KB gzip'd (with wasm-opt) |
| Version        | v0.4.0 (release-please managed)                  |
| Publish        | release-please + GitHub Actions → npm (needs NPM_TOKEN refresh) |

---

## Phase 1: Migration & Quick Wins — ✅ COMPLETE

Merged as PR #40 (consolidated).

| Item | Status | PR | Notes |
|------|--------|-----|-------|
| P1.1 Adapter Audit | ✅ | #40 | `sweepPipeShell`, `loftAdvanced`, boolean options fixed |
| P1.2 OCCT Migration | ✅ | #40 | brepjs adapter points to brepkit-wasm |
| P1.3 STL Watertight | ✅ | #40 | STL writer uses `tessellate_solid()` shared-edge mesh |
| P1.4 Grid Pattern | ✅ | #40 | `grid_pattern()` + WASM `gridPattern()` binding |
| P1.5 Integration Tests | ✅ | #40 | Gridfinity bin + baseplate end-to-end tests |
| P1.6 Perf Baseline | ✅ | #40 | Criterion benchmarks, brepkit wins 13/13 vs OCCT |

---

## Phase 2: Robustness — ✅ COMPLETE

| Item | Status | PR | Notes |
|------|--------|-----|-------|
| P2.1 Boolean Reliability | ✅ | #42 | 100-case stress suite, coplanar fix, configurable deflection (0.1 default), degenerate result errors. Pass rate >95%. |
| P2.2 Fillet Vertex Blend | ✅ | #43 | Convex 3-edge corners: planar blend patches with angular vertex ordering. Fillet all 12 box edges → 26 faces (6 planar + 12 NURBS + 8 blend). |
| P2.3 Non-Planar Fillet | ⏭️ Skipped | — | P2.2 shipped; P2.3 was pick-one alternative. |
| P2.4 Healing & Validation | ✅ | #44 | `repair_solid()` chains validate→heal→validate. Wire closure check, degenerate face area check. `RepairReport` struct. WASM `repairSolid` binding. |

### Implementation details for future agents

**Boolean deflection** (`boolean.rs:37`): `DEFAULT_BOOLEAN_DEFLECTION` changed from 1.0→0.1 in P2.1. This improves accuracy but caused 3-5× regression on curved-surface booleans (cut box-cyl: 4.4ms→26.7ms). Acceptable trade-off — still within interactive budgets.

**Vertex blend dedup** (`fillet.rs` Phase 5b): At a 3-edge box corner, 6 contact entries collapse to 3 unique positions. Dedup MUST use spatial proximity, NOT face index (each face has 2 different contact points from 2 different fillet edges).

**`transform_solid` returns `()`** — mutates in place, does NOT return a new SolidId.

**`solid_volume` takes 3 args**: `(topo, solid, deflection)`.

---

## Phase 3: Performance & Polish — ✅ COMPLETE

| Item | Status | PR | Notes |
|------|--------|-----|-------|
| P3.1 Performance | ✅ | #45 | All targets met. 1×1 bin: 33µs (3,030× under 100ms target). 3×3 baseplate: 27.5ms (18× under 500ms). Fixed `intersect(box,sphere)` bench panic. |
| P3.2 WASM Bundle | ✅ | #46 | Optional `io` feature on brepkit-wasm. Core-only (no IO): 369 KB gzip'd with wasm-opt. Full: 516 KB. Target was <400 KB core-only. |
| P3.3 Examples | ✅ | #47 | 10 CI-verified examples in `crates/operations/tests/examples.rs`. Covers fillet, booleans, patterns, measurement, transforms, tessellation. |
| P3.4 npm Prep | ✅ | #48 | Keywords, categories, description. release-please + publish.yml already handles CHANGELOG + npm publish. |

### Implementation details for future agents

**WASM feature gate**: `brepkit-wasm` has `features = ["io"]` (default). Build with `--no-default-features` for core-only. Each IO method in `kernel.rs` has `#[cfg(feature = "io")]`.

**wasm-opt command**: `wasm-opt -O3 --strip-debug -o output.wasm input.wasm` — strips function names (21% of binary) and applies WASM-specific optimizations.

**Bundle size breakdown** (raw, pre-wasm-opt):
- function names: 426 KB (stripped by wasm-opt)
- .rodata: 163 KB
- core::/alloc:: stdlib: 420 KB
- brepkit_operations: 312 KB
- brepkit_wasm: 129 KB
- brepkit_io + deps: 168 KB
- brepkit_math: 74 KB

---

## Release Status

| Version | Tag | Status |
|---------|-----|--------|
| v0.4.0 | release-please PR #41 merged | ⚠️ npm publish failed — NPM_TOKEN expired. Re-run workflow or `npm publish` manually from `crates/wasm/pkg/`. |

To publish manually:
```bash
wasm-pack build crates/wasm --target nodejs --release
cd crates/wasm/pkg
npm login
npm publish --access public
```

---

## Remaining Known Issues

| Issue | Severity | Context |
|-------|----------|---------|
| ~~`intersect(box,sphere)` returns Err~~ | ~~Medium~~ | ✅ Fixed — sphere now uses two-hemisphere topology with N-gon equatorial wire. PR #50. |
| ~~NURBS tessellation seam stitching~~ | ~~Medium~~ | ✅ Fixed — NURBS faces use CDT-constrained boundary tessellation for watertight seams. Analytic faces keep snap-based stitching. |
| Concave vertex blending | Low | Only convex 3-edge corners supported. Concave deferred. |
| NPM_TOKEN expired | Blocker for publish | GitHub Actions secret needs refresh. |

---

## Appendix: Gridfinity Operation Map

| Gridfinity Feature             | brepkit Operation                        | Status  |
|--------------------------------|------------------------------------------|---------|
| Bin body (rounded box)         | `make_box` + `fillet_rolling_ball`       | ✅ Ready |
| Stacking lip profile           | `sweepSketch` with `withContact`         | ✅ Fixed |
| Shell (hollow box)             | `shell` (remove top face, offset inward) | ✅ Verified |
| Base foot (chamfered step)     | `chamfer` or `extrude` + `boolean cut`   | ✅ Ready |
| Magnet pockets                 | `make_cylinder` + `boolean cut`          | ✅ Ready |
| Screw holes                    | `make_cylinder` + `boolean cut`          | ✅ Ready |
| Internal dividers              | `make_box` + `boolean cut`               | ✅ Ready |
| Label slot (45 deg cut)        | `extrude` + `boolean cut`                | ✅ Ready |
| Scoop (cylindrical cut)        | `make_cylinder` + `boolean cut`          | ✅ Ready |
| Baseplate grid                 | `grid_pattern`                           | ✅ Ready |
| STL export                     | `write_stl` (shared-edge tessellation)   | ✅ Fixed |
| Socket/pocket ruled lofts      | `loftWith({ ruled: true })`              | ✅ Fixed |
| Edge visualization             | `meshEdges()`                            | ✅ Fixed |
| 3MF export                     | `write_3mf`                              | ✅ Ready |
