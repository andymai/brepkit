# brepkit-wasm — brepjs Compatibility Status (2026-03-08)

brepkit-wasm version: **0.7.1**
Comparison baseline: **OCCT** via brepjs-opencascade
Environment: Node.js 24, Linux x86_64, 5 iterations per benchmark

---

## Performance (v0.7.1 vs OCCT)

brepkit is faster than OCCT across all benchmarks:

| Operation | OCCT (ms) | brepkit (ms) | Speedup |
|-----------|-----------|-------------|---------|
| makeBox | 5.9 | 0.3 | **20x** |
| makeCylinder | 2.4 | 0.1 | **24x** |
| makeSphere | 1.5 | 0.6 | **2.5x** |
| fuse(box,box) | 82.7 | 1.3 | **64x** |
| cut(box,cyl) | 124.6 | 11.8 | **11x** |
| intersect(box,sphere) | 108.3 | 4.9 | **22x** |
| translate ×1000 | 69.2 | 18.7 | **3.7x** |
| rotate ×100 | 7.1 | 1.8 | **3.9x** |
| mesh sphere (fine) | 62.2 | 0.8 | **78x** |
| volume ×100 | 7.9 | 2.0 | **4x** |
| exportSTEP ×10 | 18.9 | 0.9 | **21x** |
| multi-boolean | 52.1 | 12.7 | **4.1x** |

Previous regressions (cut: 626ms, multi-boolean: 18,511ms in v0.6.0) are fully resolved
via analytic sphere boolean with O(1) classification and edge deduplication fixes.

---

## Test Coverage

- **3897 tests pass**, 11 skipped, 0 failures across 225 test files
- 6 skipped: IGES not in WASM (4), split with shape tools (1), double fillet (1)

---

## Feature Gaps

### Gap 1: `meshEdges` returns empty (stub)

**Severity:** Low — wireframe rendering only

`BrepkitAdapter.meshEdges()` returns empty arrays. brepkit doesn't expose per-edge
tessellation in WASM. Wireframe/edge overlay rendering unavailable with brepkit kernel.

**Fix:** Expose `tessellate_edge()` in WASM that returns a polyline per edge.

### Gap 2: IGES not exposed in WASM (4 skipped tests)

**Severity:** Low — IGES reader/writer exists in Rust but not in WASM bindings.

**Fix:** Add `importIGES`/`exportIGES` to `kernel.rs` WASM bindings.

### Gap 3: Split with arbitrary shape tools (1 skipped test)

**Severity:** Medium

brepkit's `split()` only accepts a planar face (converts to plane). OCCT's
`BRepAlgoAPI_Splitter` supports arbitrary shapes as cutting tools.

**Fix:** Extend `split.rs` to accept solid/shell tools, not just planes.

### Gap 4: Double/sequential fillet (1 skipped test)

**Severity:** Medium

Filleting edges adjacent to NURBS blend faces (from a prior fillet) fails.
Single fillet works fine; the issue is vertex blending at NURBS junctions.

### Gap 5: Variable-radius fillet

**Severity:** Medium — fully unimplemented

brepkit's `fillet()` only supports constant radius. OCCT supports varying
radius along an edge via `BRepFilletAPI_MakeFillet::Add(r1, r2, edge)`.

### Gap 6: Curved geometry measurement accuracy

**Severity:** Medium — partially fixed

Primitive measurement is now exact (cone, sphere, cylinder all 0.00% error).

Boolean cut(box,cyl) with centered cylinder: **fixed** (0.13% error, was 26% error).
Root cause was `create_band_fragments` computing the band normal from the polygon
centroid, which falls on the cylinder axis for full-circle bands, yielding a
degenerate zero-length direction. Ray-casting with this direction misclassified
the bore fragment as Outside, dropping the cylinder bore face from the result.

Remaining issue: cylinder at box corner (chord-splitting path) still produces
incorrect volumes. This affects partial-overlap boolean configurations.

### Gap 7: Assembly STEP export

**Severity:** Low

`exportSTEPAssembly()` falls back to concatenating shapes — loses part names/colors.

### Gap 8: Excluded test suites (use raw OCCT API for setup)

These test files are excluded from brepkit runs because they construct inputs via
raw OCCT classes (`gp_Pnt_3`, `BRepBuilderAPI_MakeEdge`, etc.). The underlying
operations are brepkit-supported — the gap is test infrastructure, not features.

| Test file | Operations covered |
|-----------|-------------------|
| fn-booleanFns | section, split, slice, fuseAll, cutAll |
| fn-extrudeFns | complexExtrude, twistExtrude, supportExtrude |
| fn-guidedSweepFns | guidedSweep (profile along spine) |
| fn-hullFns | convex hull |
| fn-minkowskiFns | Minkowski sum, offset |
| fn-multiSweepFns | multi-section sweep / loft |
| fn-measureFns | curvatureAt, volumeProps, surfaceProps |
| fn-meshFns | meshEdges |
| fn-batchOps | measureBulk, transformBatch |

### Gap 9: Fully unimplemented features

| Feature | Test file | What it does |
|---------|-----------|-------------|
| Blueprint/2D | fn-blueprintFns | 2D shape ops, SVG path, bounds/orientation |
| 2D wire offset | fn-offsetWire2D | Wire offset with chamfer/arc/intersection joins |
| Section to face | fn-sectionToFace | Fill section plane cuts into faces |
| Playground examples | fn-examples | Validate example code snippets against API |

---

## Priority Order

1. **meshEdges** — unblocks wireframe rendering, low effort
2. **IGES WASM bindings** — code exists, just needs wiring
3. **Measurement accuracy** — verify on v0.7.1, add analytic formulas if needed
4. **Variable fillet** — significant feature for real-world CAD
5. **Split with shape tools** — extend existing split operation
6. **Blueprint/2D operations** — new feature area
