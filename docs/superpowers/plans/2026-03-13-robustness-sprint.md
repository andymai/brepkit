# Robustness Sprint Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-impact open robustness items from the Tier 2/3 roadmap: fix the Cyrus-Beck concave polygon bug, add post-boolean healing, implement relative vertex merge, add analytic area for cone/torus, and clean up outdated comments.

**Architecture:** All changes are isolated to individual crates — no cross-crate ripples. Each task targets a single file or function. Cyrus-Beck fix is the highest-correctness risk (silent wrong results on curved faces); vertex merge and healing are robustness improvements. All follow TDD: write test first, verify failure, implement, verify pass.

**Tech Stack:** Rust, `cargo test`, `cargo clippy`, `thiserror`, arena topology (`brepkit_topology`), existing CDT/winding infra.

---

## Task 1: Fix outdated ear-clipping doc comments

**Files:**
- Modify: `crates/operations/src/tessellate.rs` (lines 233, 2869, 2975)

- [ ] **Step 1: Locate the three comments**

```bash
grep -n "ear.clip" crates/operations/src/tessellate.rs
```

Expected: 3 hits near lines 233, 2869, 2975.

- [ ] **Step 2: Update each comment**

Replace:
- Line ~233: `"Tessellate a planar face via ear-clipping triangulation."` → `"Tessellate a planar face using CDT (Constrained Delaunay Triangulation), with fan-triangulation fallback."`
- Line ~2869: any mention of "ear-clipping" → CDT equivalent
- Line ~2975: `"Simple ear-clip for faces without holes."` → `"Triangulate faces without holes via CDT with fan fallback."`

- [ ] **Step 3: Verify tests still pass**

```bash
cargo test -p brepkit-operations -- tessellate 2>&1 | tail -5
```

Expected: all pass, no warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "docs(operations): fix outdated ear-clipping comments — CDT is already used"
```

---

## Task 2: Fix Cyrus-Beck called on non-convex polygons

**Background:** `plane_plane_chord_analytic()` in `boolean.rs` calls `cyrus_beck_clip()` on face boundary polygons (lines 7166–7167). These polygons come from `face_polygon()` which samples arbitrary trimmed surfaces — they are frequently non-convex. `cyrus_beck_clip()` is documented as convex-only. The correct function is `polygon_clip_intervals()` which handles concave cases via winding-number interval testing. This is a silent correctness bug affecting boolean operations on curved/concave faces.

**Files:**
- Modify: `crates/operations/src/boolean.rs` (lines ~7166–7200)
- Test: `crates/operations/src/boolean.rs` (test module at bottom)

- [ ] **Step 1: Write a failing test for concave face boolean**

Add to the test module in `boolean.rs`:

```rust
#[test]
fn test_boolean_cut_concave_face() {
    // L-shaped prism cut by a box — the L-shape creates a concave face
    // that will be incorrectly clipped by Cyrus-Beck
    use crate::primitives::make_box;
    use brepkit_topology::Topology;

    let mut topo = Topology::new();
    // Build L-shaped solid (two boxes unioned)
    let box1 = make_box(&mut topo, 20.0, 10.0, 5.0).unwrap();
    let box2 = make_box(&mut topo, 10.0, 10.0, 5.0).unwrap();
    // ... union and cut
    // Verify volume is correct to within 1%
}
```

> Note: The exact test geometry depends on what `make_box` + `boolean_solid` signatures accept. Check existing boolean tests in `boolean.rs` for the pattern (search for `#[test]` near the bottom of the file).

- [ ] **Step 2: Run to see current behavior**

```bash
cargo test -p brepkit-operations -- test_boolean_cut_concave_face 2>&1 | tail -20
```

- [ ] **Step 3: Understand `plane_plane_chord_analytic` signature**

Read `boolean.rs` around line 7100 to understand:
- What does `cyrus_beck_clip` return? → `Option<(f64, f64)>` (single interval)
- What does `polygon_clip_intervals` return? → `Vec<(f64, f64)>` (multiple intervals)
- How is the result used after line 7167?

The caller uses the result to find the chord endpoints. With multiple intervals, we want the **outermost** interval (min of all t_min, max of all t_max) for a single chord, OR we generate one chord segment per interval.

- [ ] **Step 4: Replace `cyrus_beck_clip` calls with `polygon_clip_intervals`**

In `plane_plane_chord_analytic()` (around line 7166):

```rust
// Before:
let t_range_a = cyrus_beck_clip(&line_origin, &line_dir, verts_a, &normal_a, tol);
let t_range_b = cyrus_beck_clip(&line_origin, &line_dir, verts_b, &normal_b, tol);

let (t_min, t_max) = match (t_range_a, t_range_b) {
    (Some(a), Some(b)) => (a.0.max(b.0), a.1.min(b.1)),
    _ => return None,
};
```

```rust
// After:
let intervals_a = polygon_clip_intervals(&line_origin, &line_dir, verts_a, tol);
let intervals_b = polygon_clip_intervals(&line_origin, &line_dir, verts_b, tol);

// Intersect the union of intervals from A with the union from B
// For a chord clipped to both faces, take the overlap of their combined extents
let (t_min_a, t_max_a) = match intervals_a.iter().copied().reduce(|(lo1,hi1),(lo2,hi2)| (lo1.min(lo2), hi1.max(hi2))) {
    Some(r) => r,
    None => return None,
};
let (t_min_b, t_max_b) = match intervals_b.iter().copied().reduce(|(lo1,hi1),(lo2,hi2)| (lo1.min(lo2), hi1.max(hi2))) {
    Some(r) => r,
    None => return None,
};
let t_min = t_min_a.max(t_min_b);
let t_max = t_max_a.min(t_max_b);
if t_min >= t_max { return None; }
```

> **Note:** Read the full function body before making this change — the exact shape of the existing match arms and return types may differ. Check if `polygon_clip_intervals` signature matches: it likely takes `(&Point3, &Vec3, &[Point3], Tolerance)` — verify against line 1330 of boolean.rs.

- [ ] **Step 5: Verify clippy is clean**

```bash
cargo clippy -p brepkit-operations -- -D warnings 2>&1 | grep -E "error|warning" | head -20
```

- [ ] **Step 6: Run full boolean tests**

```bash
cargo test -p brepkit-operations -- boolean 2>&1 | tail -10
```

Expected: all pass (no regressions from convex cases).

- [ ] **Step 7: Commit**

```bash
git add crates/operations/src/boolean.rs
git commit -m "fix(operations): replace cyrus_beck_clip with polygon_clip_intervals for non-convex faces

plane_plane_chord_analytic was calling cyrus_beck_clip on face boundary polygons
from trimmed curved surfaces, which are frequently non-convex. This silently
produced incorrect chord extents. Replaced with polygon_clip_intervals which
handles concave faces via winding-number interval testing."
```

---

## Task 3: Add post-boolean shape healing option

**Background:** After boolean assembly, `boolean.rs` currently calls `remove_degenerate_edges()` and optionally `unify_faces()`. The full `heal_solid()` pipeline (`close_wire_gaps`, `merge_coincident_vertices`, `remove_degenerate_edges`, `remove_small_faces`, `remove_duplicate_faces`, `fix_face_orientations`) is not called, because it can be expensive and may corrupt small features. The fix: add a `heal_after_boolean: bool` field to `BooleanOptions` (default `false`), analogous to the existing `unify_faces` field. When set, run `heal_solid()` on the result.

**Files:**
- Modify: `crates/operations/src/boolean.rs` (BooleanOptions struct, post-assembly block)
- Modify: `crates/wasm/src/kernel.rs` (if BooleanOptions is exposed to WASM)
- Test: `crates/operations/src/boolean.rs` (test module)

- [ ] **Step 1: Write a failing test**

Find a case where `heal_solid` would fix the result. Look at the existing test that uses `unify_faces` as a model. Add a test like:

```rust
#[test]
fn test_boolean_heal_after() {
    // After a complex boolean, validate_solid should pass when heal_after_boolean=true
    // even if it fails with false
    let mut opts = BooleanOptions::default();
    opts.heal_after_boolean = true; // This field doesn't exist yet — test will fail to compile
    // ...
}
```

- [ ] **Step 2: Locate `BooleanOptions` struct**

```bash
grep -n "BooleanOptions" crates/operations/src/boolean.rs | head -10
```

- [ ] **Step 3: Add the field to `BooleanOptions`**

Add `heal_after_boolean: bool` next to the existing `unify_faces: bool` field, with `default = false`.

```rust
pub struct BooleanOptions {
    // ... existing fields ...
    /// If true, run shape healing on the boolean result.
    /// Useful for final results; avoid for intermediates fed into further booleans.
    pub heal_after_boolean: bool,
}
```

Update `Default` impl to set `heal_after_boolean: false`.

- [ ] **Step 4: Wire into post-assembly block**

Find the post-assembly block (around line 691 where `remove_degenerate_edges` is called). After the existing healing, add:

```rust
if opts.heal_after_boolean {
    let _ = crate::heal::heal_solid(topo, result, tol.linear)?;
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p brepkit-operations -- boolean 2>&1 | tail -10
```

- [ ] **Step 6: Check WASM exposure**

```bash
grep -n "BooleanOptions\|unify_faces" crates/wasm/src/kernel.rs | head -10
```

If `BooleanOptions` fields are mapped from JS, add `heal_after_boolean` to the WASM mapping. If it's not exposed, no change needed.

- [ ] **Step 7: Commit**

```bash
git add crates/operations/src/boolean.rs
git commit -m "feat(operations): add heal_after_boolean option to BooleanOptions

Adds opt-in post-boolean shape healing via heal_solid(). Default false to
preserve existing behavior and avoid corrupting intermediates fed into
further boolean operations."
```

---

## Task 4: Relative vertex merge threshold in boolean

**Background:** Boolean vertex deduplication uses `resolution = 1.0 / tol.linear` — a fixed grid at 1e-7 scale regardless of model size. This causes drift in sequential booleans on large models (10m+ scale) and over-merges on small models (sub-mm). Fix: before assembly, compute the bounding box of all input vertices and scale the resolution to `1.0 / (bbox_diagonal * relative_factor)` where `relative_factor` defaults to `1e-7`. Fall back to absolute tolerance if bbox is degenerate.

**Files:**
- Modify: `crates/operations/src/boolean.rs` (assembly/quantization block around line 3060)
- Test: `crates/operations/src/boolean.rs` (test module)

- [ ] **Step 1: Understand current quantization**

Read lines 3055–3100 in `boolean.rs`:
- Where is `resolution` computed?
- Where is `quantize_point(p, resolution)` called?
- What are the input vertices at that point in the code?

- [ ] **Step 2: Write a failing test**

A test with large-scale geometry (100m bounding box) where the fixed 1e-7 grid causes incorrect vertex merging. Compare vertex count between fixed and relative approaches:

```rust
#[test]
fn test_boolean_large_scale_no_vertex_collapse() {
    // Make two boxes at 100m scale, cut one with the other
    // With fixed 1e-7 grid: vertices near 100m distance may quantize to same cell
    // With relative grid: should preserve distinct vertices
    // ...
    // Assert resulting face count is sane (not blown up from merged/missing vertices)
}
```

- [ ] **Step 3: Compute bbox before quantization**

Before the `resolution = 1.0 / tol.linear` line, collect all vertex positions and compute the diagonal:

```rust
// Compute scale-relative merge resolution
let all_pts: Vec<Point3> = all_input_vertices.iter().copied().collect();
let resolution = if all_pts.len() >= 2 {
    let bbox = brepkit_math::aabb::Aabb::from_points(&all_pts);
    let diagonal = (bbox.max - bbox.min).length();
    if diagonal > tol.linear {
        1.0 / (diagonal * 1e-7_f64)
    } else {
        1.0 / tol.linear // fallback for degenerate/point models
    }
} else {
    1.0 / tol.linear
};
```

> **Check:** Confirm `brepkit_math::aabb::Aabb::from_points` exists and its API. Search `crates/math/src/aabb.rs` for the correct constructor. Adjust as needed.

- [ ] **Step 4: Run tests**

```bash
cargo test -p brepkit-operations -- boolean 2>&1 | tail -10
```

All existing boolean tests must pass — the default behavior should be equivalent for typical unit-scale geometry.

- [ ] **Step 5: Commit**

```bash
git add crates/operations/src/boolean.rs
git commit -m "fix(operations): scale boolean vertex merge resolution to model bounding box

Fixed 1e-7 grid caused drift in sequential booleans at large scales and
over-merging at sub-mm scales. Resolution now adapts to bbox diagonal with
1e-7 relative factor. Falls back to absolute tolerance for degenerate models."
```

---

## Task 5: Analytic area for cone and torus faces

**Background:** `measure.rs` computes face area by tessellation for all surface types. Volume was fixed for cone/sphere/torus in the boolean audit (Phase 1). Area still uses tessellation for cone and torus. Fix: add `analytic_cone_face_area()` and `analytic_torus_face_area()` using closed-form formulas, wire into `face_area()`.

**Files:**
- Modify: `crates/operations/src/measure.rs`
- Test: `crates/operations/src/measure.rs` (test module)

- [ ] **Step 1: Understand existing analytic volume pattern**

Read `measure.rs` to find:
- `analytic_cone_signed_volume()` — what inputs does it take? (ConicalSurface, angular range?)
- `analytic_torus_signed_volume()` — same
- `face_area()` function — how does it dispatch on FaceSurface variants?

```bash
grep -n "analytic_cone\|analytic_torus\|face_area\|FaceSurface" crates/operations/src/measure.rs | head -30
```

- [ ] **Step 2: Write failing tests**

```rust
#[test]
fn test_cone_face_area_analytic() {
    // A full cone (r=1, h=1): lateral area = π * r * slant = π * 1 * √2 ≈ 4.443
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 1.0, 0.0, 1.0).unwrap();
    // Get the lateral face (not caps)
    let area = solid_surface_area(&mut topo, cone, 0.01).unwrap();
    // Full cone surface = lateral + base = π*r*√2 + π*r² = π(√2 + 1) ≈ 7.584
    assert!((area - std::f64::consts::PI * (2.0_f64.sqrt() + 1.0)).abs() < 0.001);
}

#[test]
fn test_torus_face_area_analytic() {
    // Full torus (R=2, r=0.5): area = 4π²Rr = 4π²*2*0.5 = 4π² ≈ 39.478
    let mut topo = Topology::new();
    let torus = make_torus(&mut topo, 2.0, 0.5).unwrap();
    let area = solid_surface_area(&mut topo, torus, 0.01).unwrap();
    assert!((area - 4.0 * std::f64::consts::PI.powi(2) * 2.0 * 0.5).abs() < 0.01);
}
```

- [ ] **Step 3: Run to see tessellation vs analytic error**

```bash
cargo test -p brepkit-operations -- test_cone_face_area test_torus_face_area 2>&1 | tail -20
```

- [ ] **Step 4: Implement `analytic_cone_face_area`**

Lateral area of a partial cone (angular range [u0, u1], v range [v_min, v_max]):
- Slant height = `(v_max - v_min) / cos(half_angle)` (half_angle from radial plane)
- Mean radius = average of radius at v_min and v_max
- Area = `(u1 - u0) / (2π) × π × (r_min + r_max) × slant`

For full cone (u0=0, u1=2π): `π × (r_min + r_max) × slant`

```rust
fn analytic_cone_face_area(surf: &ConicalSurface, u_range: (f64, f64), v_range: (f64, f64)) -> f64 {
    let (u0, u1) = u_range;
    let (v0, v1) = v_range;
    let angle_frac = (u1 - u0) / (std::f64::consts::TAU);
    // radius at v is v * cos(half_angle) for ConicalSurface parameterization
    // check surface.rs for exact formula: P(u,v) = apex + v*(cos(a)*radial(u) + sin(a)*axis)
    // radial distance at v = v * cos(half_angle)
    let r0 = v0 * surf.half_angle().cos();
    let r1 = v1 * surf.half_angle().cos();
    let slant = (v1 - v0).abs(); // v parameterizes along slant directly
    std::f64::consts::PI * (r0 + r1) * slant * angle_frac
}
```

> **Important:** Check `crates/math/src/surfaces.rs` for the exact `ConicalSurface` parameterization (`half_angle` definition per the PR #148 fix: angle is from radial plane). Adjust formula if needed.

- [ ] **Step 5: Implement `analytic_torus_face_area`**

Full torus area = `4π² × R × r`. For partial (u in [u0,u1], v in [v0,v1]):
```
area = (u1-u0) × (v1-v0) × R × r  (parametric area element = R×r dθ dφ, from first fundamental form)
```
Wait — check `ToroidalSurface` in `surfaces.rs` for the exact parameterization and whether it includes `R + r×cos(v)` terms. The standard torus surface element is `(R + r cos(v)) × r dθ dv`, so the area integral is:
```
area = r × ∫[v0,v1] ∫[u0,u1] (R + r cos(v)) dθ dv
     = r × (u1-u0) × [R(v1-v0) + r(sin(v1)-sin(v0))]
```

- [ ] **Step 6: Wire into `face_area()` dispatch**

In `face_area()`, add arms for `FaceSurface::Cone` and `FaceSurface::Torus` before the tessellation fallback, following the same pattern as the volume computation.

- [ ] **Step 7: Run tests**

```bash
cargo test -p brepkit-operations -- measure 2>&1 | tail -10
```

- [ ] **Step 8: Commit**

```bash
git add crates/operations/src/measure.rs
git commit -m "feat(operations): add analytic area computation for cone and torus faces

Replaces tessellation-based area with closed-form formulas for ConicalSurface
and ToroidalSurface faces. Matches existing analytic volume pattern. Completes
boolean audit Phase 1 measurement fixes."
```

---

## Task 6: Verify and test winding checks in extrude/sweep/revolve

**Background:** Investigation found all three already have winding checks (extrude.rs:451, sweep.rs:449, revolve.rs:235). This task verifies they handle edge cases correctly and adds regression tests.

**Files:**
- Test: `crates/operations/src/extrude.rs`, `sweep.rs`, `revolve.rs` (test modules)

- [ ] **Step 1: Write CW-winding test for extrude**

Find in `extrude.rs` how wires are passed. Create a wire with vertices in CW order, extrude it, check the resulting solid has positive volume:

```rust
#[test]
fn test_extrude_cw_profile_produces_valid_solid() {
    // CW square wire (vertices going clockwise when viewed from +Z)
    // Extrude should still produce a valid solid with correct orientation
    let mut topo = Topology::new();
    // Build CW wire: (0,0), (0,1), (1,1), (1,0) — reversed order from CCW
    // ...
    let solid = extrude_wire(&mut topo, wire_id, Vec3::Z * 1.0, &Extrude::default()).unwrap();
    let vol = solid_volume(&mut topo, solid, 0.01).unwrap();
    assert!(vol > 0.0, "CW profile should produce positive-volume solid, got {}", vol);
}
```

- [ ] **Step 2: Same for sweep**

- [ ] **Step 3: Same for revolve**

- [ ] **Step 4: Run all three tests**

```bash
cargo test -p brepkit-operations -- test_extrude_cw test_sweep_cw test_revolve_cw 2>&1 | tail -20
```

If they all pass immediately, the winding checks are working. Commit the tests as regression coverage.

- [ ] **Step 5: Commit**

```bash
git add crates/operations/src/extrude.rs crates/operations/src/sweep.rs crates/operations/src/revolve.rs
git commit -m "test(operations): add CW-winding regression tests for extrude, sweep, revolve

Verifies existing winding checks (PR #182 pattern) handle reversed profiles
correctly. All three ops already had checks; tests confirm they work."
```

---

## Execution Order

Tasks are independent. Suggested order by impact/risk:

1. **Task 2** (Cyrus-Beck) — correctness bug, highest impact
2. **Task 5** (analytic area) — completes boolean audit Phase 1
3. **Task 3** (post-boolean heal) — small, clean addition
4. **Task 4** (relative vertex merge) — careful with existing test suite
5. **Task 6** (winding verification) — likely passes immediately
6. **Task 1** (comment fix) — trivial, do last

After completing all 6:
- Update `brepkit-robustness-roadmap.md` to mark #13 (Cyrus-Beck) and #17 (heal) as ✅
- Update `brepkit-boolean-audit-2026-03-12.md` to mark the two remaining Phase 1 items ✅
- Re-run the gridfinity dual-kernel benchmark suite to validate measurement improvements
