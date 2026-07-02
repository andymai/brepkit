---
name: analytic-preservation
description: Use when a boolean, revolve, offset, or other operation degrades exact analytic geometry (planes, cylinders, cones, spheres, tori, circles) to NURBS or a mesh fallback, when face counts explode from a handful to hundreds, when running or interpreting the approx_census, when adding an exact analytic-analytic intersection arm, or when deciding whether an analytic-recovery task is worth chasing at all.
---

# Analytic Preservation

Core product doctrine: operations must preserve exact analytic geometry
(`FaceSurface::Plane/Cylinder/Cone/Sphere/Torus`, `EdgeCurve::Line/Circle/Ellipse`)
instead of degrading to NURBS approximation or the boolean mesh fallback.

Terms used below (GFA, FF phase, marcher, mesh fallback, SD, seam): glossary in
[reference.md](reference.md) section 6.

## Why it matters

- An analytic face is a few floats; a mesh-fallback face is baked triangles. Analytic
  surfaces can be shipped as parameters and meshed on the GPU per frame at adaptive LOD
  (`crates/render/src/compute_mesh.rs`). Baked triangles can never re-LOD.
- Analytic results are exactly measurable (closed-form volume, area) and tiny
  (single digits to tens of faces vs hundreds to thousands of planar facets).
- Analytic paths beat the mesh fallback by one to two orders of magnitude on the same
  inputs; the census prints per-row timings, so the margin is cheap to regenerate. One
  git-history example (#1006): box-sphere intersect went from 190 ms and 956 mesh faces
  to under 2 ms and 8 analytic faces. This margin is why brepkit beats the reference
  kernel head to head. Re-measure via the parity-benchmarking skill before quoting any
  number.

## Quick reference

```bash
# The census: which operations degrade, and via which path
cargo run --release --example approx_census -p brepkit-operations

# Where the boolean mesh fallback (the one path that loses analytic types) fires
rg -n 'analytic surface types will be lost' crates/operations/src/boolean/mod.rs

# All degradation probe sites (log target "brepkit_approx")
rg -l 'brepkit_approx' crates/
```

| Symptom | Likely cause | Where to look |
|---|---|---|
| Face count jumps from ~10 to hundreds, all planar | Boolean mesh fallback fired | `mesh_boolean_fallback` in `boolean/mod.rs`; the gate is `euler_ok && open_shell_ok && validate_boolean_result` (rg `open_shell_ok`) |
| Clean circular rim comes back as many short NURBS pieces | Generic marcher ran instead of an exact arm | `crates/math/src/analytic_intersection.rs`; see reference.md section 2 |
| Result has free edges after a curved boolean | Operands not split at the same crossing vertices, or a co-endpoint arc pair collapsed | reference.md sections 3c and 3d |
| Closed rim wrap-trimmed, out-of-domain evals | Rim emitted as closed NURBS instead of `EdgeCurve::Circle` | reference.md section 3a |
| Census says "exact analytic" but volume is wrong | The census only checks probes; it does not check correctness | See "The census caveat" below |

## Procedure: run and read the census

1. `cargo run --release --example approx_census -p brepkit-operations`
2. Expect one row per operation, shaped like:
   `boolean   sphere-cyl cut   0.84ms  faces=3   exact analytic`
   or `... FALLBACK xN: <probe texts>` or `[ERR]` rows.
3. `exact analytic` means no `brepkit_approx` probe fired during that op.
   `FALLBACK xN` lists which degradation path ran.
4. The final "remaining paths" section deliberately fires every approximation path;
   fallbacks there are expected, not regressions.
5. If a row regressed from exact to FALLBACK, the probe text names the path; map it to
   a probe site via reference.md section 1.

**The census caveat (critical):** "exact analytic" does NOT mean geometrically correct.
Precedents exist of results that were fully analytic and still wrong: a revolved cone
band touching the apex read double the angular span, inflating volume by half (fixed
in #1012), and a bored-sphere band over-integrated the removed polar cap while the
mesh skinned over the tunnel mouth, so it also rendered non-watertight (fixed in
#1005). Every analytic-preservation change must ship
with independent verification: volume vs exact or reference value, free edges == 0,
manifold check, and a regression fixture. Template fixture:
`crates/io/tests/gridfinity_wallcut_seq_inmem.rs` (asserts `free == 0`, `over == 0`,
and a minimum curved-face count). See the solid-verification skill.

## Doctrine (details and code locations in [reference.md](reference.md))

1. **Closed rims are `EdgeCurve::Circle`, never closed NURBS.** Downstream seam
   adoption and whole-loop handling pattern-match the Circle variant. (ref. section 3a)
2. **Add exact arms in `crates/math/src/analytic_intersection.rs`** instead of letting
   the generic marcher fragment a clean circle. Gate each arm to the tractable
   configuration and return `None` otherwise. (ref. section 2)
3. **Watertight seams require BOTH operands split at the SAME exact crossing
   vertices.** Resolve endpoints through the shared-crossing registry. (ref. section 3c)
4. **Co-endpoint seam arcs get a splitter-side midpoint split.** Never make
   `merge_duplicate_edges` smarter; control the geometry you emit. (ref. section 3d)
5. **Sequential booleans preserve analytic surfaces.** Split lateral bands, do not
   tessellate them. (ref. section 3e)

## Scoping filter: what to chase, what to skip

A boolean result face is a trimmed patch of an INPUT surface, so it is always closable
with the right split. Fillet, sweep, loft, and NURBS offset INTRODUCE a new blend
surface with no closed form.

- **Chase:** operations that RE-CREATE existing analytic surface types. Example:
  revolve maps analytic profile edges to exact Cone/Plane/Cylinder/Torus bands
  (`revolution_band_surface` in `crates/operations/src/revolve.rs`).
- **Do not chase:** fillet walls beyond the analytic fast paths, general sweep/loft
  side faces, offsets of NURBS input. Their output is inherently approximate.
- **Solve the NARROW problem.** Every merged exact arm is gated (coaxial-only, one
  perpendicular case) and defers to the marcher otherwise. An exact arm never has to
  be general to be shippable. Full table: reference.md section 4.

## Terminal and open cases (do not re-burn effort)

- **Perpendicular cylinder-union render is terminal.** The exact Steinmetz seam is
  singular (figure-eight pinch); exact curves make the topology WORSE. The shipped
  B-Rep with exact closed-form volume stands. `exact_cylinder_cylinder` does not exist
  in the codebase; do not go looking for it. (ref. section 5)
- **Plane-through-sphere across the seam is SOLVED** and is the reusable pattern:
  never test exact curves against discretized boundary chords; reconstruct the exact
  boundary curve and intersect analytically. (ref. section 5)
- `intersect_plane_torus` is still a grid march plus fit, not the exact quartic; the
  FF phase (GFA's surface-surface intersection stage,
  `crates/algo/src/pave_filler/phase_ff.rs`) compensates by snapping arc endpoints to
  exact crossings.

## Anti-patterns

- Do NOT conclude a change is correct because the census says "exact analytic".
- Do NOT add type-based or deviation-threshold keys to `merge_duplicate_edges`; both
  directions are load-bearing for different shapes (ref. section 3d).
- Do NOT write a general surface-surface exact intersector; gate to the narrow case.
- Do NOT chase analytic recovery for blend/offset-of-NURBS surfaces.
- Do NOT trust in-code comments about which cases "fall back to mesh"; some predate
  later fixes. Re-run the census and check face counts instead.

## Related skills

boolean-debugging (when the fallback fires and you need to know why GFA, the native
boolean engine in `crates/algo/src/gfa.rs`, failed),
solid-verification (the mandatory correctness pairing), tessellation (curved-band
meshing that analytic splits expose), numerical-robustness (seam crossings,
quantized merges, chord discretization), parity-benchmarking (head-to-head
harness), add-operation (wiring a new op so it stays analytic from day one).
