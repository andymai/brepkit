---
name: boolean-debugging
description: Diagnose and fix failures of the GFA boolean engine (fuse, cut, intersect). Use when a boolean result mesh-falls-back to hundreds of all-planar faces, comes out non-manifold or open-shelled, is missing faces, has wrong volume, fails the Euler or validation gates, errors in shell assembly, or gives different results across runs.
---

# Boolean Debugging (GFA engine)

Glossary of GFA terms (PaveFiller, FF sections, SD, paves) and the bug-class catalog live in [reference.md](reference.md).

## When to use

A boolean via `brepkit_operations::boolean::boolean` (or wasm `fuse`/`cut`/`intersect`) produces: silent mesh fallback, non-manifold or open shell, a vanished face, wrong volume, `AssemblyFailed`, or cross-process nondeterminism.

## Quick reference

| Symptom | First move | Detail |
|---|---|---|
| Result has hundreds of all-planar faces | It mesh-fell-back. Find WHY the gate rejected the GFA result | Step 1, Step 2 |
| Result looks fine but volume is off | Do NOT trust volume. Ray-cast classify points in the cut region | Prior 3 |
| Coincident-contact fuse/cut wrong | Suspect the classifier, not same-domain detection | Prior 1 |
| `AssemblyFailed("no outer shell found...")` | Shell sign or a never-created face; dump per-face data upstream | reference.md catalog |
| Non-manifold + inflated volume | Wire spur; check for consecutive same-edge opposite-orientation pairs | reference.md catalog |
| Differs across processes, stable within one | HashMap iteration order; sort the driving iteration | Step 6 |
| Bug only reproduces in the consuming tool | Unfaithful repro; export real operands as STEP or arena bytes | Step 5 |

## Step 1: Detect mesh fallback. Face count is the ONLY reliable signal

A clean analytic result is roughly 3 to 80 faces with curved surface types present. A mesh fallback is hundreds to thousands of faces, all `Plane`. Do not threshold high: a 190-face fallback has slipped past a ">500" heuristic. Zero curved faces on operands that had cylinders/cones/spheres is the tell.

Triangle count and validity checks BOTH mask fallback: the fallback result is watertight and valid (it only passes `validate_boolean_result_lenient`). Volume masks it too. Only the face census tells the truth.

Rust probe:
```rust
let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, solid)?;
for fid in brepkit_topology::explorer::solid_faces(&topo, solid)? {
    let tag = topo.face(fid)?.surface().type_tag();
}
```
`{plane: 439}` and nothing else = fallback. `{plane: 41, cylinder: 24, cone: 12}` = analytic.

Tool probe (wasm kernel, `crates/wasm/src/bindings/query.rs`): `getEntityCounts(solid)` returns `[faces, edges, vertices]`; `getSolidFaces(solid)` + `getSurfaceType(face)` gives the census (`"plane" | "cylinder" | ... | "bspline"`; NURBS that exactly represent analytics are recognized). Re-probe face count on your known cases after EVERY GFA change; a fix for one case can silently regress another.

## Step 2: Understand the gate in `crates/operations/src/boolean/mod.rs`

Flow (verify current condition names with `rg -n 'euler_ok|open_shell_ok|hollow_ok' crates/operations/src/boolean/mod.rs`):

1. Pre-GFA shortcuts: identical-solid, containment (`EmptyResult`, `build_contained_cut_hollow`), planar-NURBS flattening (`flatten_planar_nurbs_faces`).
2. GFA runs: `brepkit_algo::gfa::boolean`. Two `BooleanOp` enums exist (engine: `crates/algo/src/bop.rs`; public: `crates/operations/src/boolean/types.rs`); operations converts.
3. On `Ok`, heals run BEFORE gating: `remove_degenerate_edges`, `heal::remove_wire_spurs`, `unify_coincident_boundary_edges` (only if free edges), `merge_result_vertices`, up to 3x `heal::unify_faces`.
4. Gates: `open_shell_ok` (Intersect rejects free edges; Cut/Fuse tolerate them), `hollow_ok` (inner shells require closed manifold), `euler_ok` (hole-aware `euler_balanced`, cavity-shell surplus subtracted). Accept = `euler_ok && open_shell_ok && validate_boolean_result(..).is_ok()`.
5. Reject or GFA `Err`: `log::warn!` "falling back", then for Cut with a multi-component input A a `cut_multi_region_input` retry (per-component cuts, recombined), then a `log::debug!(target: "brepkit_approx", ...)` probe and `mesh_boolean_fallback`.

So when you see fallback, run with `RUST_LOG=debug` to learn which path fired, then find which gate condition failed and why the GFA output violated it.

## Step 3: The debugging loop

1. Reproduce with a MINIMAL primitive case (extend `crates/operations/examples/debug_boolean.rs`, see reference.md for what it prints and how to extend it). Run: `cargo run --release --example debug_boolean -p brepkit-operations`.
2. Bypass the operations layer: call `brepkit_algo::gfa::boolean` directly. Raw GFA output and post-heal output diverge; isolate at the raw layer first.
3. Dump LITERAL data, never reason from theory: face count per surface type, per-face reversed/inner_wires/area, Euler V-E+F, edge-usage counts (1 = free, 3+ = non-manifold; `has_free_edges`/`is_closed_manifold` are private to boolean/mod.rs, reimplement the short usage-count loop), wire-edge traversal per face, per-face signed volume.
4. Vary ONE variable per run (one dimension, one offset, one primitive swap).
5. Fix at the layer that owns the artifact (a bad section curve is a phase_ff bug, not an assembler bug). Trust no "this alone closes it" claim without a raw-GFA face trace; multi-bug chains are the norm.

The approximation census: `cargo run --release --example approx_census -p brepkit-operations` runs a boolean matrix over primitives and reports exact vs fallback per op. Caution: census "exact" only proves no fallback probe fired, NOT that the geometry is correct.

## Step 4: Priors that save days

1. Coincident-contact bugs: suspect the CLASSIFIER, not same-domain detection. Instrument `detect_same_domain` (`crates/algo/src/builder/same_domain.rs`) FIRST to rule SD out. Twice an SD diagnosis was wrong and the pairs were present both times; the real roots were classification (interior sample on a coincident rim). Two interior-sample paths exist and must BOTH be checked: `sample_face_interior` (`crates/algo/src/builder/mod.rs`, unsplit faces) and `interior_point_3d` (`crates/algo/src/builder/face_splitter/mod.rs`, split faces).
2. Verify the boolean INPUTS for malformed faces before blaming the engine. An "engine bug" has turned out to be a wrong-face input shell. Run the face census and validation on both operands first (see the solid-verification skill).
3. Verify cut geometry with the ray-cast classifier, NEVER with volume. Tessellation volume can read HIGH and nearly mask an un-carved slot. Use `brepkit_check::classify::classify_point(topo, solid, point, &ClassifyOptions::default())` on points that must be Inside/Outside after the op. The winding-number classifier (`classify_point_winding`/`classify_point_robust`) is categorically WRONG for non-analytic or faceted solids; do not use it here.

## Step 5: Faithful repro discipline

Export the ACTUAL failing operands from the consuming tool (brepjs: `k.unwrap(k.exportSTEP(operand))`), read back with `brepkit_io::step::reader::read_step(input: &str, topo) -> Result<Vec<SolidId>, IoError>`, then confirm the round-trip is ANALYTIC (faces cylinder/cone/plane, edges circle, not bspline). If STEP normalizes the bug away (sub-ULP vertex noise), use byte-exact arena serialization: wasm `serializeSolid`/`deserializeSolid` and native `brepkit_io::arena_io::{serialize_solid, deserialize_solid}`. Capture per-step OPERANDS, not batch results. Full recipe and precedent fixtures: reference.md, "Faithful repro".

## Step 6: Nondeterminism signature

Identical within a process but varying across processes = std `HashMap` random seed driving iteration order into downstream branching. Fix by sorting the iteration at the point that feeds branching (`sort_unstable_by_key`; precedents in `crates/algo/src/builder/same_domain.rs`). Do not reach for tolerance theories when the signature is process-boundary-shaped.

## Anti-patterns: what NOT to conclude

- "Triangle counts match and the mesh is watertight, so the boolean worked." Fallback results are watertight. Check the face census.
- "Volume is within 2%, so the cut happened." Volume can read high on an un-carved cut. Ray-cast classify.
- "Same-domain detection missed the coincident pair." Instrument `detect_same_domain` before believing this; it has been present both times it was blamed.
- "The winding classifier says Outside, so the point is outside." Not on faceted or stepped solids.
- "This one fix closes the case." Assume a chain of bugs until a raw-GFA trace on the real operands is clean.
- "My synthetic nudge repro behaves like the tool's case." Validate any native proxy against the real tool operands first.
- "The census says exact, so the result is correct." It only says no fallback fired.

Related skills: solid-verification (validity and volume checks), analytic-preservation (keeping curved surface types through ops), debugging-doctrine (the general vary-one-variable method), tessellation (mesh-side issues).
