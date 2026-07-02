# Debugging Doctrine: Reference

Companion to [SKILL.md](SKILL.md). All symbols verified against the repo; locate them with the `rg` patterns given (line numbers rot, symbols survive).

## Faithful fixtures

Capture mechanics, the three fixture tiers (native primitive repro, STEP, arena `.bin`), and templates to copy: see the testing skill. The doctrine-level rules:

- STEP tier: replay via `brepkit_io::step::reader::read_step`; full template `crates/io/tests/lipfuse_fixture.rs`. **Faithfulness gate before debugging:** iterate the imported faces and match on `FaceSurface`. Cylinder/Cone/Plane coming back as `Nurbs` means the export went lossy, the numbers differ, and any fix will overfit the fixture. This gate is the exact difference between a tractable case (lip fuse: analytic round trip) and a thrash (scoop: lossy NURBS reconstruction).
- Arena tier: failures living in the in-memory id layout (edge/vertex ordering, arena numbering) vanish under STEP renumbering. Snapshot losslessly with `serialize_solid` / `deserialize_solid` in `crates/io/src/arena_io.rs` (JS: `serializeSolid` / `deserializeSolid`); the `*_inmem.rs` tests in `crates/io/tests/` are the pattern. This tooling exists because a five-pass thrash proved "no stable repro" was the real blocker. When you hit the same wall, capture, do not attempt another debugging pass on a proxy.

### Faithfulness checklist

Before spending a single pass, the repro must match the real case on ALL of:

- [ ] Same operands (captured, not rebuilt from remembered parameters; a wrongly remembered corner radius cost ten passes)
- [ ] Same surface types per face (analytic stays analytic)
- [ ] Same face/edge/vertex counts on the inputs
- [ ] Same failure signature (same error, same wrong volume, same open-edge count)

## Instrumentation catalog

### The canonical dump harness

`crates/operations/examples/debug_boolean.rs`. Edit it in place for the investigation at hand (swap in your fixture load, your op). Run:

```bash
cargo run --release --example debug_boolean -p brepkit-operations
```

What it dumps and the output shape:

```
Box volume: 15000
Result has 7 faces
  Face 0: Plane(n=0.00,0.00,-1.00 d=-0.00)  reversed=false  inner_wires=1  area=1421.46
  Face 6: Cylinder  reversed=true  inner_wires=0  area=314.16
Per-face signed volume (divergence theorem):
  Face 0: signed_vol = 0.00  (38 tris)
Total signed volume: 14214.60
measure volume:  14214.60
Expected volume: 14214.60
```

Per-face signed volume is the sharpest tool here: a face contributing the wrong sign or a wildly wrong magnitude localizes the bad face immediately, where a total volume only says "something is off".

Caveat: the example walks `outer_shell()` only. For hollow solids use `brepkit_topology::explorer::solid_faces` (see CLAUDE.md, "Walking faces in a solid").

### Edge-incidence dump (free / over-shared edges)

Pattern: `edge_use` in `crates/io/tests/lipfuse_fixture.rs` (`rg -n 'fn edge_use' crates/io/tests/lipfuse_fixture.rs`). Counts wire-edge uses keyed by a quantized, orientation-independent endpoint pair. `free` = used by exactly 1 face (open shell), `over` = used by 3+ (non-manifold). Watertight-manifold means both are 0.

Subtlety recorded in its doc comment: the quantization grid matters. Too coarse (1e-5) false-merges legitimate sub-micron corner arcs into phantom over-shares; 1e-6 is the working value. If the dump reports over-shares on tiny arcs, suspect your grid before the topology.

### Entity counts and Euler

- Rust: `brepkit_topology::explorer::solid_entity_counts` returns `(faces, edges, vertices)`.
- JS: `getEntityCounts` returns `[faces, edges, vertices]`, `getSurfaceType` per face (`crates/wasm/src/bindings/query.rs`).

**Face count is the only reliable mesh-fallback tell.** A clean analytic boolean result has roughly 3 to 80 faces with the expected curved types; mesh fallback produces hundreds of faces, all planar, zero curved. Triangle count and mesh validity both MASK the fallback (this exact trap silently invalidated a parity scorecard). Re-probe face counts after every GFA change.

### Wire-traversal dump and spurs

Dump each face's wire as the literal ordered edge list with orientations. The spur signature: a wire that walks the same edge forward then immediately reversed (consecutive out-and-back pair). Zero area contribution, but the edge is counted twice, producing a phantom 3-face edge and inflated volume. Sound repair exists: `remove_wire_spurs` (`rg -n 'pub fn remove_wire_spurs' crates/operations/src/heal.rs`), but dump first, repair second: a spur is a symptom of an upstream splitter bug.

### Raw-engine trace vs the gated pipeline

The full boolean pipeline is:

```
operations::boolean::boolean
  -> brepkit_algo::gfa::boolean          (the real GFA engine)
  -> gate: euler_ok && open_shell_ok && validate_boolean_result(...)
  -> on gate failure, Cut only: multi-region acceptance (N disjoint closed
     manifolds, Euler = 2N, plus a B-interior component check)
  -> then, Cut only: per-component retry (cut_multi_region_input) when the
     INPUT solid is already multi-region
  -> last resort: mesh_boolean_fallback (logs a warning, returns Ok;
     silent to callers)
```

`rg -n 'pub fn boolean' crates/algo/src/gfa.rs` and `rg -n 'fn mesh_boolean_fallback|euler_ok && open_shell_ok' crates/operations/src/boolean/mod.rs`.

Consequences:

- To see what the engine actually produced, call `gfa::boolean` directly and dump faces, surface types, Euler. The gated result can look plausible (it is a valid mesh) while the GFA output was rejected and discarded.
- A predicted fix must be validated with a raw trace on the real operands. One diagnosis agent predicted a single AABB fix would close a sphere-minus-cylinder case; the raw trace showed the post-fix result had 3 faces, all cylinder, sphere entirely dropped at split/keep. The trace re-scoped one fix into five.

For phase-by-phase PaveFiller and section dumps, see the **boolean-debugging** skill.

### Temporary flag-gating (does the path fire?)

Technique, not infrastructure: add a temporary early-return, a counter, or an `if false` around the suspect branch; rebuild; rerun the fixture. There are deliberately no permanent env-var toggles in the engine, so add and remove the gate within the investigation.

- If the failing case is byte-identical with the branch disabled, the branch never fires for this case. The diagnosis is wrong regardless of how plausible it reads. (Real case: an arc-split branch in `crates/algo/src/builder/builder_solid.rs` was blamed; a counter showed zero executions; the actual fix was `weld_coincident_vertices` in the same file.)
- Compare INTERMEDIATE state across the toggle (shell-closed, per-face signed volume, edge-incidence), not just the final error string. A final-error-only comparison once missed the real isolation signal entirely.

## Ground truth: point classification

- **Use:** `brepkit_check::classify::classify_point(topo, solid, point, &ClassifyOptions::default())` in `crates/check/src/classify/mod.rs`. Ray casting along three irrational directions with majority vote. This is the ground truth for in/out/on.
- **Never:** `classify_point_winding` / `classify_point_robust` on faceted, stepped, or NURBS-heavy solids. They return OUT for points clearly inside such solids; that behavior sent three debugging agents down a wrong "multi-surface envelope" theory. The GFA builder itself uses ray casting (`crates/algo/src/classifier/ray_cast.rs`) for the same reason.
- **Never volume as ground truth.** `measure::solid_volume` is tessellation-based with deflection clamped to `bbox_diag * 5e-5` (`crates/operations/src/measure/volume.rs`); it once read 1.4 percent HIGH and nearly masked a fully un-carved slot. It also has four short-circuit paths, tried in order inside `solid_volume`: closed-form analytic (`try_analytic_solid_volume`), per-face analytic Gauss quadrature (`analytic_faces_solid_volume`), analytic surface-of-revolution (`analytic_revolution_solid_volume`), then tessellation. "Volume looks right" can mean the wrong path answered.

## Case studies (layered bugs and misdirection)

### Coaxial cone-cone cut: five successive correct diagnoses

Each fix exposed the next layer. In order: same-domain handling; face keep/discard selection; classification (the interior sample of `interior_point_3d` landed exactly on a coincident rim, returned nothing, and the fallback classified Outside); the cone-cone intersection marcher emitting ~97 garbage micro-curves instead of one circle (fixed via `exact_cone_cone` in `crates/math/src/analytic_intersection.rs`); and finally emitting that circle as `EdgeCurve::Circle` rather than NURBS so the FF phase's seam-adoption logic recognizes it. Lesson: re-dump after every fix; the layer-N dump says nothing about layer N+1.

### Sphere minus cylinder: five bugs in sequence

Missing exact (Sphere, Cylinder) intersection arm (`exact_sphere_cylinder`, same file); `split_noseam_face_direct` skipping full circles (`crates/algo/src/builder/face_splitter/special_cases.rs`); both hemisphere interior sample points degenerating to the north pole; the outer-wire-only same-domain hash false-merging two distinct bands (`detect_same_domain`, `crates/algo/src/builder/same_domain.rs`); volume over-integration in tessellation. The initial diagnosis predicted bug one alone would close the case.

### The nine-pass symptom chase (fuse)

"Shell open", "inside-out", "Euler unbalanced" were all DOWNSTREAM symptoms. Nine passes instrumented the shell assembler; the bugs were two layers up: two sources of degenerate sections in `build_section_edges` (`rg -n 'fn build_section_edges' crates/algo/src/builder/fill_images_faces.rs`), zero-length or zero-span sections and sections coinciding with a face's own inner-wire boundary. Durable rule: when a fuse shell is non-manifold and inside-out, dump each contact face's INPUT sections before touching the assembler.

### Classifier, not same-domain (twice)

Two separate coincident-contact failures were confidently diagnosed as "same-domain detection missed the overlap". Instrumenting `detect_same_domain` showed the pairs WERE detected both times. The real bugs were in classification: an interior sample landing on a coincident rim, whose failed evaluation fell back to a ray cast that answered Outside. Two interior-sample paths exist and must both be checked: `sample_face_interior` (`crates/algo/src/builder/mod.rs`, unsplit faces) and `interior_point_3d` (`crates/algo/src/builder/face_splitter/mod.rs`, split sub-faces). A rim-avoidance defense in one can be absent in the other. Instrument SD first only to RULE IT OUT, then move to the classifier.

### Narrow beats general

A general "arc-identity merge key" for duplicate-edge merging is provably impossible here: the stacking-lip ring needs co-endpoint seam arcs to COLLAPSE to one edge, while a torus-minus-box lens needs the same-signature pair kept DISTINCT. No local key discriminates; the difference is global. The shipped solution was per-case and upstream: a splitter-side midpoint split, controlling the geometry emitted instead of patching the shared merge. When a "general primitive" keeps breaking a second caller as you fix the first, that is the same signal: solve the narrow problem.

## Glossary

- **GFA**: brepkit's general fuse algorithm, the boolean engine in `crates/algo`. Raw entry `gfa::boolean`; `operations::boolean::boolean` wraps it with the acceptance gate and mesh fallback.
- **PaveFiller**: GFA phase computing pairwise interferences (VV/VE/EE/VF/EF/FF) and splitting edges at intersection parameters. `crates/algo/src/pave_filler/`.
- **FF phase**: face-face intersection (`pave_filler/phase_ff.rs`), producing section curves. Historically the richest bug habitat.
- **Section**: an FF intersection curve threaded into a face's splitting arrangement (`build_section_edges`). Degenerate sections poison everything downstream.
- **Same-domain (SD)**: detection of coincident overlapping faces across operands (`builder/same_domain.rs`) so they merge to one representative.
- **Mesh fallback**: triangle-mesh re-run when the GFA result fails the gate. Logs a warning but returns Ok, so it is silent to callers. Correct-ish geometry, hundreds of all-planar faces. Face count is the tell.
- **Free / over-shared edge**: edge used by exactly 1 face (open) or 3+ faces (non-manifold). Both zero means watertight-manifold.
- **Spur**: consecutive out-and-back same-edge pair in a wire. Zero area, breaks manifoldness.
- **Faithful fixture**: the real failing operands captured losslessly (STEP if analytic round trip holds, arena snapshot otherwise) and replayed in a Rust test.
- **The reference kernel**: the incumbent C++ B-Rep kernel brepkit replaces. Its source may be consulted for algorithm structure (see the fuse note: it keeps sub-blocks whose midpoint classifies in-or-on both faces, rather than clipping to the opposing polygon). Never name it in commits, PRs, code, or docs. For head-to-head benchmarks, use the brepjs harness (see the **parity-benchmarking** skill).
