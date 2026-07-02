# Testing reference

Deep catalog for the testing skill. Everything here is verified against the current tree; when line numbers would rot, symbols and rg patterns are given instead.

## 1. Test taxonomy, with live examples

### 1a. Module unit tests

`#[cfg(test)] mod tests` inside the source file. Workspace lints deny `unwrap_used` and `panic`, so every test module opens with a module-level allow. Real headers in the tree:

- `crates/operations/src/boolean/tests.rs`: `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::print_stderr, clippy::cast_precision_loss, clippy::cast_possible_wrap)]`
- `crates/operations/src/tessellate/tests.rs`: `#![allow(clippy::unwrap_used, deprecated)]`
- Standalone test files use the same pattern, e.g. `crates/operations/tests/regress_hexwall_cuts.rs`.

Find them: `rg -n '#!\[allow\(clippy::unwrap_used' crates -g '*.rs'`

Large modules split tests into a sibling file (`boolean/tests.rs`, `tessellate/tests.rs`, `fillet/tests.rs`) wired in via `mod tests;` under `#[cfg(test)]`.

### 1b. proptest property tests

Invariants over generated inputs. Examples:

- `crates/operations/tests/proptest_operations.rs`: `prop_boolean_volume_conservation` asserts V(A) + V(B) = V(A union B) + V(A intersect B), inside `proptest! { #![proptest_config(ProptestConfig::with_cases(20))] ... }`. Keep case counts low for expensive CAD ops.
- `crates/operations/tests/coincident_proptest.rs`: coincident-contact boolean fuzzing.
- Heavy use inside `crates/math/src/` modules. Find all: `rg -l 'proptest!' crates --type rust`

Failing seeds auto-persist to `.proptest-regressions` files next to the test (e.g. `crates/operations/tests/proptest_operations.proptest-regressions`) and under `crates/math/proptest-regressions/`. Commit them; each seed is a permanent regression case replayed on every run. Never hand-edit or delete them to make a run green.

### 1c. Golden-file tests

- Data: `tests/golden/data/*.golden` at workspace root (e.g. `box_10x20x30.golden`, `boolean_box_minus_cylinder.golden`, `tessellation_sphere.golden`). Docs: `tests/golden/README.md`.
- Test code: `crates/operations/tests/golden_regression.rs`. Helper `golden_path()` joins `CARGO_MANIFEST_DIR/../../tests/golden/data`; `assert_golden()` regenerates when `UPDATE_GOLDEN=1` is set, otherwise diffs trimmed strings.
- `rg -n 'UPDATE_GOLDEN' crates -g '*.rs'` hits only that file.
- Naming: `{shape}_{operation}.golden`. Keep files small; golden tests are for output stability, not for exhaustive geometry checks.

### 1d. Integration tests (per-crate `tests/` dirs)

Root `tests/integration/` contains only a README describing the patterns. Runnable tests:

- `crates/operations/tests/`: the main suite. Boolean edge cases and invariants (`boolean_edge_cases.rs`, `boolean_invariants.rs`, `boolean_stress.rs`), coincident and coaxial families (`coincident_planes.rs`, `coaxial_cylinders.rs`, `coaxial_cones.rs`, `coaxial_torus.rs`, `concentric_spheres.rs`), regressions (`regress_hexwall_cuts.rs`), tessellation (`tessellate_watertight.rs`), and the parity corpus.
- Parity corpus: `parity_boolean_planar.rs`, `parity_boolean_curved.rs`, `parity_boolean_empty.rs` share `parity_support/mod.rs`, which defines `struct Case` and `run_corpus(cases: &[Case])`. Oracles are exact closed-form values, never face counts: `enum Oracle` in `parity_support/mod.rs` has `Volume(f64)` (the most common), `Area(f64)`, and `Empty` (the result must be empty). Add new boolean scenarios as `Case` entries here when they have a closed-form expected value.
- `crates/io/tests/`: fixture-based regressions (section 2).
- Others: `crates/math/tests/ssi_robustness.rs`, `crates/offset/tests/integration.rs`, `crates/render/tests/` (offscreen render and compute-mesh tests).

Doctrine from `tests/integration/README.md`: measure, do not inspect topology. Verify with `volume()` / `bounding_box()`; tolerances around 1e-6 for direct computation, 1e-3 for I/O round-trips. Test the workflow, not the internals.

### 1e. Criterion benches

All in `crates/operations/benches/`: `cad_operations.rs`, `boolean_perf.rs`, `boolean_tracking.rs`, `compound_cut_perf.rs`, `fuse_perf.rs`. Bench targets are declared in `crates/operations/Cargo.toml`. Run one:

```bash
cargo bench -p brepkit-operations --bench boolean_perf
```

For profiling a bench, see CLAUDE.md, Profiling, and the profiling skill. For cross-kernel comparison against the reference kernel, use the brepjs harness (see the parity-benchmarking skill).

### 1f. WASM contract tests via `execute_batch`

`crates/wasm/src/bindings/gridfinity_tests.rs` is the template (`mod gridfinity_tests` under `#[cfg(test)]` in `bindings/mod.rs`). Pattern:

```rust
let mut k = BrepKernel::new();
let result = k.execute_batch(
    r#"[{"op": "makeBox", "args": {"width": 10.0, "height": 5.0, "depth": 10.0}},
        {"op": "volume", "args": {"solid": 0}}]"#,
);
```

then parse the JSON with the file's local helpers (`parse_batch`, `assert_ok`, `assert_no_crash`, `ok_f64`, `ok_bbox`). `execute_batch` is defined in `crates/wasm/src/bindings/batch.rs`.

Why this shape: `JsError` cannot be constructed on non-wasm targets, so binding methods cannot be called directly in native tests. `execute_batch` takes and returns strings, so `cargo test -p brepkit-wasm` runs these on the host. See CLAUDE.md, Recipe 4.

Handle gotcha, documented in the file header: solid handles in a batch are arena indices, not batch result indices. The header lists which ops create new solids (`makeBox`, `fuse`, `cut`, `extrude`, ...) versus which return the same handle or a float (`transform`, `volume`, `boundingBox`, ...). Count created solids to compute the handle for a later op.

### 1g. Ignored ready-repro tests

An `#[ignore = "..."]` test that reproduces a known-open bug, kept compiling, with the ignore string naming the open condition. The next session starts from a runnable repro instead of re-deriving one. Live examples:

- `crates/io/tests/dovetail_cornerclip_intersect_inmem.rs`, test `dovetail_corner_clip_intersect_is_watertight`. Ignore string starts "open: coincident-wall + analytic-corner intersect drops the boundary". Its assertions encode the acceptance target: watertight, analytic, compact face count.
- `crates/operations/src/boolean/tests.rs`, test `fuse_shelled_box_with_socket_loft`. Ignore string: "known bug: non-manifold edge at shelled-box + socket fuse boundary".

Two neighboring variants that also use `#[ignore]` but are NOT ready-repros:

- Diagnostic tools: explicit-only harnesses that print internal state and are expected to fail or never pass, e.g. `crates/operations/tests/perf_64cut_determinism.rs` (nondeterminism bisect tool) and `crates/operations/tests/profile_intersect.rs` (topology-inspection probes). Their ignore strings start with "diagnostic".
- Slow tests: e.g. `staircase_fuse_with_cylinders` in `boolean/tests.rs`, ignore string starting "slow (~2 min)".

Convention for the ignore string prefix: `open:` for a bug awaiting a fix, `known bug:` acceptable too, `diagnostic` for tools, `slow` for runtime. Always include enough context that a reader with no history understands what is broken.

Find all: `rg -n '#\[ignore' crates -g '*.rs'`

Lifecycle: bug found, ready-repro written and ignored, fix ships, `#[ignore]` removed in the fix PR, test becomes a permanent regression guard.

## 2. Fixture tiers in detail

### Tier 1: native primitive repro

Rebuild the failing scenario from primitives and operations inside a crate test. `crates/operations/tests/regress_hexwall_cuts.rs` is the template: hollow thin-walled box, sequence of hex-prism cuts, each cut asserted to remove exactly the analytic prism volume. The historical failure it guards: later cuts silently no-opped while returning Ok, which is why per-step volume deltas are asserted, not just the final value.

### Tier 2: STEP fixture from real tool geometry

When the failing geometry cannot be reconstructed exactly from a native rebuild (e.g. a tapered multi-section loft), capture the tool's literal operands as STEP and commit them. Template: `crates/io/tests/lipfuse_fixture.rs` with `crates/io/tests/data/lipfuse_*.step` (also `wallcut_*.step`, `scoop_*.step`, `multicavity_*.step`). Load with `brepkit_io::step::reader::read_step`.

The lipfuse header states the doctrine: the fixtures are the tool's literal operands, exported via STEP, round-tripping as exact cylinders, cones, and planes. Faithfulness gate: after import, confirm the surfaces are analytic types, not NURBS approximations. If STEP degraded them, the fixture no longer exercises the analytic code path where the bug lives.

These tests live in `crates/io/tests/` because they need the io readers, and io (L3) sits above operations. Do not add io as a dependency of operations tests.

### Tier 3: arena `.bin` fixture

Some bugs exist only in a specific in-memory id/vertex layout. A STEP round-trip renumbers entities and re-derives vertices, which can erase the bug. For those, serialize the arena directly:

- Capture tool-side: the `serializeSolid` wasm binding in `crates/wasm/src/bindings/io.rs` returns bytes.
- Commit as `crates/io/tests/data/<name>.bin`.
- Load in the test with `brepkit_io::arena_io::deserialize_solid` (module `crates/io/src/arena_io.rs`).
- Name the test file `*_inmem.rs`. Existing examples: `dovetail_cornerclip_intersect_inmem.rs`, `gridfinity_wallcut_seq_inmem.rs`, `scoop_fix_inmem.rs`, and others in `crates/io/tests/`.

Decision rule: try tier 1. If the repro passes while the real tool fails, capture tier 2. If the STEP round-trip also passes while the in-memory case fails, capture tier 3. Never debug on a repro from a lower tier that does not reproduce.

## 3. What to assert

| Property | How | Do NOT use |
|---|---|---|
| Cut/fuse actually happened | `brepkit_check::classify::classify_point` probes at points that must be inside/outside | volume alone (tessellated volume can read high over an un-carved cut) |
| Correct volume | `brepkit_operations::measure` volume vs a closed-form analytic value, tolerance ~1e-6 | comparing against a previously recorded float with no derivation |
| Watertight / manifold | count edge uses via quantized endpoint pairs, every edge used exactly twice (see the `edge_use` / `edge_health` helpers in the io fixture tests) | "it tessellates without error" |
| Result stayed analytic | face count: analytic results have a handful of faces, mesh fallback produces hundreds to thousands of all-planar facets | triangle count or validity, both mask the fallback |
| In/out classification on non-analytic solids | ray-cast classifier `classify_point` | winding-number classifier (`classify_point_winding`), wrong for non-analytic solids |

Boolean gate context: `boolean()` in `crates/operations/src/boolean/mod.rs` validates the analytic GFA result and falls back to `mesh_boolean_fallback` (same file, `rg -n 'fn mesh_boolean_fallback'`) when validation fails. A silent fallback is itself a regression for analytic scenarios; assert face counts to catch it.

## 4. Commands with expected output shapes

```bash
cargo test --workspace
# per test target: "test result: ok. N passed; 0 failed; K ignored; ..."

cargo test -p brepkit-operations --test parity_boolean_curved
# runs one integration-test binary

cargo test -p brepkit-io --test dovetail_cornerclip_intersect_inmem -- --ignored
# runs the ready-repro; EXPECTED TO FAIL while the bug is open

cargo test --release -p brepkit-operations --test perf_64cut_determinism -- --ignored --nocapture
# diagnostic tool; its own file header documents this exact invocation

UPDATE_GOLDEN=1 cargo test --workspace golden
# rewrites tests/golden/data/*.golden; inspect git diff before committing

cargo bench -p brepkit-operations --bench cad_operations
# criterion output with per-benchmark timing and change-vs-baseline

cargo run --release --example approx_census -p brepkit-operations
# enumerates ops that degrade analytic to approximation/mesh; rerun after a fix to confirm a flip

cargo clippy --all-targets -- -D warnings
# test code compiles under the strict lints; missing allow headers fail here

./scripts/check-boundaries.sh
# layer-dependency gate; CI runs it in the boundaries job
```

Failure shapes worth recognizing:

- `Golden file mismatch: <name>` plus a diff: output changed; decide intentional vs regression before regenerating.
- `Golden file not found: ... Run with UPDATE_GOLDEN=1 to create it.`: new golden test, generate the file.
- A proptest failure prints the minimal failing input and writes a seed to the `.proptest-regressions` file; commit that file.
- `test result: ok` but a scenario silently fell back to mesh: no test failure at all unless a face-count assertion exists. This is why fixtures assert face counts.

Hooks (see `.husky/pre-commit` and `.husky/pre-push` for the authoritative contents): pre-commit runs fast checks only, fmt + clippy + taplo + cargo-machete in parallel, and runs no tests. Pre-push runs nothing locally; all validation (nextest, deny, boundaries, and more) is delegated to CI in `.github/workflows/ci.yml`, gated by the `ci-pass` job. Nothing on the local path runs the test suite for you: run tests manually before pushing. Never bypass the hooks.

## 5. Glossary

- **GFA**: brepkit's general boolean engine (`crates/algo`), a pave-filler plus builder pipeline that intersects, splits, classifies, and reassembles faces.
- **PaveFiller**: GFA phase 1, computes interferences between entity pairs and splits edges at paves (intersection points). Phases VV/VE/EE/VF/EF/FF; FF (face-face) creates intersection sections and hosts most curved-boolean bugs.
- **Section**: an intersection curve segment between two faces, produced in FF, later stitched into face-splitting wires.
- **Same-domain (SD)**: detection that two operand faces lie on the same surface (coincident contact), so they merge instead of splitting each other.
- **Mesh fallback**: when the analytic GFA result fails validation, `boolean()` falls back to a triangle-mesh co-refinement boolean. Correct but slow and non-analytic; the telltale is exploding face count.
- **Analytic vs NURBS**: analytic means exact Plane/Cylinder/Cone/Sphere/Torus surfaces and Line/Circle/Ellipse curves. The quality bar is preserving these through operations instead of degrading to fitted NURBS or mesh. See the analytic-preservation skill.
- **Watertight / manifold**: every edge shared by exactly two faces. Free edges (used once) or over-shared edges (three or more) mean a broken shell.
- **Ray-cast classifier**: `brepkit_check::classify::classify_point`, the trustworthy in/out oracle. The winding classifier is unreliable on non-analytic solids.
- **Approx census**: `crates/operations/examples/approx_census.rs`, enumerates operations that still degrade analytic input.
- **Parity corpus**: the `parity_boolean_*.rs` tests scoring booleans against exact analytic volume oracles, built to match and beat the reference kernel.
- **Reference kernel**: the mature C++ CAD kernel brepkit benchmarks against via the brepjs harness (see the parity-benchmarking skill).
- **Ready-repro**: an `#[ignore]`d, compiling test that reproduces a known-open bug and whose assertions encode the fix's acceptance target.
