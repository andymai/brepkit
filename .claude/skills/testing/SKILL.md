---
name: testing
description: Use when writing, placing, running, or updating tests in brepkit; when a bug fix needs a regression fixture; when deciding whether a repro is faithful to the real failing geometry; when a golden file mismatches; or when ending a session with unverified work (verify-or-revert). Covers unit, proptest, golden, integration, wasm contract, bench, and ignored ready-repro tests.
---

# Testing in brepkit

## When to use

- Adding tests for a new operation or a bug fix, and choosing where they go.
- A test fails with "Golden file mismatch" or "Golden file not found".
- You fixed a boolean bug and need the fixture that proves it.
- You built a repro and are about to spend hours debugging on it.
- End of session and the fix is not verified.

Deep catalog, fixture-capture procedures, and glossary: see [reference.md](reference.md).

## Quick reference

```bash
cargo test --workspace                                        # everything
cargo test -p brepkit-operations                              # one crate
cargo test -p brepkit-operations --test regress_hexwall_cuts  # one integration file
cargo test -p brepkit-operations some_test_name_substring     # filter by name
cargo test -p brepkit-wasm                                    # wasm contract tests (run native)
cargo test -p brepkit-io --test dovetail_cornerclip_intersect_inmem -- --ignored   # a ready-repro
UPDATE_GOLDEN=1 cargo test --workspace golden                 # regenerate golden files
cargo bench -p brepkit-operations --bench cad_operations      # criterion bench
rg -n '#\[ignore' crates -g '*.rs'                            # list all ignored repros/diagnostics
```

Expected shape: `cargo test` prints per-target `test result: ok. N passed; 0 failed; K ignored`. Ignored counts are normal, they are known-open repros and diagnostics, not rot.

## Where a test goes (decision table)

| You are testing | Put it | Example to copy |
|---|---|---|
| One module's logic | `#[cfg(test)] mod tests` in that file, with the clippy allow header | `crates/operations/src/boolean/tests.rs` |
| A workflow across modules in one crate | `crates/<crate>/tests/<name>.rs` | `crates/operations/tests/regress_hexwall_cuts.rs` |
| An invariant over an input range | proptest, in either location above | `crates/operations/tests/proptest_operations.rs` |
| Exact output bytes/text stability | golden test + file in `tests/golden/data/` | `crates/operations/tests/golden_regression.rs` |
| A JS-facing binding | wasm contract test via `execute_batch` | `crates/wasm/src/bindings/gridfinity_tests.rs` |
| A bug fixed via STEP or `.bin` fixture | `crates/io/tests/` (needs the io readers, io is L3) | `crates/io/tests/lipfuse_fixture.rs` |
| Performance | criterion bench in `crates/operations/benches/` | `cad_operations.rs` |
| A bug you could NOT fix this session | `#[ignore = "open: ..."]` ready-repro, kept compiling | `crates/io/tests/dovetail_cornerclip_intersect_inmem.rs` |

Every test module or file starts with the allow header, because workspace lints deny `unwrap_used` and `panic`:

```rust
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
```

Note: root `tests/integration/` holds only a README of patterns. Runnable integration tests live in each crate's `tests/` dir. Root `tests/golden/` holds only the `.golden` data files; the test code is `crates/operations/tests/golden_regression.rs`.

## Regression-fixture doctrine

Every bug fix ships a fixture that fails before the fix and passes after. Run it against the pre-fix tree at least once to confirm it actually fails. Three tiers, escalate only when the lower tier cannot reproduce:

1. **Minimal native primitive repro** (preferred). Rebuild the failing scenario from primitives in a crate test. Copy the shape of `regress_hexwall_cuts.rs`.
2. **STEP fixture captured from the real failing tool geometry**, when the geometry does not reconstruct exactly from a native rebuild. Files in `crates/io/tests/data/*.step`, loaded via `brepkit_io::step::reader::read_step`. Faithfulness check: the operands must round-trip as analytic surfaces (Cylinder/Cone/Plane, not NURBS). A NURBS round-trip means the fixture is unfaithful, do not use it.
3. **Arena `.bin` fixture**, when STEP round-trip renumbers ids or re-derives vertices and the bug disappears. Captured tool-side via the `serializeSolid` wasm binding, loaded via `brepkit_io::arena_io::deserialize_solid`. Tests named `*_inmem.rs` in `crates/io/tests/` are this tier.

Assert on measurements (volume against an exact analytic value, edge-use counts for watertightness, face counts for analytic-vs-mesh-fallback), not on internal topology layout. See `tests/integration/README.md` and reference.md section "What to assert".

## Faithful-repro-first

Before grinding on a repro, verify it matches the REAL failing geometry: same dimensions, same layout, same failure signature (same face-count explosion, same free-edge pattern). A proxy repro that diverges from the real case burns many debugging passes and its fixes do not transfer. Known failure mode: a proxy with a rounded corner radius that straddles a numeric boundary the real geometry never crosses produced an entire false root-cause theory.

Checkpoint: run the repro, confirm it fails the same way the real case does, THEN debug. If a STEP round-trip cleans the failure away, escalate to a tier-3 `.bin` fixture instead of debugging the cleaned version.

## Verify-or-revert

Work that cannot be verified by the end of the session is reverted to a clean tree. An unverified "fix" is worse than no fix: it corrupts the next investigator's baseline. The reverted session's deliverables are:

1. Findings written down (PR description, issue, or investigation notes in the ready-repro's doc comment).
2. An `#[ignore = "open: ..."]` ready-repro test that compiles, whose assertions encode the acceptance target for the eventual fix. `dovetail_cornerclip_intersect_inmem.rs` is the template: its assertions check watertight, analytic, and compact, exactly what the fix must produce.

Verification is empirical, not plausible: vary one variable, dump literal data, and check cut geometry with the ray-cast classifier (`brepkit_check::classify::classify_point`), never with volume alone (tessellated volume can mask an un-carved cut). See the solid-verification and debugging-doctrine skills.

When a fix ships for a previously ignored repro, remove the `#[ignore]` in the same PR so it becomes a permanent regression test.

## Golden-file update procedure

1. A golden test fails with `Golden file mismatch: <name>` and a diff.
2. Decide: is the output change intentional (better tessellation, new field) or a regression? Read the diff; do not skip this step.
3. If intentional: `UPDATE_GOLDEN=1 cargo test --workspace golden`, then inspect `git diff tests/golden/data/` and commit the data change with the code change.
4. If not intentional: it is a regression, do not regenerate.

Missing file panics with "Run with UPDATE_GOLDEN=1 to create it." New golden files follow `{shape}_{operation}.golden` and stay small.

## Anti-patterns

- Do not conclude a boolean fix works because volume looks right. Tessellation-based volume can read high and mask failures. Use `classify_point` probes and edge-use counts.
- Do not conclude a result is analytic because it is valid and manifold. Face count is the reliable tell: analytic results have a handful of faces, mesh fallback has hundreds to thousands of all-planar facets.
- Do not delete or skip an `#[ignore]` test because it fails when run with `--ignored`. It is supposed to fail, that is its job.
- Do not call wasm binding methods directly in tests. `JsError` cannot be constructed on non-wasm targets; go through `execute_batch` (see CLAUDE.md, Recipe 4).
- Do not hand-edit `.proptest-regressions` files. proptest writes failing seeds there automatically; commit them, they are regression tests.
- Do not leave a half-verified fix in the tree "for the next session". Revert and write the ready-repro instead.
- Do not skip git hooks to get a red tree pushed. Pre-commit and pre-push gates are the contract.

## Sibling skills

boolean-debugging (root-causing the failures these fixtures capture), solid-verification (the measurement oracles), debugging-doctrine (vary-one-variable, diagnosis instability), parity-benchmarking (scoring against the reference kernel via the brepjs harness), add-operation and wasm-bindings (where new-feature tests slot in), pr-workflow (review gates before merge).
