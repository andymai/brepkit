---
name: wasm-bindings
description: Use when adding or changing methods on BrepKernel in crates/wasm, wiring ops into executeBatch, writing tests for bindings, building the wasm package (wasm-pack or cargo xtask wasm-build), testing a local kernel build inside brepjs or another JS consumer, or debugging wasm-only failures such as tests that will not compile with JsError, panics that abort the wasm instance, "recursive use of an object" errors, or a JS app silently loading a stale wasm build. Also use when the CI "WASM Binary Size" PR comment shows growth or a change adds a dependency to the wasm graph.
---

# wasm-bindings

The WASM layer exports one class, `BrepKernel` (`crates/wasm/src/kernel.rs`). JS holds opaque `u32` handles (arena indices into the kernel's `Topology`, NOT batch result indices). Bindings live in `crates/wasm/src/bindings/<domain>.rs`, one `#[wasm_bindgen] impl BrepKernel` block per module (wasm-bindgen supports multiple impl blocks across files; non-exported helpers go in a separate plain `impl BrepKernel` block).

## Quick reference

| Task | Command | Expect |
|---|---|---|
| Type-check for wasm target | `cargo build -p brepkit-wasm --target wasm32-unknown-unknown` | Clean build; catches target-specific errors |
| Run contract tests | `cargo test -p brepkit-wasm` | Native target; see "Contract tests" below |
| Build for node/vitest | `wasm-pack build crates/wasm --target nodejs --release` | `crates/wasm/pkg/` with CJS entry `brepkit_wasm.js` |
| Full release package | `cargo xtask wasm-build` | Dual-target merge + wasm-opt; entry `brepkit_wasm_node.cjs`. See [reference.md](reference.md) section 1 |
| Smoke-test the package | `node scripts/test-wasm-smoke.mjs` | `ok - ...` lines, ends `All smoke tests passed`. Only works after xtask build (needs `brepkit_wasm_node.cjs`) |

Never `wasm-pack build --target web` for node consumption: the web target's init uses `fetch()` and fails under node. The bundler ESM entry also fails under vitest.

## Procedure: add a binding

CLAUDE.md Recipe 4 covers the skeleton (js_name, `validate_positive`/`validate_finite`, `?` via the blanket `From<E> for JsError`). What it omits or gets stale:

1. Pick or create `crates/wasm/src/bindings/<domain>.rs`. Do not add bindings to `kernel.rs` (it holds the struct, constructor, and private helpers only, despite Recipe 1's stale wording).
2. Reads go through `self.topo` / `self.topo()`; writes MUST go through `self.topo_mut()` (it is `Rc::make_mut` copy-on-write shared with checkpoints; rg: `fn topo_mut` in `kernel.rs`).
3. Resolve incoming handles with `resolve_solid` / `resolve_face` / etc. from `crates/wasm/src/handles.rs`; convert results back with `*_id_to_u32`.
4. **Batch dispatch**: there are no `batch_*` companion functions (Recipe 4's wording is stale; the only `batch_*` names are test helpers). To add an op to `executeBatch`, add a match arm in `fn dispatch_op` in `crates/wasm/src/bindings/batch.rs`. Parse args with `get_f64` / `get_u32` from `crates/wasm/src/helpers.rs`, map errors with `.map_err(|e| e.to_string())`, return `Ok(serde_json::json!(...))`. Copy the `"makeBox"` arm as the template. Note: `tessellateSolid` is deliberately not in the dispatcher.
5. **Panic safety** for panic-prone ops (booleans, fillets): wrap the body in `std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ...))` and build the error with `panic_message` (`crates/wasm/src/helpers.rs`, rg: `fn panic_message`). Reason: a panic crossing the WASM FFI boundary aborts the entire wasm instance. Live examples: `fillet` in `bindings/operations.rs`, `compound_cut` in `bindings/booleans.rs` (that one also checks and sets `self.poisoned`; subsequent guarded calls refuse with "Kernel poisoned after panic"). Do NOT use the `#[wasm_binding]` attribute from `crates/wasm-macros`: the macro is fully written but unwired (not a dependency of `brepkit-wasm`, zero usages). The manual pattern is the live one.
6. Checkpoint: `cargo build -p brepkit-wasm --target wasm32-unknown-unknown` compiles, then write a contract test (below), then `cargo test -p brepkit-wasm`.

## Contract tests: why execute_batch, not method calls

`JsError::new` panics on non-wasm targets. Any `#[wasm_bindgen]` method returning `Result<_, JsError>` or `JsValue` cannot be exercised on the error path (or at all, for `JsValue`) under native `cargo test`. Three patterns that work, all in-tree:

1. **Batch contract test** (default): `let mut k = BrepKernel::new(); let r = k.execute_batch(r#"[{"op":"makeBox","args":{"width":10.0,"height":10.0,"depth":10.0}}]"#);` then parse `r` with `serde_json` and assert on `{"ok": ...}` / `{"error": "..."}` entries. Examples: `mod batch_tests` in `kernel.rs`, tests in `bindings/booleans.rs`, `bindings/gridfinity_tests.rs`.
2. **`pub(crate)` `_impl` functions returning `WasmError`** for logic that must be unit-testable natively (rg: `fn from_brep_impl` in `kernel.rs`).
3. **Error paths through `resolve_*`** (return `WasmError`, safe natively). See the comment in `bindings/query/tests.rs`.

Batch semantics to assert against: errors do not stop later ops in the array; invalid JSON returns `[{"error": "invalid JSON: ..."}]`.

## Symptom-to-cause

| Symptom | Cause | Fix |
|---|---|---|
| Test compiles-fails or panics with a JsError construction | Calling a `JsError`/`JsValue` method under native `cargo test` | Rewrite as an `execute_batch` contract test or `_impl` fn |
| JS reports "recursive use of an object detected which would lead to unsafe aliasing" | Observed symptom: a panic inside wasm on an earlier call, often `std::time::Instant::now()` (panics on wasm32, no monotonic clock). The re-entrancy error surfaces on the next call, masking the real cause | Find bare `Instant::now()` reachable from wasm (`operations` and below). Use the cfg-gated `timer_now` / `timer_elapsed_ms` pattern (rg: `fn timer_now` in `crates/operations/src/boolean/mod.rs`) |
| Wasm instance dead after one bad op | Uncaught panic crossed the FFI boundary | Wrap the op in `catch_unwind` + `panic_message` (step 5 above) |
| `cargo xtask wasm-build` bails: wasm-bindgen-cli version mismatch demanding 0.2.121 | Stale `WASM_BINDGEN_VERSION` constant in `xtask/src/wasm.rs`; workspace `Cargo.toml` pins `wasm-bindgen = "=0.2.125"` | Update the constant to match the workspace pin, or uninstall the local cli and let wasm-pack fetch its own |
| `node scripts/test-wasm-smoke.mjs` cannot find `brepkit_wasm_node.cjs` | Plain wasm-pack build names the entry `brepkit_wasm.js`; only the xtask merge produces the `.cjs` | Run `cargo xtask wasm-build` first, or point your consumer at `brepkit_wasm.js` directly |
| JS consumer still runs old kernel behavior after overlaying a local build | pnpm `.pnpm` store copy not overlaid, or stale Vite dep-optimizer cache | See [reference.md](reference.md) section 2: copy into BOTH node_modules locations, `rm -rf node_modules/.vite*`, verify with md5 or a new binding key |

## Testing a local kernel in a JS consumer

Cheap path first: brepjs already aliases `'brepkit-wasm'` in `~/Git/brepjs/vitest.config.ts` (also `vitest.bench.config.ts`, `vitest.stress.config.ts`). Build `wasm-pack build crates/wasm --target nodejs --release`, then temporarily point that alias at `~/Git/brepkit/crates/wasm/pkg/brepkit_wasm.js`. No npm install, no lockfile churn. Full overlay recipe for pnpm+Vite apps: [reference.md](reference.md) section 2.

## Anti-patterns

- Do not conclude "the binding is broken" from a native test that fails constructing `JsError`; that is the test pattern, not the binding.
- Do not conclude "re-entrancy bug" from the "recursive use of an object" error; look for an earlier panic first.
- Do not trust that a JS consumer picked up your local build; verify (md5 the `.wasm`, or probe a binding that only exists in the new build).
- Do not add timing code with bare `std::time::Instant` anywhere reachable from `brepkit-wasm`.
- Do not follow CLAUDE.md Recipe 4's "add a `batch_*` companion fn" literally; add a `dispatch_op` match arm.
- Do not reach for the `#[wasm_binding]` macro; it is unwired.

## Binary size

Small wasm size is a headline competitive property of this kernel (several times smaller than the reference kernel's wasm). The `wasm-size` CI job only posts a PR comment ("WASM Binary Size") and never fails, so read that comment as a review item on every wasm-touching PR. Rule: an unexplained increase above roughly 1% needs a justification sentence in the PR body; a new dependency in the wasm graph always does. Measurement details and the twiggy diagnosis recipe: [reference.md](reference.md) section 6.

## Related skills

`add-operation` (the L3 operation a binding usually wraps), `testing` (workspace test conventions), `release-flow` (publishing the package), `parity-benchmarking` (running the brepjs bench harness against a local build), `layer-boundaries` (what `wasm/src` may import).
