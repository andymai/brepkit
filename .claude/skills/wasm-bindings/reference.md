# wasm-bindings reference

Deep detail backing SKILL.md. Everything here is verified against the repo; re-verify with the given rg patterns if a symbol has moved.

## 1. cargo xtask wasm-build (the release-grade package)

`xtask` is aliased in `.cargo/config.toml` (`xtask = "run --manifest-path xtask/Cargo.toml --"`) and is intentionally not a workspace member. Source: `xtask/src/main.rs`, `xtask/src/wasm.rs`.

`cargo xtask wasm-build [--no-simd] [--skip-opt]` runs five steps:

1. `check_tools`: requires `wasm-pack`. If a local `wasm-bindgen-cli` is installed it must equal the pinned `WASM_BINDGEN_VERSION` constant in `xtask/src/wasm.rs`. Known gotcha: that constant is `0.2.121` while the workspace `Cargo.toml` pins `wasm-bindgen = "=0.2.125"`, so a correctly matched local cli fails the check. Fix the constant or uninstall the cli (wasm-pack fetches its own). `wasm-opt` is optional; missing means a warning and a skipped step.
2. `build_both_targets`: two wasm-pack builds from `crates/wasm` with `RUSTFLAGS="-Dwarnings"` (plus `-C target-feature=+simd128` unless `--no-simd`): `--target bundler --out-dir pkg` and `--target nodejs --out-dir pkg-node`.
3. `run_wasm_opt`: `wasm-opt -O3` in place on `pkg/brepkit_wasm_bg.wasm`, prints before/after KB.
4. `merge_packages`: copies `pkg-node/brepkit_wasm.js` into `pkg/` as `brepkit_wasm_node.cjs` (renamed to `.cjs` because the bundler package.json has `"type": "module"`), patches `pkg/package.json` (`name: brepkit-wasm`, `main: brepkit_wasm_node.cjs`, `module: brepkit_wasm.js`, an `exports` map with node/import/default conditions, `files`), then deletes `pkg-node/`.
5. `validate_output`: checks the required files exist, `.wasm` size in a 500 KB to 20 MB window, the `.d.ts` contains `export class BrepKernel` with at least `MIN_METHOD_COUNT` methods, and package.json fields.

Output lands in `crates/wasm/pkg/`, git-ignored by wasm-pack's own generated `crates/wasm/pkg/.gitignore`.

`cargo xtask wasm-publish [--dry-run]` runs the same pipeline, then `node scripts/test-wasm-smoke.mjs`, then `npm publish --provenance --access public`. It requires a `TAG_NAME` env var matching the `package.json` version.

### Which build for which purpose

| Purpose | Build | Entry file |
|---|---|---|
| Node / vitest, quick iteration | `wasm-pack build crates/wasm --target nodejs --release` | `crates/wasm/pkg/brepkit_wasm.js` (CJS despite the name) |
| Published package, smoke test, overlay into a real app | `cargo xtask wasm-build` | `crates/wasm/pkg/brepkit_wasm_node.cjs` (node) + `brepkit_wasm.js` (bundler ESM) |
| Browsers via a bundler | the same xtask package; the bundler resolves the `import` condition | `brepkit_wasm.js` |

Never `--target web` for node: its init path uses `fetch()`. The bundler ESM entry also fails under vitest because Vite resolves the `import` exports condition to an entry that uses the unsupported WASM ESM integration proposal (see the comment in `~/Git/brepjs/vitest.config.ts`). Node and vitest must get the CJS entry.

## 2. Running a local kernel build inside a JS consumer

### Cheap path: vitest resolve.alias (preferred, no npm install)

brepjs maps `'brepkit-wasm'` to `resolve(__dirname, 'node_modules/brepkit-wasm/brepkit_wasm_node.cjs')` in `vitest.config.ts`, `vitest.bench.config.ts`, and `vitest.stress.config.ts`. To test a local kernel:

```bash
wasm-pack build crates/wasm --target nodejs --release
```

Then temporarily change the alias target to `~/Git/brepkit/crates/wasm/pkg/brepkit_wasm.js` (plain nodejs build; only the xtask merge renames the entry to `.cjs`). Run the tests or bench, then revert the alias. This sidesteps npm install and the lockfile and lint-config churn that comes with it.

### Full overlay: pnpm + Vite consumers (e.g. gridfinity-layout-tool)

pnpm resolves through its `.pnpm` store, so `node_modules/brepkit-wasm/` is a link-like copy of `node_modules/.pnpm/brepkit-wasm@<ver>/node_modules/brepkit-wasm/`. Overlaying only one location silently loads the old build; Vite's dep-optimizer cache adds a second layer of staleness. The full recipe (copy into BOTH node_modules locations, `rm -rf node_modules/.vite*`, md5-verify all copies, restore with `pnpm install --force`) is owned by the parity-benchmarking skill, Procedure 2. Get the package contents from `crates/wasm/pkg/` after `cargo xtask wasm-build`, or unpack `npm pack brepkit-wasm@<ver>`. Alternative overlay check: probe for a binding key that only exists in the new build (`typeof kernel.myNewMethod === 'function'`).

## 3. dispatch_op anatomy

`executeBatch` (`pub fn execute_batch(&mut self, json: &str) -> String`, rg: `js_name = "executeBatch"` in `crates/wasm/src/bindings/batch.rs`) takes a JSON array of `{"op": "...", "args": {...}}` and returns a JSON array of `{"ok": <value>}` / `{"error": "<msg>"}`. One boundary crossing for N ops. An error entry does not stop later ops; invalid input JSON returns `[{"error": "invalid JSON: ..."}]`.

Adding an op is a match arm in `fn dispatch_op(&mut self, op: &str, args: &serde_json::Value) -> Result<serde_json::Value, String>`:

```rust
"makeBox" => {
    let w = get_f64(args, "width")?;
    let h = get_f64(args, "height")?;
    let d = get_f64(args, "depth")?;
    let solid = brepkit_operations::primitives::make_box(self.topo_mut(), w, h, d)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!(solid_id_to_u32(solid)))
}
```

Arg helpers `get_f64` / `get_u32` live in `crates/wasm/src/helpers.rs` and return `Result<_, String>` so `?` composes directly. Optional args: `get_u32(args, "segments").unwrap_or(16)`.

Handles in batch args are arena indices returned by earlier calls (batched or not), NOT positions in the batch result array. The header of `bindings/gridfinity_tests.rs` states this and its tests show the pattern.

`tessellateSolid` is not in the dispatcher (stated in the `bindings/gridfinity_tests.rs` header); call it as a direct method. Its tessellation reproducer tests therefore live in `crates/operations/src/tessellate/` instead of the wasm contract tests.

## 4. Panic safety, verbatim pattern

Two live variants, both hand-rolled with `panic_message` from `crates/wasm/src/helpers.rs`:

- Non-poisoning (fillet): `bindings/operations.rs`, rg: `catch_unwind` in that file. Wraps the body, converts a panic payload into a `JsError` via `panic_message(&panic_info, "Fillet")`, kernel stays usable.
- Poisoning (`compoundCut`): `bindings/booleans.rs`, in its own impl block. Checks `self.poisoned` first and refuses with "Kernel poisoned after panic. Create a new BrepKernel instance."; on a caught panic sets `self.poisoned = true` because the topology may be half-mutated.

Choose poisoning when the op mutates `self.topo` before the panic-prone phase; choose non-poisoning when the mutation is atomic-ish or recoverable. Either way the point is the same: an uncaught panic across the FFI boundary aborts the whole wasm instance, taking every handle with it.

`crates/wasm-macros/src/lib.rs` contains a complete `#[wasm_binding]` attribute macro (js_name plus a `recoverable` mode with snapshot restore) but it is unwired: `brepkit-wasm-macros` is not in `crates/wasm/Cargo.toml` and `rg -n 'wasm_binding' crates/wasm/src` finds nothing. Treat it as a possible future consolidation, not the recipe.

## 5. scripts/test-wasm-smoke.mjs

Loads `crates/wasm/pkg/brepkit_wasm_node.cjs` via `createRequire` (CJS from ESM), then asserts in order:

1. `new BrepKernel()` constructs.
2. `makeBox(10, 20, 30)` returns a number handle.
3. `volume(box, 0.1)` is within 1e-6 of 6000.
4. `tessellateSolid` returns non-empty positions and indices, both multiples of 3.
5. `exportStl` if present; feature-detected, skipped when the `io` cargo feature is off (`io` is a default feature per `crates/wasm/Cargo.toml`).

Output shape: `ok - ...` per check, final line `All smoke tests passed`. Run it after every `cargo xtask wasm-build` (the build prints a reminder); `cargo xtask wasm-publish` runs it automatically. It will not work after a plain wasm-pack build because that names the entry `brepkit_wasm.js`, not `brepkit_wasm_node.cjs`.

## 6. Binary size: budget and diagnosis

Small wasm size is a headline competitive property of this kernel. The shipped `.wasm` is several times smaller than the reference kernel's. Nothing in CI enforces it, so erosion is silent unless someone reads the size comment.

### What CI measures

The `wasm-size` job ("WASM Size Report") in `.github/workflows/ci.yml` runs on PRs only. It builds `cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release` for both the PR head and the base branch, stats `target/wasm32-unknown-unknown/release/brepkit_wasm.wasm`, and posts or updates one PR comment titled "WASM Binary Size" with a main/PR/delta table. It is informational only: it has no failure path, and growth above 50 KB just adds a "Please verify this is expected" line to the comment. The only hard gate anywhere is the coarse `validate_output` window in `xtask/src/wasm.rs` (`MIN_WASM_SIZE` 500 KB, `MAX_WASM_SIZE` 20 MB). Treat the size comment as a review item on every PR that touches `crates/wasm` or anything in its dependency tree.

The CI number is the un-optimized cargo build. The shipped artifact goes through `cargo xtask wasm-build`, which runs `wasm-opt -O3` on `pkg/brepkit_wasm_bg.wasm` and prints before and after KB. The two deltas usually track each other; confirm a suspicious delta on the wasm-opt output before acting on it.

### Budget rule

- An unexplained size increase above roughly 1% of the base size needs a justification sentence in the PR body.
- Adding any new dependency to the wasm dependency tree always needs one, even if the measured delta is small.

### Diagnosis recipe

1. Reproduce the CI measurement on both branches (main via a worktree or stash):

```bash
cargo build -p brepkit-wasm --target wasm32-unknown-unknown --release
stat -c %s target/wasm32-unknown-unknown/release/brepkit_wasm.wasm
```

2. Confirm on the shipped path: `cargo xtask wasm-build`, read the wasm-opt before/after line, or stat `crates/wasm/pkg/brepkit_wasm_bg.wasm`.
3. Attribute the growth with twiggy. Check availability first (`command -v twiggy`; `cargo install twiggy` if missing):

```bash
twiggy top -n 30 target/wasm32-unknown-unknown/release/brepkit_wasm.wasm
twiggy diff base.wasm pr.wasm
twiggy monos target/wasm32-unknown-unknown/release/brepkit_wasm.wasm
```

Without twiggy, optimize both files with `wasm-opt -O3` and compare the optimized sizes. Growth that survives `-O3` is real code or data, not padding.
4. Check the dependency graph for a new crate and diff against main:

```bash
cargo tree -p brepkit-wasm --target wasm32-unknown-unknown -e normal
```

### Usual suspects

- A new crate pulled into the wasm graph, often transitively. The `cargo tree` diff finds it.
- Format and panic string bloat: `format!` in error paths, derived `Debug` on large types, long message literals. Shows up in twiggy as `core::fmt` frames and data-segment growth.
- Code that should be feature-gated. `crates/wasm/Cargo.toml` defines `default = ["io"]` and `io = ["dep:brepkit-io"]`; a heavy optional capability should follow that pattern instead of landing unconditionally.
- Monomorphization bloat from a generic instantiated with many types. `twiggy monos` lists it.

The release profile is already size-conscious (`codegen-units = 1`, `lto = true` in the root `Cargo.toml` `[profile.release]`). Attribute the growth before reaching for profile tweaks.

## 7. Glossary

- **BrepKernel**: the single WASM-exported modeling context; owns all topology; JS holds only opaque `u32` handles.
- **Handle**: a `u32` arena index into the kernel's `Topology`, resolved via `resolve_*` in `crates/wasm/src/handles.rs`. Not a batch result index.
- **executeBatch / dispatch_op**: the one-boundary-crossing JSON batch API and its string-keyed dispatcher in `bindings/batch.rs`.
- **Contract test**: a native `cargo test` exercising bindings via `execute_batch` JSON or `_impl` fns, because `JsError`/`JsValue` panic when constructed off-wasm.
- **Poisoned kernel**: `poisoned: bool` set after a caught panic in a non-recoverable op; guarded calls then refuse.
- **bundler vs nodejs target**: wasm-pack output flavors. ESM for bundlers (uses the WASM ESM integration proposal, unsupported in node) vs CJS for node. The published package merges both.
- **Dual-target merge**: the xtask step producing `brepkit_wasm_node.cjs` and the patched `exports` map.
- **Overlay**: hand-copying a local build over an installed `brepkit-wasm` in a consumer's node_modules (both direct and `.pnpm` copies) plus clearing the `.vite` cache.
- **tsify**: derive crate generating TypeScript types for the typed result structs in `crates/wasm/src/types.rs`.
- **io feature**: default cargo feature gating `bindings/io.rs` (STEP, STL, and the other formats).
