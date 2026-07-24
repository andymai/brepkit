---
name: parity-benchmarking
description: Use when proving brepkit matches or beats the reference kernel, running the brepjs wasm head-to-head benchmark, overlaying a local kernel build into the gridfinity tool, re-probing scenario face counts after a GFA or boolean change, capturing faithful fixtures (STEP or arena .bin) from a failing tool scenario, or running local criterion micro-benchmarks. Also use whenever a speedup or parity claim is about to be quoted anywhere.
---

# Parity benchmarking

Repo rule: never write the competing kernel's name in commits, PRs, code, or skill files. Call it "the reference kernel". The brepjs bench harness and the gridfinity tool already know it by their own internal kernel ids, so you never need to spell it out.

## When to use

- After ANY GFA (General Fuse Algorithm, the analytic boolean engine in `crates/algo`) or boolean-engine PR merges: re-probe face counts (see Probe discipline). This is mandatory, not optional.
- Before quoting any performance or parity number.
- When a gridfinity tool scenario fails and you need a Rust-side repro.
- When asked whether brepkit "wins" on some operation.

## The parity bar

Acceptance, per real gridfinity scenario: triangle-count parity, volume parity, manifold/watertight validity, AND wall-clock at least matching the reference kernel. All four, per case. The end goal is replacing the reference kernel's wasm build in `~/Git/gridfinity-layout-tool`.

Two traps baked into the bar:

1. **Mesh fallback passes correctness while failing everything that matters.** When the analytic boolean result fails the gate in `crates/operations/src/boolean/mod.rs` (find it: `rg -n 'mesh_boolean' crates/operations/src/boolean/mod.rs`), the op reruns as a triangle-mesh boolean. The result is valid, watertight, and often has FEWER triangles than the reference kernel, so triangle and validity checks both mask it. Face count is the tell: analytic results are roughly 3 to 80 faces with curved-surface types present; fallback is hundreds to thousands of faces, all planar.
2. **Timing is not correctness.** The head-to-head harness times ops but never verifies output equivalence. A 100x speedup usually means an analytic fast path short-circuited work the reference kernel actually did. Verify equal face counts and volumes on both sides before quoting any ratio. `cut(box, cylinder)` is the trustworthy apples-to-apples row.

A hang or multi-hour suite runtime is itself a perf-bar defect, not an environment problem to wait out.

## Quick reference

```bash
# Fast head-to-head (no npm install, no lockfile churn)
cd ~/Git/brepkit && wasm-pack build crates/wasm --target nodejs --release --out-dir pkg-node-bench
# then alias-swap in brepjs, see Procedure 1

# Tool overlay build+copy loop (release build with wasm-opt skipped, direct node_modules only)
./scripts/parity-loop.sh

# Face count from JS (the fallback tell): getEntityCounts(id)[0]
# binding: rg -n 'getEntityCounts' crates/wasm/src/bindings/query.rs

# Unified criterion + JS report (slow, batteries included)
./scripts/bench-compare.sh ~/Git/brepjs

# Analytic-vs-approximation census
cargo run --release --example approx_census -p brepkit-operations
```

## Procedure 1: fast head-to-head (wasm vs wasm)

The fair fight is brepkit-wasm vs the reference kernel's wasm build, both driven through brepjs. This recipe needs no `npm install`, so it sidesteps the brepjs lockfile and knip traps entirely.

1. Build: `wasm-pack build crates/wasm --target nodejs --release --out-dir pkg-node-bench` from the brepkit root. Dedicated out-dir so `crates/wasm/pkg` is not clobbered. Rename the emitted `.js` entry to `.cjs` (Node loads the bench as CJS; the published package ships `brepkit_wasm_node.cjs`).
2. Point brepjs at it: copy `~/Git/brepjs/vitest.bench.config.ts` to a scratch config and change only the `brepkit-wasm` entry in `resolve.alias` to the absolute path of your `pkg-node-bench/*.cjs`. Do not touch `node_modules`.
3. Run: `cd ~/Git/brepjs && BENCH_KERNELS=<ids> npx vitest run --config <your-config> kernel-comparison`. The `BENCH_KERNELS` syntax (single id, comma list) is documented in `~/Git/brepjs/benchmarks/setup.ts`; the actual kernel ids are defined in `~/Git/brepjs/tests/helpers/kernelInit.ts`. Include `brepkit` plus the reference kernel's wasm id.
4. Checkpoint: the run takes seconds, not minutes, and regenerates `benchmarks/results/latest.md`. If it hangs for minutes, you included a wrong kernel id (see reference.md, Head-to-head).
5. Before quoting any ratio, verify equivalence: same face counts (via `getEntityCounts`) and volumes within tolerance on both kernels for each quoted row.

Rules: never set `BENCH_KERNELS=all` or `both`, and never include `manifold` (mesh/CSG kernel, apples to oranges, and it hangs the suite for hours). Single-kernel runs OVERWRITE `latest.md`; `git checkout benchmarks/results/latest.md` after one-offs.

Details, bench-file inventory, and iteration multipliers: see reference.md, "Head-to-head".

## Procedure 2: gridfinity tool overlay (tool-level parity)

Heavier: replaces the tool's installed `brepkit-wasm` in place so the real generators run on your kernel.

1. Get the files: either `npm pack brepkit-wasm@<ver> --pack-destination <tmp>` and use `<tmp>/package/*`, or build locally with `cargo xtask wasm-build` (add `--skip-opt` for iteration speed) and use `crates/wasm/pkg/*`.
2. Copy the 6 package files into BOTH locations in `~/Git/gridfinity-layout-tool`: `node_modules/brepkit-wasm/` AND `node_modules/.pnpm/brepkit-wasm@<ver>/node_modules/brepkit-wasm/`. pnpm resolves through `.pnpm`; overlaying only the direct path is the classic wasted-probe mistake.
3. Clear the Vite cache: `rm -rf node_modules/.vite node_modules/.vite-temp` in the tool.
4. **Verify the overlay took effect before trusting any probe**: `md5sum` the `.wasm` in both locations against your source, or feature-detect a binding that only exists in the new build.
5. Probe: `BREPJS_KERNEL=brepkit pnpm exec vitest run --config vitest.profile.config.ts <probe>` (set `BREPJS_KERNEL` to the reference kernel's wasm id for the comparison side).
6. Restore afterward: `pnpm install --force` in the tool.

`scripts/parity-loop.sh` automates the local-build flavor but copies only the direct `node_modules` location; add the `.pnpm` copy yourself when the probe imports resolve through pnpm. Its built-in probe step targets a test file that no longer exists in the tool, so bring your own probe vitest. File list and probe assets: see reference.md, "Tool overlay".

## Probe discipline: re-probe face count after every GFA change

Scorecards go stale silently. A GFA PR once shipped on top of a "locked in" scorecard and had silently broken the stacking-lip fuse for every 2x2-and-larger bin (the tool's DEFAULT config): 78 analytic faces became 190 all-planar mesh-fallback faces. Triangle counts and validity both still passed. It was found only when the probe was rerun.

The rule: after ANY GFA or boolean PR, rerun the scenario matrix and record face counts per scenario and bin size. Face count via `getEntityCounts(solidId)[0]` (returns `[faces, edges, vertices]`; Rust side is `brepkit_topology::explorer::solid_entity_counts`). Do not use a fixed face-count threshold as the fallback detector; check the surface-type mix. All-planar with zero curved surfaces on a shape that should have cylinders is fallback regardless of the number.

Vitest gotcha: the forks pool swallows `console.log`. Write probe results to a file, do not rely on stdout.

**STL edge-use probe** — the cheapest tool-level correctness oracle, and the only one for `assert: 'snapshot'` scenarios, which pin `triangleCount` and never check validity. `exportBin(params, 'stl')`, `parseSTLBinary`, quantize each vertex to 1e-4 and count edge uses: used once = boundary (hole), used >2 = non-manifold. Watertight is 0/0. Pair it with wall-clock — a sibling scenario taking 20x longer is the fallback tell. Working example: `gomaBoundaryProbe.test.ts` in the tool's `__kernel-tests__/`; run with `--config vitest.profile.config.ts` (that dir is excluded from the normal projects).

## Faithful fixture capture

Never grind on a hand-built repro without proving it matches the real tool geometry first. Diagnosis that flip-flops across debugging passes is itself the signal that your repro is unfaithful.

- **Tier 1, STEP round-trip:** in a tool vitest, build up to the failing op, export the actual operands with `exportSTEP`, write to files, read into a Rust harness via `read_step` in `crates/io/src/step/reader.rs`, run the boolean directly. Checkpoint: confirm the operands round-trip ANALYTIC (cylinder/circle type tags, not NURBS). A lossy fixture produces overfit fixes that do not transfer.
- **Tier 2, arena serialization:** some failures exist only in the tool's in-memory id layout; STEP renumbers everything and the bug vanishes. Use `serializeSolid`/`deserializeSolid` (`crates/wasm/src/bindings/io.rs`) and `serialize_solid`/`deserialize_solid` (`crates/io/src/arena_io.rs`). Capture `.bin` operands during a tool probe, commit under `crates/io/tests/data/`, replay in a `*_inmem.rs` test. Existing examples: `crates/io/tests/dovetail_cornerclip_intersect_inmem.rs`, `gridfinity_lipfuse_dividers_inmem.rs`.

Capture gotcha and full patterns: see reference.md, "Fixture capture".

## Local micro-benchmarks

- `./scripts/bench-compare.sh ~/Git/brepjs`: unified criterion + JS report into `bench-results/`. It runs `npm install` into brepjs, so it is the slow path; use Procedure 1 for iteration.
- Criterion benches: `crates/operations/benches/` (`cad_operations.rs` is the main suite; also `boolean_perf.rs`, `boolean_tracking.rs`, `compound_cut_perf.rs`, `fuse_perf.rs`). Run one: `cargo bench -p brepkit-operations --bench cad_operations`.
- Census: `cargo run --release --example approx_census -p brepkit-operations` shows which ops stayed analytic vs which approximation path fired, with wall-clock and face counts.
- Flamegraphs: see the profiling skill and CLAUDE.md, "Profiling".

## Anti-patterns

- Do NOT conclude parity from triangle count plus validity. Both pass under mesh fallback. Check face count and surface-type mix.
- Do NOT quote a speedup from the bench harness alone. Verify output equivalence first.
- Do NOT trust any scorecard measured before the most recent GFA merge. Re-measure.
- Do NOT trust a tool probe until you verified the overlay landed in both node_modules locations and the Vite cache was cleared.
- Do NOT debug from a hand-built repro whose dimensions differ from the real scenario (real bins use values like corner radius 1.485, not round numbers).
- Do NOT keep debugging an in-memory-only failure through STEP fixtures. Switch to arena serialization.

Symptom-to-cause table and glossary: see [reference.md](reference.md).
