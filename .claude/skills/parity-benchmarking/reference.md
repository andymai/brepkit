# parity-benchmarking reference

Naming rule applies here too: the competing kernel is "the reference kernel". Its ids inside the brepjs harness are defined in `~/Git/brepjs/tests/helpers/kernelInit.ts` (`~/Git/brepjs/benchmarks/setup.ts` only documents the `BENCH_KERNELS` syntax); the tool's ids live in its kernel-test helpers. Do not write them into brepkit files.

## Symptom-to-cause table

| Symptom | Likely cause | First check |
|---|---|---|
| Scenario suddenly has hundreds of all-planar faces | Analytic boolean failed the gate, mesh fallback fired | `getEntityCounts(id)[0]` before/after; surface-type mix; then `rg -n 'mesh_boolean' crates/operations/src/boolean/mod.rs` and instrument the gate |
| Triangle count and validity pass but perf regressed | Mesh fallback (masked by both headline metrics) | Face count, not triangles |
| 100x+ speedup in the bench | Analytic fast path short-circuited; the kernels did different work | Compare face counts and volumes of both outputs |
| Bench suite hangs or runs for hours | Wrong kernel id in `BENCH_KERNELS` (`all`, `both`, or `manifold`) | Rerun with an explicit two-id list |
| Tool probe shows no change after overlay | Only one of the two node_modules locations was overlaid, or Vite cache stale | `md5sum` both `.wasm` copies; `rm -rf node_modules/.vite node_modules/.vite-temp` |
| Probe prints nothing | Vitest forks pool swallows `console.log` | Write results to a file from the test |
| Rust repro of a tool bug does not reproduce | Unfaithful fixture (wrong dimensions, or STEP normalized the failing state) | Verify STEP operands round-trip analytic; if the bug needs the in-memory id layout, switch to arena `.bin` fixtures |
| Fix works on the fixture but not in the tool | Fixture was lossy (NURBS where the tool had cylinders) | Re-capture and check type tags on read-back |
| `latest.md` shows only one kernel | A single-kernel run overwrote it | `git checkout benchmarks/results/latest.md` in brepjs |
| JS monkey-patch on `fuse`/`cut` intercepts nothing | The tool's batch executor bypasses JS-level kernel methods | Capture inside the kernel: instrumented wasm build that serializes operands |

## Head-to-head (brepjs bench harness)

Files in `~/Git/brepjs/benchmarks/`:

- `kernel-comparison.bench.test.ts`: the main comparison suite. Regenerates `benchmarks/results/latest.md` on every run that produced results, including single-kernel runs, which is why one-offs clobber it.
- `boolean.bench.test.ts`: fuse/cut/intersect through the brepjs top-level API via `benchBoth` from `setup.ts`.
- `setup.ts`: kernel init; reads `BENCH_KERNELS` (single id, comma list, or `all`/`both`; never use the last two).
- `harness.ts`: `bench`, `createMultiKernelBench`, `collectResults`, `printResults`, `writeResultsJSON`, `generateReport`. The `benchAndCollect` wrapper is local to `kernel-comparison.bench.test.ts`; the `benchBoth` used by `boolean.bench.test.ts` is exported from `setup.ts`.
- `results/latest.md`: tracked in git, regenerated per run.

Iteration multipliers inside `kernel-comparison.bench.test.ts` (one measurement wraps a loop): booleans and STEP export x10; translate x1000; rotate, volume, boundingBox, primitives x100; meshing, chamfer, fillet, multi-boolean single-op. Divide accordingly when comparing to criterion numbers.

Config template (`~/Git/brepjs/vitest.bench.config.ts`, verified):

```ts
resolve: {
  alias: {
    '@': resolve(__dirname, 'src'),
    'brepkit-wasm': resolve(__dirname, 'node_modules/brepkit-wasm/brepkit_wasm_node.cjs'),
  },
},
test: {
  globals: true,
  setupFiles: ['./tests/setup.ts'],
  testTimeout: 120000,
  pool: 'forks',
  execArgv: ['--max-old-space-size=6144'],
  include: ['benchmarks/**/*.bench.test.ts'],
},
```

For the fast recipe, copy this and repoint only the `brepkit-wasm` alias at your local `pkg-node-bench` `.cjs`. That is the entire integration: no install, no lockfile change, no type-sync, which avoids two documented brepjs traps (a lockfile-churn dependency problem and knip false-flagging exports on push).

The report baselines its "vs" column to the native reference-kernel id and warns when the subset omits it. That 3-kernel table is for the README-style overview; for parity work the wasm-vs-wasm pair is the fair fight.

Sanity anchors for quoted numbers: `cut(box, cylinder)` makes both kernels do comparable general-boolean work. Rows where brepkit hits an analytic fast path are legitimate wins only after you confirm the outputs are equivalent (face count, volume).

## Tool overlay (gridfinity-layout-tool)

Package file set (copy all 6):

```
brepkit_wasm_bg.js  brepkit_wasm_bg.wasm  brepkit_wasm.d.ts
brepkit_wasm.js     brepkit_wasm_node.cjs package.json
```

Both destinations in `~/Git/gridfinity-layout-tool`:

```
node_modules/brepkit-wasm/
node_modules/.pnpm/brepkit-wasm@<ver>/node_modules/brepkit-wasm/
```

Find the exact `.pnpm` dir with `ls node_modules/.pnpm | grep brepkit-wasm`. There may also be a combined `brepjs@...` entry in `.pnpm`; the one to overlay is the standalone `brepkit-wasm@<ver>` dir, but check where the probe's import actually resolves if in doubt.

Sources:

- Published: `npm pack brepkit-wasm@<ver> --pack-destination <tmp>`, then copy `<tmp>/package/*`.
- Local: `cargo xtask wasm-build` (flags: `--no-simd`, `--skip-opt`; builds both targets, runs wasm-opt unless skipped, merges into `crates/wasm/pkg/`).

`scripts/parity-loop.sh` = xtask release build with wasm-opt skipped (`--skip-opt` skips only the wasm-opt post-pass; the compile is still `--release`, so timings are near-representative), copy to the DIRECT location only, then a `vitest run ... topologyParity` step. That probe file has been removed from the tool, so the filter matches nothing and the script today only builds and overlays; write your own probe vitest (consistent with "do not expect ready-made probe files" below). The script also echoes the name of a JSON cache of the reference kernel's topology results at the tool root; delete that cache to re-measure the reference side.

Overlay verification (do this before every probe batch):

```bash
md5sum crates/wasm/pkg/brepkit_wasm_bg.wasm \
  ~/Git/gridfinity-layout-tool/node_modules/brepkit-wasm/brepkit_wasm_bg.wasm \
  ~/Git/gridfinity-layout-tool/node_modules/.pnpm/brepkit-wasm@*/node_modules/brepkit-wasm/brepkit_wasm_bg.wasm
```

All three hashes must match. The documented failure mode is several probes wasted on a half-applied overlay.

Probe assets in the tool (`src/features/generation/worker/generators/__kernel-tests__/`): `testCases.ts` (the canonical kernel parity matrix, roughly a dozen cases), `meshAssertions.ts`, `scenarioRunner.ts`, `scenarioTypes.ts`, `kernelInit.ts`/`dualKernelInit.ts`. Beyond that, the repo has ~80 `*.scenario.*.test.ts` files: a wide pre-validated oracle. Kernel selection per run: `BREPJS_KERNEL=<id> pnpm exec vitest run --config vitest.profile.config.ts <pattern>`.

Ad-hoc probes (per-scenario face-count sweeps, perf sweeps) are write-your-own: a small vitest that runs the generator per scenario and dumps `getEntityCounts` results to a file. Do not expect ready-made probe files in the tool repo.

Restore: `pnpm install --force` in the tool when done.

## Face-count probe details

Binding (`crates/wasm/src/bindings/query.rs`, `rg -n 'getEntityCounts'`):

```rust
#[wasm_bindgen(js_name = "getEntityCounts")]
pub fn get_entity_counts(&self, solid: u32) -> Result<Vec<u32>, JsError>
```

Returns `[faces, edges, vertices]` via `brepkit_topology::explorer::solid_entity_counts`. Tool-side JS shape: `getRawBrepkitKernel().getEntityCounts(getSolidId(solid))[0]` (helpers in the tool's `__kernel-tests__/dualKernelInit.ts`). For the surface-type mix, the same raw kernel exposes `getSolidFaces(solidId)` and `getSurfaceType(faceId)`; count planar vs curved.

Reading the number:

- Healthy analytic result: order of 3 to 80 faces, curved surface types present where the inputs had them.
- Mesh fallback: hundreds to thousands of faces, all planar. A fixed threshold once missed a 436-face all-planar fallback, so classify by surface-type mix (planar vs cylinder/cone/sphere/torus counts), not by a cutoff.
- Rust side, the only reliable post-hoc tell that the gate rejected the analytic result is the face count and type mix of the RESULT; the gate itself lives in `crates/operations/src/boolean/mod.rs` (`rg -n 'mesh_boolean'`).

Record per scenario AND per bin size. The shipped-regression case broke only 2x2-and-larger bins; a 1x1-only probe would have shown green.

## Fixture capture details

Fixture-tier APIs and test templates: see the testing skill. Parity-specific notes:

- Tier 1 (STEP): in a tool vitest, construct the case up to (not including) the failing op, export both operands with `k.unwrap(k.exportSTEP(operand))`, replay in a Rust harness via `read_step`, call the boolean directly. Checkpoint before debugging: the read-back faces/edges must carry analytic type tags (Cylinder, Circle) where the tool had them; NURBS-ified operands mean a lossy fixture and a fix that will not transfer.
- Tier 2 (arena `.bin`): some failures depend on the exact in-memory edge/vertex id layout; STEP renumbers into a clean layout and the repro vanishes, and repeated debugging passes then overturn each other. That instability is the signal to switch tiers. Templates: `crates/io/tests/dovetail_cornerclip_intersect_inmem.rs`, `gridfinity_lipfuse_dividers_inmem.rs`, `scoop_fix_inmem.rs`, `gridfinity_wallcut_seq_inmem.rs`.
- Capture gotcha: the tool's batch executor bypasses JS-level kernel methods, so monkey-patching `fuse`/`cut` in JS intercepts zero booleans. Capture requires an instrumented wasm build that serializes operands inside the kernel, run through the tool overlay. That occupies the tool checkout; coordinate if another agent owns it.

Repro fidelity: before grinding on any repro, diff its dimensions against the real scenario (real gridfinity bins use values like corner radius 1.485; a round-number stand-in once cost ~10 debugging passes).

## Micro-benchmark details

`scripts/bench-compare.sh <path-to-brepjs>`, five steps:

1. `cargo bench -p brepkit-operations` (criterion) into `bench-results/criterion.log`.
2. `wasm-pack build crates/wasm --target nodejs --release --out-dir crates/wasm/pkg`.
3. `npm install` of the built package into brepjs with `--no-save` (this is why it is the slow path; it also touches node_modules state).
4. `BENCH_OUTPUT_JSON=1 npx vitest run benchmarks/kernel-comparison.bench.test.ts --config vitest.bench.config.ts --reporter=verbose`, JSON extracted between `--- BENCHMARK RESULTS JSON ---` sentinels.
5. `npx tsx scripts/bench-report.ts --criterion-dir target/criterion --js-json <file> --output-dir bench-results` producing `bench-results/report.md` and `comparison.json`.

Criterion benches in `crates/operations/benches/`:

- `cad_operations.rs`: primitives x100, fuse/cut/intersect x10, sphere meshing, chamfer, fillet, multi-boolean model, gridfinity 1x1 bin, 3x3 baseplate, 64-cut 8x8 grid, tessellation cases. The names inside the file are the `--bench` filter strings for flamegraphs.
- `boolean_perf.rs`, `boolean_tracking.rs`, `compound_cut_perf.rs`, `fuse_perf.rs`: targeted boolean perf suites.

Census: `crates/operations/examples/approx_census.rs` (`boolean_matrix()` plus per-op sections). Run `cargo run --release --example approx_census -p brepkit-operations`. Output rows show, per operation, whether it stayed exact-analytic or which approximation fallback fired, with wall-clock and face count. This is the standing scoreboard for the analytic-preservation goal (see the analytic-preservation skill).

Flamegraphs: `cargo flamegraph --profile profiling --bench cad_operations -p brepkit-operations -o /tmp/flamegraph.svg -- --bench "<filter>"` (see the profiling skill).

## Glossary

- **GFA**: brepkit's General Fuse Algorithm, the analytic boolean engine (`crates/algo/src/gfa.rs`): intersect all faces (PaveFiller), split, classify, reassemble.
- **PaveFiller**: GFA phase 1 (`crates/algo/src/pave_filler/`), pairwise interference phases VV/VE/EE/VF/EF/FF; edges split at intersection points ("paves").
- **FF phase**: face-face intersection (`pave_filler/phase_ff.rs`), producer of section curves; historically the richest bug vein.
- **SD / same-domain**: detection of coincident/overlapping faces between operands (`detect_same_domain`), merged rather than intersected. Coincident-contact bugs usually live in the classifier, not SD; rule SD out by instrumenting it first (see the boolean-debugging skill).
- **Mesh fallback**: rerun of a failed analytic boolean as triangle-mesh co-refinement. Correct and watertight, but slow and it destroys analytic surfaces.
- **Analytic face**: exact surface (Plane/Cylinder/Cone/Sphere/Torus) rather than NURBS or mesh facets.
- **Entity counts**: `getEntityCounts(solid)` returning `[faces, edges, vertices]`; faces is the fallback tell.
- **Overlay**: in-place replacement of the tool's installed `brepkit-wasm` files (both pnpm locations) to test an unpublished kernel.
- **Faithful fixture**: captured actual tool operands (STEP or arena `.bin`) replayed in a Rust test, as opposed to a hand-built repro.
- **Arena serialization**: binary dump of the exact in-memory topology arena, preserving failing states that STEP normalizes away.
- **Scorecard**: per-scenario brepkit-vs-reference table (time, faces, triangles, validity). Stale the moment any GFA PR merges; never cite one without re-measuring.
- **Census**: the `approx_census` example output, exact-analytic vs approximation per operation.

## Sibling skills

boolean-debugging (GFA failure triage), solid-verification (validity checks), analytic-preservation (keeping surfaces exact), tessellation (triangle counts), profiling (flamegraphs), release-flow (publishing brepkit-wasm and the brepjs sync), wasm-bindings (adding probe bindings), testing, debugging-doctrine.
