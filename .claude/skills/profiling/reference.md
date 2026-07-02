# Profiling reference

Everything here is verified against the current tree. Symbols move; re-find them with the rg patterns given, not line numbers.

## Tooling catalog

### The profiling profile

Workspace `Cargo.toml`:

```toml
[profile.profiling]
debug = true
inherits = "release"
lto = false
opt-level = 3
```

Release-speed code, debug symbols, no LTO. Plain `release` has `lto = true` and `codegen-units = 1`: slow to rebuild and flamegraph frames get inlined/mangled beyond use.

### Flamegraphs

`cargo-flamegraph` is installed. Two targets:

```bash
cargo flamegraph --profile profiling --bench <bench_file> -p brepkit-operations \
  -o /tmp/flamegraph.svg -- --bench "<filter>"

cargo flamegraph --profile profiling --example profile_boolean -o /tmp/flame.svg -- honeycomb
```

The trailing `-- --bench "<filter>"` is criterion syntax: `--bench` switches the harness to bench mode, the string is a substring filter selecting benchmark IDs. Output is an SVG; open in a browser, click frames to zoom. Wide frame = time spent (inclusive). Read bottom-up from `main` to find the hot subsystem, then top-down inside it.

### Criterion benches

All five files in `crates/operations/benches/`, each `harness = false` in `crates/operations/Cargo.toml`.

- `cad_operations.rs`: the head-to-head suite: `makeBox(10,20,30) x100`, `fuse(box,box) x10`, `cut(box,cyl) x10`, `intersect(box,sphere) x10`, `mesh box (tol=0.1)`, `mesh sphere (tol=0.01)`, `tessellate 64-hole plate`, `boolean 64 cuts (8x8 grid)`, `gridfinity 1x1 bin (box+shell+chamfer)`, and more. Native and JS numbers line up via the hand-maintained `NAME_MAP` table in `scripts/bench-report.ts` (rg: `NAME_MAP`), which pairs each criterion name with its counterpart in `brepjs/benchmarks/kernel-comparison.bench.test.ts`. The names do not match literally: criterion names carry an ` xN` repetition suffix (`fuse(box,box) x10`), the JS names do not (`fuse(box,box)`, `translate ×1000`). Any new head-to-head benchmark must be added to `NAME_MAP` or it will be silently missing from `bench-results/report.md`.
- `boolean_perf.rs`: scaling behavior. Group `sequential_cylinder_cuts` with IDs `N=4`, `N=16`, `N=64`; group `single_boolean_at_face_count` for cost vs target complexity. This is where O(N²) regressions show first.
- `boolean_tracking.rs`: the small fast suite CI runs for trend tracking (group `boolean`; IDs include `cut_cylinder_through_box`, `perforated_cut_36`). Keep it under a minute; it runs on every push to main.
- `compound_cut_perf.rs`: `compound_cut` vs sequential `boolean(Cut)`. Groups `compound_cut_cylinders` (grids of 4/16/36/64) and `compound_cut_honeycomb` (hex grids, IDs like `compound_rings=2_N=19`).
- `fuse_perf.rs`: `fuse_all` vs sequential left-fold. Group `fuse_balanced` has both `balanced_N={n}` and `sequential_N={n}` (N in 4/9/16/25); group `fuse_touching` has only `balanced_N={n}` (N in 4/9/16).

Cargo aliases (`.cargo/config.toml`):

```bash
cargo bench-fast   # bench -p brepkit-operations --bench cad_operations
cargo bench-full   # bench -p brepkit-operations
```

Output shape per benchmark:

```
sequential_cylinder_cuts/N=64
                        time:   [1.021 s 1.0324 s 1.0442 s]
                        change: [-2.1% +0.3% +2.8%] (p = 0.81 > 0.05)
```

The `change:` line appears once `target/criterion/` holds a prior run. That directory is your baseline mechanism: run before the fix, run after, criterion computes the delta.

### CI benchmark tracking

`.github/workflows/benchmark.yml` runs `cargo bench -p brepkit-operations --bench boolean_tracking -- --output-format bencher` and feeds github-action-benchmark with `alert-threshold: '200%'`, `comment-on-alert: true`, `fail-on-alert: false`. Shared runners are noisy, so it comments instead of failing. Consequence: a merged PR can regress perf 1.9x with zero CI signal. The deterministic guard that does hard-fail is the complexity test `scaling_perforated_cut_is_subquadratic` (rg: `fn scaling_` in `crates/operations/src/boolean/tests.rs`), which asserts a growth ratio, not wall clock. Add tests in that style for any scaling fix.

### Profiling examples (`crates/operations/examples/`)

- `profile_boolean.rs`: per-phase timing harness. Scenarios (from its header): `honeycomb`, `cylinders`, `fuse`, `large-honeycomb`, `scale` (scaling study), `xl` (200+ tool stress), `tess` (tessellation focus).

  ```bash
  cargo run --profile profiling --example profile_boolean -- <scenario>
  ```

  This is the preferred flamegraph payload for boolean work: it runs one workload continuously instead of criterion's sample loop, so the SVG is cleaner.
- `tess_profile.rs`: builds the 64-hole plate (100x100x10 box minus an 8x8 grid of r=2 cylinders), prints per-face surface-type/edge/inner-wire breakdown and arena sizes, then times `tessellate_solid(&topo, result, 0.1)` three times with vertex/triangle counts. Use it to see which surface types and face counts drive triangle counts before touching tessellation code.
- `debug_boolean.rs` and `approx_census.rs` also live there; the latter is analytic-degradation auditing, not perf.

### Cross-kernel comparison

```bash
./scripts/bench-compare.sh ~/Git/brepjs
```

Pipeline: (1) full native criterion run, log to `bench-results/criterion.log`; (2) `wasm-pack build crates/wasm --target nodejs --release`; (3) install the built package into brepjs with `--no-save`; (4) run `benchmarks/kernel-comparison.bench.test.ts` under vitest with `BENCH_OUTPUT_JSON=1`; (5) `npx tsx scripts/bench-report.ts` produces `bench-results/report.md` and `comparison.json`. That report is the only valid evidence for "faster than the reference kernel" claims. See the parity-benchmarking skill for interpreting it and for a lighter one-off head-to-head (wasm-pack build plus a vitest `resolve.alias` swap in brepjs, which avoids the npm install step).

## Bug classes in depth

### Class A: pairwise N-way accumulation is quadratic

Symptom: `for tool in tools { acc = boolean(acc, tool) }` where each iteration is slower than the last. Cause: the accumulator's face count grows every step, and every pairwise boolean pays full GFA arena/assemble cost proportional to it. Total cost O(N²) even when each individual call is optimal.

Fix shape, in order of preference:

1. Skip booleans entirely when operands are provably disjoint. `boolean()` already does this for Fuse: `solids_provably_disjoint` + `merge_disjoint_solids` in `crates/operations/src/boolean/mod.rs` (rg: `solids_provably_disjoint`). Two soundness traps if you extend it: disjoint requires a strictly positive gap (`b.min - a.max > margin`; touching is NOT disjoint, coincident faces need real boolean handling), and the accumulator can span many components, so test per-component AABBs, not one global box.
2. Batch entry points: `pub fn fuse_all(topo, compound)` in `crates/operations/src/compound_ops.rs` (bbox-partitions into overlapping groups, merges disjoint groups without booleans, tree-reduces within groups) and `pub fn compound_cut` in `crates/operations/src/boolean/mod.rs`. Both are exposed to wasm; `fuseAll` lives in `crates/wasm/src/bindings/booleans.rs`.
3. Balanced tree-reduce as a floor: fusing pairs of similar size keeps intermediate face counts logarithmically bounded. `fuse_perf.rs` measures exactly this against the left-fold.

Historically this class produced two-to-three orders of magnitude wins. Verify a fix with `fuse_perf` or `compound_cut_perf` (the `sequential_` vs `balanced_`/`compound_` IDs are the before/after built in), then guard it with a growth-ratio test.

### Class B: HashMap iteration order feeding downstream branching

Symptom: the identical operation on identical input varies enormously (100x to 500x observed historically) across process runs, or a criterion bench appears to hang. It is not a hang: `HashMap`'s per-process random seed changes iteration order, and when downstream logic branches on that order, some orderings steer the pipeline into a pathological path. Criterion's sampling then repeatedly draws the slow branch. The same mechanism also produces nondeterministic results, so this class is both a perf bug and a correctness bug.

Diagnostic recipe:

```rust
for i in 0..10 {
    let mut topo = Topology::new();
    let start = std::time::Instant::now();
    run_the_op(&mut topo);
    println!("iter {i}: {:?}", start.elapsed());
}
```

Fresh `Topology` each iteration (arena state must not carry over). Fixed code shows all iterations within roughly 2x of each other; a 10x+ spread confirms the class. Note this loop shares one process, so one seed; run the binary a few times if the spread hides.

Fix shape: sort, and usually dedup, any collection derived from HashMap iteration before it drives decisions. Live examples of the pattern: `neighbors.sort_unstable_by_key(|id| id.index())` in `crates/topology/src/adjacency.rs`; `nonmanifold.sort_by_key(|(eid, _)| eid.index())` and `neighbors.sort_unstable()` in `crates/operations/src/boolean/assembly.rs`. Sorting by arena index is the house convention.

Hunting for new instances: HashMaps still exist across the GFA pipeline, so treat any unsorted `.values()`, `.keys()`, `.iter()`, or `for (k, v) in map` whose output feeds ordering-sensitive logic as a suspect:

```bash
rg -n '\.values\(\)|\.keys\(\)|HashMap' crates/algo/src/builder crates/algo/src/pave_filler
```

Do not assume any specific file is currently buggy; several determinism fixes have landed (git history around #681, #683, #901, #907 if you want the archaeology). The pattern, not the file list, is durable.

### Class C: tessellation deflection dominates preview cost

`tessellate_solid(topo, solid, deflection)` in `crates/operations/src/tessellate/solid.rs` (rg: `pub fn tessellate_solid`); variants `tessellate_solid_with_tolerance`, `tessellate_solid_grouped_with_tolerance`, `tessellate_solid_for_boolean` live in the same file. Deflection is max chord deviation: on curved surfaces triangle count grows superlinearly as it shrinks. The bench suite encodes the sensitivity: `mesh box (tol=0.1)` is nearly free, `mesh sphere (tol=0.01)` is not, and `tessellate 64-hole plate` adds boolean-heavy topology.

Fix shape: preview paths use coarse deflection and cache the mesh; never re-tessellate at export tolerance for an interactive path. Drill down with `tess_profile` to see whether triangles come from many faces (topology problem, maybe a boolean left fragments) or few finely-meshed curved faces (deflection problem). Hard boundary: deflection is a rendering/export knob only. Introducing tessellation into core geometry or boolean classification to "speed things up" is forbidden; see the tessellation and analytic-preservation skills.

### Class D: error-variant fallthrough into a silent slow path

The largest single perf win in this codebase's history was not an algorithmic change. An intermediate Cut returned an unexpected `Err` variant, the multi-region handling matched only the expected variant, and the operation fell through to the mesh-boolean fallback (`crates/operations/src/mesh_boolean.rs`), which is orders of magnitude slower and lossy. The fix widened a match arm.

Lesson: a perf cliff can hide entirely inside error-handling plumbing. The flamegraph signature is unmistakable: a subsystem that should not be running (mesh boolean on analytic inputs) dominating the graph. When you see it, find the fallback's entry condition and trace why the primary path bailed; do not optimize the fallback. This overlaps with the boolean-debugging skill: the same fallthrough that costs time also degrades analytic geometry to mesh.

## Glossary

- **GFA**: the general-fuse-style boolean engine in `crates/algo` (`gfa.rs`): intersect everything, split faces, classify, reassemble. Every pairwise boolean pays its arena/assemble overhead.
- **PaveFiller**: GFA's intersection orchestrator (`crates/algo/src/pave_filler/`); runs interference phases pairwise by entity type.
- **FF phase**: face-face intersection (`pave_filler/phase_ff.rs`), typically the hottest GFA phase; produces section curves.
- **SD (same-domain)**: detection and merging of coincident overlapping faces from the two operands (`crates/algo/src/builder/same_domain.rs`); a classic source of order-sensitivity.
- **Mesh fallback**: `crates/operations/src/mesh_boolean.rs`, tessellation-based co-refinement used when the B-Rep path fails. Slow and lossy; appearing in a flamegraph on analytic inputs is itself a bug.
- **Deflection**: max chord deviation for tessellation, the third argument to `tessellate_solid`. Smaller means superlinearly more triangles on curved faces.
- **Arena / Topology**: the arena allocator owning all B-Rep entities (`crates/topology`). Booleans against a growing accumulator solid slow down as its face count grows.
- **Criterion baseline**: history in `target/criterion/` that makes re-runs print change-%; the before/after mechanism.
- **The reference kernel**: the incumbent C++ CAD kernel brepkit replaces. The perf bar is beating it, measured through `bench-compare.sh`.
