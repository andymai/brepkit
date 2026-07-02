---
name: profiling
description: Use when investigating or fixing performance in brepkit: a benchmark got slower, an operation (boolean, fuse, tessellation) is unexpectedly slow or hangs, a criterion bench times out or shows wild variance, or a PR needs before/after perf numbers. Covers flamegraphs, the criterion bench suite, cross-kernel comparison, and the perf bug classes this codebase has actually had.
---

# Profiling and Performance Debugging

## The bar

brepkit must beat the reference kernel on performance, not merely pass tests. Perf regressions are release blockers. CI runs `boolean_tracking` on a shared runner and only comments on regressions over 200% (`.github/workflows/benchmark.yml`, `fail-on-alert: false`), so the automated gate is looser than the real bar. You are the gate: any PR touching a hot path pastes before/after criterion numbers in its body. Cross-kernel claims require `./scripts/bench-compare.sh ~/Git/brepjs` output, not native-only numbers (see the parity-benchmarking skill).

## Quick reference

```bash
cargo bench-fast                                          # kernel-comparison suite (cad_operations, ~2 min)
cargo bench-full                                          # all 5 bench files
cargo bench -p brepkit-operations --bench boolean_perf -- "N=64"   # isolate one bench by substring
cargo flamegraph --profile profiling --bench cad_operations -p brepkit-operations \
  -o /tmp/flamegraph.svg -- --bench "<filter>"            # flamegraph a specific criterion bench
cargo run --profile profiling --example profile_boolean -- honeycomb   # per-phase boolean timing
cargo run --profile profiling --example tess_profile      # tessellation drill-down (64-hole plate)
./scripts/bench-compare.sh ~/Git/brepjs                   # native + wasm vs the reference kernel
```

The `[profile.profiling]` block in the workspace `Cargo.toml` is release optimization plus debug symbols with `lto = false`: fast rebuilds, full symbol names in flamegraphs. Always profile with it, never with plain `release` (LTO mangles frames) or `dev` (measures nothing real).

Bench files (`crates/operations/benches/`), one line each:

| Bench | Covers |
|---|---|
| `cad_operations.rs` | Head-to-head suite (`fuse(box,box) x10`, `mesh sphere (tol=0.01)`, `tessellate 64-hole plate`, ...); paired with `brepjs/benchmarks/kernel-comparison.bench.test.ts` via the `NAME_MAP` table in `scripts/bench-report.ts`, so a new head-to-head bench must be added to that table or it will not appear in the comparison report |
| `boolean_perf.rs` | Boolean scaling: `sequential_cylinder_cuts/N={4,16,64}` and `single_boolean_at_face_count` |
| `boolean_tracking.rs` | Small fast suite CI tracks for trend (`boolean/cut_cylinder_through_box`, `boolean/perforated_cut_36`) |
| `compound_cut_perf.rs` | `compound_cut` vs sequential cuts: cylinder grids and honeycomb grids |
| `fuse_perf.rs` | `fuse_all` tree-reduce vs sequential left-fold (`fuse_balanced` has `balanced_N=` and `sequential_N=`; `fuse_touching` has only `balanced_N=`) |

## Method: the perf loop

1. **Baseline.** `cargo bench -p brepkit-operations --bench <file> -- "<filter>"`. Expect a line like `sequential_cylinder_cuts/N=64  time: [x.xx s x.xx s x.xx s]`. Save it verbatim. Criterion stores history in `target/criterion/`, so later runs print change-% automatically.
2. **Isolate.** Narrow the filter to one benchmark ID. If the bench seems to hang or shows wild run-to-run variance, do not fight criterion: write a plain timed loop that runs the op N times on fresh `Topology` instances and prints per-iteration wall clock. Huge iteration spread is itself the diagnosis (see bug class B).
3. **Flamegraph that exact workload.** Use the quick-reference command with the same filter, or for boolean work use `--example profile_boolean -- <scenario>` (scenarios: `honeycomb`, `cylinders`, `fuse`, `large-honeycomb`, `scale`, `xl`, `tess`; the example header documents them). Open the SVG in a browser and look for wide frames. Checkpoint: if the widest frames are in `mesh_boolean` on analytic inputs, stop, that is the bug (class D).
4. **Fix.** Vary one variable at a time. Match the symptom to a known class first (table below) before inventing a new theory.
5. **Re-run the same bench with the same filter.** Criterion prints the delta against the stored baseline. Checkpoint: expect `change: [-XX% ...] Performance has improved`. If the number moved less than the flamegraph suggested, the wide frame was not on the measured path; re-check the filter.
6. **PR body.** Paste both criterion lines verbatim. For scaling fixes, also add a deterministic complexity-guard test so the win cannot regress on noisy runners; pattern: `scaling_perforated_cut_is_subquadratic` in `crates/operations/src/boolean/tests.rs`. Historical absolute times are machine-specific; never write them into tests as thresholds, assert growth ratios instead.

## Known perf bug classes

| Symptom | Likely cause | Fix shape | Detail |
|---|---|---|---|
| N-tool boolean scales worse than linear; each cut/fuse slower than the last | Pairwise accumulation: `acc = fuse(acc, next)` pays growing cost every step in GFA (the general-fuse boolean engine in `crates/algo`), O(N²) total | Batch it: `compound_ops::fuse_all` (bbox-partition + disjoint merge + tree-reduce) or `boolean::compound_cut`; `boolean()` already fast-paths provably disjoint Fuse | reference.md, class A |
| Same op varies 100x+ between identical runs; criterion "hangs"; results differ across runs | HashMap iteration order feeding downstream branching; the random seed sometimes samples a pathological path | Sort (and dedup) any collection derived from HashMap iteration before it drives decisions | reference.md, class B |
| Preview/mesh path slow; triangle counts explode | Deflection too fine for the use; re-tessellating at export tolerance | Coarse deflection for preview; drill down with `tess_profile` | reference.md, class C |
| Flamegraph dominated by `mesh_boolean` on analytic inputs | B-Rep path bailed via error-variant fallthrough into the mesh fallback | Fix the error handling, not the algorithm; the fallthrough is the bug | reference.md, class D |

## Anti-patterns: what NOT to conclude

- A hung or 100x-variance bench does NOT mean "the algorithm is slow". Rule out nondeterminism (class B) first with the fresh-Topology timed loop.
- A green CI benchmark run does NOT mean no regression. The alert threshold is 200% and it only comments. Compare criterion output yourself.
- Native criterion wins do NOT prove "beats the reference kernel". Only `bench-compare.sh` output does; wasm behaves differently.
- Do NOT profile with `--release` (LTO destroys symbols) or trust `dev`-profile timings at all.
- Do NOT "fix" a scaling problem by making each pairwise call faster. If cost grows with accumulator size, the fix is batching or disjointness detection, not micro-optimization.
- Do NOT paste historical numbers from old PRs as expected timings. Re-measure on the current machine and commit.
- Do NOT speed up tessellation by loosening deflection inside core geometry or booleans. Mesh approximation in core paths is a fidelity bug (see the tessellation and analytic-preservation skills).

## Deep detail

- [reference.md](reference.md): full command catalog, bug classes A-D with verified symbols and rg patterns, `bench-compare.sh` pipeline, profiling-example internals, glossary.
- Sibling skills: parity-benchmarking (cross-kernel harness), tessellation (deflection semantics), boolean-debugging (when the slow path is also wrong), debugging-doctrine (bisection discipline), pr-workflow (getting the numbers merged).
