window.BENCHMARK_DATA = {
  "lastUpdate": 1782282829715,
  "repoUrl": "https://github.com/andymai/brepkit",
  "entries": {
    "Boolean perf": [
      {
        "commit": {
          "author": {
            "email": "hi@andymai.com",
            "name": "Andy Aragon",
            "username": "andymai"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "06f0dda7f9eb574e6b1e30ffa3560a2bc0dfa1d8",
          "message": "ci: track broad boolean-perf regressions via benchmark workflow (#991)\n\n## Summary\n\nFollow-up to #990. That PR added a **deterministic complexity guard**\nthat fails CI if a boolean hot path regresses to O(N²) — sharp, but it\nonly watches the two paths it counts. This adds the complementary\n**broad absolute-perf tracking** so general slowdowns (anywhere in the\nboolean pipeline) are visible too.\n\n## What it does\n\n- A new, intentionally **small and fast** `boolean_tracking` criterion\nbench — box cut/fuse, cylinder-through-box cut, and a 36-hole perforated\npanel (the #987 case) — reduced sample count so the whole suite runs in\nwell under a minute.\n- Emitted in the libtest `bencher` format (`--output-format bencher`)\nand fed to\n[`benchmark-action/github-action-benchmark`](https://github.com/benchmark-action/github-action-benchmark)\n(pinned to v1.22.1).\n- **On `main`:** records a baseline and renders a trend chart on the\n`gh-pages` branch.\n- **On a PR:** compares against that baseline and comments only on a\nclear regression.\n\n## Deliberately conservative (won't add flaky failures)\n\nShared CI runners are noisy, so:\n- `alert-threshold: 200%` — only a clear ~2× regression alerts.\n- `fail-on-alert: false` — a regression **comments**, never fails the\nbuild.\n- `auto-push` only on `main`; PRs compare without writing.\n\nThe first `main` run bootstraps the `gh-pages` baseline; until then PRs\nsimply run the bench with nothing to compare.\n\n## Verification\n\n- Bench emits clean `bencher` lines on stdout (box cut/fuse ~1ms,\ncylinder cut ~0.4ms, perforated_cut_36 ~15ms).\n- `benchmark.yml` is valid YAML and interpolates only trusted `github.*`\ncontext + `secrets.GITHUB_TOKEN` (no untrusted-input injection surface).\n- clippy `--all-targets` and `cargo fmt` clean.",
          "timestamp": "2026-06-23T22:29:11-07:00",
          "tree_id": "4f46ff714ed6c7c4b67eb40c42aa817566683418",
          "url": "https://github.com/andymai/brepkit/commit/06f0dda7f9eb574e6b1e30ffa3560a2bc0dfa1d8"
        },
        "date": 1782279103677,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1418301,
            "range": "± 2640",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1517162,
            "range": "± 2455",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13115,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 590378,
            "range": "± 8699",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21612762,
            "range": "± 56911",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hi@andymai.com",
            "name": "Andy Aragon",
            "username": "andymai"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0c03188803e3c885e953275618b108119ffc20de",
          "message": "test(algo): extend boolean complexity guard to all five #990 hot paths (#992)\n\n## Summary\n\nFollow-up to #990. That PR fixed **five** independent O(N²) hot paths in\nthe boolean `Cut` pipeline, but the deterministic work-counter guard\n(`scaling_perforated_cut_is_subquadratic`) only instrumented **two** of\nthem — the pave-vertex probe and the same-domain polygon clip. The other\nthree were watched only by the noisy wall-clock tracking bench (#991),\nso a regression in them could silently reintroduce a quadratic without\ntripping the sharp guard.\n\nThis adds work-counters for the remaining three paths and extends the\nguard to assert sub-quadratic scaling on all five.\n\n## New counters (`algo::perf`)\n\nZero-cost `#[inline]` no-ops when `perf-counters` is off, so the\ninstrumented hot loops pay nothing in normal/release builds.\n\n| Counter | Path | Site |\n|---|---|---|\n| `ray_geom_builds` | classify sub-faces | `collect_face_geoms` (built\nonce per solid) |\n| `face_split_probes` | face-splitter scans | `find_splits_on_*`,\narrangement pairwise, `plane_internal_line_loops` |\n| `local_vertex_inserts` | `build_topology_face` vertex pool |\n`layered_vertex` materialization |\n\n`snapshot()` now returns a `PerfSnapshot` struct instead of a 2-tuple.\n\n## Verification — each counter catches its regression\n\nReverting each #990 fix in isolation makes the corresponding counter\nexplode (81→324 holes, a 4× input step):\n\n| Counter | fix in | fix reverted | bound |\n|---|---|---|---|\n| `ray_geom_builds` | **2→2** | **170→656** | absolute `< 64` |\n| `face_split_probes` | **4.1×** | **15.5×** | ratio `< 8.0` |\n| `local_vertex_inserts` | **4.0×** | **15.8×** | ratio `< 8.0` |\n\n`ray_geom_builds` regresses to O(N) (a *constant* becoming linear), so a\nratio bound would miss it — it uses an absolute bound. The other two\nregress to O(N²), where the scaling **ratio** is the sharp test. Counts\nare fully deterministic across runs, so the thresholds are tight with no\ntiming flakiness.\n\n## No residual quadratic found\n\nAll five paths measure linear (~4.0× for 4× input). The arrangement\npass's `for i { for j }` pairwise loop is structurally O(N²) but\nbbox-pruned and cheap; on disjoint holes it stays linear (work\npartitions into small per-hole arrangements), matching #990's\n\"diminishing returns\" call. Left as-is.\n\n## CI\n\nThe existing complexity-guard step (`cargo nextest run -p\nbrepkit-operations --features perf-counters -E 'test(scaling_)'`) runs\nthe extended guard — **no workflow change needed**.\n\n## Checks\n- Full workspace suite: **0 failures**\n- `clippy --all-targets -D warnings`: clean in **both** default and\n`perf-counters` configs\n- `cargo fmt`, `check-boundaries.sh`: clean",
          "timestamp": "2026-06-24T06:31:46Z",
          "tree_id": "360d00e8cb79e73bc78eb580281c21dfb0598444",
          "url": "https://github.com/andymai/brepkit/commit/0c03188803e3c885e953275618b108119ffc20de"
        },
        "date": 1782282828834,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1353094,
            "range": "± 5958",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1431370,
            "range": "± 1905",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12057,
            "range": "± 169",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 592480,
            "range": "± 1324",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21116334,
            "range": "± 353298",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}