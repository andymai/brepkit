window.BENCHMARK_DATA = {
  "lastUpdate": 1782279104050,
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
      }
    ]
  }
}