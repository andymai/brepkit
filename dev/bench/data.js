window.BENCHMARK_DATA = {
  "lastUpdate": 1783493037981,
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
          "id": "de57aa59eb81bf6ce87d1586ccca9f42e1f71faf",
          "message": "test(algo): address review feedback on boolean complexity guard (#993)\n\n## Summary\n\nFollow-up to #992, addressing the Copilot and Greptile review comments\non that (auto-merged) PR.\n\n## Changes\n\n**1. Harden the scaling-ratio helper** (Greptile P2 + Copilot)\nThe `ratio` helper returned `0.0` for a `0→nonzero` ratio, which passes\nthe `< 8.0` bound — so if a fixture or instrumentation change stopped\nexercising a path at the smaller size (`g=9`), the scaling guard for\nthat counter would silently switch off. It now asserts the baseline is\nexercised:\n\n```rust\nlet ratio = |a: u64, b: u64| {\n    assert!(a > 0, \"scaling-ratio baseline counter was not exercised at g=9\");\n    b as f64 / a as f64\n};\n```\n\n(Copilot also flagged `f64::from(bool)` as non-compiling — that's been\nvalid since Rust 1.68 and CI compiled/ran it on the 1.88 MSRV, but the\nmasking concern was real and is now fixed by removing it.)\n\n**2. Fix two docstrings** (Copilot)\n`bump_face_split_probe` and `PerfSnapshot::face_split_probes` described\nthe counter as only \"grid-query candidate endpoints\", but it is also\nbumped for arrangement chord pairs that survive the bbox broad-phase in\n`arrangement_regions_from_inputs`. Both docs now reflect that.\n\n## Scope\n\nDoc-only and test-only changes; the counters remain\n`perf-counters`-gated, so shipping builds are unaffected.\n\n## Checks\n- `scaling_perforated_cut_is_subquadratic` still passes (4.1× / 4.0×,\nbaselines nonzero)\n- `clippy --all-targets -D warnings` clean in default and\n`perf-counters` configs\n- `cargo fmt` clean",
          "timestamp": "2026-06-24T14:13:26Z",
          "tree_id": "b4ccbdc4db4f8d4cd3838f287c0080f5d046f40b",
          "url": "https://github.com/andymai/brepkit/commit/de57aa59eb81bf6ce87d1586ccca9f42e1f71faf"
        },
        "date": 1782310529848,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1420567,
            "range": "± 2888",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1510830,
            "range": "± 2429",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13116,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 587858,
            "range": "± 1395",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21582575,
            "range": "± 69035",
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
          "id": "7b7efde1bb5a54e58ec78d08284f21a243d097af",
          "message": "chore(observability): probe analytic-fallback paths + prune dead constants (#994)\n\n## What\n\nTwo cleanups from an audit of every brepkit operation that can degrade\nfrom an exact analytic B-Rep to an approximation.\n\n### 1. Observability probes — `chore(observability)`\n\nPermanent `log::debug!(target: \"brepkit_approx\", …)` probes at all 7\napproximation branches, so any run can report exactly where an op left\nthe analytic path:\n\n| Path | Site |\n|------|------|\n| Boolean → mesh (co-refinement) fallback — *the only path that loses\nanalytic surface types* | `operations/boolean/mod.rs` |\n| Fillet analytic → Newton-Raphson walker | `blend/fillet_builder.rs` |\n| Chamfer analytic → `UnsupportedSurface` (no v1 walker) |\n`blend/chamfer_builder.rs` |\n| Offset NURBS face → 16×16 sampled-NURBS refit | `offset/offset.rs` |\n| Offset trim → grid-sampling | `operations/offset_trim.rs` |\n| Offset face → raw untrimmed surface | `operations/offset_face.rs` |\n| Rolling-ball fillet → flat planar corner |\n`operations/fillet/rolling_ball.rs` |\n\nNew `crates/operations/examples/approx_census.rs` installs an in-process\nlogger that captures the probes and prints **path + wall-clock + face\ncount** per op:\n\n```\ncargo run --release --example approx_census -p brepkit-operations\n```\n\nSample output (overlapping primitives): boolean stays exact and\nsub-millisecond on `box−cyl` and coaxial `cyl∩cyl`, but `box∩sphere`\n(956 faces, 187 ms), `sphere−cyl` (1392 faces, 357 ms), and `torus−box`\n(1733 faces, 204 ms) drop to the mesh fallback — roughly 100–1000×\nslower with a 10–200× face explosion. Offset/fillet/chamfer stay exact\non every analytic primitive.\n\n`offset` gains a `log` dependency (used by the new probe).\n\n### 2. Prune dead constants — `chore(boolean)`\n\n`MESH_BOOLEAN_PER_SOLID_THRESHOLD` / `MESH_BOOLEAN_FACE_THRESHOLD` were\nunconsumed leftovers from the pre-GFA \"chord-based tessellated\" boolean\npipeline (the `collect_face_data` face-count pre-gate). That pipeline is\ngone — the engine is GFA-primary with a mesh fallback decided by result\nvalidation, not a face-count threshold. Their doc comments held the last\nreference to the removed `collect_face_data`.\n\n## Why\n\nPure observability + dead-code cleanup, no behavior change: the probes\nare `debug`-level (silent unless a logger opts in) and the removed\nconstants had zero references.\n\n## Verification\n\n- `cargo test --workspace` green — probes regress nothing.\n- `cargo clippy --all-targets` clean under `-D warnings`.\n- The example empirically captures all three primary fallback families\n(boolean→mesh, fillet→walker, offset→sampled-NURBS).",
          "timestamp": "2026-06-24T16:34:06-07:00",
          "tree_id": "18d410814a2b81a80096707b768a52c52446b778",
          "url": "https://github.com/andymai/brepkit/commit/7b7efde1bb5a54e58ec78d08284f21a243d097af"
        },
        "date": 1782344179563,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1344459,
            "range": "± 2248",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1435414,
            "range": "± 2133",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11990,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 592366,
            "range": "± 1657",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20553038,
            "range": "± 68626",
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
          "id": "8ecca9f23cfbe36c39ae56454bced46b31a40968",
          "message": "chore(observability): trigger all 7 fallback paths in census example (#995)\n\n## What\n\nExtends `approx_census.rs` so it empirically triggers **all 7**\n`brepkit_approx` approximation probes, not just the 3 the primitive\nmatrix reached.\n\nThe primitive matrix already fired:\n1. boolean → mesh fallback\n2. fillet → Newton-Raphson walker\n3. offset NURBS face → sampled-NURBS refit\n\nA new `remaining_paths()` section constructs the inputs the primitive\nmatrix can't reach to fire the other four:\n\n| Probe | Trigger added |\n|-------|---------------|\n| chamfer → `UnsupportedSurface` (v1 has no walker) | `chamfer_v2` on a\n**torus** (analytic declines Torus pairs) |\n| offset trim → grid-sampling | `offset_face` on a **NURBS loft face**,\ngentle offset (no self-intersection → SSI finds nothing) |\n| offset face → raw surface | `offset_face` on a **sharply-waisted NURBS\nloft**, large inward offset (self-intersection > 50% → trim errors) |\n| rolling-ball fillet → planar corner | `fillet_rolling_ball` on a\n**square pyramid** (4-valence apex → non-triangular corner) |\n\n## Verification\n\n`cargo run --release --example approx_census -p brepkit-operations` now\nshows every probe firing. De-duped probe families:\n\n```\nboolean: GFA → mesh\nfillet: → walker\noffset: NURBS face → sampled-NURBS\nchamfer: → UnsupportedSurface\noffset_trim: → grid-sampling\noffset_face: → raw-offset-surface\nfillet(rolling-ball): → planar corner\n```\n\nThe `offset_face` inward cases also surface the genuine error detail\n(`SSI trim failed: ...covers 100%...`), confirming the raw-surface\nfallback path.\n\n- `cargo clippy -p brepkit-operations --example approx_census\n--all-features -- -D warnings` clean.\n- Example-only change; no engine/library code touched.",
          "timestamp": "2026-06-24T17:16:45-07:00",
          "tree_id": "a301b2724c142cc78d2b0531a8e1ca51cb5b5f29",
          "url": "https://github.com/andymai/brepkit/commit/8ecca9f23cfbe36c39ae56454bced46b31a40968"
        },
        "date": 1782346726235,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1424526,
            "range": "± 1989",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1521486,
            "range": "± 9035",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13196,
            "range": "± 125",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 595738,
            "range": "± 1824",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21719304,
            "range": "± 171542",
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
          "id": "35a98cf744df7efad4a499be1d1f99e957c17199",
          "message": "chore(observability): report faces=1 for single-face offset_face in census (#996)\n\nAddresses a Copilot review nit on #995: the census printed `faces=0` for\n`offset_face` success rows, which reads like an empty result.\n`offset_face` returns a single `FaceId` (not a solid), so report\n`faces=1` instead.\n\nExample-only, one-line change. Verified: the `offset_face` rows now\nprint `faces=1` on success; `cargo fmt`/`clippy` clean via pre-commit.",
          "timestamp": "2026-06-25T00:31:12Z",
          "tree_id": "da4e6351e7794208b82cf136225cd1fdb1bd1818",
          "url": "https://github.com/andymai/brepkit/commit/35a98cf744df7efad4a499be1d1f99e957c17199"
        },
        "date": 1782347587984,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1350597,
            "range": "± 1989",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1436577,
            "range": "± 5761",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11890,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 607722,
            "range": "± 10581",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20740358,
            "range": "± 147393",
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
          "id": "c53af2f637bf7d93c1c3039157294547c93cf41a",
          "message": "fix(algo): keep cylinder slot-cut analytic (closed-circle section AABB) (#997)\n\n## What\n\nFixes the first of the boolean→mesh fallbacks surfaced by the\napproximation census: **a box cutting a slot into a cylinder's side**\nnow stays analytic and watertight instead of dropping to a 62-facet\nmesh.\n\n## Root cause\n\nA box has two faces **perpendicular to the cylinder axis** (the slot's\ntop/bottom walls). Each intersects the cylinder lateral surface in a\n**closed circle**; only a small arc of that circle lies within the box\nface, so phase_ff splits the closed circle at its boundary crossings and\nemits the in-face arc (`emit_split_circle_arcs`).\n\nThat emit step rejects each candidate arc whose midpoint falls outside\n*either* face's AABB. The face-AABB helper built its bounds from edge\n**endpoint vertices only** — but a cylinder lateral face's circular\nedges are *closed* (start == end at the seam vertex), so its AABB\ncollapsed to a thin line at the seam. Every slot-arc midpoint then\ntested \"outside\" the cylinder AABB → **all arcs dropped → the\nperpendicular box faces were never created → 8 free edges → mesh\nfallback.**\n\n(The box faces *parallel* to the axis intersect in line segments via a\ndifferent, correct path, which is why only the perpendicular walls were\nlost.)\n\n## Fix\n\n`emit_split_circle_arcs`'s face-AABB now **samples along each edge\ncurve** (8 points), exactly as the engine's primary `compute_face_bbox`\nalready does, so a closed circular edge contributes its full radial\nextent instead of just the seam point. One-function change in\n`phase_ff.rs`; the sphere-hemisphere surface-union path is untouched.\n\n## Verification\n\n- New regression test\n`cut_cylinder_by_box_slot_perpendicular_walls_is_watertight`: asserts\nthe result is closed-manifold, free-edge-free, keeps the analytic\ncylinder face, is compact (<20 faces), **and** — via the robust ray-cast\nclassifier — that a point in the slot is `Outside` and the cylinder body\nis `Inside` (the cut geometry is correct, not just topologically\nclosed).\n- Census: `cyl − box (slot)` went from **62 faces / mesh fallback** to\n**8 faces / exact analytic / 0.71 ms**. No other census case changed\n(box∩sphere, sphere−cyl, cyl∪cyl still fall back — separate root\ncauses).\n- `cargo clippy --all-targets` clean; full `cargo test --workspace`\ngreen.\n\n## Note\n\nThe tessellation-based volume measure reads ~+1.4% high on the resulting\narc-edged notched cylinder (a known, separate limitation — it diverges\nupward with finer deflection), so the test verifies the cut\ngeometrically via classification rather than by volume.",
          "timestamp": "2026-06-25T01:35:41Z",
          "tree_id": "8ba649ec118e169ada8dd2f04b8a0ff71906bea8",
          "url": "https://github.com/andymai/brepkit/commit/c53af2f637bf7d93c1c3039157294547c93cf41a"
        },
        "date": 1782351479207,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1343142,
            "range": "± 2291",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1431030,
            "range": "± 1440",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12058,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 594777,
            "range": "± 2596",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20819649,
            "range": "± 62034",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4bd07d0a22bed527e4ac25ffbd9d966441c56a92",
          "message": "chore(main): release 2.120.1 (#998)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.1](https://github.com/andymai/brepkit/compare/v2.120.0...v2.120.1)\n(2026-06-25)\n\n\n### Bug Fixes\n\n* **algo:** keep cylinder slot-cut analytic (closed-circle section AABB)\n([#997](https://github.com/andymai/brepkit/issues/997))\n([c53af2f](https://github.com/andymai/brepkit/commit/c53af2f637bf7d93c1c3039157294547c93cf41a))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-25T01:40:46Z",
          "tree_id": "fa5b19132ce76f5df9de7221a56790700d3df527",
          "url": "https://github.com/andymai/brepkit/commit/4bd07d0a22bed527e4ac25ffbd9d966441c56a92"
        },
        "date": 1782351764238,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1337156,
            "range": "± 1803",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1426627,
            "range": "± 7887",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11909,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 595623,
            "range": "± 2277",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20492468,
            "range": "± 115858",
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
          "id": "6327ebe977f9d7a12ef0b503422bca20a085b811",
          "message": "fix(offset): assemble torus offsets analytically (doubly-periodic seam wire) (#999)\n\n## What\n\nFixes the torus-offset failure the approximation census surfaced:\n`offset(torus)` errored with **\"no faces could be assembled for the\noffset solid\"**. Offsetting a torus now produces a clean analytic torus.\n\n## Root cause\n\nA torus face is doubly-periodic — it has a *fundamental-polygon*\nboundary wire (`a · b · a⁻¹ · b⁻¹`) built from **two degenerate `v0→v0`\nseam `Line` edges** and a single vertex (see `make_torus`). The offset\nwire-builder (`loops.rs::build_loops_for_face`) only knows three\nstrategies — circle+seam (cylinder/cone/sphere), direct vertex chaining,\nand line-line corner intersection — none of which can reconstruct a wire\nfrom degenerate seam lines. So the torus face got **empty wires**, the\nassembler skipped it, and with no faces left it errored out.\n\nCylinder/cone/sphere offsets already worked because their lateral faces\nuse Circle edges (handled by the circle+seam strategy); the torus is the\nonly primitive with a doubly-periodic seam.\n\n## Fix\n\nThe offset of a torus is a **concentric torus** (same center/major/axis,\nminor ± distance) with the identical seam structure. So\n`build_loops_for_face` now detects a torus offset face and rebuilds its\nfundamental-polygon wire directly from the offset `ToroidalSurface` —\none seam vertex at `evaluate(0,0)`, two degenerate seam edges, wire `a ·\nb · a⁻¹ · b⁻¹` — mirroring `make_torus`. ~25 lines in `loops.rs`, no\nchange to the generic strategies.\n\n## Verification\n\n- New regression test `offset_torus_stays_analytic` (outward, inward,\nlarger): asserts the result is a single analytic `Torus` face.\n- Census: `offset torus` went from **error** to a **1-face analytic\ntorus**.\n- `cargo clippy --all-targets` clean; full `cargo test --workspace`\ngreen.",
          "timestamp": "2026-06-25T02:13:20Z",
          "tree_id": "632ce2e0f0d06608f705abde8f1321e5c6d16bf8",
          "url": "https://github.com/andymai/brepkit/commit/6327ebe977f9d7a12ef0b503422bca20a085b811"
        },
        "date": 1782353709446,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1090236,
            "range": "± 2148",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1166668,
            "range": "± 2853",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10317,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 457056,
            "range": "± 920",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 16839137,
            "range": "± 269082",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7628eda3123fb59a8f192ccd10a8ebde780d2c70",
          "message": "chore(main): release 2.120.2 (#1000)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.2](https://github.com/andymai/brepkit/compare/v2.120.1...v2.120.2)\n(2026-06-25)\n\n\n### Bug Fixes\n\n* **offset:** assemble torus offsets analytically (doubly-periodic seam\nwire) ([#999](https://github.com/andymai/brepkit/issues/999))\n([6327ebe](https://github.com/andymai/brepkit/commit/6327ebe977f9d7a12ef0b503422bca20a085b811))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-25T02:18:25Z",
          "tree_id": "261d64a3a581e1377efec972e60871cc4bdb46b9",
          "url": "https://github.com/andymai/brepkit/commit/7628eda3123fb59a8f192ccd10a8ebde780d2c70"
        },
        "date": 1782354025931,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1337699,
            "range": "± 2059",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1423700,
            "range": "± 3119",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11889,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 590605,
            "range": "± 3452",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21044268,
            "range": "± 241846",
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
          "id": "2a8d97dc1e2fae79c5960b135dc41657c9ec1d67",
          "message": "fix(offset): restrict torus-wire rebuild to full untrimmed torus faces (#1001)\n\n## What\n\nFollow-up to #999 addressing a Copilot review finding: the torus-wire\nrebuild fired for **any** toroidal offset face, so a **trimmed torus\npatch** (e.g. a fillet/blend's torus face, which carries real\nboundary/intersection edges) would wrongly get the full\nfundamental-polygon seam wire instead of using its actual edges.\n\n## Fix\n\nGate the rebuild on the face having **no real (non-degenerate) edge**. A\nfull doubly-periodic torus offset face has only degenerate `v0→v0` seam\nedges (which the generic strategies can't use); a trimmed torus patch\ncarries real edges. So the rebuild now fires only for the genuine\nfull-torus case, and trimmed patches flow through the normal\ncircle/seam, chaining, and line-intersection strategies as before.\n\nAlso moved the check to after `face_edges` collection (it now needs them\nto compute `has_real_edge`).\n\n## Verification\n\n- `offset_torus_stays_analytic` still passes (the full untrimmed torus\nhas no real edges → still rebuilt → 1 analytic torus face).\n- `cargo clippy --all-targets` clean; full `cargo test --workspace`\ngreen.",
          "timestamp": "2026-06-25T02:23:06Z",
          "tree_id": "d97d17baec5533aa4bd48ddc97c02bc7564d9857",
          "url": "https://github.com/andymai/brepkit/commit/2a8d97dc1e2fae79c5960b135dc41657c9ec1d67"
        },
        "date": 1782354308903,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1337678,
            "range": "± 7850",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1427950,
            "range": "± 1677",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11938,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 592196,
            "range": "± 3235",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20574217,
            "range": "± 42129",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4d73734d56b2ea9dc0da86dc33e419d4409373f1",
          "message": "chore(main): release 2.120.3 (#1002)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.3](https://github.com/andymai/brepkit/compare/v2.120.2...v2.120.3)\n(2026-06-25)\n\n\n### Bug Fixes\n\n* **offset:** restrict torus-wire rebuild to full untrimmed torus faces\n([#1001](https://github.com/andymai/brepkit/issues/1001))\n([2a8d97d](https://github.com/andymai/brepkit/commit/2a8d97dc1e2fae79c5960b135dc41657c9ec1d67))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-25T02:28:04Z",
          "tree_id": "1abe83a5c7d543f49ee9d33530bf532e3cce67e0",
          "url": "https://github.com/andymai/brepkit/commit/4d73734d56b2ea9dc0da86dc33e419d4409373f1"
        },
        "date": 1782354604973,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1342123,
            "range": "± 24597",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1431243,
            "range": "± 3476",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12109,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 597739,
            "range": "± 1511",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20946477,
            "range": "± 380588",
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
          "id": "e034ed0013a8c01c779647b9a7f9b690e243a7ca",
          "message": "fix(algo): bound sphere/torus faces by surface extent in boolean broad-phase (#1003)\n\n## What\n\nMakes the boolean **broad-phase face AABB sound for sphere and torus\nfaces**.\n\n`compute_face_bbox` (in `phase_ff.rs`) derived each face's AABB from its\nboundary edges alone. A curved analytic face bulges *between* its\nboundaries,\nso the box collapsed:\n\n- a **sphere hemisphere** flattened to its equatorial disk (`z ∈ [0,\n0]`), and\n- a **full torus** — whose entire boundary is two degenerate seam-point\n`Line(v0, v0)` edges — collapsed to a single **point** `(R, 0, 0)`.\n\nThe FF broad-phase then wrongly rejected genuinely intersecting face\npairs (and\nthe 16-sample section filter dropped the curve), forcing every boolean\non those\nprimitives down the mesh co-refinement fallback (100–1000× slower,\nanalytic\nsurface types lost).\n\n## How\n\n- Add closed-form `aabb()` to `SphericalSurface` (`center ± radius`) and\n`ToroidalSurface` (ring + tube extent, orientation-aware) in\n`brepkit-math`,\n  with unit tests.\n- Union it into the face bbox for `Sphere`/`Torus` faces in\n`compute_face_bbox`.\n\nThe bbox only ever **widens**, so it can add broad-phase candidates but\nnever\ndrop a needed pair — the precise in-both restriction downstream still\ntrims\nexactly. Cylinder, cone, and plane boundaries already bound their faces\nand are\nuntouched. Same class as the cylinder seam-collapse fix in #997.\n\n## Effect (raw GFA, measured)\n\nThis is a **foundational** fix in a multi-PR campaign to eliminate the\nfour\nremaining boolean→mesh fallbacks (`box∩sphere`, `sphere−cyl`, `cyl∪cyl`\nperp,\n`torus−box`). It advances all three curved cases but closes none on its\nown —\nthe analytic results land in the follow-up split/assembly PRs:\n\n| case | before | after |\n|---|---|---|\n| `torus − box` | hard error: *\"no faces selected\"* (pairs rejected at\nbroad-phase) | reaches shell assembly (*\"all shells classified as\nholes\"*) |\n| `sphere − cyl` | cyl×sphere section dropped at broad-phase | sections\nsurvive (sphere split/keep is the follow-up) |\n| `box ∩ sphere` | box-cap sections dropped | sphere face survives the\nbroad-phase |\n\n## Test\n\n- 4 new unit tests for `SphericalSurface::aabb` /\n`ToroidalSurface::aabb`\n  (origin, off-center, canonical, and x-axis-oriented torus).\n- Full workspace suite green: **2444 passed, 9 skipped**, no regressions\n(incl. all gridfinity in-mem fixtures and\n`intersect_box_sphere_succeeds`).",
          "timestamp": "2026-06-24T21:37:46-07:00",
          "tree_id": "021e6ef53eb0bb410b7c354cb86485194985eb78",
          "url": "https://github.com/andymai/brepkit/commit/e034ed0013a8c01c779647b9a7f9b690e243a7ca"
        },
        "date": 1782362382787,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1321585,
            "range": "± 33010",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1399592,
            "range": "± 32674",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11977,
            "range": "± 105",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 590833,
            "range": "± 10784",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21087655,
            "range": "± 424014",
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
          "id": "78887da7756da191be667986daad745ec4a16372",
          "message": "fix(operations): assemble and render sphere−cyl Cut analytically (#1005)\n\n## What\n\nMakes `Cut(sphere, through-cylinder)` — and the bored-quadric family —\nproduce an **exact analytic, watertight, correctly-rendered** result\ninstead of the mesh fallback. Census: `sphere − cyl` **484ms / 1392\nplanar faces → 0.84ms / 3 analytic faces**.\n\nSecond PR in the campaign to close the four remaining boolean→mesh\nfallbacks (after #1003, the surface-aware face-AABB keystone).\n\n## The boolean fix (the sphere was being dropped)\n\nGFA returned 3 cyl faces with the sphere entirely gone. Root-caused to\n**five coupled gaps**, each unblocking the next:\n\n1. **math** — no exact `(Sphere, Cylinder)` intersection arm: the\ngeneric marcher returned `NurbsCurve`, so the closed-circle FF split\nnever fired. Added `exact_sphere_cylinder` (coaxial `z = ±√(R²−r²)`\ncircles).\n2. **phase_ff** — emit the section as `EdgeCurve::Circle`.\n3. **face_splitter** — route an all-closed-circle-section hemisphere to\n`split_face_with_internal_loops` (cap + band-with-hole) instead of\nunsplitting; aim the sphere-band interior point toward the hole's\nlatitude.\n4. **same_domain** — don't fuse two curved same-surface faces that share\nan outer-wire edge set but cover **distinct regions** (the two\nhemisphere bands of a bored sphere share the equator polygon) —\ndiscriminate by interior sample.\n5. **measure/check** — the annular spherical band over-integrates the\nremoved polar cap; route bored-quadric volume to the orientation-aware,\nhole-clipped analytic integrator.\n\nResult: 2 spherical bands-with-hole + 1 inner cylinder wall, watertight,\nvolume = `V_sphere − V_bore`.\n\n## The tessellation fix (so it renders correctly)\n\nA sphere/torus band between two constant-`v` latitude circles is\ndegenerate in UV (each latitude projects to a zero-area segment), so the\nCDT path filled the removed polar cap (mesh area 646 vs 587.7, tunnel\nmouth skinned over, not watertight). Added\n`tessellate_latitude_band_shared` — the curved analogue of the\ncylinder/cone band path: reuses the shared rim vertices at the two\nlatitudes for watertight stitching, and inserts intermediate latitude\nrows for the surface curvature. Gated conservatively to full-revolution\nlatitude bands; every other sphere/torus face takes the unchanged path.\n**Mesh area 646 → 586 (→587.9 at fine deflection), watertight, tunnel\nmouth open.**\n\n## Verification\n\n- Raw GFA: 3 analytic faces (2 sphere bands + cyl wall), free edges 0,\nmanifold.\n- Census: `sphere − cyl 0.84ms faces=3 exact analytic` (was 484ms/1392\nFALLBACK); the other 8 boolean cases unchanged (no regression).\n- Volume = **587.671** = `V_sphere − V_bore`.\n- Tessellation: watertight at deflections 0.5→0.01, area converges to\nthe analytic 587.7.\n- Full workspace suite: **2454 passed, 9 skipped**; clippy clean; fmt\nclean; layer boundaries valid.\n- Regression fixtures:\n`cut_sphere_by_through_cylinder_is_analytic_watertight` (boolean) +\n`bored_sphere_band_area_and_watertight` (tessellation).\n\n## Known tech debt (not introduced here, flagged for follow-up)\n\n`solid_volume` for a holed sphere routes to\n`analytic_sphere_signed_volume`, a **pre-existing** analytic function\nwhose cap-vs-band logic is wrong for bored spheres (it caps only one of\nthe two bands). This PR routes bored-quadric volume around it via the\nhole-clipped Gauss integrator (the `analytic_faces_solid_volume` fast\npath). Fixing `analytic_sphere_signed_volume` directly is a separate,\ndeeper cleanup (three volume paths converge there) and is left for a\ndedicated PR.\n\n## Does not touch\n\n`box ∩ sphere` (different root — disjoint great-circle arcs), `cyl ∪\ncyl`, `torus − box` remain fallback; their analytic splits are later\nPRs. The curved-band tessellation here will be reused by `torus − box`.",
          "timestamp": "2026-06-24T23:51:04-07:00",
          "tree_id": "e36c992ec3c1f7e376836ea0f9a7db8c06864e86",
          "url": "https://github.com/andymai/brepkit/commit/78887da7756da191be667986daad745ec4a16372"
        },
        "date": 1782370386850,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1348220,
            "range": "± 4293",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1436089,
            "range": "± 2520",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11832,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 593183,
            "range": "± 1797",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20778966,
            "range": "± 35345",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8ed71a79743dc4b78b8c4937de03fe19aed2d9c3",
          "message": "chore(main): release 2.120.4 (#1004)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.4](https://github.com/andymai/brepkit/compare/v2.120.3...v2.120.4)\n(2026-06-25)\n\n\n### Bug Fixes\n\n* **algo:** bound sphere/torus faces by surface extent in boolean\nbroad-phase ([#1003](https://github.com/andymai/brepkit/issues/1003))\n([e034ed0](https://github.com/andymai/brepkit/commit/e034ed0013a8c01c779647b9a7f9b690e243a7ca))\n* **operations:** assemble and render sphere−cyl Cut analytically\n([#1005](https://github.com/andymai/brepkit/issues/1005))\n([78887da](https://github.com/andymai/brepkit/commit/78887da7756da191be667986daad745ec4a16372))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-25T06:55:55Z",
          "tree_id": "42fa6be8854580b71f80c7721d538476ed87c42f",
          "url": "https://github.com/andymai/brepkit/commit/8ed71a79743dc4b78b8c4937de03fe19aed2d9c3"
        },
        "date": 1782370655267,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 813330,
            "range": "± 2363",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 876367,
            "range": "± 6183",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 7624,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 324244,
            "range": "± 1365",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 12824645,
            "range": "± 22428",
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
          "id": "6b4e781988f377a3decc5b5c441f95a955bd13d7",
          "message": "fix(operations): close box∩sphere boolean analytically (seam split + collar render/volume) (#1006)\n\n## What\n\nMakes `Intersect(box, sphere)` produce an **exact analytic, watertight,\ncorrectly-rendered, correctly-measured** result instead of a 956-face\nmesh fallback. Census: **190ms / 956 mesh → 1.94ms / 8 analytic faces**\n(6 plane disks + 2 spherical collar patches).\n\nThird PR in the campaign to close the four remaining boolean→mesh\nfallbacks (after #1003 keystone AABB, #1005 sphere−cyl). This is the\nhardest case — the centered sphere's collars are full-longitude-wrap\nbands with a *scalloped* floor.\n\n## The fix — four coupled layers\n\n**1. Section-circle split (`algo/phase_ff::emit_split_circle_arcs`)** —\nsplit arcs strictly *below* π. A span of exactly π is a diametric\nsemicircle whose two halves share both endpoints, so the assembler's\nendpoint-keyed `merge_duplicate_edges` collapsed the distinct\nnorth/south arcs into one edge (failing the Euler gate). Forcing a\nmidpoint vertex for any span ≥ π gives the two arcs distinct endpoints —\n**no merge-key change**, so the reversed-vertex merge of\ngenuinely-shared arcs is untouched (this is what makes it\nregression-safe where a merge-key discriminator was not).\n\n**2. Seam crossings (`algo/phase_ff::sphere_seam_plane_crossings`)** — a\nbox face's great-circle section must cross the sphere's *faceted*\nequator to split, but testing against the inscribed chords missed the\ncrossings by the polygon sagitta. Fit the seam plane (Newell) and solve\ncircle∩plane analytically — facet-independent.\n\n**3. Arrangement walk\n(`algo/face_splitter::split_noseam_by_arrangement`)** — the disjoint\ngreat-circle arcs share no endpoints, so the generic wire builder (which\nU-turns on the degenerate v=0 seam confluence) can't chain them. A\ndedicated DCEL arrangement walk assembles the box-face disks + the two\ncollar patches, selecting the collar by longitude winding.\n\n**4. Collar render + volume:**\n- **Render** (`operations/tessellate`): generalize the revolution-band\nmesher to a *varying-v* collar — outer ring = scalloped\ngreat-circle/seam floor, inner ring = latitude-cap hole — with\ncolumn-aligned curvature rows and a single whole-run orientation (the\nper-triangle normal flip is unstable on the thin stitch triangles).\nWatertight: 304 → 0 boundary edges.\n- **Volume** (`operations/measure`): a scalloped collar's analytic\nintegral is a deferred u-dependent lune trim, so its volume comes from\nthe (now watertight) whole-solid mesh via the divergence theorem. The\nconstant-v bored-quadric fast path (#1005 sphere−cyl) is gated to stay\nanalytic and is unchanged.\n\n## Verification\n\n- Raw GFA: F=8 (6 disks + 2 collars), free edges 0, manifold.\n- Census: `box ∩ sphere 1.94ms faces=8 exact analytic`; the other 8\nboolean cases unchanged.\n- Render: tessellated mesh watertight (0 boundary edges) at deflections\n0.05 and 0.005; area matches analytic (no cap-fill).\n- Volume: `solid_volume` = 797.4 (analytic 797.97, <0.1%), convergent\nacross deflections.\n- No regression: `sphere − cyl` still 587.671 analytic/watertight; plain\nsphere/cyl/cone/torus tessellate unchanged.\n- Full suite: **2456 passed, 9 skipped**; clippy clean; fmt clean; layer\nboundaries valid.\n- Fixtures: `intersect_box_centered_sphere_is_analytic_collar` (boolean)\n+ `box_centered_sphere_collar_tessellates_watertight` (tessellation).\n\n## Reused by `torus − box` (PR5)\n\nLayer 1 (split) and the Layer-4 collar render/volume machinery are\nsurface-agnostic / already dispatch for `Torus`, so the toroidal collar\nof `torus − box` will reuse them; only its analytic *split* (the\nplane×torus quartic) remains case-specific work.",
          "timestamp": "2026-06-25T08:24:03-07:00",
          "tree_id": "45693599f03507f8c4fa03cb9aa2b0dd65608869",
          "url": "https://github.com/andymai/brepkit/commit/6b4e781988f377a3decc5b5c441f95a955bd13d7"
        },
        "date": 1782401173022,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1455604,
            "range": "± 79401",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1495281,
            "range": "± 3944",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13184,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 589424,
            "range": "± 1775",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21609153,
            "range": "± 81112",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e59959d0a9a17946cbfea18bf446de82d6e9b9ca",
          "message": "chore(main): release 2.120.5 (#1007)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.5](https://github.com/andymai/brepkit/compare/v2.120.4...v2.120.5)\n(2026-06-25)\n\n\n### Bug Fixes\n\n* **operations:** close box∩sphere boolean analytically (seam split +\ncollar render/volume)\n([#1006](https://github.com/andymai/brepkit/issues/1006))\n([6b4e781](https://github.com/andymai/brepkit/commit/6b4e781988f377a3decc5b5c441f95a955bd13d7))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-25T15:29:08Z",
          "tree_id": "b85e9a0203bee9801c8f26b856bd37dd458e33da",
          "url": "https://github.com/andymai/brepkit/commit/e59959d0a9a17946cbfea18bf446de82d6e9b9ca"
        },
        "date": 1782401474162,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1402162,
            "range": "± 77487",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1495280,
            "range": "± 1918",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13115,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 589205,
            "range": "± 1608",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21522289,
            "range": "± 30572",
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
          "id": "0dadfc9d982bf59243eb2495c94d8737a76fba13",
          "message": "fix(algo): assemble perpendicular cyl∪cyl Fuse analytically (#1008)\n\n## What\n\nMakes `Fuse(cyl, cyl)` for two perpendicular equal-radius cylinders (a\n**Steinmetz solid**) produce an analytic, manifold, correct-volume B-Rep\ninstead of a 138-face mesh fallback. Census: **19ms / 138 mesh → 1.78ms\n/ 6 analytic faces** (2 mutually-trimmed cylinder walls + 4 caps).\n\nFourth PR in the campaign to eliminate the boolean→mesh fallbacks (after\n#1003 keystone, #1005 sphere−cyl, #1006 box∩sphere).\n\n## The fix (analytic B-Rep + volume)\n\n- **`phase_ff`** — keep the closed Steinmetz seam loop whole when its\nin-both run is near-complete (≥ N−3), and clamp the seam-wrap-trim\nparameter to the curve domain. The closed-NURBS seam's wrap returned an\nout-of-domain `t` that `NurbsCurve::evaluate` extrapolated to a garbage\npoint, collapsing the carved loop → walls dropped.\n- **`face_splitter`** — emit the complementary outer-wall remainder\n(boundary-with-hole) for the Steinmetz internal-loop split, with a\ncurved-remainder interior point placed clear of the lens, so Fuse keeps\nboth mutually-trimmed walls (not just the inside lobe).\n- **`check`/`measure`** — integrate the holed cylinder/cone band\nexcluding its inner loops (even-odd over the *combined* lens arrangement\n— each lens ellipse is a full-u sinusoid bounding no area alone), so the\nanalytic volume subtracts the Steinmetz intersection.\n\n## Verification\n\n- Raw GFA: 6 analytic faces, **manifold** (euler 4), free edges 0.\n- Census: `cyl ∪ cyl 1.78ms faces=6 exact analytic`; other 8 boolean\ncases unchanged.\n- Volume: 985.3 (analytic 987.0, **0.17%**), via the holed-band analytic\nintegrator.\n- Full suite **2457 passed**; clippy clean; fmt clean; layer boundaries\nvalid.\n- Regression fixture:\n`fuse_perpendicular_cylinders_is_analytic_watertight`.\n\n## Known limitation — watertight render mesh deferred (precisely\nroot-caused)\n\nThe preview tessellation renders **two full interpenetrating tubes**\n(correct silhouette, hidden internal walls). `solid_volume` is analytic\nand correct, **independent of the mesh**, so volume/STEP/B-Rep are\nright; only the preview mesh lags.\n\nA watertight render needs the *exact* ellipse seam, which **self-touches\nat (0,±3,0)** — a genuine boundary singularity of the Steinmetz union.\nEight measured tessellator/GFA approaches traced this to one missing\ncore-GFA primitive: **arc-identity-aware edge merging**. The exact\nseam's four co-endpoint arcs (a \"double-theta\" graph between the two\ncrossing vertices) are collapsed by the current endpoint-keyed\n`merge_duplicate_edges` → non-manifold (euler odd). This is the **same\nprimitive that gates box∩sphere's exact closure and torus−box**, is\nregression-prone (a naive midpoint key breaks the coplanar-cap-cylinder\ntests), and is a **standalone dedicated core-GFA effort** — not part of\nthis PR. The current marched-NURBS seam is manifold precisely because\nits loops are ~0.11 apart and dodge the singularity.\n\n## Does not touch\n\nThe render-blocking arc-identity primitive is the campaign's terminal\nfinding and the highest-leverage next step (unblocks cyl∪cyl render +\nbox∩sphere exact + torus−box). `torus − box` remains a future PR.",
          "timestamp": "2026-06-25T18:36:59-07:00",
          "tree_id": "31f3e58a992a20243eb28a6c7735130e0bcac9e7",
          "url": "https://github.com/andymai/brepkit/commit/0dadfc9d982bf59243eb2495c94d8737a76fba13"
        },
        "date": 1782437951535,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1357175,
            "range": "± 17173",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1444155,
            "range": "± 36988",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12035,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 648955,
            "range": "± 1822",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20695582,
            "range": "± 3350549",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "548424ed321749409b5538b2ea35939a3ee4e21f",
          "message": "chore(main): release 2.120.6 (#1009)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.6](https://github.com/andymai/brepkit/compare/v2.120.5...v2.120.6)\n(2026-06-26)\n\n\n### Bug Fixes\n\n* **algo:** assemble perpendicular cyl∪cyl Fuse analytically\n([#1008](https://github.com/andymai/brepkit/issues/1008))\n([0dadfc9](https://github.com/andymai/brepkit/commit/0dadfc9d982bf59243eb2495c94d8737a76fba13))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.120.6 for `brepkit-wasm`, fixing the Fuse operation for\nperpendicular cylinder∪cylinder by assembling it analytically for more\nstable and accurate results.\n\n- **Bug Fixes**\n- Use analytical assembly for perpendicular cyl∪cyl Fuse to avoid\nartifacts and tolerance errors in boolean unions.\n\n<sup>Written for commit 1b09ab31f70f80f5afc3b10cccf12b6354b9bb8e.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1009?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T01:42:13Z",
          "tree_id": "40a1199963f7b84a9a4a4015a1d27e2b2d787383",
          "url": "https://github.com/andymai/brepkit/commit/548424ed321749409b5538b2ea35939a3ee4e21f"
        },
        "date": 1782438256801,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 1348985,
            "range": "± 3130",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1436722,
            "range": "± 1178",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11908,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 604104,
            "range": "± 1407",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20187378,
            "range": "± 69819",
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
          "id": "ead6f717904265047b3af89b9871d8b5d9828444",
          "message": "fix(algo): close torus−box boolean analytically (plane×torus seam + toroidal band) (#1010)\n\n## What\n\nMakes `Cut(torus, box)` fully analytic — B-Rep + watertight render +\ncorrect volume — instead of a 1733-face mesh fallback. Census: **233ms /\n1733 faces → 14ms / 5 faces**.\n\n**The campaign's fourth and final case** (after #1003 keystone, #1005\nsphere−cyl, #1006 box∩sphere, #1008 cyl∪cyl). With this, **all\nprimitive-boolean cases in the census are analytic — zero mesh\nfallbacks.**\n\nThe case: torus (major 10, minor 3) minus a side-8 box that severs one\nside of the ring, leaving a capped C-tube (genus-0).\n\n## The fix (staged)\n\n**B-Rep**\n- **plane×torus marcher closure** — the marcher emitted\nnear-closed-but-*open* ovals (~0.15mm gap, one grid step) that fell into\na no-man's-land: too open to be recognized as a loop, never clipped.\nClose the fit when the chain wraps (a grazing inner-tangent figure-eight\nstays open). Adds exact `intersect_line_torus` (quartic, Durand-Kerner)\nfor box-edge crossings.\n- **`FaceExtent` for a whole torus** — a whole untrimmed torus is now\nfull extent, so `restrict_curves_to_faces` clips the ovals against the\nbox (it was bailing on the `None` v-range).\n- **exact-crossing trim + arc-split + band tracer** — trim each oval at\nits **exact** box-edge∩torus crossings (z=±1.105, shared vertices →\nwatertight), always split the kept arc at its midpoint, and trace the\nkept toroidal band (a φ-wrapping u-band, the long 294° kept side).\n\n**Render + volume**\n- `tessellate_torus_notch_band` — sweeps the band along u, sharing seam\nvertices with the notch walls (crack-free). The u-sweep starts at each\nboundary loop's **kept-side edge** (a mean start folds the stitch back\nover the boundary strip → undercount).\n- volume via `signed_volume_from_mesh` off the now-watertight mesh (the\nband isn't closed in isolation), gated to the notch-band signature.\n\n**General engine improvements** (motivated here, gated narrow)\n- SD `source_face` guard: complementary sub-faces split from the same\ninput face are never same-domain duplicates of each other.\n- `perform_areas`: a single result shell is always growth (no enclosing\nshell to be a cavity of); multi-shell results keep the volume-sign\nsplit.\n\n## Why not the \"arc-identity merge-key primitive\"\n\nThe long-discussed shared *arc-identity-edge-merge primitive* is **not\nbuildable**: `merge_duplicate_edges`'s endpoint-pair key is load-bearing\nfor the gridfinity lip ring (Line-chord + Circle-arc co-endpoint pairs,\ndeviating up to 2.4mm, that **must** collapse for manifoldness), while\ntorus−box's notch-wall lens needs the **opposite** (Line + co-endpoint\narc must stay distinct). Same local configuration, opposite required\noutcome → no merge *key* can separate them. The **splitter-side\narc-split** (give the lens arc a midpoint vertex so the face is 3 edges)\ncontrols the geometry *we* emit and sidesteps it — so\n**`merge_duplicate_edges` is byte-identical to main**.\n\n## Verification\n\n- census `torus − box`: 14ms / 5 faces (1 toroidal band + 4 plane notch\nwalls) / exact analytic; free=0, manifold.\n- render watertight: whole-solid mesh bd-edges == 0, stable across\ndeflection 0.1/0.05/0.02 (no seam cracks).\n- volume: `solid_volume` = 1537.5 vs 1543.07 (Monte-Carlo, 20M) =\n**0.36%**, convergent upward (inscribed-mesh).\n- **No regression:** gridfinity d1/d3/d4/d5 + coplanar caps (#909/#859)\n+ #696 seams green; box∩sphere (8) / sphere−cyl (3) / cyl∪cyl (6) /\ncyl∩cyl (3) unchanged; **plain torus unchanged** (the new branches gate\non the notch-band signature); `merge_duplicate_edges` byte-identical to\nmain; full suite **2476 passed**.\n- 2 fixtures: `cut_torus_by_box_notch_is_analytic_watertight`,\n`torus_box_notch_band_tessellates_watertight`.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nMakes the torus − box cut fully analytic, watertight, and fast. Replaces\na 1733‑face mesh fallback with a 5‑face B‑Rep (233ms → 14ms).\n\n- **New Features**\n- Analytic trim of plane×torus ovals at exact box‑edge×torus crossings\nusing `intersect_line_torus` (quartic), then split the kept arc at its\nmidpoint; keeps all in‑box arcs.\n- Torus notch band tracing with `split_torus_band_by_arrangement`\n(φ‑wrapping u‑band), and structured `tessellate_torus_notch_band` that\nsweeps along u from each loop’s kept‑side edge and shares seam vertices;\ninterior point from the boundary loops’ long‑arc midpoint.\n- Volume for this case is taken from the watertight whole‑solid mesh via\n`signed_volume_from_mesh` when a torus notch band is detected.\n\n- **Bug Fixes**\n- Plane×torus marcher: wrapped ovals now close; grazing inner‑tangent\nfigure‑eight stays open.\n- A whole, untrimmed torus is treated as full `FaceExtent` so ovals are\nclipped against the box face.\n- Same‑domain detection skips only complementary splits from the same\n`source_face` (distinct interiors); coincident same‑source duplicates\nstill de‑duplicated.\n- Single‑shell classification: a lone negative‑volume shell is growth\nonly if outward per a curvature‑robust divergence‑flux check; rejects\ngenuine inward cavity results. Added cube orientation tests.\n- Hardened torus pieces: tighter arc‑join tolerance; notch‑band v‑wrap\ndetection samples along edges; quartic root finder filters by residual\nand de‑dups. Added stronger torus−box fixtures (exact 5‑face\ndecomposition; watertight across deflections).\n\n<sup>Written for commit bcd2832a3932c27827637514ef834a7bd45e9c77.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1010?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-25T22:38:07-07:00",
          "tree_id": "cdfdac18892a25392c9570f1052dd9976fb19445",
          "url": "https://github.com/andymai/brepkit/commit/ead6f717904265047b3af89b9871d8b5d9828444"
        },
        "date": 1782452413219,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 755279,
            "range": "± 2345",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 851700,
            "range": "± 1052",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13116,
            "range": "± 191",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 621787,
            "range": "± 1757",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19832762,
            "range": "± 335991",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a9ef5afd59a67871bb4b16260d01fdd615ebe3e5",
          "message": "chore(main): release 2.120.7 (#1011)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.120.7](https://github.com/andymai/brepkit/compare/v2.120.6...v2.120.7)\n(2026-06-26)\n\n\n### Bug Fixes\n\n* **algo:** close torus−box boolean analytically (plane×torus seam +\ntoroidal band) ([#1010](https://github.com/andymai/brepkit/issues/1010))\n([ead6f71](https://github.com/andymai/brepkit/commit/ead6f717904265047b3af89b9871d8b5d9828444))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.120.7 for `brepkit-wasm` with a robustness fix for torus–box\nboolean operations to prevent gaps in results.\n\n- **Bug Fixes**\n- Analytically closes torus–box booleans by handling the plane×torus\nseam and the toroidal band.\n\n<sup>Written for commit 37c6bb2d04825ecfb23bec07e263a4d531daf483.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1011?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T05:43:05Z",
          "tree_id": "6bf0d6447628fd42b4baae79b5a97b38df58b193",
          "url": "https://github.com/andymai/brepkit/commit/a9ef5afd59a67871bb4b16260d01fdd615ebe3e5"
        },
        "date": 1782452707546,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 713858,
            "range": "± 2105",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 806712,
            "range": "± 560",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11962,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 585864,
            "range": "± 11715",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18351257,
            "range": "± 44962",
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
          "id": "45c1375881609a08edd6cdf906066954b3c58797",
          "message": "feat(operations): recover analytic surfaces of revolution + exact volume (#1012)\n\n## What\n\nRecovers **analytic surfaces of revolution** from `revolve`. Previously\n`revolve` produced NURBS faces (≈0.04% volume error, up to 2.3% on\npointed cones) for everything except axis-parallel-line→`Cylinder`. Now\nevery analytic profile edge becomes its exact surface of revolution with\nexact volume, and full-turn disc-cap convex revolutions match the\nprimitives' face counts.\n\nFirst PR of the **analytic-recovery campaign for non-boolean ops**\n(after the primitive-boolean campaign #1003/#1005/#1006/#1008/#1010).\nRevolve was the diagnosis's #1 target: genuinely closable (the closed\nforms already exist in `make_cone`/`make_torus`), unlike\nfillet/offset/sweep which *introduce* blend surfaces with no closed\nform.\n\n## Recognition — all 4 profile-edge types (`revolution_band_surface`)\n\n| profile edge | → surface | layer |\n|---|---|---|\n| axis-parallel line | `Cylinder` | existing |\n| **oblique line** | **`Cone`** (apex + half-angle, mirroring\n`make_cone`) | L1 |\n| **perpendicular line** | **`Plane`** annular cap | L4 |\n| **circular arc** | **`Torus`** band | L2 |\n\nA genuine NURBS/spline profile still declines to NURBS — correct, no\nclosed form. (The decline comment at the old `revolve.rs:159` claiming\n\"oblique lines have no simpler exact form\" was factually wrong — an\noblique line revolved *is* a cone.)\n\n## Volume (coupled work)\n\n- **`analytic_revolution_solid_volume`** — per-face analytic sum,\n**tightly gated** to a genuine surface of revolution (quadric walls\nsharing one axis line; every cap's arcs centered on that axis). The\ntight guard is load-bearing: a loose version (any arc-bounded planar\nface) **regressed 4 boolean tests 3×** (rounded-rect caps) — the\naxis-centered check cleanly separates revolve caps from corner caps.\n- **`planar_cap_signed_volume`** — exact disc/annulus/sector area via\n**Green's theorem** (exact circular-arc bulge `±ρ²(|α|−sin|α|)`, never\nchorded). This is the prerequisite that makes the `Plane` cap arm\nnon-regressing (boundary chording was the original reason caps stayed\nNURBS).\n- **apex-singularity fix** — a cone band touching the apex (where the\nangular parameter is undefined on the axis) read a 2× angular span →\n**+50% volume on pointed cones**. Fixed by skipping the apex vertex's\n`u` (the cone analog of the #968 cylinder-integrator midpoint fix).\n\n## Periodic-face merge\n\nFor a full revolution of a fully-analytic disc-cap profile, build **one\nperiodic face per profile edge** (shared rim circles + seam line,\nmirroring `make_cylinder`/`make_cone`) instead of 4 angular segments:\n**frustum 16 → 3 faces** (= `make_cone`), watertight (the u=0≡2π seam\nreuses the primitives' proven shared-rim topology, so #696 doesn't\nbite). Also fixes a latent topology bug — the segmented path's\ndegenerate on-axis bands faked χ=0; the merge gives the correct genus-0\nχ=2.\n\n## Verification\n\n- All 4 profile types analytic + exact volume. frustum/cylinder → 3\nfaces, watertight (bd=0 at defl 0.1/0.05/0.02), matching\n`make_cone`/`make_cylinder`. Pointed cone 2.31% → **0.0000%**.\n- **No regression:** `make_cone`/`make_cylinder`/`make_torus` (which\nrevolve-fallback through this path), sweep/loft, all booleans\n(box∩sphere, torus−box), gridfinity `*_inmem`; full suite **2481\npassed**. clippy/fmt/boundaries clean.\n- Tests: cone/torus recognition + exact volume + watertight + the merged\nface counts; `revolve` cases added to `approx_census`.\n\n## Deferred (noted, not blocking — all stay analytic + exact, just\nover-segmented)\nPointed-cone apex periodic-merge (needs a degenerate apex seam wire);\nannulus/washer-cap merge (inner-wall-toward-axis orientation — caught\nvia volume verification and scoped out rather than risk it);\npartial-turn closed-circle → partial torus.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRecovered exact analytic surfaces in `revolve` (Cylinder, Cone, Plane\ncaps, Torus) and added exact, tessellation‑free volume integration for\nanalytic surfaces of revolution. Also fixed disc‑cap and torus seam edge\ncases so multi‑section revolves compute exact, deflection‑independent\nvolume.\n\n- **New Features**\n- Profile edge → surface: parallel line → `Cylinder`, oblique line →\n`Cone`, perpendicular line → `Plane` cap, circular arc → `Torus`.\n- Exact volume for analytic revolutions via per‑face integrals; planar\ncaps use exact disc/annulus/sector area (Green’s theorem).\n- Full‑turn analytic profiles build one periodic face per edge (e.g.,\nfrustum/cylinder → 3 faces), watertight; NURBS only when needed.\n- Added a revolve survey to `approx_census`; new watertight/volume\ntests; `transform` NURBS test now uses `loft_smooth`.\n\n- **Bug Fixes**\n- Fixed apex‑singularity on cone bands touching the apex (+50% volume\nerror).\n- Closed‑circle caps now integrate (α=2π → πρ²); arc‑bulge uses the\ncurve’s domain midpoint and is orientation‑consistent, so annular caps\nsubtract the inner rim correctly.\n- Torus‑band minor‑range ambiguity at the v=0/2π seam now defers to\ntessellation to avoid mis‑integration.\n- Tight axis‑centered guard for analytic‑volume recognition; cached\nplanar‑cap volumes; `approx_census` `surf_tags` is now exhaustive\n(counts `Sphere`).\n\n<sup>Written for commit 126a0421acdc52dd0dca5b49ee993022fb590814.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1012?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T09:32:08-07:00",
          "tree_id": "8562fe048798902787d01755bf1fdd54450ea03c",
          "url": "https://github.com/andymai/brepkit/commit/45c1375881609a08edd6cdf906066954b3c58797"
        },
        "date": 1782491654110,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 734768,
            "range": "± 3886",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 820473,
            "range": "± 2648",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11862,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 622897,
            "range": "± 4127",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19082699,
            "range": "± 96447",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d84b95c0d9d74dd39835bc62d73b750426cc985",
          "message": "chore(main): release 2.121.0 (#1014)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.121.0](https://github.com/andymai/brepkit/compare/v2.120.7...v2.121.0)\n(2026-06-26)\n\n\n### Features\n\n* **operations:** recover analytic surfaces of revolution + exact volume\n([#1012](https://github.com/andymai/brepkit/issues/1012))\n([45c1375](https://github.com/andymai/brepkit/commit/45c1375881609a08edd6cdf906066954b3c58797))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.121.0 of `brepkit-wasm` adds an operations feature to recover\nanalytic surfaces of revolution and compute exact volume. This improves\nprecision and enables exact volume results for revolved geometry.\n\n<sup>Written for commit 57fe2eb807b07f67842a3ce619eb1690df244788.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1014?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T16:37:01Z",
          "tree_id": "25a020be37ccc0d64c6c62b96aca0cf272375835",
          "url": "https://github.com/andymai/brepkit/commit/3d84b95c0d9d74dd39835bc62d73b750426cc985"
        },
        "date": 1782491935891,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 591694,
            "range": "± 1819",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 667729,
            "range": "± 1216",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10207,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 488122,
            "range": "± 4871",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 15759151,
            "range": "± 319486",
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
          "id": "f7d30008e660d233acbd0727eeaa9f12c3f96c99",
          "message": "feat(render): brepkit-render M1 — offscreen wgpu renderer (#1013)\n\n## What\n\n**Milestone 1 of a GPU renderer for brepkit** — a new `crates/render`\n(`brepkit-render`) leaf crate that renders a B-Rep `Solid` **offscreen**\nvia wgpu: Lambert-shaded mesh + crisp topological edges + a per-pixel\n`FaceId` buffer for picking. Offscreen-first so it's verifiable\nheadlessly (and independently useful for thumbnails / server-side\nrenders / tessellator visual-regression tests).\n\nVerified working — box and cylinder render correctly (shaded, correctly\nframed, crisp edges), reproducible via the test (writes PNGs to the temp\ndir).\n\n## API\n```rust\nrender_solid_offscreen(topo, solid, &Camera, &RenderOpts) -> Result<RenderOutput, RenderError>\n// RenderOutput { color: image::RgbaImage, id_buffer: Vec<u32>, width, height }\n//   .face_id_at(x, y) -> Option<u32>   // 0 = background\n```\n\n## Pipeline (consumes existing brepkit output — no kernel changes)\n- `tessellate_solid_grouped_with_tolerance` → positions/normals/indices\n+ **`face_offsets`** → per-triangle `FaceId`.\n- `sample_solid_edges` → topological edges as crisp depth-biased lines.\n- **RTC precision:** tessellate in f64, upload f32 **relative to the\nmodel-center origin**, fold the f64 center into the view matrix.\n- Three targets: `Rgba8UnormSrgb` color, `Depth32Float`, **`R32Uint`\nid** (each face's triangles write its `FaceId`); two-sided Lambert mesh\npass + optional edge pass.\n- **Raw wgpu + pollster** (per the renderer research: not Bevy's\nf32-Transform game shape, not archived rend3, not WebGL-only three-d).\n\n## Roadmap (this is M1)\nM1 offscreen renderer (this PR) → M1.5 interactive winit window → **M2\nanalytic-quadric view-dependent compute meshing** (the differentiator —\nship surface parameters, mesh per-frame at LOD) → M3 direct quadric\nray-cast → later WebGPU/wasm. Rationale\n([research](.claude/agent-memory-local/researcher/cad-gpu-renderer-landscape.md)):\nbrepkit's kernel runs **in-browser via wasm**, so it can mesh\nview-dependently client-side — an architecture neither Zoo (server-side\nGPU + pixel streaming) nor Onshape (server tessellation) can use.\n\n## Verification\n- Box + cylinder render non-blank, silhouette plausible, shaded with\ncrisp edges; `id_buffer` maps every non-background pixel to a real\n`FaceId`. 2 tests, **gated on GPU-adapter availability** (skip cleanly\nif the runner has no GPU/software-Vulkan).\n- clippy `-D warnings`, fmt, `check-boundaries.sh`, doctest,\nwhole-workspace build all clean; no `unwrap`/`expect`/`panic` in lib\ncode.\n\n## Consideration for review\nThe heavy `wgpu` dep tree is isolated to this leaf crate (nothing\ndepends on it), but `cargo build --workspace` (and CI) will now compile\nit. If kernel-CI time matters, we can later move `render` out of the\ndefault workspace build (or to its own workspace). Flagging rather than\ndeciding unilaterally.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds a new leaf crate `brepkit-render` that renders B‑Rep solids\noffscreen via `wgpu`, producing a Lambert‑shaded image and a per‑pixel\nface‑id buffer for picking. Hardened with strict size checks, adapter\nfallback, and MRT correctness for reliable headless runs.\n\n- New Features\n- API: `render_solid_offscreen(topo, solid, &Camera, &RenderOpts) ->\nRenderOutput { color, id_buffer, width, height }` with `face_id_at(x,\ny)`.\n- Pipeline: RTC upload (f64 tessellation → f32 positions around model\ncenter), two‑sided Lambert mesh pass, optional edge pass, and R32 `id`\ntarget (0 = background).\n- Tests: box and cylinder offscreen; gated on adapter via\n`probe_adapter`; write PNGs to the temp dir.\n\n- Bug Fixes\n- Robustness: reject zero and oversized renders (`InvalidSize`,\n`SizeTooLarge`), fall back from real GPU to software adapter for device\ncreation, stricter mesh validation (indices/face offsets/ranges), and\nguarded camera math (degenerate up, projection clamps) with unit tests.\n- Edge pass: add write‑masked `@location(1)` id output to satisfy\nmulti‑render‑target rules; picking unchanged.\n- Tooling: `deny.toml` allowlist updated for `wgpu` transitive licenses;\n`check-boundaries.sh` and `CLAUDE.md` enforce `render` as an L4 leaf.\n\n<sup>Written for commit 91a0e83ac3752e846e329367761db73138315010.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1013?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T10:12:15-07:00",
          "tree_id": "404ac7cf8eb1568c01499bde7b0ec57d2d45e629",
          "url": "https://github.com/andymai/brepkit/commit/f7d30008e660d233acbd0727eeaa9f12c3f96c99"
        },
        "date": 1782494075570,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 734803,
            "range": "± 1830",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821311,
            "range": "± 2513",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11913,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 622591,
            "range": "± 11703",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19077998,
            "range": "± 415456",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ea21ba69a8d25088714639e7bbeebd86eb3dba3c",
          "message": "chore(main): release 2.122.0 (#1015)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.122.0](https://github.com/andymai/brepkit/compare/v2.121.0...v2.122.0)\n(2026-06-26)\n\n\n### Features\n\n* **render:** brepkit-render M1 — offscreen wgpu renderer\n([#1013](https://github.com/andymai/brepkit/issues/1013))\n([f7d3000](https://github.com/andymai/brepkit/commit/f7d30008e660d233acbd0727eeaa9f12c3f96c99))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.122.0 of `brepkit-wasm`, adding the offscreen `wgpu` renderer\n(M1) from `brepkit-render` for headless rendering.\n\n<sup>Written for commit 0a567078c2658c50d78a26acc13c41add44b9e8a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1015?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T17:17:55Z",
          "tree_id": "5cda83fb33a4db47ed49d7146df208f9c7b49542",
          "url": "https://github.com/andymai/brepkit/commit/ea21ba69a8d25088714639e7bbeebd86eb3dba3c"
        },
        "date": 1782494404021,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 755644,
            "range": "± 4938",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 852134,
            "range": "± 5268",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13156,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 620744,
            "range": "± 1386",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19816518,
            "range": "± 139564",
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
          "id": "362d8a71c2edffb8d39d10404b4fdbcf01e169c6",
          "message": "feat(render): interactive viewer — orbit, pan, zoom, click-to-pick (M1.5) (#1016)\n\n## What\n\n**Interactive viewer for brepkit-render** (M1.5) — an orbit/pan/zoom\nwindow with **click-to-pick face highlighting**, built on the merged M1\noffscreen renderer (#1013). Behind a `window` feature (winit 0.30,\nmatched to wgpu 29's `rwh_06`).\n\n**Live-verified:** launched on a real display and captured a frame from\nthe running window — it renders the box∪cylinder fuse correctly\n(Lambert-shaded, crisp topological edges), and click-to-pick resolves\nthe kernel `FaceId` under the cursor.\n\n## API + controls\n```rust\nview_solid(topo, solid, &ViewOpts) -> Result<(), RenderError>   // opens the window, runs the event loop\n```\n| action | control |\n|---|---|\n| Orbit | left-drag |\n| Zoom | scroll |\n| Pan | right-drag / Shift+left-drag |\n| **Pick a face** | left-click → highlights orange + reports its\n`FaceId` (click again to clear) |\n\nRun the demo: `cargo run -p brepkit-render --example viewer --features\nwindow`\n\n## How it reuses M1 (no duplication)\nThe viewer shares M1's render passes verbatim. M1's monolithic\n`render()` was factored into building blocks both paths use:\n- `GpuContext` (optional `compatible_surface` for presentation) +\n`acquire_device` (real→software adapter fallback, kept from M1's review\nfixes).\n- format-parametrized `Pipelines` (offscreen `Rgba8UnormSrgb` vs the\nsurface's preferred sRGB format), `GlobalsBinding`,\n`encode_scene(PassTargets)`.\n- Picking re-renders the `R32Uint` id pass for the current view and\nreads back the pixel under the cursor — the same id buffer M1 already\nproduces.\n\nThe offscreen `render()` and its tests are unchanged. M1's review fixes\n(edge MRT, degenerate-`up`, render-size validation, mesh error handling)\nare all present (this branch was rebased onto the fixed M1), so the\nviewer is valid on strict GPUs, not just NVIDIA.\n\n## Verification\n- `cargo build -p brepkit-render` (default / `--features window` /\n`--example viewer`): clean.\n- `cargo nextest run -p brepkit-render`: 6/6 (M1 offscreen + camera/size\nunit tests).\n- clippy `-D warnings` (default + window), fmt, `cargo deny check`,\n`check-boundaries.sh`: all clean. No `unwrap`/`expect`/`panic` in lib.\n- The interactive window is inherently not CI-verifiable (needs a\ndisplay); it was verified live by capturing a frame from the running\nwindow.\n\nMilestone 1.5 of the renderer roadmap (next: M2 compute-mesher —\nseparate PR).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds an interactive viewer window with orbit/pan/zoom and click-to-pick\nface highlighting, behind a `window` feature so headless/offscreen users\nstay unaffected. Includes review fixes for stable zoom framing,\nstrict-backend surface setup, and more robust picking.\n\n- New Features\n- `view_solid(topo, solid, &ViewOpts)` opens a `winit` window; orbit\n(left-drag), pan (right-drag or Shift+left), zoom (scroll); left-click\npicks a face and toggles its highlight. Example: `cargo run -p\nbrepkit-render --example viewer --features window`.\n- Picking reuses the `R32Uint` id pass; mesh shader adds a `selected_id`\nuniform to tint the selected face.\n- Optional `window` feature pulls in `winit 0.30` (aligned with `wgpu\n29`/`rwh_06`); new windowing errors added to `RenderError`.\n\n- Refactors\n- Pipeline split into reusable parts for offscreen and viewer:\n`GpuContext`, `GlobalsBinding`, `Pipelines`, `GeometryBuffers`, and\n`encode_scene` (offscreen `render()` unchanged; pipelines are\ncolor-format agnostic; edge pass masks id writes).\n- Stability/robustness: orbit camera derives near/far per-frame and\nfloors min distance to avoid dolly clipping; viewer creates the surface\nand device from the same `wgpu::Instance`, prefers FIFO present mode,\nclamps surface/targets to device limits, floors+clamps pick coordinates\nand logs readback errors; `deny.toml` allows BSD-2/3 for the `winit`\nsubtree.\n\n<sup>Written for commit a5c7a37d4987432ce485ab5ec19dedc4fe6cb374.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1016?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T10:51:51-07:00",
          "tree_id": "cb05a4ebcba9b4346ccf238ca6220523dd6735a7",
          "url": "https://github.com/andymai/brepkit/commit/362d8a71c2edffb8d39d10404b4fdbcf01e169c6"
        },
        "date": 1782496436676,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 726221,
            "range": "± 1228",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 815480,
            "range": "± 689",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12019,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 618654,
            "range": "± 624",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18840245,
            "range": "± 28392",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2de24b31ec30f32dcc2ae46d773e387d3ef060c6",
          "message": "chore(main): release 2.123.0 (#1018)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.123.0](https://github.com/andymai/brepkit/compare/v2.122.0...v2.123.0)\n(2026-06-26)\n\n\n### Features\n\n* **render:** interactive viewer — orbit, pan, zoom, click-to-pick\n(M1.5) ([#1016](https://github.com/andymai/brepkit/issues/1016))\n([362d8a7](https://github.com/andymai/brepkit/commit/362d8a71c2edffb8d39d10404b4fdbcf01e169c6))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.123.0 adds an interactive viewer to the renderer in\n`brepkit-wasm` to make model inspection easier.\n\n- **New Features**\n- Interactive viewer controls: orbit, pan, zoom, and click-to-pick to\nidentify entities under the cursor.\n\n<sup>Written for commit ac89e2aa09692425067dab6a20b7500f4b6f5241.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1018?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T17:56:47Z",
          "tree_id": "da8d48985ed3e0144b42d7b65f6f59148650d8f1",
          "url": "https://github.com/andymai/brepkit/commit/2de24b31ec30f32dcc2ae46d773e387d3ef060c6"
        },
        "date": 1782496732153,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 727501,
            "range": "± 1998",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821390,
            "range": "± 714",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11906,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 617741,
            "range": "± 900",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18814188,
            "range": "± 1004551",
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
          "id": "cf1dc6e0c6c845f8e43f1f5a28e44bb936f3f5a1",
          "message": "feat(render): compute-shader quadric mesher for cylinders (M2) (#1017)\n\n## What\n\n**GPU compute-shader quadric mesher** (M2) — the renderer's\ndifferentiator. Instead of CPU-tessellating an analytic face and\nuploading thousands of triangles, brepkit can ship the surface's\n**parameters** and let a compute shader evaluate the parametric surface\ninto a vertex buffer at a caller-chosen LOD. This PR implements it for\nthe **cylinder**; cone/sphere/torus + screen-space-adaptive LOD follow\nthe same pattern.\n\nBuilt on the merged M1 offscreen renderer (#1013).\n\n## How it works\n- `extract_cylinder_descriptor(topo, FaceId)`: `FaceSurface::Cylinder` →\n`CylinderDescriptor` (center/axis/x_ref/y_ref/radius + axial trim\n`v0..v1` from the outer wire, full-revolution `u0..u1`); RTC center\nfolded into the camera like M1.\n- `quadric_mesh.wgsl`: `cs_vertices` writes `pos(u,v)=origin+r(cos\nu·x+sin u·y)+v·axis` + the radial normal into a flat `array<u32>` (**7\nwords/vertex = M1's `Vertex` stride**), `cs_indices` writes 2 tris/quad.\nThe *same buffer* is bound STORAGE (compute writes it) then VERTEX/INDEX\n(the draw reads it), so **M1's mesh shader draws the compute output\nunchanged** — no CPU round-trip, no format conversion.\n- Seam: the wrap quad references column 0 → watertight by construction.\n- `render_cylinder_compute_offscreen(..)` renders it through M1's mesh\npass.\n\n`★ why this matters` — brepkit's kernel runs in-browser via wasm, so it\ncan mesh view-dependently *client-side*. Shipping surface parameters (a\nfew floats) instead of a fixed mesh is the foundation for smooth\ninfinite-zoom LOD — an approach server-side CAD renderers can't use.\nWebGPU has no tessellation/mesh shaders, so the mesher *must* be a\ncompute shader; that's exactly what this is.\n\n## Verification (live RTX 4080 — the tests actually mesh + render)\n- **Geometric correctness**: `compute_mesh_matches_cpu_silhouette` — the\ncompute-meshed cylinder's silhouette bbox is **identical** to M1's CPU\ntessellation at the same camera;\n`compute_mesh_matches_cpu_for_off_origin_cylinder` exercises the\n`axis_origin` term.\n- **LOD**: triangle count scales `2·n_u·n_v`; the coarse 6-gon is\nvisibly faceted (5px shortfall vs a 256-gon reference), the fine 48-gon\nis smooth (0px).\n- **Watertight seam**: a dedicated test centers the u=0 seam on the\ncamera and asserts zero interior holes.\n- `cargo nextest run -p brepkit-render`: **11/11** (5 compute + M1's 6).\nclippy `-D warnings`, fmt, `cargo deny check`, `check-boundaries.sh`:\nclean. No `unwrap`/`expect`/`panic` in lib.\n\n## Notes\n- `pipeline.rs`: reuses main's review-fixed `acquire_device()`\n(real→software fallback) for the compute path; M1's readback/format\nhelpers widened to `pub(crate)`; M1's `SizeTooLarge` over-size guard\nmirrored onto the compute path.\n- Overlaps PR #1016 (M1.5) only in `pipeline.rs` visibility (#1016\ntouches window/surface code, this touches readback/device helpers) —\nthey merge sequentially; whichever lands first, the other rebases.\n\nNext: cone/sphere/torus descriptors (pole/seam handling) +\nscreen-space-adaptive LOD (the marked `TODO` in `TessFactor`).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdd a GPU compute-shader mesher for cylinders that evaluates the surface\non the GPU and draws it with the existing mesh pass. This removes CPU\ntessellation/uploads and sets up client-side, view-dependent LOD.\n\n- **New Features**\n- `quadric_mesh.wgsl`: `cs_vertices`/`cs_indices` write positions,\nnormals, face-id, and indices; seam wraps to column 0 for a watertight\nmesh; uses a CPU-provided `full` flag and guards divides; column-major\nindexing.\n- `TessFactor::new(..)` clamps to `[3,16384]` (u) / `[1,16384]` (v) to\nkeep counts in `u32`; `CylinderDescriptor` +\n`extract_cylinder_descriptor(..)` (axis frame, radius, axial trim,\nfull-rev u, RTC center); `render_cylinder_compute_offscreen(..)` meshes\nwith `TessFactor` and draws via the existing mesh pipeline with the same\nbuffers bound as STORAGE then VERTEX/INDEX.\n- Export `CylinderDescriptor`, `TessFactor`,\n`extract_cylinder_descriptor`, `render_cylinder_compute_offscreen` from\n`lib.rs`. Headless tests cover silhouette vs CPU, LOD scaling, seam\nwatertightness, off-origin handling, plus `TessFactor` clamp unit tests.\n\n- **Refactors**\n- Make `pipeline::acquire_device`, `unpad_to_rgba`, and `unpad_to_u32`\npublic; reuse device/readback/padding helpers; mirror the `SizeTooLarge`\nguard.\n- Use a `WORDS_PER_VERT` constant and checked `u32::try_from` for index\ncounts to avoid truncation.\n\n<sup>Written for commit e4871bbe608a8cd2c039906318c0af21e6044276.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1017?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T11:18:59-07:00",
          "tree_id": "7641d19ab082e9230f81f952217d51406a0a6814",
          "url": "https://github.com/andymai/brepkit/commit/cf1dc6e0c6c845f8e43f1f5a28e44bb936f3f5a1"
        },
        "date": 1782498064926,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 735773,
            "range": "± 9017",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821024,
            "range": "± 1195",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12140,
            "range": "± 167",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 623397,
            "range": "± 1301",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19529279,
            "range": "± 208619",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d417c887185cee9abb241559facf3651c438b09f",
          "message": "chore(main): release 2.124.0 (#1019)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.0](https://github.com/andymai/brepkit/compare/v2.123.0...v2.124.0)\n(2026-06-26)\n\n\n### Features\n\n* **render:** compute-shader quadric mesher for cylinders (M2)\n([#1017](https://github.com/andymai/brepkit/issues/1017))\n([cf1dc6e](https://github.com/andymai/brepkit/commit/cf1dc6e0c6c845f8e43f1f5a28e44bb936f3f5a1))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds a GPU compute-shader quadric mesher for cylinders in the renderer\nto improve performance and mesh quality. Releases `brepkit-wasm`\n2.124.0.\n\n<sup>Written for commit 31057dd9fcd60934c895d221d8a4e65869fbe30d.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1019?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-26T18:24:00Z",
          "tree_id": "d350404f85df577e6f3cc4eced8687004afb32cd",
          "url": "https://github.com/andymai/brepkit/commit/d417c887185cee9abb241559facf3651c438b09f"
        },
        "date": 1782498363139,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 732460,
            "range": "± 2430",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 820724,
            "range": "± 1270",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12046,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 619225,
            "range": "± 15522",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18842198,
            "range": "± 50412",
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
          "id": "1d06afc42687c050cfa081a4da3a0cfc64393ece",
          "message": "ci: force HTTP/1.1 + retries for crates.io fetches (fix HTTP/2 download flake) (#1020)\n\n## What\nAdd `[http] multiplexing = false` + `[net] retry = 10` to\n`.cargo/config.toml` so dependency fetches stop reddening CI.\n\n## Why\ncrates.io downloads on the runners intermittently fail with `[16] Error\nin the HTTP2 framing layer` while fetching `criterion`'s transitive deps\n(`cast`, `alloca`, `oorandom`, `itertools`) and others. **Today alone\nthis flaked Coverage, Test, MSRV (1.88), Boolean perf, and WASM Size\nReport across the render + revolve PRs** — each a spurious red that only\ncleared on a manual re-run.\n\n## The fix\n- `[http] multiplexing = false` — cargo uses HTTP/1.1 instead of HTTP/2,\nsidestepping the HTTP/2 framing fault that causes the resets.\n- `[net] retry = 10` — retries transient network failures.\n\n## Tradeoff\nHTTP/1.1 has no connection multiplexing, so cold-cache fetches open more\nconnections (marginally slower). Downloads only happen on a cold cargo\ncache — almost always CI — so local-dev impact is negligible, and the\nreliability win removes a recurring class of spurious CI failures +\nmanual re-runs.\n\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nForce HTTP/1.1 and add retries for Cargo dependency downloads to stop\ncrates.io HTTP/2 flakiness in CI. This removes intermittent red jobs\nwithout affecting local dev.\n\n- **Bug Fixes**\n  - Set `[http] multiplexing = false` to use HTTP/1.1 for fetches.\n  - Set `[net] retry = 10` to retry transient network errors.\n- Stabilizes Coverage, Test, MSRV, Boolean perf, and WASM Size jobs;\nonly minor cold-cache slowdown in CI.\n\n<sup>Written for commit 3243e4d426f337795a7da8a696803ad34e1eaaf5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1020?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T11:31:07-07:00",
          "tree_id": "aa056ea47c13190f7b63e26726acba4c6d9314b5",
          "url": "https://github.com/andymai/brepkit/commit/1d06afc42687c050cfa081a4da3a0cfc64393ece"
        },
        "date": 1782498802550,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 733831,
            "range": "± 3079",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 824067,
            "range": "± 2512",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12261,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 623276,
            "range": "± 17129",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18996661,
            "range": "± 48747",
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
          "id": "252efcefcdf06abf32d39a77e184096e8d7f92a9",
          "message": "feat(render): screen-space adaptive LOD for the compute-mesher (M2.1) (#1021)\n\n## What\n\n**Screen-space adaptive LOD** (M2.1) — the compute-mesher now derives\nits tessellation factor **per-frame from the cylinder's projected pixel\nsize**, so detail tracks zoom under a target chord-error budget. This\nreplaces the caller-supplied `TessFactor` (the `// TODO: view-dependent\nLOD` marker from M2) and is the payoff of the compute-mesher: ship the\nsurface's *parameters* and mesh at exactly the detail the current view\nneeds.\n\n## The math\n- Projected radius (perspective): `r_px = r · (H/2) / (d ·\ntan(fov_y/2))`, where `d = |cam.eye − desc.center|`, `H` = viewport\nheight.\n- An inscribed `n_u`-gon has chord error `ε = r·(1 − cos(π/n_u))`.\nBounding the *screen-space* error by `target_px` and solving: `n_u =\nceil(π / acos(1 − clamp(target_px/r_px, 0, 2)))`. A sub-pixel cylinder\nfloors to the `TessFactor` minimum (3).\n- `n_v = 1` — a cylinder's lateral face is **ruled** (straight +\nconstant normal along the axis), so one axial segment is exact.\n(Sphere/torus will need `n_v` adaptivity too — noted for the extension.)\n- Always finishes through `TessFactor::new`, so the `[3, MAX_TESS]`\nclamp + overflow guard still apply; every divisor is guarded, so\ndegenerate inputs (camera on center, zero/NaN fov/budget) clamp cleanly.\n\n## API\n- `screen_space_tess_factor(desc, cam, viewport, target_px) ->\nTessFactor`\n- `render_cylinder_compute_screen_lod(desc, face_id, cam, opts,\ntarget_px) -> RenderOutput` (computes the LOD internally)\n- `pub const DEFAULT_TARGET_PX: f64 = 0.5`\n\n## Verification (live RTX 4080)\n- **Adaptive:** same cylinder → near `n_u=39` (78 tris) **>** far\n`n_u=16` (32 tris), monotonic with distance.\n- **Bound holds:** rendered silhouette chord error near **0.00px** / far\n**0.39px**, both ≤ the 0.5px target.\n- **Near-minimal:** at a fixed view the chosen `n_u` gives 0.00px;\nquartering it → 0.93px (exceeds budget) — so the LOD is tuned, not\nwasteful.\n- `render_cylinder_compute_screen_lod` is byte-identical to rendering\nwith the explicitly-chosen factor.\n- **23/23 render tests** (8 new incl. 5 `screen_space_tess_factor` unit\ntests + degenerate-input handling), stable across 3 runs. clippy `-D\nwarnings`, fmt, `cargo deny check`, `check-boundaries.sh`, doctest:\nclean. No unwrap/expect/panic in lib.\n\n## Note\nGPU render tests run under `cargo nextest` (each test in its own\nprocess). `cargo test`'s multi-threaded harness can SIGSEGV the\nlavapipe/Vulkan driver on concurrent cross-thread device creation —\npre-existing (M2's tests too); nextest (already the project's bar) is\nthe correct answer, so no `--test-threads=1` hack was added.\n\nNext: cone (silhouette radius varies along the axis — use the larger\nend) then sphere/torus (need `n_v` adaptivity for the second curved\ndirection).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds screen-space adaptive LOD for the compute mesher on cylinders. Mesh\ndetail now tracks zoom under a pixel chord‑error budget and handles\nextreme views robustly.\n\n- **New Features**\n- Per-frame LOD from projected radius (uses view-space depth); `n_u`\nmeets the pixel error target, `n_v=1` for the ruled side.\n- API: `screen_space_tess_factor(..)`,\n`render_cylinder_compute_screen_lod(..)`, and `DEFAULT_TARGET_PX = 0.5`\n(re-exported in `render`).\n- Adoption: keep `render_cylinder_compute_offscreen(.., TessFactor, ..)`\nas-is, or switch to `render_cylinder_compute_screen_lod(.., target_px)`.\n\n- **Bug Fixes**\n- `r_px=+∞` (camera engulfed) now requests `MAX_TESS`;\nbehind-camera/degenerate inputs floor to the minimum.\n- Use view-space depth `d = view_dir · (center − eye)` to avoid\nunder-tessellating off-axis surfaces.\n  - Clamp `fov_y` to `(1e-4, π−1e-4)` for bounded, stable LOD.\n\n<sup>Written for commit f61d4edc93b390b80ab8555abe40a3dc8cc63e0b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1021?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-06-26T12:14:49-07:00",
          "tree_id": "25cc06fe94be7a953f91ca4c1bc1f6c66b9fa5ec",
          "url": "https://github.com/andymai/brepkit/commit/252efcefcdf06abf32d39a77e184096e8d7f92a9"
        },
        "date": 1782501413641,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 738878,
            "range": "± 27917",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 829315,
            "range": "± 5398",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11988,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 626244,
            "range": "± 1582",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18975264,
            "range": "± 67622",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "74da44bc4213935dbb7ebaeabda4ec992ddf9ca3",
          "message": "chore(deps): bump the actions group with 3 updates (#1022)\n\nBumps the actions group with 3 updates:\n[actions/checkout](https://github.com/actions/checkout),\n[actions/download-artifact](https://github.com/actions/download-artifact)\nand [taiki-e/install-action](https://github.com/taiki-e/install-action).\n\nUpdates `actions/checkout` from 6.0.3 to 7.0.0\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/actions/checkout/releases\">actions/checkout's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v7.0.0</h2>\n<h2>What's Changed</h2>\n<ul>\n<li>block checking out fork pr for pull_request_target and workflow_run\nby <a href=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2454\">actions/checkout#2454</a></li>\n<li>Bump actions/publish-immutable-action from 0.0.3 to 0.0.4 in the\nminor-actions-dependencies group across 1 directory by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2458\">actions/checkout#2458</a></li>\n<li>Bump flatted from 3.3.1 to 3.4.2 by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2460\">actions/checkout#2460</a></li>\n<li>Bump js-yaml from 4.1.0 to 4.2.0 by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2461\">actions/checkout#2461</a></li>\n<li>Bump <code>@​actions/core</code> and\n<code>@​actions/tool-cache</code> and Remove uuid by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2459\">actions/checkout#2459</a></li>\n<li>upgrade module to esm and update dependencies by <a\nhref=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2463\">actions/checkout#2463</a></li>\n<li>Bump the minor-npm-dependencies group across 1 directory with 3\nupdates by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2462\">actions/checkout#2462</a></li>\n<li>getting ready for checkout v7 release by <a\nhref=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2464\">actions/checkout#2464</a></li>\n<li>update error wording by <a\nhref=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2467\">actions/checkout#2467</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a href=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> made\ntheir first contribution in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2454\">actions/checkout#2454</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/actions/checkout/compare/v6.0.3...v7.0.0\">https://github.com/actions/checkout/compare/v6.0.3...v7.0.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/actions/checkout/blob/main/CHANGELOG.md\">actions/checkout's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>Changelog</h1>\n<h2>v7.0.0</h2>\n<ul>\n<li>Block checking out fork PR for pull_request_target and workflow_run\nby <a href=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2454\">actions/checkout#2454</a></li>\n<li>Bump actions/publish-immutable-action from 0.0.3 to 0.0.4 in the\nminor-actions-dependencies group across 1 directory by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2458\">actions/checkout#2458</a></li>\n<li>Bump flatted from 3.3.1 to 3.4.2 by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2460\">actions/checkout#2460</a></li>\n<li>Bump js-yaml from 4.1.0 to 4.2.0 by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2461\">actions/checkout#2461</a></li>\n<li>Bump <code>@​actions/core</code> and\n<code>@​actions/tool-cache</code> and Remove uuid by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2459\">actions/checkout#2459</a></li>\n<li>upgrade module to esm and update dependencies by <a\nhref=\"https://github.com/aiqiaoy\"><code>@​aiqiaoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2463\">actions/checkout#2463</a></li>\n<li>Bump the minor-npm-dependencies group across 1 directory with 3\nupdates by <a\nhref=\"https://github.com/dependabot\"><code>@​dependabot</code></a>[bot]\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2462\">actions/checkout#2462</a></li>\n</ul>\n<h2>v6.0.3</h2>\n<ul>\n<li>Fix checkout init for SHA-256 repositories by <a\nhref=\"https://github.com/yaananth\"><code>@​yaananth</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2439\">actions/checkout#2439</a></li>\n<li>fix: expand merge commit SHA regex and add SHA-256 test cases by <a\nhref=\"https://github.com/yaananth\"><code>@​yaananth</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2414\">actions/checkout#2414</a></li>\n</ul>\n<h2>v6.0.2</h2>\n<ul>\n<li>Fix tag handling: preserve annotations and explicit fetch-tags by <a\nhref=\"https://github.com/ericsciple\"><code>@​ericsciple</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2356\">actions/checkout#2356</a></li>\n</ul>\n<h2>v6.0.1</h2>\n<ul>\n<li>Add worktree support for persist-credentials includeIf by <a\nhref=\"https://github.com/ericsciple\"><code>@​ericsciple</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2327\">actions/checkout#2327</a></li>\n</ul>\n<h2>v6.0.0</h2>\n<ul>\n<li>Persist creds to a separate file by <a\nhref=\"https://github.com/ericsciple\"><code>@​ericsciple</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2286\">actions/checkout#2286</a></li>\n<li>Update README to include Node.js 24 support details and requirements\nby <a href=\"https://github.com/salmanmkc\"><code>@​salmanmkc</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2248\">actions/checkout#2248</a></li>\n</ul>\n<h2>v5.0.1</h2>\n<ul>\n<li>Port v6 cleanup to v5 by <a\nhref=\"https://github.com/ericsciple\"><code>@​ericsciple</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2301\">actions/checkout#2301</a></li>\n</ul>\n<h2>v5.0.0</h2>\n<ul>\n<li>Update actions checkout to use node 24 by <a\nhref=\"https://github.com/salmanmkc\"><code>@​salmanmkc</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2226\">actions/checkout#2226</a></li>\n</ul>\n<h2>v4.3.1</h2>\n<ul>\n<li>Port v6 cleanup to v4 by <a\nhref=\"https://github.com/ericsciple\"><code>@​ericsciple</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2305\">actions/checkout#2305</a></li>\n</ul>\n<h2>v4.3.0</h2>\n<ul>\n<li>docs: update README.md by <a\nhref=\"https://github.com/motss\"><code>@​motss</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/1971\">actions/checkout#1971</a></li>\n<li>Add internal repos for checking out multiple repositories by <a\nhref=\"https://github.com/mouismail\"><code>@​mouismail</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/1977\">actions/checkout#1977</a></li>\n<li>Documentation update - add recommended permissions to Readme by <a\nhref=\"https://github.com/benwells\"><code>@​benwells</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2043\">actions/checkout#2043</a></li>\n<li>Adjust positioning of user email note and permissions heading by <a\nhref=\"https://github.com/joshmgross\"><code>@​joshmgross</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2044\">actions/checkout#2044</a></li>\n<li>Update README.md by <a\nhref=\"https://github.com/nebuk89\"><code>@​nebuk89</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2194\">actions/checkout#2194</a></li>\n<li>Update CODEOWNERS for actions by <a\nhref=\"https://github.com/TingluoHuang\"><code>@​TingluoHuang</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2224\">actions/checkout#2224</a></li>\n<li>Update package dependencies by <a\nhref=\"https://github.com/salmanmkc\"><code>@​salmanmkc</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/2236\">actions/checkout#2236</a></li>\n</ul>\n<h2>v4.2.2</h2>\n<ul>\n<li><code>url-helper.ts</code> now leverages well-known environment\nvariables by <a href=\"https://github.com/jww3\"><code>@​jww3</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/checkout/pull/1941\">actions/checkout#1941</a></li>\n<li>Expand unit test coverage for <code>isGhes</code> by <a\nhref=\"https://github.com/jww3\"><code>@​jww3</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/1946\">actions/checkout#1946</a></li>\n</ul>\n<h2>v4.2.1</h2>\n<ul>\n<li>Check out other refs/* by commit if provided, fall back to ref by <a\nhref=\"https://github.com/orhantoy\"><code>@​orhantoy</code></a> in <a\nhref=\"https://redirect.github.com/actions/checkout/pull/1924\">actions/checkout#1924</a></li>\n</ul>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0\"><code>9c091bb</code></a>\nupdate error wording (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2467\">#2467</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/1044a6dea927916f2c38ba5aeffbc0a847b1221a\"><code>1044a6d</code></a>\ngetting ready for checkout v7 release (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2464\">#2464</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/f0282184c7ce73ab54c7e4ab5a617122602e575f\"><code>f028218</code></a>\nBump the minor-npm-dependencies group across 1 directory with 3 updates\n(<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2462\">#2462</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/d914b262ffc244530a203ab40decab34c3abf34d\"><code>d914b26</code></a>\nupgrade module to esm and update dependencies (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2463\">#2463</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/537c7ef99cef6e5ddb5e7ff5d16d14510503801d\"><code>537c7ef</code></a>\nBump <code>@​actions/core</code> and <code>@​actions/tool-cache</code>\nand Remove uuid (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2459\">#2459</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/130a169078a413d3a5246a393625e8e742f387f6\"><code>130a169</code></a>\nBump js-yaml from 4.1.0 to 4.2.0 (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2461\">#2461</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/7d09575332117a40b46e5e020664df234cd416f3\"><code>7d09575</code></a>\nBump flatted from 3.3.1 to 3.4.2 (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2460\">#2460</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/0f9f3aa320cb53abeb534aeb54048075d9697a0e\"><code>0f9f3aa</code></a>\nBump actions/publish-immutable-action (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2458\">#2458</a>)</li>\n<li><a\nhref=\"https://github.com/actions/checkout/commit/f9e715a95fcd1f9253f77dd28f11e88d2d6460c7\"><code>f9e715a</code></a>\nblock checking out fork pr for pull_request_target and workflow_run (<a\nhref=\"https://redirect.github.com/actions/checkout/issues/2454\">#2454</a>)</li>\n<li>See full diff in <a\nhref=\"https://github.com/actions/checkout/compare/df4cb1c069e1874edd31b4311f1884172cec0e10...9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `actions/download-artifact` from 7.0.0 to 8.0.1\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/actions/download-artifact/releases\">actions/download-artifact's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v8.0.1</h2>\n<h2>What's Changed</h2>\n<ul>\n<li>Support for CJK characters in the artifact name by <a\nhref=\"https://github.com/danwkennedy\"><code>@​danwkennedy</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/download-artifact/pull/471\">actions/download-artifact#471</a></li>\n<li>Add a regression test for artifact name + content-type mismatches by\n<a href=\"https://github.com/danwkennedy\"><code>@​danwkennedy</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/download-artifact/pull/472\">actions/download-artifact#472</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/actions/download-artifact/compare/v8...v8.0.1\">https://github.com/actions/download-artifact/compare/v8...v8.0.1</a></p>\n<h2>v8.0.0</h2>\n<h2>v8 - What's new</h2>\n<blockquote>\n<p>[!IMPORTANT]\nactions/download-artifact@v8 has been migrated to an ESM module. This\nshould be transparent to the caller but forks might need to make\nsignificant changes.</p>\n</blockquote>\n<blockquote>\n<p>[!IMPORTANT]\nHash mismatches will now error by default. Users can override this\nbehavior with a setting change (see below).</p>\n</blockquote>\n<h3>Direct downloads</h3>\n<p>To support direct uploads in <code>actions/upload-artifact</code>,\nthe action will no longer attempt to unzip all downloaded files.\nInstead, the action checks the <code>Content-Type</code> header ahead of\nunzipping and skips non-zipped files. Callers wishing to download a\nzipped file as-is can also set the new <code>skip-decompress</code>\nparameter to <code>true</code>.</p>\n<h3>Enforced checks (breaking)</h3>\n<p>A previous release introduced digest checks on the download. If a\ndownload hash didn't match the expected hash from the server, the action\nwould log a warning. Callers can now configure the behavior on mismatch\nwith the <code>digest-mismatch</code> parameter. To be secure by\ndefault, we are now defaulting the behavior to <code>error</code> which\nwill fail the workflow run.</p>\n<h3>ESM</h3>\n<p>To support new versions of the @actions/* packages, we've upgraded\nthe package to ESM.</p>\n<h2>What's Changed</h2>\n<ul>\n<li>Don't attempt to un-zip non-zipped downloads by <a\nhref=\"https://github.com/danwkennedy\"><code>@​danwkennedy</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/download-artifact/pull/460\">actions/download-artifact#460</a></li>\n<li>Add a setting to specify what to do on hash mismatch and default it\nto <code>error</code> by <a\nhref=\"https://github.com/danwkennedy\"><code>@​danwkennedy</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/download-artifact/pull/461\">actions/download-artifact#461</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/actions/download-artifact/compare/v7...v8.0.0\">https://github.com/actions/download-artifact/compare/v7...v8.0.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c\"><code>3e5f45b</code></a>\nAdd regression tests for CJK characters (<a\nhref=\"https://redirect.github.com/actions/download-artifact/issues/471\">#471</a>)</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/e6d03f67377d4412c7aa56a8e2e4988e6ec479dd\"><code>e6d03f6</code></a>\nAdd a regression test for artifact name + content-type mismatches (<a\nhref=\"https://redirect.github.com/actions/download-artifact/issues/472\">#472</a>)</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3\"><code>70fc10c</code></a>\nMerge pull request <a\nhref=\"https://redirect.github.com/actions/download-artifact/issues/461\">#461</a>\nfrom actions/danwkennedy/digest-mismatch-behavior</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/f258da9a506b755b84a09a531814700b86ccfc62\"><code>f258da9</code></a>\nAdd change docs</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/ccc058e5fbb0bb2352213eaec3491e117cbc4a5c\"><code>ccc058e</code></a>\nFix linting issues</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/bd7976ba57ecea96e6f3df575eb922d11a12a9fd\"><code>bd7976b</code></a>\nAdd a setting to specify what to do on hash mismatch and default it to\n<code>error</code></li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/ac21fcf45e0aaee541c0f7030558bdad38d77d6c\"><code>ac21fcf</code></a>\nMerge pull request <a\nhref=\"https://redirect.github.com/actions/download-artifact/issues/460\">#460</a>\nfrom actions/danwkennedy/download-no-unzip</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/15999bff51058bc7c19b50ebbba518eaef7c26c0\"><code>15999bf</code></a>\nAdd note about package bumps</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/974686ed5098c7f9c9289ec946b9058e496a2561\"><code>974686e</code></a>\nBump the version to <code>v8</code> and add release notes</li>\n<li><a\nhref=\"https://github.com/actions/download-artifact/commit/fbe48b1d2756394be4cd4358ed3bc1343b330e75\"><code>fbe48b1</code></a>\nUpdate test names to make it clearer what they do</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/actions/download-artifact/compare/37930b1c2abaa49bbe596cd826c3c89aef350131...3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `taiki-e/install-action` from 2.81.11 to 2.82.2\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/releases\">taiki-e/install-action's\nreleases</a>.</em></p>\n<blockquote>\n<h2>2.82.2</h2>\n<ul>\n<li>\n<p>Update <code>xh@latest</code> to 0.26.1.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.23.</p>\n</li>\n<li>\n<p>Update <code>trivy@latest</code> to 0.71.2.</p>\n</li>\n<li>\n<p>Update <code>sccache@latest</code> to 0.16.0.</p>\n</li>\n</ul>\n<h2>2.82.1</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.4.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.22.</p>\n</li>\n<li>\n<p>Update <code>osv-scanner@latest</code> to 2.4.0.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.11.</p>\n</li>\n<li>\n<p>Update <code>martin@latest</code> to 1.11.0.</p>\n</li>\n<li>\n<p>Update <code>just@latest</code> to 1.53.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-zigbuild@latest</code> to 0.23.0.</p>\n</li>\n</ul>\n<h2>2.82.0</h2>\n<ul>\n<li>\n<p>Support <code>cargo-vet</code>. (<a\nhref=\"https://redirect.github.com/taiki-e/install-action/pull/1908\">#1908</a>,\nthanks <a\nhref=\"https://github.com/jakewimmer\"><code>@​jakewimmer</code></a>)</p>\n</li>\n<li>\n<p>Support <code>cargo-crap</code>. (<a\nhref=\"https://redirect.github.com/taiki-e/install-action/pull/1905\">#1905</a>,\nthanks <a\nhref=\"https://github.com/BartoszCiesla\"><code>@​BartoszCiesla</code></a>)</p>\n</li>\n<li>\n<p>Support <code>cargo-leptos</code>. (<a\nhref=\"https://redirect.github.com/taiki-e/install-action/pull/1903\">#1903</a>,\nthanks <a\nhref=\"https://github.com/404Simon\"><code>@​404Simon</code></a>)</p>\n</li>\n<li>\n<p>Update <code>kingfisher@latest</code> to 1.103.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-xwin@latest</code> to 0.23.0.</p>\n</li>\n<li>\n<p>Update <code>wasmtime@latest</code> to 45.0.2.</p>\n</li>\n<li>\n<p>Update <code>cargo-deny@latest</code> to 0.19.9.</p>\n</li>\n<li>\n<p>Update <code>prek@latest</code> to 0.4.5.</p>\n</li>\n<li>\n<p>Update <code>trivy@latest</code> to 0.71.1.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.10.</p>\n</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md\">taiki-e/install-action's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>Changelog</h1>\n<p>All notable changes to this project will be documented in this\nfile.</p>\n<p>This project adheres to <a href=\"https://semver.org\">Semantic\nVersioning</a>.</p>\n<!-- raw HTML omitted -->\n<h2>[Unreleased]</h2>\n<h2>[2.82.6] - 2026-06-29</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.7.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.25.</p>\n</li>\n<li>\n<p>Update <code>syft@latest</code> to 1.46.0.</p>\n</li>\n<li>\n<p>Update <code>dprint@latest</code> to 0.55.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-auditable@latest</code> to 0.7.5.</p>\n</li>\n</ul>\n<h2>[2.82.5] - 2026-06-26</h2>\n<ul>\n<li>\n<p>Update <code>wasmtime@latest</code> to 46.0.1.</p>\n</li>\n<li>\n<p>Update <code>wasm-bindgen@latest</code> to 0.2.126.</p>\n</li>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.6.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.14.</p>\n</li>\n<li>\n<p>Update <code>cargo-rdme@latest</code> to 2.1.0.</p>\n</li>\n</ul>\n<h2>[2.82.4] - 2026-06-25</h2>\n<ul>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.24.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.13.</p>\n</li>\n<li>\n<p>Update <code>just@latest</code> to 1.54.0.</p>\n</li>\n<li>\n<p>Update <code>biome@latest</code> to 2.5.1.</p>\n</li>\n</ul>\n<h2>[2.82.3] - 2026-06-24</h2>\n<ul>\n<li>Update <code>zizmor@latest</code> to 1.26.1.</li>\n</ul>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/9e1e5806d4a4822de933115878265be9aaa786d9\"><code>9e1e580</code></a>\nRelease 2.82.2</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/788896b6163ee187bf02f51161e573dbc028dba0\"><code>788896b</code></a>\nUpdate zizmor manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/7631577b669c94d73e58cb9f217d0f56abd33a48\"><code>7631577</code></a>\nUpdate <code>xh@latest</code> to 0.26.1</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/e0f1a05cc9f43f4fd57d15fe7ee20ca5b78f65fc\"><code>e0f1a05</code></a>\nUpdate <code>uv@latest</code> to 0.11.23</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/3cda1e20d17fde3f9958653d219495e0591233ec\"><code>3cda1e2</code></a>\nUpdate <code>trivy@latest</code> to 0.71.2</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/11ac3210af3c20497864c3d27d4499b6a7108098\"><code>11ac321</code></a>\nUpdate tombi manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/b5f9e335d3eaba5117342d9d9fda485aea29c524\"><code>b5f9e33</code></a>\nUpdate <code>sccache@latest</code> to 0.16.0</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/4e48cd5f5170589da6cad450be3e62ca61534cc1\"><code>4e48cd5</code></a>\nUpdate cargo-tarpaulin manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/e0a923573389b28b6cefdbe8309332b471f55583\"><code>e0a9235</code></a>\nUpdate cargo-rdme manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/8b3c737da4b541bf0fb5a3e0488ff20535badac9\"><code>8b3c737</code></a>\nRelease 2.82.1</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/taiki-e/install-action/compare/15449e3094499af05d8d964a1c884208e4b8b595...9e1e5806d4a4822de933115878265be9aaa786d9\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate GitHub Actions used in our workflows to the latest majors for\nsecurity and reliability: `actions/checkout` v7,\n`actions/download-artifact` v8, and `taiki-e/install-action` v2.82.2.\nThis brings ESM support and stricter artifact integrity checks.\n\n- **Dependencies**\n- `actions/checkout` v7: security hardening; blocks checking out fork\nPRs for `pull_request_target`/`workflow_run`.\n- `actions/download-artifact` v8.0.1: migrated to ESM; stops unzipping\nnon-zip files; hash mismatches now error by default; supports\n`skip-decompress` and `digest-mismatch` inputs.\n- `taiki-e/install-action` v2.82.2: refreshed tool manifests; no\nbreaking changes.\n\n- **Migration**\n- If any job needs zipped artifacts without auto-unzip, set\n`skip-decompress: true`.\n- If existing artifacts trigger hash mismatch failures, either\nregenerate the artifacts or set `digest-mismatch: warn` to keep prior\nbehavior.\n\n<sup>Written for commit c79f5e4ec1ee0dbbc611ff89bfadf59b9b087343.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1022?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-29T10:16:29-07:00",
          "tree_id": "cce2d32bcc90c5d601a9700b3aaf01c93c946647",
          "url": "https://github.com/andymai/brepkit/commit/74da44bc4213935dbb7ebaeabda4ec992ddf9ca3"
        },
        "date": 1782753516414,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 738987,
            "range": "± 7251",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 828842,
            "range": "± 1353",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12029,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 625204,
            "range": "± 1226",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19075998,
            "range": "± 83144",
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
          "id": "262676d5e280a8dbf0947bac8fb6d9f0fd6f0aba",
          "message": "fix(deps): upgrade quick-xml to 0.41 for RUSTSEC-2026-0194/0195 (#1024)\n\n## Why\n\nTwo DoS advisories landed against `quick-xml` on untrusted XML input,\nfailing Cargo Deny and Security Audit on `main` and on every open PR:\n\n- [RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194):\nquadratic time checking a start tag for duplicate attribute names.\n- [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195):\nunbounded namespace-declaration allocation in `NsReader`.\n\nBoth are fixed in `quick-xml >= 0.41.0`.\n\n## What\n\n- Bump the direct dependency `quick-xml` from `0.40` to `0.41` (root\n`Cargo.toml`). The only path that parses untrusted input is the 3MF\nreader in `brepkit-io`; it and the 3MF writer build and pass unchanged\nagainst 0.41.\n- Scope a documented advisory ignore in `deny.toml` for the residual\n`quick-xml 0.39.4`, which enters only through `wayland-scanner` (a\nbuild-time proc-macro under `winit`, behind `brepkit-render`'s `window`\nfeature). It generates code from vendored, trusted Wayland protocol XML\nat compile time and never touches runtime input, and it is pinned to\n`^0.39` upstream so it cannot advance yet. The ignore carries a re-check\ntrigger for when the `winit` chain moves off `quick-xml 0.39`.\n\n## Verification\n\n- `cargo build -p brepkit-io` and `cargo test -p brepkit-io` pass\nagainst 0.41.\n- `cargo deny --all-features check` reports `advisories ok, bans ok,\nlicenses ok, sources ok` locally (all-features reproduces the CI\naction's graph breadth, which includes the window/wayland path).\n\n## Note\n\nUnblocks the repo's security checks, which are currently red on `main`\nindependent of any single PR (including the docs-only #1023).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpgrade `quick-xml` to 0.41 to patch `RUSTSEC-2026-0194`/`0195` DoS\nissues and restore passing security checks. 3MF parsing/writing in\n`brepkit-io` builds and tests unchanged.\n\n- **Dependencies**\n- Add scoped `deny.toml` ignores for `RUSTSEC-2026-0194` and\n`RUSTSEC-2026-0195` on the residual `quick-xml 0.39.4` via\n`wayland-scanner` under `winit` (`brepkit-render` `window` feature);\nbuild-time codegen only. Taplo-formatted and includes a re-check note\nfor when the chain updates.\n- Mirror the same ignores in `.cargo/audit.toml` so the Security Audit\njob (`cargo-audit`) also passes.\n\n<sup>Written for commit 89f7898476290ef41e39ba6e224917f903418381.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1024?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-02T13:33:49-07:00",
          "tree_id": "e16e7b50d51e0e32c50a290c084a08757b8c47ae",
          "url": "https://github.com/andymai/brepkit/commit/262676d5e280a8dbf0947bac8fb6d9f0fd6f0aba"
        },
        "date": 1783024557981,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 741461,
            "range": "± 983",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 830130,
            "range": "± 1013",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12076,
            "range": "± 785",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 629009,
            "range": "± 13190",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19043634,
            "range": "± 20814",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1813c64a575876a247c4c1d9130496d52149528d",
          "message": "chore(main): release 2.124.1 (#1025)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.1](https://github.com/andymai/brepkit/compare/v2.124.0...v2.124.1)\n(2026-07-02)\n\n\n### Bug Fixes\n\n* **deps:** upgrade quick-xml to 0.41 for RUSTSEC-2026-0194/0195\n([#1024](https://github.com/andymai/brepkit/issues/1024))\n([262676d](https://github.com/andymai/brepkit/commit/262676d5e280a8dbf0947bac8fb6d9f0fd6f0aba))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release for `brepkit-wasm` upgrading `quick-xml` to 0.41 to\naddress RUSTSEC-2026-0194 and RUSTSEC-2026-0195. This resolves the\nadvisories and bumps the crate to version 2.124.1.\n\n<sup>Written for commit 40026c044589b93743b1d41a883767a067ff2f2e.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1025?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-02T20:39:26Z",
          "tree_id": "e38a3896bb78c62371ba84b1e674b41007268e9d",
          "url": "https://github.com/andymai/brepkit/commit/1813c64a575876a247c4c1d9130496d52149528d"
        },
        "date": 1783024893502,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 740563,
            "range": "± 1672",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 827180,
            "range": "± 1612",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11885,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 627705,
            "range": "± 1092",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18978613,
            "range": "± 19994",
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
          "id": "f57997351b7540f0115f5ef0e3de58b133c31171",
          "message": "docs(skills): add engineering skill library under .claude/skills (#1023)\n\n## What\n\nEighteen skills under `.claude/skills/`, each capturing the method and\ntraps for one recurring class of work, so engineers and smaller models\ncan debug, extend, validate, benchmark, and ship this kernel without\nprior project context. `README.md` indexes them with a suggested reading\norder and a glossary.\n\n**Doctrine and verification:** `debugging-doctrine`,\n`solid-verification`, `numerical-robustness`, `testing`.\n**Engine internals:** `boolean-debugging`, `analytic-preservation`,\n`tessellation`, `fillet-blend`.\n**Building:** `layer-boundaries`, `add-operation`, `wasm-bindings`,\n`render-verify`, `io-formats`.\n**Shipping:** `pr-workflow`, `profiling`, `parity-benchmarking`,\n`release-flow`.\n**Work selection:** `roadmap`, a living index of open, deferred, and\nterminal cases with the chase filters and the acceptance bar.\n\n## How it was built\n\nEach skill went through research, authoring, adversarial verification,\nand fix stages, then cross-skill consistency and coverage passes. Every\nfile path, symbol, and command was checked against the current tree.\nSeveral beliefs carried in from earlier work were found stale and\ncorrected in place (tooling that now exists, a heal function that is\nimplemented rather than a stub, a deprecation already removed).\n\n## Scope\n\nDocumentation only. No source, test, or build changes. The `roadmap`\nskill declares a maintenance contract: sessions that close, defer, or\ndiscover a work item update it in the same PR.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds the brepkit engineering skill library under `.claude/skills` with\n18 task-focused guides and references so contributors can debug, extend,\nverify, benchmark, and ship the kernel. Docs-only; includes an indexed\n`README.md` and a maintenance rule for the `roadmap` skill.\n\n- **New Features**\n- Eighteen skill guides across doctrine/verification, engine internals,\nbuilding, shipping, and work selection, each with `SKILL.md` +\n`reference.md` verified against current symbols, paths, and commands.\n  - `README.md` index with suggested reading order and glossary.\n- `roadmap` adds a maintenance contract: sessions that\nclose/defer/discover work must update it in the same PR.\n\n<sup>Written for commit 3f6b64818afdb31a27b07edef926fc63b9ba623b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1023?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-02T13:49:26-07:00",
          "tree_id": "a20266c4af14b49d2380cbf76a670477746538b4",
          "url": "https://github.com/andymai/brepkit/commit/f57997351b7540f0115f5ef0e3de58b133c31171"
        },
        "date": 1783025496367,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 736584,
            "range": "± 984",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 824550,
            "range": "± 2728",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12019,
            "range": "± 589",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 623327,
            "range": "± 15568",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19114121,
            "range": "± 54773",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "168ac6e4d2808c59b4e259b903f70cbce66b6703",
          "message": "chore(deps): bump taiki-e/install-action from 2.82.2 to 2.82.6 in the actions group (#1028)\n\nBumps the actions group with 1 update:\n[taiki-e/install-action](https://github.com/taiki-e/install-action).\n\nUpdates `taiki-e/install-action` from 2.82.2 to 2.82.6\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/releases\">taiki-e/install-action's\nreleases</a>.</em></p>\n<blockquote>\n<h2>2.82.6</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.7.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.25.</p>\n</li>\n<li>\n<p>Update <code>syft@latest</code> to 1.46.0.</p>\n</li>\n<li>\n<p>Update <code>dprint@latest</code> to 0.55.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-auditable@latest</code> to 0.7.5.</p>\n</li>\n</ul>\n<h2>2.82.5</h2>\n<ul>\n<li>\n<p>Update <code>wasmtime@latest</code> to 46.0.1.</p>\n</li>\n<li>\n<p>Update <code>wasm-bindgen@latest</code> to 0.2.126.</p>\n</li>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.6.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.14.</p>\n</li>\n<li>\n<p>Update <code>cargo-rdme@latest</code> to 2.1.0.</p>\n</li>\n</ul>\n<h2>2.82.4</h2>\n<ul>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.24.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.13.</p>\n</li>\n<li>\n<p>Update <code>just@latest</code> to 1.54.0.</p>\n</li>\n<li>\n<p>Update <code>biome@latest</code> to 2.5.1.</p>\n</li>\n</ul>\n<h2>2.82.3</h2>\n<ul>\n<li>\n<p>Update <code>zizmor@latest</code> to 1.26.1.</p>\n</li>\n<li>\n<p>Update <code>wasmtime@latest</code> to 46.0.0.</p>\n</li>\n<li>\n<p>Update <code>tombi@latest</code> to 1.1.5.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.6.12.</p>\n</li>\n<li>\n<p>Update <code>kingfisher@latest</code> to 1.104.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-tarpaulin@latest</code> to 0.35.5.</p>\n</li>\n<li>\n<p>Update <code>cargo-nextest@latest</code> to 0.9.138.</p>\n</li>\n<li>\n<p>Update <code>cargo-crap@latest</code> to 0.3.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-binstall@latest</code> to 1.20.1.</p>\n</li>\n</ul>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md\">taiki-e/install-action's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>Changelog</h1>\n<p>All notable changes to this project will be documented in this\nfile.</p>\n<p>This project adheres to <a href=\"https://semver.org\">Semantic\nVersioning</a>.</p>\n<!-- raw HTML omitted -->\n<h2>[Unreleased]</h2>\n<h2>[2.82.9] - 2026-07-05</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.9.</p>\n</li>\n<li>\n<p>Update <code>prek@latest</code> to 0.4.8.</p>\n</li>\n<li>\n<p>Update <code>cargo-tarpaulin@latest</code> to 0.37.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-leptos@latest</code> to 0.3.7.</p>\n</li>\n</ul>\n<h2>[2.82.8] - 2026-07-03</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.8.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.26.</p>\n</li>\n<li>\n<p>Update <code>typos@latest</code> to 1.48.0.</p>\n</li>\n<li>\n<p>Update <code>trivy@latest</code> to 0.72.0.</p>\n</li>\n<li>\n<p>Update <code>tombi@latest</code> to 1.1.7.</p>\n</li>\n<li>\n<p>Update <code>prek@latest</code> to 0.4.6.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.0.</p>\n</li>\n<li>\n<p>Update <code>just@latest</code> to 1.55.1.</p>\n</li>\n<li>\n<p>Update <code>biome@latest</code> to 2.5.2.</p>\n</li>\n</ul>\n<h2>[2.82.7] - 2026-06-30</h2>\n<ul>\n<li>\n<p>Update <code>tombi@latest</code> to 1.1.6.</p>\n</li>\n<li>\n<p>Update <code>kingfisher@latest</code> to 1.105.0.</p>\n</li>\n<li>\n<p>Update <code>gungraun-runner@latest</code> to 0.19.3.</p>\n</li>\n</ul>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/9bcaee1dcae34154180f412e2fa69355a7cda9f6\"><code>9bcaee1</code></a>\nRelease 2.82.6</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/e43cd7ce2e2c5a6d08d652405a0ead8cda7bf2db\"><code>e43cd7c</code></a>\nUpdate <code>vacuum@latest</code> to 0.29.7</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/3761407ad0b5e4e5b42ccb62e48ffa2712e94703\"><code>3761407</code></a>\nUpdate <code>uv@latest</code> to 0.11.25</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/cb6ad0dba1ff078e589e75ae88bb03120688feef\"><code>cb6ad0d</code></a>\nUpdate tombi manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/6022671c93f20efad71663e3adc06d63e7f1ec8a\"><code>6022671</code></a>\nUpdate <code>syft@latest</code> to 1.46.0</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/ab5aae8354703341b0939f4e6c1bb66372b446db\"><code>ab5aae8</code></a>\nUpdate gungraun-runner manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/867613b49949fba1c8fe194d323c7c51b77b24d2\"><code>867613b</code></a>\nUpdate editorconfig-checker manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/c5837ef63439ac9ebc0ecce97e23cca6a4a5aac4\"><code>c5837ef</code></a>\nUpdate <code>dprint@latest</code> to 0.55.0</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/7d41d74582778d0345db89bc1054e86f3802b6d2\"><code>7d41d74</code></a>\nUpdate <code>cargo-auditable@latest</code> to 0.7.5</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/bffeee26d4db9be238a4ea78d8826604ebcb594d\"><code>bffeee2</code></a>\nRelease 2.82.5</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/taiki-e/install-action/compare/9e1e5806d4a4822de933115878265be9aaa786d9...9bcaee1dcae34154180f412e2fa69355a7cda9f6\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\n[![Dependabot compatibility\nscore](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=taiki-e/install-action&package-manager=github_actions&previous-version=2.82.2&new-version=2.82.6)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpgrade `taiki-e/install-action` from 2.82.2 to 2.82.6 across CI\nworkflows to keep tool installers current and pull in patch fixes.\nUpdated `.github/workflows/ci.yml`, `mutants.yml`, and `publish.yml`; no\nbehavior changes expected.\n\n<sup>Written for commit dea109b8bab59e0fc110c199896d78f0855132a7.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1028?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T07:11:27-07:00",
          "tree_id": "01a7468b2e6822bef00c78b297faa4b92025fe1e",
          "url": "https://github.com/andymai/brepkit/commit/168ac6e4d2808c59b4e259b903f70cbce66b6703"
        },
        "date": 1783433642327,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 735872,
            "range": "± 4593",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 824169,
            "range": "± 6968",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11878,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 625586,
            "range": "± 20790",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19007288,
            "range": "± 134558",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be8ef9d212d0012aa7abf8dd75af51aafa00ed7b",
          "message": "chore(deps-dev): bump the npm group with 3 updates (#1026)\n\nBumps the npm group with 3 updates:\n[@commitlint/cli](https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli),\n[@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/config-conventional)\nand [prettier](https://github.com/prettier/prettier).\n\nUpdates `@commitlint/cli` from 21.0.2 to 21.1.0\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/releases\">@​commitlint/cli's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v21.1.0</h2>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">21.1.0</a>\n(2026-06-23)</h1>\n<h3>Bug Fixes</h3>\n<ul>\n<li>fix: remove duplicate es-toolkit@1.47.1 keys from lockfile by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4833\">conventional-changelog/commitlint#4833</a></li>\n</ul>\n<h3>Features</h3>\n<ul>\n<li>feat(cli): add --default-config flag to lint without a config file\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4805\">conventional-changelog/commitlint#4805</a></li>\n<li>feat(lint): allow for custom commit parser function by <a\nhref=\"https://github.com/esatterwhite\"><code>@​esatterwhite</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4829\">conventional-changelog/commitlint#4829</a></li>\n</ul>\n<h3>Docs, chore, etc.</h3>\n<ul>\n<li>test(cli): verify --cwd redirects config resolution <a\nhref=\"https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli/issues/997\">#997</a>\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4796\">conventional-changelog/commitlint#4796</a></li>\n<li>docs: add ai agent support (skill, guide, llms.txt) by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4804\">conventional-changelog/commitlint#4804</a></li>\n<li>docs: add TDD commit flow and review expectations for contributors\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4836\">conventional-changelog/commitlint#4836</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a\nhref=\"https://github.com/esatterwhite\"><code>@​esatterwhite</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4829\">conventional-changelog/commitlint#4829</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/blob/master/@commitlint/cli/CHANGELOG.md\">@​commitlint/cli's\nchangelog</a>.</em></p>\n<blockquote>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">21.1.0</a>\n(2026-06-23)</h1>\n<h3>Features</h3>\n<ul>\n<li><strong>cli:</strong> add --default-config flag to lint without a\nconfig file (<a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/issues/4805\">#4805</a>)\n(<a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/7af27ba1bcfe2347d02df2efd6dca7203b6768c5\">7af27ba</a>),\ncloses <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/issues/3662\">#3662</a></li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/6f1c0af6a7631cee854587dafc6beb0ccf274b1e\"><code>6f1c0af</code></a>\nv21.1.0</li>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/7af27ba1bcfe2347d02df2efd6dca7203b6768c5\"><code>7af27ba</code></a>\nfeat(cli): add --default-config flag to lint without a config file (<a\nhref=\"https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli/issues/4805\">#4805</a>)</li>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/51a2d7f233338d1f09ffb93d70c7658ee78720df\"><code>51a2d7f</code></a>\ntest(cli): verify --cwd redirects config resolution (<a\nhref=\"https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli/issues/997\">#997</a>)\n(<a\nhref=\"https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli/issues/4796\">#4796</a>)</li>\n<li>See full diff in <a\nhref=\"https://github.com/conventional-changelog/commitlint/commits/v21.1.0/@commitlint/cli\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `@commitlint/config-conventional` from 21.0.2 to 21.2.0\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/releases\">@​commitlint/config-conventional's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v21.2.0</h2>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">21.2.0</a>\n(2026-06-30)</h1>\n<h3>Features</h3>\n<ul>\n<li>feat(resolve-extends): resolve pure-ESM presets\n(conventional-changelog v7/v9/v10) by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4859\">conventional-changelog/commitlint#4859</a></li>\n</ul>\n<h3>Chore</h3>\n<ul>\n<li>ci: install git in stock-Ubuntu baseline job by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4847\">conventional-changelog/commitlint#4847</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0</a></p>\n<h2>v21.1.0</h2>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">21.1.0</a>\n(2026-06-23)</h1>\n<h3>Bug Fixes</h3>\n<ul>\n<li>fix: remove duplicate es-toolkit@1.47.1 keys from lockfile by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4833\">conventional-changelog/commitlint#4833</a></li>\n</ul>\n<h3>Features</h3>\n<ul>\n<li>feat(cli): add --default-config flag to lint without a config file\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4805\">conventional-changelog/commitlint#4805</a></li>\n<li>feat(lint): allow for custom commit parser function by <a\nhref=\"https://github.com/esatterwhite\"><code>@​esatterwhite</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4829\">conventional-changelog/commitlint#4829</a></li>\n</ul>\n<h3>Docs, chore, etc.</h3>\n<ul>\n<li>test(cli): verify --cwd redirects config resolution <a\nhref=\"https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/config-conventional/issues/997\">#997</a>\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4796\">conventional-changelog/commitlint#4796</a></li>\n<li>docs: add ai agent support (skill, guide, llms.txt) by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4804\">conventional-changelog/commitlint#4804</a></li>\n<li>docs: add TDD commit flow and review expectations for contributors\nby <a href=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4836\">conventional-changelog/commitlint#4836</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a\nhref=\"https://github.com/esatterwhite\"><code>@​esatterwhite</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4829\">conventional-changelog/commitlint#4829</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/blob/master/@commitlint/config-conventional/CHANGELOG.md\">@​commitlint/config-conventional's\nchangelog</a>.</em></p>\n<blockquote>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">21.2.0</a>\n(2026-06-30)</h1>\n<h3>Features</h3>\n<ul>\n<li><strong>resolve-extends:</strong> resolve pure-ESM presets\n(conventional-changelog v7/v9/v10) (<a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/issues/4859\">#4859</a>)\n(<a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/fdb566fe59457a786eac80e2a8cbb994638daba0\">fdb566f</a>)</li>\n</ul>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.0.2...v21.1.0\">21.1.0</a>\n(2026-06-23)</h1>\n<p><strong>Note:</strong> Version bump only for package\n<code>@​commitlint/config-conventional</code></p>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/1b4e5bc0e095294ad421b3ac83f6b66665429e60\"><code>1b4e5bc</code></a>\nv21.2.0</li>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/fdb566fe59457a786eac80e2a8cbb994638daba0\"><code>fdb566f</code></a>\nfeat(resolve-extends): resolve pure-ESM presets (conventional-changelog\nv7/v9...</li>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/6f1c0af6a7631cee854587dafc6beb0ccf274b1e\"><code>6f1c0af</code></a>\nv21.1.0</li>\n<li>See full diff in <a\nhref=\"https://github.com/conventional-changelog/commitlint/commits/v21.2.0/@commitlint/config-conventional\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `prettier` from 3.8.4 to 3.9.1\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/releases\">prettier's\nreleases</a>.</em></p>\n<blockquote>\n<h2>3.9.1</h2>\n<ul>\n<li>CLI: Fix ignored file has been cached incorrectly (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19483\">#19483</a>\nby <a href=\"https://github.com/kovsu\"><code>@​kovsu</code></a>)</li>\n</ul>\n<p>🔗 <a\nhref=\"https://github.com/prettier/prettier/blob/3.9.1/CHANGELOG.md#391\">Changelog</a></p>\n<h2>3.9.0</h2>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.8.5...3.9.0\">diff</a></p>\n<p>🔗 <a href=\"https://prettier.io/blog/2026/06/27/3.9.0\">Prettier 3.9:\nMajor parser upgrades and Formatting improvements</a></p>\n<h2>3.8.5</h2>\n<ul>\n<li>Fix Flow variance annotation print (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19022\">#19022</a>\nby <a\nhref=\"https://github.com/marcoww6\"><code>@​marcoww6</code></a>)</li>\n</ul>\n<p>🔗 <a\nhref=\"https://github.com/prettier/prettier/blob/3.8.5/CHANGELOG.md#385\">Changelog</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/blob/main/CHANGELOG.md\">prettier's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>3.9.1</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.9.0...3.9.1\">diff</a></p>\n<h4>CLI: Fix ignored file has been cached incorrectly (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19483\">#19483</a>\nby <a href=\"https://github.com/kovsu\"><code>@​kovsu</code></a>)</h4>\n<p>Bug details <a\nhref=\"https://redirect.github.com/prettier/prettier/issues/18016\">prettier/prettier#18016</a></p>\n<h1>3.9.0</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.8.5...3.9.0\">diff</a></p>\n<p>🔗 <a href=\"https://prettier.io/blog/2026/06/27/3.9.0\">Release\nNotes</a></p>\n<h1>3.8.5</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.8.4...3.8.5\">diff</a></p>\n<h4>Flow: Support <code>readonly</code> as a variance annotation (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19022\">#19022</a>\nby <a\nhref=\"https://github.com/marcoww6\"><code>@​marcoww6</code></a>)</h4>\n<p>Flow now accepts <code>readonly</code> as a property variance\nannotation, equivalent to <code>+</code> (covariant/read-only).</p>\n<!-- raw HTML omitted -->\n<pre lang=\"jsx\"><code>// Input\ntype T = {\n  readonly foo: string,\n};\n<p>// Prettier 3.8.4<br />\nSyntaxError</p>\n<p>// Prettier 3.8.5<br />\ntype T = {<br />\nreadonly foo: string,<br />\n};<br />\n</code></pre></p>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/c47654c003fe525572e10d5cc1ea64d7b9c0ee55\"><code>c47654c</code></a>\nRelease 3.9.1</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/06159aa254e662514d1c6f4de13fbac805984232\"><code>06159aa</code></a>\nFix bug in release script</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/4bc5ab40582921f5283af4ff5d6511b58d25ec00\"><code>4bc5ab4</code></a>\nUpdate file-entry-cache to 11.1.5 (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19483\">#19483</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/b7fd58bba027282038ad312af0522d4598e8b769\"><code>b7fd58b</code></a>\nRelease <code>@prettier/plugin-oxc@0.2.0</code> and\n<code>@prettier/plugin-hermes@0.2.0</code></li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/3006400fc2560e297b54d82c58cbc331ec87902c\"><code>3006400</code></a>\nRevert changes in release script</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/7bef7dba7e99423e8f781228e8a73163f26ca9e9\"><code>7bef7db</code></a>\nGit blame ignore 3.9.0</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/bb817b1bd04c04e0a8d89cb52c256a38e17fd0f5\"><code>bb817b1</code></a>\nBump Prettier dependency to 3.9.0</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/05cf896cfcc0890f58790c380f3da1d98872d071\"><code>05cf896</code></a>\nClean changelog_unreleased</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/79f6cdfd9873a91be9b25c9c6a41d26dcd9a6656\"><code>79f6cdf</code></a>\nDisable finished steps</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/3613b1e5a309f5b4a74acf9436946a77e4dddf69\"><code>3613b1e</code></a>\nAdd blog post for v3.9 (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19414\">#19414</a>)</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/prettier/prettier/compare/3.8.4...3.9.1\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdates dev tooling to keep linting and formatting current and reliable.\nBumps `@commitlint` packages for minor improvements and parser updates,\nand `prettier` to 3.9.1 to fix a CLI cache bug and refine formatting.\n\n- **Dependencies**\n- `@commitlint/cli` 21.0.2 → 21.1.0 (adds `--default-config`; no\nbreaking changes)\n- `@commitlint/config-conventional` 21.0.2 → 21.2.0 (resolves pure ESM\npresets)\n- `prettier` 3.8.4 → 3.9.1 (fixes ignored-file cache; minor formatting\ntweaks)\n\n<sup>Written for commit ee4514f49915b5eea614d95509e1ea48149dcc24.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1026?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T07:11:35-07:00",
          "tree_id": "ea8a6259bb532a0fa4eb5b603eb8380fa2bf192f",
          "url": "https://github.com/andymai/brepkit/commit/be8ef9d212d0012aa7abf8dd75af51aafa00ed7b"
        },
        "date": 1783433777171,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 729945,
            "range": "± 8035",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 814953,
            "range": "± 8346",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12010,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 622103,
            "range": "± 5226",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18753122,
            "range": "± 25983",
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
          "id": "e209d0cf3555b6cc4f0d18b13628156fc9670db9",
          "message": "fix(operations): watertight, parity-density tessellation for cylinder/cone bands (#1029)\n\n## Summary\n\nTwo tessellation defects, both root-caused from the gridfinity tool's\nliteral kernel operands (captured via the arena serializer during tool\nprobes) and pinned with committed fixtures:\n\n1. **Ruled-direction interior over-mesh (`interior_grid_resolution`)** —\ncylinder/cone faces fed `dv` (the axial span, **millimeters**) into the\nchord-deviation formula, which expects an **arc angle in radians**. A\n28mm hex-cut corner cylinder meshed at ~7700 triangles; a 2×2×4\nhoneycomb-wall bin blew up to ~63k triangles where ~4.3k is expected\n(~15×). The ruled direction has zero chord sag — two interior rows\nsuffice for CDT quality. Post-fix the same body meshes at ~2.1k\ntriangles, watertight.\n\n2. **Partial-band rim cracks at fine deflection** — non-full-revolution\ncylinder/cone bands (gridfinity socket-profile corner rings) fell\nthrough to the snap mesher, which re-samples the rim independently and\nreconciles by 1e-6 proximity. At export tolerance (0.01mm) its segment\ncount diverges from the shared edge pool's (the #696 off-by-one class),\nleaving ~200 one-sided mesh edges on the compartment cavity cut —\nnon-watertight STL exports (the `compartmentBuilder.scenario.manifold`\nfailure family in the tool). Hole-free partial bands now triangulate via\nCDT over the shared pool ids, watertight by construction. Faces **with**\ninner wires keep the snap path: this CDT does not constrain inner wires\nand would skin holes over.\n\n## Verification\n\n- Replayed the tool's captured operand chain (cavity cut + 44 sequential\nhex cuts): every step watertight (`boundary_edge_count == 0`,\n`non_manifold_edge_count == 0`) at deflections 0.01/0.03/0.1/0.5.\n- New fixtures + tests: `crates/io/tests/tessellation_parity_inmem.rs`\nasserts watertightness across all three tool quality tiers and bounds\nhex-cut body density (pre-fix 14272 triangles, post-fix ~2100, bound\n5000).\n- Full workspace suite green.\n\n## Impact\n\n- Honeycomb wall-pattern bins drop ~15× in triangle count (63k → ~3.4k\nfull body), restoring triangle-count parity.\n- Compartmented-bin STL exports become watertight at export tolerance\n(the 0/13 manifold scenario family).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFix tessellation of cylinder/cone bands to be watertight and match\nexpected triangle density. Prevents rim cracks in partial bands at fine\ntolerances and reduces over-meshing (up to ~15× fewer triangles on\nhoneycomb bins).\n\n- Bug Fixes\n- Ruled-direction grid: treat only the angular u direction as\ncurvature-driven; clamp v (rulings) to 2 interior rows. This corrects\nusing dv (mm) as radians and slashes triangle counts.\n- Partial bands: prefer CDT over shared edge ids for hole-free\nrevolution bands; fall back to the snap mesher when inner wires exist.\nEliminates rim cracks at 0.01 mm export tolerance, and drops stale\nmerge-map entries on CDT rollback to avoid invalid vertex references.\n- Tests: added captured tool operands and assertions for watertightness\nacross quality tiers and a bound on mesh density.\n\n<sup>Written for commit 7fb104278b0612050b643a91978523bf2267bbc0.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1029?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T08:46:08-07:00",
          "tree_id": "b2cf402cb989a948f1d0ca9bd410dcba2f0a0115",
          "url": "https://github.com/andymai/brepkit/commit/e209d0cf3555b6cc4f0d18b13628156fc9670db9"
        },
        "date": 1783439310252,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 728994,
            "range": "± 3179",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821907,
            "range": "± 2530",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11893,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 627718,
            "range": "± 865",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18783291,
            "range": "± 350386",
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
          "id": "e059c3bde67155a2b2e6f5d494c71409ac73ca5d",
          "message": "docs(skills): roadmap — tessellation-parity wave + fresh scenario baseline (#1031)\n\nLiving-doc maintenance required by the roadmap skill itself: records the\n2026-07-07 tessellation-parity wave (#1029, #1030), the fresh full\nscenario-matrix baseline measured against the tool, the new deferred\nrows (stacking-lip corner doubled faces = next target, tilted-divider\nresiduals, honeycomb+handles kernel-poisoning panic), and the \"capture\nbefore assuming GFA\" triage lesson with the probe-kernel recipe pointer.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdates `SKILL.md` for the roadmap skill with the 2026-07-07\ntessellation-parity wave (#1029, #1030), a fresh scenario-matrix\nbaseline, and a clear triage note with a probe-kernel capture recipe.\nRecords key results (honeycomb bins 63k→~3k triangles, cavity cuts\nexport watertight, reversed-tool-face cuts no longer invert) and adds\ndeferred rows for stacking-lip corner doubled faces, tilted-divider\nresiduals, and a honeycomb+handles kernel-poisoning panic.\n\n<sup>Written for commit 7e889315677b0d4aaf3b05a6f5c8d30987f5d73c.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1031?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T15:46:52Z",
          "tree_id": "feb857a2919de65338a80c8a025c2156a6954700",
          "url": "https://github.com/andymai/brepkit/commit/e059c3bde67155a2b2e6f5d494c71409ac73ca5d"
        },
        "date": 1783439447502,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 723743,
            "range": "± 5284",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 815360,
            "range": "± 849",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11954,
            "range": "± 608",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 623602,
            "range": "± 29228",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18789634,
            "range": "± 36388",
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
          "id": "a20df5536da8a7dda1a49c9ecc892e8271480e73",
          "message": "fix(algo): toggle orientation of flipped cut tool faces, reject open hole shells (#1030)\n\n## Summary\n\nTwo coupled assembler defects, root-caused from the gridfinity tool's\nliteral compartment-cavity operands (captured with the arena serializer,\ncommitted as fixtures):\n\n1. **Flip is a toggle, not a set.** `SelectedFace::reversed` is a flip\nrequest relative to the face's *current* orientation, but\n`build_solid_with_origins` built flipped copies with\n`Face::new_reversed` unconditionally. A tool face already stored\nreversed (e.g. a planar-NURBS extrusion side wall) came out of a Cut\n*unchanged*, its effective normal pointing INTO the result material. The\nB-Rep still paired every edge (the pairing walk is orientation-blind) so\nfree/over gates passed — but tessellation faithfully emitted those walls\nwound backwards: 12 one-sided mesh edges at every deflection, STL\nexports flagged non-manifold by slicers (the compartmented-bin export\nfamily), and mesh volume short by the inverted walls' contribution.\n\n2. **Open hole-shell fragments.** Correcting the flip surfaced a latent\ngap the buggy flip had masked: a residual coincident-wall duplicate face\nwhose corner-fan sign previously read as an *open growth fragment*\n(dropped by the existing closed-shell requirement on non-outer growth\nshells) now reads negative and became a 1-face \"inner shell\",\nover-sharing its edges against the outer shell (caught by the\n`a2hcomb_pcut1` honeycomb fixture). Hole shells now face the same\nclosed-shell requirement: a cavity boundary must be watertight in\nitself.\n\n## Verification\n\n- New fixture test `crates/io/tests/cut_reversed_tool_faces_inmem.rs`:\nreplays the captured cavity cut; asserts watertight tessellation\n(boundary=0, non-manifold=0) at deflections 0.01/0.1/0.5 **and**\nmesh-volume consistency `vol(cut) = vol(body) − vol(body ∩ tool)`\n(within 2%). Fails pre-fix (12 one-sided edges at every tier), passes\npost-fix.\n- The existing honeycomb fixture suite\n(`gridfinity_honeycomb_cut_inmem`) passes: the `pcut1` result stays\nmanifold (over-shared = 0) with the flipped tool NURBS faces now\ncorrectly oriented in the outer shell.\n- Full workspace suite green.\n\n## Impact\n\nAny cut whose tool solid carries `is_reversed = true` faces\n(planar-NURBS extrusion walls are the common producer) previously\nemitted inverted faces in the result. Tool-level: 5 more compartment\nmanifold scenarios pass; the remainder trace to a separate\nsocket-interface defect under investigation.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes Cut assembly to toggle tool face orientation and drop non-closed\nhole shells. Results are manifold with correct cavity volumes; adds a\nregression test and minor doc-comment cleanup.\n\n- **Bug Fixes**\n- Toggle face orientation: treat `SelectedFace::reversed` as a flip\nrelative to the face’s current orientation instead of always\nconstructing `Face::new_reversed`.\n- Reject open hole shells: require hole shells to be closed before\nadding as inner shells to prevent edge over-sharing from stray\nfragments.\n\n<sup>Written for commit d8294891e35bd968b87310ddb733ce9753600f39.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1030?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T15:50:22Z",
          "tree_id": "0200adcb9738c06b6dfa48a4d6deaac5a4d67067",
          "url": "https://github.com/andymai/brepkit/commit/a20df5536da8a7dda1a49c9ecc892e8271480e73"
        },
        "date": 1783439592568,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 733441,
            "range": "± 3754",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 822403,
            "range": "± 2560",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11840,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 621291,
            "range": "± 11769",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18878378,
            "range": "± 1413809",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3e640a8af38c41e165f47bd0da1d31e68b1d28d",
          "message": "chore(main): release 2.124.2 (#1032)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.2](https://github.com/andymai/brepkit/compare/v2.124.1...v2.124.2)\n(2026-07-07)\n\n\n### Bug Fixes\n\n* **algo:** toggle orientation of flipped cut tool faces, reject open\nhole shells ([#1030](https://github.com/andymai/brepkit/issues/1030))\n([a20df55](https://github.com/andymai/brepkit/commit/a20df5536da8a7dda1a49c9ecc892e8271480e73))\n* **operations:** watertight, parity-density tessellation for\ncylinder/cone bands\n([#1029](https://github.com/andymai/brepkit/issues/1029))\n([e209d0c](https://github.com/andymai/brepkit/commit/e209d0cf3555b6cc4f0d18b13628156fc9670db9))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T15:56:20Z",
          "tree_id": "a7943cf2bad797de8d1c8cb9ba09d2e4350c37ed",
          "url": "https://github.com/andymai/brepkit/commit/f3e640a8af38c41e165f47bd0da1d31e68b1d28d"
        },
        "date": 1783439929075,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 719846,
            "range": "± 1512",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 809470,
            "range": "± 2452",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12278,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 586427,
            "range": "± 1953",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18528889,
            "range": "± 82517",
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
          "id": "b6e21e5d37d7a2769cea35011d85ca0db7256e02",
          "message": "fix(algo): scale the EF endpoint-contact window by crossing angle (#1033)\n\n## Summary\n\nRoot-caused from the gridfinity tool's literal export operands (captured\nwith the arena serializer, backtrace-trapped to\n`phase_ef::check_edge_face_pairs`):\n\nThe stacking lip's peak corner **arc** ends exactly on the tangent line\nwhere the body's coincident outer-wall **plane** grazes the lip's corner\ncylinder. A tangential edge-face contact's position along the curve is\nonly accurate to √residual — the EF crossing solver converged **3.1µm\nalong the arc** from the true endpoint (residual ~1e-12). The fixed\n`tol.linear` (1e-7) endpoint-contact window missed it, so the phase\nminted a near-duplicate vertex 3.1µm from the arc's own endpoint at\n**every lip corner**.\n\nThe result stays index-watertight (a micron-wide sliver triangle bridges\nthe pair) and volume-exact — but under the 1e-4 STL quantization the\ngridfinity tool and slicers use, the sliver's two long edges collapse\nonto one another: **8 \"non-manifold\" STL edges, two per corner** — the\ncompartment-export failure family.\n\n## Fix\n\nScale the existing endpoint-contact drop window by the crossing angle:\n`tol.linear / |curve_tangent · surface_normal|`, capped at 1mm.\nTransversal crossings (sin ≈ 1) keep the exact current behavior; grazing\ncontacts at an endpoint are recognized as the vertex-face incidence they\nare (the guard's original intent — its comment even names \"a cap-rim arc\ntangent to a coplanar wall corner\").\n\n## Verification\n\n- New fixture test `crates/io/tests/lipcorner_tangent_inmem.rs`: replays\nthe captured body+lip fuse through the provenance path\n(`boolean_with_evolution` — the tool's export route), asserting zero\nnear-duplicate vertex pairs below the STL quantization step,\nindex-watertight tessellation at the export tier, and zero defects under\nthe tool's exact 1e-4 quantized-STL oracle. Fails pre-fix, passes\npost-fix.\n- Full workspace suite green.\n- Tool-level: `compartmentBuilder.scenario.manifold` goes **5/13 →\n7/13** (both basic divider cases flip to passing; defect counts halve on\nseveral others — the residuals are smaller instances of the same class\nat other features, re-scoped in the roadmap row updated in this PR).\n- honeycombJunction and the fixture suites unchanged.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nScales the EF endpoint-contact window by crossing angle and gates it to\nendpoints that lie on the crossed surface to correctly detect grazing\nendpoint contacts. This removes the lip-corner duplicates and fixes\nnon-manifold STL edges in gridfinity exports without changing\ntransversal behavior.\n\n- **Bug Fixes**\n- Window is now `tol.linear / |curve_tangent · surface_normal|` (capped\nat 1 mm) and applies only toward an endpoint that lies on the crossed\nsurface; transversals and off-surface endpoints keep `tol.linear`.\nComment notes full-cap drops require `sin(angle) < 1e-4`.\n- Eliminates near-duplicate endpoint vertices from tangential edge–face\ncontacts at lip corners; removes 8 non-manifold STL edges in exports.\n- Tests: added `crates/io/tests/lipcorner_tangent_inmem.rs` with\ncaptured operands; quantized-STL oracle now counts sub-quantum slivers\n(three vertices in one bin) as defects; fails pre-fix, passes now.\n- Tool impact: `compartmentBuilder.scenario.manifold` improves 5/13 →\n7/13; honeycomb raw residuals re-pinned — pcut3 15→0, pcut2 holds (34),\npcut1 raw 53→65 due to previous noise splits; production result\nunchanged.\n\n<sup>Written for commit d0d736901d133081d8c9e21496899e00baf6beeb.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1033?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T17:29:50Z",
          "tree_id": "40db98fdb5affd62b56049631d84c31d4f1efc49",
          "url": "https://github.com/andymai/brepkit/commit/b6e21e5d37d7a2769cea35011d85ca0db7256e02"
        },
        "date": 1783445534004,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 748747,
            "range": "± 2852",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 834765,
            "range": "± 33615",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12027,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634680,
            "range": "± 28848",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19110500,
            "range": "± 56279",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8aa383111d4d43a254aeeaf6966348bc79b178af",
          "message": "chore(main): release 2.124.3 (#1034)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.3](https://github.com/andymai/brepkit/compare/v2.124.2...v2.124.3)\n(2026-07-07)\n\n\n### Bug Fixes\n\n* **algo:** scale the EF endpoint-contact window by crossing angle\n([#1033](https://github.com/andymai/brepkit/issues/1033))\n([b6e21e5](https://github.com/andymai/brepkit/commit/b6e21e5d37d7a2769cea35011d85ca0db7256e02))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.124.3 for `brepkit-wasm`. Adjusts the edge–face (EF)\nendpoint-contact window based on crossing angle to improve contact\ndetection and reduce false misses.\n\n<sup>Written for commit 53ab534d7bbefe17e9b735155bc6c12b357ac191.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1034?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T17:35:18Z",
          "tree_id": "4f21d85d6539c34696175715805fb4e294867805",
          "url": "https://github.com/andymai/brepkit/commit/8aa383111d4d43a254aeeaf6966348bc79b178af"
        },
        "date": 1783445849074,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 780527,
            "range": "± 1023",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 882856,
            "range": "± 18885",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13451,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 650490,
            "range": "± 21554",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20218348,
            "range": "± 22499",
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
          "id": "0132c1e8d6b25125077645559cca9e55876fdd77",
          "message": "fix(algo): drop boundary-re-tracing sections and weave straight NURBS hole rims (#1035)\n\n## Summary\n\nA stacking-lip fuse onto a compartmented body emits FF/PaveBlock\nsections that re-trace the lip ring's own boundary edges (the body's\ncorner cylinders and wall planes meet the lip's bottom plane exactly at\nits flush rims). Replaying one captured operand pair from the gridfinity\ntool exposed three stacked defects:\n\n1. **Sub-span boundary re-traces corrupt the arrangement.** Threading a\n45°-split half of a whole corner-arc edge (or a straight run split at a\ndivider crossing) wove a snake wire that multiply-traversed the rim. The\nshell flood-fill saw every junction edge as already-manifold and\norphaned the entire interior (85 faces) as an open fragment, which the\nhole-shell guard silently dropped — the fuse returned the 14-face body\nexterior while still passing the free/over edge checks. Fix:\n`section_on_existing_boundary` rejects sections whose interior lies on\nan existing boundary edge, in both section-source arms.\n- **Exemption (the load-bearing discriminant):** an exact whole-edge\nduplicate on the OUTER wire is kept — threading it routes the face\nthrough the split/rebuild that aligns coincident-face partitions;\ndropping it regressed the plain shelled-cup lip fuses\n(`gridfinity_d3/d4/d5`) to mesh fallback. Inner (hole) wires keep the\nunconditional drop: a hole-ring re-trace weaves the zero-area annulus of\nthe 2×1/1×2 lip-fuse failure.\n2. **Uniform hole probes miss thin material.** The divider cap's bridge\nsections were discarded as pure air: 8 uniform probes over a 246 mm span\nalways missed the 1.2 mm cap sliver between the cavity openings. Fix:\nalso probe midpoints between consecutive hole-boundary crossings — the\nsame sub-segment structure the hole weave itself uses.\n3. **Geometrically straight NURBS rims defeated the hole weave.** The\ntilted cavity rims are straight `NurbsCurve` edges; the weave's nominal\n`Line` filter sent them to the bail-on-crossing arc branch, so the cap\nface was never split. Fix: geometric straightness test.\n\n## Impact\n\n- Tool compartment manifold suite: **7/13 → 9/13** (tilted-divider +\ntop-row-merged lip fuses now watertight and analytic at export\ntolerance).\n- Three halfSockets-tilt cases that previously \"passed\" only via a\nwatertight mesh-fallback mask now pass analytically.\n- Honeycomb pcut1 raw-residual pin **improves 65 → 52**; pcut2 re-pinned\n34 → 38 (same noise-lean class as prior re-pins — production results in\nthat suite unchanged).\n\n## Verification\n\n- New fixtures `crates/io/tests/lipfuse_boundary_retrace_inmem.rs`\n(arena-serialized operands captured from the live tool): both fuses\nassert ≥90/≥100 faces, curved surface types present, zero free B-Rep\nedges, watertight export-tolerance mesh.\n- Full workspace suite green, including `brepkit-wasm --lib gridfinity`\n(d1–d5) and all lip-fuse fixtures (1×1, 2×1, 3×3).\n- Roadmap skill updated in-PR per the living-doc rule.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPrevents boundary re-tracing sections from breaking lip-fuse\narrangements and correctly weaves straight NURBS hole rims. Fixes\ntilted-divider and top-row-merged cases; tool manifold suite improves\n7/13 → 9/13.\n\n- **Bug Fixes**\n- Drop section segments that lie on existing boundary edges to avoid\nsnake wires and open fragments; keep exact whole-edge duplicates on the\nouter rim with a 10 µm endpoint band; still drop all inner (hole)\nre-traces.\n- Improve thin-material detection by probing midpoints between\nhole-boundary crossings on geometrically straight hole edges (exact\nchord crossings), complementing uniform samples.\n- Treat geometrically straight `NurbsCurve` rims as straight using\ncontrol-polygon collinearity, so the hole weave splits faces instead of\nbailing.\n\n<sup>Written for commit 02c0aa78008b6a9ee8c412b2270fc1417617bf07.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1035?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T20:35:46Z",
          "tree_id": "639487e95578126dcfb938cd664ee200133a7fbe",
          "url": "https://github.com/andymai/brepkit/commit/0132c1e8d6b25125077645559cca9e55876fdd77"
        },
        "date": 1783456680204,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 753219,
            "range": "± 1505",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 844721,
            "range": "± 46078",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12053,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 632878,
            "range": "± 10548",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19382898,
            "range": "± 253479",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "afacd91026261caf58c9a611151ef0dca5e10062",
          "message": "chore(main): release 2.124.4 (#1036)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.4](https://github.com/andymai/brepkit/compare/v2.124.3...v2.124.4)\n(2026-07-07)\n\n\n### Bug Fixes\n\n* **algo:** drop boundary-re-tracing sections and weave straight NURBS\nhole rims ([#1035](https://github.com/andymai/brepkit/issues/1035))\n([0132c1e](https://github.com/andymai/brepkit/commit/0132c1e8d6b25125077645559cca9e55876fdd77))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.124.4. Fixes NURBS hole handling by removing\nboundary re-tracing and weaving straight hole rims to eliminate\nartifacts and improve robustness.\n\n<sup>Written for commit 199db8dc8caa3d5e720dec37df3d6d236523cdd2.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1036?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T20:42:13Z",
          "tree_id": "8e7b7859819c88d3ab1c76fd1acf1bd12107d96b",
          "url": "https://github.com/andymai/brepkit/commit/afacd91026261caf58c9a611151ef0dca5e10062"
        },
        "date": 1783457063915,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 756397,
            "range": "± 931",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 844468,
            "range": "± 1145",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12285,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639272,
            "range": "± 22481",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19310571,
            "range": "± 17616",
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
          "id": "43bda38c8876379ee2596cbba7457fa29f35f876",
          "message": "fix(algo): arc-true hole polygons for the region classifier seed search (#1037)\n\n## Summary\n\n`find_point_outside_holes` — which picks the classifier seed for planar\nregions with holes (thin annular rings especially) — tested candidates\nagainst **chord-approximated** hole polygons. At a rounded corner, the\nchord under-covers the true hole by the arc's sagitta (~0.75 mm for the\nhalfSockets clip's r = 2.55 inset corner arcs), so a seed could be\naccepted in the gap between chord and arc — a point **inside the real\nhole**.\n\nCaptured from the live tool: the halfSockets base-clip cut (a\nrounded-rect slab trimmed by a flared clip tool, leaving a 1.2 mm\nperimeter wall). The wall's ring floor got exactly such a seed,\nclassified **Inside** the tool, and was discarded — an open shell that\nfailed validation, fell back to the mesh boolean (all analytic surfaces\nlost), and poisoned every downstream fuse of the export chain.\n\n## Fix\n\nWhen the plane frame is available (both call sites have it for planar\nfaces), curved hole edges are densified by sampling their **3D curve**\nand projecting through the frame — exact for arcs, and sidestepping the\ndocumented garbage-domain-pcurve trap that motivated chords in the first\nplace. Without a frame, the historical chord-midpoint densification is\nkept. `is_inside_any_hole`'s chord under-approximation is deliberately\nunchanged: for its drop-air-region use, under-covering is the\nconservative direction.\n\n## Impact\n\n- Tool compartment manifold suite: **9/13 → 10/13** — `2×2 crossing\ntilts` closed outright (was 15 non-manifold STL edges), `2×6 halfSockets\n±40` down from 26 non-manifold edges to 1 (small separate residual at\nthe socket interface, tracked on the roadmap).\n- The clip cut itself: 515-face all-planar mesh fallback → 49-face\nwatertight analytic result.\n\n## Verification\n\n- New fixture `crates/io/tests/halfsockets_clipcut_inmem.rs` (captured\noperands): asserts analytic face count, curved surfaces present, zero\nfree B-Rep edges, watertight export-tolerance mesh.\n- Full workspace suite green (pre-commit + pre-push gates), including\nwasm gridfinity d1–d5 and all lip-fuse fixtures.\n- Roadmap updated in-PR per the living-doc rule.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFix region-classifier seed selection for planar faces with holes by\ntesting candidates against arc-true hole polygons sampled from the 3D\ncurve and projected to UV, with correct handling of reversed arc edges.\nThis prevents seeds landing in rounded-corner sagitta gaps and restores\na watertight analytic result for the halfSockets clip cut.\n\n- **Bug Fixes**\n- Make hole polygons arc-true for seed rejection by sampling 3D curves\nthrough the plane frame; fall back to chord midpoints when no frame is\navailable.\n- Sample reversed hole arcs in the edge’s native orientation and reverse\nthe samples to preserve wire order; avoids selecting the complementary\narc.\n- Update `find_point_outside_holes` to accept a `frame` and adjust call\nsites; keep `is_inside_any_hole`’s conservative chord\nunder-approximation unchanged.\n- Add regression test `crates/io/tests/halfsockets_clipcut_inmem.rs`\nwith captured operands; the clip cut now yields a 49‑face watertight\nanalytic result (was a 515‑face mesh fallback).\n- Suite impact: 9/13 → 10/13; closes `2×2 crossing tilts`; `2×6\nhalfSockets ±40` improved from 26 non‑manifold edges to 1; roadmap entry\nand seed‑search doc comment corrected.\n\n<sup>Written for commit 6c2beccc63658d6d4c1c81019652cc379870a2bf.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1037?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T21:41:39Z",
          "tree_id": "069cc35078bf369452f7b11827b60e80d4978534",
          "url": "https://github.com/andymai/brepkit/commit/43bda38c8876379ee2596cbba7457fa29f35f876"
        },
        "date": 1783460639483,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 752921,
            "range": "± 24702",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 840801,
            "range": "± 2462",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11918,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 638299,
            "range": "± 7724",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19473121,
            "range": "± 51881",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d323f202f306e237d982fd4469d7ecc4a0f15a90",
          "message": "chore(main): release 2.124.5 (#1038)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.5](https://github.com/andymai/brepkit/compare/v2.124.4...v2.124.5)\n(2026-07-07)\n\n\n### Bug Fixes\n\n* **algo:** arc-true hole polygons for the region classifier seed search\n([#1037](https://github.com/andymai/brepkit/issues/1037))\n([43bda38](https://github.com/andymai/brepkit/commit/43bda38c8876379ee2596cbba7457fa29f35f876))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.124.5 for `brepkit-wasm`. Fixes the region classifier seed\nsearch by using arc-true hole polygons, improving accuracy on curved\nholes.\n\n<sup>Written for commit ff784499e4c80a3823f5732f613b95c58eeef3fd.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1038?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T21:47:34Z",
          "tree_id": "46b51095163ce8a5816b9f94c0b360b28171b1e9",
          "url": "https://github.com/andymai/brepkit/commit/d323f202f306e237d982fd4469d7ecc4a0f15a90"
        },
        "date": 1783461014251,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 747326,
            "range": "± 2078",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 840377,
            "range": "± 1405",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11889,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634979,
            "range": "± 1815",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19392828,
            "range": "± 33919",
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
          "id": "c709987ab100e8c50d62ba5cf81b99e16f84f841",
          "message": "fix(algo): decide planar hole nesting from the whole loop boundary (#1039)\n\n## Problem\n\nThe gridfinity tool's bin × socket-assembly fuse at the z=5 base\ninterface shipped broken geometry in two compartment scenarios:\n\n- **`1.5×6 ±40° no-halfSockets`**: the fuse shipped `free=24` B-Rep\nedges through the evolution path → 64–96 boundary edges in the export\nmesh at every bin corner.\n- **`1×4 2×8 compartments`**: the GFA result failed validation → mesh\nfallback whose output was itself non-manifold (867 all-planar faces,\n`free=113 nm=3`).\n\n## Root cause\n\nAt each bin corner, the bin bottom's rounded-corner arc (r = 3.75)\noverhangs the base socket outline's chamfer by a ~0.1 mm **crescent**\nthat must survive as a real face. The wire builder hands these crescents\nback as CW loops, and the planar loop splitter's hole-promotion pass\ndecided nesting with a **single interior probe**. On a 0.1 mm-thin\nregion the probe slips across the shared boundary into the adjacent\nsocket-square outer, so the crescent stayed a \"hole\"; hole matching then\nprobed its **first vertex** (exactly ON a neighboring boundary, where\nthe strict ray-cast answers false for every outer) and dumped all four\ncrescents onto the first sub-face — one of them geometrically unrelated,\nwith inner wires far outside its own outer wire. That face is\nsame-domain-dropped, so every corner crescent vanished.\n\n## Fix\n\n`loop_containment` decides containment from **every sampled point** of\nthe loop (exact for loops from a planar subdivision):\n\n- `Nested` (all points inside-or-on, ≥1 strictly interior) or\n`BoundaryCoincident` (all points ON the outline) → stays a hole.\n- `Outside` of **every** outer → promoted to a region.\n\nBoundary-coincident loops (whole-edge re-traces of a sibling outline)\ndeliberately stay holes: both promoting and dropping them regressed the\nshelled-cup lip fuse (d3/d4/d5 family) by un-threading the split that\naligns coincident-face partitions.\n\n## Verification\n\n- New fixtures `crates/io/tests/socket_assembly_fuse_inmem.rs` (captured\ntool operands, both fail pre-fix): both fuses now watertight and\nanalytic. The compartments case: 867-face non-manifold fallback →\n**403-face analytic result, 5× faster** (37 ms vs 201 ms).\n- Tool scenario suite (kernel overlay): **compartment manifold 10/13 →\n12/13**. The last failure (`2×6 halfSockets ±40` = 1 NM edge) is a\ndistinct pre-existing residual.\n- Full workspace suites green including `brepkit-wasm --lib gridfinity`\n(27/27).\n- Pre-existing (unchanged by this PR, verified on clean main):\n`binGenerator.scenario.halfSockets` 1×1/1.5×1.5/2×2 emit zero triangles\n— added to the roadmap.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes planar hole nesting in the face splitter by checking containment\nacross the whole loop boundary. This makes the bin × socket-assembly\nfuse watertight and analytic, and removes non-manifold fallbacks.\n\n- **Bug Fixes**\n- Introduced `loop_containment` to decide nesting from all sampled\npoints; promote only when some points are strictly outside every outer.\nNested and boundary-coincident loops (whole-edge re-traces) stay holes;\nkept the first-vertex probe and documented why to preserve the\nshelled-cup lip fuse split.\n- Added regression fixtures for the two failing fuses; compartments case\nnow 403-face analytic (~5× faster). Tool suite improves from 10/13 to\n12/13 manifold.\n\n<sup>Written for commit c07a087d84d8619ce6a77d5cf0848812ea1caac2.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1039?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-07T23:23:12Z",
          "tree_id": "f798dbd9192629c43d84f04eafc4f8e910df47ad",
          "url": "https://github.com/andymai/brepkit/commit/c709987ab100e8c50d62ba5cf81b99e16f84f841"
        },
        "date": 1783466719732,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 793344,
            "range": "± 5420",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 875592,
            "range": "± 5005",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13700,
            "range": "± 387",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631441,
            "range": "± 1152",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20245076,
            "range": "± 67770",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "567d9e3e71e98399dc78ce7ab064d056403d9b69",
          "message": "chore(main): release 2.124.6 (#1040)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.6](https://github.com/andymai/brepkit/compare/v2.124.5...v2.124.6)\n(2026-07-07)\n\n\n### Bug Fixes\n\n* **algo:** decide planar hole nesting from the whole loop boundary\n([#1039](https://github.com/andymai/brepkit/issues/1039))\n([c709987](https://github.com/andymai/brepkit/commit/c709987ab100e8c50d62ba5cf81b99e16f84f841))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.124.6 for `brepkit-wasm` with a fix to planar hole nesting\nthat evaluates the full loop boundary to avoid misclassified holes. This\nimproves geometric correctness in planar faces.\n\n- **Bug Fixes**\n- Planar hole nesting now computed from the entire loop boundary,\npreventing incorrect hole inclusion/exclusion in edge cases.\n\n<sup>Written for commit 712f6839d40ebf33c7425a670462a3b0f63fa457.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1040?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T23:29:03Z",
          "tree_id": "e20c62583d3ea224379be4c7c52f33c6b15e61b3",
          "url": "https://github.com/andymai/brepkit/commit/567d9e3e71e98399dc78ce7ab064d056403d9b69"
        },
        "date": 1783467080674,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 747539,
            "range": "± 2007",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 835666,
            "range": "± 2414",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11993,
            "range": "± 244",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635122,
            "range": "± 7037",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19248664,
            "range": "± 24338",
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
          "id": "0a77a6346016a2b194d662c47b49b9354693cc06",
          "message": "fix(algo): normalize inner-wire winding at the face splitter entrance (#1041)\n\n## Problem\n\nThe `2×6 halfSockets ±40°` compartment scenario — the last failing case\nof the compartment manifold family — exported a non-manifold STL. A\nfresh operand capture of the export chain isolated the root to an early\n**body × stacking-lip fuse**: it shipped `free=11` B-Rep edges (46\nboundary edges + 1 non-manifold edge in the export mesh) that propagated\nthrough the final socket-assembly fuse.\n\n## Root cause\n\nThe body's cavity cut emits its top-ledge **hole wire wound the SAME way\nas the outer wire** (non-standard). `integrate_holes_plane` weaves hole\npieces trusting their stored orientation, so where the lip's inner\nprofile crossed the hole's tilted-divider diagonal mid-span, the angular\nwire builder traced a **double-cover instead of a partition**: a\nspurious loop spanning the whole opening (kept — a membrane across the\nbin throat with every edge unpaired) plus the real throat-ledge region\nwound CW (hole-matched onto a face that same-domain dropping erased).\n\n## Fix\n\nNormalize inner-wire winding — flipped to oppose the outer wire in the\nprojected UV frame — where original inner wires enter `split_face_2d`.\nThis fixes the weave by construction.\n\nDetection-side alternatives (rerouting mis-woven faces to the even-odd\narrangement, triggered by residual CW holes / area balance / containment\ntests) were tried and rejected: every discriminant also caught the\n**load-bearing** whole-edge re-trace weaves (the shelled-cup d4 lip\nfuse, honeycomb caps) that must stay on the loops path.\n\n## Verification\n\n- New fixture `crates/io/tests/halfsockets_lipfuse_inmem.rs` (captured\ntool operands, fails pre-fix): fuse now watertight and analytic (`free\n11→0, bnd 46→0, nm 1→0`).\n- Tool scenario suite (kernel overlay): **compartment manifold 12/13 →\n13/13 — the family is closed.**\n- Honeycomb raw-residual pins hold exactly (pcut1 52, pcut2 38, pcut3\n0); shelled-wall notch cut, d-series lip fuses, gridfinity wasm suite\n(27/27), full workspace (2148) all green.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nNormalize inner-wire winding at the face splitter entrance so inner\nloops always wind opposite the outer loop in UV. Adds robust outer sign\nsampling and a sliver guard, fixing the mis‑woven cavity opening that\nproduced free edges and a non‑manifold STL in `2×6 halfSockets ±40°`,\nclosing the compartment manifold family.\n\n- **Bug Fixes**\n- In `split_face_2d`, for planar faces flip inner loops that match the\nouter’s winding (reverse edges, swap endpoints, toggle `forward`);\ncompute the outer sign from sampled pcurves and skip sliver holes where\n|area| ≤ perimeter×`tol.linear`.\n- Add regression test `crates/io/tests/halfsockets_lipfuse_inmem.rs`\nwith captured operands `hslipfuse_body.bin` and `hslipfuse_lip.bin`;\nresult is watertight and analytic (free 11→0, mesh boundary 46→0, nm\n1→0).\n  - Update roadmap entry to mark compartments at 13/13 passing.\n\n<sup>Written for commit 698a6558f0dbc8d34633bc94eb544bca85093391.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1041?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T02:05:13Z",
          "tree_id": "7b65a040117babaa92c9c4d83161dfd1c09661d3",
          "url": "https://github.com/andymai/brepkit/commit/0a77a6346016a2b194d662c47b49b9354693cc06"
        },
        "date": 1783476448052,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 744733,
            "range": "± 1754",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 833359,
            "range": "± 1143",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11801,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 630150,
            "range": "± 867",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19160848,
            "range": "± 23864",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "423f6e273e7931943c5542736a67ffb66f8269b1",
          "message": "chore(main): release 2.124.7 (#1042)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.7](https://github.com/andymai/brepkit/compare/v2.124.6...v2.124.7)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **algo:** normalize inner-wire winding at the face splitter entrance\n([#1041](https://github.com/andymai/brepkit/issues/1041))\n([0a77a63](https://github.com/andymai/brepkit/commit/0a77a6346016a2b194d662c47b49b9354693cc06))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.124.7 with a fix that normalizes inner‑wire\nwinding at the face splitter entrance to prevent incorrect face splits\nand reduce topology errors.\n\n<sup>Written for commit 9fb78092d89ce487ea5fd79c9dc6541f3511eb4b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1042?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T02:11:52Z",
          "tree_id": "dadb6e8df237a4ef9d3f8ab1b9c5a7893fbc368d",
          "url": "https://github.com/andymai/brepkit/commit/423f6e273e7931943c5542736a67ffb66f8269b1"
        },
        "date": 1783476851621,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 752071,
            "range": "± 3383",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 842111,
            "range": "± 3872",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12079,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 636057,
            "range": "± 2000",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19349308,
            "range": "± 259762",
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
          "id": "75221875982746a3c2a7ccdf0181a08136d3682d",
          "message": "fix(algo): resolve disconnected section loops in the planar arrangement splitter (#1043)\n\n## Root cause\n\nA closed section loop lying strictly inside a plane face — touching\nneither the face boundary nor any other section — is a **disconnected\ncomponent** of the arrangement trace graph, so the minimal-face trace\nwalks its cycle once per orientation. The flat emission path\n(`arrangement_regions_from_inputs`, `even_odd_nesting=false`) then:\n\n1. emitted **both** traces as duplicate overlapping regions, and\n2. left the region that geometrically contains the loop **without** the\nloop as an inner wire, so the hole-less container covered the\nduplicates.\n\nDownstream, same-domain detection glued container + duplicates + the\ncoincident opposing faces into one group (`planar_faces_overlap`'s hole\nguards all key on `inner_wires()`, which the woven container doesn't\nhave) and dropped every piece; the assembler's cap fill then patched the\nopenings with membranes lying **inside** solid material. The mesh showed\nsame-direction half-edge pairs along every interior loop rim — `bnd>0`\nwith `nm=0` while the orientation-blind B-Rep checks passed\n`free=0/over=0` — and a −13% signed mesh volume.\n\nHit by the halfSockets `2×2` bin × socket-assembly export fuse: the four\ninterior socket outlines on the bin bottom are exactly such loops\n(smaller halfSockets bins put every outline on the bin boundary, so they\nnever hit this).\n\n## Fix\n\nResolve twin cycle pairs in the flat emission: two traced faces whose\nhalf-edge sets are exact `h ↔ h^1` twins with opposite winding are one\ndisconnected loop. Emit it once as a solid region and attach the\nreversed twin as an inner wire of the smallest region that geometrically\ncontains it, with a hole-safe precomputed interior seed\n(`find_point_outside_holes`). Connected arrangements are unaffected —\ntwin pairs cannot occur there.\n\n## Verification\n\n- New fixture `crates/io/tests/halfsockets_socketfuse_inmem.rs`\n(captured tool operands): fails before, passes after.\n- Both halfSockets capture chains (hs1x1: 6 ops, hs2x2: 18 ops) replay\nfully clean — every op `bnd=0 nm=0`, all analytic; the affected fuse got\n35% faster (83→55ms) and 4 fewer faces (the membranes).\n- Full workspace suite green (40 suites), including the gridfinity wasm\ncanary 27/27.\n- `check-boundaries.sh` clean; clippy/fmt clean.\n\n## Notes\n\nThis closes the defect gating the ready loft branch\n`fix/loft-recognize-sketch-arcs` (curve-preserving socket lofts) — that\nlands separately on top of this.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes duplicate emission of disconnected section loops in the planar\narrangement splitter. Each interior loop is emitted once and its twin is\nattached as a hole, preventing dropped regions and interior membranes in\ndownstream fuses (e.g., the halfSockets 2×2 export).\n\n- **Bug Fixes**\n- Detect twin cycle pairs by matching half-edge sets (`h` ↔ `h^1`) with\nopposite winding.\n- Attach the reversed twin to the smallest containing region using an\nasymmetric all-vertex containment check with boundary tolerance; robust\nto nesting and logs if no parent is found.\n- Build inner wires with correct CW winding and seed interior samples\nwith `find_point_outside_holes`; hole cycles are not emitted. Connected\narrangements are unchanged.\n- Added regression test\n`crates/io/tests/halfsockets_socketfuse_inmem.rs` (captured operands).\nResult is watertight and analytic; the affected fuse runs ~35% faster\nwith fewer faces.\n\n<sup>Written for commit 3a374ae87dc9e077b5f267b1471eb5b31ec9e461.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1043?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T06:35:20Z",
          "tree_id": "8a370097cebdd13d0fb49c5cc1cca7a4536dc646",
          "url": "https://github.com/andymai/brepkit/commit/75221875982746a3c2a7ccdf0181a08136d3682d"
        },
        "date": 1783492656875,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 754885,
            "range": "± 1482",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 844264,
            "range": "± 1623",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11938,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631831,
            "range": "± 1459",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19485474,
            "range": "± 225726",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "265643962+brepkit[bot]@users.noreply.github.com",
            "name": "brepkit[bot]",
            "username": "brepkit[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "521f250ea2aa3f8d27be68fab7708db757298acd",
          "message": "chore(main): release 2.124.8 (#1044)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.8](https://github.com/andymai/brepkit/compare/v2.124.7...v2.124.8)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **algo:** resolve disconnected section loops in the planar arrangement\nsplitter ([#1043](https://github.com/andymai/brepkit/issues/1043))\n([7522187](https://github.com/andymai/brepkit/commit/75221875982746a3c2a7ccdf0181a08136d3682d))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.124.8 fixes the planar arrangement splitter to correctly\nhandle disconnected section loops. This prevents invalid splits and\nstray loops in `brepkit-wasm`.\n\n<sup>Written for commit 71de3fa52f1c3424f98c8c2ee103410f8c21c8ca.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1044?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T06:41:29Z",
          "tree_id": "fa2e45e59325e725f91e0817f54288c199db5186",
          "url": "https://github.com/andymai/brepkit/commit/521f250ea2aa3f8d27be68fab7708db757298acd"
        },
        "date": 1783493037032,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 758807,
            "range": "± 3046",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 850055,
            "range": "± 1117",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11939,
            "range": "± 265",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639542,
            "range": "± 38854",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19490456,
            "range": "± 111907",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}