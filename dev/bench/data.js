window.BENCHMARK_DATA = {
  "lastUpdate": 1782353710364,
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
      }
    ]
  }
}