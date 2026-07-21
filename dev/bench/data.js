window.BENCHMARK_DATA = {
  "lastUpdate": 1784617688935,
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
          "id": "c8d644b3137bf1c821b510f4719008c2c5eb77ec",
          "message": "fix(operations): curve-preserving loft for sketch arcs and downward stacks (#1045)\n\n## What\n\nTwo independent bails sent every real-world loft to the faceting polygon\npath:\n\n1. **NURBS profile edges** (the form a brepjs sketch can deliver) failed\nthe curve-type gate. `profile_oriented_edges` now recognizes NURBS edges\nback to their analytic form before matching.\n2. **Profiles wound opposite the stacking axis** were rejected outright\n— a loft stacked *downward* from CCW-sketched sections (the gridfinity\nsocket: top face at z=0, lofting to −height) winds every profile CW\nabout the axis. Such profiles are now reversed instead: edge order\nflipped, endpoints swapped, and arc circle normals negated (endpoint\nswapping alone selects the complementary arc span).\n\n## Impact\n\nEvery gridfinity socket loft previously came out ALL-PLANE — a\nz-histogram localized a 1.2–2.5% bin-volume deficit entirely to the z0–5\nfeet. With this fix the sockets are exact analytic (cones + cylinders),\nand tool-level volume parity is EXACT on the probed bins (13527.2 vs\n13527.3 reference).\n\n## Relationship to #1043\n\nThe analytic sockets initially un-masked the z=5 interface defect (hs2×2\nfuse bnd=314, −13% export volume) that gated this branch — that was the\narrangement disconnected-loop bug, fixed and merged in #1043. This\nbranch is rebased on top; both halfSockets capture chains replay fully\nclean (every op bnd=0 nm=0, analytic).\n\nSupersedes the old `fix/loft-recognize-sketch-arcs` branch (same loft\ncommit, rebased; the old branch can be deleted).\n\n## Verification\n\n- New unit tests in `crates/operations/src/loft/tests.rs` (NURBS-profile\nand downward-stack lofts stay analytic).\n- `loft_probe` example: 4/4 variants analytic (8 cones + 8 cylinders),\nidentical volume.\n- Full workspace suite green; gridfinity wasm canary 27/27.\n- Note: `fuse_shelled_box_with_socket_loft` (ignored) still fails\neuler≠2 even with this fix — the \"gated on curve-preserving loft\"\nframing was wrong; roadmap updated to reflect it needs fresh diagnosis.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nMake the loft curve-preserving for sketch-built profiles by recognizing\nNURBS arcs and safely handling downward-stacked sections. Gridfinity\nsockets now loft to analytic cones/cylinders instead of a faceted\nfrustum, restoring volume parity with the reference.\n\n- **Bug Fixes**\n- Recognize NURBS profile edges back to analytic `Circle`/`Line` in\n`profile_oriented_edges`, so curve-matching no longer falls back to\npolygon faceting.\n- Reverse downward-stacked profiles using a dimensionless cosine gate\n(stable winding detection); flip edge order, swap endpoints, and negate\ncircle normals. Mixed or degenerate windings still bail.\n- Add rounded-rect loft tests in `crates/operations/src/loft/tests.rs`\ncovering native/NURBS arcs and up/down stacks; roadmap updated.\n\n<sup>Written for commit 9af4ae12330d8ef9c371c014f159fc5ccfe46838.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1045?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T06:55:43Z",
          "tree_id": "dc75439dca44c4732354865b1ff3d9bce9cd54c7",
          "url": "https://github.com/andymai/brepkit/commit/c8d644b3137bf1c821b510f4719008c2c5eb77ec"
        },
        "date": 1783493893953,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 756434,
            "range": "± 1641",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 845938,
            "range": "± 1766",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11874,
            "range": "± 378",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631355,
            "range": "± 1452",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19519548,
            "range": "± 512546",
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
          "id": "06276a7bf9573a63ad413ed7580716a267a4cd7b",
          "message": "chore(main): release 2.124.9 (#1046)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.9](https://github.com/andymai/brepkit/compare/v2.124.8...v2.124.9)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **operations:** curve-preserving loft for sketch arcs and downward\nstacks ([#1045](https://github.com/andymai/brepkit/issues/1045))\n([c8d644b](https://github.com/andymai/brepkit/commit/c8d644b3137bf1c821b510f4719008c2c5eb77ec))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.124.9 fixes loft behavior in operations to preserve\ncurve shape for sketch arcs and downward stacks, improving geometry\naccuracy and continuity. Bumps `brepkit-wasm` to 2.124.9.\n\n<sup>Written for commit c4773514ccf3daceb3eac092a339c41407da6919.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1046?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T07:01:18Z",
          "tree_id": "78874958fd652f4f66e79014beeb8e2c7b2b9c88",
          "url": "https://github.com/andymai/brepkit/commit/06276a7bf9573a63ad413ed7580716a267a4cd7b"
        },
        "date": 1783494228797,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 763981,
            "range": "± 1832",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 853794,
            "range": "± 965",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12032,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 641962,
            "range": "± 1894",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19472074,
            "range": "± 69099",
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
          "id": "e564fb85ac47adf59f0fd37831e201cbb2b12c82",
          "message": "docs(roadmap): post-merge tool matrix — halfSockets closed, new fractional-width family (#1047)\n\nRoadmap-only update after #1043 + #1045 merged, recording the post-merge\ntool scenario matrix:\n\n- **halfSockets suite: 8/11** — the 3 fails are kernel-pin snapshots\n(benign by design); triangle counts now run ~45% above the reference\npins because analytic feet replace sparse faceted planes (noted as a\npossible density follow-up, not a defect class).\n- **New deferred row: post-loft fractional-width halfSockets family** —\n`1.5×6 halfSockets` baseline + all tilt variants fail bnd=104\n(tilt-independent, deterministic), `1×4 2×8-comps` nm=12. Discriminated:\na #1043-only (pre-loft) kernel passes the same cases, so this is\nnew-geometry un-masking from the analytic sockets, not a regression.\nFirst probe documented (fresh capture via the rebased probe branch).\n- Caveat added to the compartments 13/13 row (measured on pre-loft\nfaceted sockets; closed roots stay closed).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate the roadmap matrix after #1043 and #1045 to reflect post-merge\ntool results. Closes the halfSockets suite and adds a new post-loft\nfractional-width family to track.\n\n- **Roadmap updates**\n- halfSockets: 8/11 passing; 3 are benign kernel-pin snapshots. Triangle\ncount ~+45% vs reference due to analytic feet (tessellation follow-up,\nnot a defect).\n- Compartments row: adds pre-loft caveat; closed roots still replay\nclean. Probes rebased to `probe/boolean-capture-2`.\n- New fractional-width family: `1.5×6 halfSockets` baseline + all tilts\nfail with bnd=104; `1×4 2×8-comps` STL shows nm=12. Deterministic,\ntilt-independent, and not a regression (pre-loft kernel passes). First\nprobe captured.\n\n<sup>Written for commit c27bcc263e2343b33d9f5e3727ad1408e79a99d9.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1047?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T07:12:54Z",
          "tree_id": "0fc45b7d138bb29549efafdadae4b0b78e96d7f6",
          "url": "https://github.com/andymai/brepkit/commit/e564fb85ac47adf59f0fd37831e201cbb2b12c82"
        },
        "date": 1783494913021,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 753122,
            "range": "± 1479",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 847733,
            "range": "± 1625",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11897,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635127,
            "range": "± 3936",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19617054,
            "range": "± 60892",
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
          "id": "0647ce1eee7214cc697fa1178fcaa1ff2544c113",
          "message": "docs(roadmap): make compartment and halfSockets row titles match their status (#1048)\n\nFollow-up to #1047, addressing the two Copilot consistency findings that\nlanded after its merge:\n\n- The compartment row was titled \"CLOSED, 13/13 pass\" while its own\ncaveat said the tool matrix no longer reads 13/13. Retitled to make the\ndurable claim explicit: the six roots are closed; the 13/13 score was\nmeasured on pre-loft geometry, and the live matrix lives in the\nfractional-width row.\n- The \"halfSockets faceted-loft family\" row described a fixed defect\nunder a name implying it was still present, and ended with a\nself-contradictory \"CLOSED except …\". Retitled to \"halfSockets loft\nfaceting — CLOSED (#1045)\" and pointed remaining suite work at the\nfractional-width row.\n\nRoadmap-only; no code changes.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRetitled two roadmap rows for clarity and consistency. The compartment\nrow now explicitly calls out six closed roots and timestamps the 13/13\nscore to pre-loft geometry; the halfSockets loft faceting row is marked\nCLOSED (#1045) and points remaining work to the fractional-width row.\n\n<sup>Written for commit 79f9995212aa54734ed866f887985ae2ed5bafdb.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1048?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T07:34:42Z",
          "tree_id": "447040759f712181bd16c9bc593bdf6627e8aadf",
          "url": "https://github.com/andymai/brepkit/commit/0647ce1eee7214cc697fa1178fcaa1ff2544c113"
        },
        "date": 1783496224657,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 755715,
            "range": "± 1485",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 847767,
            "range": "± 9736",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12001,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635381,
            "range": "± 2109",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19562648,
            "range": "± 50958",
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
          "id": "90d0c6a8beac14e8b5a2e3c6c15e2b938bfa2c01",
          "message": "fix(algo): orientation-safe interior points for plane sub-faces (#1049)\n\n## Summary\n\nCloses the post-loft **fractional-width halfSockets export family**:\nevery `1.5×6 halfSockets` compartment scenario (baseline + all tilt\nvariants) exported with 104 mesh boundary edges, clustered at the four\nbin corners on the z=5 socket×body interface.\n\n### Root cause\n\nAt each bin corner, the analytic socket outline's r=4 circle (tangent to\nboth bin wall lines, new geometry since #1045) and the bin's r=3.75\ncorner arc bound a ~0.1–0.25 mm sliver on the bin bottom. The\narrangement split emitted the sliver region correctly — but\n`interior_point_3d` built the region's UV polygon from stored\n**pcurves**, which reach a wire under two orientation conventions:\n\n- **section arcs** carry the curve's natural parameterization plus a\ntraversal flag;\n- **boundary arcs** are fit in traversal order but keep the topology\norientation flag.\n\nThe flag-driven sampler traced reversed boundary arcs **backwards**,\nfolding the sliver polygon into a self-crossing zig-zag whose \"interior\"\npoint landed in the adjacent socket-imprint region. The classifier then\nread solid material at the wrong point, marked the sliver `Inside`, and\ndropped it — 5 unpaired rim edges per corner (socket r=4 arc + two bin\nr=3.75 arc pieces + two 0.25 mm wall stubs).\n\n### Fix\n\n- `interior_point_3d` now samples plane-face wire polygons from each\nedge's **3D curve through the face's `PlaneFrame`**\n(orientation-unambiguous — the same arc-true pattern as\n`find_point_outside_holes`), never the pcurves. Deliberately scoped to\ninterior-point computation: widening it to the shared\n`sample_wire_loop_uv` changes split decisions and regressed the\nscooplabel over-share pin.\n- `find_point_outside_holes` hole polygons densified 3 → 15 interior\nsamples: a single-edge closed bore hole sampled at 4 points is an\ninscribed square whose sagitta gap accepted annulus seeds well inside\nthe bore (caught by the drilled-tube volume regression test).\n\n### Verification\n\n- New fixture `crates/io/tests/fracwidth_corner_crescent_inmem.rs`:\ncaptured 1.5×6 halfSockets bin × captured corner half-socket (translated\n+5z to its final-assembly position). Pre-fix: 5 free rim edges, bnd=40;\npost-fix: watertight, manifold, analytic.\n- Fresh 40-op capture of the full 1.5×6 export chain replays clean\n(every op free=0/bnd=0/nm=0, analytic).\n- Tool-level: all six `1.5×6` compartment manifold variants pass. The\nremaining integer-width non-manifold failures (`1×6`/`2×6`,\nnm=76/136/140) are **pre-existing** — counts identical under a pre-fix\nkernel build (77/136) — and now tracked as their own roadmap row.\n- Full workspace suites + wasm gridfinity canary green; honeycomb pins\nand all prior fixtures (scooplabel, halfsockets, socket-assembly,\nlipfuse) unchanged.\n\nRoadmap updated in the same PR (crescent row closed, integer-width nm\nfamily row added).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes the dropped corner crescent on plane sub-faces by sampling wire\npolygons from 3D curves via the face `PlaneFrame`, removing the 104\nboundary-edge artifact in all `1.5×6 halfSockets` exports. Also\ndensifies hole sampling to avoid seeds slipping inside single-edge\ncircular bores.\n\n- **Bug Fixes**\n- `interior_point_3d` (plane faces): sample each edge’s 3D curve through\nthe `PlaneFrame` instead of pcurves to avoid mixed-orientation folds;\nkeep `sample_wire_loop_uv` unchanged.\n- `find_point_outside_holes`: increase interior samples from 3 to 15 to\nclose the sagitta gap in bore holes.\n- Added regression fixture `fracwidth_corner_crescent_inmem.rs` with\ncaptured bin/socket operands; result stays analytic and watertight;\nroadmap updated.\n\n<sup>Written for commit b963eef282b3f13e4c96d94cb5f429afec6d64c4.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1049?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T08:50:48Z",
          "tree_id": "dbc9defa754c9d689b1440671d7f8165436b1332",
          "url": "https://github.com/andymai/brepkit/commit/90d0c6a8beac14e8b5a2e3c6c15e2b938bfa2c01"
        },
        "date": 1783500783761,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 747424,
            "range": "± 2194",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 837647,
            "range": "± 3331",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11896,
            "range": "± 153",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 633008,
            "range": "± 60306",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19396192,
            "range": "± 82156",
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
          "id": "6e0140bf1e6093f1a4f0f90d8d5c2c19c18d0c21",
          "message": "chore(main): release 2.124.10 (#1050)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.10](https://github.com/andymai/brepkit/compare/v2.124.9...v2.124.10)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **algo:** orientation-safe interior points for plane sub-faces\n([#1049](https://github.com/andymai/brepkit/issues/1049))\n([90d0c6a](https://github.com/andymai/brepkit/commit/90d0c6a8beac14e8b5a2e3c6c15e2b938bfa2c01))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.124.10 for `brepkit-wasm` fixes orientation-safe\ninterior point selection for plane sub-faces to improve robustness. No\nAPI changes.\n\n- **Bug Fixes**\n- Interior points now respect face orientation on plane sub-faces,\npreventing misclassification and downstream issues like triangulation\nerrors.\n\n<sup>Written for commit a2353e7b163ca343ffc45862d1ab321db06d3e90.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1050?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T08:57:39Z",
          "tree_id": "4207d08956f0cbd3c9913ef7f3c10c474453e9a9",
          "url": "https://github.com/andymai/brepkit/commit/6e0140bf1e6093f1a4f0f90d8d5c2c19c18d0c21"
        },
        "date": 1783501193769,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 752923,
            "range": "± 4473",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 836615,
            "range": "± 4466",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11931,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 629512,
            "range": "± 3822",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19293736,
            "range": "± 316276",
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
          "id": "190419ae8d55a10bbc04c4146246549235ca27f7",
          "message": "fix(math,algo): exact tangential intersections at socket-outline wall tangencies (#1051)\n\n## Summary\n\nCloses the **integer-width halfSockets non-manifold export family**\n(`1×6`/`2×6` compartment scenarios at nm=76/136/140, and the `1×4\n2×8-comps` reporter case at nm=12). With this fix the **full compartment\nmanifold scenario matrix is 13/13 at tool level on analytic sockets** —\nthe first honest 13/13 since #1045 made socket geometry analytic.\n\n### Root cause\n\nA half-socket outline's r=4 corner circles are **exactly tangent** to\nthe bin wall lines, and the outline's straight runs continue along those\nwalls from the tangency points — which therefore exist as exact operand\nvertices. Two solvers recomputed those tangential intersections ~1µm off\ninstead of landing on the exact points (a grazing contact's positional\nerror grows as √(2r·residual): a 1e-13 residual at r=4 is a full micron,\nabove vertex-merge tolerance):\n\n1. **`Circle3D::intersect_segment`** solved the near-tangent quadratic\ninto a root pair straddling the foot of the center on the line. Both\nphase EE and phase FF's closed-circle splitter consume it, each minting\na near-duplicate vertex next to the exact one.\n2. **Phase EF's grazing edge×surface refinement** can land anywhere\ninside the tolerance *well* (distance to the surface grows only\nquadratically around a tangency).\n\nThe ~1e-6mm line edges bridging the near-duplicates were used by three\nfaces each (one out-and-back on the bin-bottom web plane), so the\nanalytic fuse tripped the non-manifold gate and fell back to the mesh\nboolean — whose output was itself non-manifold (nm=76 in the export).\n\n### Fix\n\n- `Circle3D::intersect_segment`: when the chord between the two roots\nimplies **sub-tolerance penetration** (`disc ≤ 2r·tol·a`), collapse to\nthe well-conditioned double root — the foot. Genuine secants\n(penetration > tol) keep both crossings (unit-tested both ways).\n- Phase EF: tangential crossings **snap to an existing pave vertex**\nwithin the angle-scaled window (`tol/|tangent·normal|`, capped 1e-3 —\nthe same conditioning model as the #1033 endpoint windows), gated on\nthat vertex lying ON both the crossed surface and the edge's curve. The\nwidened lookup linear-scans pave endpoints because the spatial index\nstencil is exhaustive only at tolerance radius.\n\n### Verification\n\n- Final bin×socket-assembly fuse: 889-face gate-reject + broken mesh\nfallback → **891 analytic faces, free=0/over=0, watertight+manifold at\nexport tolerance, ~40× faster** (193ms vs 7s).\n- Fixture `crates/io/tests/intwidth_tangency_inmem.rs` (captured bin ×\ncaptured wall-adjacent socket): pre-fix `free=20 over=10 bnd=24 nm=10`,\npost-fix clean.\n- Full 30-op capture chain replays clean; **tool matrix 13/13** (all\n1.5×6, 1×6, 2×6, crossing-tilts, and 1×4 2×8-comps cases).\n- Full workspace suites + wasm gridfinity canary green; all prior\nfixtures (crescent, scooplabel, halfsockets, socket-assembly) unchanged;\ntwo new math unit tests pin the collapse and the genuine-secant\nbehavior.\n\nRoadmap updated in the same PR (integer-width row closed with the 13/13\nresult).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes exact wall‑tangency intersections in analytic socket outlines by\ncollapsing near‑tangent circle×segment roots and snapping grazing\nedge×surface crossings, with safeguards for ambiguous junctions. The\nbin×socket fuse stays analytic, watertight, and manifold (tool matrix\n13/13) and runs ~40× faster than the mesh fallback.\n\n- **Bug Fixes**\n- `crates/math`: `Circle3D::intersect_segment` collapses near‑tangent\nroot pairs to the foot when `disc ≤ 2r·tol·a`; genuine secants still\nreturn two hits (unit‑tested).\n- `crates/algo`: Phase EF snaps tangential crossings to an existing pave\nvertex within an angle‑scaled window; adds\n`find_nearby_pave_vertex_widened` to scan pave endpoints, now declines\nto snap if candidates span multiple positions (ambiguous), and requires\nthe candidate to lie on the crossed surface and inside face containment;\nfor line edges, recomputes the exact parameter.\n\n- **Verification**\n- Final fuse: 891 analytic faces; watertight and manifold at export\ntolerance; ~193 ms vs 7 s mesh fallback.\n- New fixture `intwidth_tangency_inmem.rs` (captured operands)\nreproduces the pre‑fix failure and passes post‑fix; all suites and wasm\ngridfinity canary remain green.\n\n<sup>Written for commit a8134e838876420164251fbfdc2746118c72564a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1051?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T10:22:06Z",
          "tree_id": "1450722f2cb9b7913b0b1e48f73973d09d38ec6e",
          "url": "https://github.com/andymai/brepkit/commit/190419ae8d55a10bbc04c4146246549235ca27f7"
        },
        "date": 1783506259703,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 773266,
            "range": "± 1685",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 874833,
            "range": "± 3756",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13051,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 633657,
            "range": "± 1290",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20334222,
            "range": "± 36138",
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
          "id": "880f774c5de5ac9a7c9d4a6d99a3b98dcc3dd0b5",
          "message": "chore(main): release 2.124.11 (#1052)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.11](https://github.com/andymai/brepkit/compare/v2.124.10...v2.124.11)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **math,algo:** exact tangential intersections at socket-outline wall\ntangencies ([#1051](https://github.com/andymai/brepkit/issues/1051))\n([190419a](https://github.com/andymai/brepkit/commit/190419ae8d55a10bbc04c4146246549235ca27f7))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.124.11 for `brepkit-wasm` improves geometric accuracy by\nfixing exact tangential intersections at socket-outline wall tangencies.\nAlso updates the version and changelog.\n\n<sup>Written for commit 867b73187795d3de91fcaba7aafed5c0bbbc2f2a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1052?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T10:27:50Z",
          "tree_id": "b7d7d5551ef09083288a5655155c4ebce48f88c1",
          "url": "https://github.com/andymai/brepkit/commit/880f774c5de5ac9a7c9d4a6d99a3b98dcc3dd0b5"
        },
        "date": 1783506618106,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 750054,
            "range": "± 2666",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 839835,
            "range": "± 1874",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11879,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634696,
            "range": "± 20710",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19526399,
            "range": "± 484891",
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
          "id": "97069b1f50389aecd96d9fa12c8bfe3ad2023590",
          "message": "docs(roadmap): fresh socket-loft fuse diagnosis (#1053)\n\nRoadmap-maintenance PR (the living-doc mandate): records the fresh\n`fuse_shelled_box_with_socket_loft` diagnosis measured on post-#1051\nmain.\n\n- Raw GFA improved euler 36→9 across the tessellation-parity wave; the\neuler=−54 headline is the mesh-fallback output, not the analytic result.\n- The 2026-06-13 plan's Layers 1–2 are effectively done: the cup cap now\nsplits `arrangement n=5` (interior + 4 crescents, the intended shape).\n- Two residual roots identified and recorded with probes: co-endpoint\narc/chord lenses collapsed by `merge_duplicate_edges` (sanctioned\nsplitter-side midpoint-split pattern, torus-box precedent), and ~25µm\nphantom arc-break vertices minted from face-splitter UV data\n(VERTEX_WATCH-verified as builder-side, not pave-phase).\n\nNo code changes.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate the roadmap with a fresh diagnosis for\n`fuse_shelled_box_with_socket_loft`.\nIt documents raw GFA improving Euler 36→9 (mesh fallback is −54),\nconfirms Layers 1–2 are complete (cup splits into interior + four\ncrescents), and records the two remaining roots—arc/chord lens collapse\nunder `merge_duplicate_edges` and splitter-minted phantom arc-break\nvertices—plus next probes via `RAW_GFA` and `DUMP_ARR`.\n\n<sup>Written for commit 844acb57e037e3a5bd1828ccd5a73556c26100c5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1053?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T15:18:48Z",
          "tree_id": "9c3c3a3d9d8a5e2713f17e54cd688f1d4272e778",
          "url": "https://github.com/andymai/brepkit/commit/97069b1f50389aecd96d9fa12c8bfe3ad2023590"
        },
        "date": 1783524068345,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 746221,
            "range": "± 1691",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 842061,
            "range": "± 1444",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11941,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635092,
            "range": "± 2530",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19900017,
            "range": "± 616274",
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
          "id": "bb9b1c9248ea701c6d60895dac914810b294702c",
          "message": "fix(algo): close dovetail corner-clip intersect chord/arc lens (#1054)\n\n## Summary\n\nCloses the deferred `dovetail_corner_clip_intersect_is_watertight` case:\nthe baseplate corner-rounding Intersect (slab-with-pockets ×\nrounded-rect prism) now returns a compact, watertight, fully analytic\nresult (41 faces, both corner barrel pieces preserved) instead of\nfalling back to a 6116-facet mesh slab that poisoned every downstream\ndovetail connector boolean (~11 min per tile, non-manifold STL).\n\nTwo stacked GFA roots, which only close **together** (chord-fix-only:\nfree 1→39; arc-fix-only: free 41; both: free 0):\n\n### 1. Coplanar chord/arc lens (`phase_ff_coplanar.rs`)\n\nThe FF-coplanar phase projects boundary edges as straight 2D segments,\nso a rounded-corner boundary **arc** became a **chord** section — while\nthe true arc already existed as an FF section (the barrel×cap circle,\nsplit at the operand's diagonal seam vertex). The\n`has_existing_section_at` midpoint dedup can never catch this: emitted\ncircle arcs store the full-circle bbox, whose midpoint is the circle\n**center**. The resulting co-endpoint chord/arc lens breaks the wire\nweave — the chord lands in the outer wire and the true arc is orphaned\nas a zero-area slit (the free edge).\n\nFix: `matching_arc_section_exists` — skip a **Circle** boundary edge\nwhen its exact arc (same circle, same endpoints, shared face) already\nexists as a section. Line boundary edges are never skipped: a\nco-endpoint line/arc pair can be a genuine lens with material between\nthe curves (the torus-box in-tube case).\n\n### 2. Reverse-twin phantom arc splits (`face_splitter`)\n\n`split_sections_at_t_junctions` normalized arc split parameters via\n`domain_with_endpoints`, which always returns the **CCW** span. Sections\nare pushed as forward/reverse pairs — for the reverse twin the CCW span\nis the long complement (315° for a 45° corner arc), so a point on the\ncircle but *outside* the arc normalized to an interior `t` (45/315 =\n1/7), and the split evaluator (shorter-arc convention) minted a\n**phantom vertex on the true arc's interior**. That desynced the\ncoincident caps' partitions and killed their same-domain pairing.\n\nFix: `find_splits_on_section_arc` — section splits use the same\nshorter-arc convention as `evaluate_edge_at_t` (sections are ≤π by\nconstruction). Boundary-edge splitting deliberately keeps the CCW-domain\nconvention: boundary arcs may genuinely exceed π — switching them\nregressed the d1/d3/d4/d5 gridfinity lip fuses (caught by the wasm\ncanary during development, reverted).\n\n## Verification\n\n- `dovetail_corner_clip_intersect_is_watertight` un-ignored and green\n(full operations boolean, no fallback)\n- Raw-GFA guard tightened from `free<=1` to `free==0` (renamed\n`cornerclip_intersect_raw_gfa_stays_watertight`)\n- `cargo test -p brepkit-wasm --lib gridfinity` green (27/27 — the d4/d5\nlip-fuse canary)\n- Full workspace suite green (77 suites; `brepkit-render`\n`compute_mesh_lod` SIGSEGV is pre-existing on main in this environment,\nunrelated)\n- clippy `-D warnings` clean, boundaries clean\n- Roadmap skill updated in-PR (cornerclip row → CLOSED; socket-loft row\nupdated — that case is a different mechanism and remains deferred,\nops-level state unchanged)\n\n## Expected downstream effect\n\nThe corner-rounding intersect is shared by all baseplate generators —\nthe dovetailKey (bnd=108), fit-offset (bnd=184/144), snapClip\n(nm=14/16/11) scenario failures and the >25-min dovetail suite timeout\nall consume its mesh-fallback slab. Tool-side re-probe follows after\nrelease.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes the dovetail corner-clip Intersect by removing a chord/arc lens\nand a reverse‑twin arc split bug. The Intersect now returns a compact,\nwatertight analytic result (41 faces; both corner barrel pieces) instead\nof a slow, non‑manifold mesh fallback.\n\n- **Bug Fixes**\n- FF‑coplanar: added `matching_arc_section_exists` to skip projecting a\nCircle boundary edge as a straight chord when the exact arc section\nalready exists; compares center, radius, endpoints, ARC MIDPOINT, and\nrequires parallel normals so same‑center different‑plane circles don’t\nmatch. Line edges are never skipped.\n- Splitter: introduced `find_splits_on_section_arc` and used the\nshorter‑arc convention for Circle section splits only; kept the CCW\ndomain for boundary arcs and all ellipse sections.\n- Tests: un‑ignored `dovetail_corner_clip_intersect_is_watertight` and\ntightened the raw‑GFA guard to `free == 0`; both tests pass.\n- Result: no mesh fallback; analytic output stays small and watertight.\n\n<sup>Written for commit 617db9badf44dac16c4e410d65aafc7d8679e60a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1054?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T10:43:17-07:00",
          "tree_id": "5a188105ba8a5b19d12e19273e94235cad4aa47d",
          "url": "https://github.com/andymai/brepkit/commit/bb9b1c9248ea701c6d60895dac914810b294702c"
        },
        "date": 1783532740305,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 758795,
            "range": "± 1612",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 847597,
            "range": "± 2005",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11984,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 643492,
            "range": "± 1511",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19606209,
            "range": "± 155253",
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
          "id": "5131efb8d957bd7685eaf36ddec8a07a8b807848",
          "message": "chore(main): release 2.124.12 (#1055)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.12](https://github.com/andymai/brepkit/compare/v2.124.11...v2.124.12)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **algo:** close dovetail corner-clip intersect chord/arc lens\n([#1054](https://github.com/andymai/brepkit/issues/1054))\n([bb9b1c9](https://github.com/andymai/brepkit/commit/bb9b1c9248ea701c6d60895dac914810b294702c))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.124.12. Fixes an algorithm issue where\nchord/arc intersections in dovetail corner-clip cases didn’t close the\nlens.\n\n- **Bug Fixes**\n- Properly closes the dovetail corner-clip lens for chord–arc\nintersections to prevent small gaps in geometry.\n\n<sup>Written for commit 4fbb2162466aa31b27537867bac69ada76142f3f.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1055?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T17:48:48Z",
          "tree_id": "8ba88fa817da118a312bca21500b3f632965d18c",
          "url": "https://github.com/andymai/brepkit/commit/5131efb8d957bd7685eaf36ddec8a07a8b807848"
        },
        "date": 1783533070677,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 792343,
            "range": "± 2743",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 906956,
            "range": "± 14937",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13142,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 644754,
            "range": "± 2142",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20534895,
            "range": "± 326057",
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
          "id": "ce7d8dda8b50dc638b49af4d6af60288e29ad780",
          "message": "docs(roadmap): post-#1054 baseplate re-probe baseline (#1056)\n\nLiving-doc update after #1054 released as 2.124.12 and the baseplate\nsuites were re-probed tool-side.\n\n**Headline**: the dovetail scenario suite went from a >25-minute timeout\nto **355s total** — the corner-rounding intersect no longer\nmesh-falls-back, so connector fuses stopped consuming the 6116-facet\nslab. The A1-canonical corner tile dropped from ~597 non-manifold STL\nedges (11 min/tile) to nm=3 at 468ms.\n\nRecords the residual failure families for the next fix cycles:\n- bnd=108: 5×4 middle-column + inverted + dovetailKey (one shared root)\n- bnd=144: 4×4 interior tiles — no rounded corners, i.e. the\nfully-coincident-walls intersect variant\n- bnd=5 + 265s: 5×4.5 fractional edge tile; nm=6 magnet variant; nm=3\ncorner tile\n- snapClip: partial movement (nozzle case nm 11→1), clip-solid nm is a\nseparate (fillet-family) root\n- combinedFeatures honeycomb+handles: 'handle holes'/'funnel cutout'\nPASS at engine level in 62–87s (masked by the 60s per-test timeout —\nperf item); the real defect is the 'handles + label (back skip)'\nswallowed panic that poisons the wasm borrow flag.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate the roadmap with the post-#1054 re-probe baseline on 2.124.12.\nNotes the dovetail suite speedup (>25 min → 355s), the A1 corner tile\nimprovement (~597 non‑manifold edges → 3 at 468ms), the remaining\nfailure families (bnd=108/144/5; nm=6/3), and a swallowed‑panic defect\nin combinedFeatures that’s hidden by the 60s test timeout.\n\n<sup>Written for commit 907f8bf6d774737d5240bd5ada0f36faeb1e9f0a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1056?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T11:24:51-07:00",
          "tree_id": "3e76e3f94eca899288ba26eb6f11b2790fac4619",
          "url": "https://github.com/andymai/brepkit/commit/ce7d8dda8b50dc638b49af4d6af60288e29ad780"
        },
        "date": 1783535219136,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 613460,
            "range": "± 13091",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 697047,
            "range": "± 15643",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 9468,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 506417,
            "range": "± 2161",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 15998121,
            "range": "± 218373",
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
          "id": "d2a98fcd5f2b4b43b706b35681ca247433303382",
          "message": "fix(operations): route trivial operand pairs around the evolution GFA path (#1057)\n\n## Summary\n\nCloses the dominant post-#1054 baseplate residual: the **bnd=108/144 STL\nfamily** across the dovetail, dovetailKey, and fit-offset scenario\nsuites.\n\n**Root**: the tool's export booleans run `boolean_with_evolution` (face\nprovenance for feature tags). Its faithful raw-GFA branch mis-splits\n**identical/contained operand pairs** — the fully-coincident-boundary\nconfiguration that `boolean()` short-circuits on purpose. Coincident\nwalls drop into an open shell whose position-duplicate free edges pass\nthe by-edge-id validation gate (every edge id used ≤2×, so a\npositionally-unpaired edge is invisible), and the broken result is\nreturned as \"valid\".\n\nThe gridfinity interior dovetail tile hits exactly this: with all four\nedges set to `join`, the corner-rounding profile degenerates to a plain\nbox matching the slab bounds, and the identity intersect collapsed 134\nfaces → 38 with 32 free edges. Captured stage probes (`serializeSolid`\nat each generator milestone) localized it; the same operands replay\nclean through `boolean()` (containment shortcut) and broken through the\nevolution path.\n\n**Fix**: extract `boolean()`'s identical/containment detection into\n`detect_trivial_relation` (verbatim move — the logic is unchanged and\n`boolean()`'s behavior is identical) and consult it in\n`boolean_with_evolution` before the faithful path. Trivial pairs route\nthrough `boolean()`'s shortcuts, where the geometry-heuristic evolution\nattributes a copied result's faces exactly (1:1 normal+centroid match).\n\n## Verification\n\n- New fixture tests\n(`crates/io/tests/dovetail_interior_identity_intersect_inmem.rs`, the\ntool's exact serialized 4×4-interior-tile operands): identity intersect\nwatertight with all 134 faces + 64 pocket cones preserved through BOTH\nentries, and the evolution map still attributes faces.\n- `cargo test -p brepkit-wasm --lib gridfinity` 27/27; full workspace\ngreen; clippy `-D warnings` clean.\n- **Tool-verified with a local overlay**: dovetailKey **2/2** (was 1/2),\nfit-offset **2/2** (was 0/2), dovetail suite **6/9** (was 2/9) — 5×4\nmiddle-column, 4×4 interior (both variants), inverted, and magnet tiles\nall closed. Remaining failures (fractional edge tile bnd=4,\ndoubled-dovetail nm=21, A1-corner nm=3) are separate residuals.\n\n## Notes for review\n\n- The existing gridfinity evolution-path fixtures (socket/lip fuses) use\noverlapping operands — `detect_trivial_relation` returns all-false for\nthem, so the faithful provenance path is unchanged there.\n- The detection adds O(faces + vertices) work per evolution boolean\n(classifier build + AABB + vertex classification), negligible against\nGFA cost.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRoutes identical/contained operand pairs around the evolution GFA path\nin `boolean_with_evolution` to stop mis-splitting on coincident walls.\nFixes identity-intersect failures in gridfinity dovetail tiles\n(bnd=108/144) while preserving full face provenance.\n\n- **Bug Fixes**\n- Extracted identical/containment detection into\n`detect_trivial_relation` and use it in both `boolean` and\n`boolean_with_evolution`; skip detection when `a == b`; trivial pairs\ntake `boolean`’s copy/empty shortcuts before evolution attribution.\n- Added fixtures and\n`crates/io/tests/dovetail_interior_identity_intersect_inmem.rs`;\nidentity intersect stays watertight (134 faces, 64 cones) through both\npaths and the evolution map attributes every copied slab face.\n\n<sup>Written for commit 1df55c29514e00e3494151578f621c08ed5ec629.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1057?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-08T19:05:21Z",
          "tree_id": "18280406697691a813cc119e37068e1084ef0227",
          "url": "https://github.com/andymai/brepkit/commit/d2a98fcd5f2b4b43b706b35681ca247433303382"
        },
        "date": 1783537668773,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 782602,
            "range": "± 2114",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 880243,
            "range": "± 8959",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13111,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634621,
            "range": "± 678",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20327828,
            "range": "± 29399",
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
          "id": "6bca50dd6b76f1ef56ba9f258a368f8d337fd26e",
          "message": "chore(main): release 2.124.13 (#1058)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.124.13](https://github.com/andymai/brepkit/compare/v2.124.12...v2.124.13)\n(2026-07-08)\n\n\n### Bug Fixes\n\n* **operations:** route trivial operand pairs around the evolution GFA\npath ([#1057](https://github.com/andymai/brepkit/issues/1057))\n([d2a98fc](https://github.com/andymai/brepkit/commit/d2a98fcd5f2b4b43b706b35681ca247433303382))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes operations by routing trivial operand pairs around the evolution\nGFA path to avoid unnecessary evolution and edge-case errors. Releases\n`brepkit-wasm` 2.124.13.\n\n<sup>Written for commit 8739366c325dcfbd1e71293cae24652ca63af060.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1058?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-08T19:12:33Z",
          "tree_id": "f8ae210cf9833df71eeaf0c1fde17df83b458b4d",
          "url": "https://github.com/andymai/brepkit/commit/6bca50dd6b76f1ef56ba9f258a368f8d337fd26e"
        },
        "date": 1783538089911,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 753739,
            "range": "± 1182",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 845457,
            "range": "± 1065",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11948,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 633483,
            "range": "± 5449",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19512442,
            "range": "± 272605",
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
          "id": "4fe072fc086c3ebc2094b24c13e715884b2baf89",
          "message": "feat(wasm): capture panic text for post-poison diagnosis (#1059)\n\n## Root cause\n\n`wasm32-unknown-unknown` compiles with `panic=abort` (verify: `rustc\n--print cfg --target wasm32-unknown-unknown`). A panic inside any\n`BrepKernel` method therefore traps inside the wasm-bindgen shim's\n`WasmRefCell` borrow scope: the `RefMut` never drops, the object's\nborrow flag stays locked forever, and every subsequent call throws\n\"recursive use of an object detected which would lead to unsafe aliasing\nin rust\" — masking the original panic text. `catch_unwind` is\nstructurally inert on this target, so the four existing manual wrap\nsites (fillet x2 in `bindings/operations.rs`, `compound_cut` in\n`bindings/booleans.rs`, fillet in `bindings/batch.rs`) protect only\nnative test consumers, and the `#[wasm_binding]` proc-macro is unwired\n(not a dependency of `brepkit-wasm`, zero usages, and its poisoned-path\nmessage references a nonexistent `reset()`). No Rust code can reset the\nborrow flag; the only recovery is a new `BrepKernel`.\n\nThe one thing that CAN be saved is the panic text: the panic hook still\nruns before the abort.\n\n## Fix\n\nNew `crates/wasm/src/panics.rs`:\n- `install_hook()` — idempotent chained panic hook, installed by\n`BrepKernel::new` (and thus `Default`). Records message + location into\na static and mirrors it to `console.error` as `[brepkit] panic: ...`\nbefore the abort, so the root cause survives JS callers that swallow the\ntrap's `RuntimeError`.\n- Free functions `lastPanicMessage()` / `clearLastPanicMessage()` —\nremain callable after the kernel object is poisoned (they never touch\nits borrow flag), so JS can read the root-cause text post-mortem.\n\nAlso swept: zero bare `Instant::now()` reachable from wasm (all\ncfg-gated `timer_now` or test-only).\n\nThe 2026-07-08 combinedFeatures \"handles + label (back skip)\"\nswallowed-panic case does NOT reproduce on stock 2.124.13 (independent\noverlay re-run: 1/1 structural pass, ~154s, no panic) — likely closed by\nthe #1054/#1057 GFA merges. If the family resurfaces it now self-reports\nvia `lastPanicMessage()`. Roadmap and wasm-bindings skill updated with\nthe abort semantics and the diagnosis recipe.\n\n## Verification\n\n- New regression test: `crates/wasm/src/panics.rs`\n`tests::hook_records_panic_message_and_location` (hook records panic\nmessage + location; clear resets)\n- `cargo test -p brepkit-wasm`: 229 passed, 0 failed\n- Canary: `cargo test -p brepkit-wasm --lib gridfinity`: 27 passed\n- `cargo build -p brepkit-wasm --target wasm32-unknown-unknown`: clean\n- `cargo fmt --all --check`, `cargo clippy --all-targets -- -D\nwarnings`, `./scripts/check-boundaries.sh`: all clean\n- Pre-push hook (full workspace tests + cargo-deny): passed\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdds a panic-capture hook for wasm so panic text survives `panic=abort`\nand can be read from JS after a poisoned kernel, improving post-mortem\ndebugging without changing op behavior.\n\n- **New Features**\n- New `crates/wasm/src/panics.rs` installs a chained panic hook (from\n`BrepKernel::new`); records the last panic and mirrors it to\n`console.error` as “[brepkit] panic: …”.\n- Adds `lastPanicMessage()` and `clearLastPanicMessage()` free\nfunctions; remain callable after borrow poisoning.\n- Exposes the module via `pub mod panics`; docs note\n`wasm32-unknown-unknown` is `panic=abort` and `catch_unwind` is inert on\nwasm.\n\n- **Bug Fixes**\n- Stabilized the regression test by serializing access to the\nprocess-global panic state to avoid parallel test races.\n\n<sup>Written for commit 01f7728f6fa2212e458e33426a8ddbb789ddd861.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1059?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T20:45:07Z",
          "tree_id": "18771963ad88f361d6e353f7850cc89ba2f22f41",
          "url": "https://github.com/andymai/brepkit/commit/4fe072fc086c3ebc2094b24c13e715884b2baf89"
        },
        "date": 1783716446212,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 757775,
            "range": "± 908",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 845992,
            "range": "± 12228",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12073,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 637833,
            "range": "± 1834",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19489802,
            "range": "± 1442820",
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
          "id": "3fc10a29db913d1f346249d757ad3552f42b1299",
          "message": "chore(main): release 2.125.0 (#1064)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.125.0](https://github.com/andymai/brepkit/compare/v2.124.13...v2.125.0)\n(2026-07-10)\n\n\n### Features\n\n* **wasm:** capture panic text for post-poison diagnosis\n([#1059](https://github.com/andymai/brepkit/issues/1059))\n([4fe072f](https://github.com/andymai/brepkit/commit/4fe072fc086c3ebc2094b24c13e715884b2baf89))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.125.0 adds panic text capture in `brepkit-wasm` to improve\npost-poison diagnosis and make failures easier to debug.\n\n<sup>Written for commit 043ad6d6c73b0c4501d152e49866bac8ba1d28c2.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1064?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T20:50:32Z",
          "tree_id": "a8ba2521e9746ce852a712433e4dde8e59702052",
          "url": "https://github.com/andymai/brepkit/commit/3fc10a29db913d1f346249d757ad3552f42b1299"
        },
        "date": 1783716781168,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 762094,
            "range": "± 1672",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 846445,
            "range": "± 1071",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12025,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634763,
            "range": "± 1552",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19562690,
            "range": "± 188538",
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
          "id": "f44d487a9bf4615fdc34e62a60961dfca5fceac2",
          "message": "fix(blend): propagate trimmer edge splits into neighbor face wires (#1060)\n\n## Root cause\n\n`split_edge_at` in `crates/blend/src/trimmer.rs` rebuilt only the\ntrimmed face's wire when a fillet contact curve crossed a boundary edge\nmid-edge. A neighbor cap/rim face sharing that boundary edge kept\nreferencing the stale unsplit edge, so the stale edge and the kept\nsub-edge each ended up used by exactly one face — opening the shell\nalong the shared span (box single-edge `fillet_v2`: 16 free B-Rep edges,\n28 boundary mesh edges at export tolerance).\n\n## Fix\n\n- New `propagate_split` rewrites **every** wire referencing the split\nedge onto its two sub-edges, honoring each occurrence's traversal\norientation (snapshot-then-allocate over `topo.wires()`).\n- `trim_face`'s closing contact-edge orientation was inverted: chain1\nruns va→…→vb, so the va→vb contact edge must close it **reversed** —\ntrimmed wires were silently disconnected head-to-tail.\n- `trim_face_general`'s inline edge splits are now routed through\n`split_edge_at`, so they propagate too.\n- Same-edge double-hit now bails **before** any wires are mutated.\n\n## Verification\n\n- New unit test\n`crates/blend/src/trimmer.rs::tests::split_propagates_into_neighbor_wire`\n— trims a square with an attached neighbor face traversing the shared\nedge reversed; asserts no stale-edge refs, wire connectivity, and\nexactly one shared sub-edge.\n- New regression test\n`crates/operations/tests/regress_blend_trim_neighbor_split.rs::fillet_v2_box_edge_propagates_boundary_splits`\n— fails on main exactly on \"stale pre-split edge still referenced\"\n(independently re-verified by swapping in main's trimmer.rs); passes\nwith the fix. Asserts stale refs 0, no over-shared edges, all wires\nconnected, end faces gain both split vertices (4→6 edges), and boundary\nmesh edges drop below the pre-fix 28.\n- Measured: free B-Rep edges 16→12, boundary mesh edges 28→22 at export\ntolerance (0.01mm/5°). Residual 12 free edges are separately\ncharacterized v2 trim gaps, documented in the roadmap row (keep-side\nselection degeneracy, `create_blend_face` duplicate contact edges,\nmissing end-cap notch trim, chamfer_v2 external-tangent branch).\n- `cargo fmt` clean; `clippy --all-targets -D warnings` clean;\n`check-boundaries.sh` pass; brepkit-blend 94/94; canary `cargo test -p\nbrepkit-wasm --lib gridfinity` 27/27.\n- Roadmap row updated: latent → CLOSED, with the four discovered\nstill-open v2 trim gaps recorded.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPropagates boundary-edge splits from the blend trimmer into all neighbor\nface wires to prevent open shells and T-junction cracks. Also fixes\ncontact-edge orientation, ensures inline splits propagate, rejects\nrepeated-edge hits, and cleans up stale pcurves.\n\n- **Bug Fixes**\n- New `propagate_split` rewrites every wire referencing the split edge\nonto its two sub-edges, preserving each occurrence’s orientation; inline\nboundary splits now route through `split_edge_at` so they propagate.\n- Fixed `trim_face` closing contact-edge orientation (chain1 closes with\n`va→vb` reversed).\n- Rejects same-edge and seam-style repeated-edge double hits before any\nwire mutations.\n- Drops stale per-face pcurve entries for the replaced edge so\nregistries can’t reference the old span.\n- Added unit and regression tests; box single-edge `fillet_v2`: free\nedges 16→12, boundary mesh edges 28→22, stale-edge refs 0.\n\n<sup>Written for commit 2293e26d3b789551eb7688a68d138f9f6ada7213.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1060?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T20:50:36Z",
          "tree_id": "29c5f3fc79802a2b7b8c3ff4c52bfe48e57e31ab",
          "url": "https://github.com/andymai/brepkit/commit/f44d487a9bf4615fdc34e62a60961dfca5fceac2"
        },
        "date": 1783716927225,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 758095,
            "range": "± 1320",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 845211,
            "range": "± 3322",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11881,
            "range": "± 108",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 636465,
            "range": "± 957",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19528518,
            "range": "± 884122",
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
          "id": "775774723b537c4dfcbaf51afdde7037e1df5cfb",
          "message": "chore(main): release 2.125.1 (#1065)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.125.1](https://github.com/andymai/brepkit/compare/v2.125.0...v2.125.1)\n(2026-07-10)\n\n\n### Bug Fixes\n\n* **blend:** propagate trimmer edge splits into neighbor face wires\n([#1060](https://github.com/andymai/brepkit/issues/1060))\n([f44d487](https://github.com/andymai/brepkit/commit/f44d487a9bf4615fdc34e62a60961dfca5fceac2))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.125.1 for `brepkit-wasm`, fixing blend topology by propagating\ntrimmer edge splits to neighboring face wires. This prevents desynced\nface wires during blends and addresses #1060.\n\n<sup>Written for commit 7cb2d143aba09a0b5a65cfca69afa49404e2a015.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1065?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T20:57:01Z",
          "tree_id": "8d5e68997548ea99e2f7a477842b881ba2df2e66",
          "url": "https://github.com/andymai/brepkit/commit/775774723b537c4dfcbaf51afdde7037e1df5cfb"
        },
        "date": 1783717167629,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 768080,
            "range": "± 1231",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 853258,
            "range": "± 1466",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12111,
            "range": "± 124",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639297,
            "range": "± 15200",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19905334,
            "range": "± 120368",
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
          "id": "5011607dfdb1b142005b532627d944287bbdd67b",
          "message": "fix(operations): make mesh-boolean fallback output conforming and manifold (#1061)\n\n## Root cause\n\nThe mesh-boolean safety net (the path `boolean()` falls back to when the\nanalytic GFA result fails its gates) itself emitted open / non-manifold\nmeshes on coincident-wall contact. Three stacked defects:\n\n1. **T-junctions:** the splitter fan-split each intersected triangle in\nisolation — an on-edge insertion point was never propagated to the\nneighbor triangle sharing that edge, so co-refined triangles did not\nconform across shared edges.\n2. **Coplanar contact collapsed:** contact between coplanar triangles\nwas reduced to a single \"longest segment\" instead of full mutual edge\nclipping, so neither mesh conformed to the other's edges inside the\nshared plane.\n3. **Winding coin-flip:** the generalized-winding-number classifier was\nconsulted for sub-triangles lying exactly on the other mesh's surface,\nwhere winding is exactly 1/2 — inside/outside became a coin flip on\nshared walls.\n\nOn the captured dovetail relief pair the raw fallback emitted bnd=11 and\nthe full `boolean()` export bnd=15 nm=1; the same path produced the\nnm=76 integer-width socket-fuse exports.\n\n## Fix\n\nRewrite `crates/operations/src/mesh_boolean.rs` as conforming\nco-refinement:\n\n- Per-host CDT re-triangulation with a **global cross-triangle\nedge-point map**, so points landing on a triangle edge are fed to the\nneighbor sharing that edge (no T-junctions). The legacy fan split\nremains only as a per-triangle fallback when CDT fails.\n- **Mutual coplanar edge clipping** (`coplanar_corefine_segments`): each\nmesh conforms to the other's edges inside a shared plane.\n- Explicit **`OnSame` / `OnOpp` coincident-surface classification**\nconsumed by assembly (mesh A owns the single kept copy of any coincident\nregion), instead of asking the winding number where it is exactly 1/2.\n- `MeshBooleanResult` self-reports position-welded boundary /\nnon-manifold edge counts; `boolean/mod.rs::mesh_boolean_fallback`\nwarn-logs a non-watertight fallback result instead of consuming it\nsilently.\n- The issue-#696 planar-midpoint-drop metadata path\n(`mesh_boolean_with_metadata` + `infer_planar_triangle_flags`) is\ndeleted — conforming splits subsume it, and all #696-era tests pass.\n\n## Verification\n\n- New fixture regression test\n`crates/io/tests/relief_meshbool_fallback_inmem.rs` on the tool's exact\nserialized operands (`crates/io/tests/data/relief_tongue.bin` /\n`relief_cutter.bin`):\n- `relief_cut_raw_mesh_boolean_is_manifold` — raw `mesh_boolean` path,\nbnd=0 nm=0 with self-check flags\n- `relief_cut_boolean_fallback_export_is_manifold` — full `boolean()`\nfallback export at export tolerance, quantized health check + volume\nsanity\n- New unit tests in `mesh_boolean.rs`:\n`mesh_boolean_overlapping_cut_watertight`,\n`mesh_boolean_coincident_wall_cut`, `mesh_boolean_coincident_wall_fuse`,\n`mesh_boolean_coplanar_top_stack_fuse` (9/9 lib tests pass)\n- Fixed build emits bnd=0 nm=0 at every quantization grid 1e-3..1e-6,\nverified for all 6 relief-op operand variants (fix proven load-bearing\nby A/B against a rebuilt baseline)\n- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,\n`./scripts/check-boundaries.sh` all clean\n- Canary `cargo test -p brepkit-wasm --lib gridfinity`: 27/27\n- Full workspace suites green (render compute suite passes standalone; a\nparallel-run wgpu contention SIGSEGV is environmental)\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRewrite the mesh-boolean fallback to produce closed, 2‑manifold,\nconforming results on coincident-wall contact. Also clamp coincident\nclassification to the co‑refinement tolerance to keep near‑disjoint\nfuses intact and warn on non‑watertight output.\n\n- **Bug Fixes**\n- Conforming co-refinement: per-triangle CDT re-triangulation with a\nglobal edge-point map to remove T-junctions.\n- Mutual coplanar edge clipping so each mesh conforms to the other's\nedges in shared planes.\n- Explicit coincident-surface classes (`OnSame`/`OnOpp`) to avoid\nwinding-number 1/2 coin flips; mesh A owns coincident regions.\n- `MeshBooleanResult` reports boundary/non-manifold edge counts;\n`boolean()` fallback now warns on non-watertight output.\n- Clamp `OnSame`/`OnOpp` on-surface window to the intersection tolerance\nto avoid dropping facing walls on near-disjoint solids; add a\nnear-disjoint fuse regression test.\n- Share `COINCIDENT_DEDUPE_GRID` between coincident-triangle dedupe and\nthe output self-check; scale the crossing-resolution round cap with\nsegment count and warn on exhaustion.\n\n- **Migration**\n- Remove `mesh_boolean_with_metadata` and `infer_planar_triangle_flags`.\nCall `mesh_boolean(...)` directly; conforming splits subsume the old\nplanarity path.\n\n<sup>Written for commit 9317980e418d4ca6257d666d529147f49c009b10.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1061?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T20:59:13Z",
          "tree_id": "b02eb4f3365e2017e8ca4d9f0a7df3de50ee94f3",
          "url": "https://github.com/andymai/brepkit/commit/5011607dfdb1b142005b532627d944287bbdd67b"
        },
        "date": 1783717318108,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 782315,
            "range": "± 1475",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 881054,
            "range": "± 1286",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13275,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 632966,
            "range": "± 1464",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20670813,
            "range": "± 42785",
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
          "id": "6126459fd348528b4892a9d361a4d58396a93085",
          "message": "chore(main): release 2.125.2 (#1066)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.125.2](https://github.com/andymai/brepkit/compare/v2.125.1...v2.125.2)\n(2026-07-10)\n\n\n### Bug Fixes\n\n* **operations:** make mesh-boolean fallback output conforming and\nmanifold ([#1061](https://github.com/andymai/brepkit/issues/1061))\n([5011607](https://github.com/andymai/brepkit/commit/5011607dfdb1b142005b532627d944287bbdd67b))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.125.2 for `brepkit-wasm` fixes the mesh-boolean fallback to\nensure conforming, manifold meshes. This improves the reliability of\nboolean operations in edge cases.\n\n- **Bug Fixes**\n- Mesh-boolean fallback now outputs conforming, manifold meshes to\nprevent non-manifold artifacts.\n\n<sup>Written for commit 6f2440a57beff6edbeac15b336dcc3698ca4c265.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1066?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T21:06:10Z",
          "tree_id": "818d33f71a0e6d03c74eebc46e6321864537d502",
          "url": "https://github.com/andymai/brepkit/commit/6126459fd348528b4892a9d361a4d58396a93085"
        },
        "date": 1783717717113,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 776307,
            "range": "± 1567",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 875911,
            "range": "± 20283",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13111,
            "range": "± 147",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631576,
            "range": "± 764",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20388650,
            "range": "± 33788",
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
          "id": "8b783b78e5721a63a894e82c6ea0413b69b674ae",
          "message": "feat(operations): merge analytic revolve segments — apex cone, annulus caps, partial-turn torus (#1062)\n\n## Root cause\n\nThe analytic full-revolution fast path over-segmented three profile\nshapes even though every face stayed exact analytic:\n\n- **Pointed cone (apex on axis):** an apex-touching profile edge\nrevolved into 4 on-axis NURBS wall patches instead of one\ndegenerate-seam cone wall (census: 12 faces).\n- **Washer (off-axis rectangle):** each annulus cap was split into 4\nplanar sectors instead of one plane face carrying the smaller rim as a\nhole wire (census: 16 faces).\n- **Partial turn of a circle profile:** fell off the analytic path\nentirely instead of building one trimmed `Torus` band plus two disc\ncaps.\n\nFixing these exposed three latent defects downstream:\n\n1. Torus bands with a doubled seam cracked under tessellation — CDT\ndegenerates on the fully-wrapping UV image and the snap path re-samples\nrims independently.\n2. `planar_cap_signed_volume` trusted stored hole-wire winding; a\nboolean's same-wound inner rim **added** its disc (drilled tube read\n1487 vs the true 420π ≈ 1319.47).\n3. `analytic_torus_signed_volume` used raw `[v_min, v_max]` for the\nperiodic tube angle, so a rim at z = −1.8e−16 projecting to v = 2π−ε\nintegrated the complementary arc (647.53 vs 888.26).\n\n## Fix\n\n- `revolve.rs`: classify profile edges for the analytic path —\napex-touching walls build the degenerate seam wire; annulus caps keep\nthe smaller rim as a hole wire with winding-derived orientation; a\npartial-turn circle profile builds one trimmed `Torus` band + 2 disc\ncaps with exact `V = π·R·ρ²·Δu` via the new\n`partial_torus_sector_volume` (seam-arc sweep angle cross-checked\nagainst the cap-plane dihedral).\n- `tessellate/nonplanar.rs`: new `tessellate_torus_two_rim_band` —\nstructured band from shared rim-pool vertices, handling both rim\norientations (constant-v latitude rims and constant-u tube rims); sweep\nside chosen by the doubled seam arc's midpoint.\n- `measure/volume.rs`: `planar_cap_signed_volume` now subtracts holes by\nmagnitude (winding-agnostic — a hole is inside the outer by definition);\nthe torus integrator picks the periodic v-range gap-wise like u, keeping\nthe 2-sample >π ambiguity guard.\n\n## Verification\n\n- Census (`approx_census`, `revolve_matrix`): pointed cone 12→2 faces,\nwasher 16→4, half-disc 8→2, new `circle 120° (trimmed Torus)` row at 3\nfaces; frustum/cylinder unchanged at 3 — all rows exact analytic, none\nregressed.\n- New/strengthened fixture tests in `crates/operations/src/revolve.rs`:\n- `revolve_circle_partial_turn_is_trimmed_torus` — 3-face topology,\nvalidate, watertight at 2 deflections, mesh-signed-volume material\ncheck, exact volume to 1e-9\n- `revolve_arc_profile_reversed_edge_torus_band` — reversed-arc-edge\nprofile; pins the torus v-seam integrator bug\n- `revolve_pointed_cone_apex_band_volume_is_exact` — 2-face apex-merged\ntopology, watertight, volume tolerance tightened 1e-3 → 1e-9\n- `revolve_washer_walls_are_exact_cylinders` — 4 faces, annulus caps\ncarry hole wires, watertight, exact volume\n- `revolve_arc_profile_edge_is_torus_band` — 2 faces, one periodic torus\nband, watertight\n- `cargo fmt --check`, `clippy --all-targets -D warnings`,\n`check-boundaries.sh` clean; gridfinity canary 27/27; operations volume\n+ tessellate suites green.\n\nDiscovered and roadmap-logged (not fixed here): both ray-cast\nclassifiers misread a torus band whose wire has <3 distinct vertices as\na full surface, so interior points classify Outside; repro is the\npartial-turn test's 3-face solid.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nMerged the analytic revolve path to produce minimal exact faces for apex\ncones, annulus caps, and partial‑turn circle profiles, with NURBS‑arc\nseam support and a one‑sided guard for axis‑straddling profiles. This\ncuts face counts and keeps meshes watertight with exact volumes.\n\n- **New Features**\n- Apex-touching wall builds one periodic cone with a degenerate seam\n(12→2 faces).\n- Annulus caps are single planar faces that keep the smaller rim as a\nhole; walls reuse the original profile edge as the seam, and face\nnormals come from the profile’s (radial, axial) winding (washer 16→4).\n- Partial-turn circle builds one trimmed `Torus` band plus two disc caps\nwith exact volume V = π·R·ρ²·Δu; the structured torus-band tessellator\nuses shared rim vertices, accepts NURBS-circle seams, supports both rim\norientations, and picks the swept side via the seam arc’s midpoint.\n\n- **Bug Fixes**\n- `planar_cap_signed_volume` now subtracts hole wires by magnitude\n(fixes same-wound inner rims adding area).\n- `analytic_torus_signed_volume` picks the periodic v-range gap-wise to\navoid integrating the complementary tube arc near v ≈ 2π.\n- Torus band tessellation avoids CDT/snap cracks on fully-wrapping UVs\nand accepts NURBS seam edges; the analytic path defers for\naxis‑straddling profiles to prevent wrong outer/inner rim selection.\n\n<sup>Written for commit ccbfd5b8699767bdf51e1f20b105a33d31db542c.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1062?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T21:06:55Z",
          "tree_id": "ab72dda93adde18941dbb8427da9be70134c412b",
          "url": "https://github.com/andymai/brepkit/commit/8b783b78e5721a63a894e82c6ea0413b69b674ae"
        },
        "date": 1783717878169,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 765867,
            "range": "± 1259",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 851103,
            "range": "± 1749",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11977,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634404,
            "range": "± 4434",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19629583,
            "range": "± 90941",
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
          "id": "a633c5fc947315c7a4fe69b03672b22beff84412",
          "message": "fix(algo): close the dovetail tongue-relief cut family (#1063)\n\n## Root cause\n\nThe doubled-dovetail baseplate interior tile exported STLs with nm=21:\neach relieved nub — `cut(trapezoid tongue prism, tapered socket pocket)`\n— arrived at the connector fuse already broken (bnd=13-15, nm=1-2 per\nnub, through BOTH `boolean()` and `boolean_with_evolution`, from\nwatertight operands). Four stacked roots, fixed bottom-up:\n\n1. **Graze-refinement gap** (`restrict_curves_to_faces`): the 24-sample\nin-both test dropped a real socket-mouth corner circle crossing (~8°\nsubtended on a 2 mm tongue face) as a \"tangency\". Dropping it gapped the\nsection chain at the corner; the whole top face then classified by a\nsingle interior point and the cut collapsed to an open hole shell.\n2. **Unclipped open conic sections**: an open marched-NURBS conic (plane\n× cone) spans the whole cone extent; the generic sample-clip kept it\nwhole and the downstream splitter never trims an open curved section to\na plane face's boundary, so the face never partitioned.\n3. **No analytic cone path in the ray-cast classifier**: the cutter's\ntapered corner patches fell to the flat Newell-polygon fallback, which\nmis-counts crossings for sub-face interior points ~0.2 mm inside the\npocket walls — two in-chunk pieces classified Outside and were kept (10\nfaces, 8 bad-use edges).\n4. **Tessellation NURBS edge orientation**: GFA section edges can store\ntraversal-order vertices over an unreversed NURBS curve; samplers\ntrusting `oe.is_forward()` plus natural domain order folded the boundary\npolyline back on itself (mesh nm=6-11 on a B-Rep-clean 8-face result).\n\n## Fix\n\n- `phase_ff.rs`: graze test now refines with a sample density scaled to\nthe smaller face extent before dropping (a true point tangency stays\nsub-segment at any resolution and is still dropped). New\n`trim_open_curve_to_plane_face_lines` clips open marched-NURBS conics to\nexact crossings with the plane face's straight boundary edges AND the\ncone partner's angular-window rulings, trimming the stored NURBS to each\nkept span via `trim_nurbs_to_span` (consumers normalize over\n`domain_with_endpoints`, which for a NURBS is the full knot domain).\n- `face_splitter`: `find_splits_on_nurbs_section` — T-junction splits on\nmarched-NURBS sections by sampled point-to-curve projection (the\nchord-based line search misses a junction that lies on the curve but off\nits chord).\n- `classifier/ray_cast.rs`: analytic `FaceGeom::Cone` path —\nray/double-cone quadratic filtered to the face's slant range (which also\nrejects mirror-nappe hits) and angular patch, mirroring the partial-arc\ncylinder precedent.\n- `tessellate/edge_sampling.rs` + `planar.rs`: `nurbs_runs_end_to_start`\nendpoint-alignment check so samplers orient NURBS edges by the edge's\nstart/end vertices, not knot-domain order alone. (Normalizing vertex\norder at the minting site instead regressed the calibrated torus-box\nnotch landscape — documented as do-not-retry in the fixture doc and\nroadmap.)\n\n## Verification\n\n- New regression fixtures from the tool's exact serialized operands (two\nnub positions, tip + mirrored flank):\n`crates/io/tests/dovetail_relief_cut_inmem.rs::relief_cut_tip_nub` and\n`::relief_cut_flank_nub` — assert the 8-face analytic nub with one\nsurviving cone face, zero free edges, and a watertight/manifold mesh\n(bnd=0, nm=0) through both boolean entries. Was bnd=13-15, nm=1-2.\n- `cargo fmt --all --check`, `cargo clippy --all-targets -- -D\nwarnings`, `./scripts/check-boundaries.sh` green.\n- Canary: `cargo test -p brepkit-wasm --lib gridfinity` 27/27.\n- Spot-checks: brepkit-algo suite 150/150, operations tessellate lib\ntests 71/71, curved boolean parity corpus green. approx_census\nunchanged.\n- Post-merge follow-up: the doubled-dovetail tool suite needs a fresh\noperand capture on a kernel with this fix (old stage fixtures embed\npre-fix broken nubs), tracked in the roadmap skill.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes broken dovetail tongue‑relief cuts by correctly clipping and\nclassifying cone/plane sections, producing a watertight 8‑face nub.\nEliminates bnd/nm errors for doubled‑dovetail baseplates in both\n`boolean()` and `boolean_with_evolution`.\n\n- **Bug Fixes**\n- Clip open marched‑NURBS conics (plane × cone) to exact plane‑edge and\ncone angular‑window crossings; trim NURBS to kept spans; treat on‑edge\nsamples as hits; defer to generic clip if trim fails.\n- Split NURBS section T‑junctions via sampled point‑to‑curve projection;\nreturn splits in edge order; forward/reversed tests added.\n- Add analytic cone path to the ray‑cast classifier (quadratic with\nslant‑range and angular filtering) to avoid miscounted crossings on\ntapered patches.\n- Orient NURBS edge sampling by endpoint alignment to prevent folded\nboundary polylines in meshes.\n- Refine the graze test with density scaled to the smaller face extent\nto keep real short crossings.\n- Add regression fixtures\n(`crates/io/tests/dovetail_relief_cut_inmem.rs` + operand bins)\nasserting the 8‑face nub, zero free edges, and watertight mesh across\nboth boolean paths.\n\n<sup>Written for commit 6c8a56a32b19caa88f2cce98eb29c93f93b69459.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1063?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T21:11:52Z",
          "tree_id": "1ef2ca27c21608da54b267d8e5b21f3995d5be1a",
          "url": "https://github.com/andymai/brepkit/commit/a633c5fc947315c7a4fe69b03672b22beff84412"
        },
        "date": 1783718060139,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 735841,
            "range": "± 7505",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 823626,
            "range": "± 958",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12057,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 597485,
            "range": "± 5141",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19112766,
            "range": "± 31609",
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
          "id": "329ba35395845d6dbfe365b238d47d09a4041204",
          "message": "chore(main): release 2.126.0 (#1067)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.0](https://github.com/andymai/brepkit/compare/v2.125.2...v2.126.0)\n(2026-07-10)\n\n\n### Features\n\n* **operations:** merge analytic revolve segments — apex cone, annulus\ncaps, partial-turn torus\n([#1062](https://github.com/andymai/brepkit/issues/1062))\n([8b783b7](https://github.com/andymai/brepkit/commit/8b783b78e5721a63a894e82c6ea0413b69b674ae))\n\n\n### Bug Fixes\n\n* **algo:** close the dovetail tongue-relief cut family\n([#1063](https://github.com/andymai/brepkit/issues/1063))\n([a633c5f](https://github.com/andymai/brepkit/commit/a633c5fc947315c7a4fe69b03672b22beff84412))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T21:18:13Z",
          "tree_id": "0b29272bed3cf81db5a412ddda19ef2ee6ec0cee",
          "url": "https://github.com/andymai/brepkit/commit/329ba35395845d6dbfe365b238d47d09a4041204"
        },
        "date": 1783718435503,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 755367,
            "range": "± 1604",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 843844,
            "range": "± 14038",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11952,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 640699,
            "range": "± 1536",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19441957,
            "range": "± 23095",
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
          "id": "6e21bdfab809f4d54288a5678aed398b1e2cfeac",
          "message": "fix(check,operations): classify trimmed-torus bands correctly (#1068)\n\n## Root cause\n\nThe trimmed-torus band produced by a partial-turn circle revolve (R=6,\nrho=2, 2pi/3 sweep; 3-face solid) misclassified 7/12 probe points —\nevery interior point read Outside in BOTH the check and operations\nray-cast classifiers. Three stacked roots, all instrumentation-verified:\n\n1. **Both local Ferrari ray-torus quartic solvers were numerically\nbroken** (`crates/check/src/classify/ray_surface.rs`,\n`crates/operations/src/classify.rs`): for oblique irrational rays at\nmoderate radii they missed real roots (zero roots for rays cast from\ninside the tube) AND emitted off-surface spurious roots (a \"hit\" at\nz=4.6 on a torus spanning z in [-2, 2]), flipping crossing parity. The\nsmall axis-aligned unit tests never exposed either failure mode.\n2. **check-only:** `face_aabb` (`crates/check/src/util.rs`) collapsed\neach cap disc — a single closed-circle wire has ONE vertex, and Plane\nsurfaces get no curvature expansion — to a point AABB, so the\nclassifier's BVH prefilter never offered the caps and their ray\ncrossings were silently dropped from the parity count.\n3. **operations-only:** `boolean::face_polygon` samples closed rims from\nthe curve's own parameter origin, so the band wire's two rim circles\nentered the periodic unwrap at incoherent phases and the UV boundary\nsheared into a self-inconsistent parallelogram that rejected real band\nhits.\n\nThe previously hypothesized mechanism (\"<3 distinct vertices trips the\ndegenerate full-surface branch\") was wrong — both crates densify closed\nedges into a 66-point polygon and the UV containment itself worked.\n\n## Fix\n\n- Both classifiers now delegate ray-torus intersection to\n`brepkit_math::analytic_intersection::intersect_line_torus`\n(residual-verified Durand-Kerner with Newton polish); the dead local\ncubic/quartic solvers are deleted.\n- `face_aabb` expands per-edge via `expand_aabb_for_curve`: exact\nfull-curve extent for circles/ellipses (conservative superset for\npartial arcs), control-point hull for NURBS.\n- A seam-anchored boundary sampler local to `classify.rs` keeps\nconsecutive closed edges phase-coherent through the periodic unwrap.\n`boolean::face_polygon` is untouched — its sampling is calibrated for\nband-fragment sharing.\n\nDiscovered, deliberately left open (roadmap row updated): the algo\nray-cast classifier (`crates/algo/src/classifier/ray_cast.rs`) has no\nTorus arm at all — torus faces fall to the flat Newell-polygon fallback.\nCalibrated boolean landscapes pin its current behavior, so it needs its\nown re-probe before adding the arm.\n\n## Verification\n\n- New regression tests: `partial_turn_torus_band_classification` +\n`full_turn_torus_classification` (operations),\n`partial_torus_band_interior_points` +\n`face_aabb_covers_closed_circle_boundary` (check),\n`ray_torus_oblique_from_inside_tube` (check ray_surface);\n`revolve_circle_partial_turn_is_trimmed_torus` now asserts ray-cast\nprobes directly.\n- 12/12 probe points correct in both stacks (was 5/12).\n- `cargo fmt --all --check`, `cargo clippy --all-targets -- -D\nwarnings`, `scripts/check-boundaries.sh` clean.\n- brepkit-check 47/47; operations classify 22/22, revolve 25/25;\ngridfinity canary 27/27; full `cargo test --workspace --release` exit 0.\n- Rebased onto 2.126.0 release head; all of the above re-verified\npost-rebase.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes ray-cast misclassification of partial-turn circle revolves\n(trimmed torus band + two disc caps) in both `check` and `operations`.\nInterior points now classify correctly as Inside.\n\n- **Bug Fixes**\n- Replaced local torus quartic with\n`brepkit_math::analytic_intersection::intersect_line_torus`; removed\nbroken cubic/quartic solvers.\n- `check`: `face_aabb` now expands per-edge extents (exact for\ncircles/ellipses; NURBS control-hull) so single-circle caps are included\nby the BVH.\n- Shared seam-anchored UV boundary sampler: `check::util::face_polygon`\n(now public) keeps closed circle/ellipse/NURBS rims phase-coherent\nthrough periodic unwrap; `operations` now delegates to it;\n`boolean::face_polygon` unchanged.\n\n<sup>Written for commit a4db67b472345fe24653215cb6716a7883b2c4da.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1068?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T22:31:24Z",
          "tree_id": "699c2665fb3f1a2e9992d89aa03f8f2bcdcca457",
          "url": "https://github.com/andymai/brepkit/commit/6e21bdfab809f4d54288a5678aed398b1e2cfeac"
        },
        "date": 1783722816580,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 794230,
            "range": "± 2010",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 893195,
            "range": "± 750",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13422,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 646170,
            "range": "± 2730",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20725906,
            "range": "± 261650",
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
          "id": "61e5ca5270adedf22c26c9107a07eb6d1f6abdfc",
          "message": "chore(main): release 2.126.1 (#1070)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.1](https://github.com/andymai/brepkit/compare/v2.126.0...v2.126.1)\n(2026-07-10)\n\n\n### Bug Fixes\n\n* **check,operations:** classify trimmed-torus bands correctly\n([#1068](https://github.com/andymai/brepkit/issues/1068))\n([6e21bdf](https://github.com/andymai/brepkit/commit/6e21bdfab809f4d54288a5678aed398b1e2cfeac))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.1 with a bug fix that correctly classifies\ntrimmed-torus bands in `check` and `operations`, improving geometric\naccuracy for torus trims.\n\n<sup>Written for commit 900d237f1a34fa02169595aeaaafc91f95d90d3f.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1070?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T22:38:25Z",
          "tree_id": "ad65ad9d45d9796927076065c26529766cba5400",
          "url": "https://github.com/andymai/brepkit/commit/61e5ca5270adedf22c26c9107a07eb6d1f6abdfc"
        },
        "date": 1783723253623,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 733170,
            "range": "± 7643",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 797518,
            "range": "± 24891",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11405,
            "range": "± 132",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 595577,
            "range": "± 9396",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18575795,
            "range": "± 322161",
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
          "id": "c9626ea7ac66bcea8de12c06cb5b8c52eaa640c4",
          "message": "fix(algo): make plane-plane FF section clipping robust to collinear boundary edges (#1069)\n\n## Root cause\n\n`fuse_shelled_box_with_socket_loft` (long-ignored, mesh-fallback output\nwith euler −54): the socket wall facets meet the box bottom plane\nexactly along their top chords, so every plane×plane FF section line is\ncollinear with a clip-polygon edge. `clip_line_to_polygon`\n(`crates/algo/src/pave_filler/phase_ff.rs`) judged parallelism with an\nabsolute epsilon (`denom.abs() < 1e-15`) on an unnormalized dot product\nwhose natural scale here is |n|·|d| ≈ 100, so the collinear edge read as\na genuine crossing and the span was clipped by the ratio of two roundoff\nresidues. Result: a nondeterministic subset of sections emitted (18/36,\nseveral sliver-length partials), an inconsistent bottom-face partition,\nand the previously mapped downstream symptoms (over-shared chord edges,\nphantom arc-break vertices) were noise from the missing/partial\nsections.\n\n## Fix\n\nScale-relative parallel + outside thresholds: treat the edge as parallel\nwhen `|denom| < |n|·|d|·1e-9` (i.e. sin(angle) < 1e-9) and use a\n`|n|`-relative distance band for the outside rejection, so on-boundary\nlines take the parallel path deterministically. This is the recurring\ntangential-contact conditioning class, fixed at the primitive.\n\nPost-fix, raw GFA output equals the gated operations output (no\nfallback): F=55/E=148/V=96, watertight (every edge exactly 2-use by id),\nmanifold, analytic (4 corner cylinder barrels preserved), volume =\noperand sum, deterministic across fresh processes.\n\nThe test is un-ignored with strengthened assertions: hole-aware Euler\n(the shelled cup's top rim is a genuine annulus face, so naive V−E+F is\n2+1 — the old naive `euler==2` assert was wrong for this shape),\nid-level edge-use==2 watertightness, analytic face census (≥4 cylinders,\n<100 faces), and volume-sum.\n\nKnown residual (documented in the roadmap row, deliberately below engine\ncoincidence semantics): the 32 chord/arc corner lenses at z=0 (height 19\nµm = r(1−cos 5.625°)) collapse to the chord via the endpoint-keyed edge\nmerge; full crescent fidelity needs the midpoint-split cascade and is\nonly worth chasing if a consumer needs sub-20 µm corner fidelity.\n\n## Verification\n\n- `cargo fmt --all --check`, `cargo clippy --all-targets -- -D\nwarnings`, `./scripts/check-boundaries.sh` — clean\n- `fuse_shelled_box_with_socket_loft` green across repeated fresh\nprocesses\n- operations lib suite 765+ passed / 0 failed; algo suite 152/0\n- Canary: `cargo test -p brepkit-wasm --lib gridfinity` — 27/0\n- Full `cargo test --release -p brepkit-io` calibrated fixture set — 31\ntargets, 0 failures (pins unmoved)\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nMake plane–plane FF section clipping robust for collinear edges and\nnear-parallel/degenerate cases using scale‑relative checks. This removes\nnondeterministic partial section emission in the socket‑loft fuse and\nrestores a deterministic, watertight analytic result.\n\n- **Bug Fixes**\n- In `crates/algo/src/pave_filler/phase_ff.rs` (`clip_line_to_polygon`),\njudge parallelism relative to |n|·|d| (sin(angle) < 1e-9) and use a\n|n|-relative outside band; reject zero‑length segments, skip zero‑length\npolygon edges, and only drop near‑parallel spans when both endpoints are\noutside. Prevents treating on‑boundary lines as crossings and\nsliver/empty spans.\n- Un-ignored and strengthened `fuse_shelled_box_with_socket_loft` in\n`crates/operations/src/boolean/tests.rs`: checks edge-use==2\nwatertightness, analytic face count, ≥4 cylinder faces, volume equals\noperand sum, and hole‑aware Euler.\n- Added focused unit tests in `phase_ff.rs` for collinear-span keep,\nparallel-outside drop, zero-length drop, and near-parallel entering;\nupdated `.claude/skills/roadmap/SKILL.md` to mark the case closed.\n\n<sup>Written for commit 4f8b4d15c2353007735c9546f598be0a3534772a.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1069?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-10T22:47:31Z",
          "tree_id": "008d8647e51bb63ad06964443bf0096fc732eabc",
          "url": "https://github.com/andymai/brepkit/commit/c9626ea7ac66bcea8de12c06cb5b8c52eaa640c4"
        },
        "date": 1783723770273,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 613785,
            "range": "± 3982",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 687965,
            "range": "± 902",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10228,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 495357,
            "range": "± 11575",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 16230639,
            "range": "± 532691",
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
          "id": "548cd27324abc783208c051436842f52255c521d",
          "message": "chore(main): release 2.126.2 (#1071)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.2](https://github.com/andymai/brepkit/compare/v2.126.1...v2.126.2)\n(2026-07-10)\n\n\n### Bug Fixes\n\n* **algo:** make plane-plane FF section clipping robust to collinear\nboundary edges ([#1069](https://github.com/andymai/brepkit/issues/1069))\n([c9626ea](https://github.com/andymai/brepkit/commit/c9626ea7ac66bcea8de12c06cb5b8c52eaa640c4))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPublish `brepkit-wasm` v2.126.2 with a fix that makes plane–plane FF\nsection clipping robust when boundary edges are collinear, preventing\nbad sections and errors.\n\n<sup>Written for commit 6010c868d3edf64e7ee80bb7b1d8ae8103b0af89.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1071?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-10T22:53:35Z",
          "tree_id": "4b982a6d66b232f2f39263ad653244cc0073e75a",
          "url": "https://github.com/andymai/brepkit/commit/548cd27324abc783208c051436842f52255c521d"
        },
        "date": 1783724157510,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 755073,
            "range": "± 4647",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 844885,
            "range": "± 1531",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12078,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 632729,
            "range": "± 4544",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19503250,
            "range": "± 19119",
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
          "id": "fe6c44096c3fa73cddc4d67ecbfd51680144aa4a",
          "message": "chore(deps): bump taiki-e/install-action from 2.82.6 to 2.82.9 in the actions group (#1074)\n\n[![Dependabot compatibility\nscore](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=taiki-e/install-action&package-manager=github_actions&previous-version=2.82.6&new-version=2.82.9)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate GitHub Actions dependency `taiki-e/install-action` from 2.82.6 to\n2.82.9 across CI workflows to pick up upstream fixes and maintain\nsecurity. No workflow logic or build-step changes.\n\n<sup>Written for commit a391f891def2a1788bc90e51e062f8c8854e2997.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1074?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-13T07:44:47-07:00",
          "tree_id": "2606f2b3e1182acc271c6f99dffbf6f935324cfb",
          "url": "https://github.com/andymai/brepkit/commit/fe6c44096c3fa73cddc4d67ecbfd51680144aa4a"
        },
        "date": 1783954029459,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 606767,
            "range": "± 13087",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 677715,
            "range": "± 7308",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10377,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 490646,
            "range": "± 1735",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 15902739,
            "range": "± 73107",
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
          "id": "d644fee87ac4ee7c0ed8a955ab84abf7e11cc14e",
          "message": "chore(deps-dev): bump the npm group with 2 updates (#1072)\n\nBumps the npm group with 2 updates:\n[@commitlint/cli](https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli)\nand [prettier](https://github.com/prettier/prettier).\n\nUpdates `@commitlint/cli` from 21.1.0 to 21.2.0\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/releases\">@​commitlint/cli's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v21.2.0</h2>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">21.2.0</a>\n(2026-06-30)</h1>\n<h3>Features</h3>\n<ul>\n<li>feat(resolve-extends): resolve pure-ESM presets\n(conventional-changelog v7/v9/v10) by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4859\">conventional-changelog/commitlint#4859</a></li>\n</ul>\n<h3>Chore</h3>\n<ul>\n<li>ci: install git in stock-Ubuntu baseline job by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4847\">conventional-changelog/commitlint#4847</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/blob/master/@commitlint/cli/CHANGELOG.md\">@​commitlint/cli's\nchangelog</a>.</em></p>\n<blockquote>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">21.2.0</a>\n(2026-06-30)</h1>\n<h3>Features</h3>\n<ul>\n<li><strong>resolve-extends:</strong> resolve pure-ESM presets\n(conventional-changelog v7/v9/v10) (<a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/issues/4859\">#4859</a>)\n(<a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/fdb566fe59457a786eac80e2a8cbb994638daba0\">fdb566f</a>)</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/1b4e5bc0e095294ad421b3ac83f6b66665429e60\"><code>1b4e5bc</code></a>\nv21.2.0</li>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/fdb566fe59457a786eac80e2a8cbb994638daba0\"><code>fdb566f</code></a>\nfeat(resolve-extends): resolve pure-ESM presets (conventional-changelog\nv7/v9...</li>\n<li>See full diff in <a\nhref=\"https://github.com/conventional-changelog/commitlint/commits/v21.2.0/@commitlint/cli\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `prettier` from 3.9.1 to 3.9.4\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/releases\">prettier's\nreleases</a>.</em></p>\n<blockquote>\n<h2>3.9.4</h2>\n<ul>\n<li>Angular: Format <code>@content(name)</code> -&gt; <code>@content\n(name)</code> to align with other block syntax (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19499\">#19499</a>\nby <a href=\"https://github.com/fisker\"><code>@​fisker</code></a>)</li>\n</ul>\n<p>🔗 <a\nhref=\"https://github.com/prettier/prettier/blob/3.9.4/CHANGELOG.md#394\">Changelog</a></p>\n<h2>3.9.3</h2>\n<ul>\n<li>Markdown: Fix unexpected removal of characters in liquid syntax (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19489\">prettier/prettier#19489</a>\nby <a href=\"https://github.com/seiyab\"><code>@​seiyab</code></a>)</li>\n<li>TypeScript: Allow decorators to be used with declare on class fields\n(<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19492\">prettier/prettier#19492</a>\nby <a\nhref=\"https://github.com/evoactivity\"><code>@​evoactivity</code></a>)</li>\n</ul>\n<p>🔗 <a\nhref=\"https://github.com/prettier/prettier/blob/3.9.3/CHANGELOG.md#393\">Changelog</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/blob/main/CHANGELOG.md\">prettier's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>3.9.4</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.9.3...3.9.4\">diff</a></p>\n<h4>Angular: Format <code>@content(name)</code> -&gt; <code>@content\n(name)</code> to align with other block syntax (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19499\">#19499</a>\nby <a href=\"https://github.com/fisker\"><code>@​fisker</code></a>)</h4>\n<!-- raw HTML omitted -->\n<pre lang=\"html\"><code>&lt;!-- Input --&gt;\n&lt;FancyButton [label]=&quot;title&quot;&gt;\n  @content (icon) {\n    &lt;span&gt;Icon!&lt;/span&gt;\n  }\n  @content (description) {\n    &lt;span&gt;Description text&lt;/span&gt;\n  }\n  &lt;span&gt;Other children&lt;/span&gt;\n&lt;/FancyButton&gt;\n<p>&lt;!-- Prettier 3.9.3 --&gt;\n&lt;FancyButton [label]=&quot;title&quot;&gt;\n<a href=\"https://github.com/content\"><code>@​content</code></a>(icon) {\n&lt;span&gt;Icon!&lt;/span&gt;\n}\n<a\nhref=\"https://github.com/content\"><code>@​content</code></a>(description)\n{\n&lt;span&gt;Description text&lt;/span&gt;\n}\n&lt;span&gt;Other children&lt;/span&gt;\n&lt;/FancyButton&gt;</p>\n<p>&lt;!-- Prettier 3.9.4 --&gt;\n&lt;FancyButton [label]=&quot;title&quot;&gt;\n<a href=\"https://github.com/content\"><code>@​content</code></a> (icon) {\n&lt;span&gt;Icon!&lt;/span&gt;\n}\n<a href=\"https://github.com/content\"><code>@​content</code></a>\n(description) {\n&lt;span&gt;Description text&lt;/span&gt;\n}\n&lt;span&gt;Other children&lt;/span&gt;\n&lt;/FancyButton&gt;\n</code></pre></p>\n<h1>3.9.3</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.9.1...3.9.3\">diff</a></p>\n<h4>Markdown: Fix unexpected removal of characters in liquid syntax (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19489\">#19489</a>\nby <a href=\"https://github.com/seiyab\"><code>@​seiyab</code></a>)</h4>\n<!-- raw HTML omitted -->\n<pre lang=\"md\"><code>&lt;/tr&gt;&lt;/table&gt; \n</code></pre>\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/b693cb22b412b759784bc2298fc86880e351cd3a\"><code>b693cb2</code></a>\nRelease 3.9.4</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/2e92ac0ee29fb83b7ce32277cb914c2eee192955\"><code>2e92ac0</code></a>\nAngular: Format <code>@content(name)</code> -&gt; <code>@content\n(name)</code> to align with other blo...</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/abed2c22db3cb61e922b7284c3b6c65424353ac2\"><code>abed2c2</code></a>\nBump Prettier dependency to 3.9.3</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/6cfbc00921de451e918b18bb5ab6b80e80f5cd34\"><code>6cfbc00</code></a>\nClean changelog_unreleased</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/3732e1dee6a36bdb2e77a722d206a79ac7e67aa3\"><code>3732e1d</code></a>\nRelease 3.9.3</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/a74a7b05dee7fbef39a6aeff378c3741e1a8ee15\"><code>a74a7b0</code></a>\nAllow decorators to be used with <code>declare</code> on class fields\n(<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19492\">#19492</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/bd9e11ab41e17a4f61f363a981c1ec24d2a4167a\"><code>bd9e11a</code></a>\nCorrect text identification in liquid syntax (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19489\">#19489</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/269eee3faa1f82b1de07bb7e4d15e1cee70f80d4\"><code>269eee3</code></a>\nBump Prettier dependency to 3.9.1</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/ec7ccd1ea47c965bda3c958239899737e899603d\"><code>ec7ccd1</code></a>\nClean changelog_unreleased</li>\n<li>See full diff in <a\nhref=\"https://github.com/prettier/prettier/compare/3.9.1...3.9.4\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpgrade `@commitlint/cli` to 21.2.0 and `prettier` to 3.9.4 to support\npure-ESM commitlint presets and pull in recent formatting fixes.\nDev-only update with no runtime impact.\n\n<sup>Written for commit 3f729236308e25dc7c2d07f98dd1a18098a51311.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1072?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-13T07:44:50-07:00",
          "tree_id": "eeeb0986cf6a03d8107c6b12beef6456e17bbfcb",
          "url": "https://github.com/andymai/brepkit/commit/d644fee87ac4ee7c0ed8a955ab84abf7e11cc14e"
        },
        "date": 1783954178556,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 750824,
            "range": "± 4638",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 843694,
            "range": "± 4577",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12125,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639360,
            "range": "± 4522",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19543694,
            "range": "± 68759",
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
          "id": "21d018edf7d819d7ebf7cef2ad6acb520b61bcc8",
          "message": "chore(deps): update wgpu requirement from 29 to 30 (#1073)\n\nUpdates the requirements on [wgpu](https://github.com/gfx-rs/wgpu) to\npermit the latest version.\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md\">wgpu's\nchangelog</a>.</em></p>\n<blockquote>\n<h2>v30.0.0 (2026-07-01)</h2>\n<h3>Major changes</h3>\n<h4>Optional vertex buffer slots</h4>\n<p>This allows gaps in <code>VertexState</code>'s <code>buffers</code>\nand adds support for unbinding vertex buffers, bringing us in compliance\nwith the WebGPU spec. As a result of this, <code>VertexState</code>'s\n<code>buffers</code> field now has type of\n<code>&amp;[Option&lt;VertexBufferLayout&gt;]</code>. To migrate, wrap\nvertex buffer layouts in <code>Some</code>:</p>\n<pre lang=\"diff\"><code>  let vertex_state = wgpu::VertexState {\n      module: &amp;vs_module,\n      entry_point: Some(&quot;vs_main&quot;),\n      compilation_options: wgpu::PipelineCompilationOptions::default(),\n      buffers: &amp;[\n-         &amp;vertex_buffer_layout\n+         Some(&amp;vertex_buffer_layout)\n      ],\n  };\n</code></pre>\n<p>By <a href=\"https://github.com/teoxoy\"><code>@​teoxoy</code></a> in\n<a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/pull/9351\">#9351</a>.</p>\n<h4>Integer shader I/O no longer defaults to\n<code>@interpolate(flat)</code></h4>\n<p>To align with the shading language specifications, <code>naga</code>\nno longer assumes that integer-typed shader I/O should have\n<code>flat</code> interpolation, i.e., should not be interpolated. Even\nthough flat interpolation is the only choice for integer I/O, it must be\nstill specified explicitly.</p>\n<p>WGSL:</p>\n<pre lang=\"diff\"><code> struct FragmentInput {\n     @location(0) tex_coord: vec2&lt;f32&gt;,\n-    @location(1) index: i32,\n+    @location(1) @interpolate(flat) index: i32,\n }\n</code></pre>\n<p>GLSL:</p>\n<pre lang=\"diff\"><code>-layout(location = 1) in int index;\n+layout(location = 1) flat in int index;\n</code></pre>\n<p>By <a\nhref=\"https://github.com/andyleiserson\"><code>@​andyleiserson</code></a>\nin <a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/pull/9321\">#9321</a>.</p>\n<h4>Empty buffer slices are now permitted</h4>\n<p>Creating a <code>BufferSlice</code> with a length of 0 no longer\ncauses a panic.</p>\n<p>Empty buffer slices can be:</p>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/8bf3e5ff4ab45e2c150e0d6c70d01d25f5b126c1\"><code>8bf3e5f</code></a>\nUpdate to v30 (<a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/issues/9790\">#9790</a>)</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/aa2b7907c020c4f046ca697895f9be1a70b27a45\"><code>aa2b790</code></a>\n[core] filter native-only features and limits if</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/2ea3a1d41b0085c537bbd016b30ca801c106e024\"><code>2ea3a1d</code></a>\n[vulkan] advertise <code>DownlevelFlags::SURFACE_VIEW_FORMATS</code> if\n`VK_KHR_swapchai...</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/69e66d8e2a8a4a37f7ad4ad0e0a2702b4bb39843\"><code>69e66d8</code></a>\nadd <code>DENO_WEBGPU_STRICT_COMPLIANCE</code> and set it for the CTS\njob</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/995ee7b3f1adfc3729d3fda2122f930c1e6b697f\"><code>995ee7b</code></a>\nadd\n<code>Limits::max_buffers_and_acceleration_structures_per_shader_stage</code></li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/0853e7bdc12cf4f0159f2dbdaeb9d957b56f4e43\"><code>0853e7b</code></a>\nfix: gate on static_dxc not static-dxc (<a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/issues/9785\">#9785</a>)</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/979ab2bca62daea08cd1e33c820ab7f4ece608d8\"><code>979ab2b</code></a>\n[core] IDed encoders needs to be dropped (<a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/issues/9782\">#9782</a>)</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/0cc48c829e3e40788ae3609ccd98cfa16cd61ab8\"><code>0cc48c8</code></a>\nfix(core): Track initialization status of 3D textures (<a\nhref=\"https://redirect.github.com/gfx-rs/wgpu/issues/9765\">#9765</a>)</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/a353ba97c98d24d5a66b8822256fdcf9e50ffb9e\"><code>a353ba9</code></a>\nFix signed % wrong for negative operands on NVIDIA (OpSRem poison\nwithout mai...</li>\n<li><a\nhref=\"https://github.com/gfx-rs/wgpu/commit/7877b766c8068b69f535fb7b9b1b3f32346de676\"><code>7877b76</code></a>\n[core] Move <code>begin_*_pass</code> to <code>CommandEncoder</code> and\ndo only ID resolve in glob...</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/gfx-rs/wgpu/compare/wgpu-v29.0.1...v30.0.0\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore this major version` will close this PR and stop\nDependabot creating any more for this major version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this minor version` will close this PR and stop\nDependabot creating any more for this minor version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this dependency` will close this PR and stop\nDependabot creating any more for this dependency (unless you reopen the\nPR or upgrade to it yourself)\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpgrade `wgpu` from 29 to 30 and migrate render code to the new API,\nincluding optional vertex buffer slots and updated surface/present\nbehavior.\n\n- **Dependencies**\n  - Bumped `wgpu` 29 -> 30 in `crates/render/Cargo.toml`.\n\n- **Migration**\n- Update `VertexState` to use `buffers: &[Option<VertexBufferLayout>]`\nand wrap layouts in `Some(...)`.\n- Adjust API usage: set `RequestAdapterOptions.apply_limit_buckets =\nfalse`, add `SurfaceConfiguration.color_space = Auto`, present via\n`Queue::present(frame)`, and handle `Result` from `get_mapped_range()`.\n\n<sup>Written for commit 7f9ecc97ab809a97798b788e9eecc9230c7a1698.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1073?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\n---------\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Andy Aragon <hi@andymai.com>",
          "timestamp": "2026-07-13T14:58:53Z",
          "tree_id": "b3fa5a934bc117e6e9db3a1a8253c24c841cfd91",
          "url": "https://github.com/andymai/brepkit/commit/21d018edf7d819d7ebf7cef2ad6acb520b61bcc8"
        },
        "date": 1783954891496,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 727836,
            "range": "± 890",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821579,
            "range": "± 1198",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12219,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 595980,
            "range": "± 3208",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18883423,
            "range": "± 81853",
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
          "id": "45050725d08925e23e0a35120296d4c548e4bedf",
          "message": "fix(algo): seam-edge flush pocket cut drops the entire slab top (#1076)\n\n## Problem\n\nA fractional-width baseplate tile on a split seam extends its seam-edge\npockets to the tile boundary — the pocket wall is exactly coplanar with\nthe slab's outer wall, and the opening must merge into the boundary as a\nnotch. Instead the cut mesh-fell-back (F=122, 44 open boundary edges)\nand poisoned the whole fractional-plate build (the dovetail `5×4.5\nedge-y-1` scenario exports non-manifold STLs; pre-existing on published\nkernels).\n\n## Root cause\n\n`find_point_outside_holes` built its hole-rejection polygon from each\nhole edge's **stored `start_uv`** — and a hole-wire vertex whose UV was\nfitted in a *different* plane frame corrupts the polygon (one\nforeign-frame vertex among consistent ones). The even-odd test then\naccepted classifier seeds **inside** the pocket opening; both top\nsub-faces sampled the cutter's interior, were classified Inside, and the\nentire slab top vanished. The Euler gate correctly rejected the analytic\nresult and the mesh fallback took over.\n\n## Fix\n\nOne hunk: with a plane frame available, derive **every** hole-polygon\nvertex from the edge's 3D endpoint through the frame — the exact\ndoctrine the function already applies to curved-edge sampling (\"the\nstored pcurve is never trusted\").\n\n## Result\n\n- The seam-edge pocket cut stays analytic: 14 faces, all 4 corner cones\nintact, watertight.\n- The full 25-pocket fractional plate chain builds clean end-to-end\n(F=206, zero open/non-manifold mesh edges at export tolerance) —\npreviously mesh-fallback from the second pocket onward.\n\n## Verification\n\n- New regression fixture\n`crates/io/tests/fracplate_seam_pocket_inmem.rs` with the tool's exact\nserialized operands; 10× flake gate clean.\n- Full suites green: algo, io (all calibrated in-mem fixtures incl. the\nhalfsockets family that pins this function), operations, wasm lib incl.\nthe gridfinity canary, pre-push full-workspace gate.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFix seam-edge flush pocket cuts that dropped the slab top by rebuilding\nhole polygons from 3D via the face frame. Cuts now stay analytic (14\nfaces, 4 corner cones) and export watertight; closure noted in the\nroadmap.\n\n- **Bug Fixes**\n- In `find_point_outside_holes`, derive hole-polygon vertices with\n`frame.project(e.start_3d)` when a frame exists; stop using stored\n`start_uv` from a different frame.\n- Prevents even-odd misclassification that seeded inside the opening,\nwhich caused Euler rejection and mesh fallback.\n- Added regression test `crates/io/tests/fracplate_seam_pocket_inmem.rs`\nwith fixtures (`fracplate_slab.bin`, `fracplate_seam_pocket.bin`) and\nrecorded the fix in `.claude/skills/roadmap/SKILL.md`.\n\n<sup>Written for commit 995b44b19fb7cd56ee0396566e601d4e08641747.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1076?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-16T11:46:37-07:00",
          "tree_id": "2b0ef778cdf15fe701485a7e446d26a9ba5e4789",
          "url": "https://github.com/andymai/brepkit/commit/45050725d08925e23e0a35120296d4c548e4bedf"
        },
        "date": 1784227759041,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 672326,
            "range": "± 17536",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 737462,
            "range": "± 13558",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10544,
            "range": "± 244",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 542534,
            "range": "± 11169",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 16680495,
            "range": "± 165456",
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
          "id": "b6d2642d0b515a97dc1a8008957df983196eea9b",
          "message": "chore(main): release 2.126.3 (#1077)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.3](https://github.com/andymai/brepkit/compare/v2.126.2...v2.126.3)\n(2026-07-16)\n\n\n### Bug Fixes\n\n* **algo:** seam-edge flush pocket cut drops the entire slab top\n([#1076](https://github.com/andymai/brepkit/issues/1076))\n([4505072](https://github.com/andymai/brepkit/commit/45050725d08925e23e0a35120296d4c548e4bedf))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.3 for `brepkit-wasm`, fixing a bug where the seam-edge\nflush pocket cut could drop the entire slab top. This prevents\nunintended slab-top removal during those cuts (#1076).\n\n<sup>Written for commit 744d212f8a53d368bf0861728e7a4b1eab0517cb.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1077?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-16T18:53:18Z",
          "tree_id": "506dbd1b47139a11e45c74e79b8ceb516f4a55d6",
          "url": "https://github.com/andymai/brepkit/commit/b6d2642d0b515a97dc1a8008957df983196eea9b"
        },
        "date": 1784228129121,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 805062,
            "range": "± 4083",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 882246,
            "range": "± 2631",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13175,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631702,
            "range": "± 1147",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20467115,
            "range": "± 141044",
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
          "id": "59cad4d661df0439b1c99045fed17e16b29788e4",
          "message": "fix(algo): dovetail tangency caps, compound relief cuts, and the fit-offset groove-mouth sliver family (#1078)\n\n## Summary\n\nFour commits closing the dovetail/fit-offset mesh-fallback families\nfound by tool-level parity probing on 2.126.2. Together they take the\ncaptured fit-offset export chain (3×3 loose +0.2 plate, 6 nub fuses + 6\ngroove cuts) from mesh-fallback (thousands of planar faces, phantom void\nwedges in the published result) to **fully analytic and watertight end\nto end (182 → 211 faces)**, and the dovetail suite's A1-corner nub\nfamily to analytic.\n\n1. **FF curve pre-filter re-sampling** — a 16-sample AABB probe missed\nshort in-both spans of long intersection curves; re-sample adaptively\nbefore dropping.\n2. **Socket-wall tangency corner caps** — v-range rim crossings in\nplane-face line clipping + iso-v rim circle splits parameterized in the\nedge's own unwrapped u-range (A1-corner doubled-dovetail nub).\n3. **Compound relief cuts with curved boundaries** — the\nall-straight-edges gate in `trim_open_curve_to_plane_face_lines`\ndeclined faces whose boundaries carry rim arcs/conics from a prior cut.\n4. **Fit-offset groove-mouth sliver family** (the big one — see the\ncommit message for the five coordinated mechanisms):\n- promotion of pave-split passthrough holes into the combined planar\narrangement (expansion kept separate from the weave input, which is\ncalibrated on unsplit hole edges),\n- clean-tiling cutoff for even-odd hole nesting (kills the double-cover\nre-attachment; overlapping traces keep nesting),\n- true circle×section-line boundary-arc splits, applied only on the\narrangement path, with a bay-mouth arrangement entry for multi-hole\ncaps,\n   - arc-true region polygons for interior probes,\n- at-seam UV endpoint resolution on periodic surfaces +\norientation-aware plane-arc split normalization.\n\n## Verification\n\n- Full workspace: 2187 tests green, clippy `-D warnings` clean.\n- Calibrated-path canaries all green simultaneously: d-series gridfinity\n(27/27, incl. d4 full bin), honeycomb pcut3 re-trace weave, divider-lip\nfuse, halfSockets family.\n- New faithful fixtures from tool-captured operands:\n`dovetail_dblcorner_nub_inmem.rs`, `fitoffset_nub_chain_inmem.rs`,\n`fitoffset_groove_mouth_inmem.rs` (full 6-groove chain, watertight at\nexport tolerance after every cut).\n\nSupersedes draft #1075 (same first three commits; that branch's remote\nhistory predates a rebase onto #1076 and will be closed).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes analytic cutting failures in dovetail tangency caps, compound\nreliefs, and fit‑offset groove‑mouth slivers so exports are watertight\nand avoid mesh fallback. The captured fit‑offset chain is now fully\nanalytic end‑to‑end (182 → 211 faces); roadmap docs mark A1‑corner and\ngroove‑mouth families closed and the dovetail suite at 9/9.\n\n- **Bug Fixes**\n- FF curve pre-filter: adaptive sampling to catch short in‑both spans\nbefore dropping.\n- Tangency caps: clip by rim v‑range and split iso‑v rim circles; place\nsplits in the edge’s own UV span.\n- Compound reliefs: allow curved boundary faces in the open‑conic clip\nto keep second cuts analytic.\n- Groove‑mouth slivers: promote passthrough hole rings; skip even‑odd\nnesting on clean tilings; split bay arcs at true circle×section\ncrossings; densify arc polygons for probes; keep weave input unsplit and\nenter arrangement with intact holes when needed.\n- Periodic seams: resolve 0 vs 2π UV endpoints; normalize plane‑arc\nsplit orientation; test on‑boundary sections against the true 3D arc.\n- Added regression fixtures for dovetail corner nubs, compound chains,\nand fit‑offset groove mouths.\n\n- **Refactors**\n  - Clean‑tiling cutoff tolerance scaled by region area.\n- Hasher‑generic edge image storage threaded through\n`fill_images_faces`.\n- Named the weave‑section source‑index sentinel `WEAVE_SECTION_SRC_BASE`\nfor clarity.\n\n<sup>Written for commit 2e90b35c8ee43a2ca0a1a956f46737373f022764.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1078?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-16T21:53:10Z",
          "tree_id": "3b59ca01140555b9589e78b776c2876c9fb768f7",
          "url": "https://github.com/andymai/brepkit/commit/59cad4d661df0439b1c99045fed17e16b29788e4"
        },
        "date": 1784239217465,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 779071,
            "range": "± 7598",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 882904,
            "range": "± 1841",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13149,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639809,
            "range": "± 2469",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20414406,
            "range": "± 24600",
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
          "id": "311e389f66f7f4e32e74751bc0a199ba07eecbce",
          "message": "chore(main): release 2.126.4 (#1079)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.4](https://github.com/andymai/brepkit/compare/v2.126.3...v2.126.4)\n(2026-07-16)\n\n\n### Bug Fixes\n\n* **algo:** dovetail tangency caps, compound relief cuts, and the\nfit-offset groove-mouth sliver family\n([#1078](https://github.com/andymai/brepkit/issues/1078))\n([59cad4d](https://github.com/andymai/brepkit/commit/59cad4d661df0439b1c99045fed17e16b29788e4))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.126.4 for `brepkit-wasm` fixes geometry edge cases in\ndovetail tangency caps, compound relief cuts, and fit‑offset\ngroove‑mouth slivers. Updates versions and changelog to 2.126.4.\n\n<sup>Written for commit 99cfd2767404aad3fcb41b936ec74eaf2331736c.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1079?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-16T22:05:28Z",
          "tree_id": "d075f21e708599a6c58383f2e7b539daff98f5a5",
          "url": "https://github.com/andymai/brepkit/commit/311e389f66f7f4e32e74751bc0a199ba07eecbce"
        },
        "date": 1784239669057,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 760312,
            "range": "± 1865",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 854391,
            "range": "± 5635",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11987,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639911,
            "range": "± 1157",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19623571,
            "range": "± 91886",
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
          "id": "d1a1c08a8583cbac4a229a31f8c05a0d63cebe44",
          "message": "fix(operations): recognize spline-encoded profile edges before extruding walls (#1080)\n\n## Summary\n\nThe snap-clip connector key (snapClip suite, bnd=326 export failure)\ntraced to the extrude source: 2D drawing pipelines ship corner-treated\nprofiles as B-splines (chamfer segments and corner-trimmed straight runs\narrive as degree-N NURBS; fillet arcs as rational splines), and extrude\nemitted ruled-NURBS side walls for geometry that is exactly\nplanar/cylindrical. Every downstream boolean against the prism degraded\n— the clip's four socket-relief cuts fell to a mesh fallback purely\nbecause the walls were spline-encoded.\n\nFix: normalize the profile wires' curves in place at extrude entry,\nmirroring the loft profile-recognition from the analytic-socket work —\nNURBS→Line and NURBS→Circle (oriented so the CCW start→end span covers\nthe spline's own midpoint; conversion skipped unless the spline's\nendpoints coincide with the edge vertices).\n\n## Verification\n\n- New unit fixture: spline-encoded chamfered/filleted profile extrudes\nwith zero NURBS walls, one true cylinder, and exact volume.\n- Captured tool chain (snap-clip key): clip prism 4 cyl + 18 plane (was\n8 NURBS walls); all four socket-relief cuts analytic and watertight\n(F=98, E=350 vs the fallback's E=3029); the connector-key scenario\npasses tool-side on an overlay build.\n- Full suites: operations 971, io 207, wasm 229 — all green; workspace\nclippy clean.",
          "timestamp": "2026-07-16T15:46:11-07:00",
          "tree_id": "d7f37cceba13c9505ea98b95608145fb7188f22b",
          "url": "https://github.com/andymai/brepkit/commit/d1a1c08a8583cbac4a229a31f8c05a0d63cebe44"
        },
        "date": 1784242099877,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 798205,
            "range": "± 20321",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 879322,
            "range": "± 18543",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13050,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 638867,
            "range": "± 13791",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20488379,
            "range": "± 3843493",
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
          "id": "3746b844494d30e6463f52a65c0a174ea6ca6443",
          "message": "test(operations): cover the reversed-spline-arc profile path; buffer the near-full-arc midpoint check (#1081)\n\nFollow-up to #1080, addressing Greptile's two P2 findings that automerge\noutran:\n\n- New fixture exercising the `rev` branch of\n`normalize_profile_wire_curves` (fillet spline stored end-to-start\nagainst the wire traversal) — recovers analytic walls and the exact\nvolume.\n- Tolerance buffer on the CCW midpoint-containment check so a near-full\narc cannot flip its normal on rounding.",
          "timestamp": "2026-07-16T22:55:14Z",
          "tree_id": "e8a7573ca7708e1f5d6b582cb5aaf2017cb981d6",
          "url": "https://github.com/andymai/brepkit/commit/3746b844494d30e6463f52a65c0a174ea6ca6443"
        },
        "date": 1784242632975,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 645387,
            "range": "± 9948",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 720694,
            "range": "± 11016",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10818,
            "range": "± 143",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 522858,
            "range": "± 2592",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 17264531,
            "range": "± 168485",
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
          "id": "c5fc0cd2d897fa790f66c9298babe2a9cf437392",
          "message": "fix(algo): keep the completed socket-junction disc when its traced loop samples degenerate (#1082)\n\n## Summary\n\nThe snapClip join-edges plate export accumulated ~1100 open mesh edges\nbecause the **8th pocket cut** — the one completing a 4-way cell-corner\njunction — lost the junction's blind-recess disc: the wire builder\ntraces the completed r=4 junction circle as a standalone two-arc loop\nwhose pcurve-sampled polygon folds to ~zero area, the sliver guard\nsilently drops it, and the planar-arrangement path (which produces the\ndisc correctly) was declined because the raw loop **count** still\nmatched. The resulting unpaired rims forced a mesh fallback whose open\noutput poisoned all 13 remaining pocket cuts.\n\nFix: the arrangement gate also fires when any traced loop's sampled\npolygon is area-degenerate — such a loop is invisible to the count\ncomparison because the sliver guard will drop it downstream, so the\nloops path is guaranteed to under-represent the face.\n\n## Verification\n\n- Captured tool chain (5×4 snapClip plate): all 20 pocket cuts analytic\n+ watertight (F=595 final; was F=6923 with bnd=930).\n- Minimal fixture (slab + the 4 junction pockets, tool-exact operands):\nfails without the fix (F=1342, bnd=130), green with it.\n- Calibrated foils all green: d-series gridfinity (27/27), honeycomb\npcut3, divider-lip fuse, groove-mouth chain, nub chains.\n- Full workspace: 2571 tests, clippy `-D warnings` clean.\n\nThird member of today's snapClip family work (#1080 closed the connector\nkey at the extrude source; this closes the plate-side chain root).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes loss of the completed socket‑junction disc by detecting\narea‑degenerate traced loops and forcing the arrangement path. Prevents\nunpaired rims and mesh fallback, keeping pocket cuts analytic and\nwatertight.\n\n- **Bug Fixes**\n- Added a degeneracy check (`wire_loops_have_degenerate_area`) in the\nface splitter; if any loop’s sampled UV polygon is zero/near‑zero area,\nuse the planar arrangement.\n- Preserves the r=4 cell‑corner junction disc even when the traced\ntwo‑arc loop samples fold to near zero area.\n- Verification: 5×4 snapClip plate now produces 20 analytic, watertight\npocket cuts (F=595; was F=6923, bnd≈930); full workspace 2571 tests\npass. Added regression fixture\n`crates/io/tests/socket_junction_disc_inmem.rs` with pocket/slab data.\n\n<sup>Written for commit f5a2de03378f9552b9b42963f2571ca107be6bdb.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1082?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-16T23:51:49Z",
          "tree_id": "a931b223f03a1d36dc5e4edcc86200d9e9dfb275",
          "url": "https://github.com/andymai/brepkit/commit/c5fc0cd2d897fa790f66c9298babe2a9cf437392"
        },
        "date": 1784246036770,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 785843,
            "range": "± 7370",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 881481,
            "range": "± 5078",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12994,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635411,
            "range": "± 1361",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 20593914,
            "range": "± 274320",
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
          "id": "af6406f897ec6c6e98afe58bb8264f0bc5bfeeaa",
          "message": "chore(main): release 2.126.5 (#1083)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.5](https://github.com/andymai/brepkit/compare/v2.126.4...v2.126.5)\n(2026-07-16)\n\n\n### Bug Fixes\n\n* **algo:** keep the completed socket-junction disc when its traced loop\nsamples degenerate\n([#1082](https://github.com/andymai/brepkit/issues/1082))\n([c5fc0cd](https://github.com/andymai/brepkit/commit/c5fc0cd2d897fa790f66c9298babe2a9cf437392))\n* **operations:** recognize spline-encoded profile edges before\nextruding walls\n([#1080](https://github.com/andymai/brepkit/issues/1080))\n([d1a1c08](https://github.com/andymai/brepkit/commit/d1a1c08a8583cbac4a229a31f8c05a0d63cebe44))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.5 of `brepkit-wasm` fixing socket-junction disc stability\nand enabling wall extrusion from spline-encoded profile edges.\n\n<sup>Written for commit 351a2fa914cc10be97021dcdc990c1980ea6ff19.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1083?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-16T23:58:35Z",
          "tree_id": "c69dbfc46679252f1d5ef4ef9c7b95f674ec38fc",
          "url": "https://github.com/andymai/brepkit/commit/af6406f897ec6c6e98afe58bb8264f0bc5bfeeaa"
        },
        "date": 1784246450680,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 758862,
            "range": "± 2029",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 847214,
            "range": "± 2056",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11817,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639611,
            "range": "± 1666",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19591802,
            "range": "± 1486497",
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
          "id": "c0fbeda3e3ec24d470a5e18c36b73d443c4beb4f",
          "message": "docs(algo): reattach the self-cross gate doc comment to its function (#1084)\n\nFollow-up to #1082 (automerge outran Copilot's finding): the new\n`wire_loops_have_degenerate_area` was inserted between\n`wire_loops_self_cross`'s doc comment and the function, misattaching the\nrustdoc. Reorders the comments.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nReattached the doc comment for the self-cross gate to\n`wire_loops_self_cross`. Fixes a misattached rustdoc caused by inserting\n`wire_loops_have_degenerate_area`, ensuring the correct function is\ndocumented.\n\n<sup>Written for commit 7cb4a956ffd746c8efa9f382616b32eb62d44523.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1084?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-16T23:58:51Z",
          "tree_id": "88fa07548a515dd78b095073504355a2ca816224",
          "url": "https://github.com/andymai/brepkit/commit/c0fbeda3e3ec24d470a5e18c36b73d443c4beb4f"
        },
        "date": 1784246577721,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 751810,
            "range": "± 2356",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 840734,
            "range": "± 3090",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11946,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631094,
            "range": "± 849",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 19371556,
            "range": "± 62268",
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
          "id": "de23c2528d71191d3c714f772ccfb52af836c88e",
          "message": "fix(algo): four section-machinery gaps behind the snap-slot hole-cut fallback (#1085)\n\n## Summary\n\nThe snapClip plate's 44 join-edge slot cuts all mesh-fell-back (first\ncut F=595→7356, open fallback output poisoning the rest). Native replay\nisolated four stacked gaps in the section machinery at the plate's\nedge-junction web, fixed together:\n\n1. **Concave-arc clip overshoot** — the per-face section clip took the\noutermost crossing pair, correct only for convex corner arcs;\ninward-bulging socket-bite circles made chords overshoot into air.\nHole-free plane faces now midpoint-classify each sub-interval against\nthe arc-true boundary polygon; holed faces keep the outermost pair\n(their sections feed the calibrated hole weave).\n2. **Multi-window sections** — a section crossing a face in two material\nwindows kept only one; the clip now returns every in-face interval and\nthe consumer emits one section per interval.\n3. **Plane×band Line clip** — Lines against cylinder/cone partners were\nnever clipped to the band's v-window; the band limits map to exact line\nfractions via `project_point` (affine v along a line on the surface).\nMixed pairs get exactly this trim and nothing else.\n4. **Fit-error junction weld** — marched-NURBS endpoints differ from\nexact chain partners by ~1e-6 (above the 1e-7 graph quantization), so\nsilhouette chains never junctioned; plane-face section endpoints now\nweld within the 100·tol band.\n\nPlus: `plane_internal_line_loops` accepts open NURBS sections.\n\n## Verification\n\n- Captured tool operands: cut(socketed 5×4 plate, slot box) analytic +\nwatertight (F=603 vs F=7356 fallback).\n- Calibration foils all green simultaneously: d-series 27/27, honeycomb\npcut3, divider-lip, groove-mouth chain, junction-disc, nub chains,\ncylinder-slot (the last two caught wrong discriminants during\ndevelopment and shaped the final gates).\n- Full workspace 2194 tests; clippy clean.\n- Fixture: `snapclip_slot_cut_inmem.rs` (tool-exact operands).\n\nFifth PR in today's snapClip/parity series (#1078 groove-mouth, #1080\nextrude splines, #1082 junction disc).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes four section-clipping bugs that caused the snap-slot hole cut to\nfall back to mesh and break the rest of the slot cuts. The face/FF\npipeline now clips and welds sections correctly, producing analytic,\nwatertight results.\n\n- **Bug Fixes**\n- Plane-face line clip now classifies sub-interval midpoints against an\narc-true boundary polygon (handles concave bites); uses extended-line\narc hits; keeps every in-face window.\n- Keeps all material windows when a section crosses a face multiple\ntimes; consumer emits one section per window with endpoint noise guard\npreserved.\n- Plane × band (cylinder/cone) line sections get an exact v-window trim\nvia `project_point`; mixed pairs apply only this trim.\n- Welds plane section endpoints to coincident boundary/section endpoints\nwithin 100·tol to form junctions; `plane_internal_line_loops` accepts\nopen NURBS sections.\n- Adds an in-memory regression test and fixtures for the snap-slot cut;\nresult stays analytic (~603 faces) and watertight.\n\n<sup>Written for commit 6f11b177606246b74a8c318f080b9464fec07795.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1085?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T02:24:48Z",
          "tree_id": "e166d7fd92945b3257a413d72e502d22239e6a90",
          "url": "https://github.com/andymai/brepkit/commit/de23c2528d71191d3c714f772ccfb52af836c88e"
        },
        "date": 1784255218291,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 878881,
            "range": "± 1771",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 965871,
            "range": "± 6315",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12118,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639820,
            "range": "± 2118",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 25420577,
            "range": "± 167489",
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
          "id": "f0c3499b9e246539724f62aeeb6f1eec1bd0916a",
          "message": "chore(main): release 2.126.6 (#1086)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.6](https://github.com/andymai/brepkit/compare/v2.126.5...v2.126.6)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** four section-machinery gaps behind the snap-slot hole-cut\nfallback ([#1085](https://github.com/andymai/brepkit/issues/1085))\n([de23c25](https://github.com/andymai/brepkit/commit/de23c2528d71191d3c714f772ccfb52af836c88e))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release for `brepkit-wasm` fixing an algorithm bug that created\ngaps when using the snap-slot hole-cut fallback. This improves geometry\ncontinuity and output reliability.\n\n- **Bug Fixes**\n- Resolve four section-machinery gaps behind the snap-slot hole-cut\nfallback (fixes #1085).\n\n<sup>Written for commit 7a84d449bce5534552711ddb5ff9aaa9c6855878.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1086?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T02:31:17Z",
          "tree_id": "c795f6205550e188697acbfb2cdbb4b2812a7ac2",
          "url": "https://github.com/andymai/brepkit/commit/f0c3499b9e246539724f62aeeb6f1eec1bd0916a"
        },
        "date": 1784255592541,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 576142,
            "range": "± 6957",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 643896,
            "range": "± 393",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 8282,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 478262,
            "range": "± 1203",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 17841817,
            "range": "± 762629",
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
          "id": "6a6ee562df3db91cb41085da7acb02c39c9dab39",
          "message": "docs(algo): align section-clip and internal-loop doc comments with the multi-interval behavior (#1087)\n\nFollow-up to #1085 (automerge outran Copilot's three stale-doc\nfindings): the clip now returns every in-face interval and the\ninternal-loop detector accepts open arcs/NURBS — the doc comments still\ndescribed the old single-interval/all-Line behavior.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAligns code doc comments with current behavior: section clipping returns\nevery in-face interval, and internal-loop detection accepts lines, open\narcs, and open NURBS conics with dedup. Also updates the skills roadmap\nto mark snapClip’s three roots closed (#1080/#1082/#1085) and note the\nremaining deepened‑notch work.\n\n<sup>Written for commit 34ceff783db6e568fcbed779ed319b484a38d832.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1087?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T02:35:07Z",
          "tree_id": "9297eb1092a0c81770c437495e0f7e1a09fc4916",
          "url": "https://github.com/andymai/brepkit/commit/6a6ee562df3db91cb41085da7acb02c39c9dab39"
        },
        "date": 1784255835296,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 873279,
            "range": "± 2057",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 958617,
            "range": "± 4625",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12009,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 630185,
            "range": "± 1210",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 25455754,
            "range": "± 83593",
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
          "id": "c89739e14630962951fd9e0b41f60398a1bd13f3",
          "message": "fix(algo): re-vote ray-cast classification when all cardinal rays graze degenerate structure (#1088)\n\n## Summary\n\nThe dovetail A1-corner nub fuse regressed to a whole-plate mesh fallback\n(dovetail suite 8/9) after the junction-disc fix changed the plate's\ncorner topology. Root cause is in the ray-cast classifier, not the\nsplitter: the plate wall strip's splitter-computed interior point lands\nat exactly (42, −4, −1.75) — the intersection of THREE axis-aligned\nfeature planes (the wall plane, the relief-bore tangency / profile-seam\nmeridian, and the dovetail flare plane). All three cardinal\nclassification rays (+Z, +X, +Y) travel along edges, seams, and the\ntangency line from there; their crossing parity is meaningless, the vote\ncame out 1/3, and the interior strip classified **Outside** — a kept\nmembrane, edge over-sharing in the analytic result, and a mesh fallback\nfor the whole plate chain (1517 faces).\n\n## Fix\n\nEach ray now reports whether any of its hits (accepted or barely\nrejected) grazed a face boundary, band limit, hole-band edge, patch\nu-gap border, or an in-plane face. When **all three** cardinal rays are\ndegenerate, the vote is re-cast with fixed generic directions\n(normalized √-prime component vectors) that never run parallel to\naxis-aligned feature planes. Any clean cardinal ray keeps its historical\nverdict.\n\nThe conservative trigger matters: two blunter variants were each\nrejected by a calibration foil —\n- generic directions unconditionally → honeycomb pcut1 over-shared edges\n0→7;\n- escalate on any split vote → wall-cutout fixture free edges 0→48.\n\nThe per-ray degeneracy design passes both foils and the new fixture\nsimultaneously.\n\n## Testing\n\n- New fixture `crates/io/tests/dovetail_a1corner_nubfuse_inmem.rs` with\nthe tool's exact serialized operands: fails on the previous classifier\nwith a 1517-face mesh fallback; now 144 analytic faces, watertight and\nmanifold at export tolerance.\n- Full workspace suite green (2192 tests; the one `compute_mesh_lod`\nSIGSEGV is a pre-existing parallel wgpu device-creation flake — the same\nbinary passes 5/5 with `--test-threads 1` on both HEAD and this branch).\n- Foils re-run explicitly: honeycomb pcut1 4/4, wallcut fixture, d4\ngridfinity canary 27/27, groove-mouth + junction-disc + snap-slot\nfixtures.\n- clippy `-D warnings` + fmt clean; pre-push full-test hook passed.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes ray-cast point classification when all three cardinal rays graze\nfeature boundaries by re-voting with generic directions. Resolves the\nA1‑corner dovetail nub fuse misclassification and removes the 1517‑face\nmesh fallback (now 144 analytic faces, watertight).\n\n- **Bug Fixes**\n- Each ray now flags degeneracy (grazed face boundary/band/in‑plane\nhit).\n- If and only if all three cardinal rays are degenerate, re-cast with\nfixed generic directions that aren’t parallel to axis-aligned planes.\n- Preserve any clean cardinal ray’s verdict to avoid regressions on\ncalibrated coincident-contact cases.\n- Added a focused in‑memory fixture with serialized operands to lock the\nA1‑corner behavior.\n\n- **Refactors**\n- Aligned inline comments with the per-ray degeneracy escalation design.\n\n<sup>Written for commit c5267e87bb6ba2881825591faf07179d178f3199.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1088?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T04:22:08Z",
          "tree_id": "fd54610ac376a30c02c382a944d14c1050bb8e7e",
          "url": "https://github.com/andymai/brepkit/commit/c89739e14630962951fd9e0b41f60398a1bd13f3"
        },
        "date": 1784262255127,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 870735,
            "range": "± 2479",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 961338,
            "range": "± 1083",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11863,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 640938,
            "range": "± 20005",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26526366,
            "range": "± 256850",
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
          "id": "eea0e97c15c158fdebd94edbd5b520c47ddf248c",
          "message": "chore(main): release 2.126.7 (#1089)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.7](https://github.com/andymai/brepkit/compare/v2.126.6...v2.126.7)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** re-vote ray-cast classification when all cardinal rays graze\ndegenerate structure\n([#1088](https://github.com/andymai/brepkit/issues/1088))\n([c89739e](https://github.com/andymai/brepkit/commit/c89739e14630962951fd9e0b41f60398a1bd13f3))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.126.7 addresses a ray-cast misclassification edge case\nby re-voting classification when all cardinal rays graze a degenerate\nstructure. Updates `brepkit-wasm` version to 2.126.7.\n\n- **Bug Fixes**\n- Re-vote ray-cast classification when all cardinal rays graze\ndegenerate geometry to avoid incorrect inside/outside results (fixes\n#1088).\n\n<sup>Written for commit 8b4091198782aa44358ae633a76ed31c8e370de7.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1089?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T04:28:39Z",
          "tree_id": "bff0264fab6326945ef609f717b5cfe00a2ad384",
          "url": "https://github.com/andymai/brepkit/commit/eea0e97c15c158fdebd94edbd5b520c47ddf248c"
        },
        "date": 1784262642355,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 868257,
            "range": "± 6936",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 958262,
            "range": "± 3340",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12001,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 630453,
            "range": "± 914",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26530741,
            "range": "± 87030",
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
          "id": "5801702d9c2569e10c3163788a9049f0ac0b62c2",
          "message": "fix(algo): weld section endpoints onto line interiors; widen arrangement on-plane band (#1090)\n\n## Summary\n\nFollow-up to #1088. With the A1-corner fuse chain analytic, the dovetail\nscenario's remaining nm=2 STL pin traced to the connector-recess hole\ncuts: each recess box's slanted side wall receives a 4-section web from\nthe doubled tongue it crosses — a U-chain of three lines plus a\nplane×cone conic whose far end lands **mid-span of the z=0 line section\n(a T-junction)**.\n\nMarched section geometry is only good to its curve-fit error (~1e-6),\nand two exact-tolerance (1e-7) gates rejected it:\n\n1. **The plane-face endpoint weld had no anchor at the T** (anchors are\nboundary and section *endpoints*), so the junction never formed and the\nconic dangled. The weld now runs a second pass projecting unmatched\nendpoints onto other Line sections' interiors, snapping to the nearest\nstrictly-interior foot within the same 100·tol band. The downstream\nT-junction split then fires at its exact tolerance.\n2. **The planar-arrangement rescue bailed** because its arc on-plane\nround-trip demanded the vertex tolerance; a fitted conic lies in the\nplane only to fit error. The band is now the weld scale — genuine\nstraddle arcs (the check's real target) are off-plane by orders of\nmagnitude more.\n\nUn-rescued, the angular wire builder walked the CW-boundary slit-web as\none grand circuit (every section traversed out-and-back) under **both**\nwinding rules; the cut failed the analytic gate and the mesh fallback\nexported a doubled coincident face pair — the scenario's nm=2.\n\n## Testing\n\n- New fixture `crates/io/tests/dovetail_a1corner_holecut_inmem.rs` with\nthe tool's exact serialized operands (forExport=false corner plate +\nboth recess boxes): pre-fix the first cut mesh-falls-back at 1168 faces;\nnow the chain is analytic (65→69 faces), watertight and manifold at\nexport tolerance.\n- Full workspace suite green (2193 tests, 72 binaries, zero failures).\n- Calibrated foils re-run explicitly: d4 canary 27/27, honeycomb pcut1\n4/4, wallcut, groove-mouth, junction-disc, snap-slot, divider-lip,\ncornerclip, a1corner nub fuse.\n- clippy -D warnings + fmt clean; pre-commit and pre-push hooks passed.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes planar web splitting by welding section endpoints onto line\ninteriors, widening the on-plane check, and limiting welds to\ncurved-section endpoints. A new in-memory fixture keeps the A1-corner\nrecess-hole cuts analytic, watertight, and manifold.\n\n- **Bug Fixes**\n- Weld now projects unmatched curved-section endpoints onto other Line\nsections’ strictly interior points within a `100*tol` band, constrained\nto the segment span (not extensions or near ends), so T‑junctions form\nand downstream splits fire at exact tolerance.\n- Planar arrangement “on-plane” check uses the weld scale (`100*tol`)\ninstead of vertex tolerance, keeping fitted plane×cone conics in-plane\nwhile still rejecting true off‑plane arcs.\n\n<sup>Written for commit fca8cb7e1c4e1a4cfad8b558d6e966d80b7fb317.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1090?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T05:07:03Z",
          "tree_id": "6144b743d487dcd412aaa437298881a931ba37af",
          "url": "https://github.com/andymai/brepkit/commit/5801702d9c2569e10c3163788a9049f0ac0b62c2"
        },
        "date": 1784264960957,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 874717,
            "range": "± 1981",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 961379,
            "range": "± 3786",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11970,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631933,
            "range": "± 13646",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26414447,
            "range": "± 50414",
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
          "id": "f097338fcce373646bdddc795c8c3848a0e37adc",
          "message": "chore(main): release 2.126.8 (#1091)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.8](https://github.com/andymai/brepkit/compare/v2.126.7...v2.126.8)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** weld section endpoints onto line interiors; widen\narrangement on-plane band\n([#1090](https://github.com/andymai/brepkit/issues/1090))\n([5801702](https://github.com/andymai/brepkit/commit/5801702d9c2569e10c3163788a9049f0ac0b62c2))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.8 to improve sectioning stability and\non-plane arrangement robustness.\n\n- **Bug Fixes**\n  - Weld section endpoints onto line interiors to prevent gaps at joins.\n- Widen the on-plane band in arrangement to better handle near-coplanar\ngeometry.\n\n<sup>Written for commit b40c97d0c79511f84fc11ae3baf2c41be75d0926.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1091?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T05:13:44Z",
          "tree_id": "d3f71cb0fc747204da1fce1e79738a92a8e7d369",
          "url": "https://github.com/andymai/brepkit/commit/f097338fcce373646bdddc795c8c3848a0e37adc"
        },
        "date": 1784265349060,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 877099,
            "range": "± 6240",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 960612,
            "range": "± 3326",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11819,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 631591,
            "range": "± 11823",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26777985,
            "range": "± 823620",
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
          "id": "5902a821c29ac7b72ad277bfa011bf21f4452619",
          "message": "fix(algo): true line-arc crossings and slit-free region emission in the planar arrangement (#1092)\n\n## Summary\n\nRoot-caused from the snapClip deepened-notch family (16-iteration dig):\na plane face whose section web mixes lines and marched-NURBS conics was\nsubdivided against the conic's **chord** — the line side split at the\nchord crossing while the arc side rejected the break (the true arc is a\nsagitta away, up to several 1e-4). The mismatched half-edge graph left\ndangling edges that the face tracer walks out-and-back (a deliberate\nretreat rule), so regions were emitted with **slit (doubled) edges**,\nthe analytic result failed the manifold gate, and the whole cut fell\nback to the mesh boolean.\n\n## Changes\n\n- **True line×arc crossings**: polyline pre-locate + bisection on the\narc's native parameter against the line's signed side; the refined point\nis validated to lie ON the line (phantom convergences from fit-error\nsign noise land far off it) and guarded away from curve endpoints\n(endpoint T-junctions belong to the endpoint-break pass). The exact UV\nregisters on **both** inputs so the graph stays consistent.\n- **Trimmed sub-arc emission**: an arc input whose breaks are all exact\nemits endpoint-trimmed true curves instead of bailing to the\n(mis-tracing) loops path.\n- **Weld-band tolerances** for the arrangement's on-plane round-trip and\nT-break tests (100·tol): marched geometry is only good to its curve-fit\nerror (~1e-6); the vertex tolerance (1e-7) rejected real junctions.\nGenuine straddle arcs — the on-plane check's target — are off-plane by\norders of magnitude more.\n- **Section split registry** in `fill_images_faces`: plane faces record\nwhere their sections split; curved faces sharing the same FF curve\n(whose marched copies carry no pave block) pre-split at the identical 3D\npoints via geometric point-on-curve matching, and the existing\nendpoint-T machinery cascades knock-on splits. Plane faces process\nfirst.\n\n## Not closed\n\nThe snapClip deepened-notch case improves (raw repro\nunpaired/over-shared edges 37 → 22, all doubled-edge signatures\neliminated) but does not fully close: the remaining desyncs are\ncross-face **boundary**-edge splits whose root is that marched FF\nsections bypass the pave machinery (`pave_block_id = None`). That fix\nbelongs at phase-FF/make_blocks altitude and is documented on the\nroadmap; three splitter-level propagation attempts each broke calibrated\nfoil chains and were rejected.\n\n## Testing\n\n- Full calibrated foil battery green: groove-mouth, junction-disc,\nsnap-slot, honeycomb pcut, wall-cutout, both A1-corner fixtures,\ncornerclip, divider-lip, d4 gridfinity canary (27/27).\n- Full io/operations/algo suites green; clippy -D warnings + fmt clean;\npre-push full-test hook passed.\n- Raw deepened-notch repro (cached operands): posBad 37 → 22 with slit\nsignatures gone.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes planar arrangements to compute true line–arc crossings and exact\nbreakpoints, eliminating slit edges and manifold failures. Plane faces\nnow register section split points; curved faces pre‑split at the same 3D\npoints for consistent partitions.\n\n- **Bug Fixes**\n- True line–arc crossings on sections: polyline pre‑locate + bisection\non the arc parameter, validated on‑line and guarded away from endpoints;\nreplaces chord hits using the arc’s sampled sagitta + weld band;\nregisters exact UVs on both inputs; boundary arcs keep the historical\nchord path.\n- Emit trimmed sub‑arcs when all breaks are exact; fall back only for\ninexact/chord‑only splits; sub‑spans drop `pave_block_id`.\n- Weld‑band tolerances (100× tol) for on‑plane and T‑break checks;\nboundary Lines are projection targets so section endpoints land exactly.\n- Plane faces process first and write a section split registry; curved\nfaces sharing the same FF curve pre‑split via geometric point‑on‑curve\nmatching; `face_splitter` reads/updates the registry.\n- Added a regression test for a fit‑error T‑junction web (line + marched\nNURBS): asserts three real regions with no slit edges; guards\nchord‑crossing fallbacks near endpoint T’s.\n- Result: slit edges removed; deepened‑notch repro unpaired edges 37→22;\nall test suites green. Known gap: remaining boundary‑edge desyncs from\nmarched FF sections with `pave_block_id=None` (tracked for\nphase‑FF/make_blocks). Cleanup: dropped a duplicated\n`restrict_curves_to_faces` call.\n\n<sup>Written for commit 46924c2c2ec791161efef7e00201e7dfe0aeb193.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1092?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T08:28:27Z",
          "tree_id": "2eaa6667d09bbafd4fcbd107e7a15662e69a3744",
          "url": "https://github.com/andymai/brepkit/commit/5902a821c29ac7b72ad277bfa011bf21f4452619"
        },
        "date": 1784277029594,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 730906,
            "range": "± 25493",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 821480,
            "range": "± 31882",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10149,
            "range": "± 229",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 512949,
            "range": "± 4048",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 22521090,
            "range": "± 385259",
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
          "id": "99fc66b69f4ce983b0109185e148b367d67dd39f",
          "message": "chore(main): release 2.126.9 (#1093)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.9](https://github.com/andymai/brepkit/compare/v2.126.8...v2.126.9)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** true line-arc crossings and slit-free region emission in the\nplanar arrangement\n([#1092](https://github.com/andymai/brepkit/issues/1092))\n([5902a82](https://github.com/andymai/brepkit/commit/5902a821c29ac7b72ad277bfa011bf21f4452619))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.9 for `brepkit-wasm` with a planar arrangement bug fix:\ncorrect line-arc crossings and avoid slit regions for clean topology.\nAlso bumps the crate version and updates the changelog.\n\n<sup>Written for commit 0956dcc1582c19e1cda3729a94708022df04d982.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1093?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T08:35:42Z",
          "tree_id": "67b748f14eb00d5ad29ea19642259237c04b871a",
          "url": "https://github.com/andymai/brepkit/commit/99fc66b69f4ce983b0109185e148b367d67dd39f"
        },
        "date": 1784277472914,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 914343,
            "range": "± 2329",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1005889,
            "range": "± 4451",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13136,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 641757,
            "range": "± 5601",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28224975,
            "range": "± 180651",
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
          "id": "a991f79ea71e0f60dfcf2d23b4f202dc50448854",
          "message": "fix(algo): split marched-NURBS boundary edges at neighbor partition anchors (#1094)\n\n## Summary\n\nFollow-up to #1092, next increment on the snapClip deepened-notch chain.\nA marched-NURBS boundary edge (a plane×cone conic minted by an earlier\nboolean) is shared between a plane face and a curved face. The plane\nside splits its copy face-locally where its section web anchors on it,\nbut `split_boundary_edges_at_3d_points` had **no NurbsCurve arm** — the\nchord-based line fallback rejects on-curve points by the sagitta — so\nthe curved side kept the edge whole and the two partitions\ndesynchronized: unpaired result edges, analytic-gate rejection, mesh\nfallback.\n\n## Changes\n\n- The boundary splitter routes NURBS edges through the existing sampled\npoint-on-curve matcher (`find_splits_on_nurbs_section`: nearest-sample\npre-locate + ternary refine, weld band) and uses the refined on-curve\nfoot as the split vertex, mirroring the Circle arm.\n- Curved faces feed the #1092 section split registry into their boundary\nsplit candidates, so anchors registered by plane neighbours reach the\nshared edges. Candidates not on an edge are inert by construction.\n\nAn earlier attempt at this arm (pre-#1092) regressed the groove-mouth\nchain; on top of #1092's refined arrangement (on-line validation,\nendpoint guards, sagitta cover) the groove landscape produces no NURBS\nboundary splits at all and stays green.\n\n## Testing\n\n- snapClip deepened-notch raw repro: unpaired/over-shared edges **22 →\n10**.\n- Full calibrated foil battery green: groove-mouth, both A1-corner\nfixtures, honeycomb pcut, wall-cutout, junction-disc, snap-slot,\ncornerclip, divider-lip, fit-offset nub chain, dblcorner nub, d4 canary\n27/27.\n- io / operations / algo suites green; clippy -D warnings + fmt clean;\npre-push full-test hook passed.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes desynced splits on marched NURBS boundary edges shared by plane\nand curved faces. NURBS edges now split at neighbor anchors, keeping\nshared edges in sync and avoiding analytic rejection and mesh fallback.\n\n- **Bug Fixes**\n- Route NURBS boundary edges through the sampled point-on-curve matcher\nand use the neighbor anchor point as the split vertex (adopted\nverbatim).\n- Provide the section split registry to all faces so curved-face\nboundary candidates receive plane-side anchors.\n- Result: snapClip deepened-notch unpaired/over-shared edges reduced\nfrom 22 to 10; full test suites green.\n\n<sup>Written for commit 71fdc9858a5383feafed05f225f33251201aeb23.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1094?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T09:53:25Z",
          "tree_id": "9cc4371e5ca5a97d4e6432fbc45b54dfad421a0f",
          "url": "https://github.com/andymai/brepkit/commit/a991f79ea71e0f60dfcf2d23b4f202dc50448854"
        },
        "date": 1784282148533,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 840662,
            "range": "± 2819",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 946720,
            "range": "± 3055",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11604,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 604023,
            "range": "± 3284",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26434701,
            "range": "± 165877",
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
          "id": "75a5ca5ecdeb109b5d9e3baa6ba39fa6edf32765",
          "message": "chore(main): release 2.126.10 (#1095)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.10](https://github.com/andymai/brepkit/compare/v2.126.9...v2.126.10)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** split marched-NURBS boundary edges at neighbor partition\nanchors ([#1094](https://github.com/andymai/brepkit/issues/1094))\n([a991f79](https://github.com/andymai/brepkit/commit/a991f79ea71e0f60dfcf2d23b4f202dc50448854))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.126.10 for `brepkit-wasm` fixes marched-NURBS boundary\nedge handling by splitting at neighbor partition anchors. This improves\ncontinuity at partition boundaries and prevents misaligned edges during\nmarching.\n\n<sup>Written for commit 1db77c1b6e59137040b0e4487f6a60e767ae005e.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1095?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T10:00:05Z",
          "tree_id": "1c68abef3b909769faed5f96d3714bb209d5e373",
          "url": "https://github.com/andymai/brepkit/commit/75a5ca5ecdeb109b5d9e3baa6ba39fa6edf32765"
        },
        "date": 1784282535340,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 870523,
            "range": "± 757",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 965485,
            "range": "± 2393",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11912,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 635816,
            "range": "± 1713",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26429676,
            "range": "± 42662",
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
          "id": "3fa26e4c1c276cf4abd92dd802f9249abc9db483",
          "message": "docs(skills): roadmap — NURBS endpoint-trimmed convention hole (#1096)\n\nRecords the deepened-notch dig's terminal finding:\n`EdgeCurve::domain_with_endpoints` for NurbsCurve ignores its endpoints\n(full knot domain), unlike Circle/Ellipse — every NURBS sub-span\nconsumer silently evaluates the whole curve. Includes the confirmed\nrepro, the first-breaking consumer (curved-lens interior search), and\nthe regression ladder for the dedicated fix.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nDocuments the NURBS endpoint-trim gap in the skills roadmap.\n`EdgeCurve::domain_with_endpoints` returns the full knot domain for\n`NurbsCurve` (unlike `Circle`/`Ellipse`), causing sub-span consumers to\nevaluate entire curves and conflate near-coincident geometry; adds a\nrepro, the first-breaking consumer, and a fix plan with a regression\nladder.\n\n<sup>Written for commit e11dac7972a6ca807c68b63f0b138cc12ff17669.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1096?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T10:48:25Z",
          "tree_id": "cdfdc5893ca3cad282b08bce7d61f8a6ca37ca4f",
          "url": "https://github.com/andymai/brepkit/commit/3fa26e4c1c276cf4abd92dd802f9249abc9db483"
        },
        "date": 1784285436466,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 910602,
            "range": "± 3423",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1007212,
            "range": "± 1854",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13156,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 633588,
            "range": "± 1138",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28257667,
            "range": "± 694770",
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
          "id": "c3575af0d02263d954db772a4095494a7ba25e1e",
          "message": "fix(topology): trim NURBS edge domains to validated forward endpoint sub-spans (#1097)\n\n## Summary\n\n`EdgeCurve::domain_with_endpoints` for `NurbsCurve` historically ignored\nits endpoints and returned the full knot span — unlike Circle/Ellipse,\nwhich project endpoints to the true sub-arc. Every consumer of a NURBS\nsub-span edge (piece pcurves, boundary sampling, wire polygons)\ntherefore silently evaluated the WHOLE shared curve. On the snapClip\ndeepened-notch raw repro this conflated twin cone rims 0.01 apart into\none loop, leaving unpaired micro-edges at the junction.\n\n## The convention (validation-gated; every ambiguous case keeps the\nhistorical full domain)\n\n- **Closed edges** (endpoint chord < 1e-9): full knot span, unchanged.\n- **Whole-curve edges** (endpoints match the natural curve ends within\n1e-6, either orientation): full knot span, unchanged — also the cheap\nfast path (two curve evaluations, no projection).\n- **Forward interior sub-span**: both endpoint projections on-curve\nwithin the 1e-5 weld band (split vertices on marched sections sit up to\n~1e-5 off the fitted curve) AND span > 1e-6 of the domain → returns the\nprojected trimmed `[t₀, t₁]`.\n- **Everything else** (reversed pairs, degenerate spans, off-curve\nendpoints, failed projections): full-domain fallback, byte-for-byte the\nold behaviour.\n\nReversed sub-spans are deliberately NOT accepted yet: accepting them\nexposes a downstream arrangement defect (a degenerate single-edge closed\nloop with a 4e-9 endpoint gap minted at the junction, which\npattern-matches the curved-lens hole signature and aborts the analytic\nsplit). That residual is precisely documented in the roadmap row updated\nin this PR, with the repro recipe.\n\n## Verification\n\n- New unit tests in `edge.rs`: whole-edge both orientations, forward\nsub-span trims and evaluates back to its endpoints, reversed sub-span\nfalls back, off-curve endpoints fall back.\n- Deepened-notch raw repro (arena-captured operands): one of the two\nmirrored junction signatures fully resolves (the use=3 triple +\nmicro-edge chain); posBad signature documented in the roadmap.\n- d4 canary green (`cargo test -p brepkit-wasm --lib gridfinity`), full\nworkspace green in release (all crates, 0 failures).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nTrim NURBS edge domains to validated forward endpoint sub-spans so edges\nonly sample their own piece of a shared curve. This prevents whole-curve\nevaluation that merged nearby rims in the deepened‑notch repro.\n\n- **Bug Fixes**\n- `EdgeCurve::domain_with_endpoints` for `NurbsCurve` now projects\nendpoints and, if both lie on-curve within 1e-5 and form a forward span\n>1e-6 of the domain, returns the trimmed `[t0, t1]`.\n- Closed edges and whole-curve matches (endpoints at natural ends within\n1e-6, either orientation) keep the full knot span.\n- Reversed or invalid spans fall back to the full domain to avoid a\ndownstream degenerate-loop defect; forward case resolves one mirrored\njunction in the deepened‑notch repro.\n- Tests cover whole-edge, forward trim, reversed fallback, and off-curve\nfallback; fallback domain comparisons now use approximate assertions to\navoid float flakiness. Roadmap updated to track the reversed-span\nfollow-up.\n\n<sup>Written for commit fad5a65143c7b052dabd46b83044eec7632415f8.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1097?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T11:23:15Z",
          "tree_id": "999be2e6c6d757757470276705b4b923292ca2f1",
          "url": "https://github.com/andymai/brepkit/commit/c3575af0d02263d954db772a4095494a7ba25e1e"
        },
        "date": 1784287536220,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 901060,
            "range": "± 3325",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1006960,
            "range": "± 8539",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13230,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639820,
            "range": "± 2279",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28538682,
            "range": "± 148517",
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
          "id": "9f94c9208ba2d273d16c58743f7cebcf7292cc97",
          "message": "chore(main): release 2.126.11 (#1098)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.11](https://github.com/andymai/brepkit/compare/v2.126.10...v2.126.11)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **topology:** trim NURBS edge domains to validated forward endpoint\nsub-spans ([#1097](https://github.com/andymai/brepkit/issues/1097))\n([c3575af](https://github.com/andymai/brepkit/commit/c3575af0d02263d954db772a4095494a7ba25e1e))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.11 with a topology fix. NURBS edge domains\nare trimmed to validated forward endpoint sub-spans to prevent invalid\nspans.\n\n<sup>Written for commit d42071ee173cbadc87e77f9ef48cd2629af84e49.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1098?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T11:30:59Z",
          "tree_id": "610b01061d02724988a193f234ce9d4d5a785058",
          "url": "https://github.com/andymai/brepkit/commit/9f94c9208ba2d273d16c58743f7cebcf7292cc97"
        },
        "date": 1784287998253,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 907376,
            "range": "± 2973",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1005908,
            "range": "± 15881",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13065,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 634362,
            "range": "± 592",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28715744,
            "range": "± 112478",
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
          "id": "95b899ad2077e58e916a25cb627cb2b252887238",
          "message": "fix(algo): accept reversed NURBS sub-spans and run lens interior search as a last resort (#1099)\n\n## Summary\n\nSecond landing of the NURBS endpoint-trimmed convention work (#1097),\nfrom the snapClip deepened-notch dig. Two coupled fixes:\n\n### 1. Reversed sub-spans on open curves (topology)\n\n`domain_with_endpoints` now also accepts a REVERSED validated sub-span\non a clearly open NURBS curve, returning `t₀ > t₁` so start→end\ninterpolation stays truthful. Closed curves keep the full-domain\nfallback: a reversed projection pair there is usually a seam-crossing\nforward sub-arc, and interpolating backward would trace the complement\narc. Unit tests updated: reversed-on-open now trims (and evaluates back\nto its endpoints); a new closed-fitted-loop test pins the reversed-pair\nfallback.\n\n### 2. Curved-lens interior search as a last resort (algo)\n\nA cylinder/cone wall can reach classification carrying a closed curved\ninner wire — e.g. a pre-existing notch outline kept through the generic\nsplit path — with no precomputed interior, because only the\ninternal-loops splitter ran the dedicated remainder search. The\ncurved-lens flag then aborted the analytic split unconditionally (\"no\ncontained interior for curved-lens wall\"). `fill_images_faces` now runs\n`cylinder_cone_remainder_interior` as a last resort at the consumption\npoint; the abort remains, but only when even the dense 256×17 grid finds\nno contained point.\n\nThese are coupled: reversed spans re-shape the deepened-notch split so a\nlegitimate pre-existing lens hole lands on a generic-split wall, which\nwithout fix 2 aborts the whole analytic cut.\n\n## Verification\n\n- Deepened-notch raw repro (arena-captured operands, RAWN=1): posBad 10\n→ 6; both mirrored micro-edge junction chains fully resolve. The\nremaining 6 unpaired edges are one already-mapped root (the cone face is\nnot split by the marched cone×plane sections — the pave-machinery-bypass\nrow), documented in the roadmap update in this PR.\n- d4 canary green; full workspace green in release (one unrelated flaky\nGPU-adapter-contention test, `compute_mesh_lod`, passes standalone).\n- New/updated unit tests in `edge.rs`.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAccept reversed NURBS sub-spans on open curves and add a last-resort\ninterior search for curved-lens walls to avoid false analytic-split\naborts. This fixes deepened-notch cases and reduces unpaired edges\n(posBad 10 → 6).\n\n- **Bug Fixes**\n- Topology: `EdgeCurve::domain_with_endpoints` accepts validated\nreversed spans on clearly open NURBS curves, returning `t0 > t1`; closed\ncurves still fall back to the full domain.\n- Algo: In `fill_images_faces`, run `cylinder_cone_remainder_interior`\nas a last resort for cylinder/cone walls with curved inner wires but no\nprecomputed interior; abort only if the dense grid finds no interior\npoint.\n\n<sup>Written for commit ad59fffeb987f2c3b8496f4cc3d958d947fcfe4f.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1099?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T04:43:13-07:00",
          "tree_id": "9896bfbe267abd548f1a7647c4a0b2b8e93cb886",
          "url": "https://github.com/andymai/brepkit/commit/95b899ad2077e58e916a25cb627cb2b252887238"
        },
        "date": 1784288712762,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 701392,
            "range": "± 5346",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 781455,
            "range": "± 6429",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 9480,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 505577,
            "range": "± 2559",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 22187734,
            "range": "± 45164",
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
          "id": "2535be91cd013fa9b0aa64286b8a1ef4f68cae06",
          "message": "chore(main): release 2.126.12 (#1100)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.12](https://github.com/andymai/brepkit/compare/v2.126.11...v2.126.12)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** accept reversed NURBS sub-spans and run lens interior search\nas a last resort\n([#1099](https://github.com/andymai/brepkit/issues/1099))\n([95b899a](https://github.com/andymai/brepkit/commit/95b899ad2077e58e916a25cb627cb2b252887238))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.12 with a robustness fix in the geometry\nalgorithm: handles reversed NURBS sub-spans and adds a last-resort lens\ninterior search to reduce missed solutions.\n\n- **Bug Fixes**\n  - Accept reversed NURBS sub-spans during evaluation.\n  - Add lens interior search fallback for edge cases.\n\n<sup>Written for commit 87fb8e6dff2303402c9fd03d0edff21930f9144e.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1100?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T11:50:04Z",
          "tree_id": "dfdda13f94a4471413a747283c0df4d8032d94b2",
          "url": "https://github.com/andymai/brepkit/commit/2535be91cd013fa9b0aa64286b8a1ef4f68cae06"
        },
        "date": 1784289127126,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 729024,
            "range": "± 4338",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 782515,
            "range": "± 6888",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10235,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 505816,
            "range": "± 3261",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21845777,
            "range": "± 61561",
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
          "id": "ece2163946b3a69b794ad8b829e01b614ca73cbe",
          "message": "docs(skills): roadmap — deepened-notch residual re-identified as the terminal stranded-rim case (#1101)\n\nIteration-19 decode: the pave-bypass theory is dead (the cone face's\nsection gate is open and it receives all three notch sections). The\nremaining posBad=6 is the sections arriving with different terminal rims\nper side (old z=−1.19 rim on the cone, new z=−1.2 rim on the wall\nplanes) — i.e. the already-documented TERMINAL deepened-notch\nstranded-old-floor-edge case. Updates the row so future sessions start\nat the detection design (polygon_union) instead of re-deriving this.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate skills roadmap to reclassify the residual as the terminal\ndeepened-notch stranded-rim case and drop the pave-bypass theory.\nDocuments the rim mismatch (cone trimmed to old z=−1.19; walls to new\nz=−1.2), confirms 2.126.12 tool verification, and sets the next step to\ndetect deepened-notches via `polygon_boolean::polygon_union` without\nnear-coincident false positives.\n\n<sup>Written for commit c75f2320ae6b5ae4c536078db3d82ab0b2012f2e.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1101?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T12:11:44Z",
          "tree_id": "f43f9ed5798d7555926a1d4ae3524ffffa5b574b",
          "url": "https://github.com/andymai/brepkit/commit/ece2163946b3a69b794ad8b829e01b614ca73cbe"
        },
        "date": 1784290435314,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 874281,
            "range": "± 1488",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 965836,
            "range": "± 7694",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11981,
            "range": "± 298",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 637372,
            "range": "± 4109",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26618840,
            "range": "± 93531",
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
          "id": "f0f8e0e1911924effce9b9f73e060fccfdfc48b6",
          "message": "fix(algo): clip curved-face sections to the outer region (deepened-notch stranded rim) (#1102)\n\n## Summary\n\nThird and final landing of the snapClip deepened-notch dig: the raw\nrepro goes **6 → 0 unpaired B-Rep edges** (37 → 0 across the whole\ncampaign). The residual was the roadmap's TERMINAL stranded-rim case — a\ncut deepening an earlier opening — solved without a detection heuristic\nby making the curved-face splitter geometrically honest.\n\n## The three fixes (all gated to partial-band quadric faces carrying\nmarched-NURBS boundary edges)\n\n1. **`clip_sections_to_outer_region`**: sections overhanging the\nreceiving face through an OUTER-wire concavity (the earlier cut's bite)\nare clipped in unwrapped UV. Fully-off-face pieces and band-hugging\nsub-span re-traces drop; whole-edge duplicates are kept; mixed sections\nsplit at bisection-refined crossings whose junctions are **snapped onto\nthe boundary curve** and returned as split anchors — so the boundary\nsplitter's exact 1e-7 on-curve gate accepts them. The polygon sampler\norders endpoints by the traversal flag (correct arc vs complement for\nreversed circle edges) then orients samples to wire order empirically (a\nwhole-edge NURBS traces the curve's own direction) — each half alone\nfails a different edge class.\n2. **Stale parent pcurves**: registry-presplit section pieces keep their\nparent's pcurve, so endpoint UVs evaluate at the parent's ends and\ndisconnect from the boundary in UV. A v-disagreement-gated refit (v is\nnon-periodic, so a mismatch is unambiguous where u could be a legitimate\n2π translate) rebuilds them on the receiving face.\n3. **Zero-extent edges** from T-junction self-splits derailed the\nangular walker into degenerate single-edge sub-faces; filtered before\nthe tracer (a UV-extent guard protects closed circle sections).\n\n## Verification\n\n- Deepened-notch raw repro: posBad 6 → 0; both cone faces split into\nexactly kept-region + bite sub-faces. Committed as\n`crates/io/tests/snapclip_deepened_notch_inmem.rs` with the tool's exact\nserialized operands.\n- **d4 canary 27/27** — it caught a real regression during development\n(the clip's polygon is garbage on full-revolution primitive laterals;\nnow gated out), exactly what the canary exists for.\n- Full workspace green in release (one unrelated GPU-adapter-contention\nflake, passes standalone).\n- Roadmap row updated (the living-doc discipline); deliberate residual\ndocumented (the sub-resolution B-side corner crescent, the corner-lens\nclass).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes the snapClip deepened‑notch stranded rim by clipping off‑face\nsections on curved faces and repairing split metadata. Cone/cylinder\nfaces now split cleanly in the repro, removing unpaired B‑Rep edges (6 →\n0).\n\n- **Bug Fixes**\n- `clip_sections_to_outer_region`: in UV, drop off‑face pieces and\nboundary re‑trace sub‑spans; keep whole‑edge duplicates; split mixed\nsections and snap junctions to the boundary; outer polygon now samples\nboundary‑edge endpoints. Gated to partial‑band `Cylinder`/`Cone` faces\nwith marched‑NURBS boundary edges; skips full‑revolution laterals.\nReturns anchors to seed the boundary splitter.\n- Refit pcurves on v‑mismatch for registry‑presplit pieces so endpoints\nevaluate on the receiving face (prevents UV disconnects).\n- Filter zero‑extent section edges before traversal, with a named\nUV‑extent guard to preserve closed circle sections.\n- Added `snapclip_deepened_notch_inmem.rs` with serialized fixtures\n(`snapclip_notch_plate.bin`, `snapclip_notch_cutter.bin`); asserts 0\nunpaired position‑quantized B‑Rep edges.\n\n<sup>Written for commit a2e8c5154acee713e281ff55b12d7b5e87b8b785.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1102?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T06:53:13-07:00",
          "tree_id": "5e44aaedc948d0cd7d97939b52c880ab80bf92cd",
          "url": "https://github.com/andymai/brepkit/commit/f0f8e0e1911924effce9b9f73e060fccfdfc48b6"
        },
        "date": 1784296530451,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 936815,
            "range": "± 12677",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1008878,
            "range": "± 2022",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12980,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 643228,
            "range": "± 1622",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28209378,
            "range": "± 55648",
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
          "id": "b5fb6e2cbb1166857a0ddd81a5a2cb7a12341e57",
          "message": "chore(main): release 2.126.13 (#1103)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.13](https://github.com/andymai/brepkit/compare/v2.126.12...v2.126.13)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** clip curved-face sections to the outer region\n(deepened-notch stranded rim)\n([#1102](https://github.com/andymai/brepkit/issues/1102))\n([f0f8e0e](https://github.com/andymai/brepkit/commit/f0f8e0e1911924effce9b9f73e060fccfdfc48b6))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.13 for `brepkit-wasm`, fixing the algorithm to clip\ncurved‑face sections to the outer region. This removes stray interior\ncuts on deepened‑notch stranded rims and improves section accuracy.\n\n<sup>Written for commit fcd02e241f3643ef628483e21285b7a6358498e6.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1103?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T13:59:53Z",
          "tree_id": "05b62c54b51e764923c0c77d7826f1097137f8e5",
          "url": "https://github.com/andymai/brepkit/commit/b5fb6e2cbb1166857a0ddd81a5a2cb7a12341e57"
        },
        "date": 1784296927466,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 877824,
            "range": "± 2835",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 968453,
            "range": "± 1505",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11825,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 644124,
            "range": "± 7680",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26475415,
            "range": "± 716584",
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
          "id": "ac46c583649d85f303d98e7dcfef67780cf1255d",
          "message": "fix(algo): merge overlapping deepened wall openings in the internal-loops splitter (#1104)\n\n## Summary\n\nThe snapClip join-edges baseplate export\n(`baseplateGenerator.scenario.snapClip`, 5×4 right/front/back join)\nfailed with 5 non-manifold STL edges, and its 44-hole connector chain\nran almost entirely on mesh-fallback booleans. Fresh operand capture on\n2.126.13 localized the break to `op-cut-3`: the plate flips from 615\nanalytic faces to an 8203-face mesh fallback and never recovers.\n\n**Root cause.** The snap-slot cutter stack cuts the same seam wall\ntwice: hole-2 opens a window in the x=105 join wall, and hole-3 (whose\ntop floats 0.01 above hole-2's floor) re-opens an overlapping window in\nthat same wall. Hole-3's sections form a closed internal loop on the\nwall face that OVERLAPS the existing inner wire.\n`plane_internal_line_loops` only tests loops against the OUTER boundary,\nso the face routed to `split_face_with_internal_loops`, which attached\nthe loop as an independent second hole. The two wires double-cover the\n0.01 overlap band: both rims stay as unpaired edges, the collinear band\npieces trace twice (use=3 micro verticals), the analytic gate rejects\nthe cut, and every subsequent cut inherits the fallback. This is the\nplane-face twin of the cone stranded-rim case closed in #1102 — the\nroadmap's deepened-notch terminal row.\n\n**Fix.** A union pre-pass in `split_face_with_internal_loops`, gated to\nplane faces with all-Line loop+hole: detect genuine geometric overlap\n(proper crossing, vertex containment, or collinear overlap span), split\nboth edge sets at mutual crossings and vertex T-points, classify pieces\nby midpoint (In/On/Out), and emit ONE merged opening outline (collinear\nOn-pieces contributed once, from the hole's copy so they pair with the\npave-split neighbor faces) plus the removable disc bounded by\nloop-pieces-outside + hole-pieces-inside. All 2D tests project 3D\nendpoints through one locally built `PlaneFrame` — stored hole-wire UVs\ncan be fitted in a foreign frame (the pcurve-convention lesson). Any\nchaining failure bails to the previous behavior.\n\n**Result.** The full 44-hole join-edges chain now replays analytic and\nwatertight natively: final F=881 (539 plane / 182 cylinder / 160 cone),\nposition-quantized edge pairing clean — versus F=8207 with 86 boundary\nedges before. Synthetic 3-box fixture included.\n\n## Verification\n\n- New fixture `crates/io/tests/deepened_wall_opening_inmem.rs`:\nstranded-rim pairing, single union hole on the wall, exact volume — 10×\nflake gate clean\n- Full io suite 30/30 targets green (groove-mouth, junction-disc,\nsnap-slot, cornerclip, divider-lip, honeycomb, halfSockets, intwidth,\nfracwidth foils all green)\n- d4 canary 27/27 (`cargo test -p brepkit-wasm --lib gridfinity`)\n- `brepkit-algo` 157 + `brepkit-operations` 765 green;\n`check-boundaries.sh` clean\n- Full workspace suite green via pre-push hook\n\nRoadmap updated in the same PR (terminal row retired; nozzle-chain\nresidual re-scoped with its dig recipe).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nMerges overlapping deepened wall openings in the internal-loops splitter\nso plane walls keep a single union hole and cuts stay analytic. Fixes\nthe snapClip join-edges export by preventing double-covered rims and\nmesh fallback.\n\n- **Bug Fixes**\n- Added a union pre-pass in `split_face_with_internal_loops` for plane\nfaces with all-Line loop+hole: detect overlap (crossing, containment, or\ncollinear span), split at intersections/T-points, classify pieces, and\nemit one union hole plus the correct removable disc; epsilons derive\nfrom kernel tolerance (`tol*100`), and all 2D tests project 3D endpoints\nthrough one local `PlaneFrame` built from `wire_pts`; bails to the old\npath on any failure.\n- Prevents unpaired rims and double-traced bands; the 44-hole snapClip\nplate now replays analytic and watertight (F=881, posBad=0; was F=8207\nwith 86 boundary edges).\n- Added `crates/io/tests/deepened_wall_opening_inmem.rs` to guard the\nstranded-rim case; updated roadmap notes; all test suites pass.\n\n<sup>Written for commit c131cf1dba68c5df506fa22a6ec2507b151fdb87.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1104?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T15:24:03Z",
          "tree_id": "b2fc96c7d4302a7a3cdc8eb43ac660162feaaf98",
          "url": "https://github.com/andymai/brepkit/commit/ac46c583649d85f303d98e7dcfef67780cf1255d"
        },
        "date": 1784301963139,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 704659,
            "range": "± 4318",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 780704,
            "range": "± 1969",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10243,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 492146,
            "range": "± 6884",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21937098,
            "range": "± 225533",
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
          "id": "540d10857edae68f70b451a002aa73752e366798",
          "message": "chore(main): release 2.126.14 (#1105)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.14](https://github.com/andymai/brepkit/compare/v2.126.13...v2.126.14)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** merge overlapping deepened wall openings in the\ninternal-loops splitter\n([#1104](https://github.com/andymai/brepkit/issues/1104))\n([ac46c58](https://github.com/andymai/brepkit/commit/ac46c583649d85f303d98e7dcfef67780cf1255d))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.14 for `brepkit-wasm`, fixing the internal-loops splitter\nto merge overlapping deepened wall openings. This prevents duplicated\nopenings and keeps wall topology valid.\n\n<sup>Written for commit e887026742f571c865df929345381f7f071f714b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1105?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T15:30:59Z",
          "tree_id": "b1af9724a6afdfb796430687f7d5a310c8a72b7a",
          "url": "https://github.com/andymai/brepkit/commit/540d10857edae68f70b451a002aa73752e366798"
        },
        "date": 1784302377276,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 703720,
            "range": "± 795",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 780260,
            "range": "± 882",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10259,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 497985,
            "range": "± 867",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 22041506,
            "range": "± 24359",
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
          "id": "5bf95348773bad782e3182b7d503ff1fd492787e",
          "message": "fix(algo): trim plane-cone circle sections to exact boundary-crossing arcs (#1106)\n\n## Summary\n\nSecond half of the snapClip join-edges export root (follows #1104). The\nexport baseplate builds with the simplified **tapered** pockets, so each\ndeep relief cutter's back corner lands on a pocket corner **cone**. The\ncutter should contribute three sections to that cone face: two marched\nconics (cone × back wall, cone × side wall) and the cone × cutter-top\narc that closes the chain.\n\n**Root cause.** A horizontal plane cuts a cone in an exact\n`EdgeCurve::Circle`. The exact-arc path that bypasses the sampling\npre-filters (`trim_ellipse_to_boundary_crossings` — built for the\ntilted-tread × cylinder family) only accepted **Ellipse** sections, so\nthe circle fell to the generic 16-sample in-both AABB filter, which\ncannot find a ~0.11-long in-both arc on an ~18-long circle. With the\nclosing arc missing, the cone face received an open two-conic chain, the\ninternal-loops splitter rejected it (open chains are dropped by design),\nand the face stayed unsplit. The resulting analytic-but-leaky solid\n(posBad=8) was accepted by the by-edge-id gate and poisoned the export\nchain into mesh fallback two cuts later (final bnd=111 nm=6 at export\ntolerance).\n\n**Fix.** The trimmer now dispatches over Circle and Ellipse sections\nthrough one angular parameterization (`SecCurve`); emitted arcs keep\ntheir exact curve type so downstream circle-calibrated machinery sees\nreal Circle sections. No other behavior changes — Ellipse handling is\nbyte-identical.\n\n**Result.** The minimal repro (`cut(plate-after-op-cut-2,\ndeep-relief-cutter)`) goes posBad 8→0 on both the raw GFA and ops paths,\nand the full 44-hole export-variant join-edges chain replays analytic\nand watertight: final **F=418, posBad=0** (was F=4842, bnd=111, nm=6).\n\n## Verification\n\n- New fixture `crates/io/tests/snapclip_export_corner_inmem.rs` (tool's\nserialized operands, 2026-07-17 export-variant capture): edge pairing +\nanalytic-cone assertions, 10× flake gate clean\n- Full io suite green (including the tilted-tread/halfSockets ramp\nlandscapes that calibrate the ellipse path)\n- d4 canary 27/27; `brepkit-algo` and `brepkit-operations` suites green\n- Full workspace suite green via pre-push hook\n\nRoadmap updated in the same PR. Known remaining snapClip residuals (out\nof scope, recorded with repro recipes): the 0.6mm-nozzle export chain's\nop-cut-3 (posBad=10, different landscape), the by-edge-id acceptance\ngate's blindness to position-duplicate leaks, and the bed-flat clip\nvolume pin.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes trimming for plane×cone intersections by treating circle sections\nas exact boundary-crossing arcs. Restores the short cutter-top closing\narcs, so cone faces split and the snapClip export chain stays analytic\nand watertight.\n\n- **Bug Fixes**\n- Updated `trim_ellipse_to_boundary_crossings` to handle both\n`EdgeCurve::Circle` and `EdgeCurve::Ellipse` via a shared angular\nparameterization (`SecCurve`); emitted arcs keep their exact type.\n- Avoids dropping ~0.11-length circle arcs in the 16-sample in-both\nfilter that left open two-conic chains on cone faces.\n- Added `crates/io/tests/snapclip_export_corner_inmem.rs` with\nserialized operands to assert analytic cones and B-Rep edge pairing.\n- Result: minimal repro goes posBad 8→0; full 44-hole export variant\nreplays analytic `F=418, posBad=0` (was `F=4842, bnd=111, nm=6`).\n\n<sup>Written for commit ca4ce7547f42bf90934563f52c3bd7f897bfe311.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1106?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T10:39:17-07:00",
          "tree_id": "a9d7e626835c27f693ae5c772d636d244bcd9da2",
          "url": "https://github.com/andymai/brepkit/commit/5bf95348773bad782e3182b7d503ff1fd492787e"
        },
        "date": 1784310077671,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 705839,
            "range": "± 1229",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 783968,
            "range": "± 2022",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10170,
            "range": "± 255",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 497173,
            "range": "± 2729",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21926980,
            "range": "± 53477",
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
          "id": "7a314b91490b39283a3217552142cbd01298c648",
          "message": "chore(main): release 2.126.15 (#1107)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.15](https://github.com/andymai/brepkit/compare/v2.126.14...v2.126.15)\n(2026-07-17)\n\n\n### Bug Fixes\n\n* **algo:** trim plane-cone circle sections to exact boundary-crossing\narcs ([#1106](https://github.com/andymai/brepkit/issues/1106))\n([5bf9534](https://github.com/andymai/brepkit/commit/5bf95348773bad782e3182b7d503ff1fd492787e))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.15 for `brepkit-wasm`, fixing plane–cone circle\nintersections to trim to exact boundary-crossing arcs. This improves\nsection precision and prevents arcs extending beyond cone boundaries.\n\n<sup>Written for commit ad4d4dca2244a18e46933cbd316d72839c79e6c8.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1107?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-17T17:46:12Z",
          "tree_id": "ea223efa82eb46c9ea9e277d3a2bff95d5b862dd",
          "url": "https://github.com/andymai/brepkit/commit/7a314b91490b39283a3217552142cbd01298c648"
        },
        "date": 1784310506337,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 883469,
            "range": "± 2163",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 971536,
            "range": "± 2975",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11949,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 644502,
            "range": "± 7066",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26830735,
            "range": "± 72169",
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
          "id": "503ae757184a0f0ed370d221f25344b6e0b6d782",
          "message": "docs(skills): roadmap — snapClip bed-flat volume pin resolved (#1108)\n\n## Summary\n\nResolves the long-standing snapClip bed-flat clip volume question\n(46.701 brepkit vs the 46.6±0.05 test pin) — **it is not a brepkit\ndefect; brepkit is the more accurate kernel.**\n\nA per-stage dual-kernel volume diff of `buildSnapClip` localized the\nentire delta to the relief cut:\n\n| stage | brepkit | reference | delta |\n|---|---|---|---|\n| base extrude (chamfered/filleted profile) | 52.5508 | 52.5502 |\n+0.0006 (negligible) |\n| lofted relief foot | 7518.3376 | 7518.4000 | reference +0.062 larger |\n| relief removed | 5.8498 | 5.9960 | reference over-removes 0.146 |\n| final clip | 46.7011 | 46.5543 | — |\n\nThe relief cutter is `buildSingleCellSocket` (a lofted gridfinity socket\nfoot). A native face census of brepkit's foot shows **F=34, {plane:18,\ncylinder:8, cone:8}, zero NURBS** — brepkit represents it as *exact\nanalytic* geometry (the #1045 loft-recognition). The reference kernel\nkeeps it as a NURBS loft that bulges ~0.062mm³ outward; used as a relief\ncutter, that bulging wall removes 0.146mm³ more from the clip's\ntop-bridge corners. The cutters' total volumes are identical to 0.001% —\nthe whole difference is the local loft-surface approximation in the\ncorner-overlap sliver.\n\nSo brepkit's 46.701 is the more accurate value, and the 46.6±0.05 pin is\ncalibrated to the reference's NURBS-loft approximation — the roadmap's\n\"snapshot pins are kernel-specific\" class. The resolution is tool-side\npin recalibration, not a brepkit change.\n\nThis also **corrects a prior roadmap claim** (\"genuine deviation, the\nreference kernel passes\") that saw the reference pass the pin but never\nchecked which kernel is more correct.\n\nDocs-only; no code change.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdates the roadmap docs to resolve the snapClip bed-flat clip volume\npin issue: a per-stage dual-kernel diff shows the delta comes from the\nrelief cut, where `brepkit`’s analytic loft is more accurate than the\nreference’s NURBS; the fix is tool-side pin recalibration, not a\n`brepkit` change. Also corrects the earlier claim that this was a\ngenuine deviation that the reference kernel passed.\n\n<sup>Written for commit 293ee607c67acb60b879a306b8b25eabdcbef214.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1108?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T19:32:05Z",
          "tree_id": "113b80c522b7c9049fbf85769e5c458cc752fe38",
          "url": "https://github.com/andymai/brepkit/commit/503ae757184a0f0ed370d221f25344b6e0b6d782"
        },
        "date": 1784316855316,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 903269,
            "range": "± 2103",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1003615,
            "range": "± 2193",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12941,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 638715,
            "range": "± 1357",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28146160,
            "range": "± 28118",
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
          "id": "85747395e443c089bb74882bba1d60fb529544b6",
          "message": "fix(algo): split circle-boundary disc faces cut by chords (#1109)\n\n## Problem\n\nA planar face whose outer boundary is a **single closed circle** — a\ncylinder cap disc, or a pocket-floor disc — that is cut by chord\nsection(s) was **dropped entirely**, leaving its boundary edges unpaired\n(non-manifold / open shell → mesh fallback or open export). The\ncanonical trigger is a box corner biting a cylindrical pocket/cap.\n\n## Root cause — two chained bugs\n\n**Bug A — the chords never reach the splitter.**\n`clip_line_to_face_boundary` bailed on `crossings.len() < 2`. A corner\nchord runs from an interior endpoint (a box-corner vertex *inside* the\ndisc) out to the rim, so it crosses the boundary circle exactly **once**\n→ dropped. The material window is `[crossing, interior endpoint]`, which\nthe midpoint-classification path just below already resolves — but the\nearly bail returned first. Relaxed to admit a single crossing on\n**hole-free plane faces only** (the holed / non-plane fallback still\nrequires a crossing pair).\n\n**Bug B — the disc arrangement can't represent a major arc.** The\ngeneric planar arrangement (`split_plane_face_by_arrangement`)\nrepresents every boundary curve by its **chord** for both crossing\ndetection and half-edge turn angle. A full circle split at 2 chord\ncrossings becomes a **major arc (> π) + minor arc**; the major arc's\nchord cuts deep across the disc, phantom-crossing the section chords and\ncarrying a turn angle far from the true tangent → the remnant is\nmistraced or dropped. Added a gated `try_split_disk_by_chords` that\nbuilds the arrangement **natively** from the analytic circle + chords\n(arcs by angular span, tangent-aware DCEL trace) and emits remnants with\ntrue `Circle`/`Line` geometry. It **defers** unless the boundary is one\ncircle and every section is a chord — so it never fires on mixed\nline/arc boundaries (rounded-rect walls, sectors).\n\n## Verification\n\n- `cargo test -p brepkit-wasm --lib gridfinity` → **27 passed; 0\nfailed**\n- `cargo test -p brepkit-algo` → **163 passed; 0 failed** (157 baseline\n+ 6 new)\n- `cargo test -p brepkit-io` → **214 passed; 0 failed**\n- clippy `-D warnings` clean; fmt clean\n- 6 new committed unit tests pin both fixes.\n\n## Scope\n\nFirst part of the cylinder/disc arrangement-rescue campaign for the\nfunnel/honeycomb boolean family. Foil-safe and independently valuable\n(fixes dropped cap/floor discs broadly); the cylinder-wall arrangement\npart follows separately.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes disc faces with a single circular boundary cut by chord sections.\nThese faces now split into valid regions instead of being dropped,\npreventing open shells and mesh fallbacks.\n\n- **Bug Fixes**\n- In `clip_line_to_face_boundary`, allow a single boundary crossing on\nhole-free plane faces to keep interior→rim chords (e.g., box-corner\nbites).\n- Add gated `try_split_disk_by_chords` that builds an arrangement from\nthe analytic circle + chords, handles major arcs correctly, traces with\ntangent-aware logic, and outputs true `Circle`/`Line` edges; defers on\nnon-disc or mixed boundaries.\n\n<sup>Written for commit 1de1cee523f82b9099739598ea6f2174eb4828e5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1109?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T21:10:31-07:00",
          "tree_id": "9a17127bccf6587ff3b937913c8cda1f40c79c64",
          "url": "https://github.com/andymai/brepkit/commit/85747395e443c089bb74882bba1d60fb529544b6"
        },
        "date": 1784347958257,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 792908,
            "range": "± 56291",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 817441,
            "range": "± 27728",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 9933,
            "range": "± 361",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 534873,
            "range": "± 30143",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 23419823,
            "range": "± 844635",
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
          "id": "686a0c2951733fac5878aa48f9c1392a150f2b1a",
          "message": "chore(main): release 2.126.16 (#1110)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.16](https://github.com/andymai/brepkit/compare/v2.126.15...v2.126.16)\n(2026-07-18)\n\n\n### Bug Fixes\n\n* **algo:** split circle-boundary disc faces cut by chords\n([#1109](https://github.com/andymai/brepkit/issues/1109))\n([8574739](https://github.com/andymai/brepkit/commit/85747395e443c089bb74882bba1d60fb529544b6))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.16 for `brepkit-wasm` fixes splitting of circle-boundary\ndisc faces when cut by chords, preventing invalid topology. Version bump\nand changelog updated.\n\n<sup>Written for commit a398ed1ebc1c7c02d67c68e6dd60ef6253321a7b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1110?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-18T04:17:15Z",
          "tree_id": "2a7739bf946d293924854a05d1d2987175a28b4e",
          "url": "https://github.com/andymai/brepkit/commit/686a0c2951733fac5878aa48f9c1392a150f2b1a"
        },
        "date": 1784348353855,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 728346,
            "range": "± 9440",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 788050,
            "range": "± 3742",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10219,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 500035,
            "range": "± 759",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21803348,
            "range": "± 119548",
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
          "id": "294af788165285f63316fc28a5d3ff8afcad92d2",
          "message": "docs(skills): roadmap — funnel/disc arrangement campaign (part 2 via #1109) (#1111)\n\nRecords the funnel/honeycomb cylinder-disc arrangement campaign in the\nliving roadmap (mandatory maintenance per the roadmap doctrine — #1109\nshipped part 2 without it).\n\n**The campaign**: curved/periodic faces have no arrangement rescue (the\nplane path's rescues are all `is_plane`-gated), so a box cut crossing a\ncylindrical pocket at partial overlap figure-eights the greedy wire\nbuilder. Three sub-gaps:\n- **(2)** plane disc (closed-circle boundary) cut by chords — **CLOSED\nin #1109**\n- **(3)** plane wall + single-arc crossing — OPEN\n- **(1)** cylinder-wall arc-DCEL — foil-safe, re-apply LAST (its\ncorrectness exposes the dropped plane faces)\n\nDetail lives in memory `project_cylinder-arrangement-rescue.md`; scratch\nrepros are `crates/io/examples/replay_{synthbox,diskcut}.rs`.\n\nDocs-only.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdates the roadmap to document the funnel/honeycomb cylinder-disc\narrangement campaign, marking part (2) (plane disc cut by chords) as\nclosed in #1109 and tracking parts (1) and (3) as open. Adds pointers to\n`project_cylinder-arrangement-rescue.md` and\n`crates/io/examples/replay_{synthbox,diskcut}.rs`; clarifies that all\nthree fixes are required for a watertight funnel.\n\n<sup>Written for commit 98af5202a1c6c076e1b17ca1efc7f2212ef044a5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1111?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-18T04:20:09Z",
          "tree_id": "0da9aaffcce81940a4db4a9fb0b5cf84e6403186",
          "url": "https://github.com/andymai/brepkit/commit/294af788165285f63316fc28a5d3ff8afcad92d2"
        },
        "date": 1784348547032,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 883913,
            "range": "± 5093",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 971738,
            "range": "± 5107",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12123,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 653212,
            "range": "± 14059",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26691561,
            "range": "± 131089",
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
          "id": "28569d668edb5587a038040b12acf4ae9a99a84d",
          "message": "fix(algo): cylinder-band arrangement rescue for partial-overlap pocket cuts (#1112)\n\n## Problem\n\nWhen a box cut notches a **cylindrical pocket wall at partial overlap**,\nthe greedy angular wire builder figure-eights — a self-crossing outer\nwire + slit inner wire — leaving the wall non-manifold. Plane faces\nalready get an arrangement rescue when the greedy trace breaks\n(`split_plane_face_by_arrangement`, and #1109's\n`try_split_disk_by_chords`); **periodic cylinder faces had none**. This\nis the root of the funnel/honeycomb boolean family.\n\n## Fix\n\nAdds a gated `split_cylinder_band_by_arrangement`. Key insight: on a\ncylinder every edge is **axis-aligned in UV** — cross-section rings are\nhorizontal (`v=const`), seam/side generators are vertical (`u=const`) —\nso the wall is a **rectilinear arrangement**. It:\n- seam-anchors the strip and derives all coordinates from\n`cyl.project_point` (the stored pcurve UV wraps the seam inconsistently\n— same seam appears at `u`, `u+2π`, and negative `u`),\n- pairs section generators by kept/removed alternation, traces minimal\nfaces with a rectilinear DCEL,\n- reconstructs each sub-edge with true analytic `Line` (generator) /\n`Circle` (ring) geometry.\n\n**Gated exactly like the plane rescue** — fires only when the greedy\nloops self-cross, overlap, or go degenerate, and defers (`None`) on any\nnon-rectilinear section — so it never changes a face the greedy already\nhandles. **Purely additive** (one new function + one call site + 2\ntests); no existing logic modified.\n\n## Verification\n\nSynthetic pocket-notch repro — strict improvement, no regression:\n\n| case | before | after |\n|---|---|---|\n| base | 0 | 0 |\n| box-past-pocket | 0 | 0 |\n| clear-corner | 6 | **0** |\n| top-inside-pocket | 10 | **1** |\n\n- `cargo test -p brepkit-wasm --lib gridfinity` → **27/0**\n- `cargo test -p brepkit-algo` → **165/0** (163 + 2 new)\n- `cargo test -p brepkit-io` → **214/0**\n- clippy `-D warnings` clean; fmt clean\n\nThe wall is now a correct manifold comb (no figure-eight). Completes the\ncylinder/disc arrangement-rescue campaign (parts 2/3 in #1109).\n\n## Known residual\n\nThe single `top-inside` residual is a **separate** bug:\n`merge_duplicate_edges` (grouped by endpoint pair only) collapses the\nfloor lens's bottom-rim **arc** and the tool-cut **chord** (same two\nendpoints). This is the known co-endpoint arc-vs-chord class the roadmap\nflags — the sanctioned fix is a splitter-side midpoint split (NOT a\nsmarter merge-key, which is proven unbuildable), tracked as a follow-up.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes non-manifold cylinder walls caused by partial-overlap box cuts by\nadding a rectilinear arrangement rescue for cylinder bands. Prevents\nfigure-eight wires and builds correct manifold combs without affecting\ncases the greedy path already handles.\n\n- **Bug Fixes**\n- Added `split_cylinder_band_by_arrangement` to handle cylinder bands as\na rectilinear arrangement (rings horizontal, generators vertical).\n- Seam-anchors the strip, derives UV from surface projection, pairs side\ngenerators, traces minimal faces, and rebuilds edges with analytic\n`Line`/`Circle`.\n- Gated to run only when the greedy trace breaks; defers on any\nnon-rectilinear section.\n  - Purely additive with two new tests; no existing logic changed.\n\n<sup>Written for commit 773b5f935cd191622812836e2570839845a44977.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1112?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T22:06:04-07:00",
          "tree_id": "80165b984dd9c2e98b137bc54b4d1c733616cee7",
          "url": "https://github.com/andymai/brepkit/commit/28569d668edb5587a038040b12acf4ae9a99a84d"
        },
        "date": 1784351292179,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 878028,
            "range": "± 1788",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 968879,
            "range": "± 2163",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11926,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 647226,
            "range": "± 1267",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26570172,
            "range": "± 169681",
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
          "id": "fc791e0313d42d5cb718e856dd0132768ebe05d9",
          "message": "chore(main): release 2.126.17 (#1113)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.17](https://github.com/andymai/brepkit/compare/v2.126.16...v2.126.17)\n(2026-07-18)\n\n\n### Bug Fixes\n\n* **algo:** cylinder-band arrangement rescue for partial-overlap pocket\ncuts ([#1112](https://github.com/andymai/brepkit/issues/1112))\n([28569d6](https://github.com/andymai/brepkit/commit/28569d668edb5587a038040b12acf4ae9a99a84d))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.17 for `brepkit-wasm` fixes the cylinder-band arrangement\nrescue for partial-overlap pocket cuts. This reduces failures and\nincorrect geometry on cylindrical surfaces.\n\n<sup>Written for commit 5a0f070d1db5c31a625a8a364900e2cb493832ef.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1113?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-18T05:14:06Z",
          "tree_id": "2d7c8d76f52c3747b8d596146f80710e94237e28",
          "url": "https://github.com/andymai/brepkit/commit/fc791e0313d42d5cb718e856dd0132768ebe05d9"
        },
        "date": 1784351757376,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 583196,
            "range": "± 20710",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 649269,
            "range": "± 1569",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 8263,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 485150,
            "range": "± 33155",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 18803323,
            "range": "± 557244",
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
          "id": "f8c46fa47829ad9b8b67035f106be51f0834226d",
          "message": "fix(algo): split co-endpoint lens arc in disc-chord split (funnel watertight) (#1114)\n\nCloses the cylinder/disc arrangement-rescue campaign (#1109, #1112) —\nthe funnel/honeycomb boolean is now watertight.\n\n## Problem\n\nAfter #1112, the synthetic pocket-notch repro reached posBad=0 on every\ncase **except** `top-inside` (posBad=1). Root: the pocket-floor crescent\nleft by a box-corner cut is a **co-endpoint lens** — a minor rim **arc**\nand the tool-cut **chord** share both endpoints. `merge_duplicate_edges`\nkeys edges by endpoint pair alone, so it collapsed the arc into the\nchord and degenerated the lens face.\n\n(The complementary `x>16` lens is fine only because its arc happens to\npass through the cylinder seam vertex, which an existing reconciliation\nsplits — the `y>16` arc has no interior vertex, so it collapses.)\n\n## Fix — the sanctioned splitter-side split (not a merge-key change)\n\nA smarter `merge_duplicate_edges` key is **proven unbuildable** (the\ngridfinity lip corner chord+arc MUST merge; the torus-box lens line+arc\nMUST stay distinct). So the fix is splitter-side: in\n`try_split_disk_by_chords`, when a **minor** arc (< π) has both\nendpoints also joined by a chord (the co-endpoint lens condition), split\nit at its angular midpoint. The new on-circle vertex breaks the\nshared-endpoint collision, and the **existing**\n`split_arc_edges_at_collinear_vertices` reconciliation propagates the\nidentical cut to the coincident cylinder-wall rim arc (position-based,\nsymmetric) — **no two-site coordination**. Restricted to minor arcs so\ndiameter half-discs and major-arc partitions keep their calibrated\nbehavior.\n\n## Verification\n\n`replay_synthbox` — **posBad=0 on every case** (`top-inside` 1→0;\nbase/clear-corner/past-pocket stay 0):\n\n- `cargo test -p brepkit-wasm --lib gridfinity` → **27/0**\n- `cargo test -p brepkit-algo` → **166/0** (+1 regression test)\n- `cargo test -p brepkit-io` → **214/0**\n- `cargo test -p brepkit-operations` → **972/0**\n- clippy `-D warnings` clean; fmt clean\n\nAlso records the campaign close in the roadmap skill.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nSplit minor co-endpoint lens arcs in the disk-by-chords splitter to\nprevent arc+chord collapsing; now guarded to avoid quantization\ncollisions. The funnel/honeycomb boolean is watertight, fixing the\npocket-floor crescent in the top-inside case and completing the\ncylinder/disc arrangement work (#1109, #1112).\n\n- **Bug Fixes**\n- In `try_split_disk_by_chords`, when a minor rim arc (< π) shares\nendpoints with a chord, split the arc at its angular midpoint and insert\nan on-circle vertex — only if the midpoint creates a new vertex (skip on\nquantization alias).\n- Propagate the cut to coincident rim arcs via\n`split_arc_edges_at_collinear_vertices`; no merge-key changes.\n  - Preserve diameter half-discs and major-arc partitions.\n- Add a regression test and update docs; `replay_synthbox` now reports\nposBad=0 for all cases.\n\n<sup>Written for commit c5b40455b5082fe1a684ad3730658db09126a89b.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1114?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-17T22:54:38-07:00",
          "tree_id": "f86b198d00ee3a8ee7be7be63b70f9a752a717ed",
          "url": "https://github.com/andymai/brepkit/commit/f8c46fa47829ad9b8b67035f106be51f0834226d"
        },
        "date": 1784354217073,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 876719,
            "range": "± 5007",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 968693,
            "range": "± 14432",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11935,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 642817,
            "range": "± 1074",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26637131,
            "range": "± 371412",
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
          "id": "f9dc76dc51a247b7ee4464efb189d62ca6e471c0",
          "message": "chore(main): release 2.126.18 (#1115)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.18](https://github.com/andymai/brepkit/compare/v2.126.17...v2.126.18)\n(2026-07-18)\n\n\n### Bug Fixes\n\n* **algo:** split co-endpoint lens arc in disc-chord split (funnel\nwatertight) ([#1114](https://github.com/andymai/brepkit/issues/1114))\n([f8c46fa](https://github.com/andymai/brepkit/commit/f8c46fa47829ad9b8b67035f106be51f0834226d))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.18 with a geometry fix to keep funnel\nsplits watertight by correctly handling co-endpoint lens arcs in\ndisc–chord splits.\n\n- **Bug Fixes**\n- Split co-endpoint lens arc during disc–chord splitting to prevent gaps\nand ensure watertight funnels.\n\n<sup>Written for commit 15951047840c2f75113705afd901726ff123beb0.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1115?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-18T06:02:26Z",
          "tree_id": "7263e4a2413efb4ba346f306121c9bb40be76814",
          "url": "https://github.com/andymai/brepkit/commit/f9dc76dc51a247b7ee4464efb189d62ca6e471c0"
        },
        "date": 1784354675722,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 940184,
            "range": "± 46212",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1007986,
            "range": "± 2308",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12938,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 646149,
            "range": "± 2333",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28228352,
            "range": "± 76637",
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
          "id": "cf463488bc7e433e0da559dcdee9248c3224c04f",
          "message": "docs(skills): roadmap — correct funnel campaign scope after tool re-probe (#1116)\n\nThe 2.126.17 tool re-probe shows the cylinder-disc arrangement campaign\n(#1109/#1112/#1114) closed an **engine sub-class**\n(cylinder-pocket-notch / disc-chord) — foil-safe (27/0) and **no tool\nregression** (export-integrity: 33 real fails = same known deferred\nfamilies; published solid-cutouts 6=6; the other 180 fails are the\npre-existing kernel-poison cascade).\n\n**But the tool's own `combined features › 2×2 honeycomb walls + funnel\ncutout` scenario is NOT fixed** — still a 533s bisect-hang + fail. Its\nroot is the separate honeycomb-cut coincident-wall assembler hang + the\nfunnel-cutout, not this cylinder-arrangement bug. The synthetic proxy\nfixed a real class but wasn't validated against the real\nhoneycomb+funnel operands first (the roadmap's own warning).\n\nDocs-only; records the scope correction so the roadmap isn't\noverclaiming tool-parity impact.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdates the roadmap docs to correct scope: the cylinder–disc arrangement\ncampaign closed only the engine sub-class (foil-safe, no regression),\nwhile the tool’s \"2×2 honeycomb walls + funnel cutout\" case still fails\ndue to a separate honeycomb coincident-wall assembler issue. Clarifies\nthat tool parity did not move and key failing families (scoop #11, screw\nbase #12, solid cutouts #13, honeycomb-cut) remain open.\n\n<sup>Written for commit 08368a4bd58628cc13d34c3a710b4210f0e093f8.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1116?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-18T12:48:48Z",
          "tree_id": "8dfe873f115a8fd77c48ea383bed9e28b6264a3b",
          "url": "https://github.com/andymai/brepkit/commit/cf463488bc7e433e0da559dcdee9248c3224c04f"
        },
        "date": 1784379065548,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 884883,
            "range": "± 4983",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 970832,
            "range": "± 52991",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12115,
            "range": "± 217",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 651245,
            "range": "± 1018",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26796862,
            "range": "± 56070",
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
          "id": "54aa016771e1a64c6a5b492d31c48ebb8ca258e7",
          "message": "fix(tessellate): keep self-intersecting planar caps watertight via fan fallback (#1117)\n\n## Summary\nFixes the gridfinity \"2×2 solid with slot (stadium) cutout\" export\nproducing a non-watertight STL (8 boundary edges) — the first of the\nsolid-cutouts parity failures. Root-caused with a faithful native repro\n(operands captured from the real tool cut site).\n\n## Root cause\n**Tessellation defect, not a boolean bug.** `boolean(Cut, bin, stadium)`\nyields a watertight, fully-analytic B-Rep (F=60, euler=2, nm=0, bd=0).\nBoundary edges appear only when tessellating at fine deflection (0 @\ndefl=0.5 → 8 @ 0.05 — the count-grows-with-refinement signature).\n\nThe offending face is the bin's top ledge plane: a weaving,\nsimply-connected planar annulus. At one corner the inner socket-foot\ncone rim (circle r=1.25) and the bin's outer rounded corner (circle\nr≈5.99) **genuinely intersect** — the inner rim bulges ~0.11mm *past*\nthe outer arc, so the ledge pinches through zero to negative width. The\nprojected boundary polygon is therefore self-intersecting.\n\n`cdt_triangulate_simple` on such a polygon: CDT recovers the crossing\nconstraints with a Steiner vertex, then the input-index mapping\n**silently drops** every triangle touching it → the pinch sliver is left\nuntriangulated → single-use mesh edges → cracks.\n\n## Fix\nDetect that CDT introduced Steiner vertices (`mapped <\ncdt_triangles.len()`) and fall back to `fan_triangulate`, which uses\nonly the original boundary vertices and is manifold by construction\nregardless of the self-overlap. Steiner vertices arise *only* for\nself-intersecting boundaries, so normal planar faces are unaffected.\nMinimal (+13/-1 in planar.rs); no L0–L2 code touched.\n\n## Verification\n- Faithful repro: boundary_edges **8 → 0** at defl 0.1/0.05/0.01; cut\nB-Rep unchanged (nm=0 bd=0).\n- `brepkit-operations --lib tessellate` **72/0** (incl. new\n`pinched_ledge_prism_is_watertight`); `brepkit-operations --lib`\n**766/0**; `brepkit-wasm --lib gridfinity` **27/0** (canary);\n`brepkit-io` green; clippy `-D warnings` clean.\n\n## Follow-up (not in scope)\nThe same Steiner-drop pattern exists in the holed CDT paths\n(`run_planar_cdt`, `tessellate_planar_shared_with_holes`) — no current\nrepro self-intersects an *inner* wire, so untouched. Noted in the\nroadmap.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes tessellation cracks in self-intersecting planar caps by falling\nback to a fan when CDT inserts Steiner vertices, keeping caps\nwatertight. The gridfinity “2×2 solid with stadium slot” now exports a\nwatertight mesh.\n\n- **Bug Fixes**\n- In `cdt_triangulate_simple`, detect Steiner insertion (`mapped <\ncdt_triangles.len()`) and fall back to `fan_triangulate`; also fall back\nif CDT maps no triangles. Only applies to self-intersecting boundaries;\nnormal faces are unchanged.\n- Adds `pinched_ledge_prism_is_watertight` regression with captured\npolygons; asserts 0 boundary and non‑manifold edges at deflections 0.1\nand 0.05.\n\n<sup>Written for commit 77f44b1d8a50c9c430348e48d370fc697bcb6193.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1117?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-18T10:00:53-07:00",
          "tree_id": "05f356594346da31bce1f210dc9ddbb87d448064",
          "url": "https://github.com/andymai/brepkit/commit/54aa016771e1a64c6a5b492d31c48ebb8ca258e7"
        },
        "date": 1784394194579,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 880014,
            "range": "± 2187",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 974173,
            "range": "± 4021",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11869,
            "range": "± 220",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 645214,
            "range": "± 12679",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 27008439,
            "range": "± 133604",
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
          "id": "d2893af8807b2e7c6c52e90cba2a3ad9cce3bfa7",
          "message": "chore(main): release 2.126.19 (#1118)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.19](https://github.com/andymai/brepkit/compare/v2.126.18...v2.126.19)\n(2026-07-18)\n\n\n### Bug Fixes\n\n* **tessellate:** keep self-intersecting planar caps watertight via fan\nfallback ([#1117](https://github.com/andymai/brepkit/issues/1117))\n([54aa016](https://github.com/andymai/brepkit/commit/54aa016771e1a64c6a5b492d31c48ebb8ca258e7))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.19 for `brepkit-wasm` with a tessellation fix to keep\nself-intersecting planar caps watertight. Prevents gaps and improves\nmesh integrity in these cases.\n\n- **Bug Fixes**\n- Tessellate: add fan fallback to keep planar caps watertight when faces\nself-intersect.\n\n<sup>Written for commit 15b57018a0f57fd89e292e3311fcc5ea18867e7c.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1118?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-18T17:08:05Z",
          "tree_id": "e263d99b99f1219d7b465fb3a07c7ff0b05dad33",
          "url": "https://github.com/andymai/brepkit/commit/d2893af8807b2e7c6c52e90cba2a3ad9cce3bfa7"
        },
        "date": 1784394627077,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 903344,
            "range": "± 1519",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1008365,
            "range": "± 5345",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13024,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 639232,
            "range": "± 14002",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28138337,
            "range": "± 62534",
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
          "id": "79a82f05041fa0d5f306cf9875badebd5e6ee791",
          "message": "fix(algo): salvage closed-circle cap sections in the planar face splitter (#1119)\n\n## Summary\nFixes the gridfinity **screw-base export family** (task #12): 9 of 13\nscenarios go from non-watertight to clean, and the 4×4 stress case drops\nfrom **1792 → 3** boundary edges. Root-caused end-to-end with a faithful\nnative repro captured from the real tool pipeline.\n\n## Root cause\nThe screw/magnet holes are cylinder cuts into the base socket —\nwatertight in isolation. The failure is the final **socket → bin-body\n`Fuse`**: the socket's top face (Plane z=5) carries the drilled-hole\nrims as **closed-circle** sections, coincident with the body's bottom\nplane. `arrangement_regions_from_inputs` retains only inputs with a\nnon-zero UV chord, and a **closed circle has start == end** (zero-length\nchord) → every hole cap is dropped. The impl's single-closed / all-Line\nfast path only fires for a *lone* circle, so two-or-more circles (or\ncircles mixed with the socket's corner-cone arcs) lose their caps → the\ndrilled cylinder rims are left as free edges → GFA opens (raw `bd=4`) →\noperations mesh-falls-back to hundreds of all-planar faces (the 112–1792\nexport boundary edges).\n\nThis was mis-diagnosed twice before landing here (first as a #696\ntessellation crack, then as a same-domain inner-wire drop); both were\nrefuted with literal data. The verified root is the closed-circle drop\nin the planar arrangement splitter.\n\n## Fix\n`split_face_2d` now wraps `split_face_2d_impl`: on a **plane** face it\npeels off **genuine** cap circles (closed `Circle` sections *strictly\ninterior* to the outline — centre inside by more than the radius, which\nrejects the tangent corner-cylinder remnants the pave machinery leaves\nat faceted junctions), splits the face by the remaining open sections\nthrough the unchanged impl, then carves each cap into its containing\nsub-face via the existing `split_face_with_internal_loops` (disc cap +\nholed remainder). A lone cap circle still routes through the impl's fast\npath untouched, and faces with no interior cap circle are byte-identical\nto before. General to the coincident-planar-interface-with-drilled-holes\nclass, not screw-specific.\n\n## Verification\n- Faithful native repro: raw `gfa::boolean(Fuse)` **bd 4 → 0**;\noperations result **672 all-plane mesh-fallback → 71 analytic faces,\nwatertight**.\n- Tool re-probe (overlay): **screw family 9/13 pass** (was 0/13); 4×4\nstress 1792→3; **no regression** (solid-cutouts unchanged 5 fail/23\npass).\n- Foils: `algo` 166/0, `operations` 767/0 (incl. new\n`fuse_capping_slab_preserves_drilled_hole_caps`, verified\nfails-without/passes-with), `wasm gridfinity` **27/0**, `io` green,\nclippy `-D warnings` clean.\n\n## Follow-ups (not in scope)\n4 residual screw scenarios: 4×4 stress (3 bd — a few near-grid-boundary\nholes the conservative interiority guard skips), 2 lightweight variants\n(448 — different base structure), and the scoop+label+lip permutation\n(132 — separate feature roots).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes planar face splitting to preserve closed-circle hole caps in\ncoincident-plane booleans, preventing free edges and mesh fallbacks.\nGridfinity screw-base exports: 9/13 now pass; 4×4 stress drops boundary\nedges from 1792 to 3.\n\n- **Bug Fixes**\n- Salvage interior closed `Circle` sections on plane faces: peel caps,\nsplit by open sections, then carve via `split_face_with_internal_loops`.\n- Prevent zero-UV-chord circles from being dropped by salvaging them\nbefore `arrangement_regions_from_inputs`.\n- Keep the single-circle fast path; non-planar and cap-free faces\nunchanged.\n- Add regression `fuse_capping_slab_preserves_drilled_hole_caps` to\nassert watertight fuse and cylinder preservation.\n\n- **Refactors**\n- Build containment polygon from traversal-ordered wire points for\ncorrect interiority on reversed edges; keep `PlaneFrame` convention\nunchanged.\n- Add `CAP_INTERIORITY_MARGIN` (5%) and single-pass section partitioning\nwith stored cap centers for robustness.\n\n<sup>Written for commit bfe7c188aeae5d1eca7ba9c28762b700a5bea9af.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1119?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-18T17:52:36-07:00",
          "tree_id": "9b85131e35ea81fc754456e455b5385fd6086a6b",
          "url": "https://github.com/andymai/brepkit/commit/79a82f05041fa0d5f306cf9875badebd5e6ee791"
        },
        "date": 1784422518860,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 890620,
            "range": "± 2340",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 980000,
            "range": "± 33082",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12223,
            "range": "± 355",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 657555,
            "range": "± 1803",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 27161818,
            "range": "± 109923",
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
          "id": "cdba97395689ad7ad80f334d71f8c4272f1c05a1",
          "message": "chore(main): release 2.126.20 (#1120)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.20](https://github.com/andymai/brepkit/compare/v2.126.19...v2.126.20)\n(2026-07-19)\n\n\n### Bug Fixes\n\n* **algo:** salvage closed-circle cap sections in the planar face\nsplitter ([#1119](https://github.com/andymai/brepkit/issues/1119))\n([79a82f0](https://github.com/andymai/brepkit/commit/79a82f05041fa0d5f306cf9875badebd5e6ee791))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.20 for `brepkit-wasm` with a fix to the planar face\nsplitter to salvage closed-circle cap sections. This prevents lost cap\ngeometry and improves robustness on circular faces.\n\n<sup>Written for commit 7788cd87310efa78eb0db630ccc882e993974ac2.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1120?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-19T00:59:50Z",
          "tree_id": "fb0422cb302620c0c4d0ce54e84b1b690054be47",
          "url": "https://github.com/andymai/brepkit/commit/cdba97395689ad7ad80f334d71f8c4272f1c05a1"
        },
        "date": 1784422938130,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 910915,
            "range": "± 2067",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1011225,
            "range": "± 2095",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13173,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 650630,
            "range": "± 1909",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28317243,
            "range": "± 92077",
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
          "id": "2599a0d4e68fa86d2792f208d14a32659d16a171",
          "message": "fix(algo): drop cap circles emerging inside holes; gate salvage early (#1121)\n\nFollow-up review pass on the cap-circle salvage (#1119).\n\n## Fixes\n\n- **In-hole cap guard**: a closed-circle rim whose centre lands inside\nan inner wire of the receiving plane is air (a drill emerging inside an\nexisting opening, e.g. a counterbore). The salvage peeled it off before\nthe impl's air filter could drop it, then carved it as a free-floating\ndisc — strictly worse than baseline. `distribute_cap_circles` now skips\ncaps whose centre lies in a sub-face hole, matching the impl's baseline\ndrop.\n- **Early gate**: return to the impl before any frame/polygon\nconstruction when no section is a closed circle — the common path for\nevery boolean face split.\n- **Frame-based containment sampling**: the distribution polygon now\nuses `sample_wire_loop_uv_via_frame`; the stored-pcurve sampler can fold\nloops carrying reversed arc sections into self-crossing polygons.\n- **Doc corrections**: only the single-closed fast path carves circles\n(the all-Line loop path never can); the interiority margin is about the\n*centre* clearing the outline, not the rim; removed the \"lands in\nexactly one sub-face\" over-claim and before/after narration.\n\n## Tests\n\n- `distribute_cap_circles_drops_caps_inside_holes` (algo unit test) —\nfails without the guard.\n- `fuse_embedded_drilled_block_carves_caps_across_split_sub_faces` —\ncaps mixed with open sections; exercises multi-sub-face distribution,\nwhich the original regression test never reached.\n- `fuse_counterbore_drops_drill_rims_inside_opening` — asserts the\nsolid-level contract (watertight, volume, classification). Documents two\npre-existing issues, out of scope here: this interface fails GFA\npost-assembly validation and mesh-falls-back, and the fallback mesh\nmisclassifies points directly under the bore as Outside.\n\nFull `brepkit-algo` + `brepkit-operations` suites green; clippy\n`--all-targets` warning-free.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFix cap-circle salvage on plane faces: drop caps whose centers fall\ninside holes and early-exit when no closed circles are present. This\nprevents free-floating discs, keeps results watertight, and maintains\nanalytic outputs.\n\n- **Bug Fixes**\n- Drop in-hole caps in `distribute_cap_circles` (centers inside sub-face\nholes are air).\n- Early gate to the impl when no section is a closed circle to avoid\nextra work.\n- Use frame-based containment sampling via\n`sample_wire_loop_uv_via_frame` to handle reversed arcs reliably.\n- Clarified salvage docs (which paths carve circles and the correct\ninteriority margin).\n\n- **Tests**\n- `distribute_cap_circles_drops_caps_inside_holes` — unit test for the\nin-hole cap guard.\n- `fuse_embedded_drilled_block_carves_caps_across_split_sub_faces` —\nmixed open/closed sections; verifies caps are carved into the correct\nsub-faces and watertightness.\n- `fuse_counterbore_drops_drill_rims_inside_opening` — ensures in-hole\nrims are dropped; asserts watertightness, volume, and classification.\n\n<sup>Written for commit 02ac398d2557ff6d74314eb87187b17b419b36ba.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1121?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-19T07:40:52-07:00",
          "tree_id": "95f96bb705edbe32ce2d8abe832aded7078f0dd1",
          "url": "https://github.com/andymai/brepkit/commit/2599a0d4e68fa86d2792f208d14a32659d16a171"
        },
        "date": 1784472199487,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 878862,
            "range": "± 36811",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 966943,
            "range": "± 29441",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12074,
            "range": "± 139",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 649840,
            "range": "± 486",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26807131,
            "range": "± 39133",
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
          "id": "b4b2b6daa55569ea8a38fc36e25dcee432c69157",
          "message": "chore(main): release 2.126.21 (#1122)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.21](https://github.com/andymai/brepkit/compare/v2.126.20...v2.126.21)\n(2026-07-19)\n\n\n### Bug Fixes\n\n* **algo:** drop cap circles emerging inside holes; gate salvage early\n([#1121](https://github.com/andymai/brepkit/issues/1121))\n([2599a0d](https://github.com/andymai/brepkit/commit/2599a0d4e68fa86d2792f208d14a32659d16a171))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.126.21 with a geometry fix that stops cap\ncircles from appearing inside holes and gates salvage earlier to avoid\nbad patches. This improves stability of boolean/repair operations and\nupdates version metadata.\n\n- **Bug Fixes**\n- Algorithm: block cap circles in hole regions; gate salvage early to\nprevent invalid surfaces.\n\n<sup>Written for commit 9f0493c3a3fca43cacbc0875f9951db82d82efd7.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1122?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-19T14:48:15Z",
          "tree_id": "751dcddee0bec049f81a74061ba1ea172fe80087",
          "url": "https://github.com/andymai/brepkit/commit/b4b2b6daa55569ea8a38fc36e25dcee432c69157"
        },
        "date": 1784472617228,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 704943,
            "range": "± 1312",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 784015,
            "range": "± 1135",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10177,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 508595,
            "range": "± 6078",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 21987131,
            "range": "± 98448",
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
          "id": "6dc03885deb64240bb0ae2ead1c2146aa8c5228b",
          "message": "fix(algo): anchor closed-rim splits at the edge's own start angle (#1123)\n\n## Summary\nFixes a **double-cover bug in closed-rim edge splitting**: a closed\ncircle's pieces came back sweeping **4π instead of 2π**. Found while\ndigging the gridfinity lightweight family; the fix is general, not\nlightweight-specific.\n\n## Root cause\nA CLOSED circle has no span between its endpoints, so\n`domain_with_endpoints` returns the circle's **intrinsic** domain `(0,\n2π)` — anchored at the circle's angular origin, not at the edge's start\npoint. `find_splits_on_circle`'s consumer chains pieces from `start_3d`\nin ascending `t`, so whenever those anchors differ (a cylinder seam away\nfrom the origin) **the edge's own start point reads as an interior\nsplit**. Instrumented output for a pad rim seamed at u=4.71239:\n\n```\nt0=0.00000 span=6.28319\n  t=0.250000 u=1.57080 ... t=0.750000 u=4.71239  <-- the edge's OWN start\n```\n\nThe 2π ring came back as **7 arcs sweeping 4π**, with the wedge between\norigin and seam covered twice — once inside the leading arc, again as a\ntrailing forward/reverse pair.\n\n## Fix (all gated to closed circles on a cylinder/cone)\n1. Anchor `t` at the edge's **start angle**, signed by traversal\ndirection, so `t` is monotone along the walk.\n2. Interpolate interior split UVs within the edge's own span (`start_u →\nstart_u ± 2π`) instead of a raw principal-value projection that drops\njoints back into `[0, 2π)` and loses phase coherence with neighbouring\nboundary edges.\n3. Derive the tail piece's `end_uv` from that span — `sample_edge_to_uv`\nignores orientation, so a reverse-traversed ring otherwise closes its\nperiod on the wrong side of its own start.\n\n## Verification\n- Regression: two unit tests pin the invariant directly — a closed rim's\npieces must sweep **exactly 2π**, forward and reversed. Verified failing\nwithout the fix (`got 12.566…` = 4π).\n- Lightweight repro: raw-GFA free edges **15 → 8**, and a pad-wall\nsub-face is recovered (cylinder 48 → 49), analytic preserved.\n- Foils: `algo` **169/0**, `operations` **769/0**, `wasm gridfinity`\n**27/0** (d4 canary), `io` clean, clippy `-D warnings` clean.\n\n## Scope — what this does NOT do\nIt does **not** close the lightweight export family. A separate,\nindependently-verified root remains: a **missing FF section** — a\nfloor-plane arc at z=−3.8 over `u ∈ (3.7518, 4.10219)` that should exist\n(confirmed by material flips in point classification) but is never\ngenerated, so the fuse still falls back. That is tracked as the terminal\nroot for that family; this PR lands the closed-rim invariant fix on its\nown merits.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes double-coverage when splitting closed circular rims on\ncylinders/cones. Pieces now tile the ring once (2π) from the edge’s own\nstart angle, matching traversal.\n\n- Anchor split parameter to the edge start angle (signed by traversal).\n- Interpolate interior split UVs within `start_u → start_u ± 2π`; set\nthe tail piece `end_uv` from that span so reversed rims close on the\ncorrect side.\n- Gate both anchoring and UV-span logic to closed circle edges on\n`Cylinder`/`Cone` surfaces.\n- Tests assert an exact 2π sweep (forward and reversed) and now check\nforward UV continuity (shared joints, monotone u).\n- In the gridfinity lightweight model, reduces raw-GFA free edges and\nrestores a pad-wall sub-face.\n\n<sup>Written for commit 56585280c797fe3f8b212a1078be90e77515e3fa.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1123?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-19T08:36:24-07:00",
          "tree_id": "e83e12bdd0403675c6d4c017200c25c0a2f53de8",
          "url": "https://github.com/andymai/brepkit/commit/6dc03885deb64240bb0ae2ead1c2146aa8c5228b"
        },
        "date": 1784475525283,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 882803,
            "range": "± 14827",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 971718,
            "range": "± 2023",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11949,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 650727,
            "range": "± 12166",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26767977,
            "range": "± 43977",
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
          "id": "845ee2c7798ba9108c346f3676c7b41a594b79b1",
          "message": "chore(main): release 2.126.22 (#1124)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.126.22](https://github.com/andymai/brepkit/compare/v2.126.21...v2.126.22)\n(2026-07-19)\n\n\n### Bug Fixes\n\n* **algo:** anchor closed-rim splits at the edge's own start angle\n([#1123](https://github.com/andymai/brepkit/issues/1123))\n([6dc0388](https://github.com/andymai/brepkit/commit/6dc03885deb64240bb0ae2ead1c2146aa8c5228b))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.126.22 for `brepkit-wasm`, fixing closed‑rim split anchoring\nby using the edge’s own start angle. This prevents misaligned splits and\nimproves geometry robustness.\n\n<sup>Written for commit 7566ea32a8d2f71b6d540a8bff19fa95b67f1365.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1124?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-19T15:43:57Z",
          "tree_id": "ce68ba22d3e67a25c420f928ae77dd7dfb7bd11f",
          "url": "https://github.com/andymai/brepkit/commit/845ee2c7798ba9108c346f3676c7b41a594b79b1"
        },
        "date": 1784475985362,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 911648,
            "range": "± 16683",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1011775,
            "range": "± 1135",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13076,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 649620,
            "range": "± 741",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28224512,
            "range": "± 72740",
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
          "id": "8dd92c47453437f700bebd8bad7e852550719d9f",
          "message": "feat(math): solve parallel-axis cone × cylinder in closed form (#1125)\n\n## Summary\nCone × cylinder had **no algebraic arm for the parallel-axis case**, so\nit fell through to the grid-seeded marcher — which returned **49\noverlapping partial traces of what is a single curve**. This replaces\nthat with an exact closed-form solution (2 branches).\n\n## Root cause\nThe marcher's seed threshold is half the partner's diagonal, and the\nmarch-result dedup only consumes seeds the resulting polyline passes\nnear. On a near-tangent parallel-axis configuration, many seeds each\nspawn their own fragment and none subsumes the others — hence 49\nfragments for one curve.\n\n## Fix\nWith parallel axes the axis separation `d` is constant, so at cone\nradius `ρ` the intersection satisfies\n\n```\nu = φ₀ ± acos((d² + ρ² − R²) / (2·d·ρ))\n```\n\ngiving two exact branches parameterised by the cone's own `v`. Emitted\ndirectly instead of marching — exact, and cheaper than seeding +\nmarching.\n\n## Verification\n- Regression: three tests asserting exactly the two branches are\nreturned. Verified failing without the fix: `expected exactly the two\nbranches / left: 49 / right: 2`.\n- Foils: `math` **459/0**, `algo` **169/0**, `operations` **769/0**,\n`wasm gridfinity` **27/0** (d4 canary), `io` clean, clippy `-D warnings`\nclean.\n\n## Scope\nFound while digging the gridfinity **lightweight** family, where a\nmagnet pad straddles a tapered socket foot and the 49 fragments starve\nthat fuse of a usable section chain. **This alone does not change that\nresult** — later links in that chain (FF graze-refinement density,\nopen-section trimming to curved faces, boundary-split gating, and an\noperations Euler gate) are still open and are deliberately not in this\nPR. It lands as a standalone correctness/robustness fix for the whole\nparallel-axis cone×cylinder class.",
          "timestamp": "2026-07-19T10:50:30-07:00",
          "tree_id": "8a3939387f966783eea8008a81ba533c887c5b9d",
          "url": "https://github.com/andymai/brepkit/commit/8dd92c47453437f700bebd8bad7e852550719d9f"
        },
        "date": 1784483574945,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 872310,
            "range": "± 2857",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 964890,
            "range": "± 2991",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12081,
            "range": "± 199",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 642658,
            "range": "± 8158",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26499434,
            "range": "± 57638",
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
          "id": "bf6129c01f1c0d6de9567485192803f1954ae774",
          "message": "chore(main): release 2.127.0 (#1126)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.0](https://github.com/andymai/brepkit/compare/v2.126.22...v2.127.0)\n(2026-07-19)\n\n\n### Features\n\n* **math:** solve parallel-axis cone × cylinder in closed form\n([#1125](https://github.com/andymai/brepkit/issues/1125))\n([8dd92c4](https://github.com/andymai/brepkit/commit/8dd92c47453437f700bebd8bad7e852550719d9f))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.127.0 for `brepkit-wasm`. Adds a closed-form solver for the\nparallel-axis cone × cylinder case in the math module.\n\n<sup>Written for commit 3d30c7d8fe093d15c71eadd1cbdbd375383ab122.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1126?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-19T17:57:46Z",
          "tree_id": "4bdfbc41bf75ff778ac9f662d2d6d9488aa78252",
          "url": "https://github.com/andymai/brepkit/commit/bf6129c01f1c0d6de9567485192803f1954ae774"
        },
        "date": 1784484012272,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 875274,
            "range": "± 1937",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 964628,
            "range": "± 2245",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12166,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 645852,
            "range": "± 3204",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26372450,
            "range": "± 54323",
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
          "id": "3f8d1c2d7fee700181a678010e5de68c9cb067c1",
          "message": "fix(operations): apply the hole correction in the multi-region boolean gate (#1127)\n\n## The defect\n\nN closed manifolds satisfy the Euler-Poincaré relation\n\n```\nV - E + F - inner_wires = 2 * (N - genus)\n```\n\nThe multi-region acceptance path in `boolean/mod.rs` compared **raw**\nEuler against `2 * components`, omitting the `inner_wires` term that the\nsingle-component gate immediately above it already applies via\n`euler_balanced(euler_eff, inner_wire_count)`.\n\nA face carrying a hole shifts raw Euler away from `2 * N` **even at\ngenus 0**, so the multi-region path only ever accepted **hole-free**\npieces. Any multi-component result whose pieces have pockets or bores\nwas rejected and silently replaced by a mesh boolean.\n\n## Why it went unnoticed\n\nThe fallback masks itself. It is watertight, passes\n`validate_boolean_result`, and has the correct volume — every downstream\ncheck agrees with it. Only the face census distinguishes the two,\nexactly as the `boolean-debugging` skill warns.\n\nSevering a pocketed bar in two — an ordinary CAD operation — is enough\nto trigger it:\n\n| | faces | census | volume |\n|---|---|---|---|\n| before | 114 | `{plane: 114}` | 302.80 |\n| after | 16 | `{cylinder: 2, plane: 14}` | 302.80 |\n\nAnalytic value is 302.80 (`20×6×3` − two `π·1.5²·1.5` pockets − a\n`2×6×3` slab), so both results are *geometrically* right; the old path\njust threw away the analytic one. The arithmetic at the gate:\n\n```\neuler=6  inner_wires=2  components=2\n  euler - inner_wires = 4  vs  2*components = 4   <- correct relation, holds\n  raw euler           = 6  vs  2*components = 4   <- what was compared, fails\n```\n\nIt needed three properties at once to surface — multi-component **and**\ngenus-0 **and** faces-with-holes — which is why it survived: a severed\n*un*-pocketed cylinder has no holes and passes fine.\n\n## The fix\n\nCompare `euler - inner_wire_count` against `2 * components`.\n\nThe disjointness, closed-manifold, cut-safety, and validation guards are\nuntouched, so the gate is no more permissive about malformed output — it\njust stops rejecting a shape class it was never meant to exclude. Worth\nnoting the hollow case cannot slip through the relaxed check: a cavity\ncomponent's AABB lies strictly inside its containing piece's, so\n`components_are_disjoint_pieces` rejects it on AABB overlap.\n\n## Verification\n\n- New regression `severing_cut_keeps_pocketed_pieces_analytic` **fails\nwithout the change** (114 planar faces, 0 cylinders) and passes with it.\nAsserts compact face count, both pocket walls analytic, 0 non-manifold\nedges, and volume against the exact analytic value.\n- Full workspace suite green (77 test binaries).\n- Gridfinity canary 27/0.\n- `brepkit-render --test compute_mesh_render` SIGSEGVs, but it does so\nidentically on clean `main` — pre-existing GPU/driver issue, unrelated.\n\n## Note on scope: extending to Fuse was tested and rejected\n\nThe multi-region path is still gated `op == BooleanOp::Cut`. I built\nwhat looked like a justification for widening it — fusing a lug onto one\npiece of an already-severed pocketed bar, which mesh-falls-back with 120\nplanar faces — then instrumented the gate before changing it. The\nwidening gains nothing:\n\n```\nop=Fuse comps=2 euler=6 iw=3 corrected=3 expected=4 closed_manifold=FALSE\n```\n\nA corrected Euler of **3 is odd**, which no set of closed manifolds can\nproduce, and `closed_manifold` is genuinely false. That GFA output (19\nfaces) is really broken, so the fallback is the *correct* behaviour\nthere; the actual defect is the pavefiller's single-connected-input\nassumption — the same root `cut_multi_region_input` works around for\nCut. So the Fuse extension is deliberately not included: it would have\nwidened a gate on a false premise.\n\nFound while digging the lightweight-base blocker. It does **not** close\nthat case — that one is a Fuse and is separately blocked by\n`closed_manifold=false` — but it is a real independent defect with a\nminimal repro, so it ships on its own.",
          "timestamp": "2026-07-19T14:14:53-07:00",
          "tree_id": "8ef2548dd539163f9cd67226eb79ff34780b48c5",
          "url": "https://github.com/andymai/brepkit/commit/3f8d1c2d7fee700181a678010e5de68c9cb067c1"
        },
        "date": 1784495837987,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 880035,
            "range": "± 853",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 968911,
            "range": "± 2620",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12036,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 645637,
            "range": "± 2555",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26462537,
            "range": "± 44724",
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
          "id": "76ff95e2abe2578a72345e2ae86bc8befe07ca2c",
          "message": "chore(main): release 2.127.1 (#1128)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.1](https://github.com/andymai/brepkit/compare/v2.127.0...v2.127.1)\n(2026-07-19)\n\n\n### Bug Fixes\n\n* **operations:** apply the hole correction in the multi-region boolean\ngate ([#1127](https://github.com/andymai/brepkit/issues/1127))\n([3f8d1c2](https://github.com/andymai/brepkit/commit/3f8d1c2d7fee700181a678010e5de68c9cb067c1))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nPatch release 2.127.1 fixes boolean operations for multi‑region shapes\nby applying hole correction, improving accuracy and preventing missing\ninteriors.\n\n- **Bug Fixes**\n- Apply hole correction in the multi-region boolean gate to handle\nholes/inner loops correctly during boolean ops.\n\n<sup>Written for commit db6dc8b81d1f3cc3dc9427a83fbba89c05c588c1.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1128?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-19T21:22:30Z",
          "tree_id": "0d2498311130077dcf56ba08a6a6cdaee881ec6d",
          "url": "https://github.com/andymai/brepkit/commit/76ff95e2abe2578a72345e2ae86bc8befe07ca2c"
        },
        "date": 1784496297377,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 871682,
            "range": "± 2704",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 960159,
            "range": "± 5345",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11794,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 640817,
            "range": "± 4156",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26403820,
            "range": "± 44295",
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
          "id": "d9a0400b8101152ce1fa8bc21f60f1691bcb6d0f",
          "message": "fix(heal): stop unify_same_domain discarding closed-curve boundary loops (#1129)\n\n## The defect\n\nA full circle is stored as a **closed edge** — `start == end` with a\nzero chord. `merge_group_with_holes` filtered edges with:\n\n```rust\nif (ep - sp).length() < lin && start == end { continue; }\n```\n\nwhich cannot tell a real circle from a genuinely degenerate zero-length\nsliver, so it discarded both. Discarding a real circle **erases a hole\nboundary**: the drilled annulus merges into its coplanar neighbour and\nthe bore rim is left used once (a free edge).\n\nThis is exactly the closed-edge trap CLAUDE.md documents — *\"whenever an\nedge can be closed, sample points along the curve; never derive from\nendpoints.\"*\n\n## The fix\n\nSample the curve interior to separate a real closed curve from a sliver,\nand **defer** such a group (leave it unmerged) rather than merge it with\na boundary silently dropped.\n\nDeferral rather than repair is deliberate: the reassembly classifies\nloops by UV polygon area, and a closed circle arrives as a\n**single-vertex loop** that yields no polygon, so the group cannot be\nreassembled correctly either way. Skipping matches the function's\nexisting contract — it already returns `Ok(None)` for cases it cannot\nrepresent (periodic surfaces with holes, non-manifold edges, unclosed\nloops).\n\nThis only makes the merge **more conservative**: groups it previously\nmerged wrongly are now skipped, and no group it previously skipped is\nnow merged. The cost is a few unmerged coplanar faces; the alternative\nis a hole in the solid.\n\n## How it was found\n\nThe gridfinity `2×2 lite + screw pads` export fails with **448 STL\nboundary edges**. The chain:\n\n`cutAll` → `compound_cut`, whose trailing `unify_same_domain` merged\neach socket cell's 5 coplanar bottom faces (1 floor + 4 drilled pad\nannuli) into 1, dropping 16 planes and orphaning all 16 bore rims.\n\nReplaying on captured tool operands isolated it to a single flag:\n\n| | F | E | V | free edges | planes |\n|---|---|---|---|---|---|\n| tool's `cutAll` output | 284 | 608 | 352 | **16** | 140 |\n| `compound_cut` unify **on** | 284 | 608 | 352 | **16** | 140 |\n| `compound_cut` unify **off** | 300 | 624 | 368 | 0 | 156 |\n| **with this fix**, unify on | 300 | 624 | 368 | **0** | 156 |\n\nThe unify-on replay reproduced the tool's result *exactly*, and the fix\nmakes it match the clean one. Worth noting the 16 sequential cuts and a\nsingle cut by a 16-component fused tool were both already clean — the\ndrill and phase FF were never at fault.\n\n## Verification\n\n- New regression `compound_cut_unify_keeps_bore_opening` — plate + pad\ncolumn fused (leaving a coplanar bottom disc), bored through. **Fails\nwithout the change** (1 unpaired edge), passes with it.\n- Full workspace suite green; `brepkit-heal` 86/0.\n- Gridfinity canary 27/0.\n- `brepkit-render` GPU tests SIGSEGV identically on clean `main` —\npre-existing, unrelated.\n\n## Scope\n\nThis closes the B-Rep root behind the screw-pad family. It does **not**\nby itself make that scenario pass end-to-end — I have not yet\nre-measured the tool suite with it, so I am claiming the root fix, not\nthe scenario. The lightweight failures are three unrelated roots (this\none, a sub-tolerance `nm=2` on the magnet family that a STEP round-trip\nheals, and two pure 30s timeouts).\n\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nStops unify_same_domain from dropping closed-circle boundary loops that\nerased hole openings and created free edges. We now sample the curve\ninterior to tell real circles from zero-length slivers and defer those\ngroups instead of merging.\n\n- Bug Fixes\n- Sample closed edges at mid-parameter; if it’s a real circle, skip\nmerging the whole group to preserve the boundary.\n- Prevents bore rims from becoming free edges when unifying coplanar\nfaces.\n  - Adds regression test `compound_cut_unify_keeps_bore_opening`.\n- Behavior is more conservative: previously wrong merges are now\nskipped; previously skipped cases remain skipped.\n\n<sup>Written for commit 737d1049d28cbdf3cd452d25c25e142a64de900d.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1129?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-19T19:21:46-07:00",
          "tree_id": "045c75c600dddfe181dec22d32f90b2aaa035ea2",
          "url": "https://github.com/andymai/brepkit/commit/d9a0400b8101152ce1fa8bc21f60f1691bcb6d0f"
        },
        "date": 1784514250939,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 876569,
            "range": "± 2973",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 970072,
            "range": "± 5247",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11919,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 642957,
            "range": "± 9633",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26700327,
            "range": "± 68664",
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
          "id": "fe6899a298fd6399cc4d1896353d6975d90e3df0",
          "message": "chore(main): release 2.127.2 (#1130)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.2](https://github.com/andymai/brepkit/compare/v2.127.1...v2.127.2)\n(2026-07-20)\n\n\n### Bug Fixes\n\n* **heal:** stop unify_same_domain discarding closed-curve boundary\nloops ([#1129](https://github.com/andymai/brepkit/issues/1129))\n([d9a0400](https://github.com/andymai/brepkit/commit/d9a0400b8101152ce1fa8bc21f60f1691bcb6d0f))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.127.2 for `brepkit-wasm`. Prevents boundary loss in models\ncaused by the healing step.\n\n- **Bug Fixes**\n- Stop `heal::unify_same_domain` from discarding closed-curve boundary\nloops (#1129).\n\n<sup>Written for commit b3118707bb6abc54bdae9f8255a8f63d80d92e71.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1130?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T02:29:01Z",
          "tree_id": "c26e272581c392da03a2d4d5c111709809e40aa3",
          "url": "https://github.com/andymai/brepkit/commit/fe6899a298fd6399cc4d1896353d6975d90e3df0"
        },
        "date": 1784514692836,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 800533,
            "range": "± 10238",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 883229,
            "range": "± 10567",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11048,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 589272,
            "range": "± 5492",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 25249957,
            "range": "± 258690",
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
          "id": "dd47ed29ffa2b520de99981de0223e35198a3226",
          "message": "fix(heal): revert a unify pass that would orphan edges (#1131)\n\n## The defect\n\n`unify_same_domain` could turn a watertight shell into a holed one.\nMerging is an **optimisation** — it must never leave the shell worse\nconnected than it found it.\n\nTwo phases can orphan edges:\n\n1. a group whose reassembly loses a boundary edge, and\n2. **`merge_collinear_edges`**, which rewrites only the NEW faces' outer\nwires. A neighbouring face still referencing the pre-merge edges is left\nunpaired.\n\nThe second is a one-sided topology edit and is the larger defect. This\nPR **bounds** its damage rather than repairing it — a two-sided rewrite\nis still worth doing, and I did not want to attempt that reshaping on\nthe back of a guard.\n\n## Why it went unnoticed\n\nIt is invisible on analytic solids, where collinear runs are rare, and\nsevere on mesh-fallback ones, where thousands of coplanar facets merge\nat once.\n\nFound digging the gridfinity `4×4 lite export (stress)` failure (48 STL\nboundary edges). Its base mesh-falls-back to a 15648-face solid, and a\n**single** drill through `compound_cut` merged 3184 faces and 1630\ncollinear edges — orphaning 4574:\n\n| | faces | free edges |\n|---|---|---|\n| `boolean(Cut)` 1 drill (no unify) | 15650 | 0 |\n| `compound_cut` unify=**false** | 15650 | 0 |\n| `compound_cut` unify=**true** — before | 12466 | **4574** |\n| `compound_cut` unify=**true** — after | 15650 | **0** |\n\nThe guard logs its reason: `reverting (3184 faces, 1630 edges) —\nunpaired edges would rise 0 -> 4574`.\n\n## Why the revert is complete\n\nRemoved faces are only dropped from the shell's face list, never deleted\nfrom the arena, and the collinear pass only mutates the **new** faces'\nwires — which the revert discards. So restoring the original face list\nfully restores the prior topology.\n\n## Verification\n\n- `compound_cut_unify_still_merges_normally` — new; pins that the guard\ndoes **not** over-trigger: a slotted bar's coplanar fragments must still\nmerge back (≤20 faces). Without it, a future change could make the guard\nfire everywhere and silently disable unify.\n- `compound_cut_unify_keeps_bore_opening` (from the closed-curve fix,\n#1129) continues to pass.\n- Full workspace suite green; `brepkit-heal` 86/0; gridfinity canary\n27/0.\n- `brepkit-render` GPU tests SIGSEGV identically on clean `main` —\npre-existing, unrelated.\n\n## Honest gap\n\n**No compact synthetic reproduces the collinear-edge path yet.** I tried\nsix shapes (slotted bars, coplanar pocket grids, stacked fuses,\nthrough-slot grids, a mesh-fallback cylinder∪cone, box mesh booleans);\nnone trigger it — adjacent facets on curved surfaces are not coplanar so\nnothing merges, and coplanar box facets get merged during mesh→B-Rep\nconversion. The reproducer is the captured 15648-face solid, too large\nto commit. The new test therefore pins the guard's *other* half (no\nover-triggering); the orphaning path itself is verified only against the\ncaptured operands, as recorded above.\n\nThis is a bound on damage, not a scenario fix — `4×4 lite export\n(stress)` also needs the pad fuse to stop falling back, which is\nseparate (and previously measured as expensive).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nAdd a connectivity guard to the unify pass and roll back each phase\nindependently to prevent orphaned edges. This keeps watertight shells\nintact on mesh-fallback solids while still merging valid coplanar faces.\n\n- **Bug Fixes**\n- Measure edge pairing before/after face merges and again after\ncollinear-edge merges; revert only the phase that increases unpaired\nedges. Snapshot new-face wires to undo collinear merges while keeping\nvalid face merges; warn with before/after counts.\n- Replaced the slotted-bar check with a differential L-shaped fuse test:\nunify reduces face count vs no-unify and the result stays watertight.\n\n<sup>Written for commit 29af4141028d933524c04f92884326d7aae372a5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1131?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-19T22:09:38-07:00",
          "tree_id": "3d7006b376ae8d7b17d5a5323b71f4727e83d713",
          "url": "https://github.com/andymai/brepkit/commit/dd47ed29ffa2b520de99981de0223e35198a3226"
        },
        "date": 1784524332798,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 884222,
            "range": "± 2626",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 974731,
            "range": "± 1625",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 12135,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 646627,
            "range": "± 2776",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26707494,
            "range": "± 73766",
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
          "id": "7952188e3457c521fb9a4998f1ad365dea634528",
          "message": "chore(main): release 2.127.3 (#1132)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.3](https://github.com/andymai/brepkit/compare/v2.127.2...v2.127.3)\n(2026-07-20)\n\n\n### Bug Fixes\n\n* **heal:** revert a unify pass that would orphan edges\n([#1131](https://github.com/andymai/brepkit/issues/1131))\n([dd47ed2](https://github.com/andymai/brepkit/commit/dd47ed29ffa2b520de99981de0223e35198a3226))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease `brepkit-wasm` 2.127.3 with a heal fix that reverts a unify pass\nto prevent orphaned edges and preserve topology integrity.\n\n<sup>Written for commit 15425aa23e862443c64cffc1956913bf3a8974d9.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1132?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T05:16:52Z",
          "tree_id": "a023d7bdc20ffbcd9bc91a0774831354a2f8f174",
          "url": "https://github.com/andymai/brepkit/commit/7952188e3457c521fb9a4998f1ad365dea634528"
        },
        "date": 1784524762336,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 882001,
            "range": "± 2257",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 970013,
            "range": "± 2207",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11918,
            "range": "± 174",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 647644,
            "range": "± 2905",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26742282,
            "range": "± 57374",
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
          "id": "8e48139fcc56c0be32a36e5d4c653f1a4194318f",
          "message": "chore(deps): bump the actions group with 3 updates (#1135)\n\nBumps the actions group with 3 updates:\n[taiki-e/install-action](https://github.com/taiki-e/install-action),\n[EmbarkStudios/cargo-deny-action](https://github.com/embarkstudios/cargo-deny-action)\nand [actions/setup-node](https://github.com/actions/setup-node).\n\nUpdates `taiki-e/install-action` from 2.82.9 to 2.83.2\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/releases\">taiki-e/install-action's\nreleases</a>.</em></p>\n<blockquote>\n<h2>2.83.2</h2>\n<ul>\n<li>\n<p>Update <code>parse-dockerfile@latest</code> to 0.1.8.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.5.</p>\n</li>\n<li>\n<p>Update <code>just@latest</code> to 1.56.0.</p>\n</li>\n<li>\n<p>Update <code>gungraun-runner@latest</code> to 0.19.4.</p>\n</li>\n<li>\n<p>Update <code>cargo-neat@latest</code> to 0.4.1.</p>\n</li>\n</ul>\n<h2>2.83.1</h2>\n<ul>\n<li>\n<p>Update <code>rclone@latest</code> to 1.74.4.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.4.</p>\n</li>\n<li>\n<p>Update <code>cargo-deny@latest</code> to 0.20.2.</p>\n</li>\n</ul>\n<h2>2.83.0</h2>\n<ul>\n<li>\n<p>Support <code>cargo-about</code>. (<a\nhref=\"https://redirect.github.com/taiki-e/install-action/pull/1924\">#1924</a>,\nthanks <a\nhref=\"https://github.com/ruffsl\"><code>@​ruffsl</code></a>)</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.28.</p>\n</li>\n<li>\n<p>Update <code>martin@latest</code> to 1.12.0.</p>\n</li>\n<li>\n<p>Update <code>kingfisher@latest</code> to 1.106.0.</p>\n</li>\n<li>\n<p>Update <code>biome@latest</code> to 2.5.3.</p>\n</li>\n</ul>\n<h2>2.82.11</h2>\n<ul>\n<li>\n<p>Update <code>wasm-tools@latest</code> to 1.253.0.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.27.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.2.</p>\n</li>\n<li>\n<p>Update <code>mdbook@latest</code> to 0.5.4.</p>\n</li>\n</ul>\n<h2>2.82.10</h2>\n<ul>\n<li>\n<p>Update <code>tombi@latest</code> to 1.2.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-nextest@latest</code> to 0.9.140.</p>\n</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md\">taiki-e/install-action's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>Changelog</h1>\n<p>All notable changes to this project will be documented in this\nfile.</p>\n<p>This project adheres to <a href=\"https://semver.org\">Semantic\nVersioning</a>.</p>\n<!-- raw HTML omitted -->\n<h2>[Unreleased]</h2>\n<ul>\n<li>\n<p>Update <code>just@latest</code> to 1.57.0.</p>\n</li>\n<li>\n<p>Update <code>cargo-semver-checks@latest</code> to 0.49.0.</p>\n</li>\n<li>\n<p>Update <code>tombi@latest</code> to 1.2.3.</p>\n</li>\n<li>\n<p>Update <code>cosign@latest</code> to 3.1.2.</p>\n</li>\n</ul>\n<h2>[2.83.4] - 2026-07-17</h2>\n<ul>\n<li>\n<p>Update <code>vacuum@latest</code> to 0.29.10.</p>\n</li>\n<li>\n<p>Update <code>uv@latest</code> to 0.11.29.</p>\n</li>\n<li>\n<p>Update <code>syft@latest</code> to 1.48.0.</p>\n</li>\n<li>\n<p>Update <code>prek@latest</code> to 0.4.10.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.7.</p>\n</li>\n<li>\n<p>Update <code>cargo-shear@latest</code> to 1.13.2.</p>\n</li>\n</ul>\n<h2>[2.83.3] - 2026-07-16</h2>\n<ul>\n<li>\n<p>Update <code>release-plz@latest</code> to 0.3.160.</p>\n</li>\n<li>\n<p>Update <code>prek@latest</code> to 0.4.9.</p>\n</li>\n<li>\n<p>Update <code>mise@latest</code> to 2026.7.6.</p>\n</li>\n<li>\n<p>Update <code>dprint@latest</code> to 0.55.2.</p>\n</li>\n<li>\n<p>Update <code>cargo-dinghy@latest</code> to 0.8.5.</p>\n</li>\n<li>\n<p>Update <code>cargo-binstall@latest</code> to 1.21.0.</p>\n</li>\n<li>\n<p>Update <code>biome@latest</code> to 2.5.4.</p>\n</li>\n</ul>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/43aecc8d72668fbcfe75c31400bc4f890f1c5853\"><code>43aecc8</code></a>\nRelease 2.83.2</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/fca47892c74f4dffa1e86ad93eca31cf898c2541\"><code>fca4789</code></a>\nUpdate prek manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/b41cc1f9ab348b9db341d8597853df93b08652f0\"><code>b41cc1f</code></a>\nUpdate <code>parse-dockerfile@latest</code> to 0.1.8</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/8d866f8ca86db77044d7d29639e24660d0b37b88\"><code>8d866f8</code></a>\nUpdate <code>mise@latest</code> to 2026.7.5</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/7ebe462223a33af951eed3c3ab1f754ddf2992e2\"><code>7ebe462</code></a>\nUpdate <code>just@latest</code> to 1.56.0</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/01ab5633b01b19988c158b5366d64947fd6cacce\"><code>01ab563</code></a>\nUpdate <code>gungraun-runner@latest</code> to 0.19.4</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/f164a682e7e0033cf72f1d5722d079fe82275c1e\"><code>f164a68</code></a>\nUpdate <code>cargo-neat@latest</code> to 0.4.1</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/2ca9b94c269419b7b0c711c09d0b21c4e1d51145\"><code>2ca9b94</code></a>\nRelease 2.83.1</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/8598f86981150fb7f6511798af658bbf80a8ea46\"><code>8598f86</code></a>\nUpdate parse-dockerfile manifest</li>\n<li><a\nhref=\"https://github.com/taiki-e/install-action/commit/76cfe4de2d710e12bae93580164f20dff9fd96e9\"><code>76cfe4d</code></a>\nUpdate <code>rclone@latest</code> to 1.74.4</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/taiki-e/install-action/compare/4684b8405694ae9dd42c9f39ba901a70ae83f4a3...43aecc8d72668fbcfe75c31400bc4f890f1c5853\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `EmbarkStudios/cargo-deny-action` from 2.0.20 to 2.1.1\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/EmbarkStudios/cargo-deny-action/commit/3c6349835b2b7b196a839186cb8b78e02f7b5f25\"><code>3c63498</code></a>\nFix use-git-cli deprecation (<a\nhref=\"https://redirect.github.com/embarkstudios/cargo-deny-action/issues/116\">#116</a>)</li>\n<li><a\nhref=\"https://github.com/EmbarkStudios/cargo-deny-action/commit/6f99e342a8f0f8f8d1bdc9dc43e9a6f2dd611259\"><code>6f99e34</code></a>\nBump to 0.20.2</li>\n<li><a\nhref=\"https://github.com/EmbarkStudios/cargo-deny-action/commit/8b229e2cbac05ffa3e4e6646023a0b4ee717c736\"><code>8b229e2</code></a>\nDeprecate use-git-cli</li>\n<li>See full diff in <a\nhref=\"https://github.com/embarkstudios/cargo-deny-action/compare/bb137d7af7e4fb67e5f82a49c4fce4fad40782fe...3c6349835b2b7b196a839186cb8b78e02f7b5f25\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `actions/setup-node` from 6.4.0 to 7.0.0\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/actions/setup-node/releases\">actions/setup-node's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v7.0.0</h2>\n<h2>What's Changed</h2>\n<h3>Enhancements:</h3>\n<ul>\n<li>Add cache-primary-key and cache-matched-key as outputs by <a\nhref=\"https://github.com/gowridurgad\"><code>@​gowridurgad</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1577\">actions/setup-node#1577</a></li>\n<li>Migrate to ESM and upgrade dependencies by <a\nhref=\"https://github.com/gowridurgad\"><code>@​gowridurgad</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1574\">actions/setup-node#1574</a></li>\n</ul>\n<h3>Bug fixes:</h3>\n<ul>\n<li>Remove dummy NODE_AUTH_TOKEN export by <a\nhref=\"https://github.com/gowridurgad\"><code>@​gowridurgad</code></a> in\n<a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1558\">actions/setup-node#1558</a></li>\n<li>Only use <code>mirrorToken</code> in <code>getManifest</code> if\nit's provided by <a\nhref=\"https://github.com/deiga\"><code>@​deiga</code></a> in <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1548\">actions/setup-node#1548</a></li>\n</ul>\n<h3>Documentation updates:</h3>\n<ul>\n<li>Add documentation for publishing to npm with Trusted Publisher\n(OIDC) by <a\nhref=\"https://github.com/chiranjib-swain\"><code>@​chiranjib-swain</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1536\">actions/setup-node#1536</a></li>\n<li>docs: Update restore-only cache documentation by <a\nhref=\"https://github.com/priya-kinthali\"><code>@​priya-kinthali</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1550\">actions/setup-node#1550</a></li>\n<li>docs: Update caching recommendations to mitigate cache poisoning\nrisks by <a\nhref=\"https://github.com/chiranjib-swain\"><code>@​chiranjib-swain</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1567\">actions/setup-node#1567</a></li>\n</ul>\n<h3>Dependency update:</h3>\n<ul>\n<li>Upgrade <code>@​actions/cache</code> to 5.1.0, log cache write\ndenied by <a\nhref=\"https://github.com/jasongin\"><code>@​jasongin</code></a> in <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1569\">actions/setup-node#1569</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a\nhref=\"https://github.com/chiranjib-swain\"><code>@​chiranjib-swain</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1536\">actions/setup-node#1536</a></li>\n<li><a href=\"https://github.com/deiga\"><code>@​deiga</code></a> made\ntheir first contribution in <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1548\">actions/setup-node#1548</a></li>\n<li><a href=\"https://github.com/jasongin\"><code>@​jasongin</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1569\">actions/setup-node#1569</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/actions/setup-node/compare/v6...v7.0.0\">https://github.com/actions/setup-node/compare/v6...v7.0.0</a></p>\n<h2>v6.5.0</h2>\n<h2>What's Changed</h2>\n<ul>\n<li>Update <code>@​actions/cache</code> to 5.1.0 and add security\noverrides for undici and fast-xml-parser by <a\nhref=\"https://github.com/HarithaVattikuti\"><code>@​HarithaVattikuti</code></a>\nin <a\nhref=\"https://redirect.github.com/actions/setup-node/pull/1579\">actions/setup-node#1579</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/actions/setup-node/compare/v6.4.0...v6.5.0\">https://github.com/actions/setup-node/compare/v6.4.0...v6.5.0</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/820762786026740c76f36085b0efc47a31fe5020\"><code>8207627</code></a>\nMigrate to ESM and upgrade dependencies (<a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1574\">#1574</a>)</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/04be95cf3511ea51ebf9f224ddfb99cc7ab87cd4\"><code>04be95c</code></a>\nAdd cache-primary-key and cache-matched-key as outputs (<a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1577\">#1577</a>)</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/7c2c68d20d402ed6a201ada70a81341941093140\"><code>7c2c68d</code></a>\ndocs: Update caching recommendations to mitigate cache poisoning risks\n(<a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1567\">#1567</a>)</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/6a61c0375d66246de94630495909f12cf8dac84d\"><code>6a61c03</code></a>\nMerge pull request <a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1569\">#1569</a>\nfrom jasongin/update-actions-cache-5.1.0</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/30eb73b41ded577900c1ebf968ef95cdf8f7434f\"><code>30eb73b</code></a>\nResolve high-severity audit issues</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/4e1a87a501d0302f99e30e2748568adcb388d09f\"><code>4e1a87a</code></a>\nUpdate dist</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/360237f0c01778d0c17291f75c56d6feae4f7574\"><code>360237f</code></a>\nStrict equality</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/4f8aac5beb2f0854bc79651567a18c67eb0b9de3\"><code>4f8aac5</code></a>\nBump <code>@​actions/cache</code> to 5.1.0, log cache write denied</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/f4a67bbeca970f103397d3d2b9462cf787cd2980\"><code>f4a67bb</code></a>\nOnly use <code>mirrorToken</code> in <code>getManifest</code> if it's\nprovided (<a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1548\">#1548</a>)</li>\n<li><a\nhref=\"https://github.com/actions/setup-node/commit/0355742c943ddb13ca8a6b700f824231caa91e75\"><code>0355742</code></a>\nRemove dummy NODE_AUTH_TOKEN export (<a\nhref=\"https://redirect.github.com/actions/setup-node/issues/1558\">#1558</a>)</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/actions/setup-node/compare/48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e...820762786026740c76f36085b0efc47a31fe5020\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate GitHub Actions across CI, mutants, and publish workflows to keep\nbuilds secure and current. Notably upgrades `actions/setup-node` to v7.\n\n- **Dependencies**\n- `actions/setup-node`: 6.4.0 → 7.0.0 (ESM migration; new cache key\noutputs)\n  - `taiki-e/install-action`: 2.82.9 → 2.83.2\n- `EmbarkStudios/cargo-deny-action`: 2.0.20 → 2.1.1 (handles\n`use-git-cli` deprecation)\n\n<sup>Written for commit 2604ea603c87b5eac934f95369bdfa47aa157ac2.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1135?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T16:25:25-07:00",
          "tree_id": "9a9783e5fe6e52e31687deda87574b7f9fedde42",
          "url": "https://github.com/andymai/brepkit/commit/8e48139fcc56c0be32a36e5d4c653f1a4194318f"
        },
        "date": 1784590067316,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 906767,
            "range": "± 11483",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1024505,
            "range": "± 22666",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13093,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 647428,
            "range": "± 6741",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28219693,
            "range": "± 348822",
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
          "id": "6c309a87ca4b21f130b2d5e038094bd006f801a3",
          "message": "chore(deps-dev): bump the npm group with 2 updates (#1133)\n\nBumps the npm group with 2 updates:\n[@commitlint/cli](https://github.com/conventional-changelog/commitlint/tree/HEAD/@commitlint/cli)\nand [prettier](https://github.com/prettier/prettier).\n\nUpdates `@commitlint/cli` from 21.2.0 to 21.2.1\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/releases\">@​commitlint/cli's\nreleases</a>.</em></p>\n<blockquote>\n<h2>v21.2.1</h2>\n<h1><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.1.0...v21.2.0\">21.2.0</a>\n(2026-06-30)</h1>\n<h3>Features</h3>\n<ul>\n<li><strong>resolve-extends:</strong> resolve pure-ESM presets\n(conventional-changelog v7/v9/v10) (<a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/issues/4859\">#4859</a>)\n(<a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/fdb566fe59457a786eac80e2a8cbb994638daba0\">fdb566f</a>)</li>\n</ul>\n<h3>Chore, doc, etc.</h3>\n<ul>\n<li>chore(read): replace deprecated git-raw-commits with git-client by\n<a\nhref=\"https://github.com/igordanchenko\"><code>@​igordanchenko</code></a>\nin <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4860\">conventional-changelog/commitlint#4860</a></li>\n<li>docs(issue-template): ask for package manager and version by <a\nhref=\"https://github.com/escapedcat\"><code>@​escapedcat</code></a> in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4867\">conventional-changelog/commitlint#4867</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a\nhref=\"https://github.com/igordanchenko\"><code>@​igordanchenko</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/conventional-changelog/commitlint/pull/4860\">conventional-changelog/commitlint#4860</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.2.0...v21.2.1\">https://github.com/conventional-changelog/commitlint/compare/v21.2.0...v21.2.1</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/conventional-changelog/commitlint/blob/master/@commitlint/cli/CHANGELOG.md\">@​commitlint/cli's\nchangelog</a>.</em></p>\n<blockquote>\n<h2><a\nhref=\"https://github.com/conventional-changelog/commitlint/compare/v21.2.0...v21.2.1\">21.2.1</a>\n(2026-07-08)</h2>\n<p><strong>Note:</strong> Version bump only for package\n<code>@​commitlint/cli</code></p>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/conventional-changelog/commitlint/commit/6e200411e63fb473d59cca1b4449a28f72067df0\"><code>6e20041</code></a>\nv21.2.1</li>\n<li>See full diff in <a\nhref=\"https://github.com/conventional-changelog/commitlint/commits/v21.2.1/@commitlint/cli\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\nUpdates `prettier` from 3.9.4 to 3.9.5\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/releases\">prettier's\nreleases</a>.</em></p>\n<blockquote>\n<h2>3.9.5</h2>\n<p>🔗 <a\nhref=\"https://github.com/prettier/prettier/blob/3.9.5/CHANGELOG.md#395\">Changelog</a></p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/prettier/prettier/blob/main/CHANGELOG.md\">prettier's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>3.9.5</h1>\n<p><a\nhref=\"https://github.com/prettier/prettier/compare/3.9.4...3.9.5\">diff</a></p>\n<h4>Markdown: Cap ordered list mark at 999,999,999 (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19351\">#19351</a>\nby <a href=\"https://github.com/tats-u\"><code>@​tats-u</code></a>)</h4>\n<p>CommonMark parsers only support ordered list item numbers up to\n999,999,999.</p>\n<p>With this change, Prettier now caps the ordered list item number at\n999,999,999 to ensure that the output is correctly parsed as an ordered\nlist by CommonMark parsers. Numbers larger than 999,999,999 are not\nparsed as list item numbers and are left unchanged in the output:</p>\n<!-- raw HTML omitted -->\n<pre lang=\"markdown\"><code>&lt;!-- Input --&gt;\n999999998. text\n999999998. text\n999999998. text\n999999998. text\n<p>1234567890123456789012) text</p>\n<p>&lt;!-- Prettier 3.9.4 --&gt;\n999999998. text\n999999999. text\n1000000000. text\n1000000001. text</p>\n<p>1234567890123456789012) text</p>\n<p>&lt;!-- Prettier 3.9.5 --&gt;\n999999998. text\n999999999. text\n999999999. text\n999999999. text</p>\n<p>1234567890123456789012) text\n</code></pre></p>\n<h4>Markdown: Avoid corrupting empty link with title (<a\nhref=\"https://redirect.github.com/prettier/prettier/pull/19487\">#19487</a>\nby <a href=\"https://github.com/andersk\"><code>@​andersk</code></a>)</h4>\n<p>Do not remove <code>&lt;&gt;</code> from an inline link or image with\nan empty URL and a title, as this removal would change its\ninterpretation.</p>\n<!-- raw HTML omitted -->\n<pre lang=\"md\"><code>&lt;!-- Input --&gt;\n[link](https://github.com/prettier/prettier/blob/main/&lt;&gt;\n&quot;title&quot;)\n<p>&lt;!-- Prettier 3.9.4 --&gt;\n[link](<a\nhref=\"https://github.com/prettier/prettier/blob/main/\">https://github.com/prettier/prettier/blob/main/</a>\n&quot;title&quot;)</p>\n<p>&lt;!-- Prettier 3.9.5 --&gt;\n&lt;/tr&gt;&lt;/table&gt;\n</code></pre></p>\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/b6c7d1806807162658fd5694d002b54b778c3756\"><code>b6c7d18</code></a>\nRelease 3.9.5</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/cd54ccc7288f2a4585ba5bf96883e2ed3b18e781\"><code>cd54ccc</code></a>\nAvoid corrupting empty Markdown link with title (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19487\">#19487</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/2bb67ce8cd326dbaccbdec97e7578985f6009a14\"><code>2bb67ce</code></a>\nPreserving comments' <code>placement</code> property (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19567\">#19567</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/91bcac88ec1d784767ebbf5ec4686a5dbf37c3c1\"><code>91bcac8</code></a>\nAdd more tests for comment-only object type (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19587\">#19587</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/cbee7377a1875686558bfc9416b789d66771d9e7\"><code>cbee737</code></a>\nRemove space in empty object type (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19583\">#19583</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/6394c732090efdf7798e0c8dc44c46583b4131c3\"><code>6394c73</code></a>\nAlign empty module declaration with TS (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19568\">#19568</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/a4e6f7a0264bad45f3c153773111f60a5d406d55\"><code>a4e6f7a</code></a>\nPrevent the addition of space in <code>type()</code> with <code>+</code>\n(<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19516\">#19516</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/3d063b57b1a2fdc660341b6e1c719e09997cc72c\"><code>3d063b5</code></a>\nIgnore dangling comments when checking type parameter comments (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19572\">#19572</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/908503e9ee02718ef0ec21a023f2c0fd5fb00603\"><code>908503e</code></a>\nHandle dangling comments in <code>SwitchStatement</code> (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19581\">#19581</a>)</li>\n<li><a\nhref=\"https://github.com/prettier/prettier/commit/943a475c3592149aad6062ad235338da548365ea\"><code>943a475</code></a>\nAngular: Support expression for exhaustive typechecking (<a\nhref=\"https://redirect.github.com/prettier/prettier/issues/19571\">#19571</a>)</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/prettier/prettier/compare/3.9.4...3.9.5\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore <dependency name> major version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's major version (unless you unignore this specific\ndependency's major version or upgrade to it yourself)\n- `@dependabot ignore <dependency name> minor version` will close this\ngroup update PR and stop Dependabot creating any more for the specific\ndependency's minor version (unless you unignore this specific\ndependency's minor version or upgrade to it yourself)\n- `@dependabot ignore <dependency name>` will close this group update PR\nand stop Dependabot creating any more for the specific dependency\n(unless you unignore this specific dependency or upgrade to it yourself)\n- `@dependabot unignore <dependency name>` will remove all of the ignore\nconditions of the specified dependency\n- `@dependabot unignore <dependency name> <ignore condition>` will\nremove the ignore condition of the specified dependency and ignore\nconditions\n\n\n</details>\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nUpdate dev tooling to keep commit checks and formatting current: bump\n`@commitlint/cli` to 21.2.1 and `prettier` to 3.9.5. No app code\nchanges.\n\n- **Dependencies**\n- `@commitlint/cli`: 21.2.0 → 21.2.1 (moves to\n`@conventional-changelog/git-client`, replacing deprecated\n`git-raw-commits`)\n  - `prettier`: 3.9.4 → 3.9.5 (minor Markdown fixes)\n\n<sup>Written for commit 7a32bc343dee95aa395cdbf5f413ab771640da62.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1133?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T16:25:33-07:00",
          "tree_id": "68b222176741cfa4aa1c2c82c11d3cad8975fcbd",
          "url": "https://github.com/andymai/brepkit/commit/6c309a87ca4b21f130b2d5e038094bd006f801a3"
        },
        "date": 1784590229678,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 876949,
            "range": "± 1861",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 970642,
            "range": "± 10132",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11997,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 651303,
            "range": "± 1779",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26723952,
            "range": "± 86735",
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
          "id": "8d626c74e46c118800ac826e7ce8cc4ed04a93bd",
          "message": "fix(algo): rescue corner-window cone-cylinder sections and accept multi-piece fuses (#1136)\n\n## Root\n\nThe lite magnet-pad fuse (the lightweight export family's `4×4 stress` /\n`solid bin + magnet` root): a pad wall (r=4.45) grazes a socket-profile\ncorner cone by 0.094 mm, and the parallel-axis cone×cylinder\nintersection branches exit the cone patch through its **angular-window\ncorner**. The in-both span is a 0.097 mm piece of a 1.4 mm curve — far\nbelow the graze-refinement's extent-scaled minimum crossing — so\n`restrict_curves_to_faces` dropped both branch curves. With the\nconnecting pieces missing, the wire builder dead-ended at the junctions\nand backtracked into zero-area out-and-back slits (free=5 over=19), the\nfuse fell back to a ~9755-face mesh, and every downstream drill\ninherited the poison.\n\n## Fixes (all five required — each unmasked the next)\n\n1. **math**: `Circle3D::intersect_circle` (coplanar pair, near-tangent\ndouble-root collapse in the `sqrt(2r·δ)` well);\n`closed_circle_boundary_crossings` now also crosses Circle boundary\nedges (the Line-only scan produced ODD crossing sets, desynchronizing\nthe cyclic arc pairing) and inserts a midpoint between same-arc hit\npairs so a kept span never shares both endpoints with the boundary arc\n(the co-endpoint-lens class — `merge_duplicate_edges` would fold the\npair and collapse the lens to a zero-area slit).\n2. **phase FF**: `rescue_corner_crossing` — at both graze-drop sites,\nbisect the in/out window transitions, emit the trimmed sub-span, and\nsnap its endpoints to the exact boundary-curve × partner-surface triple\njunction (the boundary *foot* alone is displaced ~1e-6 along the\nboundary by the curve-fit error and mints a duplicate vertex). A\nstrict-interior midpoint gate keeps true tangency grazes dropped.\n3. **fit-error weld plumbing** (the recurring 1e-6-fit vs 1e-7-gate\nclass): `curve_endpoints` returns pave-vertex positions when within the\nweld band; section↔boundary UV copies of one 3D junction are reconciled\nbefore the pendant filter (they landed in different 1e-7 graph cells);\n`find_splits_on_line` dedups candidates at weld scale in 3D; plane faces\ndrop zero-extent section remnants.\n4. **face splitter**: mirrored-winding retry when the greedy trace on a\nu-periodic cylinder is broken and the rectilinear arrangement declines\n(oblique ellipse/conic cuts are outside its domain); adopted only with\nno NEW broken-loop signatures (`wire_loops_self_cross` legitimately\nstays set on full-period band loops — the seam vertex appears at both\nunwrapped copies).\n5. **ops gate + tessellation**: the multi-component balance (euler − L\n== 2N disjoint closed pieces) is checked **before** `unify_faces` (the\nlite base at this stage is legitimately 16 disjoint feet; unify was\nmangling a clean result), and Fuse is admitted to the multi-region\nacceptance gate. Tessellation edge sampling now honors the\nendpoint-trimmed NURBS convention — all three `edge_sampling.rs` NURBS\narms sampled the full knot domain, ripping a bd=119 crack along the\nparent curve of a trimmed junction spline.\n\n## Result\n\nCaptured single-pad fuse (real tool operands): ops `boolean()` returns\nthe **analytic 951-face result** (194 cyl / 256 cone / 501 plane),\nposition-manifold (free=0 over=0 posBad=0), mesh **bd=0 nm=0** at export\ndeflection, **1.2 s vs 2.7 s** for the fallback it replaces.\n\n## Verification\n\n- New fixture `crates/io/tests/lite_pad_graze_fuse_inmem.rs` (two\ncropped lite feet + the grazing pad, 73 KB of arena bins) — **fails on\nthe pre-fix engine, passes now** (differential-verified by stashing the\nengine changes).\n- New unit tests for `intersect_circle` (two-point, real pad/rim\nconfiguration, near-tangent collapse, disjoint/skew/concentric empties).\n- Foil web green: wasm gridfinity canary 27/27, ops lib 772, algo 169,\nmath 463, full io fixture suite (groove-mouth, junction-disc, snap-slot,\nnub, deepened-notch, d-series lip fuses all pass).\n\nNot yet covered (follow-ups tracked in the roadmap): the 64-pad\nwhole-base fuse + drill chain, and the tool-side lightweight re-probe\nafter release.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes a graze-case cone×cylinder intersection that dropped corner-window\nsections and forced a mesh fallback. Restores an analytic, watertight\nfuse and admits multi-piece operands; 951 faces, 0 cracks, 1.2 s vs 2.7\ns.\n\n- Bug Fixes\n- `math`: `Circle3D::intersect_circle` collapses near-tangent double\nroots; circle sections also split at circle boundaries with a midpoint\nto avoid co-endpoint lenses.\n- `phase_ff`: rescue corner-window crossings; trim the kept sub-span\nclamped to the boundary edge span; snap endpoints to the exact\nboundary×surface junction (searches hole wires too); strict-interior\ngate keeps true tangency grazes dropped and fails closed on projection\nerrors.\n- Weld plumbing: `curve_endpoints` returns pave-vertex positions within\nthe weld band; reconcile section/boundary UV copies; weld-scale dedup in\n`find_splits_on_line`; drop zero-extent plane remnants.\n- Face splitter: mirrored-winding retry on u-periodic cylinders when the\ngreedy trace breaks; adopted only when it adds no new broken-loop\nsignatures.\n- Operations: check multi-component Euler balance before `unify_faces`,\nand allow Fuse results with multiple disjoint regions.\n- Tessellation: NURBS edge sampling honors endpoint-trimmed domains to\nprevent cracks along trimmed junction splines.\n\n<sup>Written for commit 42381dd3051393709068b166eee5362cb4014943.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1136?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-21T06:15:15Z",
          "tree_id": "8ecd0c9c8c0bfd3c9e24137db92c74f015763f40",
          "url": "https://github.com/andymai/brepkit/commit/8d626c74e46c118800ac826e7ce8cc4ed04a93bd"
        },
        "date": 1784614655456,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 949173,
            "range": "± 1860",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1051849,
            "range": "± 1367",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13110,
            "range": "± 260",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 662147,
            "range": "± 10301",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28888797,
            "range": "± 36998",
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
          "id": "f0fcaee8c4cc3d69a25bfb78a714c2142f85d7af",
          "message": "chore(main): release 2.127.4 (#1137)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.4](https://github.com/andymai/brepkit/compare/v2.127.3...v2.127.4)\n(2026-07-21)\n\n\n### Bug Fixes\n\n* **algo:** rescue corner-window cone-cylinder sections and accept\nmulti-piece fuses\n([#1136](https://github.com/andymai/brepkit/issues/1136))\n([8d626c7](https://github.com/andymai/brepkit/commit/8d626c74e46c118800ac826e7ce8cc4ed04a93bd))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nRelease 2.127.4 updates `brepkit-wasm` with an algorithm fix for\ncorner-window cone–cylinder intersections and support for multi-piece\nfuses. This reduces failures in complex boolean operations.\n\n- **Bug Fixes**\n- Recover cone–cylinder sections in corner-window cases and allow\nmulti-piece fuses.\n\n<sup>Written for commit 938579b1e487f88a188ffa939ebd968873ec4531.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1137?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-21T06:22:35Z",
          "tree_id": "0cdbbbbbd12dfa03e02f5955acc983a97b11187b",
          "url": "https://github.com/andymai/brepkit/commit/f0fcaee8c4cc3d69a25bfb78a714c2142f85d7af"
        },
        "date": 1784615088247,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 750336,
            "range": "± 6802",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 850217,
            "range": "± 16448",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 10311,
            "range": "± 127",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 549977,
            "range": "± 5187",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 23474649,
            "range": "± 338440",
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
          "id": "f5ef4df7f3490bdfb993e3024c1220409c7282b6",
          "message": "fix(operations): fold a multi-component fuse tool in piece by piece (#1138)\n\n## Root\n\nFollow-up to #1136. The lite base's 64 magnet pads arrive at the fuse as\nONE 64-component union; feeding the whole thing to GFA collapses the\noperands (the pavefiller mishandles many-piece operands, same class as\nthe existing multi-piece-input Cut limitation) and the entire fuse falls\nback to mesh — flattening every analytic socket and poisoning the\ndownstream drills (the lightweight `4×4 stress` chain).\n\n## Fix\n\nFuse distributes over a disjoint-union tool. When GFA has already failed\nand the tool splits into 2+ disjoint components, fold them into the\ntarget one at a time through the full `boolean()` entry — after #1136,\neach per-piece pad fuse is exactly the configuration the engine handles\nanalytically. Mirrors the existing `cut_multi_region_input` precedent\n(including the fresh per-component deep copy). Gated strictly behind the\nexisting GFA failure path, so nothing changes for single-piece tools.\n\n## Result on the captured lite 4×4 chain\n\n- 64-pad fuse export mesh: 18 non-manifold edges → **watertight (bd=0\nnm=0)**; pad-wall cylinders survive analytically; F 15648 → 1753.\n- Known residual (tracked as follow-up): one per-piece pad fuse against\nthe accumulated base still hits a pathological GFA case (87 faces out\nafter 169 s) and mesh-falls-back, taking its foot's cones with it — the\nfold contains the damage to that piece instead of the whole base.\n\n## Verification\n\n- Foils green: ops lib 772, algo 169, wasm gridfinity canary 27/27, full\nio fixture suite (including the #1136 `lite_pad_graze_fuse_inmem`\nfixture), clippy/fmt clean.\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nFixes fuse with multi-piece tools by folding each disjoint component one\nat a time when GFA fails, preserving analytic geometry and preventing\nwhole-model mesh fallback. On the lite 4×4 chain, the 64‑pad fuse\nbecomes watertight and keeps pad-wall cylinders analytic.\n\n- **Bug Fixes**\n- Only for `Fuse` after GFA failure when the tool splits into 2+\ndisjoint components and has no inner shells.\n- Builds per-piece solids from face components, deep-copies them, and\nfuses sequentially via `boolean()`.\n- Exact result (fuse distributes over disjoint union); mirrors\n`cut_multi_region_input`; single-piece tools unchanged; includes a unit\ntest folding a two-post tool into a slab (volume + manifold).\n- Result: non-manifold edges 18 → 0; faces 15648 → 1753. One per-piece\nfallback remains and is tracked separately.\n\n<sup>Written for commit de8df0ae4b8b89a14eef9a70a8b734a7b943e9ed.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1138?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->",
          "timestamp": "2026-07-21T06:57:29Z",
          "tree_id": "a2f550c86e79cd068d16e2d980168c2cc6b2e018",
          "url": "https://github.com/andymai/brepkit/commit/f5ef4df7f3490bdfb993e3024c1220409c7282b6"
        },
        "date": 1784617191471,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 935294,
            "range": "± 2081",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 1037949,
            "range": "± 1751",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 13034,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 652647,
            "range": "± 2249",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 28399277,
            "range": "± 60137",
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
          "id": "1732122b678427a794833f06548f09b93d967c83",
          "message": "chore(main): release 2.127.5 (#1139)\n\n:robot: I have created a release *beep* *boop*\n---\n\n\n##\n[2.127.5](https://github.com/andymai/brepkit/compare/v2.127.4...v2.127.5)\n(2026-07-21)\n\n\n### Bug Fixes\n\n* **operations:** fold a multi-component fuse tool in piece by piece\n([#1138](https://github.com/andymai/brepkit/issues/1138))\n([f5ef4df](https://github.com/andymai/brepkit/commit/f5ef4df7f3490bdfb993e3024c1220409c7282b6))\n\n---\nThis PR was generated with [Release\nPlease](https://github.com/googleapis/release-please). See\n[documentation](https://github.com/googleapis/release-please#release-please).\n\n<!-- This is an auto-generated description by cubic. -->\n---\n## Summary by cubic\nBump `brepkit-wasm` to 2.127.5 with a fix to the multi-component fuse\noperation. The fuse tool now processes components piece by piece for\nmore reliable results when combining parts.\n\n<sup>Written for commit 083fb1eac6e214222968b7e82d31fcb5d02790e5.\nSummary will update on new commits.</sup>\n\n<a\nhref=\"https://cubic.dev/pr/andymai/brepkit/pull/1139?utm_source=github\"\ntarget=\"_blank\" rel=\"noopener noreferrer\"\ndata-no-image-dialog=\"true\"><picture><source\nmedia=\"(prefers-color-scheme: dark)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"><source\nmedia=\"(prefers-color-scheme: light)\"\nsrcset=\"https://www.cubic.dev/buttons/review-in-cubic-light.svg\"><img\nalt=\"Review in cubic\"\nsrc=\"https://www.cubic.dev/buttons/review-in-cubic-dark.svg\"></picture></a>\n\n<!-- End of auto-generated description by cubic. -->\n\nCo-authored-by: brepkit[bot] <265643962+brepkit[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-21T07:05:47Z",
          "tree_id": "ab9ee946325bdf44b33bd1d75dcf93769bc12fb8",
          "url": "https://github.com/andymai/brepkit/commit/1732122b678427a794833f06548f09b93d967c83"
        },
        "date": 1784617687730,
        "tool": "cargo",
        "benches": [
          {
            "name": "boolean/cut_box_box",
            "value": 868930,
            "range": "± 20004",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/fuse_box_box",
            "value": 967477,
            "range": "± 2742",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/intersect_box_box",
            "value": 11830,
            "range": "± 355",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/cut_cylinder_through_box",
            "value": 608765,
            "range": "± 7408",
            "unit": "ns/iter"
          },
          {
            "name": "boolean/perforated_cut_36",
            "value": 26313940,
            "range": "± 63793",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}