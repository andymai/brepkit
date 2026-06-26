window.BENCHMARK_DATA = {
  "lastUpdate": 1782496436997,
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
      }
    ]
  }
}