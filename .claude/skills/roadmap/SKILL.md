---
name: roadmap
description: Use at the start of an autonomous or unsupervised session to pick what to work on, when deciding whether a geometry case is worth chasing, when a task looks like something a past session already tried, or before claiming a case is closed. The sanctioned work-selection doctrine: what is open and ready, what is terminal, the chase filters, and the acceptance bar.
---

# Roadmap: choosing what to work on

This is the sanctioned work-selection doctrine for autonomous sessions. It says what
is open and ready, what is TERMINAL (do not re-attempt without new tooling), which
work to chase and which to skip, and the bar a case must clear to be called closed.

## This is a LIVING document: maintenance is mandatory

When a session **closes, defers, or discovers** a work item, it MUST update this skill
in the same PR. A stale roadmap is worse than none: past sessions burned large budgets
rediscovering dead ends this file was supposed to name. Keep every entry to ONE line
with a pointer (a test path, a git-history PR number, a memory-free source file) that
carries the detail. Never duplicate the detailed truth here; point at the repro.

The `#[ignore]` inventory below is the load-bearing artifact. Before quoting any
"deferred" claim, regenerate and reconcile it:

```bash
rg -n -A2 '#\[ignore' crates/    # filter the 3 doc-comment false hits by hand
```

## When to use

- Starting a session with no assigned task and needing to pick high-value work.
- A task resembles something that may already be tried, closed, or proven impossible.
- Deciding whether an analytic-recovery or parity case is worth the budget.
- Before writing "this case is closed" anywhere.

## The north star

Replace the incumbent kernel in the gridfinity layout tool (`~/Git/gridfinity-layout-tool`)
at full parity, across all its generator scenarios: 100% triangle correctness, volume
correctness, manifold correctness, AND generation performance at least as good. Parity
first, then beating it, is the acceptance bar. See `parity-benchmarking` for the harness.

Campaign history, one paragraph: gridfinity bin parity reached (10/11 kernel-suite
cases, PRs through #938); the four primitive-boolean mesh-fallbacks eliminated and made
exact analytic that now beat the reference kernel 2.9-9.5x head to head (box-sphere
intersect #1006, sphere-cyl cut #1005, perpendicular cyl-union #1008, torus-box #1010,
all on the keystone surface-aware-AABB fix #1003); revolve made exact-analytic (#1012);
GPU render milestones shipped (offscreen #1013, interactive viewer #1016, compute-mesher
#1017, screen-space adaptive LOD #1021); tessellation-parity wave (2026-07-07, #1029
ruled-direction grid + partial-band CDT, #1030 cut orientation toggle + open-hole-shell
guard) — honeycomb bins 63k→~3k triangles, compartment cavity cuts watertight at export
tolerance, cut results with reversed tool faces no longer inverted.

**A 2026-07-07 lesson that reshapes triage: not every scenario failure is a boolean
fallback.** The honeycomb 15x triangle blow-up and the compartment non-manifold STL
family both replayed with ZERO mesh fallbacks — the roots were in tessellation density,
shared-rim meshing, and face orientation. Before assuming GFA, capture the actual
boolean traffic with the probe kernel (branch `probe/boolean-capture`, local-only:
`telemetry` hook in `operations::boolean`, wasm `probeEnableCapture`/`probeSummary`
bindings, replay driver `crates/io/examples/replay_captures.rs`) and replay the
operands natively. Also: the tool's `*.scenario.*` snapshot tests pin EXACT
reference-kernel triangle counts — a different kernel can never match them; treat
received-below-expected as benign density difference, received-10x-above as a defect.

## The priority filters (rules with reasons)

1. **Chase operations that RE-CREATE an existing analytic surface type. Do NOT chase
   ops that INVENT a blend or approximation surface.** A boolean or revolve result face
   is a trimmed patch of an *input* surface, so it is always closable with the right
   split. Fillet and chamfer walls, general sweep and loft side faces, and offsets of
   NURBS input introduce a NEW surface with no closed form; they are fundamentally
   approximate. See `analytic-preservation`.
2. **Solve the NARROW case (coaxial, perpendicular, equal-radius), not the general
   problem.** Every primitive-boolean win was gated to one specific configuration and
   defers to the generic marcher otherwise. Sessions that reached for a general solver
   burned budget and shipped nothing.
3. **Prefer work with a stable primitive repro over work that needs tooling first.**
   The four primitive-boolean cases (stable repros in
   `crates/operations/examples/approx_census.rs`) were picked over the tooling-blocked
   scoop case for exactly this reason.
4. **After ANY GFA or boolean change, re-probe scenario face counts before claiming
   anything.** Scorecards rot silently; a stale one once hid a regression through a
   whole release. This is mandatory, not optional (see `parity-benchmarking`).

## TERMINAL cases: do not re-attempt without the named missing primitive

Several past sessions burned large budgets rediscovering these. Each needs a component
that does not exist yet; without it, stop.

- **Equal-radius perpendicular cylinder-union RENDER.** The exact seam is a
  self-touching figure-eight (a genuine non-manifold singularity, odd Euler). The
  shipped artifact (#1008: analytic B-Rep whose marched-NURBS seam dodges the touch,
  plus exact closed-form volume) STANDS. Needs a face-split-at-pinch primitive on a
  periodic wall, or a periodic-aware crossing-holes mesher. There is no
  `exact_cylinder_cylinder` symbol; do not go looking for one.
- **Plane-by-sphere splitting across the chord-discretized equator.** The general
  capability behind box-sphere; a section circle's crossings miss a polygon-approximated
  equator by the sagitta. Box-sphere was closed (#1006) with a case-specific seam-plane
  fit (`rg -n 'seam_plane' crates/`). The general fix is a UV-space arrangement
  splitter, a dedicated multi-day component not yet built. The boundary-plane
  crossing technique is proven and reusable.
- **Gridfinity scoop fuse (3x3 scoop+label+lip).** Root: a lip-foot cone must be split
  with a coordinated staircase cone-split plus bracket-cap re-trim sharing the new edge;
  every one-sided attempt regresses. Many sequential autonomous passes exhausted.
  Parity is already MET via a correct-but-slow mesh fallback (this is perf-only). Note:
  STEP-faithful in-memory repros now EXIST (`crates/io/tests/scoop*_inmem.rs`); the old
  "needs serialization tooling first" framing is stale. The real blocker is the
  coordinated split.
- **Snap-clip deepened-notch case.** A later cut deepening an earlier opening strands
  the old floor edge. The 2D polygon-union primitive it needs now EXISTS
  (`crates/math/src/polygon_boolean.rs::polygon_union`). What remains is sound
  deepened-notch detection that does not false-positive on near-coincident faces.
- **A universal smarter merge-key for duplicate edges. PROVEN UNBUILDABLE.** The
  gridfinity lip corner (chord + arc, same endpoints) MUST merge; the torus-box in-tube
  lens (line + co-endpoint arc) MUST stay distinct. No merge-key discriminant separates
  them; the distinction is global. Sanctioned pattern: splitter-side midpoint splits,
  per case, so no two edges share both endpoints, and leave
  `merge_duplicate_edges` (in `crates/algo/src/builder/builder_solid.rs`) alone. Control
  the geometry you emit; do not make the shared merge smarter.

## DEFERRED but ready: open items with a repro

Regenerate the inventory (command above) and reconcile before trusting this table.
Current genuine `#[ignore]` items worth work:

| Item (repro) | Layer | Symptom / first probe |
|---|---|---|
| **Compartment manifold residuals** (re-scoped again after the corner-crescent fix; 12/13 pass) | algo/GFA + unknown | Two roots CLOSED: the grazing-EF lip-corner vertex (`phase_ef` angle-scaled endpoint window, `crates/io/tests/lipcorner_tangent_inmem.rs`) and the boundary-re-trace section family (`section_on_existing_boundary` in `fill_images_faces.rs` + straightness-aware hole weave + crossing-midpoint hole probes in `face_splitter/mod.rs`; fixtures `crates/io/tests/lipfuse_boundary_retrace_inmem.rs` — that fix un-masked 3 halfSockets-tilt cases that only "passed" via a watertight mesh fallback, then fixed them for real). The retrace guard's discriminant: an exact whole-edge duplicate section is KEPT (threading it routes the face through the split/rebuild that aligns coincident-face partitions — dropping it regressed the plain shelled-cup lip fuses d3/d4/d5 to mesh fallback), while a SUB-SPAN re-trace (45deg-split half-arc, straight run split at a divider crossing) is dropped. A third root CLOSED (chord-sagitta classifier seed, `find_point_outside_holes` in `face_splitter/containment.rs`; fixture `crates/io/tests/halfsockets_clipcut_inmem.rs`): the halfSockets base-clip cut's 1.2mm ring floor got its seed in the corner-arc sagitta gap of the chord-approximated hole polygon → ring classified Inside → open shell → mesh fallback poisoned the whole export chain. Closed `2×2 crossing tilts` outright and took `2×6 halfSockets ±40` 26→1 NM; 10/13 pass. A FOURTH root CLOSED (corner-crescent hole promotion, `loop_containment` in `face_splitter/mod.rs`; fixtures `crates/io/tests/socket_assembly_fuse_inmem.rs`): the bin×socket-assembly fuse at the z=5 base interface leaves a ~0.1mm crescent of bin bottom overhanging each corner socket's chamfered outline; the wire builder hands the crescents back as CW loops, the hole-promotion pass's SINGLE interior probe slipped across the thin boundary into the adjacent socket-square outer, so the crescents stayed "holes", were first-vertex-matched into nothing, and got dumped onto an arbitrary first sub-face that same-domain-dropping then erased — free edges at all four bin corners; the compartments variant went further into a GFA-reject + non-manifold mesh fallback. Fixed by whole-boundary containment (promote only loops with points STRICTLY outside every outer; boundary-coincident re-trace loops must stay holes — promoting or dropping them un-threads the d3/d4/d5 shelled-cup lip fuse). Closed `1×4 2×8-comps` (now analytic, 5× faster) and `1.5×6 no-halfSockets ±40`; 12/13 pass. Remaining 1 failure: `2×6 halfSockets ±40` = 1 NM residual — needs a FRESH operand capture (the session's chain captures are stale post-fix). Probe recipe: instrumented-kernel capture in `all` mode (hooks `boolean_with_evolution` — the tool's export fuses ALL go through the provenance path, invisible to a `boolean()`-only hook) + `VERTEX_WATCH` backtrace trap in `Vertex::new` (probe branch `probe/boolean-capture`). |
| **halfSockets zero-triangle bins** (`binGenerator.scenario.halfSockets`: `1×1`, `1.5×1.5`, `2×2` with half sockets → `triangleCount=0`, generator emits an EMPTY result) | unknown | Pre-existing, verified identical on clean main @43bda38c (2.124.5-era) — NOT caused by the corner-crescent fix. The fractional/taller variants (0.5×1, 2.5×2.5, 3×3, screw/magnet/weighted bases) all pass, so the root is specific to small integer bins with half sockets. First probe: capture the export chain with the probe kernel and find which op returns the empty solid ("a fuse can lose its WHOLE interior and still pass free=0"). |
| **Honeycomb+handles kernel-poisoning panic** (`binGenerator.scenario.combinedFeatures` "2×2 honeycomb walls + handle holes") | wasm/operations | 205s runtime, then a panic mid-call leaves the wasm borrow flag locked — every later kernel call dies with "recursive use of an object". Two downstream tests die at 0ms. The probe kernel's free-function bindings still read post-poison. |
| `dovetail_corner_clip_intersect_is_watertight` (`crates/io/tests/dovetail_cornerclip_intersect_inmem.rs`) | algo/GFA | Coincident-wall + analytic-corner Intersect drops the boundary; tool mesh-falls-back to a slab; the full dovetail scenario suite still exceeds a 25-min timeout (re-measured 2026-07-07). The sibling non-ignored test reaches free<=1 raw, so the residual is the last corner-cap sub-face. Probe: `cargo test -p brepkit-io --test dovetail_cornerclip_intersect_inmem -- --ignored --nocapture`. |
| `fuse_shelled_box_with_socket_loft` (`crates/operations/src/boolean/tests.rs`) | operations/loft + algo | Non-manifold edge at shelled-box + socket-loft fuse. Root: `loft` builds `EdgeCurve::Line` ring edges (`crates/operations/src/loft.rs`, `rg -n 'EdgeCurve::Line' crates/operations/src/loft.rs`) so an octagonal frustum cannot annihilate an arc-cornered cap. Gated on a curve-preserving loft. Probe: run it `-- --ignored`. |
| Fillet closed-rim free-edges (latent, no ready repro) | blend | `split_edge_at` in `crates/blend/src/trimmer.rs` rebuilds only the trimmed face's wire, never the neighbor cap face, leaving boundary edges. The d5 repro was reworked to dodge it and now passes. See `fillet-blend`. |
| Revolve follow-ups (no ignored test) | operations/revolve | Pointed-cone apex merge, annulus/washer-cap merge, partial-turn torus. All stay analytic+exact, merely over-segmented. Probe: `cargo run --release --example approx_census -p brepkit-operations` (`revolve_matrix`). |

Fresh full scenario-matrix baseline (2026-07-07, kernel 2.124.0 + #1029/#1030 overlay,
`BREPJS_KERNEL=brepkit pnpm exec vitest run --project generators <suite>` in the tool):
compartment manifold 5/13 (was 0/13; remainder = the lip-corner + tilted-divider rows
above), honeycombJunction engine-fixed (63k→~3k tris; snapshot pins remain
kernel-specific), groupedScoop + splitBin manifold PASS, combinedFeatures 3/11
(handles-panic row above; scoop case near-parity), snapClip 1/4, fit-offset 0/2,
dovetailKey 1/2 (all three: watertight-STL asserts — re-probe after the lip-corner fix
lands, they may share it), dovetail suite timeout >25min (cornerclip row above).

The remaining `#[ignore]` entries are diagnostics or slow perf cases, not open bugs:
the `profile_intersect.rs` box-sphere probes are stale leftovers (box-sphere shipped
analytic in #1006), `staircase_fuse_with_cylinders` is a ~2 min perf run, and the two
`#696` dovetail entries plus `diverge_first_cut` are print-only diagnostics.

CLOSED, do not re-open as deferred: honeycomb wall-pattern cut (#925/#928,
`crates/io/tests/gridfinity_honeycomb_cut_inmem.rs` passes), reversed-edge periodic-copy
top-face (#932, `extrude_half_*_reversed_edge_volume` pass), multi-arc hemisphere gap
(#1006).

## Subsystem trap notes (crates without their own skill)

- **heal `fix_duplicate_faces` IS implemented** (solid-scoped, `crates/heal/src/fix/solid.rs`,
  returns `Status::DONE2`), not a no-op stub. It compares only centroid, normal, and
  edge count, so it can miss true-but-differently-wound duplicates; do not rely on it
  for subtle cases. Verify current state before quoting either way.
- **heal, offset, and sketch have no distilled campaign knowledge.** They follow the
  same `debugging-doctrine`, but no skill covers their internals. Treat any diagnosis
  there as first-of-kind and write findings down (a test comment or a new note).
- **The v1 fillet deprecations are entangled with the public wasm API.**
  `operations/src/fillet/mod.rs::fillet` and `fillet/rolling_ball.rs::fillet_rolling_ball`
  are `#[deprecated]` yet still reached through the wasm `fillet` binding, via
  `wasm/src/helpers.rs::try_fillet` (it tries `fillet_rolling_ball` and `fillet` in its
  engine-preference chain). Migrating them changes public behavior; that is a product
  decision, not safe cleanup. The offset v1 path was already dropped in #850.
  `offsetSolid` now routes through the non-deprecated `offset_v2::offset_solid_v2`. See
  `fillet-blend` and `wasm-bindings`.

## Acceptance bar for a geometry campaign case

Every box before "closed":

- [ ] **Exact analytic result** where the inputs are analytic (typed faces, single to
      low-tens face count, not hundreds).
- [ ] **Watertight** tessellation (zero boundary edges).
- [ ] **Manifold** B-Rep (every edge used by exactly two faces, Euler balanced).
- [ ] **Full workspace suites green, INCLUDING** `cargo test -p brepkit-wasm --lib gridfinity`
      (running only algo/io/operations has shipped a gridfinity regression before).
- [ ] **Regression fixture shipped** with the fix (STEP or arena `.bin`; see `testing`).
- [ ] **Census clean or improved:** the row flips FALLBACK to analytic
      (`cargo run --release --example approx_census -p brepkit-operations`).
- [ ] **Head-to-head timing at least parity** (the brepjs wasm bench; see
      `parity-benchmarking`).
- [ ] **Release published** when user-facing (see `release-flow`).

## Anti-patterns

- Do NOT re-attempt a TERMINAL case hoping this time is different; it needs the named
  missing primitive, not another pass.
- Do NOT reach for the general solver when the narrow case is what parity needs.
- Do NOT call a case closed on an "exact analytic" census row alone; the census does not
  check correctness (see `analytic-preservation`).
- Do NOT quote a "deferred" or face-count claim without regenerating the inventory and
  re-probing scenarios; both rot silently.
- Do NOT close, defer, or discover an item and leave this skill unchanged.

## Related skills

`analytic-preservation` (the chase filters in depth), `parity-benchmarking` (the
scenario re-probe and head-to-head), `debugging-doctrine` (before any multi-pass dig),
`solid-verification` (the acceptance oracles), `testing` (fixtures and ready-repros),
`fillet-blend` (the blend traps), `release-flow` (shipping a user-facing close).
