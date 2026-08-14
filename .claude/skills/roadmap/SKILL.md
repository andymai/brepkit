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
**Closed-campaign narrative is rot: when a case closes, collapse its entry to one line
plus its fixture/PR pointer in the Closed section, and delete the dig log.**

The `#[ignore]` inventory is the load-bearing artifact. Before quoting any
"deferred" claim, regenerate and reconcile it:

```bash
rg -n -A2 '#\[ignore' crates/    # filter the doc-comment false hits by hand
```

**Inventory status (2026-08-13): ONE deferred-defect pin** —
`kumiko_diagonal_strut_fuse_is_exact` (`crates/io/tests/kumiko_strut_fuse_inmem.rs`),
the corner-window frontier's acceptance bar. Every other `#[ignore]` is an explicit
diagnostic or a slow-test marker. Known stale-but-harmless:
the `profile_intersect.rs` box-sphere probes (box-sphere shipped analytic in #1006),
`staircase_fuse_with_cylinders` (~2 min perf run), the two `#696` dovetail entries and
`diverge_first_cut` (print-only).

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

Where that stands: **REACHED AND SHIPPED (2026-08-13)** — the tool pins brepkit-wasm
3.2.38, its full generator suite is green on every bump (issue #1517 closed at
0/2790 on 3.2.36; see the Closed entries). The 3.2.38 parity matrix reads 0.45x
aggregate with brepkit faster on 25 of 26 rows (the last is a 1.04x noise-band
watch) and 0 non-manifold scenarios vs the reference's 5; all four
primitive-boolean fallbacks are exact analytic. Per-PR history and per-row
numbers live in git, MEMORY.md, and the bench harness — do not re-record them
here. Native criterion CAVEAT: the
cad_operations "mesh sphere" case runs a bench-local PER-FACE shim ~40x lighter than the
solid-level path — never compare it to solid-level numbers (`perf_probe` has the matching
native figure).

**The lesson that most reshapes triage: not every scenario failure is a boolean
fallback, and many are not geometry at all.** The honeycomb triangle blow-up and the
compartment non-manifold family both replayed with ZERO mesh fallbacks (roots were
tessellation density, shared-rim meshing, face orientation), and 18 of the 21
divider/floor failures were a missing brepjs ADAPTER method that threw before any
geometry ran. So: measure where the failure actually is before assuming GFA — capture
the real boolean traffic and replay it natively (recipe under "Tool-side measurement
recipes"). A family that fails in seconds is failing pre-geometry.

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
  Parity is already MET via a correct-but-slow mesh fallback (this is perf-only).
  In-memory repros exist (`crates/io/tests/scoop*_inmem.rs`); the blocker is the
  coordinated split, not tooling.
- **A universal smarter merge-key for duplicate edges. PROVEN UNBUILDABLE.** The
  gridfinity lip corner (chord + arc, same endpoints) MUST merge; the torus-box in-tube
  lens (line + co-endpoint arc) MUST stay distinct. No merge-key discriminant separates
  them; the distinction is global. Sanctioned pattern: splitter-side midpoint splits,
  per case, so no two edges share both endpoints, and leave
  `merge_duplicate_edges` (in `crates/algo/src/builder/builder_solid.rs`) alone. Control
  the geometry you emit; do not make the shared merge smarter.
- **Pinch-shim double-cover mesh residuals (groupedScoop case2 nm=75).** Two coincident
  face meshes span the same region by construction, so shared rim edges carry 3
  triangles; inherent to the shim encoding — the alternative is the face-split-at-pinch
  primitive above. Sub-export-tolerance (tool suites 7/7); parked in this row on purpose.

## OPEN: ready or gated work

| Item | Status / next step |
|---|---|
| **4x4 mag no-lip noise-band watch (1.04x on the 3.2.38 matrix)** | The only row the reference leads; has oscillated 1.00x-1.06x across 3.2.36-3.2.38 with no kernel change targeting it. Watch, do not chase, unless a fresh same-day matrix shows a real drift |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** | A product call, not just a fix: rejecting means the op fails outright. Mitigation shipped: `boolean::mesh_fallback_count()` + wasm `meshFallbackCount()` let pipelines snapshot-and-refuse |
| **Export angular default (5°) vs the reference's coarser effective default** | Tolerance-parity product choice, not mesher waste: 5° forces 18 segments/quarter-arc on r=0.6 slot corners, ~1.7x triangles vs reference at fine deflection. Revisit only as a product decision |
| **Kumiko corner-window strut fuses: marched-section junction consistency** | Two more roots shipped 2026-08-13: (1) the SSI marcher clamped every state a 0.1%-of-span margin inside non-periodic domain boundaries (`constrain_param` + `refine_ssi_point` re-clamping every iteration), so adjacent patches' sections each stopped one margin short of their shared edge — the ~2e-3 junction gaps were minted IN THE MARCHER; the "minted downstream in make_blocks/link_existing" hypothesis is REFUTED (neither can split section pave-blocks; only VE/EE/EF add extra paves, and only to operand edges). Chain ends now refine onto the exact boundary (`finish_chain`/`refine_onto_boundary` in surface_marching.rs; probe `crates/io/examples/strut_junction_probe.rs` shows endpoints exact and shared; pin `marched_section_ends_on_exact_domain_boundary`). (2) The no-seam splitter shortcut intercepted every all-Line-boundary curved face, and its disjoint-section fallback is sphere-only — a NON-periodic bilinear strut quad came back unsplit ("split into 1") and its in-wedge piece dropped whole; gated to periodic/sphere so quads take the generic arrangement (pin `nonperiodic_line_bounded_nurbs_quad_splits_by_disjoint_sections`). Third root shipped same day: `intersect_plane_nurbs` dumped its grid crossings into the proximity chainer, whose threshold CANNOT serve that scan — duplicated interior-edge crossings collapsed the average-spacing statistic to its floor (one clean crossing chained as dashes: the strut end-patch, transversal contact, signed-dist span 1.29, "graze fragments" refuted), and REMOVING the duplicates inflates the threshold and over-chains the honeycomb pcut1 thin-wall branches (over=7 — the two calibrations are coupled; any distance statistic fixes one side and breaks the other). Fix: marching-squares connectivity — each crossing is computed once per unique grid edge and linked through shared cells (saddle cells resolved by the center sample), ordered chains bypass the threshold chainer entirely via `build_curves_from_chains` (pin `plane_crossing_between_grid_columns_is_one_curve`; probe `strut_plane_probe.rs` shows patch Id(40)×plane as one 25-pt curve, was 10 dashes; honeycomb foil green). Fourth root same day: on the demoted-to-non-periodic wedge cylinder, the marched chain ends and the boundary split vertices share one 3D junction but carry UVs from DIFFERENT paths (projection vs boundary pcurve), ~1e-6 apart in u — above `remove_pendant_sections`' exact-tol weld, so the pruner ate both chains and the face survived unsplit ("split into 1" with sections=7; STRACE-PRE was the instrument, u=11.630971**5** vs 11.630972**5**). Fix: 3D-identity UV co-registration in split_face_2d_impl (section endpoints adopt the boundary vertex's UV when 3D positions coincide within 100·tol); the outer cylinder now partitions 3 ways with the in-strut band dropped (active pin `kumiko_wedge_outer_cylinder_partitions_at_strut_chains`). Fuse: 65 faces, free=26 over=2. LIVE ROOT (dug 2026-08-14, unfinished): the strut quads exiting the wedge SIDEWAYS (strut-file face Id(31) = combined Id(37); shares ruling edge Id(58) with Id(27)) get 2 sections — the outer-cylinder chain piece ENDING at the wedge corner (an interior point of the quad) and the side-plane section PASSING THROUGH that same corner mid-span — so the cut needs a T-junction split of the plane section at the corner, then 3 regions; today it makes 2 (SRCPART Id(37): 1.17 Inside-dropped + 5.83 Outside-kept whose wire carries the free ruling-edge spans). TWO band mismatches found in `find_splits_on_nurbs_section` (edge_splitting.rs): the broad phase gates nearest-SAMPLE distance at tol*100 while an on-curve point mid-segment sits ~len/128 from its nearest of 64 samples (rejects almost every real junction), and the post-ternary fine gate uses bare tol against ~1e-6 marched geometry. Fixing BOTH made the T-split fire but face 37 still produced 2 pieces with the small piece FLIPPING to Outside-selected (F 65->63, free 26->25, vol 126.8->122.6 — sideways): the wire builder does not yet route through the new junction and the interior-sample classification moved. REVERTED as unverified (verify-or-revert). Resume by re-applying both band fixes, then dig the wire-builder routing at the junction (STRACE-LOOP on combined face 37: loops n=4/n=5 share the u~0.08-0.24 boundary spans). Ready-repro `kumiko_diagonal_strut_fuse_is_exact` (ignored) is the acceptance bar; run the full splitter foil set on any change |
| **Marched FF sections carry `pave_block_id=None`** | Architectural note without a live repro (the snapClip op-cut-3 case replays clean, fixture `snapclip_export_corner_inmem.rs` ACTIVE). If a new leak lands here, the canonical altitude is pave-block attachment at phase-FF/make_blocks — every face-splitter-level attempt broke calibrated chains |
| **v1 fillet deprecations entangled with the public wasm API** | `try_fillet` still reaches deprecated `fillet`/`fillet_rolling_ball`; migrating changes public behavior — a product decision, not safe cleanup. See `fillet-blend`, `wasm-bindings` |
| **crates.io / GTM items** | Andy-only. Publishing infrastructure works (see MEMORY.md for the release-please `continue-on-error` masking gotcha) |

## Closed: root cause + where the detail lives

One line each; the fixture/PR carries the story. Newest first.

- **bp 6x4 magnets residual + every baseplate row (CLOSED 2026-08-13 on released 3.2.38; row 775 vs 1383ms = 0.56x, was 1.10x; aggregate 0.45x, faster 25/26)** —
  the #1488-era "no dominant stage" reading had rotted: pocketsCut owned the
  deficit (825 vs 466ms wasm). The 24 pitch-aligned pockets touch rim-to-rim
  (38 grid adjacencies), `fuse_n`'s welded union is genuinely non-manifold so
  the by-edge-id gate rejects it, and `fuse_cluster` fell back to 23 pairwise
  accumulator fuses: ~24 wasted GFA runs, 501ms native for a 31ms cut. Fix
  #1590: contact-thin compound-cut shortcut (every pairwise tool-AABB
  intersection at most 100·tol thick in some axis → combine tool shells
  verbatim, cut once; interpenetrating pairs like the coaxial magnet+screw
  drill take the fuse ladder; fallback taint falls through unchanged).
  Lifted bp 2x2 plain to 0.18x and bp 4x4 plain to 0.16x as collateral.
  Fixture `bp64_pocket_compound_cut_inmem.rs`; instrument `BK_GFA_TIME`
  (native-only per-stage wall clock, the tool that separated merge waste from
  arrangement cost). lightweightFloorCut's 253ms native is the genuine
  big-base cut, not merge waste — the remaining lever on this row if ever
  needed.
- **2x2 label bracket scenario-first row (CLOSED 2026-08-13 on released 3.2.37; matrix row 34 vs 66ms = 0.52x, was 2.3x slower)** —
  the tool's redesigned bracket (finger strips in the cavity-wall plane, changed
  since the Aug 9 #1510 capture) made its wall cuts ride collinearly on the top
  ring's inner boundary; the hole weave's whole-window midpoint sat exactly ON
  the hole polygon and the on-boundary ray-cast verdict flipped between the
  mirrored walls, so the +x corner lune never split out of the ring: exact fuse
  open (3 free edges) → 121-face mesh fallback → paid TWICE (brepjs
  fuseAllBisect: fuseAll bail discarded 95ms, pairwise redo kept 99ms — 195 of
  the row's 240ms). Fix #1587: collinear-overlap window splitting + explicit
  riding-piece drop in `integrate_holes_plane`, and 2-solid `fuse_all` groups
  take the pairwise contract (a pair has no batch to protect; the bail only
  double-billed degraded pair fuses). Fixture
  `labelbracket_fingers_fuse_is_exact_and_closed`. Same-day 3.2.37 matrix:
  aggregate 0.63x, faster 24/26, nm 0 vs 5; the three volume-FAIL rows remain
  the reference's own double-cover over-count. Durable: only scenario-first
  numbers are kernel comparisons in the labelBracketPerf harness (warm repeats
  are parameter-cache hits on both sides).

- **Tool geometric parity, #1517 (CLOSED at parity 2026-08-13, shipped in gridfinity-layout-tool#3471)** —
  generator suite **0 failed / 2790 passed** (283 files) on 3.2.36; same-day control
  146 failed on 3.2.35; the collapse is #1581's base-fuse island fix, the
  feet-to-base interface every bin scenario shares (verified: 0 raw FAIL lines, no
  snapshot writes, brepjs pin fixed across the pair). The 26-min label-bracket
  timeout runs in 3.85s, 4x4-everything in 2.77s. #1581's kernel story: the
  123-face pocket body x 544-face 16-foot base fuse, 72.7s dirty fallback ->
  ~100ms exact; three roots (promotion-path islands discarded AND hole-matched in
  original winding, gated to the promotion path; SD group demotion exempts members
  not covered by the opposite representative); fixture
  `crates/io/tests/gridbin4x4_feet_fuse_inmem.rs`. The ship bump also moved the
  tool's persisted mesh-cache revision to r2 (stale old-kernel previews evict).
  Head-to-head: 0.63x aggregate, faster on 24/26, 0 non-manifold vs the
  reference's 5. Harness `kernelParityMatrix.test.ts` +
  `scripts/compare-kernel-parity.ts`.
- **#1538 coplanar-interface fuse family (CLOSED 2026-08-13; arc #1554/#1559/#1563/#1567/#1581 over 3.2.29-3.2.36)** —
  six roots: the winding emitters (extrude CW-profile rewind, closed-edge merge/CB
  direction), open-curve windowing, the hole-dangling rescue + rim-tangent union,
  the analytic classifier's chord-sampled holes, and #1581's promotion-path islands.
  Every synthetic mode (`interface_fuse_probe.rs`) and captured chain (circleinsert,
  deepcutout, roundpocket4) is exact and strictly valid; the cornerRadius and 26-min
  label-bracket tool rows are green on 3.2.36. Pins: `interface_fuse_winding.rs`
  (5 tests incl. `partial_overlap_corner_hole_interface_fuse_is_exact`),
  `circleinsert_socket_fuse_is_strictly_valid`, `deepcutout_cut_inmem.rs`.
  Parked: branch `fix/doubled-faces-same-surface-gate` (`remove_doubled_faces`
  groups by edge-ID multiset and silently assumes same surface — a two-quadric
  two-edge lens is the counterexample; semantically right, unexercised post-#1559).
- **#1570 spacer export timeout (CLOSED 2026-08-13; #1573 cone-cone radical plane + #1578 residue roots; tool-side 2.37s on 3.2.35 vs the 46s timeout)** —
  the 3.2.28 "4s baseline" was a silent broken-op3 mesh blob, never a target. Four
  residue roots (in-hole coplanar probes, wholly-in-hole sections, raw-t edge
  sampling in the orientation vote, flag-vs-winding flux arbitration) live in
  `crates/io/tests/spacer_foot_fuse_inmem.rs`. Durable: never trust `is_reversed`
  alone for orientation-sensitive logic; `WINDING_CENSUS` in replay_pair measures a
  solid's flag health.
- **Extrude emitted mirrored wires for CW-wound profiles (FIXED 2026-08-12, the circleinsert pocket-cut winding root)** —
  `extrude` accepted CW-wound profiles by flipping SURFACE normals/rev flags
  while emitting the mirrored wires as-is: a solid whose every wire winds
  against its face flags. NO oracle catches this — pairwise edge opposition
  survives a global mirror, and volume/mesh orientation read surfaces, not
  wires — so the operand is "validation-clean" while GFA's face splitter
  (which trusts effective wire winding) mints same-direction rim arcs in any
  later boolean (the 8 arcs in the circleinsert floor cut; the layout tool
  authors circle profiles CW). Fixed by rewinding profile wires up front
  (outer CCW around the extrusion, holes CW). Repro modes `pocket4`/`pocket4r`
  in `interface_fuse_probe.rs` (identical tools, opposite authoring);
  regressions `cw_wound_extruded_profile_cut_has_valid_winding` +
  `circleinsert_pocket_cut_is_strictly_valid` (real bin base,
  `circleinsert_base.bin`). COLLATERAL ROOT also fixed: `chamfer_builder`
  predicted the trimmer's Left/Right keep-side in a representation-independent
  frame, but the trimmer's frame follows wire traversal — concave chamfers on
  canonically-wound prisms kept the ridge strip and grew the solid (the CW
  emission had masked it). Switched to the fillet builder's
  `TrimKeep::AwayFrom(spine_pt)`, whose side test cancels the traversal
  dependence. `audit_bin` gained `VALIDATE=1` (per-file validate_solid with
  orientation checking).
- **Free-loop cap synthesis double-covered a surviving face (FIXED 2026-08-12, the deepcutout 9-edge residue)** —
  `cap_partial_overlap_free_loops` (builder_solid) capped every closed free-edge loop
  independently. When SD's Cut+same-orientation branch drops BOTH a partially-
  overlapping annulus and the tool cap (the operand's bottom being a legitimate
  two-face coplanar tiling, and SD pairing the WHOLE annulus with a 2.55mm corner
  sliver), TWO loops free up: the outer ring and the kept disc's outline. Independent
  caps produced a hole-less full disc plus a same-sense duplicate of the disc — 9
  same-direction shared edges that volume could not see (the doubled face sat on the
  z=0 plane, zero flux through the origin). Loops on one cap plane are now
  containment-nested (arc-true sampled polygons; a vertex-only polygon misses the
  sagitta bulge, and a reverse-traversed arc must sample its stored span reversed,
  not the complement): contained loops become the container cap's holes. The whole
  deepcutout chain is exact end-to-end; all `deepcutout_cut_inmem.rs` pins active,
  `deepcutout_result_body.bin` refreshed. Instrument: `BK_CAP_TRACE`.
- **Closed-edge winding direction, three emitters (FIXED 2026-08-12, the #1538 interface family's core)** —
  synthetic clean-input probes (`crates/operations/examples/interface_fuse_probe.rs`)
  showed even cut(box, box) through-hole mints same-direction shared edges. Three
  independent roots, all invisible to the free/over census: (1) the internal-loops
  splitter normalized disc/hole winding via a signed area in the surface's own
  parameterization, inverted vs the local frame on a DOWN-facing plane
  (`special_cases.rs`, now frame-projected 3D areas); (2) `merge_duplicate_edges`
  "never flip closed edges" — two coincident circles can parameterize opposite ways
  (quarter-point comparison added; this is the merge's orientation MAP, not the
  terminal merge-key); (3) `rebuild_face_with_cb_edges` collapsed to forward=true
  for a closed rim swapped to its CommonBlock circle (same comparison). Regressions
  `crates/operations/tests/interface_fuse_winding.rs` (rect + circle chains strictly
  valid; coincident-cap pocket pinned ignored). The probe's BK_WINDING=1/2 prints
  per-edge effective directions with owning faces — the fast winding instrument.
  `expand_edge` (fill_images_faces) assumed every image sub-edge of a split boundary
  edge is minted in the parent's direction, but a CommonBlock split_edge is shared
  with the coincident partner solid and keeps THAT solid's direction. A deep corner
  cutout flush with a recessed bin's ledge got a backwards sub-edge: two unclosed
  wires + 11 same-direction shared edges, ops correctly rejected the exact result and
  paid a wrong-volume (+7) all-planar fallback that poisoned the downstream socket
  fuse (#1538's "solid mode with cutout" scenario). Images are now oriented by
  endpoint chaining. Fixture `crates/io/tests/deepcutout_cut_inmem.rs` (active:
  no-fallback + exact volume + closed wires; strict validation still ignored — 9
  same-direction shared edges remain, the family's open residue).
- **Slot cut closed the shelled bin's pocket (#1536 root, FIXED 2026-08-12)** —
  the face splitter's first-vertex hole matching attached the rim annulus's woven
  cavity-mouth loop to the tiny notch rectangle it shares two corners with
  (first-match order + strict ray-cast jitter on an exactly-on-corner probe), so
  the annulus lost its hole (emitted as a full disc) and the mouth re-emerged as a
  same-sense coincident ceiling: closed, manifold, volume 6x. Arc-cornered rims
  only — an all-line rim traces the mouth loop from a different first vertex. Fix:
  area-dominance gate on hole-attach candidates (a hole cannot be carried by a
  region smaller than itself), `builder/face_splitter/mod.rs` "Simple hole
  matching". Regression `crates/operations/tests/shelled_bin_slot_cut.rs`; probes
  `slot_cut_probe.rs` (operations example), `shell_face_census.rs` +
  `cavity_probe.rs` (io examples). The ops-layer trivial-containment shortcut and
  raw GFA both reproduced identically — the shortcut was a red herring.
- **Rim-arc crossings took the short way round (CLOSED 2026-08-10, #1540, the #1538 open shells)** —
  `circle_arc_plane_crossings` (added by #1534) decided which part of a circle a
  boundary edge covers by taking the SHORTER way between its vertices. The kernel has
  one definition and it is not that: `EdgeCurve::domain_with_endpoints` reads an open
  circle edge as the CCW span start->end, a MAJOR arc whenever a band keeps more than
  half its circumference. On a major arc the two readings are complements, so the
  predicate swaps its accept and reject sets — it drops the crossings on the edge and
  returns ones where the edge never goes. The invented crossing is the damaging half:
  it splits a section where no face boundary passes, and the existing midpoint test then
  drops a piece that should have been kept. Regressions
  `phase_ff::tests::{major,minor}_rim_arc_*`. Durable lesson in Recurring traps.
- **Compartments+scoop graze fuse (CLOSED 2026-08-10, #1517 root a)** —
  a thin planar tread meeting a corner cylinder takes a dedicated path,
  `trim_ellipse_to_boundary_crossings`, because the in-both arc is a sub-millimetre
  sliver the generic sampled filters drop. It crossed only `EdgeCurve::Line` boundary
  edges, so it split the section at the tread's boundary lines and the analytic face's
  SEAM lines but never at its RIM arcs. Nothing split the section where the band ends,
  and the single over-long arc kept its midpoint inside the extent's boundary margin, so
  the whole thing survived: tread and cylinder then bounded the same region along curves
  0.687mm apart and the shell came back open (34 free edges, ~45 non-watertight exports).
  Crossing the rim arcs too splits it at the rim and the existing midpoint test drops the
  rest — no keep/drop logic changed. 178 faces, 12 cone / 24 cyl, 0 free. Fixture
  `crates/io/tests/compartscoop_fuse_inmem.rs`, pin un-ignored. REFUTED on the way: the
  "orientation-dominant / 145 same-sense pairs" framing (predated the #1525 classifier
  fix), same-domain (the coincident scoop walls are real but not the cause), and
  `clip_line_to_face_boundary`. Also refuted: adding a conic band clip in the FF
  mutual-overlap trim — it closes the gap but the edges stay free, because the section
  never reached that clip at all. `BK_RESTRICT=1` is what showed the bypass.
- **Lid magnet-post corner fuse (CLOSED 2026-08-10, #1517 root b)** —
  `split_cylinder_band_by_arrangement` reconstructed the cut from the vertical wall
  generators alone, pairing them from the seam into removed rectangles, and used the ring
  sections only to confirm the cut was rectilinear. That models a box notch, where the
  removed sector is the only place a horizontal cut exists. A partner plane that ENDS
  inside the band cuts the arcs where it still exists — the sectors the notch KEEPS — so
  nothing capped them, and everything above stayed welded to the material below. The fix
  feeds the ring sections in as horizontals, taking their u-range from the exact 3D
  projection with the arc midpoint picking the side. The lid fuse went from 47 faces with
  6 free edges (rejected, then a 305-face mesh blob compounding across ~24 later fuses,
  the crash and the 14 timeouts) to 49 faces, 0 free, exact. Fixture
  `crates/io/tests/lidpost_fuse_inmem.rs`, pin un-ignored. Instruments that cracked it:
  `BK_SECEDGE` (exonerated FF — both faces get the same correct sections),
  `BK_SUBFACE_BOX` + `BK_SUBFACE_WIRE` (the kept piece spanned the whole band height),
  `BK_SPLIT_TRACE`.
- **Point classifier counted holes as crossings (CLOSED 2026-08-10)** —
  `classify_point` (both the `check` and `operations` copies) tested a ray hit against the
  face's OUTER wire only, so a ray leaving through the mouth of a pocket counted the ring
  face around it. A plain through-hole is parity-invisible (+2), which is why it survived;
  it bites on a blind pocket or a hole filled by a coplanar neighbour face. An open pocket
  read as solid material, and the wrong reading is deflection-independent, so no probe
  setting exposes it. Regression `operations::classify::tests::point_in_open_pocket_is_outside`.
  This is what made `POINT_IN` lie on the #1517 lid; treat pre-2026-08-10 POINT_IN readings
  on any pocketed or holed solid as suspect.
- **Label-bracket fuse open shell / mesh fallback (CLOSED 2026-08-09, #1510)** —
  `clip_line_to_face_boundary` kept chord crossings in `crossings` and true-arc ones in
  `crossings_ext`; the hole-free branch used both, the HOLED-face fallback only the former, so a
  section on an arc-cornered holed face stopped a sagitta short of the boundary and could not
  separate the piece beyond it (the bin annulus's outer corner chord meets y=40.550 at 39.200
  where the arc meets it at 40.7495). 121 all-planar fallback faces in 34ms became 58 keeping all
  8 cylinders in 11ms, watertight. Tool-side on released 3.2.18 the row moved from 1.60x slower than
  the reference to 1.74x faster (104ms to 39ms against its 68ms) at 1510 to 1072 triangles, the only
  one of the 22 scenarios whose triangle count moved; brepkit reads 0 non-manifold edges there where
  the reference reads 147. Fixture `crates/io/tests/labelbracket_fuse_inmem.rs`;
  instruments `BK_SECEDGE` (per-face clipped section extents) and `BK_CLIP` (chord vs true-arc
  crossings, with 3D points) — reach for those first when a section survives FF but the split is
  wrong.
- **4x4 mag no-lip bin row (CLOSED 2026-08-09 at parity on 3.2.16, report #1510)** —
  396 vs 398ms in-suite, 393 vs 400ms cold median against a same-session reference run; moved by the
  #1502 splice spatial hash (magnet-hole-circle-heavy tessellation), no bin-targeted work involved.
- **#1499/#1508 kumiko cutAll chain (CLOSED 2026-08-09 on released 3.2.16 + brepjs 18.124.2, kumikoProfile green)** —
  false containment EmptyResult (#1501: volume witness; regression `cut_wedge_by_thin_radial_strut_is_not_empty`)
  + four untrimmed-parent-curve consumers (`domain_with_endpoints`) + brepjs#1996 compound-base fan-out
  + #1506 wasm `Instant::now()` panic the 3.2.15 diagnostics introduced (wasm CI builds but never runs — tool-side smoke is the only runtime gate).
- **#1500 warm re-export (CLOSED 2026-08-09 on released 3.2.16: warm 490-503ms vs the reference's 890-906ms, cold 706-740ms vs 1691-1739ms)** —
  circle T-junction splice was O(circle-edges × pool-points), 90% of export tessellation; spatial hash in #1502,
  mesh hash-identical (repro `crates/io/examples/profile_export_tess.rs`, `BK_TESS_PHASES`). Next warm lever if ever needed:
  the tool's one uncached `fuseWithEvolution` (~390ms, noted in the issue).

- **#1488 baseplate perf (CLOSED 2026-08-09 on released 3.2.13, tool-side confirmed)** —
  4-27x behind became 0.84x aggregate across all 22 scenarios; 4 of 6 plates faster than
  the reference kernel, 6x4 magnets residual 1.22x with no dominant stage. Two roots: #1490 (below) and
  #1495 (edge-tangent PREVIEW pockets made every cluster fuse mesh-fallback; the
  all-planar blob poisoned every later boolean; the non-monotonic corner-clip anomaly
  and the plain-slower-than-magnets inversion were both its collateral). Guard
  `compound_cut_edge_tangent_tools_stays_analytic`; full story in the issue thread and
  memory `project_baseplate-graze-perf`
- **FF sampled plane-analytic chains fit unclipped (#1488 kernel side)** — grazing
  plane-cone hyperbolas fed ~512 points to the dense O(n³) interpolate per pair; clipped
  to the face-pair AABB overlap, closed loops stay whole-or-dropped (torus-notch canary).
  #1490, guard `tangent_graze_section_fit_is_clipped`, probe `examples/plate_probe.rs`
- **CDT lift missed constraint-recovery Steiner vertices (#1487)** — crossing splits and
  the bisection backstop mint vertices the caller never saw; masked pre-#1478 by the
  interior-grid resize; panicked and poisoned the wasm kernel. #1489, test
  `cdt_covers_steiner_vertices_from_constraint_recovery`
- **GH campaign #1445/#1446/#1447 (2026-08-08, closed on released 3.2.5)** — slots DCEL
  rescue for non-periodic bands, fillet-v2 campaign (56→0 free edges), pinch-shim SD gate,
  v2 orientation emission, CDT winding vote, pinch-u unwrap, display-density floor
  threading (#1478). Fixtures: `slots_lipcone_cut_inmem.rs`, `scoop_fillet_variable_inmem.rs`,
  `gscoop_pinch_cut_inmem.rs`, four scoop fixtures with orientation pins. Detail: MEMORY.md
  Feature Parity Status + the fixture doc comments
- **Sweep/pipe/miter placement family** — `sweep()` re-centered profiles onto the path
  (lip z-shift); perpendicular profiles now sweep as-positioned across sweep/pipe/
  sweep_with_options/miter; `compute_frames` domain-mapping fixed for split sub-paths;
  analytic spine sweep shipped (#1421/#1427/#1438, releases 3.0.1–3.1.3). `helical_sweep`
  keeps re-centering by contract (`ProfilePlacement::CentroidOnPath`). Pins:
  `*_keeps_offset_profile_position*`, `analytic_spine_sweep_lip_ring_is_exact`
- **Coincident-fuse nondeterminism** — shell_op rim assembly iterated a HashMap for
  boundary edges; wire origin rotated the splitter UV frame run-to-run. One-line sort fix;
  `exact_coincident_lip_fuse_stays_analytic` un-ignored
- **shell_op cavity corner cylinders same-sense** — three coordinated wire/rim orientation
  fixes. #1435, `shelled_rounded_box_is_orientation_clean`
- **Orientation-emission campaign** (loft/revolve/extrude/sweep/blend + splitter winding +
  fuse crescent classification + loft cylinder-arm mint) — check_orientation defaults ON;
  see MEMORY.md for the durable winding rules. #1365-#1377, #1394, #1404
- **Mixed-detail 511 residual** — 395 CDT flip-recovery stall (Steiner bisection in
  recover_edge) + 20 same-sense (#1394 pcurve-fold crescents) + 116 loft cylinder mint
  (#1404); chain verified clean on released 2.129.13. `mixed_socket_tess_inmem.rs`
- **Export matrix drift** — O-shape (ray-cast conflict re-cast, #1357) + slotted no-lip
  (SD cross-shell gate, #1360); 73/73. Fixtures volume-pinned;
  `slotted_nolip_fuse_inmem.rs`, `oshape_socket_fuse_inmem.rs`
- **Mitsukude panel cut** — missing FF section: `sample_plane_cone`'s uniform-u sweep
  aliased past the asymptote; chain ends now extend to the exact v_max boundary.
  Fixture volume-pinned; kumiko-dividers 166.6s → 25.5s
- **Kumiko lattice band fuse — closed after 29 passes** (#1302,
  `kumiko_lattice_bands_fuse_closed` un-ignored). Final mechanism, all in the face
  splitter: (1) DEMAND-GATED outer-wire pave-image expansion (3e-3 near-miss gate, both
  broader gates measured harmful); (2) pendant→boundary-vertex bridge (section-free
  targets only); (3) pendant→pendant bridge (mutually nearest within 3e-3, 10x isolation,
  twin-deduped). The pass-27 "near-coincident slope SD" framing was REFUTED by direct
  measurement (planes 15° apart). Honeycomb residuals re-pinned
- **Kumiko corner wedge coaxial cut** — NURBS boundary chord anchoring: sampled
  sign-change bisection in `clip_line_to_face_boundary` (#1343) + circle-gated NURBS
  boundary-image expansion (#1352). `kumiko_corner_wedge_inmem.rs`, volume-pinned
- **Thick-wall cavity** — two stacked roots in shell_op's collapsed-corner arm (miter fed
  both extreme normals; sharp-corner chamfer strip emitted). All cases bnd=0. Pins:
  `shell_thickness_past_corner_radius_gives_a_sharp_corner`,
  `thickwall_sharp_cavity_fuse_inmem.rs`
- **v2 trimmer residuals** — `dihedral_half_angle` returned the normals' half-angle where
  the material wedge half-angle `(pi-angle)/2` was needed; coincide only at 90°.
  `regress_blend_keepside_tangency.rs` un-ignored; refutation history in fixture docs
- **Bench intersect(corner box, center sphere)** — three stacked roots (outward-normal
  270° complement arcs, same-sense patch wire, planar-polygon containment on a
  non-planar octant patch). `bench_equiv_intersect_box_corner_sphere_is_the_octant`
- **Bench cut(box,cyl) 2.3% deviation** — endpoint-exclusion class in three boundary
  samplers dropped polygon corner vertices; plus unsigned fan areas in check::properties.
  `bench_equiv_cut_box_corner_cylinder_volume_is_exact`
- **Snap-clip deepened notch (both faces)** — cone variant via outer-region section clip
  (#1102); plane variant via `union_internal_loop_with_hole` (all-Line, interaction-gated).
  `deepened_wall_opening_inmem.rs`. Arc-bounded openings still bail by design
- **Divider scenarios 15/15** — the 3 historic defects closed by the kumiko+blend
  campaigns; the brepjs `applyMatrix` dist-patch stays tool-side BY DESIGN (brepjs pins
  the cache-alive contract) — re-target it on every brepjs bump
- **Wall-pattern honeycomb/triangle defects** — both tool-side (stamp keep-out, band
  layout); kernel exonerated. Tool #3294
- **Cone/cylinder ∪ box tangent section circle** — closed as collateral of #1357+#1360.
  `tangent_wall_fuse_configurations_stay_analytic`
- **Torus ray-cast arm** — `FaceGeom::Torus` + `math::intersect_line_torus`; TWO-RIM tube
  bands decline by design. `whole_torus_classifies_inside_and_outside`
- **Kumiko corner cut** — 4 roots (band rescue, graze scaling, chord-represented NURBS
  boundaries, reverse-twin misread). `kumiko_corner_window_inmem.rs` (fixtures gone with
  the parked branch; see OPEN)
- **Six-tool corner residual** — edge-midpoint fallback seed on a grazing ray;
  `interior_of_notched_polygon_clears_the_boundary` (pins verbatim f64 literals)
- **Segmented revolve inverted solids** — winding normalized; new oracle
  `measure::oriented_solid_volume` (plain `solid_volume` is a magnitude)
- **Arena `reserve` doubling** — bulk hint held both buffers, aborted the 4 GB wasm heap.
  `topology/src/arena.rs`
- **GFA multi-region acceptance** — rotated-bar AABBs, ring Euler surplus, ray-parity
  nesting. #1239
- **FF AABB pre-filter aliasing on straight sections** — exact slab-clip, gated to
  quadric partners. #1224, `goma_wall_band_cut_inmem.rs`
- **Tessellation nested-hole seeding** — centroid seeds identical for concentric wires;
  odd-depth rule. `oring_nested_holes.rs`
- **T-lip band cut** — depth probe overshot a 1.2 mm annulus. `lipband_cut_inmem.rs`
- **Label-sockets tab attach** — interior sampling blind to end overhang.
  `labeltab_attach_inmem.rs`
- **Intwidth wall tangency** — two solvers ±1e-6 apart on tangential intersections.
  `intwidth_tangency_inmem.rs`
- **Lite magnet-pad graze fuse** — graze heuristic keyed to face extent is blind to
  corner-window exits. `lite_pad_graze_fuse_inmem.rs`
- **Mesh-boolean co-refinement rewrite** — T-junctions, coplanar collapse, winding
  coin-flips. `relief_meshbool_fallback_inmem.rs`
- **Trimmed-torus ray-cast** — 3 stacked roots. `check/src/classify/ray_surface.rs`
- **Dovetail family** — `crates/io/tests/dovetail_*.rs`, `fracplate_seam_pocket_inmem.rs`
- **halfSockets / fractional-width / socket-assembly family** — `halfsockets_*.rs`,
  `fracwidth_corner_crescent_inmem.rs`, `socket_assembly_fuse_inmem.rs`
- **snapClip + fit-offset family** — `snapclip_*.rs`, `fitoffset_groove_mouth_inmem.rs`
- **Kernel-poison panic surface** — wasm32 is `panic=abort`, `catch_unwind` is INERT, a
  trap strands the borrow flag (recovery = new `BrepKernel`). Panic text survives via
  `crates/wasm/src/panics.rs`
- **Divider + floor pattern families** — 18 of 21 failures were ONE missing brepjs
  adapter method. Not a brepkit defect

## Refuted: do not re-try

- **Clip-level inner-wire trimming in `clip_line_to_face_boundary`** — even
  endpoint-only single-window trimming regressed three foils (groove_chain, dovetail
  a1corner holecut, exact_coincident_lip_fuse); the holed-face weave requires
  untouched sections (#1563). Do not retry any clip-level variant.
- **Vote-layer rules for on-plane samples in the ray-cast classifier** — the
  honeycomb and circleinsert flip points are formally inseparable at that
  resolution; four rule variants each broke one side. #1567 fixed the real root
  with analytic circular holes instead.
- **A universal smarter merge-key for duplicate edges** — see TERMINAL; unbuildable.
- **Placing the thick-wall collapsed corner EXACTLY** (`nᵢ·(x−C) = radius − thickness`) —
  geometrically right, measured WORSE (20 → 318/544); moot with the chamfer strip shipped.
- **Ungated pendant-chain bridging in the face splitter** — fires at healthy corners and
  over-connects (use-3); the shipped version's three gates (section-free target, mutual
  nearest, isolation) are each load-bearing.
- **Cluster-canonical vertex adoption in `JunctionRegistry::resolve`** — net-negative;
  consumers that bypass endpoint resolution keep their own anchors.
- **Narrowing the DCEL-rescue gate by kumiko loop signature** — loop shape does not
  separate the corner wall from goma's bands (goma byte-identical under it).
- **"The constraint is the SPLIT ITSELF disturbing reconciliation"** — the newly-admitted
  splits were straight axis-aligned runs stored as NurbsCurve; a span-local sagitta gate
  fixes it.
- **`shell_is_outward_oriented` / `signed_volume_of_shell` being inverted** — both exact
  on a known-good cube (`flux_orientation_probe.rs`); the operand really was inward.
- **The goma odd bands as a GFA defect** — they were brepkit's own mesh-fallback output.
- **Ellipse aliasing at FF filter 2 on the goma lump** — a genuine 2× separation, not
  aliasing.
- Also refuted, each once: coincident coaxial cylinders as the corner-cut root; a
  classification error there (independent oracle agreed with GFA); plane-gated seed
  correction; arc-cornered wires as the nested-hole trigger; helix sweep as the goma
  cause (`helical_sweep_is_watertight_across_turns_and_segments`); upstreaming the
  brepjs intersectCurves eager-release (brepjs pins the cache-alive contract).

## Recurring traps (the distilled, expensive lessons)

- **A circular edge has ONE canonical span and it is not the short one.**
  `EdgeCurve::domain_with_endpoints` is the CCW range from start vertex to end vertex,
  routinely a major arc. Any new "which part of the circle does this edge cover" test
  must call it or reproduce it exactly (#1540). Taking the short way does not merely
  lose crossings, it invents them on the complement.
- **Validation-clean is not winding-clean.** A globally mirrored solid (every
  wire wound against its face flags) passes `validate_solid` WITH orientation
  checking, positive oriented volume, and a clean directed mesh — pairwise
  opposition and surface-derived normals all survive the mirror. The only
  symptom is a downstream boolean minting same-direction shared edges from
  "clean" operands. When that happens, audit the operand CONSTRUCTOR's
  winding emission first (`extrude` shipped mirrored CW-profile prisms for a
  long time), and remember the trimmer/splitter Left/Right frames follow wire
  traversal — never predict them from geometry alone.
- **A whole-solid volume cannot tell a MISSING cavity from a COLLAPSED one** — both read
  high by the same amount. Print per-shell signed volume against each shell's own bbox
  (`cargo run --release --example cavity_probe -p brepkit-io`) before blaming either the
  boolean or the measurement (#1536).
- **A hole can only be carried by a region larger than itself; a shared-corner probe
  fakes containment.** The splitter's first-vertex hole matching + first-match order
  handed a 6577-area mouth loop to a 1.26-area notch because the traced loop STARTED
  at a shared corner and the strict ray-cast read it inside (#1536). When auditing
  hole attachment, check area dominance before trusting any point probe. Per-face
  signed-volume census (`shell_face_census`) is the cheap way to spot the resulting
  same-sense doubled cover: one face's contribution has the wrong sign for its shell.
- **Measurement and tessellation walk different face sets.** Tessellation uses
  `explorer::solid_faces` (outer + inner shells) so the MESH is complete, while several
  volume/area/CoM paths walked `outer_shell()` alone — closed, manifold and correctly
  bounded, with the cavity silently absent from the number. Fixed for volume, area and
  centre of mass; when adding a solid-scoped measurement, use `solid_faces`.
- **Never compare closed-curve directions through their parameter frames.** A closed
  circle's `domain_with_endpoints` anchors at the curve's own reference direction, so
  evaluating two coincident circles at matching parameters compares unrelated angles
  (a quarter-point test read opposing circles as same-direction). Compare TANGENTS at
  a shared 3D point (`closed_curves_same_direction` in fill_images_faces).
- **Marched/fitted section geometry is good to ~1e-6; every exact-tol (1e-7) gate it
  meets needs a weld-scale (100·tol) band.** Four separate gaps in one family were this.
- **A sampled proxy gated at an exactness tolerance** is the single most common defect
  shape here (five instances): `best_d` bounded by sample SPACING, 16-sample AABB scans,
  uniform-t restriction, chord polygons under-covering by a sagitta.
- **Interior points of notched/symmetric sub-faces land on feature-plane intersections BY
  CONSTRUCTION** — classification must survive on-plane samples, and a seed must be
  STRICTLY interior. A centroid is not an interior point for concentric or non-convex
  wires.
- **When classifying which side of a face carries material, sample the face INTERIOR
  offset along its own normal; an edge or vertex point is never valid** (at a convex edge
  both sides read empty), and offset/deflection stability does not rescue a wrong sample
  point. For a non-convex open shell neither a bbox centre nor a vertex centroid is a
  valid interior sample.
- **The face splitter is a web of mutual calibrations.** Run ALL foils on any change:
  d4 gridfinity, honeycomb pcut1/pcut3, divider-lip, groove-mouth, junction-disc,
  cylinder-slot, a1corner. Each caught a different wrong discriminant.
- **A trigger keyed to a post-hoc failure signature cannot demote a working case** — the
  cheap way past those calibrations.
- **When a point-classification oracle disagrees with the face list, distrust the oracle
  first.** Read the operand with `dump_solid` and check its volume; both are independent of
  the classifier. A whole roadmap entry once recorded an "unexplained asymmetry" that was
  purely a classifier defect, and it steered two passes at the wrong question.
- **`solid_volume` is a MAGNITUDE**; only `oriented_solid_volume` sees an inverted or
  doubled shell. It is also translation-VARIANT on a malformed boundary, which needs no
  second oracle.
- **The by-edge-id manifold gate is BLIND to position-duplicate faces and edges.** "GFA
  validated OK" never proves watertight; use the position-quantized check.
- **All-planar output with zero curved faces, on a shape that should have cylinders, is
  the fallback tell** — but weak where the construction is legitimately planar.
- **Never replay a captured operand without printing its free/over counts first.**
  Captures can be fallback-poisoned; a whole iteration has been spent inside that trap.
- **Every `BK_*` knob is NATIVE-ONLY** (`std::env::var` returns Err on wasm32).
  `setLogLevel` + a JS ring buffer is the only handle on kernel internals from JS.
- **`log::debug!` in `fill_images_faces.rs` does not reach a custom logger** that
  receives `builder_solid`'s fine — probes there read as false zeros.
- **A fast-failing scenario family is a signal the failure is PRE-geometry.** Nothing
  doing real geometry fails 9 of 11 cases in 5.4 s.
- **Read raw log lines, not summary counters.** A capture regex missed
  `GFA boolean failed … falling back` and reported 0 rejections while 12 were present.
- **Verify the instrument fired, and verify which binary/branch a measurement came
  from.** `cargo build --tests` does not rebuild examples (stale-binary readings); a
  "compare against the parked branch" experiment was already answered because the
  measured kernel WAS that branch.
- **In nondeterminism digs, dump OPERANDS first.** Differential dumps at stage boundaries
  walk a flip upstream, but operand-construction ops (shell, extrude, sweep) are as
  suspect as the boolean — the coincident-fuse root was a HashMap iteration in shell_op.
- **Noise can be born at EMISSION, not intersection.** Probes at the phase level all lied
  once; the recipe that cracked it was an env-gated backtrace in `Vertex::new` on the
  literal coordinate.

## Tool-side measurement recipes and traps

- **Scenario numbers rot.** Always run the control on the SAME DAY and SAME catalog; a
  stale baseline has twice nearly produced a false conclusion. Confirmed again 2026-08-10:
  the issue's "137 failed on 3.2.18" re-measured as **154** on the same kernel, because the
  catalog had grown. Quote a delta between two runs you did yourself, never against a
  recorded number.
- **The reference kernel's volume is NOT an oracle where its own mesh is non-manifold.**
  Three of four volume "failures" in the head-to-head matrix were against reference meshes
  carrying 970-1809 non-manifold edges (Euler 1230-2375) reading HIGHER than brepkit's clean
  ones — the double-cover over-count. Check `nonManifoldEdges` on BOTH sides before believing
  a volume delta.
- **A per-kernel `.brepkit.snap` triangle count is not a defect signal.** Any correct change
  to how a face splits moves it. 11 of 16 apparent regressions in the 3.2.22 run were stale
  baselines; separate them out before counting, or refresh them first.
- **Measurement worktree recipe** (2026-08-10, worked end to end): `git worktree add
  .worktrees/<name> origin/main --detach` inside the TOOL repo, `pnpm install`, edit the
  `brepkit-wasm` pin in its `package.json`, `pnpm install --no-frozen-lockfile`. Verify with
  `require.resolve('brepkit-wasm', {paths:[require.resolve('brepjs')]})` plus a sha256 of the
  `.wasm` — resolution THROUGH brepjs is the part that catches a nested copy. Drive
  `./node_modules/.bin/vitest` directly. `--reporter=basic` is not a vitest 4 reporter and
  fails as a missing custom module. A full generator run is ~45 min at `--maxWorkers=4`.
- **Current baseline (2026-08-07, released 2.129.13 era, stock pins): the ENTIRE tool
  generator suite is GREEN — 272 files, 2720 passed, 0 failures.** Compare against a
  fresh same-day run, not old counts (the catalog grows continuously).
- **Overlay verification is mandatory and non-obvious.** Hash the file that
  `require.resolve('brepkit-wasm', {paths:[require.resolve('brepjs')]})` returns, run
  FROM the directory vitest will use. The foils worktree has its OWN `node_modules`. For
  a brepjs-side change, `npx vite build` then copy `dist/*`; methods live in
  content-hashed `shapeTypes-*.cjs` chunks, so grepping `brepjs.cjs` reads as "fix
  missing". The vitest resolve.alias does NOT reach the CJS require path — overlay
  node_modules and hash-verify, or you silently bench the installed kernel.
- **`pnpm exec vitest` triggers a dep check that wants to PURGE `node_modules`**
  (destroying any overlay). Drive `./node_modules/.bin/vitest` directly.
- **`vitest run --project generators` EXCLUDES `__kernel-tests__`** — those need
  `--config vitest.profile.config.ts`. Vitest does not surface `console.log` through a
  pipe: write probe results to a FILE.
- **Capture recipe:** wrap the RAW kernel's boolean entry points from a tool probe and
  `serializeSolid` each operand. A hook on `fuse` alone fires ZERO times — exports drive
  `fuseWithEvolution`/`cutWithEvolution`, scoops drive `filletVariable`, and much traffic
  goes through `executeBatch` (flatten batch ops when capturing). `compoundCut` passes
  tools as a Uint32Array — `Array.isArray` misses it, `ArrayBuffer.isView` is required;
  a number-only argument filter captures the base and silently drops every tool. Replay
  with `crates/io/examples/replay_pair.rs` (`A=`, `B=`, `OP=`, `TOOLS=<paths>` for
  compound cuts) or `replay_cut_capture.rs`.
- **A multi-case tool probe MUST make a fresh kernel per case, or run one case per
  process.** The kernel is a per-worker singleton whose borrow flag strands permanently
  on a trap; the first failing case poisons every later one. Cheapest fix is a `CASE=`
  env selector and one vitest invocation per case.
- **Do not compare a standalone probe number against a suite number** — in-matrix runs
  are cache-warm; the same scenario has measured bnd=0 standalone vs bnd=6 in-suite.
- **Tool probes under `__kernel-tests__` are UNTRACKED and get cleaned.** Budget for
  re-writing one. The tool is a SEPARATE repo other sessions commit to concurrently —
  check its `git status` before running anything there.
- **The foils worktree can vanish mid-session.** The probes survive on branch
  `diag/brepkit-kernel-foils` (local and origin); restore with `git worktree add`; it
  needs its OWN `node_modules`. Do NOT run measurements in the main tool checkout.
- **`brepkit-render`'s `compute_mesh_lod` SIGSEGVs intermittently** (pre-existing, ~50%
  of runs), aborting `cargo test --workspace` early and masking later suites. Use
  `--exclude brepkit-render`. Also: `cargo test --workspace` is fail-fast per binary —
  use `--no-fail-fast` when counting failures.
- Scenario snapshot tests pin EXACT reference-kernel triangle counts; a different kernel
  can never match them. Received-below-expected is benign density difference,
  received-10x-above is a defect.
- **Durable native probes/instruments** (env-gated, grep for them before writing new
  ones): `BK_FF_DUMP` / `BK_FF_TRACE` / `BK_RAWC` (phase_ff), `BK_SD_SETS`
  (same_domain), `BK_RESTRICT` (phase_ff — the in-both window each section is trimmed to, and
  crucially whether a section reached that clip at all: a special-case emitter that bypasses
  it looks exactly like a window computed wrong), `BK_OPEN_SHELL` / `BK_SHELLS` (builder_solid shell grouping),
  `BK_SUBFACE_SRC` / `BK_SUBFACE_BOX` (builder — note BOX tests face VERTICES only) and
  `BK_SUBFACE_WIRE` (adds each sub-face's wire with per-edge curve MIDPOINTS, the only way
  to tell a short arc from the long one sharing its endpoints on a periodic wall),
  `POINT_IN` / `FREE_EDGES` / `TESS_BND` modes in `replay_pair`, `dump_solid` (per-wire
  edge ids), `audit_bin.rs` (HALFEDGE directed oracle — the authoritative winding
  oracle), `orient_scan.rs` / `fuse_orient.rs`, fillet instruments (`BK_FORCE_V2`,
  `BK_PIECES`, `BK_CORNER_TRACE`, `BK_TRIM_TRACE`, `BK_SPLIT_PREPASS`, `BK_NOTCH_TRACE`).

## Subsystem trap notes (crates without their own skill)

- **`validate_solid` mis-reports a multi-component shell as an Euler error.** A 2x2
  socket assembly is 4 disjoint feet in ONE shell (V-E+F = 8, correct at 2 per
  component); the validator expects 2+L. The ops boolean gates handle this
  (`euler_multi_ok`), the standalone validator does not — never "fix" a fixture to
  satisfy that report, and never gate a multi-foot operand on `is_valid()`.
- **The free/over edge census cannot see winding damage.** A result can read free=0
  over=0 with same-direction shared edges the validator counts (the #1538 pocket
  cuts mint 8-9). `validate_solid`'s orientation check or the halfedge oracle
  (`audit_bin`) are the instruments; `replay_wire_audit` prints unclosed chains and
  same-direction positional pairs directly.

- **heal `fix_duplicate_faces` IS implemented** (solid-scoped,
  `crates/heal/src/fix/solid.rs`, returns `Status::DONE2`), not a no-op stub. It
  compares only centroid, normal, and edge count, so it can miss true-but-differently-
  wound duplicates. Verify current state before quoting either way.
- **heal, offset, and sketch have no distilled campaign knowledge.** They follow the
  same `debugging-doctrine`, but no skill covers their internals. Treat any diagnosis
  there as first-of-kind and write findings down.

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
- Do NOT close, defer, or discover an item and leave this skill unchanged — and when
  closing, DELETE the dig log rather than appending to it.

## Related skills

`analytic-preservation` (the chase filters in depth), `parity-benchmarking` (the
scenario re-probe and head-to-head), `debugging-doctrine` (before any multi-pass dig),
`solid-verification` (the acceptance oracles), `testing` (fixtures and ready-repros),
`fillet-blend` (the blend traps), `release-flow` (shipping a user-facing close).
