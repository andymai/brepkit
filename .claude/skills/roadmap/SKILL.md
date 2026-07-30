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
- **Snap-clip deepened-notch case — NO LONGER TERMINAL; both faces of it are closed.**
  The cone-face variant closed via the outer-region section clip (the #1102 dig). The
  plane-face variant (a later cut's internal section loop OVERLAPPING an existing wall
  opening — the snapClip join-edges export root) closed via the deepened-opening union
  in `split_face_with_internal_loops` (`union_internal_loop_with_hole`, all-Line +
  interaction-gated, bails to prior behavior on any chain failure; fixture
  `crates/io/tests/deepened_wall_opening_inmem.rs`). Detection is geometric overlap in
  a locally-built frame — no heuristic. Arc-bounded openings still bail; extend the
  union to arcs only when a repro demands it.
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
| **Tangent section circle on a quadric lateral vs. box walls** — surfaces in the census as `cone ∪ box`, the ONLY remaining primitive-boolean fallback (repros `boolean::tests::cone_union_box_should_be_analytic` + `diag_cone_box_tangency_sweep` / `diag_cylinder_box_tangency` / `diag_tangency_count` / `diag_tangency_epsilon_band`, all `#[ignore]`; their doc comments carry the detail) | algo/GFA | Root-mapped 2026-07-24 (#1213). NOT cone-specific — a plain cylinder r=4 vs a tangent d=8 box fails identically. TRIGGER IS THE COUNT of tangency points: 0 or 1 is CLEAN, 2 gives 4 free edges, 4 gives 4 nm. FAILURE BAND is sub-tolerance only (gross crossing and any clearance both work), because `intersect_segment`'s near-tangent collapse returns ONE hit per wall — that collapse is CORRECT and load-bearing for the intwidth closure, do NOT touch it; the box-side splits are also REQUIRED (its square-minus-inscribed-circle region is genuinely pinched). The defect is the quadric side treating >=2 touch-splits as an open chain instead of one closed rim. CAUTION: the 2-wall and 4-wall cases are NOT proven to share a root — at 4 the quadric face survives mis-wired (outer wire == inner wire), at 2 it is DROPPED entirely. FIX SHAPE: add "inner wire whose edge set duplicates the outer wire's" as a `greedy_broken` trigger so the existing `split_cylinder_band_by_arrangement` rescue fires (face_splitter/mod.rs ~5007). Pre-validated: that signature appears on the 4-wall case and NOWHERE else across the brepkit-io fixtures or brepkit-operations lib tests, so it cannot demote a passing case — but it is INCOMPLETE, the 2-wall case carries no such signature and needs its own trigger. |
| **Ellipse sections spanning > π are outside the shorter-arc contract** (no repro yet — theoretical, flagged in #1150 review) | algo/GFA | `evaluate_edge_at_t` applies the shorter-arc convention to ALL open ellipse edges, and `find_splits_on_section_ellipse` (#1150) matches it — a genuine junction on a > π ellipse section arc is skipped, but every split site would mis-evaluate it anyway. Circle sections are ≤ π by construction (the FF closed-circle emitter splits longer spans); ellipse sections carry no such guarantee. Durable fix: enforce the ≤ π split at ellipse-section emission (restrict/window machinery), mirroring the circle emitter. First probe: instrument window emission for ellipse spans > π on a large-cutout replay. |
| **Compartment manifold roots (six) — CLOSED; the 13/13 tool score was measured on pre-loft geometry (live matrix: fractional-width row below)** | algo/GFA | Two roots CLOSED: the grazing-EF lip-corner vertex (`phase_ef` angle-scaled endpoint window, `crates/io/tests/lipcorner_tangent_inmem.rs`) and the boundary-re-trace section family (`section_on_existing_boundary` in `fill_images_faces.rs` + straightness-aware hole weave + crossing-midpoint hole probes in `face_splitter/mod.rs`; fixtures `crates/io/tests/lipfuse_boundary_retrace_inmem.rs` — that fix un-masked 3 halfSockets-tilt cases that only "passed" via a watertight mesh fallback, then fixed them for real). The retrace guard's discriminant: an exact whole-edge duplicate section is KEPT (threading it routes the face through the split/rebuild that aligns coincident-face partitions — dropping it regressed the plain shelled-cup lip fuses d3/d4/d5 to mesh fallback), while a SUB-SPAN re-trace (45deg-split half-arc, straight run split at a divider crossing) is dropped. A third root CLOSED (chord-sagitta classifier seed, `find_point_outside_holes` in `face_splitter/containment.rs`; fixture `crates/io/tests/halfsockets_clipcut_inmem.rs`): the halfSockets base-clip cut's 1.2mm ring floor got its seed in the corner-arc sagitta gap of the chord-approximated hole polygon → ring classified Inside → open shell → mesh fallback poisoned the whole export chain. Closed `2×2 crossing tilts` outright and took `2×6 halfSockets ±40` 26→1 NM; 10/13 pass. A FOURTH root CLOSED (corner-crescent hole promotion, `loop_containment` in `face_splitter/mod.rs`; fixtures `crates/io/tests/socket_assembly_fuse_inmem.rs`): the bin×socket-assembly fuse at the z=5 base interface leaves a ~0.1mm crescent of bin bottom overhanging each corner socket's chamfered outline; the wire builder hands the crescents back as CW loops, the hole-promotion pass's SINGLE interior probe slipped across the thin boundary into the adjacent socket-square outer, so the crescents stayed "holes", were first-vertex-matched into nothing, and got dumped onto an arbitrary first sub-face that same-domain-dropping then erased — free edges at all four bin corners; the compartments variant went further into a GFA-reject + non-manifold mesh fallback. Fixed by whole-boundary containment (promote only loops with points STRICTLY outside every outer; boundary-coincident re-trace loops must stay holes — promoting or dropping them un-threads the d3/d4/d5 shelled-cup lip fuse). Closed `1×4 2×8-comps` (now analytic, 5× faster) and `1.5×6 no-halfSockets ±40`; 12/13 pass. A FIFTH root CLOSED the family (hole-winding normalization, `split_face_2d` in `face_splitter/mod.rs`; fixture `crates/io/tests/halfsockets_lipfuse_inmem.rs`): the halfSockets body's cavity cut emits its top-ledge hole wire wound the SAME way as the outer wire; `integrate_holes_plane` trusts stored orientation, so where the lip's inner profile crossed the hole's divider diagonal mid-span, the angular wire builder traced a double-cover — a membrane across the bin throat (kept) + the real throat-ledge region wound CW (erased) → free=11 propagating into the final socket fuse's 1 NM edge. Fix: normalize inner-wire winding opposite the outer in UV at the splitter entrance. `2×6 halfSockets ±40` closed; **compartments 13/13 (pre-loft geometry)**. Detection heuristics for the mis-weave (residual-CW-hole triggers, area balance, containment) all FAILED to separate it from the load-bearing re-trace weaves (d4, honeycomb pcut3) — the winding NORMALIZATION at the input was the only clean cut. Probe recipe: instrumented-kernel capture in `all` mode (hooks `boolean_with_evolution` — the tool's export fuses ALL go through the provenance path, invisible to a `boolean()`-only hook) + `VERTEX_WATCH` backtrace trap in `Vertex::new` (probe branch `probe/boolean-capture-2`, rebased onto post-#1045 main). CAVEAT: 13/13 was measured with the PRE-loft faceted sockets; the analytic sockets (#1045) changed every bin's base geometry and un-masked a NEW family (row below) — the 13 closed roots stay closed (their captured chains replay clean), but the tool matrix number from that era is historical — the six closed roots are the durable claim, not the 13/13. |
| **halfSockets loft faceting — CLOSED (#1045)** (`binGenerator.scenario.halfSockets`; the old "zero-triangle" read was a MISREAD — the scenario runner logs `triangleCount:0` for ANY failure) | operations/loft + algo | TWO stacked roots. (1) Every gridfinity socket loft came out ALL-PLANE (z-histogram: 1.2–2.5% bin-volume deficit entirely in the z0–5 feet); loft fix LANDED (recognize NURBS profile edges back to analytic + reverse downward-stacked CCW sketches instead of bailing; arc reversal must NEGATE the circle normal; unit tests in `crates/operations/src/loft/tests.rs`). (2) The fix un-masked the disconnected-loop arrangement defect (row below, CLOSED in #1043): the hs2×2 socket fuse showed bnd=314 + −13% mesh volume, initially misdiagnosed as a wire-orientation/traversal bug. With both fixed, both hs capture chains replay fully clean (every op bnd=0 nm=0, analytic). Tool-verified post-merge (#1045): halfSockets suite 8/11 — the 3 fails are kernel-pin snapshots (benign); brepkit triangle counts now run ~45% ABOVE the reference pins (7512 vs 5176) because analytic feet replace sparse faceted planes — possible tessellation-density follow-up on small socket cones, not a defect class. Remaining halfSockets-suite work lives in the fractional-width row below. |
| **Arrangement disconnected-loop twins — CLOSED** (fixture `crates/io/tests/halfsockets_socketfuse_inmem.rs`) | algo/GFA | A closed section loop strictly inside a plane face (touching neither boundary nor other sections — halfSockets interior socket outlines on the bin bottom) is a DISCONNECTED component of the arrangement trace graph, so its cycle is traced once per orientation. Flat emission (`arrangement_regions_from_inputs`, `even_odd_nesting=false`) shipped BOTH traces (duplicate overlapping discs) and left the containing web region hole-less, geometrically covering them. Same-domain then glued web+duplicates+socket-tops into one group (the hole-less web defeats every `inner_wires()`-keyed guard in `planar_faces_overlap`) and dropped ALL of it; the assembler's cap fill patched the openings with interior membranes → same-direction half-edge pairs on every interior cell rim (bnd>0, nm=0, free=0/over=0 — B-Rep edge checks are orientation-blind). Fix: twin-cycle resolution in flat emission — emit each disconnected loop once, attach its reversed twin as an inner wire of the smallest containing region. DIAGNOSTIC LESSON: bnd-on-both-faces with zero nm reads like a winding bug but can be a REGION-SELECTION bug; map the material first (`classify_point_robust` probes around the rim). |
| **Post-loft fractional-width corner crescent — CLOSED** (fixture `crates/io/tests/fracwidth_corner_crescent_inmem.rs`) | algo/GFA | The bnd=104-per-tilt family root: at each bin corner the analytic socket's r=4 outline circle (tangent to both bin wall lines, new since #1045) and the bin's r=3.75 corner arc bound a ≈0.1–0.25mm sliver on the z=5 bin bottom. The arrangement emitted the sliver region correctly, but `interior_point_3d` built its polygon from the stored pcurves — and a wire can MIX pcurve orientation conventions (section arcs: natural parameterization + traversal flag; boundary arcs: fit in traversal order but carrying the topology flag), so the reversed boundary arcs sampled BACKWARD, folding the sliver into a self-crossing zig-zag whose "interior" point landed in the adjacent socket-imprint region → classified Inside → dropped → 5 unpaired rim edges per corner. Fix: plane-face wire polygons in `interior_point_3d` sample the 3D curves through the `PlaneFrame` (orientation-unambiguous; the #1037 arc-true pattern), never the pcurves; `find_point_outside_holes` hole polygons densified 3→15 interior samples (a single-edge closed bore hole sampled at 4 points is an inscribed square — its sagitta gap accepted annulus seeds well inside the bore, drilled-tube volume regression). ALL six `1.5×6` variants green at tool level. Do NOT widen the fix to the shared `sample_wire_loop_uv`: the split paths consume it and were calibrated against the flag convention (an endpoint-proximity variant changed splits and re-broke the scooplabel over-share pin). The convention mismatch itself (`boundary_edges_to_pcurve` fits traversal-order but copies `oe.is_forward()`) remains — any NEW pcurve-polygon consumer must sample 3D-via-frame instead. |
| **Integer-width halfSockets wall-tangency family — CLOSED; COMPARTMENT MANIFOLD MATRIX 13/13 ON ANALYTIC SOCKETS** (fixture `crates/io/tests/intwidth_tangency_inmem.rs`) | math + algo/GFA | The nm=76/136/140 + `1×4 2×8-comps` nm=12 family (the "all bnd=0" note was a misread — the nm assert fires before bnd; the export actually had bnd=788 too). NOT an SD/duplicate-face root: a half-socket outline's r=4 corner circles are exactly TANGENT to the bin wall lines, and the outline's straight runs continue along those walls from the tangency points, which exist as exact operand vertices. Two solvers recomputed those tangential intersections ±1e-6 off (positional error ~ sqrt(2r·residual); a 1e-13 residual at r=4 = a full micron): (1) `Circle3D::intersect_segment` solved the near-tangent quadratic into a root pair straddling the foot — hit by both phase EE and FF's `closed_circle_boundary_crossings`; (2) phase EF's grazing edge×surface refinement landed anywhere in the tolerance WELL (surface distance grows only quadratically around a tangency). The micro (~1e-6, above vertex-merge tol) line edges were used by 3 faces (one out-and-back on the bin-bottom web) → analytic fuse failed the non-manifold gate → mesh fallback whose own output was non-manifold (nm=76 exported). Fix: (1) near-tangent root collapse in `intersect_segment` — when the chord implies sub-tolerance penetration (disc ≤ 2r·tol·a), emit the well-conditioned double root (the foot); (2) EF tangential mid-edge junction snap — `find_nearby_pave_vertex_widened` (angle-scaled window, linear scan; the spatial index stencil only covers tol-radius) gated on the vertex lying ON both the surface and the edge curve. Final fuse: 891 analytic faces, watertight, ~40× faster than the fallback it replaces. VERTEX_WATCH recipe: watch the tangency coordinate, get every minting backtrace in one run — found both roots in minutes. |
| **Mesh-boolean fallback non-manifold output — CLOSED (2026-07-10)** (fixture `crates/io/tests/relief_meshbool_fallback_inmem.rs`) | operations/mesh_boolean | The safety-net co-refinement itself emitted open/non-manifold meshes on coincident-wall contact (relief-cut pair: raw bnd=11, export bnd=15 nm=1; the intwidth nm=76 export came FROM this path): the splitter fan-split each triangle without propagating on-edge points to the neighbor sharing that edge (T-junctions), coplanar contact was collapsed to a single longest segment, and winding-number classification coin-flips at winding=1/2 on shared walls. Rewritten as conforming co-refinement: per-host CDT re-triangulation with cross-triangle edge-point propagation, mutual coplanar edge clipping (`coplanar_corefine_segments`), and explicit `OnSame`/`OnOpp` coincident-surface classes in assembly (A owns the kept copy). The issue-#696 planar-midpoint-drop metadata path (`mesh_boolean_with_metadata`) is deleted — conforming splits subsume it. `MeshBooleanResult` now self-reports position-welded bnd/nm counts and `mesh_boolean_fallback` warn-logs a non-manifold fallback result instead of consuming it silently. |
| **Honeycomb+handles kernel-poisoning panic — NOT REPRODUCIBLE on 2.124.13 (2026-07-10); panic-capture hardening shipped** (`binGenerator.scenario.combinedFeatures`) | wasm/operations | Full-suite faithful-order overlay run: zero panics, zero "recursive use of an object"; back-skip PASSES structurally (7167 tris, 106s), handle holes 86s; back-skip re-confirmed same day on an independent second overlay run (1/1 structural pass, ~154s wall). AUDIT FINDINGS (durable): wasm32-unknown-unknown is `panic=abort`, so `catch_unwind` is INERT on the real target — the 4 manual wrap sites (fillet ×2 in `bindings/operations.rs`, `compound_cut` in `bindings/booleans.rs`, fillet in `bindings/batch.rs`) + the unwired `#[wasm_binding]` macro (zero usages, references a nonexistent `reset()`) cannot prevent borrow-flag poisoning; a trapping panic locks the object's `WasmRefCell` borrow flag forever and no Rust code can reset it (recovery = new BrepKernel). Shipped: chained panic hook + `lastPanicMessage()`/`clearLastPanicMessage()` free functions (`crates/wasm/src/panics.rs`, installed by `BrepKernel::new`) — the root-cause text survives JS catch-and-continue (mirrored to console.error as `[brepkit] panic:`) and stays readable post-poison. If the family resurfaces, the text now self-reports; dig from there. |
| **Dovetail corner-clip Intersect — CLOSED (2026-07-08)** (`crates/io/tests/dovetail_cornerclip_intersect_inmem.rs`, both tests un-ignored + green) | algo/GFA | Final two stacked roots: (1) the FF-coplanar phase projected the caps' rounded-corner boundary ARCS as straight CHORD sections while the true arcs already existed as FF sections (barrel×cap circle, split at the operand's 225° seam vertex) — the `has_existing_section_at` midpoint dedup can never catch it because emitted arcs store the FULL-CIRCLE bbox (midpoint = circle center); the chord+arc co-endpoint LENS then broke the weave (chord into the outer wire, true arc orphaned as a zero-area slit). Fix: `matching_arc_section_exists` in `phase_ff_coplanar.rs` — skip a Circle boundary edge when its exact arc (same circle + endpoints) already exists as a section. Line edges are NOT skipped (a co-endpoint line/arc pair can be a genuine lens — torus-box in-tube). (2) `find_splits_on_circle` normalized split params against `domain_with_endpoints`, which is ALWAYS the CCW span — for the REVERSE twin of a section pair that is the LONG complement (315° for a 45° arc), so a point on the circle OUTSIDE the arc mapped to interior t (45/315 = 1/7 — the "phantom arc-break" mechanism of the socket-loft diagnosis, now precisely characterized) and `evaluate_edge_at_t` (shorter-arc) minted a phantom vertex, desyncing the coincident caps' partitions (killed the SD pairing that had only worked pre-fix because BOTH caps adopted the same wrong chord). Fix: `find_splits_on_section_arc` — shorter-arc convention for SECTION splits only (sections are ≤π by construction; boundary arcs may exceed π and keep the CCW path — switching them broke the d1/d3/d4/d5 lip fuses, caught by the canary). CAUTION: the two fixes only work TOGETHER (chord-only → free 1→39; shorter-arc-only → free 41). The chord sections were also load-bearing as the coplanar pair's FF-interference link; with clean co-endpoint arcs the caps SD-pair instead. Tool-side dovetail suites re-probe pending. |
| **`fuse_shelled_box_with_socket_loft` — CLOSED (2026-07-10, test un-ignored + green)** (`crates/operations/src/boolean/tests.rs`) | algo/GFA (phase_ff plane×plane clip) | Root (superseding the 2026-07-08 two-defect map): the socket wall facets meet the box bottom plane EXACTLY along their top chords, so every plane×plane FF section line is COLLINEAR with a clip-polygon edge — `clip_line_to_polygon`'s ABSOLUTE parallel epsilon (`denom.abs() < 1e-15` on an unnormalized dot with |n|·|d|≈100) misread the collinear edge as a genuine crossing and clipped by the ratio of two roundoff residues → nondeterministic partial emission (18/36 sections, some sliver-length; the old "9 over-shared + phantom 0.36° breaks" were downstream noise of the missing/partial sections). Fixed by scale-relative parallel+outside thresholds (sin(angle)<1e-9, distance band 1e-9) in `clip_line_to_polygon` — the tangential-contact class again, fixed at the primitive. Post-fix raw GFA == ops output: F=55, manifold, watertight (all edges 2-use by id), analytic (4 cylinders), vol=operand sum, hole-aware euler 2 (naive euler 3 is CORRECT: the shelled cup's top rim is a genuine annulus face — the test's old naive `euler==2` assert was wrong for this shape). KNOWN RESIDUAL (below engine coincidence semantics, deliberate): the 19µm chord/arc corner lenses at z=0 (32 of them) collapse to the chord — barrel rims are bounded by the socket chords (≤19µm=r(1−cos5.625°) off-surface), the true crescent ring is not represented. Representing it needs the FULL midpoint-split cascade: midpoint paves on circle pave-blocks co-endpoint with a line edge + arrangement lens tracing (chord-space bigon = zero area) + the wall-facet grazing [circle] sections (32 circle→line merges in `merge_duplicate_edges` are these lenses folding). Only chase if a consumer needs sub-20µm corner fidelity. |
| **v2 trimmer neighbor-split — CLOSED** (fixtures `crates/blend/src/trimmer.rs::split_propagates_into_neighbor_wire`, `crates/operations/tests/regress_blend_trim_neighbor_split.rs`) | blend | `split_edge_at` now rewrites EVERY wire referencing the split edge onto its two sub-edges (`propagate_split`; `trim_face_general`'s inline splits routed through it), so untouched cap/rim faces no longer keep the stale unsplit edge (box single-edge fillet: free 16→12, bnd 28→22 at export tolerance; stale-edge refs 0). Also fixed en route: `trim_face`'s closing contact-edge orientation was inverted (trimmed wires were silently disconnected head-to-tail). DISCOVERED, still open in the v2 regular trim path (all evidenced by the regress test's residual 12 free edges): (1) keep-side selection is degenerate — `n·(center−p1)` is ∥ n by tangency, so face1 can keep the SLIVER (the top face does, in the box repro); needs a discard-side hint (spine midpoint side test), TrimSide alone is under-determined because trim chains are wire-order-dependent; (2) `create_blend_face` builds its own contact edges instead of sharing `TrimResult::contact_edge` → position-duplicate free-edge pairs; (3) no end-cap notch trim where a stripe terminates (corner.rs only covers stripe-meeting corners) — inherent v1 gap; (4) chamfer_v2 on a box edge solves the EXTERNAL tangent branch (contacts at z=11/y=−1, outside the solid) and never reaches trimming. |
| **Revolve follow-ups — CLOSED (all three)** | operations/revolve | Pointed-cone apex merge (12→2 faces, degenerate seam wall), annulus/washer-cap merge (16→4, caps keep the smaller rim as a hole wire), partial-turn circle→trimmed `Torus` band + 2 disc caps (exact `π·R·ρ²·Δu` via `partial_torus_sector_volume`). Enablers shipped with it: `tessellate_torus_two_rim_band` (structured band for a doubled-seam torus wall in EITHER rim orientation — CDT/snap cracked both) and hole-winding-agnostic `planar_cap_signed_volume` (a boolean's same-wound inner rim ADDED its disc; holes now subtract by magnitude — made the drilled-tube volume exact). Tests in `revolve.rs` (`revolve_circle_partial_turn_is_trimmed_torus` etc.); census `revolve_matrix` has rows for all cases. |
| **Trimmed-torus ray-cast misclassification — CLOSED (2026-07-10)** (fixtures `crates/operations/src/classify.rs::partial_turn_torus_band_classification`, `crates/check/src/classify/mod.rs::partial_torus_band_interior_points`, `crates/check/src/classify/ray_surface.rs::ray_torus_oblique_from_inside_tube`) | check + operations classify | The "<3 distinct vertices trips the degenerate full-surface branch" hypothesis was WRONG — both crates' `face_polygon` densify closed edges (66-pt band polygon) and the UV containment itself worked. THREE stacked roots, all instrumentation-verified: (1) BOTH local Ferrari ray-torus quartic solvers missed real roots (zero roots for rays from inside the tube) AND emitted off-surface spurious ones (hits at z=4.6 on a torus spanning z±2) at R=6/ρ=2 with oblique irrational rays — small axis-aligned unit tests never caught it; both now delegate to math's residual-verified Durand–Kerner `intersect_line_torus` (the torus-box campaign primitive), local Ferrari cubic/quartic deleted; (2) check's `face_aabb` collapsed each cap disc (single closed-circle wire = ONE vertex; Plane gets no surface expansion) to a point AABB → the BVH prefilter never offered the caps → cap crossings silently dropped from parity — fixed by exact per-curve extent expansion (`expand_aabb_for_curve`: circle/ellipse closed-form, NURBS control-hull); (3) ops-only: `boolean::face_polygon` samples closed rims from the curve's own parameter origin (not the seam vertex), so a band wire's two rims enter the periodic unwrap at incoherent phases → UV rectangle shears into a parallelogram rejecting real band hits — fixed by a seam-anchored sampler local to `classify.rs` (`boolean::face_polygon` is calibrated for band-fragment sharing, do NOT change its phase). `revolve_circle_partial_turn_is_trimmed_torus` now asserts ray-cast probes directly. DISCOVERED, open: the algo ray-cast classifier (`crates/algo/src/classifier/ray_cast.rs`) has NO Torus arm — torus faces fall to the flat Newell-polygon fallback (same parity class as the #1063 cone gap); left untouched here because calibrated boolean landscapes (torus-box) pin its current behavior — needs its own re-probe before adding the arm. |

Fresh full scenario-matrix baseline (2026-07-07, kernel 2.124.0 + #1029/#1030 overlay,
`BREPJS_KERNEL=brepkit pnpm exec vitest run --project generators <suite>` in the tool):
compartment manifold 5/13 (was 0/13; remainder = the lip-corner + tilted-divider rows
above), honeycombJunction engine-fixed (63k→~3k tris; snapshot pins remain
kernel-specific), groupedScoop + splitBin manifold PASS, combinedFeatures 3/11
(handles-panic row above; scoop case near-parity), snapClip 1/4, fit-offset 0/2,
dovetailKey 1/2 (all three: watertight-STL asserts — re-probe after the lip-corner fix
lands, they may share it), dovetail suite timeout >25min (cornerclip row above).

Baseplate re-probe on published 2.124.12 (2026-07-08, post-#1054): partial movement,
no closures — snapClip 0/4 (nm 14 unchanged, key 16→12, 0.6mm-nozzle 11→**1**, clip
volume 46.78 vs 46.6±0.05), fit-offset 0/2 (loose bnd 184→144, at-floor 144
unchanged), dovetailKey 1/2 (bnd=108 unchanged). The #1054 fixture was the CORNER
tile (one rounded corner); the residuals live in the other tile/connector configs —
next step is a fresh operand capture of one failing case (dovetailKey bnd=108 or the
nm=1 nozzle case) on ≥2.124.12. The full dovetail suite: **>25-min timeout →
355s total** (mesh-fallback slab gone in-tool; most tiles now 0.3–1.5s), 2/9 pass;
the A1-canonical corner tile fell ~597 nm → **nm=3 at 468ms**. Residual family:
bnd=108 (5×4 middle-column, inverted, AND dovetailKey — one shared root), bnd=144
(4×4 interior ×2 — interior tiles have NO rounded corners, so this is the
fully-coincident-walls intersect variant), bnd=5 (5×4.5 fractional edge tile, also
still slow at 265s), nm=6 (magnet variant, 82s), nm=3 (corner tile).

**bnd=108/144 family CLOSED (PR #1057, 2026-07-08)** (fixture
`crates/io/tests/dovetail_interior_identity_intersect_inmem.rs`): stage-probe
capture (`buildBaseplateSolid`'s `probe` callback + `serializeSolid` per milestone —
NO instrumented kernel needed) localized it to `cornerClipIntersect`; for all-join
tiles the rounding profile degenerates to a plain box matching the slab bounds, and
`boolean_with_evolution`'s faithful raw-GFA branch mis-split the fully-coincident
identity intersect (134 faces → 38, free=32) — accepted as "valid" because
position-duplicate free edges pass the by-edge-id gate (ids used ≤2×). `boolean()`
was immune via its identical/containment shortcut. Fix: `detect_trivial_relation`
extracted and consulted by `boolean_with_evolution` before the faithful path.
Tool-verified (local overlay): dovetailKey 2/2, fit-offset 2/2, dovetail 6/9 —
middle-column, interior ×2, inverted, magnet all closed. Remaining dovetail
residuals: fractional edge tile bnd=4 (+265s perf), A1-corner nm=3. DURABLE: the
by-edge-id validation gate is BLIND to position-duplicate free edges — any "GFA
result validated OK" claim about watertightness needs the position-quantized
check; and the generator's probe hook + serializeSolid is the cheap capture path
for baseplate ops.

**Doubled-dovetail interior nm=21 (tongue-relief cut) CLOSED at engine level
(2026-07-10)** (fixture `crates/io/tests/dovetail_relief_cut_inmem.rs`): each
relieved nub — cut(6-face trapezoid tongue prism, tapered socket pocket) —
arrived at the connector fuse already broken (bnd=13-15 nm=1-2 per nub through
BOTH boolean entries); the fuse merely accumulated 12 nubs' damage. FOUR stacked
roots (the fixture doc comment carries the full map): (1) restrict 24-sample
graze test dropped a real ~8° socket-mouth corner crossing → refine to the
smaller face extent; (2) open marched-NURBS conic sections kept whole → exact
clip to the plane face's straight boundary edges + the cone partner's
angular-window rulings, TRIMMING the stored NURBS to each kept span
(`domain_with_endpoints` is the full knot domain), plus sampled-projection
T-junction splits (`find_splits_on_nurbs_section`); (3) the ray-cast classifier
had NO analytic cone path — tapered corner patches fell to the flat
Newell-polygon fallback, which mis-counts crossings for interior points ~0.2 mm
inside the pocket walls, keeping two in-chunk pieces (`FaceGeom::Cone` added,
mirroring the partial-arc cylinder path); (4) GFA section edges can store
traversal-order vertices over an unreversed NURBS curve — a B-Rep-clean result
whose tessellation folds the boundary polyline (mesh nm on a watertight B-Rep);
fixed in the SAMPLERS by endpoint alignment (`nurbs_runs_end_to_start` in
`tessellate/edge_sampling.rs`) — normalizing vertex order at the minting site
(`instantiate_wire_edge`) instead broke the calibrated torus-box notch
landscape, do not retry. All six captured nub operand pairs: 8-face
analytic nub, bnd=0 nm=0, both entries. Tool-side re-probe of the doubled
dovetail suite pending fresh capture (the old stage fixtures embed pre-fix
broken nubs). DIAGNOSTIC LESSON: sub-face classification against a coned cutter
was silently polygon-approximated — when a kept-piece pattern matches "inside
the cutter but classified Outside", check `collect_face_geoms` coverage for the
partner's surface types before touching the splitter.

Fractional-plate seam-edge pocket family — CLOSED (2026-07-16, fixture
`crates/io/tests/fracplate_seam_pocket_inmem.rs`): a seam-edge pocket flush
with the tile wall mesh-fell-back and poisoned the whole 5×4.5 fractional
plate (dovetail `5×4.5 edge-y-1` nm export). Root: `find_point_outside_holes`
trusted stored `start_uv` for its hole-rejection polygon and one
foreign-frame vertex corrupted it — classifier seeds landed inside the
opening and the slab top was dropped. Fixed by deriving every polygon vertex
from 3D through the plane frame. Tool-verified: the dovetail suite's
fractional tile passes; with the tangency-nub + groove-mouth PR (#1078) the
suite reaches 9/9. DURABLE: stored `start_uv`/pcurves on hole wires can be
fitted in a FOREIGN frame — any consumer building polygons from them must
re-derive via frame.project(3D) (same class as the pcurve-convention lesson).

snapClip family — THREE roots CLOSED 2026-07-16 (#1080, #1082, #1085); deepened-notch remains:
- Connector key (#1080, fixture `extrude_spline_encoded_profile_recovers_analytic_walls`):
  2D drawings ship corner-treated profiles as B-splines; extrude emitted ruled-NURBS
  walls for exact plane/cylinder geometry and every boolean against the prism fell
  back. Profile-wire curve recognition at extrude entry (the loft pattern).
- Completed 4-way socket-junction disc (#1082, fixture `socket_junction_disc_inmem`):
  the junction circle's 2-arc traced loop samples area-degenerate, the sliver guard
  dropped it silently, and the arrangement was declined on equal loop COUNT; the
  arrangement gate now also fires on any area-degenerate traced loop. Full 20-pocket
  snapClip plate chain analytic (F=595 vs F=6923/bnd=930).
- Snap-slot hole cuts (#1085, fixture `snapclip_slot_cut_inmem`): four stacked
  section-machinery gaps — outermost-pair clip vs INWARD-bulging bite arcs
  (midpoint-classified multi-interval clip, HOLE-FREE faces only: holed faces'
  sections feed the weave, calibrated on whole pieces), multi-window sections kept
  one window, plane×band Lines never clipped to the band v-window (exact affine-v
  trim; mixed pairs get ONLY that trim — the plane-polygon clip on banded pairs
  broke seam-anchored cylinder bands), and marched-fit endpoints ~1e-6 off exact
  chain partners (weld at 100·tol). FOIL SET GREW: cylinder-slot + groove-mouth +
  junction-disc are now mandatory alongside d4/pcut3/divider for ANY section/clip
  change — three wrong gate choices were each caught by a different foil.
- REMAINING after the deepened-opening union AND the plane×cone exact-circle
  arc (fixture `snapclip_export_corner_inmem.rs` — the EXPORT chain builds with
  forExport=FALSE/tapered pockets; `trim_ellipse_to_boundary_crossings` only
  accepted Ellipse sections, so the horizontal cutter-top's exact Circle arc
  died in the 16-sample in-both filter and the corner cones never split; both
  join-edges chains now replay fully analytic posBad=0 — true-variant F=881,
  export-variant F=418): the 0.6mm-nozzle EXPORT chain still breaks at
  op-cut-3 (posBad=10 analytic-but-leaky, accepted by the gate; fresh minimal
  repro via CHAIN=1 DUMP_AT=3 on capture-snapnozzle-noexp in the 2026-07-17
  cache — captured operands are fallback-poisoned, never replay them directly),
  the by-edge-id acceptance gate is BLIND to position-duplicate leaks (poison
  propagates silently — evaluate a position-quantized gate), and the bed-flat
  clip volume 46.701 vs 46.6±0.05 pin — RESOLVED, NOT a brepkit defect: the
  per-stage dual-kernel diff localized the whole delta to the relief cut, whose
  cutter (buildSingleCellSocket) brepkit represents as EXACT ANALYTIC (native
  census F=34 {plane:18,cylinder:8,cone:8}, zero NURBS — the #1045
  loft-recognition) while the reference keeps a NURBS loft that bulges ~0.062mm³
  and over-removes 0.146mm³ from the clip corners. brepkit's 46.701 is the MORE
  accurate value; the pin is calibrated to the reference's loft approximation
  (the "snapshot pins are kernel-specific" class). Resolution is tool-side pin
  recalibration, not a brepkit change.

Fit-offset groove-mouth sliver family — CLOSED (2026-07-16, PR #1078, fixture
`crates/io/tests/fitoffset_groove_mouth_inmem.rs`): each groove cutter's mouth
clips the adjacent socket-pocket rim corners, leaving zero-width top-face
slivers; three variants of the root appear as the chain progresses (each cut
absorbs its mouth rings into the outer wire as bays). Five coordinated fixes —
pave-split hole promotion into the combined arrangement (expansion kept OUT of
the weave input, whose whole-edge re-trace discriminant is calibrated on
unsplit hole edges — pcut3 foil); a CLEAN-TILING cutoff for even-odd hole
nesting (a proper subdivision never nests; component and edge-sharing
discriminants both REFUTED by the divider-lip fuse foil); true circle×section
splits of boundary bay arcs applied ONLY on the arrangement path (global
splits broke d4) plus a bay-mouth arrangement entry (≥2 holes); arc-true
region-polygon probes; at-seam UV endpoint resolution on periodic surfaces
(a 4th-quadrant corner cone's window read as its complement — span derivation
from the circle's own parameterization REFUTED, stored normal can oppose the
surface axis) with orientation-aware plane-arc split normalization. The
captured export chain runs fully analytic+watertight (182→211 faces; the
PUBLISHED kernel's "pass" encloses phantom void wedges at every groove-mouth
corner). DURABLE: the splitter's paths are a web of mutual calibrations — d4,
honeycomb pcut3, divider-lip, and the nub fixtures are the four foils; run ALL
of them on any face_splitter change (each caught a wrong discriminant this
session that fit-offset alone blessed).

Fresh baseplate re-probe on PUBLISHED 2.126.2 (2026-07-16, overlay md5-verified):
dovetailKey 2/2 and fit-offset 2/2 CONFIRMED on the published build; dovetail
7/9 @188s (was 6/9 @355s) — the 4×4-interior doubled-dovetail (the relief-cut
family) passes end-to-end tool-side; snapClip 0/4 with ALL signatures moved
since the mesh-boolean rewrite (join nm 14→4, key nm 12→0 but bnd=326, 0.6mm
nozzle nm 1→15, clip volume 46.78→46.70 vs 46.6±0.05). Dovetail residuals:

- **2×2 A1-canonical doubled-dovetail nm=2 — CLOSED (2026-07-16, PR #1078,
  fixture `crates/io/tests/dovetail_dblcorner_nub_inmem.rs`; tool-verified on
  the overlay: dovetail 9/9 @37s, dovetailKey 2/2, fit-offset 2/2 — the
  265s fractional slow case is gone with the groove-mouth fix).** The paired tongue sits offset by exactly the socket corner
  radius, straddling the wall-plane↔corner-cylinder tangency meridian (the
  recurring tangential-contact class). THREE stacked roots: (1) the FF
  raw-curve AABB pre-filter's fixed 16-sample scan missed the flank×cone
  conic's ~2mm in-both sliver on a ~30mm marched curve — the pair vanished
  before the exact open-conic clip ever ran (mirror nub survived by sampling
  luck); now refines adaptively like the restrict graze escalation. (2)
  `trim_open_curve_to_plane_face_lines` clipped conics to the plane face's
  boundary + the cone's u-window but NOT the patch's axial v-range — the kept
  piece overshot the rim circle, dangled, and the splitter's pendant filter
  removed the whole section chain; now bisects v(t) to exact rim crossings.
  (3) `find_splits_on_circle` normalized against the CCW start→end span, but
  a rim quarter-arc traversed CW covers the 270° COMPLEMENT (the #1054
  reverse-twin mechanism on BOUNDARY arcs; `edge.forward` does NOT
  disambiguate — the cone rim is fwd=true with u decreasing); now picks the
  true arc via the edge's own UV midpoint and the consumer uses the returned
  on-circle foot. Result: 10-face analytic nub (1 cone + 1 cyl + 8 planes),
  watertight, both boolean entries. LATENT: `find_splits_on_ellipse` has the
  same complement hazard, no repro yet.
- **Fractional edge tile 5×4.5 — CLOSED by the seam-edge flush pocket fix
  (#1076, fixture `fracplate_seam_pocket_inmem.rs`; see the closure entry
  above).** The old forExport=true capture (F=7928 pocketsCut) was the wrong
  variant; the true export-variant root was the flush-wall pocket cut.
- **A1-corner nub FUSE membrane (post-#1082 plate topology) — CLOSED
  (2026-07-16, fixture `crates/io/tests/dovetail_a1corner_nubfuse_inmem.rs`):**
  the #1082 junction disc changed the plate's corner topology, and the nub
  fuse's plate-wall middle strip got its splitter interior point at exactly
  (42, −4, −1.75) — the intersection of the wall plane, the relief-bore
  tangency/profile-seam meridian, and the dovetail flare plane. All THREE
  cardinal classification rays ran along edges/seams/the tangency line, the
  parity votes were garbage (1/3), and the interior strip classified Outside —
  a kept membrane, non-manifold analytic result, mesh fallback for the whole
  plate chain (op-fuse-0 F=1517), dovetail 8/9. Fix in
  `classifier/ray_cast.rs`: each ray now reports whether any hit grazed a face
  boundary, band limit, or in-plane face; when ALL THREE cardinal rays are
  degenerate the vote re-casts with fixed generic (√-prime) directions. Any
  clean cardinal ray keeps its historical verdict — two blunter variants
  (all-generic; escalate-on-split-vote) each broke a calibrated foil
  (honeycomb pcut1 over-shared 0→7, wallcut free 0→48) before the per-ray
  degeneracy design passed all foils simultaneously. DURABLE: splitter
  interior points of notched/symmetric pieces land on feature-plane
  intersections BY CONSTRUCTION; classification must survive on-plane sample
  points. STALE-CAPTURE TRAP: the capture dir held two interleaved probe runs
  (02:47 pre-fix + 19:44 fresh); one full iteration was burned replaying the
  stale pair whose F=18 nub was an OPEN mesh-fallback operand (GIGO, already
  fixed) — check bin mtimes before replaying mixed capture dirs.
- **A1-corner recess-hole conic-web split (the scenario's nm=2 STL pin) —
  CLOSED (2026-07-16, fixture `crates/io/tests/dovetail_a1corner_holecut_inmem.rs`):**
  after #1088 made the A1 fuse chain analytic, the remaining nm=2 came from
  the forExport=false hole cuts: each recess box's slanted wall gets a
  4-section web (3-line U-chain + plane×cone conic T-ing mid-span into the
  z=0 line). Two 1e-6-fit-error-vs-1e-7-tolerance gaps: (1) the weld had no
  anchor at the T (now welds endpoints onto other Line sections' INTERIORS —
  nearest strictly-interior foot in the 100·tol band); (2) the planar
  arrangement's arc on-plane round-trip demanded 1e-7, bailing on the fitted
  conic (now 100·tol — genuine straddle arcs are off by orders of magnitude
  more). Un-rescued, the angular wire builder walked the CW-boundary slit-web
  as ONE grand circuit under BOTH winding rules (that greedy-trace weakness
  remains — the arrangement is the sanctioned rescue for plane-face webs).
  DURABLE: marched/fitted section geometry is good to ~1e-6, every exact-tol
  (1e-7) gate it meets needs a weld-scale (100·tol) band; this is the FOURTH
  such gap in this family (weld anchors, T-split, on-plane, junction-disc).
- **snapClip deepened-notch family — PARTIAL (2026-07-17, 16-iteration dig,
  full log in memory project_snapclip-plate-bore.md iterations 1-16):** the
  op-cut-3+ nm chain root-mapped. LANDED: arrangement true line×arc
  crossings (bisection-refined against the real arc, on-line validated,
  endpoint-guarded — phantom chord-crossing breaks desynchronized the
  half-edge graph and the tracer's dangling-edge retreat emitted SLIT
  regions with doubled edges), exact-UV T-break registration, weld-band
  on-plane/T-break tolerances, trimmed sub-arc emission, and a section
  split registry (plane-first ordering + geometric point-on-curve presplit
  for curved faces). Raw repro posBad 37→22; ALL calibrated foils green.
  NOT closed: the remaining 22 are cross-face BOUNDARY-edge desyncs whose
  root is that **marched FF sections on curved faces carry
  pave_block_id=None — they bypass the pave machinery that gives plane
  faces pre-split, shared-vertex sets. The canonical fix is pave-block
  attachment/splitting for marched curves at phase-FF/make_blocks altitude;
  every face-splitter-level propagation (face-web geometric, per-edge
  keyed, NURBS boundary arm) broke the groove/a1corner calibrated chains
  which DEPEND on downstream reconciliation of asymmetric splits.** Repro:
  cache replay_scplate.rs (RAWN=n) + capture-snapclip-plate-fresh.
- **NURBS endpoint-trimmed convention — FORWARD SPANS SHIPPED; reversed
  spans remain OPEN behind a named arrangement defect (2026-07-17, the
  deepened-notch dig's terminal root):** `EdgeCurve::domain_with_endpoints`
  for `NurbsCurve` historically returned the FULL knot domain, ignoring the
  endpoints — every NURBS sub-span consumer silently evaluated the WHOLE
  curve (piece pcurves carried the parent's UV endpoints; the wire builder
  conflated near-coincident structures — the snapClip deepened-notch cone's
  twin rims). SHIPPED (topology/src/edge.rs, unit tests in the same file):
  whole-edge endpoints (either orientation) and closed edges keep the full
  span; a validated FORWARD interior sub-span (both projections on-curve
  within the 1e-5 weld band, span > 1e-6·domain) returns the projected
  trimmed `[t₀, t₁]`. On the RAWN=1 raw repro this cleaned one of the two
  mirrored junction signatures (the use=3 triple + micro-edge chain at
  y=−39.4). SECOND LANDING (same day): REVERSED sub-spans accepted on
  clearly-open curves (`t₀ > t₁`, start→end interpolation stays truthful;
  closed curves keep the full-domain fallback — a reversed pair there is
  usually a seam-crossing forward sub-arc). The "degenerate phantom loop"
  that blocked reversed acceptance was a MISREAD: the single-edge closed
  NURBS inner wire on the aborting cone wall (seam gap 5e-9, source_edge_idx
  Some) is a LEGITIMATE pre-existing notch outline; the real defect was a
  COVERAGE hole — walls reaching `fill_images_faces` through split paths
  that never run the dedicated `cylinder_cone_remainder_interior` search
  aborted unconditionally on the lens flag. Fixed by running that search as
  a last resort at the consumption point (fill_images_faces; abort only if
  even the dense grid finds nothing). Together: RAWN=1 posBad 10 → 6, both
  mirrored micro-edge chains resolved; 2.126.12 tool-verified (dovetail
  9/9, dovetailKey 2/2, fit-offset 2/2 hold; snapClip nozzle nm 13→12).
  THIRD LANDING — RAW REPRO FULLY CLOSED (posBad 37 → 0; fixture
  `crates/io/tests/snapclip_deepened_notch_inmem.rs`): the residual was the
  terminal stranded-rim case, solved WITHOUT a detection heuristic by making
  the curved-face splitter geometrically honest. (1)
  `clip_sections_to_outer_region` (face_splitter/mod.rs): sections
  overhanging the face through an OUTER-wire concavity (an earlier cut's
  bite) are clipped in unwrapped UV — fully-off-face and band-hugging
  sub-span re-trace pieces dropped, mixed sections split at bisected
  crossings with junctions snapped ONTO the boundary curve so the exact
  1e-7 boundary-splitter gate accepts them as anchors; gated to
  partial-band quadric faces carrying marched-NURBS boundary edges (the
  clip's polygon is garbage on full-revolution primitive laterals — the d4
  canary caught that regression). (2) Registry-presplit pieces keep their
  PARENT's pcurve — endpoint UVs evaluate at the parent's ends,
  disconnecting them from the boundary in UV; fixed by a
  v-disagreement-gated pcurve refit (v is non-periodic ⇒ unambiguous where
  u could be a 2π translate). (3) Zero-extent section edges from T-junction
  self-splits derailed the angular walker (filtered; a UV-extent guard
  protects closed circle sections). DURABLE polygon-sampler recipe:
  endpoint order for `domain_with_endpoints` must follow the traversal flag
  (selects the correct arc vs its complement for reversed circles) AND the
  samples must then be oriented to wire order empirically (a whole-edge
  NURBS traces the curve's own direction) — each half alone fails a
  different edge class. Deliberate residual: the B-side corner crescent is
  sub-resolution (0.0016 u-width < the 1e-3 fit band) and drops — the
  corner-lens residual class. Export-level verification = the tool 4-suite
  re-probe after release.
- **Mesh-boolean fallback emits OPEN meshes that get CONSUMED — OPEN
  (discovered 2026-07-16):** on the dblcorner nub operands the co-refinement
  fallback produced bnd=5/6 output (warn-logged, then used anyway, poisoning
  every downstream boolean into a 1400-face fallback export). The safety net
  must be watertight or rejected. Repro: the dblcorner fixture operands with
  the analytic path disabled, or any pre-fix build.

- **A universal smarter merge-key for duplicate edges. PROVEN UNBUILDABLE.** The
  gridfinity lip corner (chord + arc, same endpoints) MUST merge; the torus-box in-tube
  lens (line + co-endpoint arc) MUST stay distinct. No merge-key discriminant separates
  them; the distinction is global. Sanctioned pattern: splitter-side midpoint splits,
  per case, so no two edges share both endpoints, and leave
  `merge_duplicate_edges` (in `crates/algo/src/builder/builder_solid.rs`) alone. Control
  the geometry you emit; do not make the shared merge smarter.

Lite magnet-pad graze fuse — CLOSED at engine level (2026-07-21, fixture
`crates/io/tests/lite_pad_graze_fuse_inmem.rs`; the lightweight export family's
`4×4 stress`/`solid bin + magnet` root, full dig log in memory
`project_lightweight-export-failure-map.md` §2026-07-21): a magnet pad's r=4.45 wall
clips a socket-profile corner cone by 0.094 mm, and the cone×cylinder branch curves
exit the cone patch through its ANGULAR-window corner — the in-both run (0.097 mm of a
1.4 mm curve) is far below the graze-refinement's extent-scaled minimum, so restrict
dropped both curves and the wire builder backtracked into out-and-back slits → whole
fuse mesh-fell-back and poisoned every downstream drill. FIVE coordinated fixes, all
required: (1) `Circle3D::intersect_circle` (new math primitive, near-tangent
double-root collapse) + Circle-boundary-edge crossings in
`closed_circle_boundary_crossings` (the Line-only scan left ODD crossing sets) + a
midpoint inserted between same-arc hit pairs (co-endpoint-lens sanctioned split);
(2) `rescue_corner_crossing` in the phase-FF restrict: bisect the in/out window
transitions, trim the sub-span, snap endpoints to the boundary foot then refine to
the exact boundary-curve×partner-surface triple junction (the foot alone is displaced
~1e-6 ALONG the boundary and mints a duplicate vertex); strict-interior midpoint gate
keeps true grazes dropped; (3) fit-error weld plumbing — `curve_endpoints` returns
pave-VERTEX positions within the weld band, section↔boundary UV reconciliation before
the pendant filter (same 3D junction, UV copies 1e-6 apart, 1e-7 graph cells),
weld-scale 3D dedup in `find_splits_on_line`, plane-face zero-extent section filter;
(4) cylinder-face mirrored-winding retry when the greedy loops are broken and the
rectilinear arrangement declines (oblique ellipse cuts; adopt only with NO NEW broken
flags — `wire_loops_self_cross` false-positives on full-period band loops at the seam
vertex); (5) the ops gate: multi-component balance (euler−L == 2·N disjoint closed
pieces) checked BEFORE `unify_faces` (which otherwise mangles a clean N-piece result
— the lite base at this stage is LEGITIMATELY 16 disjoint feet) + Fuse admitted to
the multi-region acceptance, and tessellation edge sampling honoring the
endpoint-trimmed NURBS convention (all three `edge_sampling.rs` NURBS arms sampled
the FULL knot domain, ripping a bd=119 crack along the parent curve of a trimmed
junction spline). Single-pad ops fuse: analytic F=951, position-manifold, mesh bd=0
nm=0 at 0.01, 1.2 s vs the 2.7 s fallback. DURABLE: the greedy walker's outcome is
CHAOTIC in junction-level geometry (each 1e-6 change flipped it between different
broken traces during the dig) — fix junction identity everywhere, then partition
health follows; and a "graze" heuristic keyed to face extent is blind to
corner-window exits, which can be arbitrarily smaller than either face. NOT yet
verified: the 64-pad whole-base fuse and drill chain, and the tool-side lightweight
re-probe (needs release + overlay).
FOLD-2 CONTINUATION (2026-07-21, PR #1150): the diagonal double-graze pad's "lens
cells"/"slit holes" (the whole DCEL v3/v4 dig) were UV PHANTOMS — all four notch
"ellipses" are windows of ONE closed ellipse (pad cyl × socket chamfer plane), and
`find_splits_on_ellipse`'s CCW-domain normalization phantom-split only the REVERSE
section twins at other-window endpoints (the finder/evaluator convention mismatch;
`evaluate_edge_at_t` is shorter-arc). Fixed by `find_splits_on_section_ellipse`
(shorter-arc twin of `find_splits_on_section_arc`). Fold-2 raw: odds 16→9, no ×3
families, wall = its true 2 sub-faces via plain greedy — do NOT resurrect the DCEL
patches for this defect. Remaining fold-2 (9 odds): the east corner-bite ring
unpaired — the wall's east-strip sub-face is built but never pairs (classification
or edge-copy identity); dig state in memory `project_parity-loop-state.md`. The
whole-chain FUSE_ALL measurement is dominated by the open-mesh-consumption row
above (deterministic bd=21 nm=12 with the fix vs bd=0 nm=6 before — the fallback
output shifted, the consumption hazard is the root).

Funnel/honeycomb cylinder-disc arrangement campaign — CLOSED (2026-07-17, memory
`project_cylinder-arrangement-rescue.md`): curved/periodic faces had NO arrangement
rescue (the plane path's rescues are all `is_plane`-gated), so a box cut crossing a
cylindrical pocket at PARTIAL overlap figure-eighted the greedy wire builder. Fixed as
three sub-gaps, all "closed/arc-bounded face cut by sections, greedy drops/mistraces":
(2) PLANE disc (closed-circle boundary) cut by chords — #1109 (`try_split_disk_by_chords`
in `face_splitter/special_cases.rs` + single-crossing bail relaxation in
`fill_images_faces.rs::clip_line_to_face_boundary`); (3) PLANE wall + single-arc crossing
— SUBSUMED by #1109's single-crossing relaxation (tool walls are hole-free planes cut by
one generator; proven by single-variable isolation, no separate work); (1) CYLINDER-wall
rectilinear-UV arrangement (`split_cylinder_band_by_arrangement` in `face_splitter/mod.rs`,
purely additive, gated `u_periodic && !v_periodic && Cylinder && greedy-broken`) — #1112.
Final residual (floor lens's co-endpoint rim-arc + tool-chord collapsing in
`merge_duplicate_edges` — the merge-key UNBUILDABLE class) closed by the SANCTIONED
splitter-side midpoint split of the minor lens arc (`try_split_disk_by_chords`; the
existing `split_arc_edges_at_collinear_vertices` propagates the cut to the shared cylinder
rim — no two-site coordination; NOT a merge-key change). Synthetic pocket-notch repro
`crates/io/examples/replay_synthbox.rs` now posBad=0 on ALL cases. Distinct from the
snapClip deepened-notch pave-bypass root above.
**SCOPE CORRECTION (2026-07-18 tool re-probe, 2.126.17 overlay):** "CLOSED" = the ENGINE
sub-class (cylinder-pocket-notch / disc-chord), foil-safe (27/0) and NO tool regression
(export-integrity: 33 real fails = same known deferred families; published solid-cutouts
6=6; the other 180 fails are the task-#14 poison cascade). **BUT the TOOL's own
`combined features › 2×2 honeycomb walls + funnel cutout` scenario is NOT fixed — still a
533s bisect-hang + fail.** Its root is the SEPARATE honeycomb-cut coincident-wall
assembler hang ([[honeycomb-wall-cut-coincident]] pcut0) + the funnel-cutout, not this
cylinder-arrangement bug. The synthetic proxy fixed a real class but was NOT validated
against the real honeycomb+funnel operands first (the roadmap's own warning). This campaign
did NOT move tool parity; the headline failing families (scoop #11, screw base #12, solid
cutouts #13, honeycomb-cut) remain open.

Label-sockets family — BOTH roots CLOSED (fixture
`crates/io/tests/labeltab_attach_inmem.rs`, one test covering both). (1) The bd=24/14
export was a GATE LEAK, not a build defect: `validate_boolean_result` checked unclosed
wires and c>2 non-manifold but never c==1 FREE edges, so a warm-cache run whose GFA
happened to complete with 8 free edges and an accidentally-balanced Euler was accepted
(#1192, free_edges>0 now hard-fails; the cold run aborted to the fallback and passed —
which is why the scenarios passed solo and failed only after an earlier scenario warmed
`getCellSocketTemplate`). (2) The remaining analytic root — the tab-attach fuse itself
never assembling ("open hole shell with 97 faces would be dropped", both variants) —
was the CORNER CRESCENT: the tab's square top corners overhang the cavity's rounded
corners, and the tab's back-plane chord RIDES the cavity's collinear back line for most
of its span while jutting ~2.55mm into each crescent. Two coordinated fixes: (a)
`line_section_boundary_extensions` in `fill_images_faces.rs` — the boundary re-trace
test samples interior points, all of which land on the covered middle, so the whole
section read as a re-trace and the crescents were never split off; the uncovered
extensions are now recovered by exact interval arithmetic (NOT sampling — the extension
fraction can be arbitrarily small) and re-queued through the interval loop, which
terminates because an extension's own coverage set is empty. (b) A third arrangement
entry condition in `face_splitter/mod.rs` keyed to a DEMONSTRATED failure rather than a
predicted one: `loops_have_out_and_back` detects the angular walker's own signature of
having woven twin section edges into a single loop (an edge immediately followed by its
exact UV reverse). Result: analytic fuse, watertight, volume 20462.5 by
inclusion-exclusion (17421.32 + 3046.86 − 5.71). DURABLE: any coverage/containment test
that decides a section's fate by INTERIOR SAMPLING is blind to overhang at the ends —
prefer exact interval math wherever the salvageable fraction can be arbitrarily small;
and an arrangement trigger keyed to a post-hoc failure signature cannot demote a
working case, which is the cheap way past this splitter's web of mutual calibrations.

Custom-shape T lip band cut — CLOSED (2026-07-24, fixtures
`crates/io/tests/lipband_cut_inmem.rs` un-ignored + `tship_lipcut_inmem.rs` corrected).
The "no outer shell found" T body+lip fuse failure traced to a malformed lip operand:
`cut(outer T-prism, inner T-frustum)` produced a DOUBLED bottom (outer ±62.75 disc
unsplit + tool's ±61.55 disc, both same-orientation) instead of one ring. Root was
CLASSIFICATION not the FF split (doctrine held): the correctly-split band-bottom ring was
mis-classified Inside and dropped because `classify_coincident_coplanar`'s depth probe
stepped tip→centroid by fractions that overshoot a ~1.2mm annulus into the hole, finding
no valid probe → unstable ray-cast → Inside. Fix (`classifier/mod.rs`): deep centroid
fractions first (honeycomb stacked-cap foil needs them), then a thin-band absolute-nudge
fallback near the tip. Band now single-covered, translation-invariant, vol 6090.8 (was
the doubled 20108.8, which `tship_lipcut_inmem` had enshrined as "watertight" — its by-edge-id
gate is blind to position-duplicate doubling). Benign residual: bottom tiles as ring + 2
tiny T-armpit pieces (3 faces) from redundant FF-coplanar concave-corner sections — exact
watertight tiling, not the defect. TOOL-SIDE RE-PROBE DONE (2026-07-24): 3x3 T export
bnd=173 nm=6 @10.4s -> bnd=0 nm=0 @0.84s; L and U already clean and byte-identical, 1
of 23 customShape triangle counts moved. DURABLE detector reaffirmed: `solid_volume` is TRANSLATION-VARIANT on a doubled/malformed
boundary (a doubled face breaks Σ(area·nz)=0) — translate and re-measure needs no second
oracle. And `validate_solid` + the by-edge-id manifold gate are both BLIND to nested
same-orientation / position-duplicate faces.

Custom-shape O-ring — CLOSED (2026-07-24, fixture
`crates/io/tests/oring_nested_holes.rs`). Found by the T re-probe and it was a
TESSELLATION bug, not a boolean one: minimal config `3x3 O, base=flat, lip=OFF`
gave nm=88 at 1800 tris in 64ms while the B-Rep was clean (F=47, 24 cylinder +
23 plane, free=0 over=0). All 88 folds sat on the z=21 wall top, whose 3 inner
wires NEST (cavity opening > island band > central hole). Two stacked roots:
(1) both SOLID tessellation paths seeded hole flood-removal at each wire's
vertex CENTROID, and a bin centred on the origin gives every concentric wire the
same centroid — first flood took the innermost cell, the rest found it gone, so
the cavity was never removed (the non-shared path already used
`find_interior_seed`); (2) removing one cell per inner wire is wrong regardless
— nesting alternates material and void, so only ODD-depth wires bound a hole.
Fixed by `hole_removal_seeds` in `tessellate/planar.rs` (per-wire interior seed +
even-odd depth by geometric containment; stored winding CANNOT classify them —
a boolean can emit a hole wound like its outer). Tool-verified: every O variant
bnd=0 nm=0, and the downstream fallback disappeared (lip=on 24.2s -> 4.2s,
off-centre 29.7s -> 4.7s); L/T/U/full unchanged. DURABLE: a centroid is not an
interior point for concentric or non-convex wires, and cell-area arithmetic
(predicted 13945.8/595.0/793.1 vs measured 13944.0/590.1/799.9) confirmed the
mechanism instead of inferring it. REFUTED en route: arc-cornered wires as the
trigger — synthetic straight-edged multi-hole faces are clean because their
holes are side-by-side, not nested.

combinedFeatures re-read (2026-07-10, 2.124.13-based overlay, full 11-case suite):
all 6 structural cases PASS including "handles + label (back skip)" (7167 tris,
106s) and "handle holes" (86s) — the 2026-07-08 swallowed-panic/borrow-poisoning
defect no longer reproduces; the 5 remaining vitest failures are benign
reference-kernel triangle-count snapshot pins (runner logs 0 tris for any
failure), and the 2 structural passes over 60s are the per-test-timeout PERF
item. Any future panic self-reports via `lastPanicMessage()` (row above).


The remaining `#[ignore]` entries are diagnostics or slow perf cases, not open bugs:
the `profile_intersect.rs` box-sphere probes are stale leftovers (box-sphere shipped
analytic in #1006), `staircase_fuse_with_cylinders` is a ~2 min perf run, and the two
`#696` dovetail entries plus `diverge_first_cut` are print-only diagnostics.

CLOSED, do not re-open as deferred: honeycomb wall-pattern cut (#925/#928,
`crates/io/tests/gridfinity_honeycomb_cut_inmem.rs` passes), reversed-edge periodic-copy
top-face (#932, `extrude_half_*_reversed_edge_volume` pass), multi-arc hemisphere gap
(#1006).

Export-integrity matrix baseline (2026-07-24, `binGenerator.scenario.export-integrity`, 408 tests
asserting zero boundary edges + bounded non-manifold on the exported STL — the tool's own version of
the STL edge-use oracle). Published 2.128.2: **43 failed / 365 passed**. Local main (T-lip #1209 +
O-ring #1212): **37 failed / 371 passed**. Fixed: both `3x3 T with lip` cases, `3x3 O-shape (ring)
with lip`, `O-shape + magnet base + lip`, plus pathfinder/permutation/scoop rows. NO regressions —
the one apparent regression (`wall patterns > slots carves 3x3x5 walls (scale 0.5)`) fails IDENTICALLY
on both kernels in isolation (63.3s published vs 64.3s main), i.e. it is a timing-borderline scenario
whose full-suite verdict depends on cache warmth and machine load, not on the kernel. Do not chase it
as a correctness bug.

Failure families on that baseline: kumiko 14, permutation matrix 7, custom-shape 6, solid cutouts 3,
then singles. **Kumiko's 14 are ONE root, and that root is PERF, not a crash.** Isolated probe on
published 2.128.2: the goma 1x1x6 export SUCCEEDS — 2.7MB of STL, `lastPanicMessage()` returns none, no
panic anywhere — but it takes **849 SECONDS (14 min)**. The "recursive use of an object detected which
would lead to unsafe aliasing in rust" seen in the suite is a CONSEQUENCE: the export blows through
vitest's per-test timeout, the abandoned async generation chain stays pending, and it re-enters the
kernel concurrently with the next test. The following kumiko scenarios then inherit the poisoned object
(two surface as "Shape handle has been disposed"); only `mitsukude bold` (bnd=571) has an independent
assertion. So do NOT chase this as a panic — `catch_unwind`/`lastPanicMessage` have nothing to report.
Chase the 849s. FULLY ROOT-CAUSED (#1215-#1219) — the whole measured chain lives in the doc
comment of the ready-repro `crates/io/tests/goma_wall_band_cut_inmem.rs`; read that, not this.
Summary: the 850s scenario trips vitest's timeout, whose abandoned async chain poisons the kernel
for every later kumiko test (14 failures, ONE root). 203s of it is a single `cutAll` of 8 lattice
bands, slow only because the analytic path is rejected for **30 free edges** and the mesh fallback
runs; the analytic path is ~12x faster and keeps all 12 cones + 24 cylinders. Those 30 edges are
**4 missing faces** in one **0.05mm plane-vs-cylinder sliver** — the tool's deliberate
`SLAB_OVERLAP = 0.05` past the corner tangent planes (a tangency workaround). Repro is ~230ms per
tool; all 8 bands fail, evens with free=30, odds aborting on an open growth shell. TARGET: the FF/section/split stage — the faces are never created. SHARPEST DATUM
(`FACES_NEAR_X=17.025`): in the result every corner CYLINDER face is trimmed at x=17.000 while
tool0's cut plane is at x=17.050 — the 0.05mm of cylinder between them has NO face at all, and
that band is exactly what the 30 free edges bound. Planes do reach 17.050; the cylinders stop
0.05mm short. INPUT SCAN (`BASE_FACES_NEAR_X`) shows 17.000 is the BASE's own geometry — the
tangent point where each corner cylinder (x[17.000,20.750]) meets the flat wall (x[-17.000,17.000])
— so it is not a trim. tool0's cut plane at x=17.050 therefore lies 0.05mm INSIDE the cylinder's
range (exactly SLAB_OVERLAP past the tangent), and should split it. The result's cylinders still
span [17.000,20.750], identical to the base: had they been split with the outer piece kept they
would read [17.050,20.750]. **So the cut plane appears not to split the corner cylinders at all** —
that is the defect, and the missing patches are the cylinder pieces it should have produced. ALGO TRACE (`BK_FF_TRACE=<x>` in phase_ff.rs, env-gated): sections ARE
emitted and DO survive clipping — at x=17.05, 64 cylinder x plane pairs pass the AABB test (736
rejected, mostly correctly) and **12 sections survive `restrict_curves_to_faces` intact**. and all 12 are then EMITTED into the arena as `line` curves —
correct geometry, since the cut plane's normal is along X while the corner cylinders' axes are
along Z, so the plane is parallel to the axis and meets them in generators, not ellipses (and only
one of the two generators lies on each quarter-arc face). So FF, restrict and emission all do their job. THE GAP IS
BETWEEN ARENA EMISSION AND THE PER-FACE SECTION LISTS (`BK_SPLIT_TRACE=1`, eprintln at the
`fill_images_faces` face loop): that loop DOES run, over 741 faces, and reports **only 2 of 24
CYLINDER faces with has_sections=true** (22 without; planes are 420 true / 285 false) — although FF
emitted 12 cylinder x plane curves. So ~10 emitted sections never reach their cylinder face's
section list, which is why those faces are never split. NEXT STEP: `build_section_map` in fill_images_faces.rs, which SKIPS any
`arena.curves` entry whose `pave_blocks` is empty (`continue`). REFUTED: that skip never fires — all
**890 curves carry exactly ONE pave block, none are empty**, so this is NOT the snapClip
deepened-notch pave-bypass root. Nor is it the `section_map` lookup: that map has 2 cylinder keys
(total 422; plane 420) and the other 22 cylinder faces are simply ABSENT (`in_map=false`), so
nothing is lost in the lookup. CAUTION — do NOT read "only 2 of 24 cylinders sectioned" as the
defect. The `FACES_NEAR_X` filter tests only the X range, and EVERY corner cylinder spans
[17.000,20.750], so all four corners pass it regardless of y/z; most of those 22 are on other
corners and SHOULD be untouched by tool0's single-wall slab. **RESOLVED: 2 sectioned cylinders is
RIGHT, not short** — localizing by the free loops' y/z (not x) puts all four missing outlines in the
SINGLE (+x,−y) corner, whose two full-height wall cylinders are exactly the outer r=3.75
(x[17.000,20.750], 30 edges) and inner r=2.55 (x[17.000,19.550], 34 edges); the other 22 are other
corners or low-z base profile that tool0 never reaches. **AND THE QUADRIC SIDE IS PROBABLY NOT THE
DEFECT AT ALL.** `FACE_WIRES=1` (new knob on the replay example) shows both cylinders carry ONE
plain outer wire and ZERO inner wires — so the cone-box row's predicted "inner wire duplicating the
outer" fix-shape does NOT transfer here, and a bayed single outer wire is the CORRECT shape for a
quarter-cylinder bitten at its θ=90 edge. Re-reading the free-loop geometry accordingly: each
missing outline spans y∈[−19.550,−20.750] — exactly the 1.2mm wall thickness between the inner and
outer corner cylinders. An intermediate read of that — "the missing patches are planar TOOL-side
faces, not cylinder pieces" — was WRONG and is retracted; it inferred the surface from the rim's
ellipse arcs without checking whether the cylinder boundary itself was notched. **THE REAL
SIGNATURE, and the sharpest datum in this whole dig: the notch forms in 5 of 8 z-bands and is simply
ABSENT in 3.** Dumping face 5678's 30-edge outer wire (`FACE_WIRES=1` prints per-edge geometry and
flags free edges) shows a clean repeating bay wherever the cut worked — `line` along the tangent
generator at x=17.000, `ellipse` out to x=17.050, `line` down the cut plane, `ellipse` back — i.e.
the quarter-cylinder trimmed from θ=90 back to θ=89.24 across the tool's 0.05mm overshoot. In the
three failing bands (z 11.507–12.338, 6.794–7.624, 2.700–3.192) the wire instead runs STRAIGHT along
x=17.000 with no bay, and **those un-notched generator segments ARE the free edges** (e17959, e17965,
e17971), free because the flat wall at y=−20.750 does have its opening there so nothing pairs.
`FREE_OWNERS=1` confirms the whole rim: 10 faces carry all 30 free edges — the two corner cylinders
(3 + 2) and eight planes, dominated by the cut-plane faces Id(6090) 10-of-12 and Id(6091) 4-of-9.
SLOPE DISCRIMINANT — PROPOSED AND REFUTED IN THE SAME SESSION, do not re-chase: the bridging
ellipse's z direction looked like the split (working bays DECREASE outward 19.140→19.111, the three
failures INCREASE 12.338→12.367), but enumerating all 8 bands kills it — the working bay at
z 9.291–10.122 goes 10.122→10.151, i.e. UP-outward exactly like all three failures, and its notch
forms fine. Full band table, top to bottom (B=notch formed, F=free): B 18.309–19.140 down,
B 16.093–16.923 down, B 13.723–14.707 down, **F 11.507–12.338 up**, B 9.291–10.122 up,
**F 6.794–7.624 up**, B 4.578–5.408 down, **F 2.700–3.192 up**. Also REFUTED as discriminants: band
width (working and failing bands are both ≈0.831, except one working band at 0.984 and one failing
at 0.492) and band spacing (centres are a near-uniform ≈2.2 lattice pitch). START THE NEXT SESSION
HERE — **THE DEFECT IS IN PHASE FF'S AABB IN-BOTH PRE-FILTER, NOT THE FACE SPLITTER AND NOT
`restrict_curves_to_faces`.** (An intermediate commit on this branch pinned it on
`restrict_curves_to_faces`; that was a MISREAD of the trace — `before_restrict` is captured after two
shadowing filters, so "restrict 0 → 0" means the curves were ALREADY gone. Stage traces afterF1 /
afterF2 show filter 1 keeps 2 of 2 on all 16 pairs and the AABB pre-filter at phase_ff.rs ~333–383
drops 2→1 on the 12 successes and 2→0 on the 4 failures.) Aggregating the
`BK_FF_TRACE=17.05` pair/restrict lines by partner face closes the accounting exactly. Isolating the
true cut-plane partners (`bx[17.050,17.050]`; the `bx[15.437,17.050]`/`[16.037,17.050]`/
`[16.756,17.050]` partners are the slanted lattice walls and all restrict 0→0): outer cylinder
`ax[17.000,20.750]` × cut plane gives raw_curves=2 restrict **1→1 on 5** pairs and **0→0 on 3**;
inner `ax[17.000,19.550]` × cut plane gives **1→1 on 7** and **0→0 on 1**. That is 5+3=8 and 7+1=8,
i.e. ALL 16 band×cylinder pairs accounted for, and the 3 outer + 1 inner failures are exactly the 3
outer + 1 inner missing notches and exactly the 4 free components. **So restrict is handed TWO raw
generator curves on all 16 geometrically-identical pairs and discards BOTH on 4 of them.** This also
corrects #1222: its "12 sections survive restrict intact" was a FALSE ALL-CLEAR — the correct
denominator is 16, and 12 is just the successful-notch count. **ROOT CAUSE, CONFIRMED BY EXPERIMENT:**
the AABB pre-filter samples each raw curve 16× and keeps it only if some sample lands in both faces'
inflated AABBs — but for `EdgeCurve::Line` it then RETURNED FALSE without the adaptive refinement it
applies to every other curve type ("straight lines are exactly represented by their endpoints; a
uniform scan cannot under-sample them at this granularity in practice"). That reasoning is wrong for
this geometry: exactness of the LINE says nothing about whether a sample lands in the tiny in-both
WINDOW. The generator spans the full ~20.3mm cylinder height, 17 uniform samples give a ~1.27mm
pitch, and each lattice opening band is only ~0.83mm tall — so whether a band is hit is aliasing
luck, which is exactly why the B,B,B,F,B,F,B,F pattern looked random. Note the same mechanism is
ALREADY documented one filter above for the faceted-ramp × cylinder ELLIPSE case ("the 16-sample
AABB pre-filter below and the uniform-t restriction both drop it (no sample lands in the band)"),
which got a bespoke exact-arc bypass — lines never got one. **FIXED (#1224):** a straight section
never needs sampling at all — the predicate is membership in `bb_a ∩ bb_b`, itself an AABB, so
`segment_meets_both_boxes` slab-clips the segment against it (exact, O(1), and strictly CHEAPER than
the 16-sample scan it replaces). tool0 and bands 2/4/6 all go free=30 → **free=0** (F=495, 24
cylinders + 12 cones preserved, ~230ms unchanged); `goma_wall_band_cut_is_closed` is un-ignored and
green. **GATED to pairs with a Cylinder/Cone partner, deliberately NOT plane×plane** — the exact and
sampled tests can only disagree when the in-both window is shorter than the sample pitch, and on
that difference set the inflated AABB is a gross over-approximation of a PLANAR face, so the exact
test also admits lines crossing `bb_a ∩ bb_b` while missing both faces: ungated it admitted exactly
two such plane×plane lines into `dovetail_a1corner_nubfuse_inmem` and took it watertight → bnd=158.
Plane×plane keeps the sampled test and its `return false`, so it stays theoretically susceptible to
the same aliasing; no repro exhibits it, and closing it needs a test against true FACE extents
rather than AABBs. **STILL OPEN after that fix — THE ODD-BAND FAMILY, now the live goma work:** bands
1/3/5/7 abort with "open growth shell with N faces" (9, 22, 23, 36 respectively). PROVEN INDEPENDENT
of the line fix — identical counts on pre-fix main, so do NOT assume a shared root. The abort is
DELIBERATE (`builder_solid.rs` ~1192): an OPEN growth shell of ≥4 faces is a genuine solid lump whose
selection left unpaired junction edges, and failing beats silently deleting its volume (the lite
fused-foot lesson). Note the odd tools carry MORE faces than the even ones (726/737/714/764 vs a
uniform 663) — they are structurally different lattice members (the diagonals), so they meet the
corner cylinder at an angle rather than square on. LOCALIZED (env-gated `BK_OPEN_SHELL` probe at the
abort site, stashed not yet committed): tool1's 9 faces are ALL PLANES clustered in the SAME (+x,−y)
corner as the even-band defect and the SAME 1.2mm wall-thickness band — x≈17.1–18.1, y≈−19.4 to
−20.75 — but in one narrow z window ≈8.2–9.0. **THE STRONGEST DATUM: BOTH LUMPS ARE ALL PLANES,
zero cylinder or cone faces.** A genuine chunk of bin WALL at a corner would necessarily include
corner-cylinder faces; the lattice tool is 100% planes. So these lumps are clusters of TOOL faces
(the cut surfaces) that formed a disjoint growth shell instead of joining the main body — i.e. the
junction between the tool's cut surface and the bin's corner cylinder never connects. Start there,
not at a stray-sliver theory. **ROOT NARROWED (2026-07-25, #1227): THE CORNER-CYLINDER SUB-FACES
THAT SHOULD CLOSE THE CHUNK ARE MISSING FROM THE SELECTION.** Four measurements, each ruling
something out. (1) Every one of tool1's 14 unpaired edges has `same_id_outside=0` AND
`coincident_other_id=0` — the partner exists nowhere in the selected set under ANY identity, so this
is neither "the lump was not walked into the main shell" nor a double-minted junction. (2) All 9
lump faces carry `src` ids 226–348; the base holds only 78 faces (ids 0–77, deserialized first), so
EVERY lump face is tool-derived and the missing partners must be BASE faces. (3) The lump's boundary
vertices lie EXACTLY on the two corner cylinders — (17.186,−20.745) and (18.114,−20.581) are at
r=3.750 from axis (17,−17), and (17.809,−19.418) is at r=2.550 — so the chunk is the wall segment
between the inner and outer corner cylinders, and those cylinders are what should close it. (4) Of
the 11 selected faces within 0.5mm of the lump bbox x[17.186,18.114] y[−20.745,−19.418]
z[8.185,9.026], ALL are planes and only ONE is base-derived (src=73). INSTRUMENT VERIFIED (the
"probe that misses the path looks like a null result" trap): the same run reports selected-total
`[cone 12, cylinder 26, plane 378]`, so cylinders ARE selected globally — their absence at the lump
is real and LOCAL. Lump signed volumes are substantial, not slivers: 3.69 / 7.39 / 9.55 / 10.13 mm³
for bands 1/3/5/7. So the over-selection reading (a spurious tool patch that should never appear) is
REFUTED; this is a genuine missing-face defect in the same corner and the same family as the
even-band notch loss, reached by a different mechanism. **FF STAGE MEASURED (#1228):**
`BK_FF_TRACE=17.65` with new per-stage `afterF1`/`afterF2` traces (they report curve KINDS, and they
exist because the old `restrict N -> M` line captures N AFTER both filters, so an earlier loss reads
as "restrict 0 → 0" and looks like restrict's doing — that misreading cost one iteration in the
even-band dig). At the lump's x the 127 surviving cylinder×plane pairs split as: **73 dropped at
filter 1 (lines), 49 ELLIPSES dropped at filter 2's AABB in-both sampling, 3 ellipses surviving, 2
line pairs.** So the dominant loss is ellipses at filter 2 — the SAME aliasing class as the
even-band Line bug, on the path #1224 does NOT cover (its exact clip is gated to Lines) and which
the existing bespoke exact-arc bypass only covers for the faceted-ramp × cylinder configuration.
**AND THE ELLIPSE-ALIASING READING OF THOSE 49 IS REFUTED — DO NOT "FIX" FILTER 2.** The caution
paid off: a temporary min-distance probe (curve samples vs the mutual box `bb_a ∩ bb_b`, since
removed) shows NONE of the 49 dropped ellipses has a mutual box overlapping the lump's z window
[8.1,9.1] — they are all elsewhere in z — and the CLOSEST drop anywhere misses by 0.108mm while
sampled at 0.055mm spacing (n_fine=404 over approx_len=22.261), i.e. a genuine 2× separation, not an
aliasing artifact. So filter 2 is working correctly here and the 49 drops are legitimate; loosening
it would break calibrated fixtures for nothing. Worth recording for a FUTURE case though: `n_fine`
is clamped at 1024 while `approx_len` reaches ~1173mm on near-axis-parallel planes (1.15mm spacing
against sub-millimetre boxes), so the clamp IS a real aliasing hazard — just not this bug's cause.
CONFIRMED, the sections ARE emitted and DO cover the lump: the `FF_TRACE emit` line now carries the
curve's y/z bbox, and at the lump's x tool1 emits `curve#26` ellipse z[6.128,12.000]
y[−20.749,−13.251] (the OUTER r=3.75 cylinder) and `curve#241` ellipse z[7.068,11.061]
y[−19.550,−14.450] (the INNER r=2.55 one) — both spanning the lump's z window [8.185,9.026] and its
y range — plus `curve#334` line. So FF, restrict, emission and coverage are all CORRECT for this
lump; **the corner-cylinder sub-face is lost DOWNSTREAM, in the face splitter or in classification,
not at section computation.** **ROOT FOUND — IT IS A CLASSIFICATION ERROR, NOT A MISSING FACE (#1228).** A new
`BK_SUBFACE_BOX=x0,x1,y0,y1,z0,z1` probe (in `builder/mod.rs`, reports every sub-face touching the
box with surface kind, source, `FaceClass`, rank, selection and extent) diffed the SAME corner
between a working and a broken band. On **tool0 (works)** the inner corner cylinder `Id(72)` yields
8 tiny **Inside** notch slivers at x[17.000,17.050] — the 0.05mm bands #1224 restored, correctly
removed — plus ONE big **Outside** remainder x[17.000,19.550] y[−19.550,−17.000] z[1.200,20.300]
that is SELECTED. On **tool1 (broken)** the identically-extented full remainder
x[17.000,19.550] y[−19.550,−17.000] z[1.200,20.300] is classified **Inside** and DROPPED, taking the
whole inner corner wall with it and leaving the tool's cut-surface patch with nothing to pair
against. So the splitter IS producing the face and #1227's "missing from the selection" stands, but
the mechanism is misclassification, NOT a failure to create. Prime suspect, and it is a KNOWN class
here: the sub-face's interior sample point. See the a1corner root ("splitter interior points of
notched/symmetric pieces land on feature-plane intersections BY CONSTRUCTION; classification must
survive on-plane sample points", `classifier/ray_cast.rs` per-ray degeneracy re-cast). **STOP — THE WHOLE ODD-BAND LINE OF INVESTIGATION WAS GIGO. THE ODD TOOL OPERANDS ARE NOT
WATERTIGHT.** Adding free/over-edge counts to the replay's operand `describe()` shows the failure
split IS the watertightness split: tool0/2/4/6 (the bands that work) are `free=0 over=0`; tool1/3/5/7
(every failing band) are `free=405 over=38`, `free=383 over=34`, `free=367 over=42`, `free=392
over=29`. Ray-cast parity against a non-closed solid is UNDEFINED, which fully explains the chain
below — the "1 crossing" rays, the misclassified corner cylinder, the dropped inner wall and the open
growth shell are all downstream of a malformed operand. This is exactly the documented trap
("captured operands are fallback-poisoned, never replay them directly", the snapClip note) and a
whole iteration was spent inside it. **ALWAYS print operand free/over counts before diagnosing a
replayed capture; the harness now does this unconditionally.** OPEN QUESTION, and the real next
step: is the breakage in the CAPTURE or does the tool genuinely build open lattice bands? If the
latter it is a real defect, but upstream of GFA — re-capture the odd bands from a current build
before spending anything further. TWO FACTS THAT NARROW IT ALREADY, both cheap and already measured:
(1) **they are NOT mesh-fallback output** — the odd tools carry 726/737/714/764 PLANAR faces against
663 for the working ones, comparable magnitude, whereas a mesh co-refinement fallback on this
geometry would be a triangle soup in the thousands. So the "fallback-poisoned capture" explanation
does not fit and should not be assumed. (2) they are non-manifold as well as open (`over` 29–42
alongside `free` 367–405), which reads as overlapping or duplicated faces — a mis-assembled analytic
shell, not a truncated or lossy capture. Note also that serialization is unlikely to be the culprit:
the SAME `arena_io` round-trip produced perfectly watertight even bands in the same capture
directory. **THE DECISIVE TEST IS DONE AND THE ANSWER IS: A REAL DEFECT, UPSTREAM OF THE GFA CUT.** The old
captures were 2026-07-24 from published 2.128.2 (pre-#1224), so the odd bands were re-captured on a
local 2.128.5 build carrying the fix (overlay md5-verified in BOTH tool `node_modules` locations;
`gomaCaptureBisect.test.ts` in the tool's `__kernel-tests__/`, 294s; fresh operands in
`~/.cache/brepkit-parity-captures/2026-07-25/goma-bisect/`). **The odd bands are STILL open and
non-manifold**: tool1 free=405 over=38 (identical to the old capture), tool3 393/33, tool5 386/36,
tool7 428/40 — while tool0/2/4/6 stay a clean 0/0. Face counts shifted slightly between captures
(tool5 714→690, tool7 764→792) so construction is not bit-identical, yet the brokenness reproduces
every time. That kills BOTH remaining benign explanations: not a stale/bad fixture, and not a
nondeterministic flake. So the DIAGONAL kumiko lattice bands really do arrive at the cut as
malformed solids — the cut is being handed garbage. (Where that garbage comes from is answered
further down, and the answer is NOT "the tool built it": the bands are brepkit's own mesh-fallback
output. An intermediate note here said the failures "were never a defect in the boolean engine";
that is retracted — they are not a defect in GFA's ANALYTIC path, but they are a brepkit defect, in
the mesh fallback.) CONSTRUCTION TRACED (`kumikoWrapBuilder.ts` in the tool): a band is the FUSE of dozens of struts —
vertical struts as small-angle revolves, near-horizontal as thin partial revolves, **rising diagonals
as `sketchHelix(...).sweepSketch(rect, {frenet:true})`**, and falling diagonals as chord boxes (a
left-handed helix is unsupported). The odd/even split maps onto that: revolve-built bands are clean,
diagonal-bearing bands are open. **HELIX SWEEP REFUTED as the direct cause** — `helical_sweep` is
watertight (free=0 over=0) at every turn count 0.25–2.0 and segment density 4/8/16 at the bin's
r=3.75; now pinned by `helical_sweep_is_watertight_across_turns_and_segments` in `helix.rs`. So a
single strut is not the problem. NEXT: the band is ASSEMBLED by fusing those struts, so chase the FUSE, not the sweep. **IT IS THE KNOWN MESH-FALLBACK DEFECT — "Mesh-boolean fallback
emits OPEN meshes that get CONSUMED" (open since 2026-07-16).** The full chain is now code-confirmed:
an analytic fuse that fails the STRICT gate (`validate_boolean_result` in `boolean/assembly.rs`,
which does reject `free_edges > 0` per #1192) drops to `mesh_boolean_fallback`; that function checks
`boundary_edge_count`/`non_manifold_edge_count`, `log::warn!`s that the output is not a closed
2-manifold, and then falls straight through to `mesh_result_to_face_specs` and uses it anyway; the
result is finally checked by `validate_boolean_result_lenient`, which by design rejects only
degenerate topology (too few faces, zero edges/vertices) because it is the terminal check with no
fallback left. So an open mesh result is accepted, exactly as the open item says.
**CORRECTION — an earlier read of mine said the bands were "almost certainly NOT that path's output"
because 726/737/690/792 planar faces is too few for a triangle soup. That was WRONG, twice over:**
(1) `mesh_boolean_fallback` runs `unify_faces` on its output, merging coplanar triangles back into
polygons, so a mesh-derived band lands at a modest planar face count; and (2) the decisive tell was
sitting in the operand dump the whole time — **EVERY band is 100% planar, the clean ones included**
(`mix=[("plane", 663)]` and friends, zero cylinders or cones), while the base carries 12 cones and
24 cylinders. A band built from revolves and helix sweeps MUST have cylindrical strut surfaces
wrapping the corner; their total absence is the parity skill's canonical fallback tell ("all-planar
with zero curved surfaces on a shape that should have cylinders is fallback regardless of the
number"). **CAVEAT, measured afterwards — the tell is weaker than it first reads and rests
specifically on the `revolve` struts.** `helical_sweep` output is all-planar BY CONSTRUCTION (the
helix is a NURBS approximated by segments and the sweep emits planar quads: `turns=0.25 segs=8`
gives `F=42 mix=[("plane", 42)]`), so the helix-swept diagonals contribute no curved faces even on a
perfectly analytic path, and neither do the chord-box falling diagonals. What still makes
zero-curved damning is that `revolve` with a proper axis-containing profile DOES emit cylinders
(census row `revolve cylinder (parallel→Cyl, caps→Plane)`, `cyl=1`), so each vertical and
near-horizontal `revolve` strut should leave a cylindrical wall in the band, and none survives. DO
NOT try to confirm this with a hand-built strut fuse without checking the profile plane first: a
probe using `make_unit_square_face` (a unit square in the XY plane) revolved about Z is DEGENERATE —
the profile is perpendicular to the axis, and it returns `F=6` all-planar, which looks like a result
and is not one. All eight bands are mesh-fallback output; four happened to come out watertight and four
did not.

**HOW A BAND IS ACTUALLY BUILT — it is a CUT, not a fuse (correcting an earlier framing here).**
`kumikoWrapBuilder.ts` starts from a `wedge` (a `revolve`, so it HAS cylindrical faces) and
iteratively carves it: `for (const family of families) cutter = cutAll(cutter, family)`. The flat-wall
span does the same with a box region (line ~413). `cutAll` is `compound_cut`, which delegates to
`boolean(Cut, …)` per tool (batched via `cluster_tools_by_aabb` + `fuse_cluster`, else sequential),
so every step runs the strict-gate-then-mesh-fallback path.

**FLAT-WALL STRUT CUT IS HEALTHY — measured, do not re-probe it.** An existing capture,
`~/.cache/brepkit-parity-captures/2026-07-23/kumiko-goma/` (`cut1-region.bin` + 180 `cut1-tool<i>.bin`;
replay it by symlinking `cut1-region.bin` to `cut1-base.bin` and using `PREFIX=cut1`), replays clean:
region F=147 all-planar free=0 over=0, every strut F=6 free=0 over=0, result **F=1146 all-planar,
free=0 over=0, 11.6s** — and F=1146 matches the "each ~F=1146" figure in `gomaCaptureBisect`'s own
doc comment. NOTE all-planar is CORRECT here and is NOT a fallback tell: a box slab cut by box prisms
has no curved surfaces to lose. So this capture cannot exercise the suspect path.
**ROOT FOUND, AND IT IS A 2ms SIX-FACE REPRO — fixture
`crates/io/tests/kumiko_corner_wedge_inmem.rs`.** The corner-wedge `cutAll` was captured
(`kumikoCornerCutCapture.test.ts`, modelled on `gomaCaptureBisect`, HEIGHT=4, 295s; six calls in
`~/.cache/brepkit-parity-captures/2026-07-25/kumiko-corner/`, replay each with `PREFIX=cut`). Calls
0–3 are the flat-wall path, all planar in and out and all clean, ending at the good F=663 band.
**Call 4 is the corner wedge and it is the defect:** base `F=6 mix=[("cylinder", 2), ("plane", 4)]`
free=0 over=0, five strut tools each identically `F=6` with 2 cylinders and watertight, result
**`F=71 mix=[("plane", 71)] free=2 over=1`**. Bisecting by tool count separates TWO failures:
`1 tool → F=60 ALL-PLANAR free=0` (the cylinders are already gone at the FIRST cut, in 2ms),
`2 → F=68 free=0`, `3 → F=72 free=3`, `4 → free=2 over=1`, `5 → free=2 over=1`. So (a) a COAXIAL
wedge×wedge cut — six analytic faces against six, two cylinders each, same corner axis — drops to
the mesh fallback immediately, and (b) openness appears from the third strut on. **The kumiko root
is therefore NOT that `mesh_boolean_fallback` consumes open output (it does, and that remains a real
defect); it is that these coaxial wedge cuts fall back AT ALL.** Fix that and no band is
mesh-derived, which is what the whole family is downstream of. **AND THE REASON IT FALLS BACK IS ALREADY MEASURED:** raw GFA on that exact pair
(`RAW=1 TOOL=0 SHELL_LOG=1`) reports `BuilderSolid: 0 growth shells, 1 hole shells` and aborts with
**"no outer shell found (all shells classified as holes)" in 0ms**. The analytic path does not
produce a wrong result — it produces NO result: one shell is built and classified INWARD, so nothing
remains to be the outer shell. For `Cut(wedge, strut)` the answer should be a single OUTWARD shell,
so the suspect is orientation or the growth-vs-hole decision in `perform_areas`
(`crates/algo/src/builder/builder_solid.rs`), NOT the face splitter. **INSTRUMENTED (`BK_AREAS=1`,
new): the growth/hole REJECTION IS CORRECT — what feeds it is wrong.** The lone shell reports
`faces=6 mix=[("cylinder", 2), ("plane", 4)] signed_vol=-182.448 lone=true outward=Some(false)`, so
both the corner-fan volume AND the curvature-robust flux test agree it is inward, and rejecting it is
right. But look at what it IS: six faces with the SAME surface signature as each operand (both are
`F=6`, 2 cylinders + 4 planes; wedge vol 284.873, strut vol 43.971), and a corner-fan magnitude
closer to the wedge than the strut. **The strut never split the wedge at all — a whole operand came
through, inverted.** A partial cut must produce MORE than six faces. **ROOT NAILED: BOTH SHELL-ORIENTATION TESTS MIS-READ A CORNER WEDGE
BOUNDED BY TWO COAXIAL CYLINDERS.** Decisive route: `BBOX=1` shows tool0 is z[-8.307,2.192] while the
base is z[2.700,20.800] — **fully DISJOINT** (tool3 likewise, above). For a disjoint `Cut(A,B)` GFA
correctly keeps all of A's faces and none of B's, so the resulting 6-face shell IS the base wedge
unmodified — and `BK_AREAS` reports it `signed_vol=-182.448 outward=Some(false) -> hole`, while
`solid_volume` measures that same wedge at **+284.873**. Both cannot be right and the
tessellation-based volume is the trustworthy one, so the corner-fan `signed_volume_of_shell` AND the
curvature-robust `shell_is_outward_oriented` are BOTH wrong on this geometry. The signature is
unmistakable: `-182.448` recurs IDENTICALLY across tool0 (6 faces), tool2 (6 faces) and tool4 (5
faces) — three different shells cannot share a volume integral. Per-tool results (each tool cut
against the base alone): all five fail "no outer shell found", tool0/tool3 with no sections at all
(disjoint, correctly AABB-rejected) and tool1/tool2 with sections emitted, so the failure is
downstream of sectioning in every case. NOTE the ops-level shortcut only detects CONTAINMENT
(A⊂B, B⊂A, A=B) — **disjointness is not handled**, which is why a non-touching strut still routes
through GFA and then the mesh fallback. **FLUX BREAKDOWN (`BK_FLUX=1`, new) NARROWS IT TO ONE UNRESOLVED FORK — resolve
this BEFORE writing any fix.** Per-face on the disjoint case: `Id(0) plane d=0.0000`,
`Id(4) cylinder d=+68.3065`, `Id(3) plane reversed=true d=-320.9000`,
`Id(2) cylinder reversed=true d=-641.4838`, `Id(1) plane d=+41.6553`, `Id(5) plane d=0.0000`,
**total = -852.42**. Compare 3V = 3 × 284.873 = **+854.6**: the magnitudes agree to 0.3% and the SIGN
is opposite. So the geometry is integrated correctly and the orientation is GLOBALLY inverted — this
is not a per-face `is_reversed` slip (flipping only the reversed faces gives ±1072.4, not ±855).
THE FORK: either (a) the flux convention is inverted and `shell_is_outward_oriented` is wrong, or
(b) **the captured base operand is genuinely inward-oriented**, the flux test is right, and the
defect is upstream in how the tool's wedge is built or serialized. Evidence does NOT yet separate
them: `solid_volume` returns `total.abs()` so its +284.873 says nothing about orientation, and
`classify_point` is ray-parity based so its Inside/Outside verdicts (material between r=1.55 and
r=4.75, hollow inside — geometrically correct) are orientation-independent too. `brepkit-check`'s
`integrate_face` DOES accumulate a signed volume but check is out of `brepkit-io`'s layer.
**FORK RESOLVED — (a) IS DEAD, THE ORIENTATION TESTS ARE CORRECT.** The discriminating experiment
(`crates/operations/examples/flux_orientation_probe.rs`: `Cut(box, far-away disjoint box)`, whose
result shell is just the box, run with `BK_AREAS=1 BK_FLUX=1`) gives for a 10×10×10 cube:
`signed_vol=1000.000000`, per-face flux `0/0/0/1000/1000/1000` totalling **3000.0 = 3V**,
`outward=Some(true) -> growth`. Both tests are EXACT and correctly signed on known-good geometry, so
the convention is not inverted and `shell_is_outward_oriented` needs no fix. **Therefore the
captured wedge operand is GENUINELY INWARD-ORIENTED, and the defect is upstream of GFA entirely.**
Do NOT "fix" `signed_volume_of_shell` or `shell_is_outward_oriented` — they are provably right.
**ROOT CAUSE FOUND, FIXED, AND FIXTURED — THE SEGMENTED `revolve` PATH DID NOT NORMALIZE THE
ORIENTATION OF THE SOLID IT EMITS.** Fix in `crates/operations/src/revolve.rs`
(`profile_chart_is_ccw` + traversal reversal at the segmented path's entrance); fixtures
`crates/operations/tests/regress_kumiko_corner_wedge.rs` (native wedges, both the overlapping and
the disjoint cut) and `revolve::tests::revolve_segmented_{is_outward,full_turn_is_outward}_for_either_winding`.
The sweep runs in `+θ = axis × e_r`, so a profile wound CCW in the (radial, axial) chart faces
AGAINST it and revolves into a consistently-wound but globally INVERTED solid (Pappus: the swept
signed volume carries the chart's signed area). `try_analytic_full_revolution` derives each face's
material-outward side from that same shoelace sign; the segmented path built faces straight from
traversal order, so it now normalizes the TRAVERSAL instead (reverse the oriented edges and negate
the cap normal when the chart shoelace is CCW; a degenerate chart is left alone). WIDER THAN THE DIG
RECORDED: the bug hits segmented FULL revolutions too (a profile whose surface is non-planar, so the
analytic path declines) — `−112.833` for a `+113.1` washer — the earlier "full revolution is correct
for both windings" reading only held because analytic profiles never reach the segmented path.
NEW ORACLE, and the reason this survived so long: `measure::solid_volume` returns a MAGNITUDE, so no
existing revolve test could see an inverted shell. `measure::oriented_solid_volume` (new, public)
keeps the divergence-theorem sign — positive iff outward. Reach for it in any orientation
assertion; the pre-fix wedges read `−143.148`.
STALE FIXTURE, deliberately still ignored: `crates/io/tests/kumiko_corner_wedge_inmem.rs` replays
operands captured PRE-FIX, i.e. already-inverted wedges that nothing downstream can recover. Its new
`captured_operands_are_inward_oriented` pins exactly that (and self-reports when a re-capture lands).
TOOL-SIDE RE-PROBE STILL OWED — see the mandatory-re-probe note below; the engine claim is closed,
the export-integrity numbers are not.
Historical dig record follows. Probe: `crates/io/examples/arena_roundtrip_orientation.rs`. The
arena round-trip is INNOCENT (a serialize/deserialize copy of a box gives the same
`signed_vol=1000.000000 outward=Some(true) -> growth` as the native one). But revolving a wedge
profile (rectangle in the XZ plane, r 1.55→4.75, z 2.7→20.8, about Z, 45°) across all four
combinations of profile winding and stored plane normal gives: ccw/−Y → `signed_vol=-129.010` hole,
disjoint cut F=48 (mesh fallback); ccw/+Y → `-129.010` hole, F=48; cw/−Y → `+129.010` growth,
**F=6 analytic**; cw/+Y → `+129.010` growth, **F=6 analytic**. So the PROFILE WINDING alone decides
the result's orientation and the face's stored `normal` is IGNORED. Wound one way `revolve` emits a
proper outward solid whose disjoint cut stays analytic at six faces; wound the other it emits an
INWARD solid that fails every downstream orientation check and drops to the mesh fallback at 48
planar faces. The tool's wedge is wound the inward way, which is why every corner cut in every
kumiko band degrades — and why the flat-wall path, which has no revolve, is clean. **SCOPED FURTHER — IT IS AN INCONSISTENCY BETWEEN REVOLVE'S TWO PATHS, NOT A MISSING CAPABILITY.**
Same probe, adding a 360° case: **FULL revolution is CORRECT for BOTH windings** (ccw and cw both
`outward=Some(true) -> growth`, F=4, a washer of 2 cylinders + 2 planes), while the PARTIAL 45°
revolve is wrong for one (ccw → hole → F=48 mesh; cw → growth → F=6 analytic). So
`try_analytic_full_revolution` (gated on `is_full`) ALREADY normalizes orientation — its doc comment
says as much, "face orientation comes from the profile's traversal winding in the (radial, axial)
chart, so inward-facing walls and cap normals are exact for both CCW- and CW-wound profiles" — and
the SEGMENTED path taken by partial revolutions never got the same treatment. The kumiko corner
wedge is a partial revolve, so it lands on the unfixed path. FIX: apply the full path's winding
normalization to the segmented path (compute the profile's signed area in the (radial, axial) chart
and reverse the oriented edges when it has the inward sign), with a working reference implementation
sitting in the same file. Incidental confirmation: the full washer reports `signed_vol=0.000000` and
is rescued by the flux test — exactly the lone-shell path `perform_areas` documents for the
torus−box notch band, so that path is load-bearing and must not be disturbed. Independently and still worth doing: short-circuit a disjoint Cut to A (the ops shortcut
detects only containment), which would spare tool0/tool3 the whole path. Reproduce per tool by symlinking
one `cut-tool<i>.bin` as `cut-tool0.bin` beside `cut-base.bin` and running
`PREFIX=cut RAW=1 TOOL=0 SHELL_LOG=1 BK_AREAS=1 BBOX=1`. Its sibling `operands_are_clean_analytic_wedges` runs unignored and guards the
fixture itself — an unvalidated operand already cost this campaign several passes.
FIRST ACTION FOR THE NEXT SESSION: the engine fix is landed and fixtured, so the open work is
MEASUREMENT — publish/overlay a build carrying it and re-run the tool's export-integrity matrix plus
the 4 baseplate suites (baseline to beat: 37 failed / 371 passed on local main; kumiko 14 failures on
one root; goma 1×1×6 at 844s with 2567 boundary edges). Expect the four diagonal bands to arrive
closed, which should take them off the mesh fallback and collapse the 844s. Do NOT quote a kumiko
closure before that re-probe. Two independent follow-ups remain, both untouched by the revolve fix:
short-circuit a DISJOINT Cut to A (the ops shortcut detects only containment, so a non-touching strut
still routes through GFA — tool0/tool3 are exactly this), and decide the product call on
`mesh_boolean_fallback`, which warns that its output is not a closed 2-manifold and consumes it
anyway (rejecting means the op fails outright, since there is no further fallback).

**MANDATORY POST-GFA RE-PROBE DONE, AND THE ANSWER IS THAT #1224 MOVED THE SCENARIO NOT AT ALL.**
`gomaBoundaryProbe` (goma 1×1×6 export + STL edge-use oracle) on the overlaid 2.128.5 build:
**844s wall-clock** (pre-fix baseline 849s — unchanged, still blows the 600s vitest timeout) and
**2567 boundary edges** on the exported STL (still not watertight). So the even-band fix, though
real and verified free=30→0 on tools 0/2/4/6 in isolation, buys NOTHING at scenario level, because
goma's cost and brokenness are dominated by the four malformed diagonal bands — an upstream
construction defect #1224 does not touch. **Do not quote #1224 as progress on the kumiko
export-integrity family; it is not.** The 14 kumiko failures, the 850s export and its
timeout-poisoning of later tests are ALL still open, and they will stay open until the diagonal
lattice bands are built as closed solids. That is now the single blocking item for this whole
family.

**RE-PROBE OF THE REVOLVE ORIENTATION FIX (2026-07-25, PR #1237) — HALF A RESULT, AND THE OTHER HALF
IS A REGRESSION THAT FIX CAUSES. DO NOT MERGE #1237 WITHOUT RESOLVING IT.** Overlay md5-verified in
BOTH `node_modules` locations, Vite cache cleared. CONTROL EXPERIMENT (the same local 2.128.5 build,
sole difference an `&& false` disabling the winding flip — the cleanest causation test available
here): fix DISABLED → goma 1×1×6 exports **OK in 851s, 2,715,884 bytes** (reproducing the recorded
849s/2.7MB baseline exactly, so the environment is validated); fix ENABLED → **THREW at 369s**. So
the orientation fix more than halves the cost (the analytic path replacing the mesh fallback, as
predicted) AND trips a trap. FAILURE SHAPE: `RuntimeError: unreachable` inside `compoundCut`, then
`recursive use of an object` (the borrow-flag stranding that follows ANY wasm trap — a consequence,
never the root). **IT IS NOT A RUST PANIC:** `lastPanicMessage()` is empty and no `[brepkit] panic:`
line is emitted, so the hook installed by `BrepKernel::new` never ran — which points at an
`abort()` such as `handle_alloc_error` (allocation failure bypasses the hook) rather than a `panic!`,
a `raw_vec` capacity overflow, or a divide-by-zero, all of which WOULD be captured. Note every panic
AND every abort lowers to `unreachable` on wasm32, so the trap kind identifies nothing by itself.
RULED OUT, each by measurement: (1) `assembly.rs:326`'s `unreachable!()` on
`FaceSpec::CylindricalFace` — tempting because the fix newly produces cylindrical faces where the
fallback emitted all-planar, but the outer arm at `:191` is unguarded so it is genuinely unreachable;
(2) a synthetic native repro — `compound_cut` of a wedge by 5 coaxial revolve struts is clean (F=26,
7 cylinders, free=0 over=0, Pappus volume); (3) the corner construction itself — the tool's
`kumikoWrapSpike` (annular wedge − vertical/horizontal/diagonal-helix struts, i.e. the exact failing
shape) PASSES in 2.5s, 4/4; (4) the SSI marcher as the unbounded allocator — it is capped four ways
(`max_steps = 200`, `max_evals`, `MAX_TURNING_POINTS`, `MAX_BRANCHES_PER_DIRECTION`). HEIGHT LADDER
(`gomaHeightLadder.test.ts`, new in the tool's `__kernel-tests__/`; ascending, stops at the first
trap because a trap strands the kernel): h=1 OK 1105ms/81kB, h=2 OK 150ms/178284B, h=3 OK
150ms/**178284B — byte-identical to h=2**, h=4 OK 489s/2.05MB, h=6 THREW 387s. **TWO CAVEATS ON THAT
TABLE, both mine:** h=2/h=3 being identical at 150ms means those heights do not exercise the lattice
at all and are evidence of nothing (real lattice work starts at h=4); and the ladder's
`wasmHeapMB` probe returned `?` at every height — the instrument did NOT fire, so there is still
ZERO memory data and the OOM hypothesis is UNTESTED. Also note the ladder reuses ONE kernel across
heights, so its h=4 489s carries accumulated arena state and is not comparable to the ~292s recorded
elsewhere; the h=6 trap is independently confirmed on two fresh-kernel runs. NO FAST REPRO YET —
h=4 is the smallest case that does lattice work and it costs 489s. NEXT: peak-RSS measurement
(`/usr/bin/time -v` around the h=6 run) discriminates the two live hypotheses without touching code —
allocation failure would show RSS climbing toward the wasm 4GB ceiling, while a wasm stack overflow
from deep recursion (equally scale-dependent, equally message-free) would show modest RSS. After
that, the sanctioned path is capturing the failing `compoundCut` operands and replaying them
natively, where an abort is legible. SEVERITY FRAMING: goma was ALREADY failing pre-fix (the 850s
export blows vitest's timeout and that single root accounts for all 14 kumiko export-integrity
failures), so this is a CHANGED FAILURE MODE on an already-failing scenario, not a newly-broken
passing one — but a trap is still a hard failure and blocks the merge. The engine-level defect and
its fixtures stand on their own (`revolve_segmented_is_outward_for_either_winding` and
`regress_kumiko_corner_wedge.rs` fail without the fix, at exactly the inverted volume).
**TRAP LOCATED, AND IT IS NOT IN THE ANALYTIC PATH — IT IS THE MESH FALLBACK.** Captured with
`setLogLevel('debug')` + a JS ring buffer over the 1,961,120 kernel log lines the export emits
(`gomaLogTail.test.ts`, new in the tool's `__kernel-tests__/`; the cjs keeps `wasm.memory`
module-local so heap size is NOT readable from JS — `setLogLevel` is the only handle). The last
lines before the abort: `BuilderSolid: 11 shells (sizes [23,17,189,14,44,158,34,23,103,26,4])` →
`assembled solid with 635 faces` → `Euler characteristic V-E+F = 12 is invalid (V=3588, E=4211,
F=635)` → `GFA result not accepted, falling back` → `boolean Cut: GFA unusable — using mesh
(co-refinement) fallback`, then the trap. So the chain is: the fix makes the band analytic → a later
Cut's GFA result has OPEN shells and is correctly rejected (11 CLOSED pieces would give euler 22, not
12) → `mesh_boolean_fallback` runs → **the fallback aborts**. TWO SEPARATE DEFECTS, only one caused
by the revolve fix: (1) GFA emits open shells on this op — the deeper geometry bug the fix newly
exposes, and the real target; (2) `mesh_boolean_fallback` can kill the whole kernel — a PRE-EXISTING
landmine (same function as the long-open "emits OPEN meshes that get CONSUMED" row). Mechanism for
(2), from `boolean/mod.rs::mesh_boolean_fallback`: it tessellates BOTH operands via
`tessellate_solid_for_boolean(.., deflection, 0.0)` — angular_tol 0 plus the circle curvature floor,
deliberately DENSE — then BVH+CDT co-refines. Pre-fix those operands were coarse mesh-derived planar
solids; post-fix they carry real cylinders and cones, so the triangle counts explode
(`mesh_boolean.rs:903` is `Vec::with_capacity(tri_count * 2)`). MEASUREMENTS BEHIND THE ALLOCATION
READING: host has 61GB with 23GB free (no physical pressure) and node's wasm ceiling is 4063MB;
peak process RSS on the trapping run is 2.74GB, i.e. BELOW that ceiling — which argues against
gradual growth reaching the cap and for a SINGLE oversized request, since a failed `memory.grow`
sends Rust to `handle_alloc_error` → `abort()` → `unreachable`, bypassing the panic hook exactly as
observed. This also explains why native replay looks clean: 61GB absorbs a multi-GB allocation that
is instantly fatal in a 4GB wasm heap, so this class of bug is INVISIBLE natively — reach for a
capped allocator or a wasm run when a native repro of an abort refuses to reproduce.
**ROOT FOUND — `Vec::reserve` DOUBLING THE TOPOLOGY ARENA, AND THE OOM READING ABOVE IS ONLY HALF
RIGHT.** Bisected by stage-bracketed `log::debug!` + `setLogLevel`, four rounds, each halving the
search: not a Rust panic (hook verified installed at `kernel.rs:57`, and the probe captures
`console.error`) → not the analytic path → not co-refinement SIZE (the failing op is a tiny
**7136 + 12 triangles, 497 intersecting pairs**, so the "single oversized allocation from
degenerate geometry" theory is REFUTED too) → `mesh_boolean` itself COMPLETES (5/5 stages, 7238
tris) → the abort is in **`assemble_solid_mixed`, inside `topo.reserve`**, whose preceding log
reads `arena before reserve V=3207122 E=5780703 W=2787018 F=2786863`. Mechanism: the arena never
reclaims (`dispose()` is a no-op), so a large export accumulates MILLIONS of entities; `Vec::reserve`
then rounds a 21714-edge bulk hint up to `max(2*capacity, len+additional)` — a full DOUBLING to
~11.5M edges — and the reallocation holds the old and new buffers at once, which fails against the
4GB wasm cap. So the peak-RSS reading was a symptom of accumulated arena, NOT of this operation.
FIXED in `crates/topology/src/arena.rs`: `reserve` now no-ops when capacity suffices and otherwise
grows by `additional.max(len / 8)` — geometric (total copying stays amortized O(n)) but capping the
transient overshoot at 12.5% instead of 100%. Plain `reserve_exact` was tried first and DOES clear
the abort, but leaves `capacity == len` so the next hint re-copies the whole arena — O(n²) across
thousands of booleans; test `reserve_hint_grows_a_large_arena_by_a_bounded_fraction` pins both
halves. MEASUREMENT TRAP that nearly produced a false perf claim: the `reserve_exact` run looked
catastrophically slower, but it ran at `LOG_LEVEL=debug` emitting **1.96 MILLION** log lines across
the wasm→JS boundary while the 851s control had logging OFF — never compare a debug-logged export
against a quiet one. DURABLE, and wider than kumiko: ANY brepkit export long enough to grow the
arena into the millions is one `reserve` away from a silent wasm abort, and the same cliff applies
to `Vec::push`'s own doubling. The real ceiling is architectural (no arena reclamation in a 4GB
address space); this fix buys headroom, it does not remove the cliff.
**SCENARIO VERDICT — THE ABORT IS GONE AND THE EXPORT COMPLETES, BUT GOMA IS 2.6x SLOWER THAN THE
CONTROL. THE KUMIKO FAMILY IS NOT CLOSED.** With both fixes (revolve orientation + arena growth),
goma 1×1×6 exports **OK in 2187s / 4,153,484 bytes**, against the fix-disabled control's **851s /
2,715,884 bytes**. Do NOT quote the revolve fix as kumiko progress at scenario level: the export
still blows vitest's 600s timeout (by 3.6x now instead of 1.4x), so all 14 kumiko failures and the
timeout-poisoning of later tests REMAIN. MECHANISM of the slowdown, and it is a direct consequence
of the fix: GFA still fails these ops (`assembly failed: open growth shell with N faces would be
dropped`, N=13/23/213/216, plus Euler and non-manifold rejects), so they still route to
`mesh_boolean_fallback` — but the operands are now ANALYTIC (cylinders + cones) where they used to
be coarse mesh-derived planes, and that fallback tessellates deliberately densely
(`tessellate_solid_for_boolean(.., deflection, 0.0)`, angular_tol 0 + circle curvature floor). Same
fallback, far more triangles: hence both the 2.6x wall-clock and the +53% STL. So making geometry
analytic makes the FALLBACK more expensive without removing the need for it. **THE BLOCKING ITEM IS
NOW UNAMBIGUOUS: the GFA open-shell defect.** Fix that and these ops stay analytic, the fallback is
never reached, and both the slowdown and the non-watertight output should go with it. Repro is cheap
and already in hand — `LOG_LEVEL=warn` on `gomaLogTail.test.ts` prints every rejection with its face
count; the smallest is `open growth shell with 13 faces`. NOT YET MEASURED: exported boundary-edge
count on the fixed kernel (needs `gomaBoundaryProbe`, control = 2567), and the 408-test
export-integrity matrix (baseline 37 failed / 371 passed) — the net effect across OTHER scenarios is
unknown, and the two engine fixes may well help elsewhere even as goma regresses.
**MATRIX MEASURED, A/B, AND THE ANSWER IS ZERO CHANGE — PLUS THE OLD BASELINE IS STALE.** Both runs
on the same day, same tooling, same 435 tests, differing ONLY in the two fixes (control = both
disabled in-place, overlay md5 2e58be3e vs treatment 4df3f2e3): **63 failed / 372 passed in BOTH**,
and diffing the failing test NAMES gives 63 of 63 identical — no treatment-only failure, no
control-only failure. Durations match too (2440s vs 2463s), because goma times out under both. So
PR #1237 is scenario-neutral: it fixes two real engine defects and changes nothing in the matrix,
neither regressing nor improving it. **CORRECT THE BASELINE: it is 63 failed / 372 passed over 435
tests, NOT the 37/371 over 408 recorded above** — the tool's catalog grew by 27 scenarios, and the
two families that look like new regressions (`divider patterns` 15, `floor patterns` 11, = 26) are
pre-existing failures in those NEW scenarios. Comparing a fresh run against the 37/371 figure
manufactures 26 phantom regressions; that number is only valid against the 408-test catalog it was
measured on. Current families: divider 15, kumiko 14, floor 11, permutation 6, solid cutouts 3,
wall/edge/custom-shape/combined 2 each, then singles. ALWAYS run the control on the SAME DAY and the
SAME catalog rather than trusting a recorded baseline — this is the second time this session that a
stale or mismatched reference nearly produced a false conclusion.
**THE TOOL'S `__kernel-tests__` PROBES ARE UNTRACKED AND GET CLEANED — RE-CREATE, DO NOT HUNT.**
`gomaPanicProbe`, `gomaCaptureBisect`, `gomaBisectProbe`, `gomaOpsProbe`, `gomaPhaseProbe`,
`gomaProfile` and the two written here (`gomaLogTail`, `gomaHeightLadder`) are/were UNTRACKED files
in `~/Git/gridfinity-layout-tool` — `git log` on them returns nothing. A `git clean` in that repo on
2026-07-26 removed all of them, so any roadmap entry naming one is a pointer to a file that may not
exist. Budget for re-writing the probe rather than treating its absence as a missing capability, and
check `git status`/recent commits in the tool before running anything there: it is a SEPARATE repo
that other sessions modify concurrently (two commits landed mid-run here). What survives a clean:
the capture `.bin` files under `~/.cache/brepkit-parity-captures/` and everything in this repo.
Recipe for the most useful one, ~40 min per goma run: `setLogLevel(level)` from `brepkit-wasm` is the
ONLY handle on kernel internals from JS (the cjs keeps `wasm.memory` module-local, so heap size is
NOT readable), a JS ring buffer over `console.log`/`warn`/`error` captures the output, and at
`warn` the whole goma 1×1×6 export emits just **1069 lines** — small enough to keep entirely, so use
a large KEEP and capture the chain from its START. **And note every `BK_*` knob (`BK_FLUX`,
`BK_AREAS`, `BK_OPEN_SHELL`, `BK_FF_TRACE`, …) is NATIVE-ONLY**: `std::env::var` returns `Err` on
wasm32-unknown-unknown, so passing them to a tool run silently does nothing.
**GFA IS THE SEED OF THE GOMA CHAIN, NOT THE MESH FALLBACK — AND THE BLOCKER IS THE MULTI-REGION
ACCEPTANCE GATE (2026-07-26).** Ordering from a complete warn-level log (1069 lines, untruncated):
first GFA rejection at line 11, first `open growth shell` at 28, first `mesh fallback output is NOT a
closed 2-manifold` at **65**. So the long-open open-mesh-consumption item AMPLIFIES the cascade but
does not start it; chase GFA. The rejected results are topologically fine (`validate=None`), so the
refusal is one term of the acceptance conjunction — and since both gates are conjunctions, the bare
rejection log says nothing. The `GFA reject detail` debug log added in that PR names the term; use it
first. TWO TERMS FIXED (branch `fix/gfa-multi-region-acceptance`, goma **2187s → 1123s** and STL
4,153,484 → 2,725,284 B, i.e. back within 0.35% of the control): (1)
`components_are_disjoint_pieces` tested AABB OVERLAP, which equals disjointness only for
axis-aligned pieces — two rotated bars a clear distance apart each span the whole diagonal envelope,
so a lattice could never be accepted (fixture
`tests::rotated_separate_pieces_are_recognised_as_disjoint`); now tests containment, since callers
reach it only after `closed_manifold` + balanced Euler, so pieces are closed and hence disjoint-or-
nested. (2) `euler_balanced` bounded the surplus at 2, assuming every component is a SPHERE — a
lattice cut yields RINGS and a closed loop is genus 1 (χ=0), so 13 pieces containing 6 rings give
χ=14 not 26; the bound is now `2·components` (fixture
`euler_balance_allows_genus_across_components`). The pre-unify check stays single-component
deliberately: it only decides whether `unify_faces` runs, and that pass can mangle a legitimate
N-piece result. **STILL OPEN, and the exact next step: 217 of 226 remaining rejections are
`disjoint=false`** (measured directly, not inferred) because AABB CONTAINMENT is also wrong for
rings — a ring's box contains the box of a separate piece sitting in its HOLE. Closing it needs a
REAL containment test; the read-only pieces exist (`tessellate::face::tessellate_with_uvs` +
`brepkit_math::ray_triangle::watertight_ray_triangle_intersect`, whose shared-edge guarantee suits
parity counting), gated behind the cheap AABB pre-filter. Do NOT reach for
`make_solid_from_face_subset`: it needs `&mut Topology` and would add temp solids per boolean to an
arena that never reclaims — the cliff fixed in #1237. REFUTED en route, do not re-chase: the Euler
validator in `validate.rs` (logged as a WARNING only, never gates — "hard-failing would reject ~25%
of currently working booleans"); absent multi-region acceptance (it exists and covers Cut); and
phantom components from position-duplicate edges (a position-keyed count matched the edge-id count
exactly, 13=13, 7=7, 23=23). TOOLING TRAP: `brepkit-render`'s `compute_mesh_lod` SIGSEGVs
intermittently — **2 of 4 runs with AND without changes**, a pre-existing flake that aborts
`cargo test --workspace` early and silently masks every later suite. Use
`--exclude brepkit-render`, and never conclude a regression from one stashed-vs-restored run.
**THIRD TERM FIXED AND THE GOMA CASCADE IS GONE — BUT THE MATRIX MOVED ONE TEST. KUMIKO IS STILL 14;
DO NOT CALL IT CLOSED.** `components_are_disjoint_pieces` now decides nesting by RAY PARITY against
the enclosing candidate's own tessellation (`component_encloses_point`), with AABB containment
demoted to a pre-filter — containment is necessary for nesting but not sufficient, because a ring's
box contains the box of a piece in its HOLE and a lattice is made of rings (that is why 217 of 226
were still `disjoint=false`). Fixtures pin the truth table: rotated-separate disjoint, nested not
disjoint, piece-in-a-ring's-hole disjoint. Read-only BY DESIGN — do not use
`make_solid_from_face_subset` (needs `&mut Topology`, adds temp solids per boolean to the
never-reclaiming arena, i.e. the #1237 cliff); uses `watertight_ray_triangle_intersect` (one hit per
shared edge, so parity works across faces) and a √-prime direction (these pieces sit on feature-plane
intersections where axis-aligned probes graze). goma standalone: **851s control → 2187s → 1123s →
499s with ZERO GFA rejections**, STL 2,128,934 B. **MATRIX: 63 → 62 failed of 435, kumiko unchanged
at 14.** The prediction "499s is inside the 600s timeout so the 14 clear" was WRONG — the same trap
recorded for #1224. WHAT DID CHANGE, and it is large: goma runs **140s inside the matrix** instead of
timing out, the rest of the kumiko block completes in **1–42s** instead of inheriting a poisoned
kernel (they now fail on their own merits, not as cascade casualties), and the residual failures are
small **boundary-edge** counts (5, 8, 12 observed) where goma previously exported **2567**. The count
barely moves because the assertion is zero-tolerance: 5 boundary edges fails exactly as hard as 2567.
NEXT: chase the residual handful of boundary edges per kumiko scenario — now isolated, reproducible
in seconds rather than a 14-minute timeout, and no longer masked by poisoning. Branch
`fix/gfa-multi-region-acceptance`, PR #1239.

Everything below this line about the odd bands describes behaviour observed on BROKEN INPUT and must
not be treated as an engine defect: **RETRACTED — the "GFA classifier misjudgement" reading (#1229)
is NOT established; the classifier was fed a non-closed solid.**

The probe now prints
each sub-face's `interior_point`, and a new `POINT_IN=x,y,z` knob on the replay classifies a point
against the base and every tool with the independent operations-level `classify_point`. The two
remainders' points differ: tool0 uses (17.244,−19.538,10.771) → GFA Outside, SELECTED; tool1 uses
(17.816,−19.416,8.070) → GFA Inside, DROPPED. But the oracle says that tool1 point is **Outside
tool1**, and the geometry agrees — z=8.070 sits BELOW the lattice opening at z[8.185,9.026]. So GFA's
classifier is wrong on this point; the face is correctly split and correctly sampled. This is the
a1corner degenerate-ray class (`classifier/ray_cast.rs`, per-ray degeneracy re-cast): the point lies
exactly ON the inner corner cylinder in a corner crowded with coincident feature planes. NEXT:
instrument GFA's ray-cast for that exact point on tool1 — which rays are cast, which hits are
counted, and whether the existing degeneracy re-cast fires. CAVEAT on the new knob: both interior
points classify oddly against the BASE (Inside vs Outside) when both lie exactly ON the inner
cylinder and should read OnBoundary — a tolerance artifact in `is_on_boundary`, harmless for the
tool verdict but the base column of `POINT_IN` is not trustworthy. CAVEAT on the probe: it tests
VERTICES against the box, so a large unsplit face whose corners sit outside the box will not
register — widen the box or add a face-bbox-overlap mode before concluding a face is absent. Each odd band hits a
DIFFERENT corner (tool1 → (+x,−y) z≈8.2–9.0;
tool3 → (+x,+y) x≈18.6–20.5 y≈+18.3–19.9 z≈6.5–9.8), consistent with one diagonal member per band.
CORRECTION, filed then retracted within the same session: an initial read called this the
near-duplicate-vertex/weld-band class and pointed at vertex minting. The near-duplicates ARE present
(Id(2721)/Id(2730) start at (17.809,−19.418,8.466) with a ~0.002mm unpaired ellipse to
(17.811,−19.418,8.465)), but the surrounding junction gaps run ~0.019–0.06mm — three to four orders
above the ~1e-6 marched-fit error that defines the weld-band class, so that filing is NOT supported.
Treat the micro-edges as a symptom of the unconnected tool/cylinder junction, not the root.
Secondary lead, the inner/outer
asymmetry: the inner
cylinder Id(6088) forms SEVEN notches and fails only at z 2.700–3.192, while the outer Id(5678)
forms five and fails at three bands — same tool, same openings, two concentric cylinders 1.2mm
apart, different outcomes. So the decision is per-face, not per-opening. The two bands where inner
succeeds but outer fails also show the cascade: at z 11.536–12.367 the inner notch IS built but its
x=17.050 line (e19454) is left FREE because the outer side never produced the partner patch. Also
worth explaining: every outer-cylinder edge sits in one fresh contiguous id block (e17930–e17972)
while the inner mixes a shared lower block (e18387–e18485, its formed notch ellipses) with fresh
e19445–e19459 — the outer's notch ellipses appear NOT to be the shared section edges the inner's
are, which may be the provenance difference behind the whole asymmetry. Everything
else in this chain is measured and solid — FF pairing, restrict, emission and the section map are
all confirmed working; three hypotheses (the empty-`pave_blocks` skip, a face-id mismatch, and
cylinder mis-wiring via a duplicated inner wire) are REFUTED. WARNING: `log::debug!` inside `fill_images_faces.rs` does NOT emit — an
adjacent `log::debug!` and `eprintln!` on the same line gave **0 vs 890** records, while
`builder_solid`'s `log::debug!` reaches the same logger fine. Log-based probes in that file read as
a FALSE ZERO. Use a temporary `eprintln!` gated on an env var (it trips `clippy::print_stderr`, so
do not commit it). Cause not diagnosed.
REFUTED, do not re-attempt: a panic (none; `lastPanicMessage()` is empty), bisect thrash
(telemetry: 1 attempt, 1 success), prism construction (322ms, 0.1%), boolean batching as the main
cost (30%), classification (defect is op-independent), and the assembly sliver-drop guard (tool0
drops nothing). TOOLING: V8 `--cpu-prof` does NOT work through vitest here (fork pool drops it via
NODE_OPTIONS and execArgv; no vite-node) — use `vi.mock` wrapping, and verify the wrapper covers
the path you are claiming about. Other 7 bands + harness:
`~/.cache/brepkit-parity-captures/2026-07-24/goma-bisect/` + `crates/io/examples/replay_cut_capture.rs`
(`RAW`, `TOOL`, `OP`, `FREE_LOOPS`, `LOOP_GEOM`, `XSCAN`, `SHELL_LOG`, `CHUNK`).

CAUTION on counting: the baseline's classified kinds (23 boundary-edge, 2 non-manifold, 4 poisoning,
1 timeout) sum to 30 of 43, so ~13 failures carry no recognised error form — probably cascade
casualties, unconfirmed. Never quote the raw failure count as a defect count.

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
