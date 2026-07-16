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
fractional tile passes; with the tangency-nub PR (#1075) the suite reaches
9/9. DURABLE: stored `start_uv`/pcurves on hole wires can be fitted in a
FOREIGN frame — any consumer building polygons from them must re-derive via
frame.project(3D) (same class as the pcurve-convention lesson).

Fresh baseplate re-probe on PUBLISHED 2.126.2 (2026-07-16, overlay md5-verified):
dovetailKey 2/2 and fit-offset 2/2 CONFIRMED on the published build; dovetail
7/9 @188s (was 6/9 @355s) — the 4×4-interior doubled-dovetail (the relief-cut
family) passes end-to-end tool-side; snapClip 0/4 with ALL signatures moved
since the mesh-boolean rewrite (join nm 14→4, key nm 12→0 but bnd=326, 0.6mm
nozzle nm 1→15, clip volume 46.78→46.70 vs 46.6±0.05). Dovetail residuals:

- **2×2 A1-canonical doubled-dovetail nm=2 — ENGINE FIX COMPLETE (2026-07-16,
  fixture `crates/io/tests/dovetail_dblcorner_nub_inmem.rs`, tool re-probe
  pending).** The paired tongue sits offset by exactly the socket corner
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
- **Mesh-boolean fallback emits OPEN meshes that get CONSUMED — OPEN
  (discovered 2026-07-16):** on the dblcorner nub operands the co-refinement
  fallback produced bnd=5/6 output (warn-logged, then used anyway, poisoning
  every downstream boolean into a 1400-face fallback export). The safety net
  must be watertight or rejected. Repro: the dblcorner fixture operands with
  the analytic path disabled, or any pre-fix build.

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
