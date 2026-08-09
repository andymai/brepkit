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

Head-to-head (wasm vs wasm, 2026-08-06, current kernel hash-verified overlaid into the bench harness — NOTE: the harness's vitest resolve.alias does NOT reach the CJS require path, so overlay node_modules and hash-verify; an alias-only run silently benches the installed kernel): brepkit leads every row — fuse(box,box) 87x, cut(box,cyl) 2.3x, chamfer 27x, fillet 21x, multi-boolean 6.4x, mesh box 5x, mesh sphere 7.3x (was 1.4x; the display-sphere curvature-floor removal, #1389 — sphere density now matches the reference at equal tolerance, 9,800 vs 10,176 tris), transforms ~4x, volume 47x (re-measured 2026-08-07 on released 2.129.13, hash-verified overlay in a brepjs bench worktree: 0.176 ms vs 8.31 ms per 100 calls — the exact all-planar divergence path #1396; was 2.3x), bbox 3x, STEP export 16x. Equivalence verified per row (volume oracle both kernels): fuse/chamfer/sphere exact, cut/fillet/multi-boolean within 0.004%; the intersect row is RE-QUALIFIED at 117x (2026-08-07, released 2.129.14 overlay: 0.597 ms vs 69.6 ms — the octant shortcut + classifier fix #1408/#1410; correctness pinned by bench_equiv_intersect_box_corner_sphere_is_the_octant). EVERY head-to-head row now leads. Native criterion CAVEAT: the cad_operations "mesh sphere" case tessellates PER-FACE through a bench-local shim (boundary-locked, ~40x lighter than the solid-level path the wasm row exercises) — never compare it against solid-level numbers; perf_probe measures the matching-parameter native figure. Where parity stands: gridfinity bin parity reached; all four primitive-boolean
mesh-fallbacks are exact analytic and beat the reference kernel 2.9-9.5x head to head;
revolve is exact-analytic; the GPU render crate shipped through screen-space LOD. Per-PR
history lives in git — do not re-record it here.

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

Regenerate the `#[ignore]` inventory (command above) and reconcile before trusting this table.
**Every closed case below is one line plus a pointer. The measured detail lives in the fixture's
doc comment — read that, not this.**

| Item | Layer | Repro / first probe |
|---|---|---|
| **Thick-wall cavity — CLOSED 2026-08-04** | operations/shell | See the dedicated entry below: the residual root was a SECOND missing-face defect in the same shell arm, and every measured case is now bnd=0 | |
| **Bench-found intersect(corner box, center sphere) — CLOSED 2026-08-07 (three stacked roots)** | operations | (1) the analytic octant shortcut (build_box_sphere_octant) built its cut arcs on circles with OUTWARD plane normals, storing the 270-degree COMPLEMENTS (arc midpoints on the far side; the 1304.8 volume) — inward normals give the intended quarters; (2) the sphere-patch wire traversed all arcs the same sense as the discs (3 same-sense edges, pre-existing) — the patch now reverses them; (3) classify's sphere arm assumed a PLANAR boundary polygon, but an octant patch's arcs span three planes and face_polygon samples only the three coplanar CORNERS, so the planar containment discarded every genuine hit (probe points read Outside even on the GFA-built octant) — non-planar-boundary patches (arcs in 2+ distinct planes) now use per-arc plane HALF-SPACE containment with the side calibrated from the boundary centroid pushed onto the sphere. GFA itself was already correct (fixed as collateral of #1394). Pin ACTIVE: bench_equiv_intersect_box_corner_sphere_is_the_octant. The head-to-head intersect row can be re-qualified |
| **Bench-found cut(box,cyl) 2.3% deviation — CLOSED same day (2026-08-05)** | operations/tessellate+measure | The geometry was EXACT; a three-site endpoint-exclusion class in boundary samplers dropped POLYGON CORNER vertices: a reversed edge's `(0..n).rev()` iteration excluded t_end = its traversal START, which no neighbour supplies, so the outline shortcut the corner with a chord whose area bite scales with the NEIGHBOUR edge's length (0.53 area from a 256-sample arc). Sites: measure/helpers `sample_edge_curve` (endpoint-inclusive now), tessellate/planar + tessellate/edge_sampling (reversed iteration is `(1..=n).rev()`, count-preserving). The volume ran through the direct-face-tessellation arm (a cut's reversed cylinder wall routes there) making it deflection-invariant. Pin ACTIVE: `bench_equiv_cut_box_corner_cylinder_volume_is_exact` (929.3231 vs exact 929.3141). Probe recipe that cracked it: chain the per-face mesh's own boundary edges and shoelace the outline — outline≠polygon means the sampler, not the CDT. SIBLING CLOSED same session: check::properties' planar fan triangulation used UNSIGNED triangle areas, so a fan over a non-convex notched cap counted the notch positively (over-count varying with vertex 0's position: top cap 98.07, bottom 94.29 for the same 92.93 shape). Signed accumulation (projected on the face normal, whole-polygon flip for CW winding to keep the hole-subtraction contract) makes the fan exact: gauss volume 946.43 -> 929.32 |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** (open since 2026-07-16) | operations | `mesh_boolean_fallback` warns its output is not a closed 2-manifold and uses it anyway; there is no further fallback, so rejecting means the op fails outright. A product call, not just a fix. PARTIAL MITIGATION (2026-08-08): the fallback is now DETECTABLE — `boolean::mesh_fallback_count()` + wasm `meshFallbackCount()` let export pipelines snapshot-and-refuse (GH #1445's ask) |
| **GH issues #1445/#1446/#1447 (user-filed 2026-08-08): grouped-scoop non-watertight, boolean-heavy 6x aggregate slower, BooleanOptions ignored** | operations + tool chains | MEASURED 2026-08-08 (slots wall-pattern 3x3x5, probe in the issue comments): the generated solid is 7766 ALL-PLANAR faces and `unifyFaces` removes 0 — the mesh fallback fires early in the chain and every later cut grinds against the faceted B-Rep; that one root explains the 2x triangles, the 56x time, AND refutes "missing simplify" as the cause (simplify would be a no-op there). Shipped: fallback counter (see row above) + `fuse/cut/intersectWithOptions(simplify)` wasm bindings + batch `simplify` flag and `meshFallbackCount` op, so brepjs can stop warn-dropping `BooleanOptions.simplify` (#1447; brepjs adapter wiring still TODO). ROOT CAPTURED AND REDUCED (2026-08-08, fixture `crates/io/tests/slots_lipcone_cut_inmem.rs`): the whole slots carve is ONE compoundCut (60 rounded-corner slot prisms; 52.8 s of the 54 s generation) — the tool fuse succeeds ANALYTICALLY, then the final cut's GFA emits 540 free boundary edges (9/slot), validation rejects, mesh fallback. The 13 ms ATOM: every SINGLE slot cut leaves exactly 6 free edges — a closed hexagonal loop at the slot corner where the tool's corner cylinder crosses the bin's LIP CONE (the loop carries a plane-cone ellipse edge; x spans the wall thickness, z 26.6-27.4); one cone sub-face piece is dropped. Ready-repro pins watertight + cone-survival (the fallback is watertight but all-planar, so cone count discriminates). Capture gotcha added to the recipe: compoundCut tools arrive as a Uint32Array — Array.isArray misses them, ArrayBuffer.isView is required. SLOTS ROOT FIXED (2026-08-08, same-day dig): three overturned framings in one pass (dropped tool pieces were CORRECTLY Outside per the oracle; "missing lip-cone piece" was really the TOOL's corner-cylinder in-wall piece; the duplicate boundary-riding section was real but not the breaker — dedup shipped anyway as hardening). TRUE ROOT: the tool's corner cylinder must split 3 ways (outside/in-wall/cavity) at two wall circles + the taper ellipse meeting at degree-3 T-junctions, and the greedy walker's order-dependent successor grand-toured three regions into one slitted loop — on a NON-PERIODIC band no rescue arm existed (all DCEL rescues were u_periodic-gated). Fix: non-periodic band arm (greedy_broken gate, DCEL adopted only when strictly more loops AND absolutely clean — no periodic-aware relaxation). Measured: single slot free=6 -> 0 analytic; full 60-tool compoundCut 14.5 s mesh-fallback -> 752 ms ANALYTIC (F=546, 256 cyl + 12 cones, free=0; the fallback volume was also wrong by ~194). Fixture un-ignored. Full foils green (algo 210, io 63 suites, ops 819, gridfinity 27). SLOTS TOOL-SIDE VERIFIED on released 3.2.1 (2026-08-08): generation 53.7 s -> 3.07 s (17.5x), 546 analytic faces (matches the native replay exactly), wallPatterns scenario 7/7 in 28 s. Density residual CLOSED (2026-08-08 night): the CDT fallback's interior grid hardcoded the curvature floor, so every boolean-result cylinder/cone face carried a floored interior lattice the display path never needed (a quarter pillar wall meshed 117 tris where tolerance needs ~50). interior_grid_resolution now threads the caller's circle_floor and skips interior points entirely for developable faces on the display/export path (rims carry the u density; the surface is exact along the rulings; boolean path bit-identical). Single-slot cut 4100 -> 2324 tris at 0.01mm/5deg, cylinders avg 65 -> 32; density + watertight pin in the slots fixture; the mitsukude mesh-derived volume pin re-calibrated (density-only shift, leak discriminant intact). GROUPED-SCOOP (#1445) ROOT CAPTURED AND REDUCED (2026-08-08, NOT the slots class): the booleans are exonerated by chain replay — the openness is MINTED by `filletVariable` (the tool's adaptive scoop: 9 bottom-outline edges of the fused rect+circle tool, radii 2/0.6/0.39/0.26), which emits 6 unstitched NURBS walls -> free=44, propagated by every later boolean into the export (Cut/Fuse tolerate open INPUTS by design). Byte-exact 0 ms repro: `crates/io/tests/scoop_fillet_variable_inmem.rs` (ignored ready-repro) + `crates/io/examples/replay_fillet_variable.rs` (spec edges match by captured ENDPOINT pairs; tool-side midpoint params are not portable). Capture gotcha: the scoop uses `filletVariable(solid, json)` — a hook on `fillet` alone fires zero times. FILLET CAMPAIGN CLOSED (2026-08-08, 16 rounds, v2 free 56 -> 0 on the captured scoop repro; this PR ships the squashed diff): mixed original+contact loop rebuilds for every stripe-adjacent face (per-stripe infinite-line trims amputated non-convex territory; contact endpoints ON boundary edges split them — Line and NURBS with properly trimmed sub-curves — interior endpoints get notch bridges, corner junctions get offset-arc bridges); HORN-TORUS corner patches for equal-radius corners (major = tube = r about the unfilleted corner edge — sphere/flat-triangle patches were the wrong surface); ruled rational transition bands for mixed-radius junctions; pairwise corner patches at multi-edge vertices; circle-aware free-edge welding (stored-curve midpoints are phase-dependent for arcs — key on centre+radius+`|axis|`, antipodal pairs excluded per the merge-key lesson); collinear split-welds; exact plane fills for residual closed coplanar loops. fillet_variable retries v2 when its v1 result comes back open and adopts a CLOSED fully-blended v2 result (v1-clean cases byte-identical); the scoop fixture watertight pin is ACTIVE via the production path. Instruments kept (env-gated): BK_FORCE_V2, BK_PIECES, BK_CORNER_TRACE, BK_TRIM_TRACE, BK_SPLIT_PREPASS, BK_NOTCH_TRACE. TOOL-SIDE RE-MEASURE on released 3.2.3 (groupedScoop export suite): 1 of 7 cases fully PASSES (asymmetric W/D), most collapse substantially (47->23, 53->18, 33->6, 33->3), two hold (different-cut-depths 65, aggressive-radius 25) — the residual open spots are fillet configs the captured repro did not cover (rotated members, asymmetric W/D radii, aggressive near-max radii, multi-depth groups). RESIDUAL CENSUS DONE (2026-08-08, parameterized probe, all 6 cases captured + replayed natively): 4 of 6 residual fillet configs are ALREADY watertight on the shipped engine (incl. rotated members with 24 torus patches, asymmetric W/D) — their remaining export boundary edges come from elsewhere in those chains (next capture: hook the FULL chain per case, not just filletVariable). LAST 2 LEAKING CONFIGS CLOSED (2026-08-08, cases 2+3 free 7/6 -> 0; all 6 captured cases + the original replay free=0): the "junction-curve sharing at build time" framing DISSOLVED under the curve-detail dump — the lenses were three mechanical defects: (1) build_two_edge_patch's flat triangle CHORDED the stripe's terminal-section arc (chord+arc share endpoints and correctly never weld; fix = build_arc_apex_patch, a rational ruled arc-to-apex patch whose boundary is the SAME circle the wall's cross edge carries, so the weld pairs them); (2) a stripe end pinched to a point minted a zero-length cross edge no weld can pair (fix = skip minting; cross_end/cross_start are Option now); (3) the residual plane-fill pass required 3-4 edge chains and skipped 2-edge arc/chord lenses (a band's straight rail against a rebuilt face's bridge arc; fix = 2-edge fills taking the plane from the arc's own circle). The depth-step rim ring closed as part of the same pass. Fixtures depthstep_/aggressive_radius_fillet_variable_is_watertight ACTIVE via the production fillet_variable path (v1 -> v2 retry adopts the closed v2 result). V2 ORIENTATION EMISSION CLOSED (2026-08-08): the "tessellation-level residue" was mostly not tessellation — the v2 fillet emitted orientation-inconsistent faces (brep same-sense pairs 55/14/12/30/0/29 across the six captured cases; directed-half-edge tess_bnd conflated same-sense mesh pairs with true cracks). Two passes now run at the end of FilletBuilder::build: propagate_orientation (BFS from carried-over input faces, set_reversed so effective wire senses oppose across every shared edge) and normalize_face_normals (interior-left boundary-walk integral, CONVENTION-CALIBRATED against the seed faces — the walk convention is a property of the input solid, a CW-profile extrude walks interior-right, so an absolute Newell test flips valid trimmed caps; repair is the sense-preserving triple flip: reverse wire order + toggle senses + toggle reversal). Both passes touch ONLY faces built by the current pass (a prior fillet's NURBS wall arriving as input must not be re-judged — try_fillet_second_pass pin). Measured: brep same-sense 0 on all six cases; tess boundary 377/193/824/592/0/74 -> 153/72/77/59/0/3. Orientation pin folded into all four scoop fixtures (same_sense_pair_count == 0). CDT WHOLE-FACE WINDING VOTE (2026-08-08): the 8 horn-torus corner patches take the tessellate_nonplanar_cdt path (past the notch/two-rim/latitude arms) and the CDT's UV winding, internally consistent, can be INVERTED as a whole against the surface on a pinched parameterization; fix = area-weighted geometric-vs-surface-normal vote deciding ONE flip for the whole face (the per-triangle variant was REFUTED by measurement — near the pinch the sampled normal scatters and it made every case worse, 153->297). Mesh boundary counts 89/38/39/59/0/3 after the vote (from 377/193/824/592/0/74 at day start), then 15/19/20/21/0/3 with nm 0/75/0/0/0/0 after the PINCH-U UNWRAP FIX (2026-08-08 late) TAIL DOCUMENTED AND PARKED (2026-08-08 night): the residual counts are SUB-EXPORT-TOLERANCE (both tool groupedScoop suites pass 7/7 on stock kernels) and split into two classes deliberately not chased: (a) case2's nm=75 are triple-used mesh edges along the depth-step arc at z=12.55 — the pinch-shim double-cover encoding makes two coincident face meshes span the same region, so their shared rim edges necessarily carry 3 triangles; INHERENT to the shim encoding (the alternative is the TERMINAL face-split-at-pinch primitive); (b) the remaining bnd counts (15/19/20/21/0/3) are thin-wedge boundary-sample drops at tangent contacts (a face's CDT collapses the last sliver against a tangent boundary while the neighbour keeps the true samples; case6's 3-edge wedge at (-12.55,-6.55,z 11.75..13.5) is the type specimen, B-Rep exact). Re-open only if an export-level regression points here.: the dominant strip-drop class was the CDT's periodic-u unwrap steered the LONG way (270 degrees) by the horn pinch point — a point ON the torus axis has no meaningful ring angle, the projection returns an arbitrary u, and the unwrap left the UV loop unclosed by a full turn so remove_exterior ate the boundary strip; degenerate-locus points now take their predecessor's u (fix in tessellate_nonplanar_cdt Step 2a; the earlier duplicate-pairs reading of the census was a 3-decimal print artifact collapsing mirrored corners, and the planar merge-map keying change measured zero and was reverted). Remaining #1445 residuals: case2 nm=75 (pinch-shim double-cover meshes overlap by construction, likely inherent to that encoding), case1's 89+nm2 and case4's 59+nm18 (uncensused), case6's 3 (a thin-wedge boundary-sample drop at a tangent point, NOT a B-Rep defect - all edge endpoints are exact), and the tool-side full-chain re-measure after release. Captures live in the scratchpad per case (gscoop_case1..6). SHIPPING COMPLETE (2026-08-08): brepjs #1966 MERGED (simplify routed through fuse/cut/intersectWithOptions with feature detection + old-kernel fallback, meshFallbackCount typed, kernel pin 3.2.1, consumer project 4568 green) — #1447 closes at the next brepjs release; tool PR #3345 bumps brepkit-wasm to 3.2.1 (wallPatterns EXPORT 8/8 — the arm that carried the slots signature — scenario 7/7; groupedScoop export 6 fails unchanged, the open fillet root). #1424 (crates.io trusted publishing) stays Andy-only. PINCH-SHIM CUT CLASS CLOSED (2026-08-08): the case-1/case-6 "watertight fillet yet open cut" residual was the CUT, not the fillet — a zero-pinching corner fillet's horn torus touches the pocket floor tangentially, and the blend closes that tangent arc with a tiny corner face coincident with (contained in) the unsplit floor; within-rank SD dedup read the shim as #696 containment residue and dropped it, orphaning the tangent arc (torus side) and stub edges (wall side) into 12 free edges (11 ms repro; the FF-coplanar chord-imprint theory was refuted by trace — no z=11 coplanar pair even exists). Gate shipped in `detect_same_domain_with_shells`: a containment-matched duplicate whose edges serve sub-faces OUTSIDE its SD group is a structural shim and is KEPT; coextensive (identical edge-set) residue and boundary-isolated residue still drop, so the honeycomb and #696 pins hold. Case-1 AND case-6 cuts now free=0 analytic (all 8 horn tori survive). Fixtures: `crates/io/tests/gscoop_pinch_cut_inmem.rs` ACTIVE (full operations::boolean path) + detection pin `contained_shim_with_outside_edge_user_is_kept`. GH CAMPAIGN CLOSED (2026-08-08 evening, all on released 3.2.5 stock npm kernel): #1445 CLOSED — groupedScoop export 7/7 AND scenario 7/7 tool-side (was 6/7 failing); #1446 CLOSED — every worst-offender row now BEATS the issue's reference baseline (split-stress 516.5->13.0s vs 16.6, combinedFeatures 526.5->16.6s vs 19.0, puzzle 146.4->5.1s vs 5.9, groupedScoop 31.8->2.2s vs 1.5, snapClip 302->13.1s vs 17.0, winding 334->15.4s vs 18.8, dovetail 441->19.5s vs 25.6) — the tail was mesh-fallback dominated and died with the #1445 chain + slots roots; #1447 closed earlier via brepjs simplify wiring. #1424 stays Andy-only. The CDT whole-face winding vote (#1469) + perf nits (#1468/#1471) land in 3.2.6 |
| **Divider residual geometry — GEOMETRY CLOSED (re-probed 2026-08-04 on the post-campaign kernel)** | algo/GFA | All 15 divider-pattern scenarios pass their assertions, including the 3 historic defects (`mitsukude lattice on dividers` was bnd=6, `dividers + scoops keep the ramp footings solid` was bnd=4, `kumiko dividers perforate the compartment walls`): the kumiko campaign + blend closures reached them. TWO residuals, neither geometry: (1) RESOLVED 2026-08-05: `kumiko dividers` passes in 166.6 s against the 180 s budget on the post-#1343 kernel (the NURBS section-clip fix shaved ~17 s off the mitsukude compound cut); the underlying mesh-fallback correctness gap remains tracked in the mitsukude panel-cut row below; (2) CLOSED 2026-08-06 (tool #3262): the tool now pins brepjs 18.119.9 + brepkit-wasm 2.129.8 on stock pins — all 15 divider scenarios pass with NO overlay (103/103 with the 5-arm export matrix). The intersectCurves bounding-box dist patch was RE-TARGETED, not upstreamed: brepjs deliberately pins the cache-alive contract (a test asserts boxes survive intersectCurves for reuse), so the tool's eager-release stays a dist patch by design — re-target it on every brepjs bump | |
| **snapClip 0.6 mm-nozzle export chain — NO LIVE REPRO (re-measured 2026-08-05)** | algo/GFA | The op-cut-3 in-repo repro replays clean on main (snapclip_export_corner operands: free=0 over=0, 80 cones analytic, 43 ms; the plane-cone exact-circle-arc fix closed it, fixture `snapclip_export_corner_inmem.rs` ACTIVE). The architectural observation stands without a repro: marched FF sections on curved faces carry `pave_block_id=None` and bypass the pave machinery; if a new leak lands here, the canonical altitude is pave-block attachment at phase-FF/make_blocks — every face-splitter-level attempt broke calibrated chains |
| **Export matrix drift: CLOSED at export level 2026-08-05 (73/73) by the ray-cast conflict re-cast; two capture-era engine repros remain open** | algo/GFA + tool | The clean/suspicious conflict re-cast (ray_cast.rs, the O-shape root fix) flipped the WHOLE matrix green: all 452 export-integrity assertions pass on the fix kernel (md5-verified overlay). The O-shape case is closed root-to-export with an ACTIVE volume-pinned fixture. The slotted and mixed-detail cases pass at export level because the chains' upstream booleans now classify differently and no longer produce the aborting operands — but the CAPTURED operands still reproduced. SLOTTED ROOT CLOSED (2026-08-05): detect_same_domain's rank-agnostic geometric-containment pass grouped the no-lip body's cavity CEILING with its coplanar exterior top (a zero-thickness roof) and dropped the ceiling as #696 within-rank residue, orphaning the cavity walls. Extent and orientation discriminants both REFUTED by measurement (honeycomb's true residue matches them exactly); the shipped gate is SHELL MEMBERSHIP (cross-shell coincidence is structural, within-shell is residue) via detect_same_domain_with_shells. Fixture ACTIVE and volume-pinned. MIXED-DETAIL ROOT SPLIT AND MOSTLY CLOSED (2026-08-05): the 511 count was two defects — 395 missing-face edges (the z=5 floor tessellated to ZERO triangles: CDT flip recovery stalled on a last-ULP-tilted 33.5mm rail and RETURNED OK without the edge, so remove_exterior's flood erased the face; fixed by Steiner-midpoint bisection in recover_edge + flood barrier union + Steiner lifting/splicing in run_planar_cdt) and 116 half-edge winding errors, whose root is UPSTREAM (2026-08-05 measurement): the tessellator is faithful, but the captured BODY operand itself carries 20 same-sense edge pairs (reversal-corrected traversal check) around its quarter-socket faces while the assembly is clean — an earlier op of the export chain emits orientation-inconsistent faces. DISCOVERY (2026-08-05): the check already existed (check_shell_orientation) but the public operations validate wrapper never ran it; it is now wired in behind ValidationOptions::check_orientation (default OFF) because enabling it fails 12 existing tests — loft, pipe, revolve (8 same-sense edges on a washer), sweep, and extruded-hollow-box all emit orientation-inconsistent shells. That op family is the likely upstream root of the mixed-socket body's 20 pairs (quarter sockets are built from these ops). NEW CAMPAIGN, ranked entry: fix orientation emission op-by-op, then flip the option default and re-capture the chain. FIRST SITE CLOSED (analytic full-revolve wall wires: rim senses now account for face reversal; the washer measures 0 same-sense pairs). REVOLVE FULLY CLOSED (second site): the segmented outer side-face builder paired an unreversed quad wire with Face::new_reversed; it now builds the reversed-winding wire (the inner side faces' own idiom) when reversed. All three paths measure 0 same-sense on the washer profile (90deg, 180deg, near-full segmented). EXTRUDE CLOSED (third site): outer side walls reverse their wire with the face (the #1367 rule), and the inner-wall quad pattern is chosen by the REVERSAL flag alone, not hole winding — the caps hold input wires reversed(bottom)/forward(top), so the wall's effective bottom sense must equal input orientation (measured truth table in the commit). Hollow box and circular-hole strict-clean. CAMPAIGN CLOSED (fourth site, same PR that flips the default): sweep_smooth side walls build reversal-aware wires, and build_cap_face (the loft/sweep/pipe SHARED end cap) picks its ring-wire pattern by start_role XOR reversed — the bilinear NURBS end cap was rev=true with an unreversed wire, and fixing it closed sweep, loft, and pipe together. CAMPAIGN CLOSED AND DEFAULT FLIPPED ON (2026-08-05): the "cut of two clean lofts" framing was WRONG — the D1 inner loft OPERAND itself carried all 32 same-sense pairs (never measured before; the ruled-NURBS taper corners set the reversal flag without reversing the wire, the #1367 rule again), and the earlier Step-0 32→28 measurement was patching a dirty-operand symptom (builder_solid's Cut-B flip and shape_store's fsnap.reversed transfer are both EXONERATED — new_reversed with unchanged wires flips every effective sense coherently). Three real roots closed: (1) loft's ruled-NURBS side walls now flip the SURFACE (swap the ruled rails, negating dS/du) instead of flagging reversal — reversal-flagged loft walls broke downstream boolean consumers (gridfinity d3/d4/d5 Euler), so an unreversed face with an outward normal beats a correctly-reversed one; (2) the internal-loops splitter (split_face_with_internal_loops) normalized plane-face discs CW-in-UV and holes CCW — backwards vs the effective-CCW-outer/CW-hole convention — so EVERY flush pocket/socket cut emitted a cap hole wound backwards against the flipped tool walls; fix is PLANE-GATED (the periodic lateral window-cut machinery is calibrated to the old winding — the cylinder-slot foil catches the ungated version); (3) blend's rim-fillet torus band used a fixed wire order + set_reversed, which cannot serve both rims — senses are now derived by opposing each contact circle's cap/wall user, and the Line-seam band is accepted by the structured two-rim mesher (the generic path skinned the correctly-wound band as its complement: area 1857→2440, volume −112). Pins: boolean::tests::{lip_ring_loft_cut,coplanar_flush_pocket_cut,fillet_v2_cylinder_rim_bands}_*_orientation_consistent + the same_sense_pairs/planar_effective_windings helpers. FULL WORKSPACE green with check_orientation defaulting ON. Probe gotcha for future winding oracles: a reversed face's effective boundary is the wire in REVERSE ORDER with flipped senses — flipping senses alone yields the same cyclic polygon and silently measures the stored winding. Capture-era operands (mixed-socket body) also remain same-sense (pinned). Each is its own wire-construction site. The fixture has an ACTIVE every-edge-two-sided guard; only the strict half-edge watertight pin stays `#[ignore]`d. Neither gates parity. RE-CAPTURED 2026-08-06 (post-campaign kernel, fresh 9-op chain capture with operands AND intermediate results serialized): every stage is orientation-CLEAN (fresh body clean, assembly byte-identical to the old capture) yet the final fuse of the two clean operands STILL emits exactly 20 same-sense pairs, tess 116 unmatched half-edges, 112 surviving into the exported STL (export tests stay green only because their oracle is undirected) — so the winding root is the GFA FUSE's own face-orientation emission on the NURBS-quarter-socket x cylinder-band configuration, NOT an upstream op; the boolean-assembler frontier is live again with a 58 ms repro. FUSE-SIDE ROOT CLOSED (2026-08-06): the 20 same-sense pairs were NOT the quarter-socket rims but 5 thin z=5 corner crescents of the body's underside (bin outline arc r=3.75 vs socket footprint arc r=4, 0.07-0.25 wide) — the greedy trace wound them CORRECTLY (arc-true UV area +0.43) but the splitter's outer/hole classification sampled via PCURVES, which fold reversed boundary arcs on a thin two-arc band (sign -6.69), so each correct crescent classified as a hole and the adjacent-not-nested promotion INVERTED it. Fix: plane faces classify loops on the arc-true via-frame polygon (split_face_2d_impl, same switch the nesting test already had). Fuse now validates clean, volume unchanged, all foils green. The 116 unmatched mesh half-edges PERSIST on the clean B-Rep — attribution CORRECTED 2026-08-07 with the DIRECTED half-edge oracle (authoritative; the offset-classification outwardness audit gives unanimous FALSE POSITIVES near concave cylinder corners — a directed-watertight cut audited '3 inverted, 10-0' — so the #1401 'top-socket cut mints double-flipped faces' claim is RETRACTED). Directed truth: assembly 0; body 116; stage capture shows call 000 healthy end-to-end, and call 001's ARGS already carry 38+78=116 — minted BEFORE any captured boolean by executeBatch-driven ops the fuse/cut hook missed (flatten batch ops in the next capture). Every captured boolean preserves the counts exactly. Fixtures: topsocket_cut_inmem.rs (healthy-cut guard + chain001 carrier pins, ~80 ms) on topsocket_{cut_base,cut_tool,chain001_a,chain001_b}.bin; oracle tooling in crates/io/examples/audit_bin.rs (HALFEDGE mode). MINT CAUGHT (capture v2, all 386 kernel calls hooked): loftWithOptions — call 080's loft result is BORN with 38 directed mismatches and call 207's with 78 (the 208 cut merges them to 116); each loft is combinatorially CLEAN with ONE coherently-flipped analytic CYLINDER side wall owning every mismatch (the campaign fixed the ruled-NURBS wall path; the arc-profile-to-cylinder-band path is the remaining emission site). Fixture: quarter_socket_loft_inmem.rs on quarter_socket_loft38.bin (clean-combinatorial + 38-mismatch pins). MINT FIXED (2026-08-07): the coaxial Cylinder/Cone arm emitted (surface,false) unconditionally — radial-outward equals solid-outward only for CONVEX corner arcs; a CONCAVE rounding needs the reversal, and the chord-cross outward cannot discriminate (concave traversal flips chord AND radial normal together), so the check is material-outward = traversal tangent x connect direction at the arc midpoint. Verified on the REAL captured profiles (thin-extrusion capture trick: extrude each loft profile face 0.5 in the hook, the top cap carries the input): without fix natively reproduces exactly 38; with fix, 38- and 78-configs both 0. Pins: quarter_socket_loft_inmem.rs (regenerates-watertight + defective-era capture). Full foils green (canary 27, ops 1019, io 250, algo 209). The STATIC chain fixtures (body 116 etc.) document the defective era and keep their counts; CHAIN VERIFIED CLEAN ON THE RELEASED 2.129.13 (tool-side re-capture: all 793 stage captures 0 unmatched directed half-edges); the mixed_socket body fixture is refreshed from that chain and the strict watertight pin is ACTIVE — the whole mixed-detail residual (511 = 395 CDT + 20 same-sense #1394 + 116 loft winding #1404) is CLOSED end to end; chain001/topsocket fixtures keep their defective-era counts as historical pins. Fixture pins updated (mixed_socket_fresh_fuse_is_orientation_clean ACTIVE); probes `crates/io/examples/orient_scan.rs` and `fuse_orient.rs` (per-face half-edge attribution + same-sense B-Rep edge attribution + operand FACES dump). Historic detail: re-probing the 4 historic export arms found 70/73, with `2x2 slotted no lip` (bnd 107-109), `2x2 mixed-detail per-cell half sockets` (bnd 259), and `3x3 O-shape + half sockets` (nm 8) failing. FULL KERNEL BISECT EXONERATES every 2026-08-04/05 engine PR: the same three failures reproduce byte-for-byte on the session-start kernel (73a4c2ce), so the tool's generator changes since the 73/73 measurement (#3223-#3227 era) altered these configurations' geometry into shapes the kernel has never handled. These are NEW scenario-coverage gaps in the slotted and per-cell/mixed-socket families. SLOTTED CAPTURED AND REDUCED (2026-08-05): `crates/io/tests/slotted_nolip_fuse_inmem.rs` — of the 10 booleans in the export chain, only the final socket-assembly fuse fails (clean F=56 slotted body + clean F=136 four-socket assembly abort with "open hole shell with 45 faces", the thick-wall abort family) and the fallback output carries the 107-109 boundary edges. BOTH HALF-SOCKETS CASES CAPTURED (2026-08-05): the `3x3 O-shape + half sockets` root is `crates/io/tests/oshape_socket_fuse_inmem.rs` — two clean 49-face socket pieces (one carrying 12 NURBS quarter-socket faces from the per-cell dispatch) abort with "open growth shell with 45 faces"; the chain's later 1022-planar-face fuse is its collateral. O-SHAPE ROOT MEASURED (2026-08-05): whole-strip ray-cast misclassification — three thin 45-degree chamfer-band plane strips of B classify Inside as whole faces (zero FF sections) while the independent point oracle says Outside; the measurement chain and fix entry live in the fixture doc. The `2x2 mixed-detail per-cell half sockets` case is CONFIRMED tessellation-parity (2026-08-05): all nine booleans replay clean and the final fused B-Rep is watertight (free=0), yet tessellating it at export tolerance yields 511 mesh boundary edges — `crates/io/tests/mixed_socket_tess_inmem.rs` pins the healthy B-Rep and carries the ignored tessellation repro; `replay_pair` gained a TESS_BND=1 mode for this discriminant. baseStyles arm stays 26/26; kumiko-dividers stays in budget. FULL EXPORT SURFACE MEASURED GREEN (2026-08-05, post plane-cone fix kernel c6dbc14a, md5-verified overlay): all 35 binGenerator.export arms, 447/447 assertions (88 in the matrix+divider run, 359 across the remaining 30 arms incl. combinedFeatures, compartments, cutouts, edgeCases, floors, permutations) — the scenario-coverage backlog's top combos now exist tool-side and all pass |
| **Mitsukude panel cut — CLOSED 2026-08-05 (the kumiko-dividers 82 s mesh-fallback root)** | math/SSI | Root was NONE of the five probe passes' candidates (splitter, classification, pave attachment, and the tangent-pinch framing all superseded by measurement): the free loop was a MISSING FF SECTION — the lip-taper cone x the prism's grazing x=38.05 end plane. `sample_plane_cone` sweeps 512 uniform-u samples and v(u)=e/(n·g) diverges near a hyperbola's asymptote, so the last kept sample sat at v=2.88 while the cap allowed v_max=4.07 — the entire 0.5-tall face-overlap window fell between two adjacent u-samples (chain cutoff z=34.8126, measured twice: F2_TRACE and a direct probe). Fix: extend each chain run's ends to the exact v_max boundary (bisect u on the monotone n·g within the dropped pitch, 8 uniform-u tail samples). Fixture ACTIVE and volume-pinned (27045.9; the leak measured 27095.9). New probes committed: BK_F2_TRACE (per-dropped-curve min distance to each face AABB inside the F2 filter) and FREE_EDGES=2 (owner-face bbox dump in replay_pair). TOOL-SIDE MEASURED on the fix-kernel overlay (single-test vitest run, same method as the 166.6 s baseline): the kumiko-dividers scenario passes in 25.5 s wall-clock (was 166.6 s against the 180 s budget) — the fallback path and its knock-on costs are gone. The 8-panel compound-cut abort is MOOT at export level on the fix kernel: the binStyles export arm (mitsukude watertightness assertions included) passes, the full export matrix + all 15 divider scenarios measure 88/88 in 84 s, and the isolated 2x2x6 mitsukude export runs 14 s error-free (historically minutes with bnd=4 nm=9). Interior-fallback counting is currently NOT measurable from JS: the setLogLevel console-tap recipe is dead (the vitest import resolves a different module instance than the kernel's cjs; totalLines=0 even at debug — caught by the instrument-fired check), and the per-op boolean capture hooks fire 0 times (the export drives executeBatch now). Re-instrumenting means either a batch-level hook or wiring setLogLevel through the tool's own kernel instance |
| **Kumiko corner wedge: overlapping coaxial cut removes NOTHING and drops the coincident cap** (re-captured 2026-08-04; characterization corrected same day) | algo/GFA | `crates/io/tests/kumiko_corner_wedge_inmem.rs` (ignored ready-repro): on post-revolve-fix operands the coaxial wedge-strut cut runs ANALYTIC in 2 ms keeping both cylinders (the old all-planar fallback is gone) but is wrong twice over: the strut genuinely OVERLAPS the wedge (point-oracle verified; it protrudes through the wedge's y=0 cap plane) yet the result keeps the full 285.861 volume, and the coincident y=0 cap is dropped (F=6 -> F=5, free=4) — its interior sample lies ON the strut boundary, `classify_coincident_coplanar` defers straddlers, and ray-cast answers Inside. Coincident-contact family (with the mitsukude pinch). Section inventory (2026-08-04): the wedge's y=0 cap receives ZERO sections although the strut's coaxial cylinder walls cross it in vertical lines — the plane-contains-axis x coaxial-cylinder FF case emits nothing — and the few sections that ARE emitted (1 on each z-plane, 4 on one strut face) each "split into 1 sub-faces", i.e. the splitter declines them. This is the [[project_coaxial-samedomain-frontier]] in its purest 12-face form. FF pair map (BK_FF_TRACE, 2026-08-04): the only 4 emitted sections all pair wedge faces 1-4 with ONE strut plane (the far angular cap; the strut's near cap is NOT coincident with the wedge's y=0 cap — its caps are slanted, y ranges [-0.57,-0.11] and [0.22,1.10], correctly AABB-rejected). MECHANISM FOUND (BK_RAWC + EF logs, 2026-08-04, after two refuted framings): every AABB rejection is geometrically CORRECT — the strut RADIALLY ENGULFS the wedge (cylinders at r 1.04/5.19 bracket the wedge's 1.55..4.75), so the y=0 cap lies wholly inside the strut and its removal is right, and the only genuine intersection curves are the 4 sections against the strut's far angular cap, ALL EMITTED. The defect surface: all four sections fail to split their faces ("split into 1 sub-faces") and nothing is cut. Two EF crossings are dropped ("edge Id(15) at t=0.031 — outside face Id(2) boundary") — but CAUTION before blaming chord containment: `build_face_containment` already carries a sagitta margin for curved edges, and the dropped edges may be the strut's r=1.04/5.19 cylinder edges piercing z=20.8 genuinely OUTSIDE the wedge sector (correct drops). ROOT ARITHMETIC (BK_SPLITW section dump, 2026-08-04): face 2 receives ONE Line section whose endpoints sit at r=4.000 and r=1.306 — EXACTLY where the section line crosses the SINGLE CHORDS of the face's two 90-degree NURBS arcs (4.75/sqrt(2)/cos(32.9deg)=4.00; 1.096/cos(32.9deg)=1.306), instead of the true arc crossings at r=4.75/1.55. So the section endpoints float 0.75 mm inside the face / overshoot the inner arc, the splitter cannot anchor them to the boundary, and it declines (1 piece). EXONERATED: `restrict_curves_to_faces`'s FaceWindow polygon samples curved edges at 16 points (sagitta 0.006 mm) — it is NOT the chord source. FIND (narrowed 2026-08-04): `clip_line_to_face` is EXONERATED — it bails Indeterminate on any non-Line boundary edge, so the wedge's arc faces are never polygon-clipped there. `build_section_edges` reads endpoints from the ARENA curve via `curve_endpoints`, so the chord crossings were paved upstream: prime suspect is the EE/EF machinery intersecting the FF section line with the wedge's NurbsCurve boundary edges via their CHORD (endpoints), placing the curve's extreme paves at r=4.00/1.31. CRITICAL ASYMMETRY (EF log, 2026-08-04): EF finds and KEEPS the four TRUE arc crossings (wedge arc edges x strut far cap, all at t=0.1436 — the exact NURBS crossings; the earlier-logged drops are the strut cylinder edges outside the sector, correct). So the true junctions EXIST as paves on the wedge's boundary arcs; only the FF curve's OWN end paves sit at the chord crossings (r=4.000/1.306, chord arithmetic exact to 4 decimals). CLIP FIXED (PR #1343, 2026-08-05; successor to the closed #1341): `clip_line_to_face_boundary` now intersects sections with NurbsCurve boundary edges by sampled sign-change bisection in the face plane (bounded cost; honeycomb stays at 0.75 s where bezier clipping took 115.6 s), and a chord crossing is dropped as a phantom border ONLY when its segment's true-arc hit lands within the section range (dropping it unconditionally broke the honeycomb residual pins — measured, narrowed). The wedge section is now ONE clean arc-to-arc piece (r 4.7497 to 1.5504). CAMPAIGN CLOSED (PR #1352, 2026-08-05): the final fix is NURBS boundary-image expansion in `boundary_edges_to_pcurve_with_images`, extended to hole-free faces of ANY surface (Line expansion stays planar-gated for the seam machinery) behind two foil-earned gates: circle-likeness (5-sample circumcircle fit; marched free-form NURBS like the snapClip walls broke under expansion and are never circles) and weld-coincidence of the junction with a section anchor. The wedge cut is watertight, analytic, and volume-pinned (285.861 -> 247.460); the fixture's cut pin is ACTIVE. RETRACTION: the "knife-edge across builds" blocker was a STALE-BINARY artifact — `cargo build --tests` does not rebuild examples, so the free=0 readings came from a binary still carrying the non-planar arm; the source was deterministic all along (verify the instrument fired). Downstream: the kumiko corner-band family and every coaxial revolve-vs-revolve cut now have the anchoring machinery this campaign built |
| **v2 trimmer residuals — ALL CLOSED 2026-08-04** | blend | Last root: `dihedral_half_angle` returned half the angle BETWEEN the inward normals where the geometry needs the material wedge half-angle `(pi - angle)/2`; the two coincide only at 90-degree dihedrals, so every box-calibrated case passed while near-tangent ridges got contacts 100*r from the edge (the "keep-side tangency" 12% loss was never keep-side; `regress_blend_keepside_tangency.rs` un-ignored and green). Earlier same-day closures: concave fillet (bisector flip via face-extent material witness + `TrimKeep::AwayFrom` resolved inside the trimmer, `regress_fillet_concave_notch.rs`), duplicate contact edges, end-cap notch, `chamfer_v2` external tangent branch. Refutation history and the constructor-winding ground truth live in the fixture doc comments and `convex_chamfer_volume_check.rs` |
| **Sweep re-centered the profile onto the path — FIXED, MERGED #1421, RELEASED 3.0.1 (2026-08-07; verified on the npm 3.0.1 overlay: lip z 0..4.4 exactly as positioned)** | operations/sweep | Found via the brepjs gridfinity-smoke test 4 lip-fuse failure (fused zMax 22.52 vs 25.4 expected; excluded from brepjs CI so long-standing on every kernel version). The fuse was EXONERATED by operand capture: `sweep()` decomposed profile vertices as offsets from the profile CENTROID and rebuilt them at the path frames, discarding the profile's position relative to the spine (the lip profile's centroid height 2.88 = the exact z-shift). Fix: `sweep()`/`sweep_smooth()` now sweep AS POSITIONED (frame-0 basis+origin make ring 0 the identity; reference-kernel pipe semantic) when the profile is perpendicular to the path start tangent (\|cos\| >= 0.99); edge-on/oblique profiles keep the legacy auto-orient re-centering (pinned by `sweep_edge_on_profile_is_auto_oriented`). `helical_sweep` keeps re-centering explicitly via `ProfilePlacement::CentroidOnPath` (its API positions the profile; occt-wasm throws for helicalSweep so brepkit defines the contract). Pins: `sweep_keeps_offset_profile_position_on_straight_path`, `sweep_closed_path_keeps_profile_position` (foil-verified: fails under re-centering). FOLLOW-UPS: (1) CLOSED for `pipe.rs` and `sweep_with_options`, MERGED #1438, RELEASED 3.1.3, CONSUMER-VERIFIED (brepjs full brepkit project 4567 tests green on the npm 3.1.3 overlay): both now resolve placement like `sweep()` — perpendicular profiles as-positioned (pipe: offsets from the path start; options family: frame-0 basis+origin, straight fast path aligned), edge-on/oblique keep centroid; pinned by pipe/sweep_with_options `*_keeps_offset_profile_position` (pipe pin foil-verified). `sweep_miter` as-positioned CLOSED with an EXACT miter (2026-08-08): the parked volume question resolved BOTH ways. The oracle was mis-modeled — area x path length holds only for centered profiles; a true miter cuts each leg prism at the bisector plane, so a profile offset toward the inside of the bend loses 2(centroid.n)/(n.t) per kink (the offset elbow's truth is 8.2668, not ~10.39). AND the geometry was broken by a bug UNDER both prior attempts: `compute_frames` sampled raw [0,1] ignoring the curve's domain, so curve_split sub-paths (domains [0,0.5]/[0.5,1]) extrapolated their clamped end spans linearly and EVERY miter leg swept the full path length — two overlapping full-length tubes glued by transition quads (also the root of main's centered-L 9.84 and the second attempt's 8.67/13.875). Fixes on the merged branch: compute_frames maps samples into the curve domain (all sweep modes benefit; identity for [0,1] curves), and sweep_miter joins legs on ONE shared bisector-plane kink ring — slide each exit-ring vertex along its own leg tangent onto the plane; for perpendicular profiles reflection-through-bisector equals the tangent-to-tangent rotation, so both legs land on identical points (no transition faces, wall quads stay in their prism side planes); degenerate kinks or slides crossing the first interior ring fall back to the old bridge. Miter is now EXACT for polyline paths: pins sweep_miter_l_shaped_volume_correct (=9.0 — the 0..1 square's 0.5 centroid offset shortens both legs) and miter_sweep_keeps_offset_profile_position (=8.2668), both at 1e-6; the elbow dissect probe stays as an ignored diagnostic. Latent sibling: pipe.rs still samples raw [0,1] (no kink split there; bites only shifted-domain paths); (2) the lip sweep emits ~2730 facet faces (interpolated-NURBS path -> ~180 rings x 14 profile edges) where the reference kernel emits per-spine-edge analytic surfaces — an analytic-preservation gap for line/arc spines (planes + cylinders/cones per segment), now ALSO correctness-relevant: it is the direct upstream of the lip-fuse chain below. WITH THE PLACEMENT FIX the smoke test moved into a NEW failure: the correctly-positioned lip (bottom coplanar with the box top rim at z=21) fuse chain is broken THREE deep (measured 2026-08-07 on captured operands, box F=27 clean vol 16191.8 + lip F=2730 all-planar clean vol 2201.7): (a) raw GFA fuse completes in ~850 ms but emits free=74 over=28 vol 15799 (LESS than the box alone — material lost; coplanar-contact class); (b) validation rightly rejects it and the mesh fallback then HARD-ERRORED "empty wire" in assemble_solid_mixed — a sub-resolution polygon loses every edge to the degenerate/duplicate skips (all vertices quantize to one id) — FIXED on fix/mesh-assembly-empty-wire (drop the polygon; pin assemble_mixed_drops_sub_resolution_polygon, foil-verified); (c) with (b) fixed the fallback completes but its output is garbage (F=6032, vol 87780 vs expected 18393 — mesh co-refinement also breaks on the coplanar contact). EMPTY-WIRE FIX MERGED #1423, RELEASED 3.0.2; the FULL gridfinity-smoke file is GREEN on the npm 3.0.2 overlay (26/26, the lip fuse completes via the mesh fallback and meets the test's assertions), so the smoke-test defect is CLOSED at test level. STILL OPEN (quality, not test-gating): the fallback result is degraded (all-planar, imprecise volume) — ANALYTIC SWEEP FACES SHIPPED, MERGED #1427, RELEASED 3.1.0, TOOL-SIDE VERIFIED (2026-08-08, npm 3.1.0 overlay: full brepjs gridfinity-smoke 26/26 in 1.57 s — was ~9-12 s; the lip is 60 exact analytic faces via the brepjs 12-edge spine, and the production lip fuse is an 86-face ANALYTIC result with exact sum volume 18396 and zMax 25.4 — the exact-coincidence config went analytic in this run; the nondeterminism note below still stands as the native repro shows): `sweep_along_edges` moved into operations with an analytic path (`sweep/spine.rs`) gated to closed planar G1 line/arc spines with all-line perpendicular profiles sketched on the plane through the chain start — per straight run planar quads, per tangent arc corner `revolution_band_surface` bands (the revolve machinery, helpers made pub(crate)); the lip ring is 40 exact faces (24 planes, 8 cylinders, 8 cones), watertight, Pappus-volume-pinned (`analytic_spine_sweep_lip_ring_is_exact`), and the NATIVE hollow-box+lip fuse is ANALYTIC with exact sum volume (`analytic_lip_ring_fuses_onto_hollow_box`, lip inset 0.25 from the outer wall; 10/10 stable) — the old GFA coplanar free=74 failure was the 2730-facet operand and is gone for this class. COINCIDENT-FUSE NONDETERMINISM: CLOSED (2026-08-08, four-pass dig, fix on fix/coincident-fuse-determinism-4). ROOT: shell_op's rim assembly collected its open-boundary edges by iterating `edge_to_face_map` (a HashMap) — the rim loop's STARTING EDGE varied per run, so the rim face's wire origin (which `PlaneFrame::from_plane_face` anchors on) rotated the splitter's UV frame run-to-run, shifting quantized edge sets so SD's within-rank dedup of the coincident z=21 faces formed or not (~10-33% bad). Fix: sort the boundary edge ids before chaining (one line). Measured: operand byte-variance 24/29 -> 0/99; bad outcome 10-33% -> 0/100; `exact_coincident_lip_fuse_stays_analytic` UN-IGNORED and 10/10 green. The dig also hardened `remove_pendant_sections` (lowest-source peel, #1432) and left durable instruments: BK_FF_DUMP (phase_ff), BK_SD_SETS (same_domain), STRACE-SEC/STRACE-IN (split trace), BK_OPERAND_DUMP (loop probe). METHOD NOTE for future nondeterminism digs: differential dumps at stage boundaries (FF sections -> SD edge sets -> operand faces) walked the flip upstream in three hops; the operand-level dump should be FIRST next time — operand construction ops (shell, extrude, sweep) are as suspect as the boolean. GATE NOTES: the profile must lie on the perpendicular plane THROUGH the chain start (rings transport junction-to-junction); the cone-exactness coplanarity gate tests the ring TRANSPORTED to each corner (testing ring 0 against far corners is the bug the first draft had). NEW FINDING while pinning the fuse: shell_op emits orientation-inconsistent cavity corner cylinders — see the shell_op row below. Repro recipe: brepjs bench worktree `tests/lip-sweep-probe.test.ts` (untracked) captures lip_fuse_{a,b}.bin + bounds JSON; replay with replay_pair (raw GFA) or an operations::boolean probe |
| **shell_op cavity corner cylinders emit same-sense edges — CLOSED, MERGED #1435, RELEASED 3.1.2, CONSUMER-VERIFIED (2026-08-08: brepjs full brepkit project on the npm 3.1.2 overlay — 307 files, 4567 tests, 0 failures)** | operations/shell_op + boolean/assembly | Was: 16 same-sense edges on every shelled rounded box (4 cavity corner cylinders x 4 edges). THREE coordinated fixes, each measured: (1) `assemble_solid_mixed`'s CylindricalFace arm now builds the WIRE with reversed winding for `reversed: true` specs (the #1367 rule) — the arc-edge geometry still derives from the given vertex order, which is why the earlier vertex-order flip broke arc pairing (that refutation stands); (2) shell's rim chaining (`sort_edges_into_loops`) chains UNDIRECTED and assigns orientation from the chain direction (the old strict-orientation chaining dead-ended on the corrected cavity wires — it was calibrated to the double-flip); (3) the rim's per-edge seed orientation accounts for the owner face's reversal flag (`!(is_forward XOR is_reversed)`), which the old code ignored (16 -> 8 -> 0 across the three fixes). `shelled_rounded_box_is_orientation_clean` UN-IGNORED and green; the lip-fuse pin upgraded to assert FULL validity (operand now clean); coincident-fuse determinism loop still 0/50 bad; io 62 suites + canary green. NOTE the planar/curved emission idioms in shell Phase 4 are now genuinely consistent (planar: reversed order + unreversed flag; curved: given order + reversed flag + assembler-side wire reversal) |
| **User-reported wall-pattern defects — BOTH ROOTS FOUND TOOL-SIDE, fix PR tool #3294 (2026-08-07)** | tool | Honeycomb corners: stamps stayed on the full inner span on LIPPED bins (the #2865 keep-out was lip-less-only), so the unstaggered row's end hex crossed into the corner-arc zone and scalloped the curved corner; fix clamps every bin's stamps to the flat span (BOX_CORNER_RADIUS - wallThickness). Triangle: apex-up/down stamped at the SAME centroid height on double pitch (checkerboard) — not a tessellation; rewritten as interlocked bands (downs at +R/2, half-period over, slant web exact from gap=(sqrt3*dx+dy)/2-R). Method: render the pure-math calculators against the wall elevation (SVG->PNG), verify numerically (0 overlaps, min web = 0.8000). Full tool suite 20882 green with re-pinned snapshots. Kernel exonerated |

The remaining `#[ignore]` entries are diagnostics or slow perf runs, not open bugs: the
`profile_intersect.rs` box-sphere probes are stale (box-sphere shipped analytic in #1006),
`staircase_fuse_with_cylinders` is a ~2 min perf run, and the two `#696` dovetail entries plus
`diverge_first_cut` are print-only.

## Thick-wall cavity: CLOSED (two stacked roots in the same shell arm)

**Root 1 (fixed, `fac84c7f`):** the miter at a swallowed corner saw only one normal, so the
neighbouring walls overshot. Fix: feed a collapsing fillet's vertices BOTH extreme normals.
That took exported boundary edges from 149-160 down to 17-23 but did NOT close the case.

**Root 2 (fixed 2026-08-04, the residual):** the SAME `new_radius > tol` arm still emitted NO
inner face for the collapsed cylinder, so the inner shell had a corner-wide gap at every
swallowed corner. The spec assembler closed it the only way it could: by threading the top
ring's inner wire down the cavity verticals and across the floor chamfer diagonals — a 16-edge
non-planar inner wire on a z=const plane face. That body is edge-paired (free=0, passes every
health check) but geometrically degenerate, so the next fuse's hole-shell grouping correctly
aborted `open hole shell with 9 faces`, fell to the mesh fallback, and the fallback's open
output carried the 17-23 boundary edges to the export. Fix: emit the sharp-corner chamfer
strip — the collapsed cylinder's WIRE vertices mapped through the already-computed miter
positions form exactly the right planar quad (z-extent included, because the shared corner
keys carry the bottom/top normals).

**Measured end-to-end** (exported mesh boundary edges, 1x1x10 halfSockets unless noted):

| case | pre root 1 | after root 1 | after root 2 |
|---|---|---|---|
| wall 3.8 / 3.9 / 4.0 | 149 / 159 / 160 | 23 / 19 / 17 | **0 / 0 / 0** |
| 2x2 wall 4 / wall 6 | 832 / 898 | 20 / 20 | **0 / 0** |

Pins: `shell_thickness_past_corner_radius_gives_a_sharp_corner` (now also asserts the 4
chamfer strips) and `crates/io/tests/thickwall_sharp_cavity_fuse_inmem.rs` (re-captured
post-fix operands, fuse pin active: F=67 analytic, free=0).

**REFUTED, do not re-implement: placing the collapsed corner EXACTLY.** Solving for the true
corner from the arc centre (`nᵢ·(x−C) = radius − thickness`) is geometrically right and
measured WORSE (2x2 wall 4: 20 → 318, wall 6: 20 → 544). With root 2 fixed the octagonal
cavity (walls ending at `half − thickness − radius`, 45-degree chamfer strips between) is
watertight and analytic end-to-end, so the true-corner formula is moot.

**Diagnostic recipes that broke this open:** `BK_SHELLS=1` dumps every assembled shell's face
membership at grouping time (builder_solid), complementing `BK_OPEN_SHELL=1`; the committed
`dump_solid` example prints every wire of a serialized operand with edge ids — the degenerate
16-edge inner wire was invisible until wires were printed edge-by-edge. Capture recipe: a
temporary tool-side kernel test monkey-patching the raw kernel's boolean methods with
`serializeSolid` on every numeric arg (flatten arrays: `compoundCut` passes tools as an
array), then `replay_pair` per captured op.


**THE TWO MITSUKUDE "DIVIDER" FAILURES ARE MISFILED: neither dividers nor compartments are
involved.** Isolation (2026-08-01, one case per process): a 2x2x6 bin with the mitsukude wall
pattern and NO compartments and NO dividers exports **bnd=4 nm=9** on its own. So this is a
WALL-PATTERN defect that the divider scenarios merely inherit — the same misfiling as the floor
`oversized elements` case one section up. Both scenarios use `wallPattern.pattern: 'mitsukude'`,
which is a KUMIKO pattern built by `kumikoWrapBuilder.ts`, i.e. the same builder as the goma
lattice bands. **These belong to the kumiko family, not to a separate divider family, and they are
therefore not independent easy wins.**
**AND THE PARKED BRANCH DOES NOT FIX THEM — already measured, no experiment needed.** The kernel
used for the divider re-run was built from `fix/kumiko-corner-window-cut` HEAD, which contains all
five kumiko roots (`9ecb2bb2`, `e58fec63`, `2fa915e9`, `de6bdcb8`, `1d41afea`); the failures persist
under it. So the corner-cut work does not reach this configuration, and "overlay the parked branch
and see" is a dead experiment. VERIFY WHICH BRANCH A MEASURED KERNEL CAME FROM before proposing a
branch-comparison — that check turned a planned experiment into an already-answered one.
**FULLY REDUCED TO A 364 ms FIXTURE — `crates/io/tests/kumiko_lattice_fuse_inmem.rs` — AND THE
ROOT CHAIN IS NOW LOCALIZED TO ONE MISSING QUAD (2026-08-03).** The 67-face open lump has exactly
FOUR free edges bounding one 0.05-wide skewed quad. Probe chain (BK_OPEN_SHELL → BK_SUBFACE_BOX →
BK_FF_TRACE → per-face section probes): the quad is a strip of B-slant face 845 that never split
at x=38.05 because its section CHAIN dangles 0.002 short of 845's boundary — the closing 0.002
micro-section (a REAL lattice end-facet feature, 20000×tol, not noise) was graze-dropped by the
sampled in-both filter, the arrangement then rejects the pendant chain, and the face under-splits.
SHIPPED prerequisite: F1's plane×plane Line filter now uses the exact polygon clip instead of
16-sample aliasing (the fix the old comment said "needs a test against true face extents" — the
AABB version regressed A1 to bnd=158; the polygon version keeps every foil green).
SEVENTH PASS (2026-08-03, shipped from `diag/kumiko-junction-snap`): free edges 43 → **2**. The
"7.7e-4 trim-arm disagreement" framing was WRONG — the twin free-edge chains were minted by the
PLANE-ARRANGEMENT EMISSION, which reconstructed every sub-face vertex as
`frame.evaluate(frame.project(p))`, PROJECTING operand facet-chain vertices (which sit up to
~2e-5 OFF their face's stored plane) onto the plane while the unsplit neighbour faces kept the
original positions. Three fixes shipped together, each measured: (1) arrangement emission carries
each registered input endpoint's EXACT 3D through `exact3d` (35 → 2); (2) crossings snap-round
onto registered vertices, endpoints pre-registered first-wins (supports 1); (3)
`weld_coincident_vertices` snap widened 10·MERGE_TOL → 100·MERGE_TOL, the recurring
fit-error-vs-exact-gate class (43 → 35). Junction registry additionally: guarded wide adoption
(≤1e-3 AND 10x closer than the next junction — the measured bimodal gap: copies ≤7.9e-4, genuine
spacing ≥1.1e-2) plus boundary-VERTEX adoption in the snap (the partner-surface refine is
degenerate when the boundary lies IN the partner surface). Foils: algo 208 green, ops green, io
green with the honeycomb residual re-pin (pcut2 IMPROVED 38 → 28, pcut1 52 → 53, pcut3 held 0).
REMAINING (eighth pass, localized 2026-08-03): the fixture aborts on an open 67-face growth
shell with exactly TWO free edges at the z=34.8 coplanar top corner — clean coordinates,
owners `Id(1436)<-365` (a DIAGONAL lattice edge (38.0,-42.75)→(38.05,-42.95), z=34.8) and
`Id(2309)<-364` (the x=38.05 top-edge piece y∈[-42.95,-42.7487]). Probed with COP_PAIR/COP_SEC
eprintlns in `process_coplanar_pair`: the z=34.8 coplanar pair is A-top `Id(364)` (outer wire =
a 4-edge RECTANGLE whose corner is exactly the shared free-edge vertex (38.05,-42.95)) ×
B-top `Id(852)` (16 outer edges, extends past x=38.05), and the pair emitted ZERO coplanar
sections. The diagonal free edge is an INNER-WIRE segment of A-top (the lattice outline hole).
NINTH PASS (2026-08-03): the coplanar-phase inner-wire theory is REFUTED for this corner. The
full inner-wire extension of `phase_ff_coplanar` was implemented (loops over outer+inner wires,
even-odd `point_in_loops`, multi-span `clip_section_to_loops` that no longer bridges holes) and
had ZERO effect: the clip then finds the corner span (B edge `Id(2784)` → 1 span into A's
material triangle) but `has_existing_section_at` correctly suppresses it — the section ALREADY
EXISTS from the transversal pair (A-top × B's south wall), so A-top 364 receives every section
it needs. Reverted per verify-or-revert (the hole-bridging clip fix is real but has no repro).
THE ACTUAL ROOT is in 364's SPLIT: its inner lattice wire PINCHES ONTO the outer rectangle
corner (the diagonal ends exactly at (38.05,-42.95), so material wedges to zero width there).
A boundary-touching "hole" is not a proper hole, and the splitter's hole distribution
(assign whole inner wires to containing sub-faces) mis-handles it: the split product `Id(2309)`
takes the outer corner path WITHOUT weaving the diagonal, over-covering the pinched wedge, so
the diagonal (wall side) and the east-edge piece (top side) are each left use-1. Fix altitude
is the face splitter's hole integration for inner wires that touch the outer wire (same pinch
class as the tessellation "hole-less containers defeat inner_wires guards" lesson), not the
coplanar phase.
An `emit_riding_sections` variant (exact-vertex sub-spans for chain-riding coplanar
candidates) was implemented and REVERTED: A/B showed zero effect on these 2 edges.
TENTH PASS (2026-08-03): the pinched-hole theory was ALSO wrong (A-top `Id(364)` has ZERO inner
wires; its split {2309, 2472} is a correct partition along B's outline). The true root of the
z=34.8 corner: `fill_face_info::fill_ef_in`'s crossing-angle gate (`IN_FACE_MAX_DEVIATION_RATIO`
0.2) admitted a LONG shallow-crossing leaf — B's slope-bottom edge crossing the east wall
`Id(365)` (x=38.05 plane) sits 0.05 OFF the plane but at only 2.7% of its 1.8 chord — so pbs
3890/3891 landed in 365's `pave_blocks_in`, the wall's splitter consumed off-plane section
endpoints, and its partition warped (faces `1436`/`2311` with top-plane vertices in wall wires;
`Id(9142)` used 3x). Fix on parked branch `diag/kumiko-in-gate-abs` (e0f3f799): an absolute
ceiling `IN_FACE_MAX_DEVIATION_ABS = 1e-2` on the ratio band. MEASURED: the z=34.8 corner
closes fully (abort 67 -> 51 faces, zero free edges at z=34.8), BUT the fixture's free count
regresses 2 -> 14: the z≈4.5/9.9 junction-copy pairs (the seventh pass's 1.7e-4/3.5e-4
disagreements) RESURFACE once those IN blocks stop feeding the corner splits — the shipped
2-edge state was partly stitched by the same off-plane blocks. Eleventh pass: keep the abs
ceiling and re-attack the resurfaced junction copies (they are the SAME positions the seventh
pass measured, so the guarded-adoption path is the place to look at why they no longer unify —
likely a registration-order change now that fewer resolve calls happen), then re-run foils
before judging the ceiling's value (1e-2 sits between the grazing class's fit-error scale and
the 0.05 wall offset; the calibrated 2-18%/24-58% ratio classes are unaffected for chords
under 5e-2).
ELEVENTH PASS (2026-08-03): registry-side theories for the 14 resurfaced edges are EXHAUSTED —
with a 2e-3 near-miss probe in `JunctionRegistry::resolve` there are ZERO near-misses at the
twin positions, and pre-seeding the registry with ALL operand boundary vertices (first-wins
exact ground truth, tried on the parked branch) leaves the 14 free edges byte-identical. So
the twin endpoints (e.g. (38.049699651,-42.747694041,4.513595527) vs its operand-exact
partner 3.5e-4 away) NEVER pass through phase-FF endpoint resolution; they are minted by the
pave machinery (VE/EE/EF vertices or make_blocks) or the builder's edge splitting. NEXT
INSTRUMENT (the recipe that cracked the seventh pass): env-gated backtrace in `Vertex::new`
on the literal coordinate 38.049699651 to get the minting call path in one run, then apply
the adopt-exact-geometry principle at that site. Also note: seeding was REVERTED (null
result), and the warped wall faces the ceiling removed were "accidentally load-bearing" at
the z=4.5/9.9 corners — geometrically invalid (0.05 off-plane vertices in plane-face wires)
but topologically stitching, which is why the shipped 2-edge state looked better than this
parked 14-edge state while being structurally worse.
TWELFTH PASS (2026-08-03, backtrace instrument): the twin vertex (38.049699651, -42.747694041,
4.5135955) is minted by `phase_ef::check_edge_face_pairs` — an EF crossing of B's slope-bottom
edge with a slope plane — and its 3.5e-4 offset from A's operand corner is REAL OPERAND-LEVEL
DISAGREEMENT: the two bands are independently faceted and their outlines miss each other by up
to ~1e-3 at lattice corners. Nothing at the EF altitude unifies the crossing with the nearby
operand vertex: the existing mid-edge tangential snap (`find_nearby_pave_vertex_widened`,
window `tol/sin_angle` capped 1e-3) only fires for near-tangential crossings AND requires the
candidate to lie on the crossed surface at weld scale, which the OTHER operand's corner fails
by the same operand noise. FIX SHAPE for the next session: guarded wide adoption at the EF
crossing (band ~1e-3 with the 10x nearest-2 ambiguity ratio guard from `JunctionRegistry`,
candidate on-surface band widened to the same scale) so interference solutions within the
operands' own inconsistency adopt one shared vertex. RISK: this touches the honeycomb-
calibrated endpoint windows (the 2.7e-7..2.2e-6 measurements and the "regressed the honeycomb
wall-cut raw residual" caution in `phase_ef.rs`) — implement with the full foil suite and the
honeycomb residual ceilings in the loop. The larger lesson for the campaign: every remaining
kumiko defect traces to the operands disagreeing at ~1e-3 while the engine welds at 1e-5;
per-vertex tolerance adoption at interference altitude is the unifying frontier.
THIRTEENTH PASS (2026-08-03, SHIPPED from the former parked branch): the EF-IN absolute
ceiling is scoped to STRAIGHT leaves (an unscoped ceiling broke `fuse_shelled_box_with_
socket_loft` — the calibrated grazing arcs legitimately exceed 1e-2 absolute; a Line's
deviation from a crossed plane grows linearly so ratio-small + absolute-large is always a
long transversal crossing), and the FF junction registry is seeded with every previously
minted pave vertex so FF refinements adopt the anchors EF already placed (14 -> 7 free
edges). All foils green, honeycomb ceilings unchanged. REFUTED this pass: (a) guarded wide
VV merge of cross-solid operand vertex pairs — byte-identical free edges (the twins are
solver-minted, not operand vertices); (b) cluster-canonical adoption in `resolve`
(CLUSTER_BAND 3e-3 / ISOLATION 1e-2, lexicographic canonical) — net-negative 7 -> 9 with
new long free edges, because consumers that bypass endpoint resolution keep their own
anchors and a canonical only helps if every consumer converges; (c) seeding alone on the
pre-ceiling main — neutral (the old 2-edge state was stitched by the invalid off-plane
wires the ceiling removes). FOURTEENTH PASS (2026-08-03, structure cracked): the 7 free edges form ONE CLOSED LOOP — a
single hexagonal hole on the slope at the z≈9.9 crossing (x∈[38.0,38.05], z 9.863→9.892
rising with x), so this is a MISSING FACE, not a weld problem. Cluster member identities
(Vertex::new backtrace): A=(38.0,-41.315108,9.863328) is an OPERAND vertex, C=(38.0,
-41.317472,9.863328) an EF crossing 2.36e-3 away, B=(37.9997,-41.315202,9.863147) an FF
junction 3.75e-4 from A — with the seeding shipped in #1284 the A/C pair sits wider than
the guarded band and the ratio guard rightly refuses. But the loop structure shows the
copies are all RIM vertices of the one hole. Rim owners: `2375<-913` (two edges),
`2388<-935` (three edges: B's slope split), `2370<-910`, `2323<-400`. FIFTEENTH PASS
(2026-08-03, answered): source `Id(935)` split into only TWO sub-faces — `Id(2066)`
Outside/kept and `Id(2067)` INSIDE/DROPPED with probe point (38.0199,-41.0126,9.7837),
whose y sits OUTSIDE the hole's y-range [-42.75,-41.30]. So 2067 STRADDLES solid A's
boundary: 935 was UNDER-SPLIT — the FF section chain that should partition the outside
hexagon from the inside strip gaps by exactly the 2.4e-3 B/C anchor disagreement (the FF
junction B and the EF crossing C at the z≈9.9 rim), the splitter rejects the pendant
chain, and single-point classification then drops the whole straddling face. SIXTEENTH-
PASS TARGET (one decision, then mechanical): determine whether the EF crossing C
(38.0,-41.317472,9.863328) is REAL geometry or operand noise — measure the distance from
operand vertex A (38.0,-41.315108,9.863328) to the face crossed by C's EF interference.
If A lies within operand-noise (~1e-3) of that face, the crossing IS A's vertex-face
incidence solved 2.36e-3 along the edge, and the EF endpoint-drop window (currently tol,
or tol/sin only when the endpoint is ON-surface at weld scale) must widen its on-surface
test to the operand-noise scale for Line edges — C then collapses to A, the FF chain
anchors agree, 935 partitions, and the hexagon face survives classification. If A is NOT
near the crossed face, the micro-facet is real and the splitter must instead accept the
2.4e-3 chain gap at 935 (bridge chain ends within the measured operand-noise scale,
gated to chain-END pairs only). Either way the campaign remains ONE decision from a
closed fixture.
SIXTEENTH PASS (2026-08-03, DECIDED — branch b): measured with an EF-site probe. The
crossing C comes from B-edge `Id(3093)` × A-face `Id(400)` at t=0.482 (both edge
endpoints ~0.18 OFF the face: clean transversal, no endpoint incidence). Operand vertex
A and FF junction B both lie ON face 400's surface to 3.5e-10/5.9e-11 — so A and C sit
on the SAME line (plane(400) ∩ {x=38.0}), 2.36e-3 apart ALONG it: B's edge genuinely
crosses 2.36e-3 away from A's vertex. The micro-structure is REAL and the fix is the
PENDANT-CHAIN BRIDGE in the face splitter: for a planar face, after collecting sections,
find pendant chain ends (section endpoints shared with no other section or boundary
within the weld band) and connect PAIRS of pendant ends within a bridge band (~3e-3,
the measured operand-noise scale; genuine features are >= 1.1e-2) with a Line section —
the micro-facet edge the true result needs. Implementation notes: measure 935's actual
section chains near the hexagon first (which ends dangle and where); gate the bridge to
pendant-END pairs only (never mid-chain), and keep the guard that a bridged pair must
be mutually nearest. Then verify sub-face 2067 splits into the inside strip + the
outside hexagon, classification keeps the hexagon, and the fixture closes.
SEVENTEENTH PASS (2026-08-03): 935's section chains measured. The rim line 935∩400 is
covered by PER-PAIR micro-pieces because the A-side is facet-fragmented: pair (935×400)
contributes [x=38.05 exit → A] (A is one of face 400's own vertices — the line exits 400
there), the NEXT facet pair contributes the 3.75e-4 fragment [A → B], and the final
[B → C] micro-piece (~2e-3, ending at the EF exit C on 935's boundary edge 3093) is
MISSING — dropped by the in-both/graze filtering in `restrict_curves_to_faces` for
whichever A-facet pair owns it. So the pendant-chain-bridge framing narrows to: the
graze-drop rescue (`rescue_corner_crossing`) declines a REAL 2e-3 corner crossing at
this rim (its strict-interior midpoint gate, or the fine-resample path, eats it).
EIGHTEENTH PASS (2026-08-03, ROOT FOUND — the deepest one yet): the (400×935) section
arrives at `restrict_curves_to_faces` ALREADY truncated at B, because the [A..C] stretch
RIDES face 400's boundary and the pair-level mutual-overlap clip legitimately ends
there. The exit pave C DOES exist: edge 3093's pave block is split at C
(`PB3093` probe: split edges 7702/7703 meet exactly at C). But
`split_face_2d_impl` builds the OUTER boundary via `boundary_edges_to_pcurve(topo,
face.outer_wire(), ..)` from the face's CURRENT topology wires — `edge_images` are
expanded ONLY for INNER wires (the `expanded_inner_wires` block). So the boundary split
at C never reaches 935's splitter, the face cannot partition at its own exit pave, and
the neighbour wall's products (which DO use images 7702/7703 through C) mismatch 935's
products along the whole edge. NINETEENTH PASS (mechanical, foil-gated): expand the
OUTER wire with `edge_images` in `split_face_2d_impl`, mirroring the inner-wire
expansion — blast radius is every split face's outer boundary, so run the full foil
suite plus the honeycomb ceilings. REFUTED this pass: the pendant-chain bridge — a
twin-aware pendant scan + mutually-nearest boundary-vertex bridge (band 3e-3,
isolation 1e-2) fires at the ALREADY-HEALTHY z=4.5 corner (whose sliver is genuine
structure) and breaks it (bridge edge use-3), while the z~9.9 target's true fix is the
boundary-image expansion above, after which its C vertex exists and no bridge is
needed. Do not re-attempt bridging; fix the boundary instead.
NINETEENTH PASS (2026-08-03, parked on `fix/outer-wire-images` @ 6eb334e6): BOTH pieces
implemented and measured. (1) `boundary_edges_to_pcurve_with_images` expands the OUTER
wire with pave-split edge images (Line edges, mirroring `rebuild_face_with_edge_images`)
— face 935 now sees its exit vertex C on the boundary. (2) The pendant bridge RETURNS,
correctly gated this time: twin-aware pendant test, own-other-endpoint exclusion,
mutually-nearest boundary vertex within 3e-3, isolation >= 1e-2, dedupe, AND a new
section-free-target gate (a target vertex any section already reaches is part of the
section network — bridging onto it over-connected the healthy z=4.5 corner to use-3;
exit paves like C are section-free). MEASURED: z=4.5 closes COMPLETELY; the z~9.9
defect shrinks to ONE six-edge hole rim ((38.05,-41.30)→A→B→C→(-42.75)→
(38.05,-42.7477)→back). Remaining defect: face 935's splitter STILL yields only 2
sub-faces (2066 Outside kept / 2067 Inside dropped, probe unchanged) — the strip region
bounded by the bridged chain is not traced as its own region. TWENTIETH PASS (2026-08-03,
answered; branch now @ cf8c53af): the plane arrangement IS the adopted path for 935,
and the bridge never reached it — the arrangement reads `sections` while the bridge
lived only in `all_edges` (the greedy input). Fixed: bridge sections now augment the
section list ahead of both paths. STILL 2 interior regions (`ARR935 traced=3
interior=2`): the bridged chain's NORTH anchor (38.05,-41.3002,9.8922) must land on
935's ridge boundary (x=38.05) for the chain to separate a region, and the T-junction
there does not register — the endpoint likely sits just past the tol*100=1e-5
endpoint-T window off the ridge chord, or the ridge edge in the arrangement inputs is
the unsplit image piece whose interior the endpoint misses. TWENTY-FIRST PASS
(2026-08-03, measured): NO boundary input lies within 5e-3 of that anchor — the point
is a MID-FACE junction where sections i=16 ((38.05,-40.4644,9.6350)→anchor, the
x=38.05 line) and i=18 (A→anchor) meet at t=1.0/t=1.0. The x=38.05 line is INTERIOR
to 935 (the neighbour wall's crossing), not its ridge. So the would-be separating
chain runs (38.05,-40.4644,9.635)→anchor→A→B→[bridge]→C: south end now
boundary-anchored at C, but the NORTH continuation past (38.05,-40.4644,9.635) — the
SEVENTH-pass z=9.635 corner, the very first near-miss/orphan this campaign measured —
must ultimately reach 935's boundary for the arrangement to separate the region, and
it evidently does not (2 interior regions traced). TWENTY-SECOND PASS
(2026-08-03, the adjacency dump REFRAMES everything): 935's chain IS boundary-anchored
at BOTH ends (north anchor (38.05,-40.4644,9.635) is a degree-3 boundary vertex; south
anchor C via the bridge), and the arrangement's 2-region partition is locally CORRECT
for its inputs. The real missing piece is a WHOLE ABSENT SECTION, not a pendant: the
result's rim edge (38.05,-41.3002,9.8922)→(38.05,-42.7477,9.8922) (1.45 long, use-1,
carried by the A-side x=38.05 wall products) shows A's material continues south of
-41.30 at x=38.05 — face 398's pair with 935 contributed only [(38.05,-40.4644)→
(38.05,-41.3002)] because 398 ENDS there, and the NEXT A-wall face's pair with 935
(which owes the [(38.05,-41.30)→(38.05,-42.7477)] continuation) produced NOTHING among
935's 20 arrangement inputs. Without it 935's partition lacks the full A-boundary
trace, the hexagon merges into the big kept east region, and the rim mismatches.
TWENTY-THIRD PASS
(2026-08-03, target corrected): there is NO "next wall" — face 398's bbox spans the
WHOLE y∈[-42.95,-38.15] at x=38.05, and the 1.45-long rim edge (38.05,y,9.8922) does
NOT lie on 935's plane at all (plane check: 0.0186 residual) — it lies on the x=38.05
WALL plane. The "hexagon on 935" model was wrong: the six-edge hole rim is a 3D loop
across several faces, and the missing cover is a product of WALL 398 (or its B-side
partner at x=38.05) whose top boundary should run at z=9.8922 between y=-41.30 and
y=-42.7477. TWENTY-FOURTH PASS
(2026-08-04, answered): wall 398 splits into THREE — `1489` Outside (y≈-38.8, kept),
`1490` INSIDE/DROPPED (probe (38.05,-41.19,6.78)), `1491` Outside (y≈-42.83, kept).
1490 is ANOTHER STRADDLE: genuinely inside B's band at its probe (low z), but its
top-south strip (z>9.86, y -41.30..-42.75) sits ABOVE B's slope and must be kept —
that strip is exactly the six-edge hole's wall side. The wall lacks a split along
z=9.8922 south of y=-41.30: the rim edges (38.05,y,9.8922) lie on x=38.05 ∩
z=9.8922, i.e. the section owed by pair (398 × B's HORIZONTAL top face at z=9.8922),
which is missing/truncated in 398's inputs. TWENTY-FIFTH PASS
(2026-08-04, answered): NO z=9.8922 plane exists in either solid, and no FF pair
produces the constant-z rim line — because the rim edges (38.05,y,9.8922) are A's OWN
OPERAND RIDGE (the boundary edge between A-slope `400` and A-wall `398`; their planes
meet along that horizontal line). The slope side of the ridge survives (product
`2323<-400`); the wall side sits inside the straddle-dropped middle band `1490`. So
the missing face is the wall strip y∈[-42.75,-41.30] bounded ABOVE by the ridge — and
the split 398 lacks is the VERTICAL cut near y≈-41.30 below the slope line (B's
boundary at the strip's north edge), which would separate the keep-strip from the
genuinely-inside middle band. TWENTY-SIXTH PASS
(2026-08-04, answered): NO vertical section exists — the only (398 × B) section near
the strip is the slope line, and it arrives at restrict ALREADY truncated at
(38.05,-41.3002,9.8922). Crucially the (400×935) section truncates at the SAME point
(twenty-second pass data): two different pairs, both against 935, stopping at the
identical spot — a COMMON CAUSE in how pair sections against face 935 are clipped
(the trimRR mutual-overlap arm or 935's FaceExtent), not per-pair noise. 935's own
outer polygon DOES extend south to y=-42.75 (IN935 dump), so the truncation is not
its true extent. TWENTY-SEVENTH PASS
(2026-08-04, THE CAMPAIGN'S FINAL FORM): the trim probe shows clip_a ([0.205,0.727],
the wall's own polygon) produces the -41.30 cutoff and it is CORRECT — the slope line
rises through the wall's top ridge (z=9.8922) there; the wall genuinely is not cut by
935 south of it. Assembling every pass: the six-edge hole is a 0.03-TALL SLIVER along
the ridge (z 9.863..9.892, y -41.30..-42.75) where A's TOP SLOPE `400` and B's BOTTOM
SLOPE `935` are NEAR-COINCIDENT — the sixteenth pass proved A/B/C sit on plane 400 to
1e-10, i.e. the junction chain rides the two slopes' common line, and the strip region
between them is thinner than the operand facet noise. This is the COAXIAL SAME-DOMAIN
FRONTIER (see the memory of the same name), planar-sloped: SD does not pair the
asymmetrically-fragmented near-coincident slope pieces (2067 Inside-dropped straddles;
2323 kept with an exposed rim), and every chain/section/straddle artifact of passes
14-26 is collateral of that one unpaired overlap. FIX ALTITUDE (twenty-eighth pass,
likely a fresh session): same-domain detection for NEAR-coincident plane pairs at
operand-noise separation (the `surfaces_same_domain` d-tolerance is `tol.linear`;
these planes differ by ~1e-3 in d and a few 1e-4 in normal) with the partial-overlap
split machinery — the exact configuration the SD scope memory
(`project_gfa-samedomain-scope`) predicted. Alternatively the mesh-fallback remains
correct for this fixture; a product decision on chasing exact-analytic here is fair
game given 27 passes of engine hardening already shipped from this campaign. ALSO worth implementing independently
(robustness backstop, foil-gated): straddle DETECTION in classification — classify
each sub-face at 3-5 spread samples instead of one; disagreement marks the sub-face
as straddling (under-split), which can at least abort with a precise diagnostic
instead of silently dropping volume. Every recent kumiko root would have been caught
at its face by that check. The campaign's
generic lesson is now sharp: every remaining defect is one instance of "a pair
section truncated where operand geometry rides a boundary, cascading into a
straddle-drop"; each pass peels one instance, and the same probe recipes
(WALL/RESTRICT/SEL) resolve each in one run apiece. Also REVERTED this pass (no effect,
principled but unverified): a degenerate-refine rescue in `snap_to_boundary_junction_
band` (detect a flat refine objective, re-refine along the nearest TRANSVERSAL boundary
edge within 3e-3) — the B endpoint never routes through a degenerate snap; keep the
idea in mind if a future corner shows the degenerate-refine signature directly.
Instrument recipe that finally localized the mint: bisect topology vertex scans across pipeline
stages, then an env-gated backtrace in `Vertex::new` on the literal coordinate — probes at the
phase level all lied because the noise was born at EMISSION, not intersection.
CAPTURE GAP worth fixing once: `compoundCut(base, tools[])` passes its tools as an ARRAY, so a
number-only argument filter captures the base and silently drops every tool — the op then cannot be
replayed at all. Flatten arrays and typed arrays in any boolean-capture hook.
`crates/io/examples/replay_pair.rs` now takes `TOOLS=<paths>` to replay a `compound_cut`, which a
pairwise replay cannot reach.
CAVEAT on probe numbers here: a standalone probe run of `scenario-dividers-on` reports
`bnd=0 nm=7` in 555 s where the in-suite run asserts `bnd=6`. Same context-dependence the goma
notes record (in-matrix runs are cache-warm); do NOT compare a standalone probe number against a
suite number.

## Open growth shell: the 364 ms lattice fuse, characterized

`crates/io/tests/kumiko_lattice_fuse_inmem.rs` is the first sub-second repro this family has had.
`BK_OPEN_SHELL=1` with `replay_pair` gives the anatomy directly:

- The lump is **67 planar faces, signed volume +153.5**, bbox x[32.792,39.722] y[-42.950,-38.150]
  z[26.253,34.800] — a genuine chunk, not a sliver, which is why the guard refuses to drop it.
- Its unpaired edges are **exactly four, and they form a CLOSED QUADRILATERAL**
  (38.050,-42.748,29.866) -> (38.000,-42.750,29.830) -> (38.000,-40.932,29.260) ->
  (38.050,-40.909,29.289). So this is ONE MISSING FACE, not a tear.
- That quad BRIDGES x=38.000 and x=38.050, i.e. it caps the tool's deliberate
  **`SLAB_OVERLAP = 0.05`** gap — the same 0.05 mm overlap the goma notes record.
- Every one of the four carries `same_id_outside=0 coincident_other_id=0`: the partner exists
  nowhere in the selection under any identity, so it was never created rather than mis-selected.
- `BK_SUBFACE_BOX` over the gap shows neighbouring 0.05 mm slivers ARE created and classified
  `Inside` (Id 1447, 1961, 1966), which is correct for a Fuse — but nothing covers the quad itself.

**REFUTED: plane-plane FF aliasing is NOT the mechanism**, despite the matching 0.05 mm signature
and the standing note that plane-plane "stays theoretically susceptible ... no repro exhibits it".
Ungating #1224's exact slab clip to plane-plane leaves the fuse failing IDENTICALLY (same 67-face
lump, 368 ms). So this repro does not exhibit that hazard and the gate should stay as it is.
The quad's four corners are coplanar (normal ~(0.570,-0.244,-0.777) — a slanted lattice-strut
plane), so it can only be a trimmed piece of an INPUT face; a boolean emits nothing else.

**REFUTED #2: it is NOT an incomplete face partition.** `BK_SUBFACE_SRC=<id>|all` (new, in
`builder/mod.rs`) totals a source face's pieces against the source's own area. The obvious suspect
`Id(371)` — whose pieces bracket the quad — tiles EXACTLY: 29.143651 = 29.076256 + 0.067395,
uncovered 0.000000. Scanning ALL source faces, only one fails to tile and by 8e-6, i.e. fan-
triangulation noise. So every source face is fully covered and the quad is not a missing piece of
any of them.
**AND THE "MISSING FACE" FRAMING IS ITSELF OVERTURNED (refuted #3, the important one).** Classifying
either side of the quad with the independent operations oracle (`POINT_IN` on `replay_pair`, now
wired; instrument verified — a far-away point reads Outside/Outside):

| point | vs A | vs B |
|---|---|---|
| quad + normal | **Inside** | Outside |
| quad − normal | **Inside** | Inside |

Both sides of the quad are INSIDE the union, so **no face belongs there at all** — one would be an
internal membrane. That much is solid: the samples sit immediately either side of a real boundary,
offset along its normal.

**CORRECTION, and the method lesson behind it.** An earlier pass added "and the lump's own interior
is inside A", concluding the whole lump is an internal artifact. That is NOT established and the
claim is withdrawn. It rested on a BBOX-CENTRE sample; adding a vertex-CENTROID sample (now printed
by the `BK_OPEN_SHELL` probe) contradicts it outright:

| pair | bbox centre | vertex centroid |
|---|---|---|
| 1+2 | Inside A | **Outside / Outside** |
| 2+3 | Outside / Outside | **Inside A** |
| 3+4 | Outside / Inside B | Outside / Inside B |

**For a non-convex OPEN shell neither a bbox centre nor a vertex centroid is a valid interior
sample** — both can land outside the lump, and here they disagree on two pairs out of three. Only
points taken adjacent to an actual face and offset along its normal mean anything. So the standing,
supported finding is the narrow one: no face belongs at the quad, hence the faces owning those four
free edges are what needs explaining. Whether the lump as a whole is spurious is OPEN.
**Connectivity probe so far, and an honest status.** `BK_SUBFACE_SRC=all` now also reports sources
whose pieces TILE but where NONE was selected — a hole an area-gap scan cannot see. There are three
such sources, all tiny (areas 0.040, 0.009, 0.001), none matching the ~0.117 quad. So the boundary
gap is not a fully-dropped source either, and the four free edges still have no face anywhere
sharing their positions (`same_id_outside=0 coincident_other_id=0`).

**Take stock before continuing.** This single failure has now had five successive diagnoses, four of
them overturned by later measurement (FF aliasing, incomplete partition, missing face, spurious
lump). The refutations are durable and the probes are reusable, but the rate of progress toward a
FIX is low and each pass has cost a full iteration. `debugging-doctrine` exists for exactly this
shape. Anyone picking this up should consider whether a different entry point is cheaper than
continuing the current thread — for example instrumenting the shell WALKER directly (why it starts a
new shell at these edges) rather than continuing to characterise the result it produced.

**NEXT: chase SHELL CONNECTIVITY, not a drop-it discriminator.** The earlier proposal — drop a
growth shell whose interior looks internal — is retargeted: the faces are legitimate, so dropping is
never right here. The question is why the shell walker emits this set as a SEPARATE shell instead of
joining it across the quad region. Useful context if that work needs the operands: they are NOT in
scope at assembly (`build_solid`/`build_solid_with_origins` take only `selected: &[SelectedFace]`
plus cap planes, and `SelectedFace` carries `face_id`, `source_face`, `reversed` — no operand tag),
and `brepkit-algo` cannot call `operations::classify_point` under the layer rules.

**SYSTEMATIC BUT PAIR-SPECIFIC.** Within op29's tool merges, `1+2` (67 faces), `2+3` (33) and `3+4`
(65) all abort while `1+3` and `1+4` fuse clean (F=872 / F=1024, free=0). So it is neither a one-off
nor universal — it depends on the pair.
**RESOLVED: every lump face is a LEGITIMATE union boundary; the lump is real material that failed
to CONNECT.** Sampling per face either side along its own normal, with the sample taken at the face
INTERIOR (vertex centroid of the outer wire), over 24 faces of the `1+2` lump:

| side | reading | count |
|---|---|---|
| minus | Inside A or Inside B | **24 of 24** |
| plus | Outside / Outside | 23 of 24 |

Material behind every face, empty in front — that is exactly what a union boundary face looks like.
Combined with the quad result (material on BOTH sides there, so no cap belongs), the coherent
picture is: **the lump is a genuine piece of the union whose faces are correct, and the failure is
that the shell walker put it in a SEPARATE shell instead of connecting it** to the neighbouring
faces across the quad region. Not a missing face, not a spurious selection, not an under-partition.
So the assembly guard is RIGHT to refuse to drop it, and a "drop it when it looks internal"
discriminator would have been the wrong fix — chase shell connectivity instead.

**RETRACTED, with the method lesson.** An earlier pass reported "8 of 24 faces bound empty space on
both sides, stable across offsets 0.02/0.05/0.15" and built a per-face-discriminator plan on it.
That was an artifact of sampling the midpoint of the face's first boundary EDGE: at a convex edge,
offsetting perpendicular to the face exits the material on BOTH sides, so perfectly good faces read
as bounding nothing. Offset-stability did not save it (the artifact is offset-independent), and
neither did deflection-stability (identical verdicts at 0.01/0.002/0.0005) — both looked like
robustness and were measuring the same wrong point. **When classifying which side of a face carries
material, sample the face INTERIOR; an edge or vertex point is never valid.** The same rule already
appears here for notched sub-face seeds; this is the classification-probe form of it.

PROPOSED DISCRIMINATOR: a growth shell whose INTERIOR classifies inside one of the operands is not
new boundary and can be dropped; the fused-foot lump was genuinely outside both.

**COST AND BLAST RADIUS, measured before attempting it — this is why it is not done yet.** The
operands are NOT in scope at assembly: `build_solid`/`build_solid_with_origins` take only
`selected: &[SelectedFace]` plus cap planes, and `SelectedFace` carries `face_id`, `source_face`,
`reversed` — no operand tag and no solid. So the discriminator needs either the operand solids
plumbed down (changes a load-bearing signature) or `assemble` returning open lumps for the caller to
judge. `brepkit-algo` also cannot call `operations::classify_point` (layer rules) and must use its
own `classifier/ray_cast.rs`; and an OPEN shell cannot be classified reliably, so any test must run
against the closed OPERANDS as the measurements above do. Do not rush this into a guard that exists
to prevent a silent material-deleting regression.

**SYSTEMATIC BUT PAIR-SPECIFIC.** Within op29's tool merges, `1+2` (67 faces), `2+3` (33) and `3+4`
(65) all abort while `1+3` and `1+4` fuse clean (F=872 / F=1024, free=0). So it is neither a one-off
nor universal — it depends on the pair.
**VALID SAMPLING DONE — the lump MIXES real boundary with faces that bound nothing.**
`BK_OPEN_SHELL_FACEPTS=1` emits, per lump face, a point either side offset 0.02 along that face's
own normal (the only sound sample for a non-convex open shell), and `POINT_IN` now takes a
semicolon-separated batch. Over 24 faces of the `1+2` lump, every PLUS-side point is Outside/Outside,
and the MINUS side splits:

| minus-side | count | reading |
|---|---|---|
| Outside / Outside | **8** | empty on BOTH sides — bounds nothing, spurious |
| Outside A / Inside B | 6 | legitimate union boundary |
| Inside / Inside | 2 | legitimate |
| OnBoundary A / Inside B | 1 | legitimate |
| OnBoundary A / Outside B | 7 | COINCIDENT with A's surface — see below, not an artifact |

**Offset-stability check (0.02 / 0.05 / 0.15) settles both groups.** The `Outside/Outside` count is
**8 at every offset** — an offset-independent fact, not a sampling accident. And the `OnBoundary A`
rows persist at 0.15 too, which is far past any tolerance: those faces are genuinely COINCIDENT with
operand A's surface rather than ambiguously sampled. So the stable decomposition of the 24 sampled
faces is roughly **8 bounding nothing + 7 coincident with A + 7 legitimate boundary**.

So the lump is neither "all real material" (which the guard assumes) nor "all internal artifact"
(the withdrawn claim): a third of its faces bound empty space on both sides, and another third sit
on an operand surface — which puts the coincident-face/same-domain machinery in scope as a likely
contributor. A discriminator therefore cannot be a single whole-lump verdict; it has to be per-face,
and it now has a measured signature to key on.
CAUTION on `BK_SUBFACE_BOX`: it tests face VERTICES against the box, so a face whose corners sit
outside will not register; do not conclude a face is absent from that probe alone (the quad's own
corners ARE its vertices, which is why it is a valid conclusion here).

## Live campaign: kumiko / goma

**STATE 2026-08-04: the lattice-band arm of this campaign is CLOSED on main** (#1302 —
`kumiko_lattice_bands_fuse_closed` un-ignored; see the Closed section for the mechanism). The
old parked branch `fix/kumiko-corner-window-cut` is GONE from the remote (its 5
`kumiko_corner_window_inmem.rs` fixtures and their data went with it); its four documented
roots remain unshipped, and re-attempting them means re-capturing fixtures first. The
thickwall ready-repro still aborts identically on the new machinery (`open hole shell with 9
faces`, pre-shell-fix operands — re-capture before drawing conclusions). TOOL-SIDE RE-PROBE DONE
2026-08-04 on the post-campaign kernel (wasm built `--skip-opt`, deployed via the
`parity-loop.sh` copy step; bypass pnpm's dep check by invoking
`./node_modules/.bin/vitest` directly or the purge prompt clobbers the copied kernel):
`gomaBoundaryProbe`, `kumikoNubProbe`, `dividerCrossLap`, `honeycombManifoldCheck` — 4 files,
8/8 tests green in 53 s. Note the tool's test files were RENAMED since the recipes below were
written (`topologyParity` and `mitsukudeNmProbe` no longer exist; the probes above are their
successors). THE FULL EXPORT MATRIX IS GREEN
(2026-08-04, same kernel): all four arms — `binGenerator.export.baseStyles`, `.binStyles`
(wall patterns incl. kumiko/mitsukude), `.customShape`, `.halfSockets` — 73/73 tests in ~31 s
total under the DEFAULT vitest config (the profile config's include misses them). The
historical "14 min/arm with failures" era is over; the matrix now runs in seconds and passes
entirely. The isolation probes (4 files, 8/8) are green on the same build.

**Historical state (pre-closure):** branch `fix/kumiko-corner-window-cut` closed four real
engine defects with fixtures but was **PARKED** — it regressed goma from 8 to 65 exported
boundary edges and 540 s to 817 s.

Measured A/B, same day, same tool commit, each overlay md5-verified through brepjs's own resolution:

| arm | boundary edges | duration |
|---|---|---|
| main `a8379d7b` | 8 | 540 s |
| **root 1 off, root 2 on** | **8** | **539 s** |
| branch (roots 1+2+3+twin+seed) | 65 | 817 s |

- **Root 1 owns the entire regression**; root 2 (graze escalation) is byte-identical to main and
  needs no work. Root 1 is the non-periodic band DCEL rescue (`face_splitter/mod.rs` ~5251).
- **The branch cannot be split**: disabling either root alone fails the same 3 kumiko fixtures, so
  there is no subset that ships the neutral work without the regressing work.
- Root 1's gate already requires `greedy_broken` (self-cross ∨ nested ∨ degenerate-area) and already
  rejects an unhealthy DCEL. So the replacement partition is structurally VALID and still worsens
  the export — the twin-classification shape, i.e. the gate is too broad, not the trace too weak.
- **No native proxy exists**: the 2026-07-24 `goma-bisect` even bands replay identically
  (F=495, free=0, ~230 ms) with root 1 on and off. Every attribution costs a ~14-minute arm until
  someone captures fresh goma operands.
- Next instrument is OBSERVATION, not another guessed predicate: `log::warn!` at root 1's adoption
  site plus a warn-level ring-buffer capture (~1069 lines for the whole export), then compare a
  firing that helps against one that hurts.

## Closed: root cause + where the detail lives

- **Cone/cylinder ∪ box tangent section circle** — the last primitive-boolean fallback; every
  tangency count (0/1/2/4 walls) now fuses clean and analytic, closed as collateral of the
  classifier conflict re-cast (#1357) and the SD cross-shell gate (#1360). Pinned ACTIVE with
  exact face counts: `boolean::tests::tangent_wall_fuse_configurations_stay_analytic`.
- **Kumiko lattice band fuse — CLOSED after 29 passes** (`kumiko_lattice_bands_fuse_closed`
  un-ignored and green). Final mechanism, all in the face splitter: (1) DEMAND-GATED outer-wire
  image expansion — a planar hole-free face expands a boundary edge's pave-split images only
  when an interior image junction sits within 3e-3 of a section endpoint but NOT within the
  weld band (coincident junctions are already served by the calibrated splitting; expanding
  them de-analytics the divider-lip fuse, and expanding periodic laterals' seam edges breaks
  the perpendicular-cylinder fuse — both gates measured); (2) pendant→boundary-vertex bridge
  (section-free targets only); (3) pendant→pendant bridge (mutually nearest within 3e-3,
  10x isolation, twin-deduped) — the final 2.3e-3 gap was TWO section pendants on face 400.
  The "near-coincident slope SD" framing of pass 27 was REFUTED by direct measurement (planes
  15° apart; A/B/C share their intersection LINE, which does not make the planes coincident).
  Honeycomb raw residuals re-pinned (pcut1 53→83, pcut2 28→30, production tests unchanged).
- **Torus ray-cast arm** — whole-torus faces (degenerate boundary, < 3 polygon verts) were
  DROPPED from parity counting entirely, and full-tube laterals fell to the flat polygon
  fallback. `FaceGeom::Torus` + `math::intersect_line_torus` (the solver already lived in
  math, no layer move needed) close both; TWO-RIM tube bands decline by design
  (side-ambiguous from boundary vertices alone). Full-u detection is largest-gap-based
  (max−min fails on sampled circles). `ray_cast.rs::whole_torus_classifies_inside_and_outside`

- **Kumiko corner cut** — 4 roots: no rescue for a partial cylinder band; graze refinement scaled to
  face extent not arc length; NURBS boundary edges represented by their CHORD; and a reverse-twin
  loop misread as a flipped complementary band. `kumiko_corner_window_inmem.rs`
- **Six-tool corner residual** — `sample_interior_point`'s edge-midpoint fallback accepted a seed a
  GRAZING ray placed exactly on the boundary; the centroid branch already had the clearance test.
  `classify_2d::interior_of_notched_polygon_clears_the_boundary` (pins verbatim f64 literals — round
  them and the case stops reproducing)
- **Segmented revolve emitted inverted solids** — the full-revolution path normalized winding, the
  segmented path did not. `revolve.rs` + `regress_kumiko_corner_wedge.rs`. New oracle:
  `measure::oriented_solid_volume` (the old `solid_volume` is a MAGNITUDE and cannot see inversion)
- **Arena `reserve` doubling** — a bulk hint rounded up to `2*capacity` on a multi-million-entity
  arena, holding both buffers and aborting the 4 GB wasm heap. `topology/src/arena.rs`
- **GFA multi-region acceptance** — AABB overlap ≠ disjoint for rotated bars; Euler surplus bounded
  at 2 assumed spheres but a lattice yields rings; containment ≠ nesting for a piece in a ring's
  hole (now ray-parity). PR #1239
- **FF AABB pre-filter aliased on straight sections** — 16 uniform samples over a 20 mm generator
  missed 0.83 mm lattice bands. Fixed by exact slab-clip, gated to quadric partners. #1224,
  `goma_wall_band_cut_inmem.rs`
- **Tessellation nested-hole seeding** — hole flood-removal seeded at each wire's CENTROID, which is
  identical for concentric wires; and only ODD-depth wires bound a hole. `oring_nested_holes.rs`
- **T-lip band cut** — a correctly-split ring was misclassified Inside because the depth probe
  overshot a 1.2 mm annulus. `lipband_cut_inmem.rs`
- **Label-sockets tab attach** — a section's fate decided by INTERIOR SAMPLING is blind to overhang
  at the ends; plus an arrangement trigger keyed to a demonstrated failure signature.
  `labeltab_attach_inmem.rs`
- **Intwidth wall tangency** — two solvers recomputed tangential intersections ±1e-6 off
  (positional error ≈ √(2r·residual)). `intwidth_tangency_inmem.rs`
- **Lite magnet-pad graze fuse** — a graze heuristic keyed to face extent is blind to corner-window
  exits, which can be smaller than either face. `lite_pad_graze_fuse_inmem.rs`
- **Mesh-boolean co-refinement rewrite** — T-junctions, coplanar collapse, winding coin-flips.
  `relief_meshbool_fallback_inmem.rs`
- **Trimmed-torus ray-cast** — 3 stacked roots; both local Ferrari quartics missed real roots, and
  check's `face_aabb` collapsed cap discs to a point. `check/src/classify/ray_surface.rs`
- **Dovetail family** (cornerclip intersect, interior identity, relief cut, A1 nub/hole/fuse,
  fractional seam pocket) — see `crates/io/tests/dovetail_*.rs` and `fracplate_seam_pocket_inmem.rs`
- **halfSockets / fractional-width / socket-assembly family** — loft faceting, disconnected-loop
  twins, corner crescent, hole-winding normalization. `halfsockets_*.rs`,
  `fracwidth_corner_crescent_inmem.rs`, `socket_assembly_fuse_inmem.rs`
- **snapClip + fit-offset** — connector key, junction disc, slot cuts, groove-mouth slivers,
  deepened notch. `snapclip_*.rs`, `fitoffset_groove_mouth_inmem.rs`
- **Kernel-poison panic** — wasm32 is `panic=abort`, so `catch_unwind` is INERT and a trap strands
  the borrow flag forever (recovery = new `BrepKernel`). Panic text now survives via
  `crates/wasm/src/panics.rs`
- **Divider + floor pattern families** — 18 of 21 failures were ONE missing brepjs adapter method
  (`applyMatrix` had no compound case); 21 → 4 after the fix. Not a brepkit defect

## Refuted: do not re-try

- **A universal smarter merge-key for duplicate edges** — see TERMINAL above; it is unbuildable,
  and the sanctioned pattern is splitter-side midpoint splits per case.
- **Narrowing root 1 to the kumiko loop signature** (`loops_have_out_and_back`, or a lone
  self-crossing grand tour). Keeps all 5 fixtures and every foil green, and leaves goma
  BYTE-IDENTICAL — so loop shape does not separate the corner wall from goma's bands.
- **Splitting the kumiko branch** to ship the neutral fixes without roots 1+2 — each root is
  independently load-bearing.
- **"The constraint is the SPLIT ITSELF disturbing reconciliation"** — the newly-admitted splits were
  all straight axis-aligned wall runs stored as NurbsCurve; a span-local sagitta gate fixes it.
- **`shell_is_outward_oriented` / `signed_volume_of_shell` being inverted** — both are exact on a
  known-good cube (`flux_orientation_probe.rs`); the operand really was inward.
- **The goma odd bands as a GFA defect** — they were brepkit's own mesh-fallback output, i.e. GIGO.
- **Ellipse aliasing at FF filter 2 on the goma lump** — the 49 dropped ellipses miss by 0.108 mm at
  0.055 mm sampling, a genuine 2× separation.
- Also refuted, each once: coincident coaxial cylinders as the corner-cut root; a classification
  error there (an independent `classify_point` oracle agreed with GFA); plane-gated seed correction;
  arc-cornered wires as the nested-hole trigger; helix sweep as the goma cause
  (`helical_sweep_is_watertight_across_turns_and_segments`).

## Recurring traps (the distilled, expensive lessons)

- **Marched/fitted section geometry is good to ~1e-6; every exact-tol (1e-7) gate it meets needs a
  weld-scale (100·tol) band.** Four separate gaps in one family were this.
- **A sampled proxy gated at an exactness tolerance** is the single most common defect shape here
  (five instances): `best_d` bounded by sample SPACING, 16-sample AABB scans, uniform-t restriction,
  chord polygons under-covering by a sagitta.
- **Interior points of notched/symmetric sub-faces land on feature-plane intersections BY
  CONSTRUCTION** — classification must survive on-plane samples, and a seed must be STRICTLY
  interior. A centroid is not an interior point for concentric or non-convex wires.
- **The face splitter is a web of mutual calibrations.** Run ALL foils on any change:
  d4 gridfinity, honeycomb pcut1/pcut3, divider-lip, groove-mouth, junction-disc, cylinder-slot,
  a1corner. Each caught a different wrong discriminant.
- **A trigger keyed to a post-hoc failure signature cannot demote a working case** — the cheap way
  past those calibrations.
- **`solid_volume` is a MAGNITUDE**; only `oriented_solid_volume` sees an inverted or doubled shell.
  It is also translation-VARIANT on a malformed boundary, which needs no second oracle.
- **The by-edge-id manifold gate is BLIND to position-duplicate faces and edges.** "GFA validated OK"
  never proves watertight; use the position-quantized check.
- **All-planar output with zero curved faces, on a shape that should have cylinders, is the fallback
  tell** — but it is weak where the construction is legitimately planar (helix sweeps, box cuts).
- **Never replay a captured operand without printing its free/over counts first.** Captures can be
  fallback-poisoned; a whole iteration has been spent inside that trap.
- **Every `BK_*` knob is NATIVE-ONLY** (`std::env::var` returns Err on wasm32), so they do nothing in
  a tool run. `setLogLevel` + a JS ring buffer is the only handle on kernel internals from JS.
- **`log::debug!` in `fill_images_faces.rs` does not reach a custom logger** that receives
  `builder_solid`'s fine — probes there read as false zeros.
- **A fast-failing scenario family is a signal the failure is PRE-geometry.** The floor family failed
  9 of 11 in 5.4 s; nothing doing real geometry fails that fast.
- **Read raw log lines, not summary counters.** `gomaLogCapture`'s regexes miss
  `GFA boolean failed … falling back` and reported 0 rejections while 12 were present.

## Tool-side measurement recipes and traps

- **Scenario numbers rot.** Always run the control on the SAME DAY and SAME catalog; a stale
  baseline has twice nearly produced a false conclusion. The catalog grew 408 → 435, so comparing
  against the old 37/371 manufactures 26 phantom regressions.
- **Current baseline (2026-08-05, kernel c6dbc14a, md5-verified overlay): the ENTIRE tool generator
  test directory is GREEN, 244 files, 2426 passed / 4 skipped, 0 failures, 297 s wall.** The
  historic 62-failed/372-passed matrix is fully closed on this surface; compare against THIS
  number, not the old one.
- **Overlay verification is mandatory and non-obvious.** Hash the file that
  `require.resolve('brepkit-wasm', {paths:[require.resolve('brepjs')]})` returns, run FROM the
  directory vitest will use. The foils worktree has its OWN `node_modules` (it does NOT resolve the
  parent's). For a brepjs-side change, `npx vite build` then copy `dist/*`; `applyMatrix` lives in a
  content-hashed `shapeTypes-*.cjs` chunk, so grepping `brepjs.cjs` reads as "fix missing".
- **`pnpm exec vitest` triggers a dep check that wants to PURGE `node_modules`** (aborts headless,
  destroying any overlay). Drive `./node_modules/.bin/vitest` directly.
- **`vitest run --project generators` EXCLUDES `__kernel-tests__`** — those need
  `--config vitest.profile.config.ts`. Vitest also does not surface `console.log` through a pipe:
  write probe results to a FILE.
- **Capture recipe:** wrap the RAW kernel's boolean entry points from a tool probe and
  `serializeSolid` each operand. A hook on `fuse` alone fires ZERO times — the export drives
  `fuseWithEvolution`/`cutWithEvolution`. Replay with `crates/io/examples/replay_pair.rs`
  (`A=`, `B=`, `OP=`) or `replay_cut_capture.rs` for base+tools chains.
- **A multi-case tool probe MUST make a fresh kernel per case, or run one case per process.** The
  brepkit kernel is a per-worker singleton whose borrow flag strands permanently on a trap
  ("recursive use of an object"), so the first failing case poisons every later one — a sweep then
  reports one real number followed by N identical bogus errors. `exportIntegrityRunner` recreates
  the kernel for exactly this reason; a hand-rolled probe does not. Cheapest fix is a `CASE=` env
  selector and one vitest invocation per case.
- **Tool probes under `__kernel-tests__` are UNTRACKED and get cleaned.** Budget for re-writing one
  rather than treating its absence as a missing capability. The tool is a SEPARATE repo other
  sessions commit to concurrently — check its `git status` before running anything there.
- **The foils worktree can vanish mid-session.** On 2026-08-01 `.worktrees/brepkit-foils` was
  removed by another session while a verification was queued, and the main checkout had moved to an
  unrelated branch. The probes survive on branch `diag/brepkit-kernel-foils` (local and origin), so
  restore with `git worktree add`; it needs its OWN `node_modules`. Do NOT run measurements in the
  main tool checkout — another session is usually working there.
- **`brepkit-render`'s `compute_mesh_lod` SIGSEGVs intermittently** (pre-existing, ~50% of runs),
  aborting `cargo test --workspace` early and masking later suites. Use `--exclude brepkit-render`.
- Scenario snapshot tests pin EXACT reference-kernel triangle counts; a different kernel can never
  match them. Received-below-expected is benign density difference, received-10x-above is a defect.

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
