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

**Inventory status (2026-09-15): FIVE deferred-defect pins**, each owned by an
OPEN row: `groupedscoop_tool_tessellates_watertight` +
`groupedscoop_cut_removes_at_most_the_tool` (the two-stripe corner model),
the two kumiko wrap cuts (`kumiko_wrap_strut_cut_inmem.rs`,
`kumiko_wrap_lattice_cut_inmem.rs`) and the fanned developable band
(`tessellate::tests::band_with_split_ruling_keeps_triangles_on_the_arc`).
Every other `#[ignore]` is an explicit
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
here. **Drift measured 2026-09-14 (tool HEAD 4a66decb, brepkit 3.3.9 and 3.4.0,
`BREPJS_KERNEL=brepkit`, forks pool, 2 workers, both runs stopped at ~350 of the
catalog's files after an hour of hung stragglers): 108 files fail per kernel, 103 of
them identically on both (421 vs 399 failing tests), and the sampled ones pass on
the reference kernel.** The tool's own CI never sets
`BREPJS_KERNEL`, so its generator suite runs on the reference kernel by default and
brepkit is opt-in via Labs; a month of tool feature growth (wall-cutout corner
radii, text tracking, label-plate icons, click rails) was developed against the
reference kernel only. Roots found so far (three probes, one kernel fix; see OPEN):
the trimmed-cylinder bounding box (closed below), brepjs `scaleDrawing` on the
brepkit path, and brepjs `compound()` of compounds. Native criterion CAVEAT: the
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
  crossing technique is proven and reusable. Measured 2026-09-24 against
  `make_sphere(3, 32)`: every box face crossing the equator falls back (a half
  at any turn or tilt, a wedge, a slab at x > 0.5, a corner reaching below
  z = 0), though a half or wedge crosses it at chord vertices: its first
  failure is `split_noseam_face_direct`, which cannot chain two arcs meeting at
  the pole, handing them to `split_noseam_by_arrangement`, which keeps only a
  collar. REFUTED: an exact-circle equator in `make_sphere` as a drop-in (half
  cuts turned 5, 11.25 or 30 degrees come back uncut, 2 faces, silently wrong).
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
| **Two-stripe convex corner with an unfilleted spoke: the corrected engine's closure is not a solid** (`crates/blend/src/corner.rs` `build_junction_fan` 2-stripe arm: horn torus, mixed-radius band, runout; fixture `crates/io/tests/groupedscoop_fillet_cut_inmem.rs`, ignored ready-repros `groupedscoop_tool_tessellates_watertight` + `groupedscoop_cut_removes_at_most_the_tool`) | Both stripes run full-length to the far wall (each end section lies IN the neighbouring wall's plane), so the two bands cross each other in the 2x2x2 corner cube, the bottom face's two contact lines cross at Q and the chord between their far ends closes a reversed lobe (a bowtie outer wire), and the horn torus pinching at the spoke point P spans the cube a third time. 3.3.9 had the same model: its "exact" cut of the bin removed 2.1x the tool's volume with 144 open mesh edges (`replay_fillet_variable` `CUT=` mode), which is the tool file's 2 failures on that version. After the twin fix (Closed below) the tool is manifold by position and the cut stays exact and manifold by id and position, but removes 1.5x the tool volume, fails Euler, and its mesh has 105 open edges; the tool mesh itself has 38 (the mixed-radius NURBS band's boundary does not follow the shared circle edge's polyline, and the bowtie lobe traverses its edges same-sense). The fix is the reference kernel's two-corner: intersect the two stripe surfaces (perpendicular equal-radius cylinders at A/C: the exact diagonal ellipse from Q to P; cylinder x torus at D: marched), trim both bands to it, build no patch, end the bottom contacts at Q, split the spoke at P and drop the piece below. B and E are a further class: the r=2 bands on edges 0 and 8 are tangent at x=-7.55 and swallow the 0.584 mm notch (edge 4, wall 5, the arc near E) entirely, which needs stripe-vs-neighbour-face intersection beyond the vertex (the reference kernel's intersection-at-stripe-end procedure). Tool-side same-day pair on tool 4a66decb (3.4.0 stock vs the #1650 build overlaid in the same worktree): `export.solidCutouts` 1 -> 0, `assemblyGenerator.scenario` 13 -> 10 (block/comb/riser at the base centre), `export.groupedScoop` 4 -> 4 (boundary edges 36/31/69/15: this row), `combriser` 4 -> 4 (boundary edges 117/61/120 + 14 degenerate triangles), `scenario.solidCutouts` 5 -> 5 (snapshot mismatches, also on 3.3.9). The 3.3.9 counts were 2 / 0 / 6, so the bump gate is still shut; combriser and the remaining assembly failures ("cluster fuse degraded to mesh fallback", open meshes) are UNATTRIBUTED and the next dig |
| **Rounded-prism rim ease (the assembly parts' second stage): the plane x cylinder analytic stripe rounds the cove side of a rounded-prism cap** (diagnostic `comb_part_probe` with `STAGE=rim FLUX=1`; mechanism `plane_material_inside_cylinder` on branch `wip/prism-corner-fillets`) | After the corner fix the comb's r=1 rim ease (four lines and four r=2.5 corner arcs, eight tangent junctions over a smooth spoke) fails cleanly and the part keeps its sharp rim: the plane x cylinder stripe classifies the cap as a plate around a post (`plane_is_bounded_disc` only knows a full disc), places the four corner-arc stripes at (±32.5, ±8, 35) outside the part, and no cross-section can then be shared at the junctions (16 registered fresh). A local material-side test at the spine (is the cap's interior toward the cylinder axis) puts them right and the rim builds its 8 cylinders + 4 tori + 6 planes, but one trimmed corner cylinder comes out inverted (the pairing walk reaches it through a torus), and the test shifts the `cross_one_row` volume oracle by 79 mm³ (64,889 vs 64,982; oracle 64,968 within 65) while the dumped faces at both stripes it reclassifies (the r=7 boss top and the r=12 step rim, origin (58, 42)) are identical in both modes: untraced. Tool-side this is the difference between a comb or riser with eased rims and one without; the parts are valid either way. Tool-side same-day triple on tool 4a66decb, one worktree (3.4.0 stock -> the #1650 build -> the #1654 build): `assemblyGenerator.scenario` 13 -> 10 -> 6 (the 3.3.9 count, back at parity), `combriser` 4 -> 4 -> 4 was the EMPTY 2x1 base's plate x socket fuse, not the parts (Closed below: coaxial same-domain orientation). `export.groupedScoop` 4 -> 4 -> 4 (the two-stripe corner model), `export.solidCutouts` 1 -> 0 -> 0, `scenario.solidCutouts` 5 -> 5 -> 5 (snapshots) |
| **Assembly cluster fuses degrade to mesh** (`assemblyGenerator.scenario.test.ts`: "arch at the base center" and "tilted block and cradle" throw `cluster fuse degraded to mesh fallback`; operands captured with the `captureScoopOps.test.ts` wrapper pattern, replayed with `replay_pair FUSE_ALL=1 FUSE_MEMBERS=<third member>`) | TILTED only, since the arch closed (Closed entry below): block x cradle (two tilted parts overlapping 5 mm in x) fails GFA ("open growth shell with 18 faces would be dropped") and the mesh fallback is non-manifold; plate x cradle is accepted closed by id with 100 open mesh edges and a volume 2037 mm³ below the raw result. Block x cradle alone (`BK_OPEN_SHELL=1`): the 18-face open shell is the cradle's part outside the block, and its free edges are the cradle's cut boundary on the block's +x face and the block's -y corner cylinder (a line, the groove arcs and two ellipse arcs at x=8): those block faces were never split there, six EF crossings of the cradle's rounded-edge curves with the +x face having been dropped as `outside face boundary`. TUBE: the test passes since the plate x tube nesting closed, and its mouth chamfer (`chamfer(tube, both rims at z=60, 1/3)`) takes the round-rim chamfer (Closed below; tool-side unmeasured). The six dropped EF crossings are correct (those cradle curves cross x=8 beyond the block's rounded corners). SUB-ROOT 1 FOUND: the block's +x face reached the splitter with its two straight NURBS boundary edges (the fillet contact lines) unsplit, so its five sections could anchor on no boundary vertex and were all dropped; `boundary_edges_to_pcurve_with_images` expands a NURBS edge only when circle-like, and a straight NURBS edge with a weld-coincident pave junction now expands too on branch `fix/nurbs-line-boundary-expansion` (parked: with it the pair assembles 34 faces but 26 free edges remain from plane sections overshooting into the rounded-corner region and tiny marched pieces where the cradle's 0.8 mm rounded edge meets the block's corner cylinder; no primitive pin discriminates the expansion yet, an axis-aligned or tilted NURBS-edged box splits fine either way). Next: those two remainders on the captured pair (`asmcap/tilted`, `replay_pair OP=fuse BK_OPEN_SHELL=1`) |
| **Kumiko corner-wrap cut: a cylinder band cut by helical-sweep strut lattices** (fixture `crates/io/tests/kumiko_wrap_strut_cut_inmem.rs`, ignored ready-repro; captured from `slideRailBuilder.test.ts` "is not carved away by a kumiko wrap either") | The generator-suite hang class. Tilted corner struts are rectangles swept along helix segments (`kumikoCornerSlab.ts`): 16 NURBS wall patches per wall, 66 faces per strut, inherently non-analytic. One band-by-strut cut spends 4.2 s of 4.4 s in `phase_ff` (`analytic_nurbs_intersection`, cylinder x helicoid patch), the marcher logs `SSI: curve deviates 1.8e0 from surface(s) (tolerance=1e-5), re-fitting from sample points` ten times, and the exact result has 42 free edges (35 on the strut's NURBS sub-faces) so the op falls back to a 167-face planar mesh; a 3-tool compound cut is 27 s (two arrangements, 24.4 s FF), the 8-tool one 226 s, and the 3x2x6 wrap export runs 21 of them (532 s, 23 fallbacks) where the reference kernel exports watertight. REFUTED: the strut NURBS coming from the adapter's transform refit (brepjs 19.0.4 gives an identical 532 s / 23-fallback profile). Isolated (`crates/io/examples/kumiko_pair_probe.rs`, the same `intersect_nurbs_nurbs(32, 0.01)` call on every band-cylinder x strut-wall pair): 98 AABB-overlapping pairs, 7.7 s; the marcher is ACCURATE (points within 1e-6 of both surfaces, fitted curves within 1e-4) but 8 pairs that never intersect cost 250 to 490 ms each (half the time) and a few intersecting pairs return 2-point fragments (`Id(4) x Id(22)`: 5 curves, 10 points) instead of one section, which is where the 1.8 mm re-fit drift and the free edges come from once the pieces fail to chain. SEED COST CLOSED (#1646): the grid fallback launched Newton from every sample pair within its closeness threshold, so a wall grazing the cylinder ran thousands of failing refinements; it now refines the mutual-nearest pairs and the 256 closest pairs first and runs the full close-pair pass only once one of them has converged (`surface_seeding::find_ssi_seeds_grid`, pins `grid_seeder_*`). Probe 7.9 s -> 2.0 s (the review-driven complete-search version), the single cut 4.4 s -> 0.8 s, the 3-tool compound 27 s -> 5.6 s, same seeds and accuracy. Tool-side on an overlaid wasm (brepjs 19.0.4 stack): the 3x2x6 wrap export 532 s -> 483 s; only the two box-by-3-strut compound cuts halved (53 s -> 26 s each), while the 8-tool compound cut (94-face base, 226 s -> 7516-face blob) and the fuse of that blob (125 s) are unchanged and now own the export, and those tools are themselves 645-face fallback blobs: the chain's FIRST fallback is the flat wall's 62-plane strut lattice compound-cut by four helical corner struts (fixture `crates/io/tests/kumiko_wrap_lattice_cut_inmem.rs`, ignored ready-repro; natively 9.5 s over three fuse-ladder arrangements, 5.3 s of it in `phase_ef`, ending in "open growth shell with 6 faces would be dropped" and 77 non-manifold edges). Close that and the 8-tool cut stops consuming blobs. STILL OPEN: exactness (the raw cut keeps 41 free edges on the strut's NURBS sub-faces) via fragment chaining across the 16 wall patches, and one 400 ms pair (`Id(4) x Id(50)`) |
| **Generator-suite hangs on brepkit** | Attributed for the kumiko wrap (row above); the rest untraced. Single tests block 5 to 19 minutes inside synchronous kernel calls on both 3.3.9 and 3.4.0 (`is not carved away by a kumiko wrap either` 741s, `splits bin with compartments + scoop + thick walls + connectors` 631s, `featureCacheKeyDiscipline` 740s), so a full run is ~50 minutes at 2 forks where the reference finishes in ~4. A hang is a perf-bar defect; undug. 2026-09-15: one of the five lid files `hingeSwing.scenario`, `lidScoopClearance.scenario`, `lidDividerClearance.scenario`, `lidMagnetSeating.scenario`, `lidLabelTabClearance.scenario` held a worker at full CPU for 18 minutes with no output; rerun one at a time under a 10-minute cap, `hingeSwing.scenario` is the one (it alone reached the cap with no test output; the other four pass in seconds). NOT a hang: a per-op kernel log (`KERNEL_OPLOG`/`KERNEL_OPCAP` hook in the verify-340 worktree's `wasmInit.ts`, uncommitted) shows the swing sweep's every `intersectWithEvolution` taking ~83 s in wasm because the bin arrives as a 4763-face planar blob. The blob is minted by `compoundCut(bin with knuckles, [two keyhole pin tools])` (natively 1.1 s to a 4009-face fallback; operands in the session scratchpad `pincut/`): the first tool's cut is exact since the knuckle bore roots closed (Closed below), the second tool's still falls back on two roots, both OPEN rows below (its end face coplanar with the fragmented knuckle end; the sector splitter with the seam inside a section-bounded band). Until both close the sweep stays mesh-versus-mesh |
| **Export and volume meshes fan a developable band whose ruling edges carry vertices** (ready repro `tessellate::tests::band_with_split_ruling_keeps_triangles_on_the_arc`, ignored; live: the `hingeSwing` second pin cut, scratchpad `pincut/bin.bin` x `tool1.bin`) | `interior_grid_resolution` gives cylinders and cones NO interior points on the display/export path (`(2, 1)`), and the boundary-only Delaunay of a band whose length dwarfs its arc only ladders while its ruling edges carry nothing but corners: a ruling split by a section (the pin's tip line cut by the knuckle's step plane 0.4 mm inside each rim) is fanned to a whole rim and the triangles between that fan and the far rim span the entire 70 degrees, flat. The pin's bore integrates 8.6% short, the bin's own export volume 0.7% short (45252 vs 45560 with interior rows), and the mesh divergence agrees with `oriented_solid_volume` because both read the same mesh, so the cut/intersect identity `V(A-B)+V(A∩B)=V(A)` holds while both are wrong: only the fuse-derived `V(A)+V(B)-V(A∪B)` or an oracle scan (axis and cross-section grids, `POINT_IN`) exposes it. Also bites `measure::solid_volume` on a UNIFIED full bore wall: the keyhole-pin compound cut (`compound_cut_by_two_keyhole_pins_meeting_on_a_knuckle_face_stays_exact`) reads 11 mm3 high, one bore's worth, at 0.01 and 0.001 while `oriented_solid_volume` matches the sequential oracle; pin volumes with the oriented integrator. REFUTED: one interior row at the rim's u density (fans stay at the corner vertices); aligned columns at the boundary's u samples plus rows at every vertex v (T-junctions and 173 non-manifold export edges on `dovetail_a1corner_hole0`, triangle budgets blown, and `cross_one_row_fillet` drifts 5% from its reference oracle because the winding vote flips on the refilled faces); refining fat triangles by their centroids (diverges on full-turn walls, 376k triangles on one face). The reference mesher seeds a near-isotropic grid on developable faces; the fix must keep `cross_one_row_fillet` within 0.1% of its oracle and will move the mesh-derived pins `mitsukude_panel_cut` (27027.9 -> ~27095) and `spacer_foot_fuse` (2404.44 -> ~2397.8), which were calibrated on fanned meshes |
| **4x4 mag no-lip noise-band watch (1.04x on the 3.2.38 matrix)** | The only row the reference leads; has oscillated 1.00x-1.06x across 3.2.36-3.2.38 with no kernel change targeting it. Watch, do not chase, unless a fresh same-day matrix shows a real drift |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** | A product call, not just a fix: rejecting means the op fails outright. Mitigation shipped: `boolean::mesh_fallback_count()` + wasm `meshFallbackCount()` let pipelines snapshot-and-refuse |
| **Export angular default (5°) vs the reference's coarser effective default** | Tolerance-parity product choice, not mesher waste: 5° forces 18 segments/quarter-arc on r=0.6 slot corners, ~1.7x triangles vs reference at fine deflection. Revisit only as a product decision |
| **Marched FF sections carry `pave_block_id=None`** | Architectural note without a live repro (the snapClip op-cut-3 case replays clean, fixture `snapclip_export_corner_inmem.rs` ACTIVE). If a new leak lands here, the canonical altitude is pave-block attachment at phase-FF/make_blocks — every face-splitter-level attempt broke calibrated chains |
| **Hinge bin pin cut: the two keyhole pins meeting on the knuckle end face fail as one tool** (tool `hingeSwing` op3029, operands `hinge14cap/op3029_*` in the session scratchpad; replay `TOOLS=<pin0>,<pin1>` for the compound path, `MERGE_TOOLS=1` or `B=fusedpins.bin` for the raw cut) | Pairwise and sequential cuts are exact (117 ms, 245 faces); every batched tool (the two shells concatenated, or their fuse whose 0.075 step ring is pinched along the tails' shared base) leaves the knuckle end face 133 with 23 free and 4 non-manifold edges: it receives 18 sections, the keyhole outline arriving twice (a regular FF piece and a coplanar-phase piece for the same run, `STRACE` with `BK_SPLIT_TRACE=133`), and the splitter emits two pieces that both carry the face's outer boundary. The synthetic full-interior twin closed (Closed entry below); this one differs by the outline crossing the small face's rim. `compound_cut` then SHIPS the batched mesh fallback (the fuse-ladder rung accepts it) instead of falling back to the exact sequential cuts, so the bin becomes a 4078-face blob and every swing intersect after it is a mesh boolean (~85 s each in wasm, 9+ per scenario). Fix the arrangement; the policy rung is the safety net |
| **Lid knuckle fuse: a stepped barrel (r 2.2 flanges, r 1.8 middle, ramp planes) whose axis lies on the lid plate's bottom plane and overhangs the plate's edge** (tool `hingeSwing` op20865, operands `hinge15cap/op20865_*`; five such fuses op20865..20885 build the lid's knuckle row) | GFA leaves 11 free and 3 non-manifold edges (the r 1.8 wall pieces inside the plate and the plate's bottom-face strip inside the barrel are both KEPT, so both operands' sub-faces in the overlap are classified wrong), the fuse falls back, and each later knuckle fuse grows the blob (181 -> 289 -> 397 planar faces); the swing sweep body (888 planar faces) inherits it. Undug; the bin-side knuckle fuse (closed 2026-09-15) had its axis on the bracket's EDGE, this one has it on the plate's FACE |
| **v1 fillet deprecations entangled with the public wasm API** | `try_fillet` still reaches deprecated `fillet`/`fillet_rolling_ball`; migrating changes public behavior — a product decision, not safe cleanup. See `fillet-blend`, `wasm-bindings` |
| **crates.io / GTM items** | Andy-only. Publishing infrastructure works (see MEMORY.md for the release-please `continue-on-error` masking gotcha) |

## Stability campaign: every README status row to Stable

A row flips to Stable only when its whole stated scope clears the bar:
exact result, `validate_solid` clean (orientation on), watertight mesh,
volume against an independent oracle, tests over that scope including
transformed and mirrored inputs, and wasm exposure. Auditing the Beta rows
also turned up defects inside rows the README already calls Stable; those
come first, since a Stable row that is wrong is worse than a Beta one.

| Row | Status / blocker |
|---|---|
| **Evolution (Beta)** | Faithful GFA provenance exists (`boolean_with_evolution`). Gaps: a same-domain merge keeps one origin and marks the other deleted; identical/contained operands and every fallback use `build_evolution_by_geometry`, whose 10-unit centroid cap is scale-dependent; fillet evolution is heuristic only |
| **Defeaturing (Beta)** | `defeature.rs` drops the faces and reassembles an open shell. Needs the gap closed by extending the neighbouring faces |
| **Feature recognition (Beta)** | Dihedrals are signed from outward normals and the edge tangent, adjacency reads every wire and shell, holes are concave cylinders. Pockets group coplanar split faces and open along a floor normal no face in them looks back against, one pocket per floor (a stepped pocket's landing is its own). A fillet is a curved face tangent to two neighbours that are not parallel planes; a chamfer must stand where the edge its two neighbours' planes meet along was cut away (outside it across convex edges, inside across concave ones; a scalene prism's side fails) and be at most half the larger face it bevels (tests in `feature_recognition.rs`). Still heuristic: a chamfer wider than that is missed (a regular prism's sides meet like chamfers, so size is the only discriminant); a full-round edge between parallel faces is not reported as a fillet; an undercut pocket (a face overhanging its floor) is not found; a floor split into patches (coplanar, or within the 0.01 rad flat-edge tolerance) reports its largest patch as the floor and leaves the rest out, since `Feature::Pocket` has one floor |
| **Torus booleans (Beta)** | Audited 2026-09-24 against `make_torus(4, 1.5)` (15 tools x 3 ops, probe `zz_torus_audit` in the session scratchpad): exact for planes across or through the axis, planes whose loops wind around the tube, a cube over the ring's side (a lobe and trimmed loops), coaxial tori, and balls and rods on the axis; a small box inside the tube (its cut the ring around a box cavity) is exact too, and a rod across the tube reads 2.3230 against a numeric 2.32302. The rest fall back (row below). A sweep of 216 box placements (centres on a 6 x 3 x 3 grid, half-sizes 0.5 to 3; probe `zz_torus_boxes` in the session scratchpad) leaves 245 of its 648 ops exact and the rest safe fallbacks, none silently wrong (row below) |
| **Torus booleans against boxes: loops around the ring, and small tools through the wall** (the box sweep above) | Safe fallbacks. A box around the axis spanning the tube on both sides (a 6-cube centred on it) cuts lobes that join into loops winding once around the ring, not the tube, so the torus splits into bands whose rims are chains of lobe arcs: the band counterpart of `split_torus_by_tube_loops` (latitude seams become meridian seams). Small boxes through the tube's wall (half-size 0.5 to 2 at (0, 3, 0.5), say) fall back on every op; undug. A box over half the ring whose `y` wall passes the axis (`|x| < 3`, `y > -0.5`) folds its tube loops (each side wall's section turns back on the tube at `y = 0`), so `TubeLoop::new` rejects them and the op falls back (pinned by `box_wall_on_either_side_of_the_ring_axis`) |
| **Torus booleans off the axis: curved tools** (`make_torus(4, 1.5)` against `make_sphere(1)` at (5, 0, 0), and a second torus (4, 1) turned 90 degrees about x) | All three ops fall back on each (safe meshes, volumes right). Both go through the general marcher, whose curves FF drops (721 duplicates for the crossed tori) |
| **Non-planar sweep profiles (Beta); IGES, render (Experimental)** | Not yet audited |
| **Stable row defect: walking-engine chamfer at a closed rim or a shared vertex** | `chamfer_v2` on a cylinder's or cone's circular rim builds its cone face but returns a shell whose edges are not all shared by two faces (the endpoint-sampled trims cannot take a closed contact; the corrected fillet builder's periodic-contour machinery is the model). Two chamfered edges meeting at a box corner are refused (`TrimmingFailure`, pin `chamfer_v2_refuses_edges_meeting_at_a_vertex`; unrefused they left 9 edges open): each stripe's end detour lies in the other chamfer's removed region, so the two chamfer planes need a mitre along their intersection line (three at a corner need a corner patch). brepjs calls the planar `chamfer` in `chamfer.rs`, not this builder; the wasm `chamferV2` and `chamferDistanceAngle` bindings reach it |
| **Cone cuts parallel to the axis: the remaining fallbacks** (`cone_cut_parallel_to_its_axis` in `crates/operations/tests/cone_plane_cut.rs` lists the exact cells) | Safe fallbacks, none wrong. Both cones fall back when the piece the plane cuts off holds the seam (the pointed cone turned 17 degrees at x < 0.3 and beyond or turned 200 at x < -0.7 and below, the frustum at x < 1.2 and beyond at turns 0 and 17 or at x < -1 and below at turn 200): the section is anchored where it crosses the seam ruling, and the region the seam pinches still traces wrong. The pointed cone at turn 0 builds even so, since its hyperbola's vertex lands exactly on the seam. The loop, stripe-meshing and rim changes of the Closed entry are cone-only: on cylinders they broke a box less a quarter cylinder (the closed-rim sampling start) and six rod and knuckle fixtures, so a rod cut by a wall crossing one rim (tilted 10 or 25 degrees) still falls back where they would build it exactly. Next idea: trace a cone in its unrolled window [u_s, u_s + 2 pi] with non-periodic keys so a seam vertex's two copies stay distinct and the piece the seam crosses becomes two faces sharing the seam edge; a pointed cone needs a synthetic apex edge to close (the DCEL retry on every pointed cone changed nothing) |
| **A pointed cone through a plate's edge fuses short; notched walls take their volume from the mesh** (probes in the session scratchpad, `rvpr/`) | `make_cone(1.2, 0, 10)` at `(0, 5, -4)` fused with a 10 x 10 x 2 plate reads 208.22 against 213.93, its mesh open, on main as well: both notched-wall tests need two closed rims, and a pointed cone has one. Its wall's `face_area` reads right (34.173), but the wall's own mesh keeps the window: it covers the cone's whole side (37.968) at `x = 0` in every turn, and at `x = 0.4` turned 1 radian reads 33.222 against 32.396. A notched wall's solid measures off the whole-solid mesh (the rod at `x = 0`: 228.249 against 228.274), and routing it to the per-face flux reads worse, so that flux still assumes the `(u, v)` box the area no longer does |
| **A rod through a plate's edge with its seam through a window corner can fall back; some exact cuts mesh open** (the reviewer's battery, probe `zz_probe_1787` in the session scratchpad) | On main as well: turned `-acos(-x)` the rod at `x = -0.3`, `-0.1` and `0.4` falls back in every op, and at `x = -0.5` turned `acos(0.5)` the rod less the plate and the Fuse do, each mesh up to 2.1% off. Fourteen results are exact and within `1e-6` but mesh open, among them the rod at `x = 0.4` turned 2 less the plate and the Intersects with a plate turned 0.5 about `y`; main had each open, wrong or falling back |
| **The N-way fuse keeps a disjoint operand in the same shell** (`brepkit_algo::gfa::fuse_n` of the plate, a tool through it and a far unit box, pins in `rod_through_plate_edge.rs`) | The result measures right and meshes watertight, but the far box lands in one shell with the rest, and `validate_solid` reads the Euler characteristic as wrong (V - E + F = 5 against 3 for the rod, 6 against 4 for the ball), on main as well |
| **A hole within a degree of a cylinder's seam meshes open** (a rod turned 2 radians about its axis through a plate's edge at `x = 0.4`) | The fuse is exact and measures right, but its solid mesh is open and the rod wall's per-face mesh keeps the window, on main as well: the rim's samples sit about 0.068 radians off the surface projection in `(u, v)`, so the window pokes past the seam and the hole's flood clears the sliver beside it instead. Turned 1 radian with the rod at `x = 0`, the same fuse falls back to a mesh, and the rod at `x = 0` or `x = 0.7` falls back cut by, cutting or within the plate: safe, but the fallback meshes read up to 1.6% off (the rod less the plate at `x = 0`, 27.8115 against 28.2743) |
| **Sphere measure leftovers: the chordal equator** | A solid bounded by a ball zone between the chordal equator and a section circle measures off: `make_torus(4, 1.5)` less `make_sphere(3)` reads 166.4567 against 166.3658 (`torus_coaxial_tools.rs` bounds it at 1e-3), and `classify_point` on that Cut reads (2.8, 0, 0), on the equator plane, Inside where (2.8, 0, 0.3) reads Outside; the ball less its positive octant measures 98.95105 against 98.96017 with every face's area exact, and its solid mesh is open (the same for the ball fused with the box over that octant, in any turn about `z`). Both trace to the hemispheres meeting on chords (the quirk row below). Separately, a solid with no face that sends `solid_volume` down the direct path takes the whole-solid mesh: the ball's part inside a coaxial r=1 bore (two caps and a wall) reads 18.30905 against 18.31583, where the (u, v) box and the cylinder flux would be exact. A ball less a box holding the pole measures short: corner `(-0.7, -1.1, 0.1)` reads 85.702 against 85.754, corner `(-0.3, -0.4, 0.2)` 95.230 against 95.533. Its ring face, bounded by the chordal equator with a hole winding the axis, takes the per-face mesh, which meshes it badly (that Cut's mesh volume is 90.80 at deflection 0.01, 95.18 at 0.0005); pin `a_corner_holding_the_pole_cuts_its_patch` holds the Cut within 5e-3 (relative) of the ball less the Intersect. Next for holes: a hole around the axis sends its face to the mesh, though the smaller cap beyond it has a closed flux, R times its area `R² (2π − abs(∮ sin v du))` plus (C − about) dotted with the loop's vector area signed like `∮ sin v du`, all over three |
| **The ball less a column through both poles falls back to an invalid mesh** (`make_sphere(3, 32)` less `make_box(5, 5, 10)` at `(-2.5, -2.5, -5)`) | The mesh fallback is invalid and reads 8.318 against 8.902: `split_noseam_by_arrangement` returns only the collar inside the column's walls and drops the four lunes past them, which the Cut keeps. The Intersect and the column less the ball are exact |
| **The ball within or less a column narrower than itself falls back to a mesh** (`make_sphere(3, 32)` with `make_box(3.6, 3.6, 10)` at `(-1.8, -1.8, -5)`, whose corners lie inside the ball so each wall meets the sphere in two arcs) | Undug. The Intersect returns 352 planar faces and the Cut 460, on main as well; the engine's ray cast classifies a 13-step grid over the ball right on both meshes. The dome at each end is the case `sphere_arc_loops_bound_only_their_own_region` pins the ray cast to decline |
| **The ball less a column entering it from below with its end's corners outside the ball falls back** (`make_sphere(3, 32)` less `make_box(4.1, 4.1, 10)` at `(-2.05, -2.05, -1)` or `make_box(4.4, 4.4, 10)` at `(-2.2, -2.2, -1)`: the corners at `z = -1` lie past the ball's radius there, `8^0.5`) | Undug. 555 and 545 planar faces, on main as well; columns 0.2 to 3.8 wide are exact |
| **A hole closer to a plane face's outer arc than its chords' sag meshes open** (`make_sphere(3, 32)` bored by `make_cylinder(0.3, 10)` at `(2.69, 0, -5)`, within or less the box over `(-0.7, -1.1, 0.1)`) | Exact, valid and within 1e-3 of the truth, but at deflection 0.01 the solid mesh is open: the bore's circle in the box's floor passes 0.008 from the floor's arc of the ball, whose chords sag up to 0.01 and cut into it |
| **The engine's ray cast reads a point just past an off-axis bore's mouth as in the bore** (the rod `make_cylinder(0.3, 10)` at `(1, 0, -5)` through `make_sphere(3, 32)`) | `brepkit_algo::classifier::classify_point` on the bored ball puts `(1, 0, 2.9)`, outside the ball above the bore, Inside: the bore's cylinder face is tested by its whole axial band, from its rim's lowest point (2.704) to its highest (2.917), so a ray leaving the bore's axis at `z = 2.9` crosses the wall where the ball has already ended. The check crate's classifier reads it right. No boolean reached it in the bored-ball pins |
| **A sphere face whose hole winds the axis close to the pole meshes off the sphere** (probe `zz_ringmesh` in the session scratchpad) | The ball less a box whose wall passes near the pole keeps a ring face with that hole, right by topology and point classification, but its mesh cuts through the ball: with the corner at `(-0.0001, -1.1, 0.1)` the ring meshes to area 37.52 (37.81 exact) with a flux of 29.8 where a mesh on the sphere gives the radius times its area (112.6), and the Cut reads 66.34 against 92.56 (`solid_volume` meshes that face); at `(-0.05, -1.1, 0.1)` 88.13 against 92.07, at `(-0.2, -0.2, 0.1)` 95.99 against 96.85 |
| **A sphere face whose arcs chain into one loop around a closed section falls back** | `split_noseam_face_direct` pairs the open arcs into a cap and its remainder and cannot place a closed section in either, so a box whose top cuts a full circle inside the patch around the pole (the box over `(-0.7, -1.1, 0.1)` to `z = 2.95`) fails the face and falls back. Nesting the circle as a hole of the piece that holds it, with its disc as its own face, needs a point-in-region test on the sphere for a loop that winds the axis |
| **Sphere booleans that fall back clear of the equator** (probes in the session scratchpad) | Safe fallbacks. `make_sphere(3, 32)` within the box over y > 0, z > 1 (the y = 0 arc runs over the pole), and a rod through a trimmed patch: the box-corner piece (x > 1, y > 1.2, z > 0.8) less a vertical r = 0.2 rod at (1.8, 1.8), or the octant less one at (1, 1). A section through the pole is the same splitter gap as the half cuts in the chordal-equator TERMINAL entry |
| **A cone through a ball off its axis falls back** (probe `zz_conetime` in the session scratchpad) | Safe fallbacks. `make_sphere(3, 32)` less or within `make_cone(1.2, 0.4, 10)` laid along `x` at `z = 0.3`, tilted, or upright at `(1, 0.5, -5)` falls back to a mesh. Phase FF sections the cone wall and each hemisphere in 13 marched pieces, which split neither hemisphere, and the cone wall's splitter gives up on all 26. Each reaches its fallback in 0.3 s along `x`, 6.2 s tilted and 16 s upright (release) |
| **Balls against a half-space; the upright slab Cut** (audit probe `zz_pose_audit` in the session scratchpad) | A ball against the half-space `x > 0.5` falls back in every pose (the chordal-equator TERMINAL entry). The slab Cut upright is exact but invalid: its two pieces share one shell, the representation the cylinder and cone slab Cuts share |
| **Pose audit: other primitives** (probe `zz_pose_audit`, 5 primitives x 4 tools x Cut/Intersect x upright/turned/mirrored) | Exact results whose volume moves with the pose: a turned cylinder's box-corner Cut (2e-6). Fallbacks only when mirrored: a cylinder's box corner, a cone's rod Cut. Fallbacks in every pose: a pointed cone's box-corner and rod Intersects, a frustum's rod, a torus's rod. The slab Cut that leaves two pieces is exact but invalid for the cylinder, cone and frustum in every pose |
| **The cross one-row fillet's solid mesh is open** (fixture `crates/io/tests/cross_one_row_fillet_inmem.rs`) | The fillet result is closed by topology (every edge used twice), but `tessellate_solid` leaves about 1,700 mesh edges not shared by two triangles, so its volume depends on the anchor (65,090 about the origin, 29,224 about (80, 5, 10)); `solid_volume` reads 68,449 against the 64,968 oracle. The fixture's volume criterion compared the origin-anchored number and was dropped. Its r = 0.5 reversed sphere corner patches meshed their complements (about 3.03 mm² each against 1.02) until the reversed-cap fix |
| **Shelled cylinder and cone cups are invalid** | `shell(make_cylinder(5, 10), 1, [top])` and `shell(make_cone(5, 2, 10), 1, [top])` each leave 32 shared edges with one sense (the cup measures 344.45 against 333.01), and a hollowed `make_torus(6, 2)` fails with "solid assembly produced no faces". Their inner walls (closed-seam cylinder, cone, NURBS and torus) reach `assemble_solid_mixed`'s generic arm with the reversed vertex list and the flag; the sphere arm keeps the outer face's winding instead. Giving them the same un-reversal makes the senses consistent but not the cup (Euler V-E+F = -1 with V=128, E=134, F=5, volume 451.86), so a second fault sits in the closed-seam wall or its rim. Until they match, a shelled solid where a sphere wall meets one of them would cross the seam in one sense (no brepkit operation builds that seam today: a capsule's fuse falls back to a mesh) |
| **Heal's sphere recognition keeps an inward NURBS face's side** (`crates/heal/src/custom/convert_to_elementary.rs` near line 65; found by reading, not yet reproduced) | The NURBS face's surface is swapped for the recognized sphere with its flag and wire kept, without comparing the NURBS normal (`Su x Sv`) with the sphere's outward normal: a patch whose parameterization faces inward (a mirrored NURBS sphere patch, whose transform flips the flag and keeps the wire) would come out wound against its new surface. Next: pin it with a mirrored NURBS patch, and turn the flag and the wires over when the normals oppose |
| **Stable row quirk: `make_sphere(r, segments)`** | The hemispheres meet on a chordal equator (line edges), a sagitta off the sphere |
| **Point classification reads plane discs and cylinder walls through chords** (grid probe `zz_mirrod` in the session scratchpad) | Both `classify_point`s test a ray's hit on a plane or cylinder face against its boundary sampled into chords (32 per closed circle), so a hit within a sagitta of a circular rim is misread: on the exact mirrored ball less the slab `1 < z < 2` (#1775), a point under the slab reads Outside because two of its rays cross the slab's discs within 0.006 of their rims. The operations classifier also votes with two rays and needs both for Inside, so one miss reads Outside, and it reads a point as on the boundary by its distance to a sphere face's whole sphere, trimmed or not. A sphere face bounded by one wire of two rims joined by a seam (a band, as a file may store a ball between two planes) is misread by both: neither planar nor a seam alone, it takes the polygon path along a Newell normal the two rims mostly cancel; a cap bounded by its rim and a seam takes the same path, with the rim's chords. Dropping the seam's out-and-back run from the loop would leave the rims to the side tests |
| **The engine's ray cast misreads cone faces** (`make_cone(5, 2, 10)` within the box over `|x|, |y| < 3`, and less the box over `x > 1`) | `classify_ray_cast` misreads 2056 of 23248 and 206 of 23248 points of a grid over the cone (points within 0.02 of a surface skipped), on main as well: a cone face read as a `v` band with one `u` gap takes a trim that varies with height as constant, and a closed circle makes the whole face a full band. Undug |
| **The check crate integrates a plane face through its boundary's chords** (`crates/check/src/properties/face_integrator.rs` `integrate_planar_face`, sampled by `crates/check/src/util.rs` `wire_polygon`) | A plane face's area, flux and moments come from its outer and inner wires sampled into 32 chords per curved edge, so a curved boundary reads short by its sagitta segments: the keyhole cap in `crates/operations/tests/extrude_major_arcs.rs` (a 323-degree arc of radius 1.58) reads 28.2368 where the truth is 28.2004, and a 32-chord circle holds 0.64% less than its disc. `operations::measure::solid_volume` sends solids with a bored quadric face through this integrator. Green's theorem along each edge's own parameter would read a curved boundary exactly |

## Closed: root cause + where the detail lives

One line each; the fixture/PR carries the story. Newest first.

- **The engine's ray cast read a ball's hole cut by a column by its flat polygon (CLOSED 2026-09-26; pin `the_engine_reads_a_hole_by_its_planes` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the hole's four wall circles also run below the equator, where the
  hemisphere's outer loop already ends the face, so the exact arc check
  declined it; a loop's admitted arcs are now restricted to the part of
  each circle the face's other half-space loops admit.
- **Extruding an arc past half a turn built an inside-out wall (CLOSED 2026-09-26; pins in `crates/operations/tests/extrude_major_arcs.rs`)**:
  extrude oriented a circular arc's cylinder wall by the chord against
  the radius at the arc's start, which past half a turn points back, so a
  keyhole notch or a major segment extruded into a solid that read valid
  but measured wrong with an open mesh; it reads the side of the chord
  the arc lies on now, and the volume integrators walk each arc along its
  own span.
- **The engine's ray cast read a ball's collar by its flat polygon (CLOSED 2026-09-26; pins `the_engine_reads_a_collar_by_its_planes` in `crates/operations/tests/sphere_box_corner.rs` and `sphere_arc_loops_bound_only_their_own_region` in `crates/algo/src/classifier/ray_cast.rs`)**:
  `sphere_face_loops` sampled each circle's admitted part and wanted one
  run, and the collar of the ball within a square column has four arcs
  on the equator and wall arcs cut at their crest, so it fell to the flat
  polygon; the loop's arcs on each circle must now match the exact arcs
  admitted by the other planes.
- **A rod cut at a plate's edge lost its floor and top segments (CLOSED 2026-09-25; pins `a_rod_cut_at_a_plate_edge_keeps_its_segments` and `a_turned_rod_joins_an_n_way_fuse` in `crates/operations/tests/rod_through_plate_edge.rs`)**:
  turned so its seam lies off the plate, the rod's floor section ran
  from one crossing of the plate's edge to the other, sharing both ends
  with that edge's piece; the merge folded the two, and the rod less the
  plate at `x = -0.5` kept a flat chord where its wall's window was
  (30.597 against 30.188) and the Intersect read 1.195 against 1.228. The exact-arc
  path now splits such an arc at its midpoint. The same turn put the
  rod's vertex-only box off the plate, so the N-way fuse dropped the pair
  (with a far box, 201.000 against 227.718 for the rod at `x = 0.4`); curved edges
  now bound the solid box by their arcs, and spheres, tori and NURBS faces
  by their surfaces. Four more roots in the same poses: the face-pair
  filter's boxes sampled each arc at nine points and dropped the rulings at
  a wall's widest (the uncut rod at `x = -0.3` turned 2, 31.416 against
  29.456); a crossing kept one source edge, so a seam through a window
  corner hid the lens; a half-disc beside the seam was sampled level with
  its middle edge's sample, and read its offsets wrapped past the opposite
  meridian; and the notched-wall mesh test wanted closed rims. A battery of
  1,084 rod, cone, turned-plate and N-way cases now reads 870 exact and right
  against main's 499, 53 wrong against 131, and none worse.

- **A box holding the pole less a bored ball dropped the bore's post (CLOSED 2026-09-25; pin `a_bored_ball_keeps_its_bore_in_a_box_holding_the_pole` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the ball's face bounded by the chordal equator takes a shortcut split
  (`split_noseam_face_direct`) that dropped its holes, so the Cut kept the
  box less the plain ball (972.657217) and the Intersect fell back. Off the
  axis three more readers failed on the bore's marched rim: the engine's
  ray cast (the flat equator polygon), the band split of the bore's wall
  (circle rims only) and the patch's interior sample (inside the hole). The
  sphere mesher now joins a band's winding hole to its outer loop with other
  holes beside it (pins `a_band_keeps_its_other_holes`, a bore holding the
  pole at `(0.2, 0)`), and a ball's collar inside a column through both
  poles runs with the pole on its left (pin
  `a_column_through_both_poles_keeps_its_collars`: the ball within the
  column had read 179.57 against 104.20 on main).

- **A sphere patch around the pole measured as a zone (CLOSED 2026-09-25; pin `patch_around_the_pole` in `crates/operations/tests/sphere_face_area.rs`)**:
  a face whose outer loop winds the axis has no `(u, v)` area, and
  `face_area` fell back to the zone from the loop's mean latitude to the
  pole: 34.738 against 22.761 for the patch a box over `(-0.7, -1.1, 0.1)`
  keeps. It now reads `R² (2π − ∮ sin v du)` along the loop, the region
  on its left, for either pole.

- **A rod or cone frustum through a plate's edge fused short (CLOSED 2026-09-25; pins in `crates/operations/tests/rod_through_plate_edge.rs`)**:
  where the plate's window in the tool's wall straddles the wall's seam,
  the outer wire carries it as a notch. The solid mesher fanned rim
  samples to the notch's corners through the solid, `face_area` measured
  the rims' `(u, v)` box and the per-face mesher filled it: the fuse of a
  10 x 10 x 2 plate and a radius 1 rod at its edge passed every check but
  measured 219.41 against 228.27. The notched-wall rule (developed metric
  and angular refinement) now takes a notch of arcs and lines too, the
  area reads the loop's `∮ u w(v) dv`, the per-face mesher takes the
  hole-aware path, and the refinement accepts a triangle that sags no more
  than the coarsest shared boundary edge at one of its corners, which it
  had halved toward until the face fell back to a mesher that cracked its
  boundary.

- **A box holding the pole kept the wrong patch (CLOSED 2026-09-25; pin `a_corner_holding_the_pole_cuts_its_patch` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the upper hemisphere's section loop winds the axis, and the patch's
  interior sample, the centroid of its loop's `(u, v)` polygon (open
  across the seam), landed in the ring around it; both pieces classified
  outside the box, and the Cut kept the patch: a 4-face invalid solid of
  18.18 against 85.75, accepted by the gates. The sample now aims between
  the loop's nearest approach to the pole on its left and the pole, both
  loops sampled on their edges' curves. Placed so its walls miss the ball's
  vertical extent (corner `(-2.5, -2.5, 0.1)`), the box cuts lens faces
  whose section arc and bottom line share both ends, which the edge merge
  folded until a section circle split between two crossings of one line
  took a midpoint as it did between two of one arc. On a sweep of 432
  upright corners the wrong results fell from 72 to 4, all near-pole Cuts
  that measure short.

- **Turned or mirrored balls fell back (CLOSED 2026-09-25; pin `turned_and_mirrored_balls_stay_exact` in `crates/operations/tests/sphere_box_corner.rs`)**:
  a turned hemisphere's equator projects to a rounding sliver of `v`, which
  as an extent clipped every section inside the face; `face_v_range` now
  treats a sliver on the surface as empty and extends a loop winding the
  axis to the pole on its left. Unmasked, a slab's two latitudes nested on
  one hemisphere (the internal-loops splitter now nests sphere loops), the
  splitter kept a hemisphere whole where it could not trace it (it now fails
  the face, and the collar needs three arc chains), and the check crate's
  Gauss integrator integrated a zero band for a turned equator. A rod's
  tunnel left a shell whose hemispheres' boxes were slivers on the equator,
  so the builder's orientation vote summed only the tunnel wall's flux and
  its sign followed the pose (a winding sphere face's box now reaches its
  pole). The corner, the rod and the slab are exact turned and mirrored.

- **Point classification misread sphere faces (CLOSED 2026-09-25; pins in `crates/operations/tests/sphere_classify.rs`)**:
  both `classify_point`s tested a ray's hit on a sphere face against its
  boundary polygon projected onto the nearest axis plane, over which a
  tilted face is no graph, and through the chords' sagitta: on a grid of
  2,117 points a plain `make_sphere(3, 32)` read 32 (check) and 132
  (operations) wrong upright, 452 and 489 turned. A loop in one plane now
  bounds the face by its side of the plane, a planar hole by the far side
  of its own, and any other loop is projected along its own normal.

- **A turned octant below the equator fell back (CLOSED 2026-09-25; pin `turned_octants_are_exact` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the box's top face meets the ball on the faceted equator, and phase FF
  emitted that arc once, for whichever hemisphere it paired first, skipping
  it as a duplicate for the other; the lower hemisphere kept only its two
  meridian arcs, which do not close, and stayed whole (94 faces, 13.937
  against 14.137). The arc now also sections the other hemisphere
  (`GfaArena::curve_extra_faces`).

- **A ball's octant through a second boolean (CLOSED 2026-09-25; pin `box_octant_feeds_a_second_boolean` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the octant the boolean engine builds (box and ball turned about `z`) came
  back invalid, its patch's chained cap kept as the arcs came
  (`split_noseam_face_direct` now runs each arc the cap shares with the
  remainder the other way), and cutting a box corner from any octant fell
  back or returned it unchanged. Four roots: the seam-plane crossings fitted
  a plane through an arc-bounded patch's corners (now chord-bounded faces
  only), `Circle3D::intersect_circle` gave nothing for skew planes, a plane
  face's arc band grew with its longest side's sagitta (now per chord), and
  the ray-cast classifier stood a sphere face in by the flat polygon through
  its boundary (now the ray/sphere roots in each loop's half-spaces). The
  collar mesh path in `solid_volume` now also requires the outer wire to
  wind the axis. A rod through the patch fell back: a sphere face's point
  test took a meridian's UV chord to its pole's arbitrary `u` as boundary
  (its arcs are tested in 3D only now).

- **Point classifiers and the orientation check read a reversed face's side (CLOSED 2026-09-25; pins `reversed_caps_classify_by_their_wire` and `reversed_faces_raise_no_orientation_warning` in `crates/operations/tests/sphere_reversed_cap.rs`)**:
  `classify_point` in operations and in check negated a reversed sphere
  face's loop normal, so around a dimple (0, 0, 4) and (1.9, 0, 4.9) read
  Inside and (0, 0, 3) Outside; the face-orientation check compared a
  reversed face's wire with the negated normal (a false warning on every
  pocket wall and dimple) and took a full band's area-less rim loop as a
  winding. Both read the wire about the surface's normal now, and the check
  skips a loop with no projected area.

- **STEP export dropped a solid's cavities (CLOSED 2026-09-25; pins in `crates/io/tests/step_voids.rs`)**:
  the writer wrote only the outer shell and the reader built every solid
  without inner shells, so the box `[-5, 5]³` less a ball of radius 2 at its
  centre read back 1000.0000 against 966.4897. A solid with cavities is now
  a `BREP_WITH_VOIDS` whose voids are `ORIENTED_CLOSED_SHELL`s flagged false,
  read back with each void face turned over.

- **A hyperbola-trimmed cone wall measured off, and drifted with the pose (CLOSED 2026-09-25; pins `frustum_half_space_in_any_pose` and `cone_cut_parallel_to_its_axis` in `crates/operations/tests/cone_plane_cut.rs`)**:
  a frustum cut by the box over x > 0.5 measured 1.8e-4 off the segment
  integral, and differently again when turned or mirrored (the pose audit's
  frustum and cone drifts, 2e-6 to 2e-5): a cylinder or cone wall trimmed by
  a free-form curve left the solid on the whole-solid mesh. `solid_volume`
  now takes the direct path for it, integrating a NURBS boundary's flux per
  knot span. Phase FF builds a cone's parabola or hyperbola section as the
  exact rational quadratic arc (`plane_cone_conic_arc`, declined unless it
  meets the cone between its ends) instead of a cubic through samples.

- **Reversed sphere caps meshed their complement (CLOSED 2026-09-25; pins in `crates/operations/tests/sphere_reversed_cap.rs`)**:
  a box less a ball poking through its top leaves a reversed sphere face
  bounded by one circle; its solid mesh read 937.98 against the exact
  989.397125, because `close_loop_at_pole` took a reversed face's wire as
  running about the face's normal, while the boolean builders flip the flag
  and keep the wire about the surface's normal. `transform_solid`'s sphere
  v range made the same assumption, so the dimpled box scaled 1.5 along x
  measured 1406.77 against 1484.10. Both read the wire's turn alone now.
  `shell_op` built a hollowed ball's inner wall wound against the surface
  (the rim's hole wound with its outer loop to match), so the bowl's wall
  meshed as a dome; it keeps the outer face's winding now. STEP bounds run
  about the face's normal by ISO 10303-42: the writer flags a reversed
  face's bounds false and marks its header, and the reader turns a loop whose
  bound and face flags differ, except in an unmarked brepkit export (pins in
  `crates/io/tests/step_reversed_face_bounds.rs`).

- **A ball less a box corner, and the box-sphere octant (CLOSED 2026-09-25; pins in `crates/operations/tests/sphere_box_corner.rs`)**:
  the ball less a box whose corner pokes into it came back exact but
  invalid, its pocket's hole wound with the hemisphere's outer wire
  (`split_noseam_face_direct` kept the chained loop as it came; the loop
  is now turned counter-clockwise by its vector area, sampled along each
  edge's traversal). The octant shortcut (`build_box_sphere_octant`) built
  a corner below the equator as its arcs' complements (it assumed a
  right-handed corner) and wound every wire clockwise, so a second boolean
  read its patch backwards: a box corner cut from the octant was ignored
  and a rod through it measured 12.824527 against 12.816012. It also built
  an octant for any three cutting planes, corner in the ball or not (a ball
  at (1.8, 1.8, 1.8) came out 82.09 against 78.84); it now steps aside. A
  cap's (u, v)-box flux picked its pole from the boundary centroid, which
  sits on the axis; it reads the wire's turn in u now, the wire running
  about the surface's outward normal on a reversed face too.

- **Per-face meshes of a trimmed torus face (CLOSED 2026-09-25; pins in `crates/operations/tests/torus_face_mesh.rs`)**:
  per-face `tessellate` gridded a torus face's (u, v) box, so a notched
  ring meshed whole, a half ring meshed to nothing and a patch inside a
  thin rod meshed as the whole ring (28 of the 48 torus faces that
  `make_torus(4, 1.5)` kept against 15 tools were more than 1% off their
  exact area). The glTF, OBJ and PLY writers and the wasm UV mesh read
  these meshes. Every torus face other than the whole ring now takes the
  local mesher with the solid mesher's cascade (notch, two-rim, latitude
  band, CDT), falling back to the grid.

- **Sphere faces trimmed by a box or a rod: area, mesh and volume (CLOSED 2026-09-25; pins in `crates/operations/tests/sphere_face_area.rs`)**:
  `face_area` read a sphere face as a zone from its boundary's mean
  latitude to a pole (a box corner's 3.0867 patch read 84.70, an octant
  a hemisphere), per-face `tessellate` gridded the loop's (u, v) box, and
  the flux behind the direct volume path did the same, so a tilted rod's
  piece of a ball measured 10.996346 against 11.000032. The area is now
  Green's theorem in (u, v) with both poles free (`sphere_face_uv_area`,
  the face's mesh choosing the patch or the sphere past it, or measuring a
  face within 5% of half the sphere itself), and so is a hole's
  (`sphere_hole_area`: a pocket through a pole read 15 pi against 16.5 pi).
  A trimmed or pole-touching face meshes through the constrained CDT. The
  direct volume path takes the patch flux (`sphere_patch_flux`) for an
  outer loop without chords that winds none of u, the (u, v) box for a
  wire along the box's sides (a notch's outline declines it, pinned by
  `sphere_box_declines_a_notched_hemisphere`), and the face's mesh for the
  rest, a hole around the axis included. An arc over a pole between its
  vertices is split there (`half_cap_whose_arc_runs_over_the_pole`).

- **A cone cut by a plane parallel to its axis (CLOSED 2026-09-25; pins `cone_halved_through_its_axis` and `cone_cut_parallel_to_its_axis` in `crates/operations/tests/cone_plane_cut.rs`)**:
  over 168 cases (both cones, four turns, seven offsets, three ops) main
  built 6 exactly, fell back on 119 and returned 43 wrong solids; now 96,
  72 and 0. Through the apex the section is the two rulings with
  n·g(u) = 0 (`plane_cone_apex_rulings`), the wall splits into sectors
  between them (a pointed cone's close at the apex), and the exact volume
  takes a side face through the axis of cone walls
  (`ruled_wall_is_rectangle`). Off the axis the sampled hyperbola stopped at
  eight vertex radii, short of the rims an open one crosses (FF now asks
  for the faces' reach); it splits at its vertex so no arc shares both ends
  with a cap's chord; a loop along the boundary and back along a section is
  a region, not a hole; a pointed cone's seam copies lie a period apart
  through the apex, every piece samples its interior from its 3D curves
  (unwrapping folds rim pieces over a half turn and the apex jump), and a
  pointed cone's broken trace retries with the DCEL; an untouched rim is rejoined whole
  for its cap, and a rim a section touches at the seam's antipode is
  quartered so its halves never share both ends; a section crossing the seam is anchored there; and the
  stripe mesher takes conic-trimmed walls, which the rim ladder fanned into
  chords (a third of the volume lost). A face its sections cross but which
  splits into nothing now fails the build, and a result edge off the face
  it bounds fails validation.

- **A box that cuts a torus into two bands around the tube (CLOSED 2026-09-25; pin `box_over_half_the_ring` in `crates/operations/tests/torus_plane_cut.rs`)**:
  a 6-cube over |x| < 3, y > 0 against `make_torus(4, 1.5)` cut to 100.2
  (open mesh, truth 127.49) and fused open, and seven placements of the
  box sweep did likewise, all silently: the notch tracer emitted only the
  band the long way round the ring, and its mesher swept that way, so when
  the box covered the long way the kept band was never built and Cut and
  Fuse accepted the open shell. The tube-loop sector splitter now takes
  loops stitched from several walls' arcs (a lobe and a tube cross-section
  here) and seams each pair of loops between their vertices nearest in tube
  angle, a latitude arc when they share one, else a curve straight in
  (u, v); the notch tracer is gone. The two-rim band mesher takes rims
  that are chains of arcs, orients its stitches in (u, v) (a sliver along
  a lobe near its pinch is too thin for its 3D normal), and lays a row
  column level with every rim vertex, so a row turns with a rim that runs
  nearly along a latitude instead of cutting the corner and folding.

- **A cube over a torus's side (CLOSED 2026-09-24; pin `cube_over_the_rings_side` in `crates/operations/tests/torus_plane_cut.rs`)**:
  a 4-cube over x in [3, 7] against `make_torus(4, 1.5)` fell back on
  every op. Its x = 3 wall cuts a lobe that does not wind and its y = ±2
  walls cut loops around the tube that its edges trim, so the torus keeps
  (or loses) one disc bounded by eight arcs. The internal-loops splitter
  took the ring's seam placeholders for a boundary at v = 0, where the
  loops now start, so it never ran; a whole ring has no boundary, and its
  disc and holed remainder now take interior points from (u, v) (a disc's
  loop centroid is off the surface). Two walks of a hole in (u, v), the
  whole-ring classification guard's and the whole-ring mesher's, sampled
  NURBS edges by their knot span, which runs from the end vertex back on
  these arcs, and read the hole as winding. A ring around a cavity (a small
  cube cut from inside the tube, pin `cube_inside_the_tube`) read its volume
  off the mesh (177.4342 against 177.43688); a whole ring's face now sends
  `solid_volume` per face, where its flux is exact.

- **A plane whose loops wind around a torus's tube (CLOSED 2026-09-24; pins `slab_over_one_side_of_the_ring`, `bar_through_the_ring`, `slab_tilted_off_the_axis` in `crates/operations/tests/torus_plane_cut.rs`)**:
  a slab over x > 1 and a 2 x 20 x 4 bar through the ring of
  `make_torus(4, 1.5)` fell back to meshes on every op, the fuses having
  first dropped the ring. A plane that crosses every tube cross-section
  twice meets the tube in two loops, each `u = phi ± acos(rhs(v))`; phase
  FF now samples them exactly from v = 0, the whole-torus sector splitter
  takes any closed section that winds once around the tube from one
  latitude (not only tube cross-sections), a plane face carves a closed
  NURBS loop strictly inside it as a cap, the torus area and flux follow a
  free-form boundary by Green's theorem (so `solid_volume` takes such a
  face per face, not off the mesh), and the two-rim band mesher sweeps
  between rims whose u wanders with v. The volumes hold to the loops' fit
  (7e-8 on the bar's common part, whose four loops are cubic fits through
  exact points 2 pi / 128 apart).

- **A ball or a rod on a torus's axis (CLOSED 2026-09-24; pins in `crates/operations/tests/torus_coaxial_tools.rs`)**:
  `make_torus(4, 1.5)` fused with `make_sphere(3)` at its centre spent
  435 s in the general surface marcher and fell back to a 1294-face mesh;
  a rod of radius 4.2 through the ring's hole fell back on every op, and
  one of radius 2 standing clear in the hole fused to the rod alone
  (125.66). A
  surface of revolution sharing the torus's axis meets it in circles about
  that axis, where the two cross-sections in a half-plane through the axis
  cross, so phase FF now intersects those (`meridian_crossings`, which the
  coaxial-tori arm uses too) and emits the circles; all three ops build
  exactly. The ring's two zero-length seam placeholders no longer merge
  into one common block, the out-and-back spur pass leaves a whole ring
  alone, and the check integrator unwraps a torus boundary in v as well as
  u (the ball's Cut read 36.02 against 166.37). A whole ring that no
  section cut is sampled around its tube and sent to the fallback when the
  samples disagree, so a section FF missed can no longer keep or drop the
  whole ring: a 2x20x4 box through the ring and a slab over x > 1 had
  fused to the box alone (160 and 4000) and now fall back with the ring in.

- **A torus cut by a plane across or through its axis (CLOSED 2026-09-24; pins in `crates/operations/tests/torus_plane_cut.rs`)**:
  `make_torus(4, 1.5)` less the half-space above z = 0 or z = 0.5 came
  back as the whole torus (its volume read 177.65, the full ring), below
  z = -1 it fell back to a 282-face mesh, and halving it through its axis
  fell back too. The plane's sections were fitted NURBS loops, and the
  internal-loops splitter took each for a hole with its centre as the
  sample. A plane across or through the axis now meets the torus in exact
  circles; a new splitter cuts the whole torus into bands around the tube
  (level circles) or sectors around the ring (tube cross-sections), seamed
  along the reference meridian or equator where phase FF now starts those
  circles; a point in a plane face's hole no longer reads on the face; and
  the torus area and flux take a seamed band's side from its seam arc, not
  from its rims' short way round. A tipped tool's box enclosed the ring and
  its one vertex sat inside the tool, so the cut read "fully contained"; a
  whole ring is now probed toward the tool's flat faces, like a ball. That
  probe also stopped two overlapping coaxial tori (R 3 r 0.5, R 4 r 0.7)
  from intersecting to the whole first torus (14.80); their marched section
  then fitted thousands of points in one dense solve and never finished, so
  coaxial tori now meet in exact circles (their tube cross-sections cross in
  a half-plane), and the lens between them builds exactly (1.8879). Every
  torus band's or sector's seam is split at its middle, so the lens's two
  bands, seamed between the same two vertices, share no edge ends.

- **A ball cut by a plane clear of its equator (CLOSED 2026-09-24; pin `crates/operations/tests/sphere_plane_cut.rs`)**:
  `make_sphere(3, 32)` less the half-space above z = 0.5 fell back to a
  415-face mesh reading 69.65 against 70.55, keeping the cap above it
  fell back too, and a tilted plane's cut declared the ball "fully
  contained" in the tool. Four roots: the internal-loops splitter took a
  sphere loop's interior at its centroid, inside the ball (it now takes
  the sphere point straight out through it); the doubled-face pass dropped
  a cap and its disc, which share their one circle edge (two faces on
  different surfaces that no other face touches now stay); the containment
  witness probed only edges, and a ball's run only round its equator (a
  ball is now probed toward each flat face of the tool and along the axes);
  and the collar mesher wanted the level ring inside (a hemisphere less a
  tilted cap has it outside). Measure: a ball less disc caps holds
  (r A_sphere + sum d A_disc) / 3 about its centre, and a sphere face
  bounded by one circle is the cap on its boundary's winding side.

- **A tube cut by an oblique plane (CLOSED 2026-09-24; pin `tube_cut_by_an_oblique_plane` in `crates/operations/tests/oblique_rod_cut.rs`)**:
  a tube (radius 3 bored to 1.5) less the half-space above a tilted plane
  fell back to a 62-face mesh reading 63.38 against 20.25 pi = 63.62 at
  every tilt tried, where the level cut was exact. The plane face carries
  two nested closed ellipses, and a plane face with more than one closed
  section reaches the wire builder, which ignores closed curves; the salvage
  pass that peels interior closed circles off first and carves them as
  nested loops skipped ellipses (it now takes them, clearing the outline by
  the semi-major axis).

- **A rod halved through its seam (CLOSED 2026-09-24; pin `crates/operations/tests/rod_halved_at_every_angle.rs`)**:
  `make_cylinder(3, 4)` less the half-space y < 0 (a plane through the
  axis and the seam) came out exact-looking at 87.39 against 18 pi, and the
  other half (y > 0) fell back to a mesh. The cap's split pieces took the
  shorter arc between their ends, a coin toss at half a turn, and the
  walker read a reversed arc's tangent from the wrong end of its pcurve and
  fell back to its chord, so an arc and the chord across it tied; and a rim
  split at one point kept one half-turn arc whole, which the edge merge
  welded to the chord. Split pieces of a boundary arc now run its own
  sense, the walker reads a pcurve from whichever end sits on the edge's
  start, and a rim split once is paved at both halves' middles (in
  `make_blocks` and the planar splitter alike).

- **A cone cut by a plane across its wall (CLOSED 2026-09-24; pin `crates/operations/tests/cone_plane_cut.rs`)**:
  `make_cone(3, 0, 6)` less the half-space above z = 3 fell back to a
  37-face mesh reading 49.21 against 49.48, and keeping the tip fell back
  to 36 faces; tilted at slope 0.2, the cut built an invalid 3-face solid
  with an open mesh reading 63.72, more than the whole cone. The band
  splitter wanted two rim circles, and a pointed cone's wall has one rim
  and a seam up to its apex, so the section circle was taken for a hole
  (the apex now ends the band stack, and the band against it closes on the
  seam alone), and the result gate wanted 3 faces unless a sphere or torus
  face closed on itself (a cone face with its apex on its boundary does
  too). Measured, the wall's area assumed a (u, v) rectangle (the tilted
  tip's read 10.12 against 17.41; a non-rectangular cone wall now takes
  cos(a) times its v-weighted (u, v) area by Green's theorem) and an
  ellipse cap left the cone to tessellation (the cone between the apex and
  a cap is a third of the section's area times the apex's height over it),
  and a plane face bounded by an ellipse read its sampled boundary (an
  elliptic arc's segment is a b / 2 (Δ − sin Δ) over its parametric sweep).

- **A rod cut by an oblique plane (CLOSED 2026-09-24; pin `crates/operations/tests/oblique_rod_cut.rs`)**:
  `make_cylinder(3, 6)` less the half-space above a tilted plane fell back
  to a mesh unless the tilt faced the rod's seam, and where it built,
  `solid_volume` read 74.43 against 27 pi at slope 0.3 and the wall's
  `face_area` 73.51 against 18 pi. The plane's ellipse started at its
  frame's origin, off the wall's seam, so the band never split (it now
  starts where the seam line crosses the plane, and a closed ellipse's
  domain runs from its vertex); at slope 0.8 and up, the plane's lines
  against the rod's caps survived the box-based filter through the caps'
  box corners (a line missing a disc or ellipse face is now dropped); the
  band mesher declined ellipse rims, and the CDT chorded through the solid
  (it takes them now); the pi r^2 h formula skipped caps tilted past 8
  degrees (any cap bounded only by conics crosses the whole wall); and the
  wall's area assumed a (u, v) rectangle (a non-rectangular wall now takes
  r times its (u, v) area by Green's theorem).

- **Point classification ignored cavities (CLOSED 2026-09-24; pins in `crates/operations/tests/classify_cavities.rs`)**:
  the ray-cast `classify_point` of both `brepkit_check::classify` and
  `brepkit_operations::classify` (and their winding variants' boundary
  test) walked only the outer shell, so every point in a closed cavity
  read Inside. They cast against every shell now. The check crate's
  winding number still fan-triangulates wire polygons, which does not cover
  a curved face, so its winding and robust variants stay unreliable there.

- **Rods cut along their axis measured by mesh (CLOSED 2026-09-24; pins `crates/operations/tests/flat_sided_rods.rs`)**:
  a half rod read `solid_volume` 18.84619 against 6 pi at any deflection
  from 0.1 to 0.001, and its half-disc caps `face_area` 6.283146 against
  2 pi. The revolution path took only caps perpendicular to the axis, so a
  face parallel to it (a half rod's cut, a D-shaft's flat) sent the solid
  to the mesh, and planar areas came from the sampled boundary. Faces
  parallel to the axis of cylinder walls now integrate exactly when every
  wall is a rectangle in (u, v) with rim arcs of at most a half turn (the
  angular-range reader takes an arc's shorter side; a rod fused with a bar
  keeps the mesh), and a planar face bounded by lines and circles takes its
  area by Green's theorem.

- **A pointed cone's open mesh (CLOSED 2026-09-24; pin `pointed_cone_tessellation_is_watertight_at_every_deflection` in `crates/operations/tests/tessellate_watertight.rs`)**:
  `make_cone(3, 0, 6)` meshed with 46, 78 and 142 open edges at 0.03, 0.01
  and 0.003. The rim circle's frame comes from its +z normal (samples from
  +y) and the cone's from its apex-to-base axis, -z (grid from -y, running
  the other way), so the snap mesher's grid met the rim's shared samples only
  when the rim took an even number of segments; an apex-down cone, both
  frames from +z, was always closed. The band mesher now fans a pointed
  cone's shared rim samples to its apex (one closed rim, one seam line up to
  the apex): exact to the rim's chords, with the same triangle count.

- **Chamfers of round rims (CLOSED 2026-09-24; pins in `crates/operations/tests/chamfer_round_rims.rs`)**:
  the planar `chamfer` (the one brepjs calls) rebuilt every face as a
  polygon and threw `cannot normalize zero vector` on a closed circle's
  single vertex, so a rod's end, a tube's mouth or a hole's rim kept its
  sharp edge. A closed circle between a flat cap and a coaxial cylinder or
  cone wall now chamfers exactly: the cap's circle moves `d` into the cap,
  the wall's rim moves `d` along its rulings, a cone band joins them and the
  seam follows. Convex rims remove a ring and concave ones (a blind hole's
  floor) fill one; a distance that would reach another boundary of the cap
  or the wall's far rim is refused. Scope: walls that are a plain band
  between two rims (or a rim and an apex), caps bounded by lines and
  circles; other rims keep the polygon path.

- **`validate_solid` rejected every solid with a cavity (CLOSED 2026-09-24; pin `solids_with_cavities_validate` in `crates/operations/src/validate/tests.rs`)**:
  its Euler check read V - E + F against 2 + L as if a solid had one
  shell, so a block with a closed cavity (V - E + F = 4) failed as invalid.
  It now expects 2(S - g) + L with S counting the cavities, and checks face
  connectivity within each shell rather than across the solid.

- **STEP `EDGE_CURVE` `same_sense` (CLOSED 2026-09-24; pin `crates/io/tests/step_edge_same_sense.rs`)**:
  the reader dropped the flag, so a `.F.` arc (its circle's axis flipped,
  as another writer may emit it) ran start to end the wrong way round and
  its faces traced the complement. `build_edge_curve` now reverses such a
  curve with `EdgeCurve::reversed` (circle and ellipse: normal and `v_axis`
  negated; NURBS: net, weights and mirrored knots), which also replaces
  extrude's private copy.

- **Per-face NURBS meshes ignored the trim (CLOSED 2026-09-24; pins in `crates/operations/tests/bspline_conversion_mesh.rs`)**:
  `tessellate::tessellate` meshed a NURBS face over its whole surface, so
  `solid_volume`'s direct path, `face_area` and the glTF, OBJ and PLY
  writers read a trimmed face as its whole patch (the converted napkin
  ring: `solid_volume` 1613 against 587.7). A NURBS face whose boundary
  leaves its surface's domain edges now meshes through the trimmed CDT,
  587.60. The analytic sphere `face_area` never subtracted a face's holes
  (the unconverted ring read 648.28 against 587.67); a hole takes
  `R² |∮ sin v du|`, or the cap beyond it when it winds round the pole. The
  L-lip fuse's `solid_volume` pin moved with the meshes, 28993 to 29034.

- **Drills into a ball and a ring, a pocket into a pointed cone (CLOSED 2026-09-24; pins in `crates/operations/tests/ball_and_ring_drills.rs` and `cone_pocket.rs`)**:
  `make_sphere(2, 16)` less an r=0.2 drill from (0.5, 0, 1) up and
  `make_torus(5, 1, 16)` less an r=0.3 drill through its tube fell back to
  meshes. The pairs went to the grid-seeded marcher (287 s in a debug build
  for the ball); `algebraic_sphere_cylinder` now sweeps the drill's rulings
  off the axis, as a parallel-axis torus-cylinder pair does, sharing the
  cylinder-cylinder sweep's helpers. Then: same-domain grouping took the
  untouched south hemisphere for a duplicate of the drilled north one (a
  closed surface's shared boundary bounds two regions, so faces oriented
  alike must traverse it the same way to coincide); the ring's whole-surface
  face, bounded by its two zero-length seams at one vertex, was dropped as a
  sliver, welded seam into seam by the edge merge and rebuilt from fresh
  edges (it keeps the parent's seams, and the merge leaves one face's
  zero-length lines alone); and a two-face solid failed the three-face
  minimum. For measurement, the CDT mesher closes every sphere cap through
  its pole along a sampled meridian, the loop started clear of any hole (the
  snap mesher's grid ignored holes, and a cap beside a conforming one cannot
  keep it), and gives a holed ring a periodic rectangle; the sphere goldens'
  counts moved. `solid_volume` subtracts a sphere's or torus's
  contractible holes by boundary integrals and counts the ring's face as
  the whole ring, and the classifier tests a full-surface face's holes.
  The cone pocket (`make_cone(3, 0, 6)` less the box x -0.5..0.5, y 1..5,
  z 1..2) lost its cone face: its seam runs up to the apex and straight
  back, which BuilderSolid's spur excision and the boolean's
  `remove_wire_spurs` both stripped. They keep it now; the CDT mesher
  splits a holed cone's seam at the apex into two sampled sides and an apex
  row and keeps the slant unscaled there; a fitted edge's mesh samples end
  exactly on its vertices (a marched end 5e-8 off split a corner in two); Step 2a
  recentres an unclosed loop by whole periods only (an arbitrary shift
  moved every sample off its point); and the classifier closes a pointed
  cone's boundary along the apex row.

- **A round bore into a rod's wall (CLOSED 2026-09-24; pins in `crates/operations/tests/rod_side_bore.rs`)**:
  `make_cylinder(1.5, 4)` less a perpendicular r=0.3 cylinder fell back to a
  mesh (through) or came back uncut (blind). `algebraic_cylinder_cylinder`
  swept the thicker cylinder's rulings, so each root traced an open arc
  forced closed; it now sweeps the cylinder whose rulings all meet the
  other. Each loop winds the bore once: the seam-anchor pre-pass starts it
  on the bore's seam, and the band builder takes any number of such chains,
  measured along each piece. Where the hole straddles the rod's seam, the
  pre-pass also cuts the loop at its seam crossings (and a two-piece loop
  at its midpoints, which the edge merge would weld into one), and
  `split_periodic_face_around_seam_holes` notches the wall's outer wire
  around the hole's two halves. The CDT mesher gives a notched wall the
  holed wall's developed metric, and `solid_volume` integrates a cylinder
  or cone face trimmed by a free-form curve along its boundary. Two loops
  of one face pair that nearly meet (equal crossing cylinders, the lens
  fuse) keep their old routing.

- **heal `convert_to_bspline` broke the solids it converted (CLOSED 2026-09-24; pins `crates/operations/tests/bspline_conversion_mesh.rs`, `closed_conics_start_at_their_vertex`)**:
  cones, spheres and tori became sampled degree-1 grids 5 to 7% off the
  surface, closed circles and ellipses started at their frame's angle
  rather than their vertex, and cylinder, cone and plane patches were sized
  from vertices alone (a disc's one vertex left its cap patch short of the
  disc). Faces now take the exact rational forms over their boundary's
  sampled range. Their meshes needed four NURBS CDT fixes: seam runs keep
  their projected v (spaced by index they assumed both seam vertices, which
  a rim usually keeps), a straight seam drops its interior samples (they
  fanned across a ruled wall with no interior rows), a band between two
  loops winding the period (the napkin ring's zones) is joined along a
  virtual seam, and the triangulation measures in surface speeds rather
  than knot values (a converted cylinder's u spans 1, its v 4).

- **`sweep_smooth`'s rails left their faces (CLOSED 2026-09-24; pin `sweep_smooth_rails_lie_on_their_faces` in `crates/operations/src/sweep/tests.rs`)**:
  each rail edge was the straight chord between its first and last ring
  vertices while each side face interpolated its own two columns of ring
  positions, so the faces' boundaries left their surfaces (a trimmed mesh
  of a quarter-circle sweep read 6.87 against 7.85). The rails are now one
  B-spline per profile vertex through its ring positions on shared
  parameters, and each side is the ruled surface between two of them, as
  in `loft_smooth`.

- **NURBS faces meshed far outside their deflection (CLOSED 2026-09-24; pin `squashed_walls_mesh_within_their_deflection` in `crates/operations/tests/non_uniform_scale_mesh.rs`)**:
  `interior_grid_resolution` fed a NURBS face's knot spans to a circle-chord
  formula with radius 1, so a squashed cylinder's wall meshed 6.2e-3 low in
  volume at deflection 0.001. The grid now follows the chords of the face's
  own iso-lines across its (u, v) box (one division along a ruling, the
  normals' turn weighed on every chord but one ending on a pole, at
  most 1024 per direction and 65,536 cells in all, with a warning when
  that binds): 8.9e-5 at 0.001. The diagnostic volume pin in
  `cross_one_row_fillet_inmem.rs` moved with the meshes.

- **The walking-engine chamfer left every chamfered edge open (CLOSED 2026-09-24; pins `chamfer_v2_closes_on_every_box_edge`, `chamfer_v2_closes_two_parallel_edges`, `chamfer_v2_concave_notch_adds_only_the_chamfer_sliver` in `crates/operations/tests/regress_chamfer_obtuse_ridge.rs`)**:
  `ChamferBuilder` trimmed the two chamfered faces but not the end faces,
  which still ran through the old corner (a 4 x 3 x 2 box's edge chamfer:
  6 edges used once, volume 23.667 against 23.5). The end faces now take the
  chamfer's cross edges in place of the corner detour. On a concave edge the
  chamfer plane's normal (spine tangent x contact span) pointed into the
  material, so the face is flagged reversed when it disagrees with its two
  neighbours' outward normals. A cross edge between a chamfer's contacts
  came out a semicircle whenever roundoff left the chord's cross product
  with its midpoint "centre" nonzero (the (0.3, 0.6) chamfer read 23.6499
  against 23.64); a chord through the centre is now a line.

- **Volumes far from the origin (CLOSED 2026-09-24; pins: `crates/operations/tests/far_from_origin_volume.rs`)**:
  every divergence sum ran about the world origin, so a solid a million
  units out multiplied its coordinates into each face's flux (a slanted
  prism, a windowed tube and a napkin ring all lost their volumes; the ring
  read 2599 against 587.67). Triple products and shoelace sums now take
  differences first, and each exact path (all-planar, revolution, direct,
  and the check crate's face integrator) sums about the origin for a solid
  near it and about its box centre past ten half-diagonals. Mesh-only
  paths stay about the origin: an open mesh's volume is comparable only
  about one fixed point. The near/far rule also keeps a faulty solid's
  reading where it was; the concave `chamfer_v2` face that surfaced this
  is its own row.

- **Draft to Stable (CLOSED 2026-09-24; pins in `crates/operations/src/draft.rs` tests and `draft_batch_cuts_the_exact_wedge` in `crates/wasm/src/bindings/operations.rs`)**:
  `draft` pushed a drafted face's vertices radially from an axis, bending the
  face and leaving its neighbours on the old vertices. It now turns each
  drafted plane about its neutral line until `n · pull = sin(angle)` and
  moves every vertex of a drafted face to its planes' meeting point; a
  vertex that meets a curved face, or would split, is refused.

- **`loft_smooth` did not close its shell (CLOSED 2026-09-23; pins `loft_smooth_waisted_squares_close_at_their_volume`, `loft_smooth_uneven_profiles_share_their_rails` in `crates/operations/src/loft/tests.rs`)**:
  each side face had its own straight rails while its surface curved through
  the middle profiles, and its normal pointed inward. The rails are now one
  B-spline per vertex index on shared mean chord-length parameters, and each
  side is the ruled surface between two of them.

- **`solid_volume` chorded the curved edges of planar faces (CLOSED 2026-09-23; pins: the `boolean_box_minus_cylinder` golden at 1000 − 90π, and the window and reversed-face volume checks at 1e-9)**:
  the direct per-face path meshed every planar face, so a cap bounded by a
  circle, ellipse or NURBS edge lost its chord segments (0.036 mm³ on a
  10 mm block less an r=3 bore). `planar_face_flux` integrates the face's
  exact area by Green's theorem (closed form on conics, Gauss per knot span
  on NURBS); a face whose holes might nest keeps the mesh.

- **The ellipsoid primitive and a squashed torus meshed to nothing (CLOSED 2026-09-23; pins `crates/operations/tests/non_uniform_scale_mesh.rs`)**:
  `makeEllipsoid` scales a unit sphere non-uniformly, so each hemisphere
  becomes an exact NURBS cap whose only wire is the equator, winding the
  periodic u once with no seam. The curved CDT could not close that region
  in (u, v) and returned no triangles. `close_loop_at_pole` continues such a
  loop into its first sample's image one winding on and back along the
  degenerate v edge on its left, welded to the pole. A torus's one face,
  bounded by its seam pair collapsed onto one vertex, encloses nothing; the
  solid mesher now falls back when the CDT emits no triangles, and a doubly
  periodic NURBS face meshes as a structured grid whose far seam rows copy
  the near ones (the adaptive quadtree left T-junction cracks).

- **Mirrors broke solids that mix NURBS and analytic faces (CLOSED 2026-09-23; pin `crates/operations/tests/mirror_mixed_faces.rs`)**:
  `transform_solid` and `copy_and_transform_solid` reversed every wire and
  also flipped each NURBS face's flag, so a NURBS face's boundary ran the
  wrong way against its planar neighbours (4 inconsistent edges on a box
  with a NURBS lid). A NURBS image's Su × Sv already turns inward, so it
  flips its flag only; faces with explicit normals reverse their wires.

- **Windows through cylinder and cone walls, and holes in reversed faces (CLOSED 2026-09-23; pins `crates/operations/tests/cylinder_wall_windows.rs`, `reversed_face_holes.rs`)**:
  a box cut through a tube wall (upright, tilted, through the u origin, blind)
  left the wall's hole wound the same way as the cutter walls, the mesh
  skinned over the hole and dropped the tunnel, and `solid_volume` counted the
  full band. The internal-loops splitter now winds curved-face loops by the
  plane rule, measured in unwrapped (u, v), and samples reversed loop edges
  along their own span; the curved CDT constrains and flood-removes inner
  wires (the per-face mesher routes through it too) and refines holed
  developable walls by angular extent; holed cylinder and cone faces integrate
  their flux by Green's theorem, each wire oriented by its own (u, v) area.
  A face flipped into a cavity keeps its wires under the reversed flag, so
  every stored outer wire runs CCW about the SURFACE normal; the splitter's
  `is_plane && !reversed` rule wound a hole cut into a reversed face (a
  cavity floor, a bore wall) with that wire. Discs are now CCW about the
  surface normal for every parent.

- **Assemblies to Stable (CLOSED 2026-09-23; pins in `crates/operations/src/assembly.rs` and `crates/wasm/src/bindings/assembly.rs`)** —
  `flatten` dropped a parent component's own solid, the bill of materials and
  its names followed `HashMap` order, and the assembly box transformed the
  corners of each instance's local box. Every component is now placed in tree
  order and the box bounds each instance in its placed frame
  (`measure::solid_bounding_box_transformed`).

- **Transform lost analytic frames (CLOSED 2026-09-23; pins in `crates/operations/src/transform/tests.rs`)** —
  `transform_solid` rebuilt a rotated torus or sphere around the world z axis
  (boundary 5 units off the surface, open sphere meshes), dropped every
  cylinder/cone reference direction, left a mirror's wires winding clockwise
  and its NURBS faces facing inward, and mapped a non-uniformly scaled circle
  to a non-principal ellipse. Surfaces are now exact images of their frames
  (NURBS where a map breaks circularity), mirrors reverse wires and flip NURBS
  face flags, and pcurves are dropped where the parameterization changed.
  Collateral fixes: the equal-count band mesh paired its rims one sample out
  of phase when a sample sat on the u seam (a twisted band, 1.7% low);
  NURBS point projection now wraps across a closed direction instead of
  clamping; the CDT mesher unwraps closed NURBS directions and starts each
  closed rim at its vertex.

- **Boundary arc crossing window and partially riding line sections (CLOSED 2026-09-15; pin `compound_cut_by_two_keyhole_pins_meeting_on_a_knuckle_face_stays_exact`)** —
  two roots in `fill_images_faces`: `arc_segment_crossings` windowed a boundary arc by its
  shorter-arc midpoint, so a major arc's complement admitted a phantom crossing (a bore
  cap's outline arc cut the pin hole's slanted edge mid-run and `link_existing` then
  replaced the hole edge with the short piece), and the classification polygon sampled
  major arcs the same way; both now walk the native span (unit pin
  `major_arc_keeps_only_crossings_on_its_own_side`). A straight section riding a
  collinear boundary edge for PART of its span (the bore's tail base along the pin hole's
  base edge) threaded the shared run as a second copy of that edge; only the uncovered
  runs are kept now (`line_section_uncovered_pieces`).
- **Knuckle pin bore: a coaxial bore cut through a rod fused to an on-axis bracket (CLOSED 2026-09-15; `hingeSwing` op3029 first tool, the bin-blob root of the swing sweep)** —
  three roots. The convex-analytic classifier accepted the non-convex
  knuckle because only its vertex CENTROID was tested against the
  half-spaces, then read the bore wall's sample (on the far side of the
  bracket plane) as outside and dropped the wall band; every boundary vertex
  now has to satisfy every constraint (`try_build_convex_analytic`). The
  planar interior sampler validated candidates against a corner-only chord
  polygon, so a C-shaped ring (the annulus remainder after #1666) rejected
  every candidate near its arcs and fell back to a hexagon interior point in
  the bore (`sample_face_interior` walks the arcs). The FF closed-circle split
  handed a coplanar sibling's arc to the first face pair and denied the
  sibling its own as a duplicate (`in_region` trims arcs to each planar
  face's trimmed loops), and the hole-free planar clipper dropped a section
  with no boundary crossing at all (a keyhole slot wall inside the disc), so
  the midpoint test now decides zero-crossing sections too. Pin
  `cut_a_pin_bore_and_slot_through_a_knuckle_flush_with_its_bracket` (#1667,
  its bore stage).
- **Partially overlapping coplanar end caps in a fuse (CLOSED 2026-09-15; the keyhole pin, likely the `op59` lid plate family)** —
  the same-domain detector sampled every boundary arc with the shorter-arc
  evaluator, so a rim split at two paves (the disc's 276 degree remainder
  beside a flush slot bar) polygonised as its short complement, the
  remainder paired with the bar's overlap rectangle and was dropped, and
  the real overlap piece went as an unpaired On face: both free. The
  detector now samples the NATIVE arc (`sample_edge_uniform_native`,
  counter-clockwise in the circle's own parameter from stored start to
  end), which is the codebase's edge convention; the old comment's fear of
  split faces storing arcs against that convention did not survive the
  suites. Pin `fuse_a_rod_with_a_flush_slot_bar_crossing_its_rim`.
- **Containment fallback returned an operand; a rim split at two opposite points lost its chord (CLOSED 2026-09-15)** —
  `detect_trivial_relation`'s AABB-only fallback called a tool contained
  when its AABB centre sat on the blank's boundary plane (the centre
  witness cannot refute that), so the intersect returned the whole tool
  and the fuse the blank; a boundary witness now probes the tool's
  vertices and edge midpoints with the robust classifier. Past the
  shortcut, a closed plane rim split at two opposite points kept a
  half-turn piece that shared both endpoints with the chord section, and
  the endpoint-keyed edge merge collapsed the two (the cap kept the removed
  half's rim in place of the chord); the plane splitter now splits such a
  piece at the seam's antipode, the vertex the periodic path already puts
  on the wall sharing the rim (`edge_splitting.rs`). Pin
  `intersect_a_pin_with_a_wide_diagonal_keep_box_keeps_half_the_pin`.
- **Coplanar sections from arc boundaries (CLOSED 2026-09-15; the `hingeSwing` second pin, op3029)** —
  the coplanar FF phase projected every boundary arc as its chord on both
  faces of a pair: a knuckle end's radius ending at the pin's centre was
  clipped where it crossed the pin cap's chord, the chord piece became a
  section of its own, a bore arc's chord crossed the pin's diameter the arc
  never reaches, and the wedge split into a sliver nothing paired with
  (three free edges, mesh fallback). The phase now keeps arcs as arcs:
  exact segment/circle and circle/circle crossings, arc-true region tests
  over every wire, real arc sections deduplicated against the regular FF's
  arcs, and coincident carriers matched by geometry, not endpoints
  (`phase_ff_coplanar.rs`). The pin cut replays exact (239 faces, no free
  edges) with the bore verified by axis and cross-section oracle scans. Pin
  `half_pin_standing_on_a_knuckle_end_touches_and_fuses_exactly`.
- **Periodic band whose rims pass through the surface seam (CLOSED 2026-09-15; the slot after the pin bore)** —
  the bore wall's face seam sits a quarter turn from the surface's u-seam,
  so its rims cross the seam at a plain vertex; once the slot's rulings pave
  those rims, the image expansion assigned principal-value u to the pieces
  past the crossing, the boundary loop folded, and the wire builder traced
  one loop with rim arcs used three times (mesh fallback). The expansion
  now unwraps every OPEN boundary piece by continuity along the wire
  (start, midpoint in the arc's native parameterisation, end), not only
  endpoints exactly on the seam; closed rims keep their conventional
  full-period pcurve, which is not anchored at the seam vertex's u
  (`resolve_seam_endpoint_uv`). Pin extended:
  `cut_a_pin_bore_and_slot_through_a_knuckle_flush_with_its_bracket`.
- **Hinge knuckle fuse: a barrel whose axis lies on a bracket's edge (CLOSED 2026-09-15; `hingeSwing.scenario` op22/27/32/37/42/47/65..85, mesh fallback to exact)** —
  the bracket's two faces through the axis cut the end caps radially, one of
  them through the rims' seam vertices, and the barrel wall keeps 270 degrees
  around its seam. Roots, all in the closed-rim machinery: a holed planar
  face dropped any section with fewer than two outer crossings, so the annulus
  never got its ring bridges (`clip_line_to_face_boundary` now classifies such
  sections against the outer wire and the holes together) and the hole
  promotion ignored a section ENDING on a hole vertex (its two-edge spur loop
  now routes the face to the arrangement); a closed rim's pave block spanned
  the circle from its own origin rather than the seam vertex, so its children
  walked backwards, and a rim split once left two arcs sharing both endpoints
  that the endpoint-keyed merge conflated (`make_blocks` anchors closed
  circle/ellipse blocks at the seam and paves the longer arc's middle; the
  planar splitter mirrors the split on rims it re-splits itself); the boundary
  winding tests sampled reversed arcs backwards through their pcurves
  (`cw_loops` and the hole `outer_sign` use the arc-true frame sampler); and
  the assembler's outward flux and the analytic cylinder volume both read a
  270 degree wall as a full turn (periodic samples unwrapped along the wire;
  `angular_range_from_wire_arcs` walks circle rims, splines keep the old
  heuristic because a blend band's rational arc carries its full circle as
  its domain). REFUTED on the way: paving FF Line sections at every boundary
  vertex of both faces (the reference kernel's approach) broke seven io
  fixtures and mis-measured op65 by 0.5 mm3 while looking watertight; the
  holed-face clip alone yields the same bridges. Pins
  `fuse_tube_with_a_bracket_edge_on_its_axis_splits_the_annulus_caps` and
  `fuse_rod_with_a_bracket_edge_on_its_axis_splits_the_end_discs`
  (`crates/operations/src/boolean/tests.rs`). The tool-side hang is still
  unmeasured (row above); `op59` (a plate with rounded corners, 12 free
  edges) is a separate open case.

- **brepjs-side roots of the label-plate icon and tracked-text families (CLOSED in brepjs #2314, 2026-09-14, 19.0.3)** —
  `transformCurve2dGeneral` keeps conics exact under similarity transforms
  (scale, rotation, reflection decomposed onto the exact 2D ops; only true
  affinities still refit) and `makeCompound` flattens nested compounds so
  `meshCompound` sees every solid. Tool-side verification waits for the
  tool's brepjs bump (its pin is 18.124.8; `labelPlateIcons.test.ts`,
  `textBuilder.scenario.test.ts`, `wallText`, `textElement`, `repeatLabels`).
- **Click-rail seating family: a probe column on the lip corner's mesh seam (CLOSED 2026-09-15 on the tool side, gridfinity-layout-tool #4284)** —
  `lidCutoutGrip.scenario.test.ts` read 1.91 mm of rail interference on
  brepkit against a 1.7 ceiling in all ten cases, the plain bin included.
  The worst columns are x = +-59.000 exactly, the lid's bounds less the corner
  radius, which is the bin lip's seam between the corner cone and the straight
  chamfer plane; both kernels' meshes have triangles at z = 40.55 and 42.25
  containing the column on their shared edge, but the tool's `columnCrossings`
  skips a barycentric weight of -1e-16 (the `1 - w0 - w1` rounding for one
  vertex order) and keeps an exact 0 (the other order), so brepkit's mesh lost
  the lip-underside crossing and the parity pairing read the whole wall as
  solid. A 0.5 mm column grid around that corner agrees between the kernels to
  0.01 mm. Fixed in the tool's metric (weights within rounding of zero count);
  the seat-interference matrix's pinned count of flush-fill scoop cases moves
  from four to three because the 1x2 footprint's "narrow flush fill" reading
  was the same edge miss on both kernels. brepkit geometry needs nothing.
  Probe: an untracked `__kernel-tests__/railColumnProbe.test.ts` (the sweep,
  the seam vertices and the containing triangles' weights, both kernels).
- **A rod tangent to the top of the post it passes through: the arch part (CLOSED 2026-09-15; pin `fuse_rod_tangent_to_the_top_of_a_post_keeps_the_rod` in `crates/operations/src/boolean/tests.rs`)** —
  two roots. (1) The rod's piece inside the post is a full-turn band whose
  interior sample sits on the ruling where it touches the post's top face; a
  ray cast from a point on a face is a coin toss (Outside), so the piece
  survived as a tunnel wall while the outside piece was stranded, and the
  pairwise fuse was ACCEPTED closed by edge id with the rod dropped. The
  box classifier now reports its tolerance band as On, and the builder
  re-samples any non-coincident sub-face whose sample is On or within
  tolerance of an opposing face (`point_on_solid_boundary`) at points
  halfway toward its outer wire's vertices (`face_interior_candidates`).
  (2) The rim circle touches the outer face's top edge at the tangent
  point, so that face's outer wire winds through the circle (a keyhole);
  `cdt_triangulate_simple` fenced the circle but never removed its
  interior, so the ring double-covered the coincident cap (hundreds of
  non-manifold mesh edges). A loop that revisits a vertex is split there:
  a sub-loop winding against the main loop is flood-removed as a hole.
  The captured three-member cluster fuse replays exact (15 faces). Re-sampling skips candidates that are themselves on the opposing boundary (a coincident face the same-domain pass left unpaired decides nothing by re-sampling; `deepcutout_cut_inmem` pins that). Tool-side (the #1661 build -> this build): `assemblyGenerator.scenario` 2 -> 1 of 23 (the arch passes; the tilted block x cradle is the last), `combriser` 0 of 4.
- **A section loop enclosing another loop: a tube standing on a plate (CLOSED 2026-09-15; pin `fuse_tube_standing_on_a_plate_nests_the_bore_loop_in_the_wall_loop`)** —
  the tube's wall circle and, inside it, its bore circle both land on the
  plate's top face; the splitter built the wall disc without the bore as its
  hole and gave the remainder both circles, so the wall circle had three
  owners and the tool's `fuseAll([plate, tube])` fell back to a 287-face blob.
  A loop enclosed by another loop is now a hole of the smallest enclosing
  loop's disc (the enclosing disc's interior sampled between its rims) and
  not of the remainder. The captured fuse replays exact (16 faces). Tool-side: `assemblyGenerator.scenario` 3 -> 2 of 23 (the counterbored, tapered tube passes; its mouth chamfer still fails cleanly, row above), `combriser` 0 of 4.
- **A section loop enclosing a face's hole: the coaxial counterbore through a tube's top annulus (CLOSED 2026-09-15; pin `cut_coaxial_counterbore_from_a_tube_splits_the_top_annulus` in `crates/operations/src/boolean/tests.rs`)** —
  the tool's counterbored, tapered tube (`assemblyGenerator.scenario`) took its
  first mesh fallback in `cutWithEvolution(tube, counterbore)`: the counterbore
  wall's circle on the tube's top annulus ENCLOSES the bore hole, and
  `split_face_with_internal_loops` emitted the loop's disc with no inner wire
  while the remainder kept the old hole (the band between the rims covered
  twice, the bore rim owned once). On planar faces an enclosed hole now moves
  into the loop's disc and both sub-faces sample their interior between their
  rims (a ring's centroid falls in its hole). The captured cut replays exact
  (6 faces). Tool-side (the #1657 build -> this build): `combriser` 0 of 4, `assemblyGenerator.scenario` 3 of 23; the tube test's export drops from 622 to 93 open mesh edges because its chain now reaches a chamfer that fails cleanly on the exact tube (`cannot normalize zero vector`, two edges at r=1/3) and a two-solid `fuseAll` that falls back (row above).
- **Coaxial same-domain pairs took their orientation from the axis sign (CLOSED 2026-09-15; pin `fuse_plate_onto_downward_extruded_cell_bands_keeps_the_corner_slivers` in `crates/operations/src/boolean/tests.rs`, unit pins `cylinders_same_domain_opposite_axis_shares_normals` + `torus_same_domain_opposite_axis_shares_normals`)** —
  the empty 2x1 assembly base (every `combriser` test) exported 46 open mesh
  edges: the plate is extruded up from z=-0.01 onto two cell socket tops whose
  top 0.25 mm is a band lofted DOWN from z=0, so the four corner cylinders
  coincide over the 0.01 mm overlap with opposite axis vectors.
  `surfaces_same_domain` returned `axis_dot > 0` as the orientation flag, the
  four sliver pairs read as opposite and the fuse dropped both faces of each
  (16 free edges). A cylinder's normal is the outward radial direction and a
  torus's points out of its tube whichever way the axis runs, so those pairs
  are always same-direction and only the reversal flags decide; cones keep the
  axis test (their normal's axial component flips). Capture recipe: an
  untracked `__kernel-tests__` probe that rebuilds the plate as
  `assemblyGenerator.ts` does and calls `buildBaseSocket(2, 1, ...)`, then
  `replay_pair OP=fuse FREE_EDGES=1 BK_SD_SEL=1` (the pass now logs every
  pair's surfaces and flags under `BK_SD_SEL`). The same rule had broken the
  3x3 L body x lip fuse on today's tool (`export.customShape`: `fuseWithEvolution`
  mesh-fell-back, the whole export a 890-face blob): the current loft marks a
  concave-corner cylinder reversed, which the axis-sign rule misread the other
  way. The `lship_lipfuse_inmem.rs` operands were re-captured (its 2.127-era
  lip encoded that orientation in the axis instead). Discovered on the way: the
  same body x lip pair under `OP=intersect` leaves 18 free edges and falls back
  to a mesh (not a tool op; undug). Tool-side (tool 4a66decb, the #1654 build
  -> this build, same worktree): `combriser` 4 -> 0 of 4, `assemblyGenerator.scenario`
  6 -> 3 of 23 (below the same-day 3.3.9 control's 6; the three left are the
  row below), `export.customShape` 0 of 23 and exact, `export.groupedScoop`
  4 -> 4, `export.solidCutouts` 0 -> 0, `scenario.solidCutouts` 5 -> 5 (snapshots).
- **Prism vertical-corner fillets, the assembly parts' first stage (CLOSED 2026-09-15; pin `prism_vertical_corner_fillets_are_watertight`, probe `comb_part_probe`)** —
  two roots. (1) A stripe's terminal end on an untouched cap was closed by a
  reversed planar runout patch overlaying the cap's corner: the end-cap notch
  ran after the corner solver and only for one-edge planar fillets, and a
  multi-contact wall never split its spoke. A terminal whose third face is a
  planar cap with two straight corner edges that both contacts end on is now
  "notchable": the spoke splits there on every face
  (`BoundarySplitPolicy::Selective`), the doubled-back tail is dropped as a
  chain, and the cap adopts the cross-section arc before the corner solver
  runs, so no runout cycle exists. Other terminals keep the runout closure
  (the aggressive scoop's contacts do not end on its caps' corner edges; a
  split there strands one sub-edge per cap). (2) With every original face
  rebuilt the orientation repair had no seeds, and the pairing walk toggled
  the two mirrored corner bands' flags (their raw wires wind clockwise around
  the surface normal) and inverted them; a notched cap keeps its source's
  surface, flag and traversal and now seeds the repair. REFUTED: seeding on
  every rebuilt original (cross_one_row volume oracle); an area-vector
  winding check at band creation or after propagation (the engine calibrates
  its walk convention per solid from seeds, and a rebuilt periodic face has
  no meaningful area vector: 180 wrong flips on cross_one_row); a plan-level
  "tangent unselected spokes are not sharp" filter (the aggressive scoop's
  mixed-radius junctions then have nothing to share). The wasm fillet
  chain's validity gate (`try_fillet`) now also requires the result's mesh
  to be watertight, so a closed-by-id overlay falls through to a clean
  failure instead of being consumed. The comb part now cuts its slots
  exact; its rim ease fails cleanly (OPEN row).
- **3.4.0 blend cutover twins at sharp-cornered fillets (CLOSED 2026-09-14; pins `groupedscoop_fillet_has_no_twin_edges`, `groupedscoop_cut_stays_exact`, `groupedscoop_plan_drops_flat_edges`)** —
  three emission roots, none of them the 4-stripe fan (it never runs once the
  flat edges are gone): (1) the plan filleted three tangent-continuous edges
  (coplanar bottom pieces of the group outline) into zero-width bands with
  twin contacts; the plan now drops tangent edges, as the reference kernel
  ignores them (`fillet_plan::faces_tangent_along_edge`). (2) The batch
  trimmer split a boundary edge at a contact end only for one-edge fillets,
  so each wall at a spoke minted its own vertex at the same point and bridged
  to it with a connector twinning the other wall's; it now splits at every
  single-contact face under `BoundarySplitPolicy` (junction spokes split,
  terminal-end spokes keep the runout closure, one-edge planar fillets split
  everything for the end-cap notch) and drops the doubled-back tail as a
  chain, since a neighbour's earlier trim can have split the spoke at another
  height. (3) Two r=2 stripes meeting across the cylinder seam each registered
  a coincident cross-section; the second band now adopts the first's edge as
  its second owner and the junction builds no patch
  (`junction_cross_sections_closed`). The bin cut's endpoint-keyed merge had
  collapsed the twins into a four-owner edge. REFUTED on the way: the
  roadmap's "3.3.9 closed it with 28 faces and the cut stayed exact" (its cut
  removed 2.1x the tool, see the OPEN corner-model row), and a simple-cycle
  requirement in `boundary_cycle`.
- **Trimmed cylinder and cone faces inflated the solid bounding box to the full circle (CLOSED 2026-09-14; root of the wall-cutout corner family)** —
  `expand_cylinder_at_vertices` / `expand_cone_at_vertices` added the whole
  circle's axis-aligned extremes at every face vertex regardless of angular
  extent, so a 0.001 mm slab intersected across an r=5 corner arc reported the
  full 40 mm width (the tool's `wallCutoutBuilder.test.ts` reads widths that
  way: "still rounds the u-shape bottom corners" and 5 siblings). The boolean
  and volume were right all along. Replaced by exact per-edge extremes over
  `domain_with_endpoints` for circles and ellipses (a ruled quadric's extreme
  lies on a ruling whose ends are boundary edges) and the subdivided Bezier
  hull of NURBS edges (conservative by the convex-hull property).
  Pins `measure::bounding_box::tests::{concave_arc_notch_does_not_inflate_the_box,thin_slab_through_arc_corners_has_tight_bounds}`.
  Spheres and tori keep their conservative full extents.
- **Plane-torus section perf (CLOSED 2026-08-28; torus−box boolean 12.4ms → 5.0ms, 2.46x, still exact analytic)** —
  `exact_plane_analytic`/`intersect_plane_torus` sent every plane-torus through a
  128² sign-change grid + Newton refinement (~49K torus evals), then a NURBS fit the
  sample path discarded. Replaced with the per-v closed form
  `cos(u−φ) = (d − n·center − r·c·sin v) / (s·(R + r·cos v))` (scan v, solve u directly,
  as `intersect_plane_cylinder` already did), routing the sample path around the wasted
  fit; a `chain_self_touches` gate keeps the inner-tangent figure-eight open. Perpendicular
  planes fall out as exact-circle sampling. Pins: `analytic_intersection::tests::plane_torus_{lobe_closes_and_stays_on_surface,inner_tangent_figure_eight_stays_open}`,
  `cut_torus_by_box_notch_is_analytic_watertight`. Do NOT reintroduce the 2D grid.
- **Kumiko corner-window strut fuse (CLOSED 2026-08-15, ten roots, #1599-#1616; pin `kumiko_diagonal_strut_fuse_is_exact` ACTIVE)** —
  exact watertight fuse (70 faces, free=0 over=0); every root was marched-geometry
  ~1e-6..1e-5 drift meeting an exact-tol gate (endpoints, vertices, tangents) — fixed by
  weld-scale bands with 10x ambiguity guards or shared-anchor adoption at each altitude.
  Fixtures and instrument census: `crates/io/tests/kumiko_strut_fuse_inmem.rs` + replay_pair. Tool-side verified same-day on 3.3.7: generator suite 2883/2883 (292 files) vs a 2883/2883 control on 3.3.0, wall 221s vs 229s; head-to-head matrix 0.49x aggregate (10476ms vs 5155ms), nm 0 vs 5, only the three known scoop-family reference double-cover volume flags.

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

- **A closed edge's samples start at its curve's parametric origin, not at its
  vertex.** An ellipse always starts at its major vertex and a converted or
  transformed conic starts wherever its new frame puts it, so any consumer
  that walks a closed rim into a seam must re-sequence the rim at the vertex
  (`anchor_closed_edges_at_vertices` in the CDT mesher). The band mesher is
  immune because it orders by angle.
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
- **A width read off `getBounds` is a measurement, not geometry.** The wall-cutout
  "fillet never applied" family was a vertex-anchored bounding box adding full-circle
  extents on trimmed cylinder faces; the intersect's volume was exact throughout.
  Before blaming a boolean for a bounds delta, print the result's volume against the
  analytic expectation (`thin_slab_through_arc_corners_has_tight_bounds` is the template).
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

- **The tool's generator suite runs on the REFERENCE kernel unless `BREPJS_KERNEL=brepkit` is set** (`__kernel-tests__/wasmInit.ts` defaults to the reference id; the tool's CI never sets it). "Green on every bump" is a reference-kernel statement; only an explicit brepkit run is a brepkit signal.
- **Run recipe that survived 2026-09-14:** per-version worktree under the tool's `.worktrees/` with the `brepkit-wasm` pin edited, then
  `systemd-run --user --unit=<name> --collect --working-directory=<wt> --setenv=BREPJS_KERNEL=brepkit --setenv=NO_COLOR=1 --setenv=PATH="$PATH" bash -c "./node_modules/.bin/vitest run --project generators --project generators-heavy --pool=forks --maxWorkers=2 --reporter=default --reporter=json --outputFile.json=<out>"`.
  The root config's threads pool at 75% of cores takes a V8 heap OOM in one worker down with the whole run (`Worker exited unexpectedly`); forks lose only that file. A sandboxed foreground call cannot outlive its timeout and a plain `setsid nohup` child died with the call; the user unit survives. Never `pkill -f` a pattern that appears in your own command line.
- **Attribute a failure to a kernel version only from a same-day pair** on the same tool commit: of 53 failing files on 3.4.0, 50 failed identically on 3.3.9.
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

- **`validate_solid` mis-reports a multi-component shell as an Euler error** (or, when one component is a whole torus, as "shell is disconnected"). A 2x2
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
