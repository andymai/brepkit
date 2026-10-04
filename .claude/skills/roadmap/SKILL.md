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

**Inventory status (2026-10-03): FIVE deferred-defect pins**, each owned by an
OPEN row: `groupedscoop_tool_tessellates_watertight` +
`groupedscoop_cut_removes_at_most_the_tool` (the two-stripe corner model),
the fanned developable band
(`tessellate::tests::band_with_split_ruling_keeps_triangles_on_the_arc`),
and the hinge lid past its stop
(`hinge_lid_past_its_stop_overlaps_the_lip_exactly` +
`hinge_left_lid_past_its_stop_overlaps_the_lip_exactly`).
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
| **Rounded-prism rim ease (the assembly parts' second stage): the plane x cylinder analytic stripe rounds the cove side of a rounded-prism cap** (diagnostic `comb_part_probe` with `STAGE=rim FLUX=1`; mechanism `plane_material_inside_cylinder` on branch `wip/prism-corner-fillets`, rebased onto the rim fix (Closed below) as `wip/rounded-prism-cap-side`) | After the corner fix the comb's r=1 rim ease (four lines and four r=2.5 corner arcs, eight tangent junctions over a smooth spoke) fails cleanly and the part keeps its sharp rim: the plane x cylinder stripe classifies the cap as a plate around a post (`plane_is_bounded_disc` only knows a full disc), places the four corner-arc stripes at (±32.5, ±8, 35) outside the part, and no cross-section can then be shared at the junctions (16 registered fresh). A local material-side test at the spine (is the cap's interior toward the cylinder axis) puts them right and the rim builds its 8 cylinders + 4 tori + 6 planes, but one trimmed corner cylinder comes out inverted (the pairing walk reaches it through a torus), and the test shifts the `cross_one_row` volume oracle by 79 mm³ (64,889 vs 64,982; oracle 64,968 within 65) while the dumped faces at both stripes it reclassifies (the r=7 boss top and the r=12 step rim, origin (58, 42)) are identical in both modes: untraced. Tool-side this is the difference between a comb or riser with eased rims and one without; the parts are valid either way. BLOCKER for landing the side test (measured 2026-10-03 on `wip/rounded-prism-cap-side`): the interior fillets it lets build reach the tool's interior-fillet bins, whose fillet-material fuse (op 1463: the bin against a fillet body with 27 tori tangent to the floor at z = 2.25, operands in the session scratchpad `if1463/`) falls back natively with free edges on that floor, so `export.interiorFillet` goes 2 -> 5 failures, `export.interiorFilletScoops` 2 -> 8 and `assemblyGenerator.scenario` 1 -> 2. Its tangent floor sections are exact since the plane-resting-on-a-torus fix (Closed below); the raw fuse then assembles 199 analytic faces with 18 free edges, and the operations path still fails on faces the splitter returns nothing for (`face ... is cut by sections but split into nothing`). Next: those faces (`replay_pair A=if1463/a.bin B=if1463/b.bin`). Tool-side same-day triple on tool 4a66decb, one worktree (3.4.0 stock -> the #1650 build -> the #1654 build): `assemblyGenerator.scenario` 13 -> 10 -> 6 (the 3.3.9 count, back at parity), `combriser` 4 -> 4 -> 4 was the EMPTY 2x1 base's plate x socket fuse, not the parts (Closed below: coaxial same-domain orientation). `export.groupedScoop` 4 -> 4 -> 4 (the two-stripe corner model), `export.solidCutouts` 1 -> 0 -> 0, `scenario.solidCutouts` 5 -> 5 -> 5 (snapshots) |
| **Assembly cluster fuses degrade to mesh** (`assemblyGenerator.scenario.test.ts`: "arch at the base center" and "tilted block and cradle" throw `cluster fuse degraded to mesh fallback`; operands captured with the `captureScoopOps.test.ts` wrapper pattern, replayed with `replay_pair FUSE_ALL=1 FUSE_MEMBERS=<third member>`) | TILTED only, since the arch closed (Closed entry below): block x cradle (two tilted parts overlapping 5 mm in x) fails GFA ("open growth shell with 18 faces would be dropped") and the mesh fallback is non-manifold; plate x cradle is accepted closed by id with 100 open mesh edges and a volume 2037 mm³ below the raw result. Block x cradle alone (`BK_OPEN_SHELL=1`): the 18-face open shell is the cradle's part outside the block, and its free edges are the cradle's cut boundary on the block's +x face and the block's -y corner cylinder (a line, the groove arcs and two ellipse arcs at x=8): those block faces were never split there, six EF crossings of the cradle's rounded-edge curves with the +x face having been dropped as `outside face boundary`. TUBE: the test passes since the plate x tube nesting closed, and its mouth chamfer (`chamfer(tube, both rims at z=60, 1/3)`) takes the round-rim chamfer (Closed below; tool-side unmeasured). The six dropped EF crossings are correct (those cradle curves cross x=8 beyond the block's rounded corners). SUB-ROOT 1 FOUND: the block's +x face reached the splitter with its two straight NURBS boundary edges (the fillet contact lines) unsplit, so its five sections could anchor on no boundary vertex and were all dropped; `boundary_edges_to_pcurve_with_images` expands a NURBS edge only when circle-like, and a straight NURBS edge with a weld-coincident pave junction now expands too on branch `fix/nurbs-line-boundary-expansion` (parked: with it the pair assembles 34 faces but 26 free edges remain from plane sections overshooting into the rounded-corner region and tiny marched pieces where the cradle's 0.8 mm rounded edge meets the block's corner cylinder; no primitive pin discriminates the expansion yet, an axis-aligned or tilted NURBS-edged box splits fine either way). Next: those two remainders on the captured pair (`asmcap/tilted`, `replay_pair OP=fuse BK_OPEN_SHELL=1`) |
| **Kumiko corner-wrap export: the 8-tool cut (op 5928) runs past 1,200 s natively** (captured from `slideRailBuilder.test.ts` "is not carved away by a kumiko wrap either"; operands in the session scratchpad `op5928/`, tool 6 replaced by the exact slab cut in `op5898/cut_exact.bin`) | With the first band's cuts exact (Closed below), the four corner bands' slot-box compound cuts (ops 1988, 3252, 4669, 5801) take 2.2 to 14.4 s in the tool and the slab cut that follows (op 5898) is exact. Op 5928 then compound-cuts a 94-face base (12 cones, 36 cylinders, 46 planes) by 8 tools: four planar lattice pieces (two of 645 faces, two of 458) and four exact corner bands (200 to 228 faces, mostly NURBS). Natively it did not finish within 1,200 s (2 GB); on 4.1.5, with mesh-blob bands, it trapped after 112 s. Profiled: the 8-way `fuse_n` fails in 5.1 s, then `fuse_cluster`'s pairwise ladder fuses a lattice with a band (two disjoint pieces after the slab) through `fuse_multi_component_tool` into a 17,756-face mesh in 65 s, and a nested boolean resets the taint flag, so the ladder carries the mesh on; 26 s of each GFA pass on that mesh was `build_sd_grouping` (746k coplanar candidate pairs whose straight edges were sampled 8 times each), 4.3 s since #1936. Cut tool by tool instead, the lattice cut is exact (0.34 s) but base less band fails (open 12-face growth shell of band faces at (62, -39)) and every later cut grinds against the resulting mesh. Base less band, dug: the band's z = 9.25 cap (from the slab) and the wall's corner cylinder meet in a circle whose crossings with the cap's NURBS groove edges `circle_face_hits` skips (`EdgeCurve::NurbsCurve(_) => continue`), so the whole quarter arc is judged by one midpoint and dropped; handling those edges (parked on `fix/kumiko-cap-circle-nurbs`) removes the 12-face open shell, and the cut then fails on the band's end plane x = 59, which runs through the tangent seam between the wall's y = -41.75 plane and its corner cylinder: the section lies on that seam edge and a duplicate edge splits off a triangle of the wall. Next: the seam-coincident section, then decline the mesh fallback inside the ladder (operands in the session scratchpad `op5928x/`, `ch5928/cur_1.bin`). Also open, off the tool's path: box 3 cut after boxes 1 and 2 one by one leaves a 6-edge hole (the compound cut takes the exact batch). Perf: the slowest band-cylinder x strut-wall pairs still spend 0.4 to 0.55 s in the grid seeder (`kumiko_pair_probe`, `PAIR=4,22`) |
| **Generator-suite hangs on brepkit** | Attributed for the kumiko wrap (row above). On 3.3.9 and 3.4.0 single tests blocked 5 to 19 minutes (`is not carved away by a kumiko wrap either` 741 s, `splits bin with compartments + scoop + thick walls + connectors` 631 s, `featureCacheKeyDiscipline` 740 s), so a full run took ~50 minutes at 2 forks where the reference finishes in ~4. Re-timed 2026-10-03 on 4.1.21, neither other file hangs. `binGenerator.scenario.split-robustness-topology` passes 14 of 14 in 23.5 s with the scoop clip exact and each split face's holes built once (Closed below; main before them failed 1 of 14 on its 90 s timeout in 176 s). Three of its booleans still fall back: the lip fuse (op 2119: trimming a plane line between two faces with arcs to the span both hold makes it exact, parked on `wip/plane-line-common-run` because it drops the spurious section that gave the hinge lid its only vertex at the barrel's nose, and the hinge footprint tests then read the lid's mesh 0.015 mm short until the mesher puts vertices at an arc's axis-aligned extremes), a fuse that falls back only in wasm (op 4412, exact natively), and an 8.6 s fuse of the bin body with its feet grid (op 7242, natively too: the body's bottom plane holds each foot corner's whole r = 3.75 rim circle, and 36 foot cones keep that circle closed and free where their own quarter-arc rim already runs, the coincident-rim class of the hinge row). `featureCacheKeyDiscipline` passes 17 of 17 in 58.7 s with the mesh order fix (Closed below; before it, 11 failures in 414 s). 2026-09-15: one of the five lid files `hingeSwing.scenario`, `lidScoopClearance.scenario`, `lidDividerClearance.scenario`, `lidMagnetSeating.scenario`, `lidLabelTabClearance.scenario` held a worker at full CPU for 18 minutes with no output; rerun one at a time under a 10-minute cap, `hingeSwing.scenario` is the one (it alone reached the cap with no test output; the other four pass in seconds). NOT a hang: every swing-sweep `intersectWithEvolution` was a mesh boolean (~22 s each in wasm, measured 2026-10-01, when both operands arrived as blobs). On a wasm built from #1937's branch (re-captured 2026-10-01, `KERNEL_OPVAL` mode of the hook validates every boolean result) the scenario's first test finishes in 59 s instead of timing out: the bin reaches the swing exact, and only the lid's keyhole pin cut still fell back, so each swing intersect was the lid blob against the exact bin (~5.8 s in wasm) and the test failed its interference floor (7.45 mm3 against 5). Re-captured 2026-10-01 on a wasm with that cut exact (#1938) and the closed-lid intersect empty (Closed below): `hingeSwing.scenario` fails 4 of 27 in 434 s where the reference kernel passes 27 in 131 s, and 46 swing intersects fall back (operands in the session scratchpad `hinge5/`). All four closed (Closed below: plane holes in the circle split; unpaired coincident faces by side): on a wasm with both, the seat test's circle split, the bin's knuckle bracket and the lid at its stop (Closed below), measured 2026-10-03, `hingeSwing.scenario` passes 27 of 27 in 186.42 s (main 219.05 s) and 5 swing intersects still fall back (scratchpad capture `hinge20/`): the overhang test's last sweep (op 153611) and the last stop intersect of each of the four stop tests (ops 158240, 162966, 166800, 170634). All five are the lid pushed past its stop (ready repros `hinge_lid_past_its_stop_overlaps_the_lip_exactly` on the seat bin and `hinge_left_lid_past_its_stop_overlaps_the_lip_exactly` on a 114-face left-wall bin, ignored): each knuckle bracket's sliver runs along the lid's r = 2.45 cove, and the section the bin's knuckle end face cuts on that cove duplicates the cove's own boundary arc at the knuckle end (rim arcs on the knuckle end faces likewise), so the loop tracers merge the sliver into the unbounded face and never split it out. Dropping every curved section that runs along its own face's boundary makes all five exact and point-exact (`wip/boundary-section-drop`) but breaks coincident-rim fuses in the same scenario (op 103114, corner cylinders whose rims lie on coplanar bottoms: 12 non-manifold edges; 7 of 27 tests fail), because the partner face keeps the section's edge while this face keeps its boundary edge. Neither face type nor a retry only on broken traces separates the two: the hinge needs the drop on both plane and curved faces. Earlier op numbers below are those of the scratchpad capture `hinge14/`. Left from that work: in the left-wall bin's two-pin compound cut (scratchpad `hinge6/fb12019*`, data `hinge_left_bin_knuckled.bin` with its pins), `compound_cut`'s merged-tools shortcut leaves 21 free edges and the sequential path's second cut 4, because the knuckle end face where the short pin (keyhole 0.925) ends and the long pin (keyhole 1.0) begins splits into a piece that joins the outer ring to the ring between the keyholes by running the r = 1.0 arc both ways (`replay_pair TOOLS=... MERGE_TOOLS=1 BK_SPLITW=103`); the batched path is exact and matches the point oracle. The overhang test's sweeps (ops 153467 on) are genuine thin overlaps, not contacts: op 153467's lid rim overlaps the bin's back lip in a sliver (7,310 of 512,000 grid points inside both in a 0.4 x 0.4 mm slab at y 41.4 to 41.8, z 46.3 to 46.7, about 0.29 mm3, where the mesh fallback reads 5.497). Perf: the four slowest swing intersects (ops 5095, 9869, 13751, 17633) take 0.24 to 0.39 s natively, from about 5.7 s each (Closed below: marching bounded to the face pair's box overlap). A hole image fix for the lid-minus-bin cut (split edges shared with the coplanar partner read in the parent's direction) is parked on `wip/hole-image-direction` until that cut can pin it. Per-op capture: `KERNEL_OPCAP=<dir>` hook in the `brepkit-kumiko` tool worktree's `kernelInit.ts` (uncommitted) saves every fallback's operands |
| **A long-lived kernel's arena only grows** (brepjs's `dispose` is a no-op) | Nothing removes entities from the arena, so one kernel serving a long session grows until the 4 GB wasm heap and the next allocation traps with no panic message; the tool's worker recovers by recreating the kernel. Measured 2026-10-03 on 4.1.21: `featureCacheKeyDiscipline.test.ts` grew its test worker to 4.16 GB and trapped in a fillet at call 1,794,666, poisoning the file's 10 later tests, while mesh order noise sent every labelTabs candidate through whole-bin rebuilds; with the order fix (Closed below) the file passes in 58.7 s. Needs reclamation: a sweep from the live solids brepjs reports (a release on dispose), or compaction with a handle remap |
| **Export and volume meshes fan a developable band whose ruling edges carry vertices** (ready repro `tessellate::tests::band_with_split_ruling_keeps_triangles_on_the_arc`, ignored; live: the `hingeSwing` second pin cut, scratchpad `pincut/bin.bin` x `tool1.bin`) | `interior_grid_resolution` gives cylinders and cones NO interior points on the display/export path (`(2, 1)`), and the boundary-only Delaunay of a band whose length dwarfs its arc only ladders while its ruling edges carry nothing but corners: a ruling split by a section (the pin's tip line cut by the knuckle's step plane 0.4 mm inside each rim) is fanned to a whole rim and the triangles between that fan and the far rim span the entire 70 degrees, flat. The pin's bore integrates 8.6% short, the bin's own export volume 0.7% short (45252 vs 45560 with interior rows), and the mesh divergence agrees with `oriented_solid_volume` because both read the same mesh, so the cut/intersect identity `V(A-B)+V(A∩B)=V(A)` holds while both are wrong: only the fuse-derived `V(A)+V(B)-V(A∪B)` or an oracle scan (axis and cross-section grids, `POINT_IN`) exposes it. Also bites `measure::solid_volume` on a UNIFIED full bore wall: the keyhole-pin compound cut (`compound_cut_by_two_keyhole_pins_meeting_on_a_knuckle_face_stays_exact`) reads 11 mm3 high, one bore's worth, at 0.01 and 0.001 while `oriented_solid_volume` matches the sequential oracle; pin volumes with the oriented integrator. The oriented integrator misreads split bands too (2026-10-02, at 0.001): the hinge seat pair's cut (`hinge_seat_bin.bin` minus `hinge_seat_lid.bin`) reads 45158 against the bin's 45117 while a 216,000-point comparison over the hinge strip matches bin minus lid exactly. REFUTED: one interior row at the rim's u density (fans stay at the corner vertices); aligned columns at the boundary's u samples plus rows at every vertex v (T-junctions and 173 non-manifold export edges on `dovetail_a1corner_hole0`, triangle budgets blown, and `cross_one_row_fillet` drifts 5% from its reference oracle because the winding vote flips on the refilled faces); refining fat triangles by their centroids (diverges on full-turn walls, 376k triangles on one face). The reference mesher seeds a near-isotropic grid on developable faces; the fix must keep `cross_one_row_fillet` within 0.1% of its oracle and will move the mesh-derived pins `mitsukude_panel_cut` (27027.9 -> ~27095) and `spacer_foot_fuse` (2404.44 -> ~2397.8), which were calibrated on fanned meshes. Also live in `hinge_swing_inmem.rs`: `compound_cut`'s unify step merges stacked corner cylinders into bands with split rulings, and the cut lid's mesh volume reads 37189.8 against the exact 37205.1, the cut bin's 44194.6 against 44502.7 (each outer corner band 5% short in area) |
| **The brepkit adapter's `simplify` does not merge coplanar faces** (`nestingFloor.geometry` "has a flat underside around magnet pockets without boss seams", export on and off; ops captured in the session scratchpad `nest5/`, `nest6/`, `nest8/`) | Every magnet boss fused onto the bed leaves its bottom as a separate face in the bed's plane on either kernel; the reference chain merges them in the tool's `simplify(skirt)` (`trayBottomStage.ts`), whose reference adapter unifies same-domain faces, while the brepkit adapter's `simplify` (brepjs `src/kernel/brepkit/modifierOps.ts`) only calls `healSolid`. With `unify_faces` sound on these bodies (Closed below) and that `simplify` patched locally to also run `unifyFaces`, `nestingFloor.geometry` passes 19 of 19 (2 failures with the stock `simplify`). Next: the brepjs change, once a kernel release carries the `unify_faces` fixes; with an older kernel the merge opens other nesting meshes |
| **4x4 mag no-lip noise-band watch (1.04x on the 3.2.38 matrix)** | The only row the reference leads; has oscillated 1.00x-1.06x across 3.2.36-3.2.38 with no kernel change targeting it. Watch, do not chase, unless a fresh same-day matrix shows a real drift |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** | A product call, not just a fix: rejecting means the op fails outright. Mitigation shipped: `boolean::mesh_fallback_count()` + wasm `meshFallbackCount()` let pipelines snapshot-and-refuse |
| **Export angular default (5°) vs the reference's coarser effective default** | Tolerance-parity product choice, not mesher waste: 5° forces 18 segments/quarter-arc on r=0.6 slot corners, ~1.7x triangles vs reference at fine deflection. Revisit only as a product decision |
| **Marched FF sections carry `pave_block_id=None`** | Architectural note without a live repro (the snapClip op-cut-3 case replays clean, fixture `snapclip_export_corner_inmem.rs` ACTIVE). If a new leak lands here, the canonical altitude is pave-block attachment at phase-FF/make_blocks — every face-splitter-level attempt broke calibrated chains |
| **Hinge lid ∩ knuckle falls back** (operands in `crates/io/tests/data/hinge_lid*.bin`, `hinge_swing_inmem.rs`: the lid compound-cut by its five tools, intersected with `hinge_lid_knuckle.bin`) | The knuckle's flat top runs in the lid's clearance bevel plane, and the intersect leaves 8 free edges along that coplanar contact; the fuse and both cuts of the pair are exact. Off the tool's path (it fuses the knuckles); undug |
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
| **Torus booleans (Beta)** | Audited 2026-09-24 against `make_torus(4, 1.5)` (15 tools x 3 ops, probe `zz_torus_audit` in the session scratchpad): exact for planes across or through the axis, planes whose loops wind around the tube, a cube over the ring's side (a lobe and trimmed loops), coaxial tori, and balls and rods on the axis; a small box inside the tube (its cut the ring around a box cavity) is exact too, and a rod across the tube reads 2.3230 against a numeric 2.32302. The rest fall back (row below). A sweep of 144 boxes centred inside the tube (x in {0.1, 1.3}, y in {2.3, 3.05, 4.1, 5.2}, z in {-0.4, 0.37, 1.1}, half-sizes 0.6 to 3; probe `zz_torusbox` in the session scratchpad) leaves 377 of its 432 ops exact, valid, watertight and within 1e-6 of a grid-integral truth, and 55 safe fallbacks (row below) |
| **Torus booleans against boxes: loops around the ring, and small tools through the wall** (the box sweep above) | Safe fallbacks. A box around the axis spanning the tube on both sides (a 6-cube centred on it) cuts lobes that join into loops winding once around the ring, not the tube, so the torus splits into bands whose rims are chains of lobe arcs: the band counterpart of `split_torus_by_tube_loops` (latitude seams become meridian seams). Of boxes through the tube's wall, one with a face tangent to the tube falls back (a face on the inner equator's plane, or on the tube's top or bottom plane, as at (0, 3, 0.5) with half-size 0.5, 1 or 2), and so does one whose two faces' sections meet at two degree-3 nodes round a sliver lens, which the loop chainer cannot take (centre (0.1, 3.05, 0.37), half-size 0.6: its `y = 2.45` face meets the tube only past `|x| = 0.4975`, 0.0025 inside its `x = -0.5` wall). The sweep's other fallbacks are undug. A box over half the ring whose `y` wall passes the axis (`|x| < 3`, `y > -0.5`) folds its tube loops (each side wall's section turns back on the tube at `y = 0`), so `TubeLoop::new` rejects them and the op falls back (pinned by `box_wall_on_either_side_of_the_ring_axis`) |
| **Plane x torus sections the `v` scan misses or misreads** (`plane_torus_loops` in `crates/math/src/analytic_intersection.rs`, 128 scan steps) | A run of `v` shorter than a scan step is dropped: on `(4, 1.5)` a plane through the centre tilted 0.001 or 0.005 rad gives no loops and one tilted 0.01 only the outer one, and a wall within 3e-4 of the outer equator gives none (the same on main). A gap between two runs narrower than a step reads as a touch, so a wall within 3e-4 outside the inner equator gives two open runs where the section is one closed loop. A plane tangent at an inner point whose `v` is an odd multiple of pi/128 closes its self-touching section through the node (17 of 62 such planes). Next: find the extrema of `|rhs|` inside each scan interval and start, split or end runs there |
| **Torus booleans off the axis: curved tools** (`make_torus(4, 1.5)` against `make_sphere(1)` at (5, 0, 0), and a second torus (4, 1) turned 90 degrees about x) | All three ops fall back on each (safe meshes, volumes right). Both go through the general marcher, whose curves FF drops (721 duplicates for the crossed tori) |
| **Non-planar sweep profiles (Beta); IGES, render (Experimental)** | Not yet audited. IGES round trips lose curved faces by design: export skips analytic surfaces and import rebuilds each plane as a unit square (README, Known Limitations), so `make_cylinder(1, 1)` less the box over `x > 0.5` comes back as three squares |
| **A second fillet on an edge beside a first fillet's blend throws its band past the solid** (`try_fillet_second_pass_does_not_break_solid` and `try_fillet_nurbs_blend_neighbor_is_watertight` in `crates/wasm/src/helpers.rs`) | On a 10 mm cube with one or two edges filleted r 1, several r 0.5 second fillets return watertight results with vertices one radius outside the cube (at -0.5, or at x 10.5). One (edge 31 after two first fillets) is inconsistent, 1010.11 exact against 985.42 meshed, and `try_fillet` now rejects any result whose exact and meshed volumes disagree by more than 1%; the others pass that gate and are still accepted. Undug: where the stripe meets the neighbouring blend |
| **Stable row defect: walking-engine chamfer at a closed rim or a shared vertex** | `chamfer_v2` on a cylinder's or cone's circular rim builds its cone face but returns a shell whose edges are not all shared by two faces (the endpoint-sampled trims cannot take a closed contact; the corrected fillet builder's periodic-contour machinery is the model). Two chamfered edges meeting at a box corner are refused (`TrimmingFailure`, pin `chamfer_v2_refuses_edges_meeting_at_a_vertex`; unrefused they left 9 edges open): each stripe's end detour lies in the other chamfer's removed region, so the two chamfer planes need a mitre along their intersection line (three at a corner need a corner patch). brepjs calls the planar `chamfer` in `chamfer.rs`, not this builder; the wasm `chamferV2` and `chamferDistanceAngle` bindings reach it |
| **Cone cuts parallel to the axis: the remaining fallbacks** (`cone_cut_parallel_to_its_axis` in `crates/operations/tests/cone_plane_cut.rs` lists the exact cells) | Safe fallbacks, none wrong. Both cones fall back when the piece the plane cuts off holds the seam (the pointed cone turned 17 degrees at x < 0.3 and beyond or turned 200 at x < -0.7 and below, the frustum at x < 1.2 and beyond at turns 0 and 17 or at x < -1 and below at turn 200): the section is anchored where it crosses the seam ruling, and the region the seam pinches still traces wrong. The pointed cone at turn 0 builds even so, since its hyperbola's vertex lands exactly on the seam. The loop, stripe-meshing and rim changes of the Closed entry are cone-only: on cylinders they broke a box less a quarter cylinder (the closed-rim sampling start) and six rod and knuckle fixtures, so a rod cut by a wall crossing one rim (tilted 10 or 25 degrees) still falls back where they would build it exactly. Next idea: trace a cone in its unrolled window [u_s, u_s + 2 pi] with non-periodic keys so a seam vertex's two copies stay distinct and the piece the seam crosses becomes two faces sharing the seam edge; a pointed cone needs a synthetic apex edge to close (the DCEL retry on every pointed cone changed nothing) |
| **A rod parallel to a pointed cone's axis still falls back where its loop surrounds the axis, or leaves through a frustum's top** (`make_cone(3, 0, 6)` at `z = -3` with `make_cylinder(0.6, 20)` at `(0.3, 0.2, -10)`; the frustum `make_cone(3, 1.5, 6)` with the rod at `(1.2, 0.5, -10)`) | Safe fallbacks. Around the axis the loop winds the cone as well; traced as one loop it broke the near-coaxial `circleinsert` socket fuse, so it stays in two branches and the op falls back, as does a rod whose wall passes nearer the axis than the rulings resolve the loop's bend there. Through a frustum's top the loop is clipped and still comes as two open branches |
| **A pointed cone through a plate's edge meshes open at some turns where the plate's window crosses its seam** (`make_cone(1.2, 0, 10)` at `(0, 5, -4)` turned about its axis, fused with `make_box(10, 10, 2)`; `a_pointed_cone_through_a_plate_edge_fuses_to_a_valid_solid` in `crates/operations/tests/rod_through_plate_edge.rs` pins validity and volume) | Measured 2026-09-28 at every sixteenth of a turn and at 5.3 to 5.6: every result is valid, but the exact fuse at `x = 0` meshes open at four turns (102 open or non-manifold mesh edges turned 1.57 and 4.71, where the seam lies on the window's ruling edge, and 3 at 1.96 and 4.32). At `x = 0.4` every exact turn meshes watertight, and turned 2.36 and 3.93 the fuse falls back to a valid mesh. Undug |
| **A rod through a plate's edge with its seam through a window corner can fall back; some exact cuts mesh open** (the reviewer's battery, probe `zz_probe_1787` in the session scratchpad) | On main as well: turned `-acos(-x)` the rod at `x = -0.3`, `-0.1` and `0.4` falls back in every op, and at `x = -0.5` turned `acos(0.5)` the rod less the plate and the Fuse do, each mesh up to 2.1% off. Fourteen results are exact and within `1e-6` but mesh open, among them the rod at `x = 0.4` turned 2 less the plate and the Intersects with a plate turned 0.5 about `y`; main had each open, wrong or falling back |
| **A hole within a degree of a cylinder's seam meshes open** (a rod turned 2 radians about its axis through a plate's edge at `x = 0.4`) | The fuse is exact and measures right, but its solid mesh is open and the rod wall's per-face mesh keeps the window, on main as well: the rim's samples sit about 0.068 radians off the surface projection in `(u, v)`, so the window pokes past the seam and the hole's flood clears the sliver beside it instead. |
| **A slab across a mitred rod's rim falls back** (`mitred_rod_cut_by_a_slab_across_its_rim` in `crates/operations/tests/oblique_rod_cut.rs`) | Safe fallbacks, valid and watertight, 0.4% short in volume; a slab level with the rim's peak is exact. A slab 1e-4 under the peak, whose tip holds 4e-10, comes back 1e-5 off in volume with 3 faces (turned 0 its mesh is open). Remaining roots, found 2026-09-28: the general splitter splits the wall into nothing (an ellipse rim is now tested in 3D, as circles are, so the wall's section reaches it), and a face kept whole after an empty split now needs a point on either side of each section running across it to classify alike (turned 22.5 degrees with the slab at 4.45, the wall had stood whole and the result came back exact and 0.15 short). The wall's split needs closed ellipse rims laid out like circle rims (split pieces' u along the rim's own span, the antipode split on the rim, pieces' pcurves on the rim's unwrapped u, the two seam copies in one period window), and the lens between the slab's arc and the rim then closes across period windows; a probe of those steps is parked on `wip/mitred-ellipse-rim`; the ellipse cap's two halves, split by the slab's line, both sample in the tip; and with the rim's peak on the seam the wall splits into two regions where the seam divides the tip into two. `clip_line_to_face` leaves a line across an ellipse cap unclipped; clipping it exactly changed no outcome here, and applied to circle caps too it broke rods halved through their axis by 2e-8 in volume (the line then ends off EF's rim paves) |
| **Sphere booleans across the chordal equator that still fall back** (`make_sphere(3, 32)`; `split_noseam_by_arrangement` in `crates/algo/src/builder/face_splitter/special_cases.rs`) | A section across the seam splits each hemisphere through the arrangement, which rebuilds the seam as its exact circle and keeps the collar holding the pole and a lune past each chain, one chain or more (`ball_cut_by_a_plane_across_its_equator`). One chain through the pole splits it into the lunes either side (`split_into_lunes`). Still falling back, safely: a chain through the pole on a face that already has a hole (a bored ball within a box whose wall passes the pole: the face's holes are attached to its pieces whole afterwards, and a wall crossing, clipping or grazing the bore left them in the wrong lune or touching its wire, so the lune split declines on any holed face; `a_wall_through_a_bore_by_the_pole_is_never_silently_wrong`). REFUTED: an exact-circle equator in `make_sphere` as a drop-in (half cuts turned 5, 11.25 or 30 degrees come back uncut, 2 faces, silently wrong) |
| **A rod resting on a cylinder's wall along the rod's own seam falls back** (`a_rod_resting_on_a_wall_along_its_seam_stays_valid` in `crates/operations/tests/rod_touching_a_wall.rs`) | Safe fallbacks, valid and watertight. The rod's outermost ruling rests on the wall and the two section loops touch there; when that ruling is the rod's seam, both loops meet the seam at the one point they share and `seam_through_a_crossing` (`crates/algo/src/builder/fill_images_faces.rs`) declines. Laying it out needs all of: the winding-chain splitter accepting two separators that share their seam vertex (no seam edge between them); each loop cut at its middle so the duplicate-edge merge keeps them apart; the seam-crossing scan ignoring a loop that only touches a seam at its start; the wall's split where its own seam also runs through that vertex (it adds the notched lobes again as holes); and `validate_solid` counting such a vertex once per sheet, which needs the edge directions sorted in each face's tangent plane, since where two wires of one face meet at the vertex their own corners pair the wrong edge ends |
| **A ball holding a pointed cone's apex and reaching past its base falls back** (`make_cone(3, 0, 6)` at `z = -3` against `make_sphere(5, 32)` at `(1, 0, 1)`; fixes parked on branch `fix/ef-signed-crossings`) | Safe fallbacks, in 0.06 to 0.11 s: every op comes back as a mesh (87, 73 and 1161 faces). Two roots found. Phase EF never splits the base rim where the ball crosses it: `find_edge_surface_crossings` tests the unsigned distance at 64 samples, so it finds an edge's transverse crossing of any curved surface only where a sample or midpoint lands within a few tolerances of it (a line across a unit cylinder at `y = 0.3` finds none; the vertical line through a saddle patch in `hull_bound_keeps_every_nurbs_crossing` is missed alike) (`make_box(2, 2, 2)` within `make_sphere(1.5, 32)` at its centre works because its edges cross the ball at samples, `t = 0.25` and `0.75`). And the loop's window on the wall ends 0.02 to 0.035 below the base: its run's end samples pass the margin-inclusive extent test but not the strict one, and `emit_curve_windows` brackets the bisection from them, two outside points. The parked branch finds crossings where the surface's signed side changes (bounded by the face's loops in `(u, v)` where they enclose an area) and brackets from the strictly inside samples, but it regresses `gridbin4x4_feet_fuse_is_exact_and_strictly_valid` (72 same-sense shared edges against 34), `slotted_nolip_socket_fuse_is_analytic_watertight` and `halfsockets_export_fuse_is_analytic_and_watertight` (failing after 1187 s) |
| **A cylinder's rim crossing a ball falls back** (`make_cylinder(3, 6)` at `z = -3` within `make_sphere(5, 32)` at `(1, 0, 1)`; needs the rim crossings of the row above) | With the parked branch the rim splits where the ball crosses it and the sphere's side closes its three sections into one loop, but the wall keeps one piece: the face splitter lays the top rim a period low and both seam uses at one `u` (4.712), so its trace cannot close the loop round the section that bites the bottom rim |
| **Pose audit: other primitives** (probe `zz_pose_audit`, 5 primitives x 4 tools x Cut/Intersect x upright/turned/mirrored) | Fallbacks in every pose: a pointed cone's box-corner Intersect (its truth is empty), and the ring's rod along `y` through `(0.5, ., 1)`, whose top rulings pass over the tube. The ruling trace takes a rod only when every ruling meets the ring; extended to turning points it gave exact volumes but rod pieces wound backwards around the two loops straddling the rod's seam (8 shared edges same-sense), so those stay with the marcher, which finds no crossing at all: it sizes a torus from two parameter corners that are one point and rejects the pair |
| **The cross one-row fillet's solid mesh is open** (fixture `crates/io/tests/cross_one_row_fillet_inmem.rs`) | The fillet result is closed by topology (every edge used twice), but `tessellate_solid` leaves about 1,700 mesh edges not shared by two triangles, so its volume depends on the anchor (65,090 about the origin, 29,224 about (80, 5, 10)); `solid_volume` reads 68,449 against the 64,968 oracle. The fixture's volume criterion compared the origin-anchored number and was dropped. Its r = 0.5 reversed sphere corner patches meshed their complements (about 3.03 mm² each against 1.02) until the reversed-cap fix |
| **Stable row defect: offsets and shells outside the exact image** (decline reasons log at debug level from `brepkit_offset::image`) | `offset_solid` and `shell` build exact results only while the offset keeps the input's topology, as the image checks it: no edge turns round or leaves its faces' offsets, and no planar face's loops meet (so walls passing each other are caught only where a planar face lies between them). A solid with cavities is declined outright. Past that the offset falls to the phased pipeline, which returns invalid solids: a 10-cube offset by -6 comes back with 24 free edges, the pointed `make_cone(5, 0, 10)` (its apex has no normal to offset along) with one misoriented edge, a bored plate offset until the hole meets its sides with 26 free edges. A shell of a curved solid also leaves the exact path when an open face has a hole, when the opening is a face split into coplanar pieces (they touch, and touching open faces are declined), or when an edge is an ellipse or NURBS; it then takes the polygon walls, which chord curved faces (the bored plate's cup reads 243.19 and meshes open). Next: an offset that changes topology needs the phased pipeline's intersections to share edges between faces (each face builds its own trimmed edges today) |
| **Stable row quirk: `make_sphere(r, segments)`** | The hemispheres meet on a chordal equator (line edges), a sagitta off the sphere. A plane face ending at the equator chords reads its region by them while the ray cast reads the sphere faces by the arc, so a ray through the equator plane between chords and circle loses a crossing (`sphere_ray_cast.rs` skips those rays) |
| **Point classification reads cylinder walls through chords** | `brepkit_check::classify::classify_point` (which the operations one calls) tests a ray's hit on a cylinder face against its boundary sampled into chords (32 per closed circle) in its (u, v), so a hit within a sagitta of an ellipse or other non-constant-v rim is misread. |

## Closed: root cause + where the detail lives

One line each; the fixture/PR carries the story. Newest first.

- **A rod whose side rests on or runs just inside a cylinder's wall gave invalid exact results or ran out of memory (CLOSED 2026-10-04; `crates/operations/tests/rod_touching_a_wall.rs`)**: the two section loops around the rod touch where its outermost ruling rests on the wall, each with a corner there that the fit rounded off, and a proximity gate left any two loops that close unanchored, so the rod's band came out as faces bounded by one loop each (the common's mesher then grew without bound). The ruling solvers (`closed_ruling_loops` in `crates/math/src/analytic_intersection.rs`) now sample both loops from the one ruling where they touch or pass closest, closer together near it, and only loops that meet a sibling along two separate stretches (equal crossing cylinders) stay unanchored
- **STEP cones were read and written with the wrong apex and angle (CLOSED 2026-10-03; `cones_read_from_their_placement_radius`, `earlier_brepkit_cones_read_as_written` in `crates/io/src/step/reader.rs`)**: the reader took a cone's placement as its apex and its STEP semi-angle (from the axis) as brepkit's angle (from the plane across the axis), and the writer wrote that angle back, so brepkit's own files round-tripped while a reference-kernel export of a nesting body cut read 19.8% over its volume and a brepkit frustum export read on the reference kernel at 2.4 times its own volume. The reader puts the apex `radius / tan(semi_angle)` back along the axis and converts the angle, the writer writes the semi-angle at a positive radius under a new header marker, and a brepkit export without the marker still reads as written
- **STEP faces on a surface of linear extrusion failed to read (CLOSED 2026-10-04; `crates/io/tests/step_linear_extrusion.rs`, `linear_extrusions_read_as_their_surfaces` in `crates/io/src/step/reader.rs`)**: the reader had no `SURFACE_OF_LINEAR_EXTRUSION`; a swept line reads as a plane, a circle swept along its axis or against it (the face turned over) as a cylinder, and a B-spline curve as the B-spline surface linear across the sweep, spanning the face's edges along the vector (the reference kernel's export with three swept cubic walls, `tests/data/swept_wall_cut.step`, reads at that kernel's volume)
- **Unifying faces broke the bodies it merged (CLOSED 2026-10-03; `unify_fills_a_ring_whose_hole_another_face_closes`, `unify_keeps_a_seam_in_a_merged_cylinder_wall` in `crates/operations/src/heal/tests.rs`)**: `unify_faces` carried a ring's hole into its merge with the face filling that hole, leaving the hole open, and merged two half-cylinders that share both seam lines into a band with no seam, which the mesher cannot cover; a hole another merged face fills now loses its shared edges, and a merge whose boundary would go once around a cylinder's or cone's axis is skipped
- **STEP edges given as curves on surfaces failed to read (CLOSED 2026-10-03; `surface_curves_read_through_their_3d_curve` in `crates/io/src/step/reader.rs`)**: the reader dispatched only plain curve entities, so a file whose edges carry `SURFACE_CURVE`, `SEAM_CURVE` or `INTERSECTION_CURVE` (the reference kernel's STEP export writes one for every edge) failed with `unsupported STEP entity`; it reads the wrapped 3D curve now
- **Healing broke the tool's closed bodies (CLOSED 2026-10-03; `fix_orientations_leaves_a_cups_cavity_alone`, `heal_keeps_a_cylinders_disc_caps` in `crates/operations/src/heal/tests.rs`)**: the tool runs brepjs `healSolid` in place on finished bodies, and `heal_solid` flipped every plane facing the solid's centroid (a cavity's floor and inner walls, the lip's inner bevels) deleted every face whose boundary has one vertex as a sliver (a magnet pocket's floor, one closed circle), and deleted as a duplicate any face sharing a plane, a corner count and a corner centroid with another (a floor filling a ring's opening), so `nestingFloor.geometry` failed 8 of 19 on open meshes while every boolean before the heal was closed (2 after: the OPEN `simplify` row). A plane now flips only when its outer wire winds against the way the rest of its shell's planes wind, read arc-true (a full circle's sweep takes its sign from the circle's axis); a face's size bounds its boundary curves (exactly for conics, by the control hull of each NURBS edge's span); duplicates must share their outer and hole corners
- **A plane resting on a torus's tube fell back (CLOSED 2026-10-03; `a_rod_rounded_at_its_foot_fuses_onto_a_box_exactly` in `crates/operations/tests/rounded_rod_on_a_box.rs`, `plane_tangent_to_a_tube_touches_it_along_one_circle`)**: the exact plane x torus solver declined a plane tangent to the tube, and the sampled section traced the contact circle twice at the tube's bottom; a level plane tangent to the tube now meets it in its one exact circle of radius `R`, ring or spindle (a torus whose `R` is effectively zero keeps the sampled path)
- **A rod's or tube's top rim would not round off (CLOSED 2026-10-03; `crates/operations/tests/fillet_round_rims.rs`)**: the plane x cylinder fillet held a convex rim's radius to half the cylinder's (the ring-torus bound), so an r = 3 fillet on an R = 5 rod failed with an empty contour stripe, and it read a ring cap (a tube's mouth) as a plate around a post, so the stripe sat outside the part and the mesh opened. A convex rim now takes spindle tori up to the cylinder radius, a cap counts as a disc when every boundary vertex, its holes' included, lies within that radius, and the closed-rim assembly keeps the cap's holes unless another stripe also runs on the cap (a holed plate's own hole rim). Both remove exactly the analytic ring volume
- **A split face could carry one hole twice (CLOSED 2026-10-03; `a_hole_reached_twice_is_built_once` in `crates/algo/src/builder/fill_images_faces.rs`)**: the face splitter can reach a hole both woven into a holed plane's arrangement and attached whole; built twice, its edges sat on three faces after the duplicate-edge merge. In wasm only (natively the same faces come out with each hole once) this sent the split bin's floor under a wedge connector and a bin top whose divider meets the wall to the mesh fallback; the face builder now skips an inner wire whose edges match one it already built
- **The feature cache read two identical builds as different (CLOSED 2026-10-03; `a_solid_meshes_in_the_same_order_after_any_history` in `crates/operations/src/tessellate/tests.rs`)**: the solid mesher numbered its shared vertices by walking a hash map keyed by arena edge ids, so the same label tab built after different earlier work came out in a different vertex order; edges are now walked in id order. `featureCacheKeyDiscipline.test.ts` passes 17 of 17 in 58.7 s (11 failures in 414 s before)
- **The split bin's scoop clip fell back (CLOSED 2026-10-03; `split_bin_scoops_clip_to_the_envelope_exactly` in `crates/io/tests/binsplit_inmem.rs`)**: the face splitter's line split guard compared a split point's parameter fraction with the length tolerance, so it skipped any point within 16.6 microns of a 166 mm edge's end, and a scoop facet's crossing with the envelope's corner cylinder 12.8 microns short of the end stayed a pendant. `binGenerator.scenario.split-robustness-topology` passes 14 of 14 in 29.7 s
- **The swung hinge lid's intersect with its bin fell back (CLOSED 2026-10-02; `hinge_lid_swung_40_degrees_only_touches_the_bin`, `hinge_left_bin_pin_cut_matches_its_tools`, `circle_intersect_segment_long_ruling_ending_on_the_circle`)**: an unpaired coincident sub-face now takes its class from the side the other solid lies on, and `Circle3D::intersect_segment` reads each end's own plane distance
- **The hinge lid at its stop fell back (CLOSED 2026-10-03 for twelve of the scenario's sixteen stop intersects, the other four in the generator-hangs row; `hinge_lid_at_its_stop_overlaps_the_lip_exactly`, #1957)**: ellipse trims read curved plane edges, plane lines the faces never share are dropped, partial-overlap caps wind against the faces on their loops, and the multi-piece intersect gate probes a point each piece holds
- **The bin against its knuckle bracket fell back in the hinge scenario (CLOSED 2026-10-03; `hinge_bin_against_its_knuckle_bracket_intersects_exactly`)**: the block's front face splits into side-by-side pieces on one bin face, kept together since #1953; and the cove's section with a tilted bin face ran along that face's own edge, so the opposing clip dropped it. A run along the opposing face's edge now stays a section when the face across that edge lies on the curved face being split (the knuckle's cylinder on the cove); plane pairs keep the coplanar phase's boundary
- **The hinge lid's intersect with its long pin fell back (CLOSED 2026-10-02; `hinge_lid_long_pin_intersect_is_exact`)**: the pin's end cap lies flush on a knuckle end in two side-by-side tiles; the same-domain residue pass demoted one as a duplicate and the intersect kept the other alone. A same-rank member is residue only where another member of its rank covers it; tiles ride with their pair, and the kept side keeps its tiles
- **Touching bored blocks fell back, and so did the hinge lid's intersect with its bin in the tool's seat test (CLOSED 2026-10-02; `crates/operations/tests/touching_bored_blocks.rs`, `hinge_lid_on_its_bin_overlaps_the_lip_exactly`)**: a section circle that leaves a plane face without crossing a straight edge's interior (a bore's rim through two corners of the other block's end face, a bin knuckle neck's circle running off a lid knuckle's end face) never counted as leaving it, so it was emitted whole (the bored blocks' fuse closed a phantom disc over the bore; tool op 147557 split the neck into nothing); and the split-at-T-junctions pass read a promoted hole's 227 degree rim piece as its shorter complement, splitting it where it does not run (cut)
- **Keyholed knuckles end to end fell back (CLOSED 2026-10-02; `crates/operations/tests/keyholed_knuckles_end_to_end.rs`)**: the splitter read a hole's winding with the pcurve sampler, which walks reversed arcs backwards, and flipped a correctly wound keyhole; and the plane hole weave ignored a hole that sections only end on (a ring arc meeting the keyhole's tail), so the arcs dangled and were pruned as pendants
- **Analytic marching bounded to the face pair's box overlap (CLOSED 2026-10-02; `marcher_keeps_to_its_region`)**: a bin lip cone against a tilted radius 0.1 lid fillet cylinder (boxes touching along one plane) marched 2.7 s to 13 curves; seeds now converge onto the curve and must land in the overlap, and a march stops where it leaves it
- **Hinge corner-sweep intersects fell back (CLOSED 2026-10-02; `hinge_left_lid_swung_18_degrees_only_touches_the_bin`)**: `circle_face_hits` now splits a section circle at a plane face's holes as well as its rim
- **The closed hinge lid's intersect with its bin fell back (CLOSED 2026-10-01; `hinge_closed_lid_only_touches_the_bin` in `crates/io/tests/hinge_swing_inmem.rs`)**: the two only touch, so no face of either lies inside the other and the builder's empty selection was an error; an intersect that selects nothing is now the empty solid (the mesh fallback had answered 5.27 mm3 of slivers, the swing tests' interference)
- **The hinge lid's keyhole pin cut fell back for both pins (CLOSED 2026-10-01; `hinge_lid_short_pin_cut_is_exact`, `hinge_lid_long_pin_cut_is_exact` in `crates/io/tests/hinge_swing_inmem.rs`, `crates/operations/tests/keyhole_pin_on_a_knuckle_end.rs`, #1938)**: plane arrangement and DCEL rescues read arc-true areas, an unpaired tool cap piece on the blank's boundary is On, and a plane x band ruling stops at the plane face's curved boundary and holes
- **The hinge lid's knuckle bores and knuckle fuses fell back, the bin's clearance cut turned a face inside out, and a rod poking past a box's side fell back or came back silently wrong (CLOSED 2026-10-01; `crates/io/tests/hinge_swing_inmem.rs`, `crates/operations/tests/rod_along_a_box_side.rs`)**: a bore cap's arc crossing a wall's plane twice at a shallow angle passed the in-face deviation gate though it ended off the wall's boundary (`fill_ef_in` keeps such a two-crossing leaf only when both ends lie on the face's boundary, as a faceted loft's corner arcs do); a plane x plane section shorter than the filter's sampling step was dropped when an outline had an arc (now clipped on the outlines' own lines and arcs); `point_in_region` read a point at a vertex as off both edges meeting there; `split_plane_boundary_arcs_at_points` read a rim piece past half a turn the short way round and split it at a point it never passes (a planar face's arc piece running between the two ends of a straight section now takes a vertex at its middle instead, which that misreading had supplied by accident on keyhole caps); the containment shortcut sampled rims at quarter turns and missed a rod's 0.05 poke (it now also probes each rim's farthest reach past the outer's planes); and `unify_same_domain` wound a merged reversed planar face against its members and listed a merged curved face's edges out of order.
- **The kumiko slab cut emitted a cap region twice (CLOSED 2026-10-01; `kumiko_wrap_first_band_slab_cut_stays_exact` in `crates/io/tests/kumiko_wrap_first_band_cut_inmem.rs`)**: the plane x cylinder circle kept an arc across a strut groove's mouth because `trim_ellipse_to_boundary_crossings` read the cylinder face by its `v` window and angular gap only; arcs now also read the face's own wires (`LateralTrim`).
- **The first kumiko band's 19-box compound cut was not exact (CLOSED 2026-10-01; `crates/io/tests/kumiko_wrap_first_band_cut_inmem.rs`)**: the walker closed a section on its own reverse where a smooth boundary curve is split, and the DCEL rescue counted that zero-area loop as a region (NURBS and plane faces); plane x cylinder arcs skipped elliptical rims; plane x plane lines ran past a plane face whose boundary has curved edges.
- **Mesh-fallback classification was quadratic in the pieces (CLOSED 2026-10-01; `region_classification_matches_each_triangle` in `crates/operations/src/mesh_boolean.rs`, `kumiko_wrap_first_band_cut_inmem.rs`)**: every split piece took a winding number over the whole other mesh (134.5 of 141 s on the first kumiko band's 19-box batch); pieces now share one per region of edges off the other mesh, with three agreeing probes, and pieces on the other surface keep their own test. `compound_cut` keeps its box-by-box cuts only while every one stays exact, else cuts the batch once.
- **Kumiko slot-box compound cut fell back to a 7,227-face mesh (CLOSED 2026-10-01; `kumiko_wrap_slot_compound_cut_stays_exact`, `kumiko_wrap_slot_chain_stays_exact` in `crates/io/tests/kumiko_wrap_slot_cut_inmem.rs`)**: the single cut against the 19 fused boxes cannot stay exact (the outer cylinder's 61-section split leaves 42.8 mm² uncovered), so `compound_cut` declines the mesh fallback in its batched probes and cuts box by box, which #1926 and this PR keep exact through box 19 (a clip crossing whose inside sample sits in the band, holes placed a period apart on a split cylinder, pinched planar loops traced by the DCEL).
- **Kumiko slot boxes cutting the strut-cut band fell back (CLOSED 2026-10-01; `kumiko_wrap_slot_cuts_stay_exact` in `crates/io/tests/kumiko_wrap_slot_cut_inmem.rs`, `descending_pave_span_splits_along_the_edge`)**: seven boxes cut the band exactly; the four roots (NURBS-partner overhang, descending pave spans, NURBS rim crossings, groove-edge expansion) are in the fixture's comment and #1924.
- **The kumiko corner band cut by a helical-sweep strut fell back (CLOSED 2026-09-30; `kumiko_corner_strut_cut_stays_exact` in `crates/io/tests/kumiko_wrap_strut_cut_inmem.rs`)**:
  the strut's sections with the band's cylinders chain across its 16 NURBS
  wall patches, and one patch's section came back as five two-point
  fragments. Three roots in the NURBS x NURBS marcher: a step past a patch
  edge was clamped back and the march stopped short of the edge (it now
  retries shorter outside the clamp band), a march taking no step from a seed
  already in that band ended unsnapped (the seed's edge point is appended),
  and the traced segments were re-chained by a proximity threshold that
  collapsed where a march steps short (they are now assembled in traced
  order). The cut is exact at 50 faces; the vertical x ruled-diagonal strut
  pose sweep goes from 99 to 102 of 120 exact.

- **A wall whose NURBS rim runs past its vertices fell back when sliced (CLOSED 2026-09-30; `a_slab_through_a_major_arc_wall_keeps_it_exact` in `crates/operations/tests/extrude_major_arcs.rs`, whose arcs include a NURBS running 0.05 past each vertex, stored either way round)**:
  three roots. The DCEL loop tracer kept the outside of a periodic wall
  short of a full turn as a third piece (the splitter now drops a traced
  loop of boundary edges alone that does not wind the period and runs
  clockwise read off the surface, whichever way the face winds).
  `split_arc_edges_at_collinear_vertices` skipped a NURBS stored against its
  edge (`t1 < t0`), so a wall kept the arc whole where the cap split it at
  its middle vertex; with the tracer fixed that returned an exact, wrong
  solid (0.99 against 0.314). And the revolution volume read a
  spline-bounded wall's extent from its vertices and arc midpoints, calling
  300 degrees 210 (an open spline arc is now walked like a circle).

- **A box through a torus's tube wall fell back or came back wrong (CLOSED 2026-09-30; `cube_through_the_tube_wall` in `crates/operations/tests/torus_plane_cut.rs`)**:
  five roots. A plane's section was chained from scattered `(u, v)`
  samples by nearest neighbour under a fixed step cap: it split a loop at
  its turns, where its two `u` branches meet, bridged a narrow waist
  between two loops, and failed at any scale far from a unit torus. Each
  run of `v` where the section exists is now one loop, out along one branch
  and back along the other between its exact turns, sampled evenly along
  the curve (`plane_torus_wall_sections_close_into_their_loops`). A whole
  ring's outer wire is
  its seam placeholders, so `BuilderSolid` read no flux from it and took
  the cut's shell for a cavity (it now integrates the whole ring). A wall's
  lobe beside the loops winding the tube made the tube-loop splitter give
  up (a lobe is now a hole of the sector holding it, whose seam may wind
  the tube to pass it). The CDT pinned a sector's seam to one `u` as it
  does a cylinder's meridian (a seam along the ring now keeps its
  unwrapped `(u, v)`), and `solid_volume` read a sector with a lobe hole
  as a notch band off its mesh. And a wall's oval section took each box
  edge's crossing at the nearest of 1,025 samples, so an arc starting there
  folded back past its exact end and meshed open (the crossing is now
  refined). A sweep of 144 boxes went from 182 of 432 results right on
  main (22 exact but wrong or open) to 377, none worse; single-wall lobes
  are pinned by `a_wall_cutting_a_lobe_from_the_ring_is_exact`.

- **A wall and floor over the ball fell back mirrored or turned (CLOSED 2026-09-30; `a_wall_and_floor_over_the_ball_are_exact_mirrored_and_turned` and `a_cavity_cut_by_a_wall_and_floor_keeps_its_band` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball | y > 0, z > 1` mirrored)**:
  the box over `y > b`, `z > c` leaves the upper hemisphere a band whose
  hole is the loop of wall and floor arcs, round the pole or through it.
  Neither splitter that takes such a loop (the no-seam shortcut, or the
  internal-loops path when the loop reads as winding nothing) sampled the
  band, and the generic sample, at the equator's mean `u`, landed in the
  hole whenever a mirror or turn put it there (a block holding a ball
  cavity, cut the same way upside down and mirrored, fused to a solid with
  the cavity dropped). Both now take `region_sample`, and the face fails
  when it finds no sample. Below the equator, a wall holding the axis runs
  through the ball's seam meridian, and at some turns rounding hands the
  floor arc's pcurve a turn out there, so the arrangement read the collar's
  winding backwards and sampled it past the other pole; the winding is read
  from the edges' curves now. A scan of 1,536 results (four
  walls, four floors, sixteen turns, mirrored or not) went from 261
  fallbacks on main to none. The row's rods through a trimmed patch were
  already exact on main; the box-corner piece's is pinned by
  `a_rod_through_a_box_corner_piece_is_exact`, the octant's by
  `box_octant_feeds_a_second_boolean`.

- **A box floor in the equator's plane, its walls meeting the equator between the chordal vertices, fell back (CLOSED 2026-09-29; `an_octant_turned_off_the_chordal_vertices_is_exact` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep cases `ball | octant rz1` and `ball | corner -0.5,0.5,0`)**:
  the floor's section rides the hemispheres' seam and ends between the
  chordal vertices, where the uncovered chords cannot reach it, so the
  cap's remainder never closed and the face failed. The arcs that do not
  ride the seam now go to the arrangement, which rebuilds the seam as its
  exact circle, and a face whose only sections ride its seam stays whole
  with its seam laid on that circle, split at their ends into pieces under
  half a turn so the endpoint-keyed merge cannot weld two of them.

- **A section through a sphere's pole, and a closed section inside a cap round it, fell back (CLOSED 2026-09-29; `a_plane_through_the_axis_keeps_half_the_ball`, `a_wedge_through_the_axis_keeps_its_quarter`, `a_wall_holding_the_axis_keeps_its_piece`, `a_stepped_block_through_the_axis_keeps_its_lunes` and `a_box_top_circling_the_pole_inside_its_footprint_is_exact` in `crates/operations/tests/sphere_plane_cut.rs`, pose sweep cases `ball | half x > 0` to `ball 10 | stepped block`)**:
  a chain from the seam through the pole and back (a plane or a wall
  holding the axis, two walls meeting on it) leaves no region holding the
  pole, so the arrangement kept no collar and the face failed. It is now
  the two lunes either side of the chain, each sampled on the great circle
  from its seam arc's middle toward the chain's middle, short of where that
  circle first meets the chain (a stepped block's chain bends back over its
  lune). A closed section inside a cap round the pole (a box top at
  `z = 2.95` whose floor and walls ring the pole) is a hole of the piece
  holding most of its samples and a patch of its own, or holds the cap as
  its patch's hole when it rings the cap (a tube's wall round a column's
  cap, `a_tube_round_a_columns_cap_is_exact`).

- **A sphere face whose hole winds the axis meshed through the ball (CLOSED 2026-09-28; `a_sphere_face_whose_hole_winds_the_axis_meshes_on_the_sphere` in `crates/operations/tests/tessellate_watertight.rs`)**:
  `make_sphere(3, 32)` less the box over `(-0.0001, -1.1, 0.1)` keeps an
  upper face whose hole's arc passes 0.0001 from the pole. The latitude
  band mesher took it as a collar, whose rows run along the floor's own
  columns, and the floor left the half turn from u = pi/2 to 3 pi/2
  unsampled: its mesh had a flux of 0.83 times its area where the radius is
  3, and the solid's mesh held 64.94 of 92.56. A sphere collar whose widest
  column gap would sag past four deflections now goes to the CDT (flux 2.996,
  mesh 92.41). The box over `(-0.7, -1.1, 0.1)` meshed 84.68 of 85.75, now
  85.60. Measured on main the same day: `make_torus(4, 1.5)` less
  `make_sphere(3)` reads 166.36583 (the truth 166.3658), the ball less its
  positive octant meshes watertight, and the pole face of the ball less the
  column `|x|, |y| < 2.2` meshes within 1% of its area.

- **A ball less a column and a rod through its cap meshed open at coarse deflections (CLOSED 2026-09-28; the deflections in `crates/operations/tests/column_and_rod_against_a_ball.rs`)**:
  the rod's outermost ruling runs 0.004 inside the ball between the two
  curves where the rod leaves it, and at deflection 0.02 and coarser both
  the rod's wall and the ball's cap bridged that waist with the same chord,
  which then bounded four triangles. `split_pinched_chords` in
  `crates/operations/src/tessellate/mesh_ops.rs` splits one face's pair at
  its own surface point over the chord's midpoint.

- **A cap ring thinner than half a percent of its rim's radius fell back (CLOSED 2026-09-28; the cap placements in `crates/operations/tests/ball_on_a_cone_axis.rs`)**:
  `make_cone(3, 1.5, 6)` at `z = -3` against `make_sphere(1.5148, 32)` or
  `make_sphere(1.5195, 32)` at `(0, 0, 2.75)` leave rings 0.006 and 0.0012
  wide. Three roots: the ring's seed search stepped past it (finer first
  steps now follow the coarse ones), its hole's 16-point polygon sagged
  0.029 inside the section circle so a seed in the hole read as material
  (a hole of lines and conic arcs is now read exactly), and turned, FF kept
  the ball's circle on the extended wall 0.016 above the top, inside the
  wall's margin band (a closed circle past an analytic face's window at
  every sample, crossing no boundary, is dropped).

- **A slab across a mitred rod's rim returned a silently wrong solid (CLOSED 2026-09-28; `mitred_rod_cut_by_a_slab_across_its_rim` in `crates/operations/tests/oblique_rod_cut.rs`)**:
  `make_cylinder(3, 6)` less the half-space over `z = 3 + 0.5 (x cos + y sin)`,
  then less the slab over `z = 4.07` to `4.3`, came back valid by
  `validate_solid` but 19.6 to 49.1 off in volume (84.4 true), or with the slab
  ignored. Three roots: `circle_face_hits` skipped ellipse edges, so the
  wall's circle was kept whole as an internal loop; the split-arc filter
  read a wall by its box, which holds the arcs running above the rim on the
  far side; and `face_v_range` sampled the rim five times, missing its peak
  when it rises off the seam, so the circle read as past the wall. The
  results now fall back (the open row).

- **A column fused with a rod beside it, against a ball wider than the column, meshed open (CLOSED 2026-09-28; `crates/operations/tests/column_and_rod_against_a_ball.rs`)**:
  `make_box(5, 5, 10)` at `(-2.5, -2.5, -5)` fused with `make_cylinder(0.1,
  10)` at `(2.75, 0)` against the ball turned `rotation_y(0.3)` then
  `rotation_z(1)`, and with `make_cylinder(0.15, 10)` at `(-2.7, -0.9)`
  against the upright ball. Measured on main 2026-09-28: the tool less the
  ball, the ball less the tool and their intersection are exact and valid,
  each within 1e-4 of the integrated volume, and mesh watertight at
  deflections 0.2 to 0.001.

- **A hole closer to a plane face's arc than the arc's chords sag meshed open (CLOSED 2026-09-28; `a_hole_nearer_a_floors_arc_than_its_chords_sag_meshes_watertight` in `crates/operations/tests/tessellate_watertight.rs`)**:
  `make_sphere(3, 32)` bored by `make_cylinder(0.3, 10)` at `(2.69, 0,
  -5)`, within or less the box over `(-0.7, -1.1, 0.1)`: the bore's circle
  in the box's floor passes 0.008 from the floor's arc of the ball, whose
  chords sag up to 0.01 at deflection 0.01 and cut into it (18 open or
  non-manifold mesh edges). The shared edge pool now samples such an arc
  until its chords sag at most half the hole's clearance.

- **A ball just past tangency inside a cone's wall read as contained (CLOSED 2026-09-28; the tangent placement in `crates/operations/tests/ball_on_a_cone_axis.rs`)**:
  `make_sphere(3 / sqrt(5) + 1e-4)` at the origin inside `make_cone(3, 0,
  6)` at `z = -3` bulges 1e-4 through the wall, and the containment test's
  probes (the ball's equator, its reach along the axes and past flat faces)
  missed it, so every op dropped the 1.65e-5 sliver. A ball whose centre
  lies nearer the other solid's boundary than its radius now refutes
  containment, measured by `point_to_solid_distance`: a cone's projection
  of a point on its axis landed on the axis, and within rounding of it the
  foot left the surface (0.6 and 1.799 for 1.342).

- **A holed frustum wall mirrored meshed short on its own (CLOSED 2026-09-28; the frustum placement in the per-face check of `ball_beside_a_cone.rs`)**:
  `make_cone(3, 1.5, 6)` less `make_sphere(0.9, 32)` at `(0.5, 2, 0.5)`,
  mirrored, meshed 83.247 of 84.850 per face: its closed rims' samples
  stand in for the seam vertex up to half a step off, and the wall's `u`
  span read off them came out 0.12 radians short of the period. Every
  cylinder or cone wall that runs its seam both ways now takes its span from
  the seam's own vertex, and the rims' stand-ins sit on its copies; the
  pointed cone through a plate's edge meshes watertight turned 3.93 with it.

- **Per-face meshes of a cone or cylinder patch a section bounds filled the wrong region (CLOSED 2026-09-28; the per-face checks in `ball_beside_a_cone.rs`, `rod_along_a_cone.rs`, `cone_cut_by_a_plane_across_its_wall` and `rod_cut_by_an_oblique_plane`)**:
  a face with no hole and no notch took the grid over its `(u, v)` box:
  the lens a ball bites from a cone's wall, split by the seam, meshed 0.459
  of 0.305 and 2.312 of 2.765, a rod's patches on a cone 0.855 of 2.529,
  and the band a tilted plane leaves 56.101 of 52.074. A cylinder or cone
  patch with a NURBS or ellipse edge now meshes through the local mesher,
  as the solid mesher does, in the developed metric: in raw `(u, v)`
  Delaunay sheared a rod's wall under a tilted plane into strips longer
  than the surface (59.533 of 56.549 at a 0.3 slope).

- **A ball holding a pointed cone's apex off its axis took minutes and fell back (CLOSED 2026-09-27; `a_ball_holding_the_apex_is_exact` in `crates/operations/tests/ball_beside_a_cone.rs`, the tip checks in `cone_cut_by_a_plane_across_its_wall`)**:
  `ruling_cone_sphere` handed an apex inside the ball to the marcher, and
  `make_cone(3, 0, 6)` against `make_sphere(2, 32)` at `(0.5, 0, 2.5)`
  fell back after 70 s (the Intersect) or ran past 150 s (the Cut and the
  Fuse). With `K < 0` the roots along a generator have opposite signs, so
  every generator leaves the ball once ahead of the apex, one loop traced
  at `v = √(h² − K) − h`, its steps halved where a ball barely holding the
  apex turns it sharply; the three ops now take 3 ms and are exact. The
  Intersect's tip, bounded by that loop and the seam up to the apex, meshed
  open (the seam's run through the apex took one copy); a pointed cone's
  tip that a section bounds now gets the apex row, the developed metric,
  and the local mesher per face, which also mends a plane-cut tip's
  per-face mesh (6.99 of 11.15 before, at a 0.8 tilt).

- **The developable wall mesher gave up a pass short (CLOSED 2026-09-27; the watertight check in `a_pointed_cone_through_a_plate_edge_fuses_to_a_valid_solid`)**:
  the pointed cone through a plate's edge at `x = 0` meshed open turned
  0.79, 5.11, 5.3, 5.6 and 5.89 (102 to 152 mesh edges): the refinement of
  its notched wall in the developed metric needed 17 passes, stopped at
  16, and the wall fell back to the snap mesher's whole cone. The cap is 32.

- **A wall a section loop winds round meshed chords through the solid (CLOSED 2026-09-27; the mesh checks in `crates/operations/tests/rod_along_a_cone.rs`)**:
  a rod through a cone's wall keeps two pieces of the rod's wall the
  section loop bounds, and the loop dips between the seam's copies. Delaunay over the
  raw `(u, v)` boundary joined the loop's two sides across the dip, so the
  pieces met their neighbours watertight but read 9 to 21% short, their
  triangles cutting through the rod, and short per face as well. Such a
  wall now takes the developed metric and angular refinement a holed wall
  takes, with its seam copies placed from the seam's own vertices and a
  closed rim's stand-in for the seam vertex snapped onto them (mirrored
  after the boolean, the upper copy had sat on the loop's last sample and
  dropped a sliver). Both meshes land within a chord of every face.

- **A rod along a pointed cone, straddling its seam, fell back (CLOSED 2026-09-27; the `(1.2, 0.5)` placement in `crates/operations/tests/rod_along_a_cone.rs`)**:
  `split_periodic_face_around_seam_holes` took only a wall with two rims.
  A pointed cone's apex now stands in for the second: the seam runs up to
  it and back, and the wall's sense is read from its one rim. Exact, valid
  and watertight in every pose, the Cut's and
  the Fuse's wall meshing on its own too. The plate-edge row's `x = 0.4`
  open meshes closed with it, and four of its `x = 0` turns.

- **A ball beside a cone's wall, and a pin laid across a ball's equator, fell back (CLOSED 2026-09-27; pins `crates/operations/tests/ball_beside_a_cone.rs` and the `across` placement in `crates/operations/tests/pin_through_a_ball.rs`)**:
  three faults. Only some of the cone's generators reach a ball beside it,
  which the generator trace did not take; they form one arc of `u` whose
  ends touch the ball, and `window_cone_sphere` traces the loop over it at
  `u = mid - half cos θ`, where the roots' split changes sign with `sin θ`
  and the loop stays smooth through both touching generators. A closed NURBS
  whose in-both run wraps its start (the ball's equator splits the loop into
  windows) was trimmed at sample indices off the equator; phase FF now
  restarts it outside every window, so each window's ends are bisected
  onto the boundary. And a pointed cone's wall whose seam the hole notches
  meshed open: the mesher now walks the wall from a seam edge, takes its `u`
  span from the seam's own vertices (a closed rim's samples miss its vertex
  by up to half a step), puts each seam run on the copy its neighbouring
  sample hangs from, and unwraps each arc from that copy; the per-face
  mesher takes the wall the same way instead of gridding the whole cone.
  Exact, valid and watertight in every pose, and mirrored after the boolean.

- **A rod parallel to a pointed cone's axis through its wall fell back (CLOSED 2026-09-27; pin `crates/operations/tests/rod_along_a_cone.rs`)**:
  `algebraic_parallel_cone_cylinder` returned the section as two open
  branches meeting at both turning points. The loop winds the rod, and the
  band split of a wall a loop winds needs it closed and anchored at the
  seam, so the rod's wall stayed whole and every op fell back (the
  Intersect 2.7% off). A loop whose turning points lie inside both faces is
  now traced along the rod's rulings, each meeting the nappe once, as one
  closed curve: exact, valid and watertight in every pose for rods clear of
  the cone's seam and axis.

- **A ball bulging through a cylinder's side wall fell back (CLOSED 2026-09-27; pin `crates/operations/tests/ball_through_a_cylinder_wall.rs`)**:
  the section loop crosses the ball's chordal equator, so phase FF gives
  each hemisphere its own window of it, but `compute_seam_anchors` cut each
  window as the whole closed loop at the wall's seam crossings and handed
  each hemisphere the other's arcs too (its two inner pieces came out doubled
  and were dropped); off the seam line the two windows share both ends and
  welded into one edge. Windows are now cut at their own seam crossings, and
  a window twinned end to end is cut at its middle. Mirrored, the wall's mesh then
  lost the strip between its seam and the rims' last samples: both rims had
  handed their seam vertex to the seam runs, whose u was taken from the
  rims' span a sample short of the period; the seam's own projection now
  fixes it. Exact, valid and watertight in every pose.

- **A thin ring on a cap with a circular rim dropped out (CLOSED 2026-09-27; pin `crates/operations/tests/thin_ring_caps.rs`)**:
  the plane sampler takes a closed rim as 8 points, whose chord midpoints
  sit 7.6% of the radius inside it, and `find_point_outside_holes` stepped
  in only from those midpoints, so a ring thinner than that seeded its
  interior point in the hole and classified with it. A thin-walled tube
  (`make_cylinder(10, 10)` less a coaxial 9.5 bore, or 2 less 1.9), the
  same bore 5 deep, and a ball through a cylinder's or a frustum's cap near
  the rim all fell back. The search now also steps in from the polygon's
  vertices, which lie on the rim: exact, valid and watertight in every pose.

- **A ball on a cone's axis hung or fell back (CLOSED 2026-09-27; pin `crates/operations/tests/ball_on_a_cone_axis.rs`)**:
  the coaxial pair went to the general marcher. A ball swallowing a pointed
  cone's apex splintered its wall into a 37,379-edge wire over 109 s, and
  `unify_faces` then ran for over ten minutes; a ball through the apex ran
  past 200 s; a ball through the middle or across a frustum's wall and top
  fell back after 26 to 42 s; across the wall and base the Fuse came back
  exact but 2.5 off. Every generator meets
  such a ball at the same distances from the apex, so the section is exact
  circles (`exact_cone_sphere`), now exact in 0.00 s within 3.6e-10 of the
  section-integrated volume in every pose.

- **A tapered pin through a ball off its axis fell back (CLOSED 2026-09-27; pin `crates/operations/tests/pin_through_a_ball.rs`)**:
  `make_sphere(3, 32)` less or within `make_cone(1.2, 0.4, 10)` upright at
  `(1, 0.5, -5)` or tilted went to the general marcher, 13 pieces per face
  pair that split neither hemisphere, falling back after up to 16 s. When
  every generator crosses the sphere twice ahead of the apex, each root of
  the generator's quadratic sweeps one closed loop around the cone, now
  traced exactly in 0.01 s; the upright Intersect matches the lens-area
  truth to about 1e-9.

- **A rod through a ring's tube fell back (CLOSED 2026-09-27; pin `a_rod_through_a_rings_tube_is_exact` in `crates/operations/tests/rod_through_a_wall.rs`)**:
  `make_torus(4, 1.5)` and a rod along `y` through the tube on both sides
  of the hole went to the general marcher, which found no crossing and
  left the ring unsplit. When every ruling of the rod meets the ring the
  same even number of times, each root of the ruling's quartic sweeps one
  closed loop around the rod, now traced exactly: Cut, Intersect and Fuse
  match the truth within 1e-4 in every pose (the Cut within 1e-9
  upright).

- **A rod through a cone's or a cylinder's wall was lost or fell back (CLOSED 2026-09-27; pin `crates/operations/tests/rod_through_a_wall.rs`)**:
  a rod across a pointed cone came back uncut (the cone whole, reported
  exact), a rod poking out through a cylinder's seam came back uncut, and
  the frustum's rod fell back. The oblique cone and cylinder went to the
  general marcher (seven fragments that never chained) and are now traced
  along the cylinder's rulings, a quadratic per ruling; a closed section
  crossing a seam exactly at a sample was cut at one crossing only; and a
  section's pcurve fit on another period copy than its end UVs made the
  splitter read its loop's polygon across the whole period.

- **A mirrored cylinder's box-corner Cut and Intersect fell back (CLOSED 2026-09-27; pin `crates/operations/tests/mirrored_cylinder_corner.rs`)**:
  a mirror leaves the cylinder's frame right-handed, so its closed rims
  turn against its u; the rim splitter put their split points where the
  rims would be had they run with u, and the wall split into nothing. The
  splitter now reads a closed rim's sense from its samples on a cylinder
  as on a cone, and a wall whose clean walk misses the band that the full
  trace finds goes to the band arrangement.

- **Two ray casts, both misreading sphere faces bounded in several planes (CLOSED 2026-09-27; pins `a_ball_less_a_tool_classifies` (column), `a_ball_windowed_by_a_box_classifies`, `a_ball_within_a_thin_wedge_classifies` and `a_seam_joined_band_classifies` in `crates/operations/tests/sphere_classify.rs`, `a_point_on_a_cap_just_inside_its_rim_is_on_the_boundary` in `classify_rims.rs`)**:
  `operations::classify` read such a face as the intersection of its arcs'
  half-spaces and voted with two rays, needing both for Inside (1002 of
  2308 grid points of `make_sphere(3, 32)` less the column
  `make_box(4.5, 4.5, 10)` at `(-2.5, -2.5, -5)` misread in every pose);
  the check crate's read it by the half-space through its loop's first
  point and a polygon of chords (the ball within the box
  `[0.5, 2] x [0.5, 1.5] x [0.3, 10]` misread near its lowest corner, a
  ball band stored as two rims and a seam misread widely). All three
  operations classifiers now call the check crate's, which reads a sphere
  face bounded by lines and circles in no one plane by the parity of a
  great-circle arc to points just inside its outer wire, and tests a point
  on a plane face against the face's own lines and arcs.

- **Heal's analytic recognition kept a NURBS face's side against its new surface (CLOSED 2026-09-27; pin `recognized_faces_keep_their_side` in `crates/operations/tests/heal_recognized_side.rs`)**:
  `convert_to_elementary` swapped a NURBS surface for the recognized one
  keeping the face's flag and wire, so a patch parameterized against the
  analytic normal came back inside out: a mirrored ball converted to NURBS
  and back read a signed volume of -112.92, and even an upright cylinder's
  top disc came back facing inward (signed 20.81, the mesh open). Each face
  whose NURBS normal opposes the new surface's is now turned over, its
  flag flipped and its wires reversed so every edge keeps its sense.

- **The engine's ray cast read a point past an off-axis bore's mouth as in the bore (CLOSED 2026-09-27, re-measured; pin `both_classifiers_read_the_bores_mouth` in `crates/operations/tests/ball_and_ring_drills.rs`)**:
  a bore whose rims are curves on a sphere is read by its own wires
  (`UvTrim`, gated by `is_uv_rectangle`), not by the band between its
  rims' lowest and highest points; both classifiers read `(1, 0, 2.9)`
  outside the bored ball, upright, turned and mirrored.

- **Point classification read plane discs through chords (CLOSED 2026-09-27; pins in `crates/operations/tests/classify_rims.rs`)**:
  both public classifiers tested a ray's hit on a plane face against its
  boundary sampled into chords, so a hit within a sagitta of a round rim
  was misread (the operations classifier read 9 of 320 points just inside
  a cylinder's rim, and 13 of 1280 in the mirrored ball less the slab
  `1 < z < 2`, as Outside). A plane hit is now read against the face's own
  lines and arcs (`brepkit_math::region2d`, the face flattened by
  `brepkit_topology::planar::face_boundary_2d`), falling back to the
  polygon only for NURBS edges or a hit on the boundary.

- **Point-to-solid distance read curved faces as whole surfaces (CLOSED 2026-09-27; pins in `crates/operations/tests/point_solid_distance.rs`)**:
  the operations distance measured a cylinder, cone, sphere or torus face
  to its untrimmed surface and walked the outer shell only (a point inside
  a cylinder 1 below its top read 5, a point over a frustum's small end
  0.30 against 3, a hollow ball's cavity 4 against 3), and the check
  crate's trim test projected a curved face's polygon along a Newell
  normal and measured fallback edges as chords (a point behind a half
  cylinder's flat read 3 against 8). Both now read a face as the ray-cast
  classifier does (`face_contains`) and edges on their curves, and the
  operations distances and boundary test delegate to the check crate.

- **A slab through a major-arc face's wall returned wrong solids (CLOSED 2026-09-27; pin `a_slab_through_a_major_arc_wall_keeps_it_exact` in `crates/operations/tests/extrude_major_arcs.rs`)**:
  a face whose chord and arc (or two arcs) share both endpoints, extruded
  and sliced, lost the region between them: the assembler keys duplicate
  edges on their endpoints and welded the pair (the TERMINAL merge-key row
  above), so a plate with a major-segment hole came back valid at 7.85356
  against 9.22004 and the two-arc disc meshed open. The pave filler now
  paves at its middle each curved edge of a solid that shares both
  endpoints with another of its edges along a different path, and faces
  no section crosses take the split pieces of open curved edges as they
  take a line's.

- **The upright slab Cut of a ball read invalid (CLOSED 2026-09-27, re-measured; pin `a_ball_less_a_slab_keeps_both_pieces` in `crates/operations/tests/sphere_plane_cut.rs`)**:
  its two pieces share one shell, which the validator now reads piece by
  piece; the cut is valid, watertight and exact upright, turned and
  mirrored, clear of the equator or across it.

- **A slab through a notched wall fell back to a mesh (CLOSED 2026-09-27; pin `a_slab_through_a_major_arc_wall_keeps_it_exact` in `crates/operations/tests/extrude_major_arcs.rs`)**:
  the keyhole extruded 0.2 less the slab `z > 0.1` came back as 30 planes
  measuring 2.827083 against 2.820039. The slab's face inside the keyhole
  outline is enclosed by a loop that is not convex, and the internal-loops
  splitter sampled that piece at its loop's centroid, which the keyhole's
  chamber puts outside the loop, so the piece read Outside and was dropped.
  On a plane the sample is now taken from the loop's own polygon: its
  centroid when that lies inside, clear of the edges, else a point walked
  in from an edge.

- **Offsets and shells of analytic solids were invalid (CLOSED 2026-09-27; pins in `crates/operations/tests/offset_exact.rs`)**:
  the offset engine trimmed each face's edges on its own, so a box's
  offset faces shared no edge (24 free), and a shelled cylinder or cone
  cup came back with 32 misoriented edges (344.45 against 333.01). While
  no face collapses and no edge turns round, the offset now copies the
  input's topology (`brepkit_offset::image`): each face moves to its
  surface's analytic offset, each vertex to where its faces' offsets
  meet, each line or circle through the moved vertices. A shell is two
  such images, the inner one turned inside out, closed by a rim on each
  open face; `shell` takes it for any solid with a curved face.
- **A partial revolve of a profile touching the axis was invalid (CLOSED 2026-09-27; pins `a_profile_touching_the_axis_revolves_part_way` and `a_disc_sector_revolves_to_a_ball_part_way` in `crates/operations/tests/revolve_arc_profile.rs`)**:
  the segmented revolve copied every profile vertex at every ring and swept
  a circle from each, the ones on the axis included, so a half-turn cone
  carried zero-length edges; and a band's orientation read the radial
  direction at the band's first end, which at an apex has none, so a cone
  standing on its tip came out reversed. A vertex on the axis now stays one
  vertex and sweeps nothing, the bands beside it close as wedges, a line
  along the axis sweeps no face, and the radial direction is read at the
  band's end off the axis. A half disc (its arc ending on the axis at both
  ends) measured 0 at every angle: an arc centred on the axis now sweeps a
  sphere band, wound by chord x sweep at the arc's midpoint.

- **A revolve of a profile with a half-circle side measured wrong (CLOSED 2026-09-27; pin `a_half_circle_side_revolves_to_its_pappus_volume` in `crates/operations/tests/revolve_arc_profile.rs`)**:
  a partial revolve built every ring after the first from lines between the
  turned vertices, so an arc's later copies were chords (the end cap lost
  the half-disc, the next band's far edge read round the tube's inner half);
  each torus band's orientation was read at its chord's centre, which a
  half circle puts on the tube's core circle; and a full revolution of a
  profile wound clockwise met its rims the wrong way round. Later rings now
  carry the profile's own curves turned about the axis, the band's side is
  read at the arc's midpoint, and the rim senses follow the winding.
- **A fuse dropped a real piece of a face as a sliver and returned an invalid exact solid (CLOSED 2026-09-27; pin `a_pointed_cone_through_a_plate_edge_fuses_to_a_valid_solid` in `crates/operations/tests/rod_through_plate_edge.rs`)**:
  the pointed cone at `x = 0.4` turned 5.3 to 5.6 fused with the plate came
  back exact and invalid (7 shared edges misoriented, 201.10 against 213.05):
  its wall split into two pieces where three belong, and `BuilderSolid`
  dropped one, alone in its own open shell, as a fragmentation sliver. An
  open shell of one to three faces spanning more than 5% of the outer
  growth shell's extent now aborts the assembly instead, and the fuse falls
  back. A gate on
  misoriented shared edges is not viable: operands and accepted results
  from the tool carry them (15 io fixtures, relative to the operands).

- **Columns ending just under the ball's pole meshed open (CLOSED 2026-09-27; pin `a_column_ending_in_the_ball_keeps_the_cap_over_it` in `crates/operations/tests/sphere_box_corner.rs`, now checking every case watertight; pose sweep cases `ball | column rx-0.2 to 2.93093` and `ball rx0.35 | column 2.5 to 2.8`)**:
  the sphere face's hole round the pole joins its outer loop along a virtual
  seam, which started at the first outer sample clear of the face's other
  holes; at the foot of a wall arc climbing from the equator it ran beside
  that arc and their samples interleaved. `join_winding_hole` now starts the
  seam where it keeps farthest from every boundary sample, and moves the
  face's other holes by whole periods into the rotated loop's span (a seam
  started elsewhere left them a period outside it, unmeshed).

- **A solid with a cavity against a box ignored or dropped the cavity (CLOSED 2026-09-27; pins in `crates/operations/tests/cavity_boolean.rs`)**:
  three roots. The analytic classifiers read only the outer shell, so a
  hollow cube classified as a box and every op against a box took the
  box-pair shortcut (7600 for 7486.90); they now decline a solid with inner
  shells. A cavity shell whose faces have no corner fan (a cap and its disc,
  each bounded by one circle) or a flat one (a ball's two hemispheres,
  cornered on their equator) read as a growth shell; the flux decides it
  when no face has three corners, or when the fan is flat, another shell
  holds it and the flux reads inward. And `remove_doubled_faces` dropped the
  two hemispheres as a doubled pair; two faces of one surface that run every
  shared edge the opposite way, flags alike, are its two halves and stay.

- **A box face on the plane of the ball's chordal equator gave invalid or wrong exact results (CLOSED 2026-09-27; pin `a_box_on_the_equator_plane_keeps_a_hemisphere` in `crates/operations/tests/sphere_plane_cut.rs`)**:
  the plane met each hemisphere in the equator circle, which in `(u, v)` is
  the hemisphere's own boundary, so as a section it split one hemisphere
  into nothing and the other into the region across it (the Intersect kept
  the upper hemisphere, the box less the ball read as the whole box). Such a
  circle now sections only the plane, once (`curve_skip_faces`), and an
  unsplit curved face whose boundary centroid lies level with its sampled
  edge takes the side its wire runs on.

- **The ball less a box corner over its pole, and a column whose top runs through a turned ball's pole, meshed open (CLOSED 2026-09-27; pins `ball_less_a_box_corner_over_its_pole_meshes_closed` and `a_column_whose_top_runs_through_the_pole_meshes_closed` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball 1 | octant`)**:
  a sphere face's loop through a pole (a corner on the axis, or a wall arc
  running over the pole) gave the pole the arbitrary u its projection
  returns, so in `(u, v)` the loop cut straight across the pole's row: a
  corner on the axis meshed a sliver of the removed patch, and a wall arc
  whose samples straddled the pole read as a collar winding the axis (half
  the face's mesh missing). A circle edge running over a sphere face's pole
  now takes the pole as a shared sample, a loop through a pole runs along
  the pole's row (the step across it is the one that closes the loop), and
  the collar mesher wants its floor to wind the axis in wire order.

- **A ball against a tilted plane across its equator fell back at most turns (CLOSED 2026-09-27; `ball_cut_by_a_plane_across_its_equator` in `crates/operations/tests/sphere_plane_cut.rs` now covers planes tilted off the centre)**:
  on a hemisphere the plane's section was one arc from the seam to the seam,
  sharing both ends with the seam arc under it; `merge_duplicate_edges`
  folds such a pair, so the lune collapsed. Vertical planes escaped only
  because their arcs span more than half a turn and are split at their
  midpoint. `closed_circle_boundary_crossings` now midpoint-splits every
  span between two seam crossings, as it does a span between two hits of
  one boundary edge.

- **A ball against a half-space across its equator fell back (CLOSED 2026-09-26; pins `ball_cut_by_a_plane_across_its_equator` in `crates/operations/tests/sphere_plane_cut.rs`, `a_box_corner_below_the_equator_splits_each_hemisphere_along_one_chain` and `a_column_crossing_one_side_of_the_ball_keeps_its_corner_loops` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep cases `ball | x > 0.5` and `ball | column 4.1x6.1 from -1`)**:
  a single chain of arcs from the seam to the seam left
  `split_noseam_by_arrangement` wanting a second one, though it keeps the
  collar holding the pole and a lune past each chain, so every face crossing
  the chordal equator fell back. One chain now splits it; arcs closing into a
  loop of their own clear of the seam become a hole and a patch, and a chain
  through a pole falls back. A battery of 180 box placements against the
  ball turned 126 of its 540 results exact with none newly invalid or open.

- **A ball within a box corner holding a pole meshed open (CLOSED 2026-09-26; pin `a_ball_within_a_box_corner_meshes_its_pole_closed` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the mesher closes a sphere face holding a pole with a virtual meridian
  from its loop's first sample to the pole, and that sample could sit beside
  a steep wall arc heading for the pole (or on a wall through the axis), so
  the meridian's samples interleaved with the arc's and left the mesh open.
  The loop now starts where its meridian runs farthest in `u` from the holes
  and from every boundary sample between it and the pole.

- **A column entering the ball from below fell back (CLOSED 2026-09-26; pin `a_column_entering_the_ball_from_below_keeps_its_corner_patches` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball | column 2.05 from -1`)**:
  near each corner the lower hemisphere holds a small patch inside the
  column, bounded by two wall arcs and an arc of the floor's circle a few
  degrees long. `restrict_curves_to_faces` sampled the floor's circle too
  coarsely to see those arcs and dropped it (a closed circle now goes on
  when an arc between its exact boundary crossings has its midpoint on both
  faces), and the hemisphere's loops, open arcs that never reach the seam,
  went to the no-seam shortcut (they now take the internal-loops path).

- **A column offset and turned with two corners inside the ball fell back (CLOSED 2026-09-26; pin `an_offset_turned_column_splits_each_hemisphere_between_two_chains` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball | column -2.6,-2.2 rz0.2`)**:
  on each hemisphere the three walls near the inside corners join into one
  chain of arcs from the seam to the seam, facing the far wall's arc, and
  `split_noseam_by_arrangement` wanted three chains around the pole (a rule
  older than its lunes), so every op fell back. Two chains now split it: the
  collar holding the pole and a lune past each.

- **The ball less a column with a corner inside it fell back (CLOSED 2026-09-26; pin `a_column_with_a_corner_in_the_ball_keeps_its_collars` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball | column -2.5 to 2`)**:
  the engine's three pieces were right, but three readers misjudged the
  wedge past the corner's two walls, whose curved faces are cornered only
  on its rims: BuilderSolid's corner-fan volume read it negative and made it
  a cavity (a negative shell no other shell's box holds, whose flux reads
  outward, is now a lump), `validate_solid`'s nesting test read its sphere
  faces as the intersection of their arcs' half-spaces (it now casts with
  the engine's ray cast, `RayCastGeoms::of_faces`), and the Cut gate found
  no plane face of it outside the tool (points along its edges now count
  when clearly outside).

- **`validate_solid` rejected a face whose hole touches its outer wire (CLOSED 2026-09-26; pin `tangent_wall_fuse_configurations_stay_analytic` in `crates/operations/src/boolean/tests.rs`)**:
  a cylinder fused with a box one of whose walls is tangent to it leaves the
  box's floor a hole whose rim shares the tangent vertex with the floor's
  outer wire. The Euler check counted that hole as a loop of its own, so it
  wanted `V-E+F = 2(S-g)+1`, odd, and found 2; it now counts each face's
  boundary pieces, two wires joining when they share a vertex.

- **A column narrower than the ball fell back to a mesh (CLOSED 2026-09-26; pin `a_column_narrower_than_the_ball_stays_exact` in `crates/operations/tests/sphere_box_corner.rs`, pose sweep case `ball | column 1.8`)**:
  each wall's section circle crosses the wall's two vertical edges at four
  points 83 and 97 degrees apart, which `closed_circle_boundary_crossings`
  took for a polygon inscribed in the circle (it tested only that the hits
  were spread within a quarter of an even gap) and dropped, so the arcs ran
  past the wall and every op fell back. An inscribed boundary now also has
  every hit at one of its own vertices and every edge's midpoint within the
  circle (`boundary_is_inscribed`).

- **The check crate read trimmed curved faces through a masked grid (CLOSED 2026-09-26; pins in `crates/operations/tests/check_curved_faces.rs`)**:
  `integrate_parametric_trimmed` kept or dropped Gauss points by the outer
  wire's sampled polygon over a box from its edges' end points, and counted
  inner wires only as full-revolution sphere holes, so the check crate's
  volume was off by more than 1e-3 in 170 of the pose sweep's 210 results
  (worst 110%). A cylinder, cone or sphere face is integrated along its
  wires' own curves in `(u, v)` by Green's theorem
  (`properties/boundary.rs`): a seam term for each wire that winds the axis,
  a pole term for a face holding one, a sphere read about an axis clear of
  its wires, a NURBS edge span by span; the sweep's worst is now 9.6e-10. A
  face runs on its wire's left: a wire the other way bounds the rest.
  `operations::measure::solid_volume` takes this path
  (`exact_solid_volume`) for a solid of planes, cylinders, cones and
  spheres before it would mesh one, so its sweep error is now 7.2e-10.

- **A mirrored ball less a box corner kept the corner's patch (CLOSED 2026-09-26; pin `a_mirrored_ball_less_a_box_corner_keeps_its_region` in `crates/operations/tests/sphere_box_corner.rs`)**:
  a mirrored ball's section pcurves, computed afresh, handed the corner's
  arcs `u` windows a turn apart, so the corner's `(u, v)` polygon crossed
  itself and its interior sample fell on the ball's far side: the Cut kept
  the corner's patch and dropped the rest (13.04 where upright it reads
  103.08). A sphere piece's interior now reads its loop on the edges' own
  curves in one continuous `u` window when the loop closes there.

- **The engine's ray cast read a sphere face that planes do not bound by its flat polygon (CLOSED 2026-09-26; pin `a_ball_less_a_box_corner_reads_by_its_wires` in `crates/operations/tests/sphere_ray_cast.rs`)**:
  the upper hemisphere less a quarter (a box corner at the ball's centre)
  is neither an intersection nor a union of half-spaces, so it took the
  flat polygon through its boundary and misread points well outside the
  ball (682 of the pose sweep's grid points). It is read in `(u, v)` about
  an axis whose poles lie clear of its wires, the ray run away from the
  pole the face holds (either way for a band or a patch, and on through
  the far pole, which it also holds, for a sphere with holes), the equator
  chords standing for the great-circle arcs they project to.

- **A box less a frustum through it fell back to a mesh (CLOSED 2026-09-26; pin `a_box_less_a_frustum_through_it_keeps_both_pieces` in `crates/operations/tests/box_less_frustum.rs`)**:
  the engine's two pieces (a slab under the frustum, the box's corners
  ringing its top) were exact, but the multi-piece gate rejected a Cut
  whose piece's bounding-box centre read inside the tool, and a ring's
  centre lies in its own hole; a centre the result does not hold passes now
  when one of the piece's plane faces lies outside the tool, which no stray
  piece of the tool's interior can show.

- **The ball less a column through both poles fell back, and results in pieces read invalid (CLOSED 2026-09-26; pins `a_column_through_both_poles_keeps_its_collars` and `a_ball_less_a_turned_column_keeps_its_four_caps` in `crates/operations/tests/sphere_box_corner.rs`, `a_solid_cut_in_two_validates` in `crates/operations/src/validate/tests.rs`)**:
  `split_noseam_by_arrangement` returned only the collar and dropped the
  lunes past the walls, and split the seam at the wall arcs' crests too;
  a cap's corner-fan volume is zero (all its corners lie on its wall), so
  its shell's sign was rounding and a turned column's caps assembled as
  cavities; and `validate_solid` counted shells in the Euler term, so any
  result the engine keeps as pieces in one shell (a block cut by a slab,
  an N-way fuse with a disjoint operand) read invalid. It now counts
  connected pieces, and rejects pieces that share a vertex or nest facing
  the same way (by ray parity). Each closed section is a hole of the
  region holding it and bounds a patch of its own (a column ending in the
  ball keeps its polar cap; pin `a_column_ending_in_the_ball_keeps_the_cap_over_it`),
  seam arcs follow their own span, and the shell flux measures from the
  shell's centre.
- **The engine's ray cast read cone and cylinder faces as bands and a plane face's curved edges by chords (CLOSED 2026-09-26; pins in `crates/operations/tests/cone_cylinder_trims.rs`)**:
  a cone or cylinder face whose edges were not all rulings and axis
  circles at its two heights was read as a constant `v` band with one `u`
  gap (on a grid over `make_cone(5, 2, 10)` within the box over
  `|x|, |y| < 3`, 1053 of 22347 points misread); such a face is read in
  `(u, v)` now against its own wires, a cone face on its own nappe with
  its ray running away from the apex (the apex has no `u`, so it also
  takes no part in a band's `u` gap), and a plane face's curved edge that
  is not a circle is read as its chord and the region between them,
  crossings solved on the curve.
- **The engine's ray cast read a ball's hole cut by a column by its flat polygon (CLOSED 2026-09-26; pin `the_engine_reads_a_hole_by_its_planes` in `crates/operations/tests/sphere_box_corner.rs`)**:
  the hole's four wall circles also run below the equator, where the
  hemisphere's outer loop already ends the face, so the exact arc check
  declined it; a loop's admitted arcs are now restricted to the part of
  each circle the face's other half-space loops admit.
- **The check crate integrated a plane face through its boundary's chords (CLOSED 2026-09-26; pins `a_plane_face_reads_its_circles_exactly` in `crates/check/src/properties/face_integrator.rs`, the check-crate volume in `crates/operations/tests/extrude_major_arcs.rs`, the moved cone in `crates/operations/tests/check_face_bounds.rs`)**:
  a curved edge was read as 32 chords, so the keyhole read 5.642508
  against 5.640079 and a cone tipped with its base in `x = 1` 2.101111
  against 2.094395; a plane face's area and moments are line integrals
  along its edges' own curves now, by Green's theorem.
- **Extruding an arc past half a turn built an inside-out wall (CLOSED 2026-09-26; pins in `crates/operations/tests/extrude_major_arcs.rs`)**:
  extrude oriented a circular arc's cylinder wall by the chord against
  the radius at the arc's start, which past half a turn points back, so a
  keyhole notch or a major segment extruded into a solid that read valid
  but measured wrong with an open mesh; it reads the side of the chord
  the arc lies on now, the cylinder wall's volume readers and the
  per-face tessellator walk each arc along its own span, and
  `offset_wire` stores a clockwise loop's arc joins counter-clockwise.
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
