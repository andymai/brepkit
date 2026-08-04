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

Where parity stands: gridfinity bin parity reached; all four primitive-boolean
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
| **Thick-wall cavity — ROOT FIXED and MEASURED; a residual remains and the "more correct" fix is REFUTED** | operations/shell | See the dedicated entry below |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** (open since 2026-07-16) | operations | `mesh_boolean_fallback` warns its output is not a closed 2-manifold and uses it anyway; there is no further fallback, so rejecting means the op fails outright. A product call, not just a fix |
| **Divider residual geometry** (3 cases, re-confirmed 2026-08-01 with the compound + shell fixes overlaid: 3 failed / 12 passed, 820 s) | algo/GFA | `mitsukude lattice on dividers` (**boundary 6** — was non-manifold 4 before the shell fix, so the signature MOVED), `kumiko dividers perforate the compartment walls`, `dividers + scoops keep the ramp footings solid` (boundary 4). The first two are the SAME 2x2x6 mitsukude bin with 2x1 compartments differing only in `dividers: true/false`, which argues against the divider carry-through being the cause; `__kernel-tests__/mitsukudeNmProbe.test.ts` isolates pattern vs compartments vs footprint |
| **snapClip 0.6 mm-nozzle export chain** breaks at op-cut-3 (posBad=10, analytic but leaky) | algo/GFA | Root is marched FF sections on curved faces carrying `pave_block_id=None`, bypassing the pave machinery. Canonical altitude is pave-block attachment at phase-FF/make_blocks; every face-splitter-level attempt broke calibrated chains |
| **algo ray-cast classifier has no `Torus` arm** — torus faces fall to the flat Newell-polygon fallback | algo | PREVENTIVE, no failing repro: re-probed 2026-08-03, every torus landscape green (lib torus tests, `parity_boolean_curved`, `cut_torus_by_box_notch_is_analytic_watertight`, census `torus − box` analytic). Rank below repro-backed work. If built: the quartic ray×torus machinery exists in `check/src/classify/ray_surface.rs` but algo cannot depend on check — the solver would have to move to math |
| **v2 trimmer residuals** (4, all evidenced by the regress test's 12 free edges) | blend | Keep-side selection is degenerate under tangency; `create_blend_face` builds its own contact edges; no end-cap notch trim; `chamfer_v2` solves the external tangent branch. `crates/operations/tests/regress_blend_trim_neighbor_split.rs` |

The remaining `#[ignore]` entries are diagnostics or slow perf runs, not open bugs: the
`profile_intersect.rs` box-sphere probes are stale (box-sphere shipped analytic in #1006),
`staircase_fuse_with_cylinders` is a ~2 min perf run, and the two `#696` dovetail entries plus
`diverge_first_cut` are print-only.

## Thick-wall cavity: fixed, measured, and one refuted "improvement"

**Root (fixed, `fac84c7f`):** `shell_op` SILENTLY DROPPED a corner cylinder whose radius the
thickness swallowed — the `new_radius > tol` arm had no else, while the Sphere arm right below it
errors for the same condition. The two neighbouring walls then kept their original tangent extent
and overshot each other by `thickness − radius`, leaving a sub-tolerance chamfer. The body stayed
closed and manifold so nothing complained, but the next fuse aborted `open hole shell with 9 faces`,
fell to the mesh fallback, and that fallback's open output poisoned every later fuse.
Fix: feed a collapsing fillet's vertices BOTH extreme normals so the miter resolves the corner.
Pinned by `shell_thickness_past_corner_radius_gives_a_sharp_corner` (differentially verified).

**Measured end-to-end** (exported STL boundary edges, 1x1x10 halfSockets unless noted):

| case | before | after |
|---|---|---|
| wall 3.8 / 3.9 / 4.0 | 149 / 159 / 160 | **23 / 19 / 17** |
| 2x2 wall 4 / wall 6 | 832 / 898 | **20 / 20** |

Everything at or below wall 3.7 was and stays 0. So this is a large improvement, NOT a closure —
a residual remains at every thickness past the corner radius.

**REFUTED, do not re-implement: placing the collapsed corner EXACTLY.** The miter is a
TRANSLATION (`inner = outer − t·m`), so a tangent vertex that does not already sit on the far
neighbour's plane lands short of the corner — the shipped fix leaves the walls ending at
`half − thickness − radius` rather than `half − thickness`. Solving for the true corner from the arc
CENTRE instead (`nᵢ·(x−C) = radius − thickness`, verified against the measured 16.95) IS
geometrically right and IS worse in practice: wall 3.8/3.9 close to bnd=0, but 2x2 wall 4 goes
20 → **318** (nm 171, 46 s → 150 s) and 2x2 wall 6 goes 20 → **544** (nm 197, 366 s). Implemented,
measured, reverted. Whatever is wrong is in how the two extreme normals are chosen on a bin with
many corners — check that before assuming the formula.

**Residual next step:** the 17-23 boundary edges left at wall ≥ 3.75. Capture is cheap
(`wallThickCapture`, 6 booleans, `replay_pair`); the fixture
`crates/io/tests/thickwall_sharp_cavity_fuse_inmem.rs` holds PRE-FIX operands, so it stays
`#[ignore]`d and replaying it does NOT test the fix — re-capture before using it.

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

## Live campaign: kumiko / goma

**State:** branch `fix/kumiko-corner-window-cut` closes four real engine defects with fixtures
(`crates/io/tests/kumiko_corner_window_inmem.rs`, 5 tests) but is **PARKED and must not ship** —
it regresses goma from 8 to 65 exported boundary edges and 540 s to 817 s.

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
- **Current baseline: 62 failed / 372 passed of 435** (families: divider 15, kumiko 14, floor 11,
  permutation 6, solid cutouts 3, then singles). The divider/floor fix should take this to ~45 once
  released — NOT yet re-measured as a full matrix.
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
