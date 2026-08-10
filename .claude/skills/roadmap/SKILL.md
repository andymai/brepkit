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

**Inventory status (2026-08-10): ONE deferred-defect pin.**
`lidpost_fuse_inmem::lidpost_fuse_is_closed_and_analytic` (#1517, see OPEN). Every
other `#[ignore]` is an explicit diagnostic or a slow-test marker. Known stale-but-harmless:
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

Where that stands: **every head-to-head bench row leads** (last full sweep 2026-08-07 on
released kernels, equivalence verified per row; the weakest rows were closed by #1389
mesh-sphere, #1396 exact all-planar volume, #1408/#1410 intersect octant). Gridfinity bin
parity is reached; the full tool generator directory is green; all four primitive-boolean
fallbacks are exact analytic. Per-PR history and per-row numbers live in git, MEMORY.md,
and the bench harness — do not re-record them here. Native criterion CAVEAT: the
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
| **Tool geometric parity: 137 failed / 2550 passed (#1517)** | First end-to-end head-to-head; prior green numbers for that suite were all on the REFERENCE kernel. Perf LEADS (0.69x aggregate, faster on 24/26, 0 non-manifold scenarios vs the reference's 5). Two roots carry the tail: (a) the compartments+scoop graze fuse behind ~45 non-watertight exports, (b) a `hashbrown` capacity-overflow panic behind 1 panic + 4 reentrancy. Count is post-brepjs#2012 (compound meshing); the #1517 opening comment quotes the earlier 152/2535. Harness `kernelParityMatrix.test.ts` + `scripts/compare-kernel-parity.ts` in the tool; refresh `.brepkit.snap` baselines first or ~110 stale counts masquerade as defects |
| **Compartments+scoop graze fuse (#1517 root a)** | ORIENTATION-dominant: raw GFA keeps the analytic faces (12 cone/22 cyl) but emits 123 unmatched directed half-edges and only 34 free edges. Divider ramp faces are SLANTED and graze the body's vertical wall within 0.034mm, against a staircase of 0.1mm sliver faces. Same class as the #1365-#1404 orientation campaign; `detect_same_domain` reports 7 within-rank duplicates (usually 0) but SD has been wrongly blamed before. Repro `crates/io/tests/compartscoop_fuse_inmem.rs` (~700ms) |
| **Lid magnet-post corner fuse (#1517 root b): the crash AND the 14 timeouts** | ONE root, not two symptoms. A post cylinder (r=4 at (-118.5,76.5), z=-2.8..-0.7) straddles a lid corner and pokes through BOTH walls x=-122 and y=80; the lid's z=-0.8 ledge cuts its cross-section into arcs. Raw GFA stays analytic (47 faces, 4 cone/17 cyl) but leaves **6 free edges**: the ledge plane and the post cylinder each keep DIFFERENT arcs of the same circle, so the two surfaces are split against each other inconsistently rather than one failing to split. Gate rejects (euler=-1) then mesh fallback turns 44 exact faces into 305 flat ones; the next ~24 post fuses compound the blob to 8852 faces; call 248 never returns, and after ~12 min the lookup table cannot grow and the kernel aborts. Repro `crates/io/tests/lidpost_fuse_inmem.rs` (**5ms**, ignored pin `lidpost_fuse_is_closed_and_analytic`); `FREE_EDGES=1 replay_pair` dumps the arcs. RED HERRING: `plane_internal_line_loops` logs 'not strictly interior' at 2.8e-14/8.9e-16/0 here, which is correct behaviour since those sections do touch the boundary. CDT constraint recovery was separately found unbounded and hardened in #1520, but it was not this root |
| **Mesh-boolean fallback emits OPEN meshes that are CONSUMED** | A product call, not just a fix: rejecting means the op fails outright. Mitigation shipped: `boolean::mesh_fallback_count()` + wasm `meshFallbackCount()` let pipelines snapshot-and-refuse |
| **Export angular default (5°) vs the reference's coarser effective default** | Tolerance-parity product choice, not mesher waste: 5° forces 18 segments/quarter-arc on r=0.6 slot corners, ~1.7x triangles vs reference at fine deflection. Revisit only as a product decision |
| **Kumiko corner-window roots (4, documented)** | Unshipped; the parked branch `fix/kumiko-corner-window-cut` is GONE from the remote with its fixtures. Re-attempting means re-capturing fixtures first |
| **Marched FF sections carry `pave_block_id=None`** | Architectural note without a live repro (the snapClip op-cut-3 case replays clean, fixture `snapclip_export_corner_inmem.rs` ACTIVE). If a new leak lands here, the canonical altitude is pave-block attachment at phase-FF/make_blocks — every face-splitter-level attempt broke calibrated chains |
| **v1 fillet deprecations entangled with the public wasm API** | `try_fillet` still reaches deprecated `fillet`/`fillet_rolling_ball`; migrating changes public behavior — a product decision, not safe cleanup. See `fillet-blend`, `wasm-bindings` |
| **crates.io / GTM items** | Andy-only. Publishing infrastructure works (see MEMORY.md for the release-please `continue-on-error` masking gotcha) |

## Closed: root cause + where the detail lives

One line each; the fixture/PR carries the story. Newest first.

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
  stale baseline has twice nearly produced a false conclusion.
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
  (same_domain), `BK_OPEN_SHELL` / `BK_SHELLS` (builder_solid shell grouping),
  `BK_SUBFACE_SRC` / `BK_SUBFACE_BOX` (builder — note BOX tests face VERTICES only),
  `POINT_IN` / `FREE_EDGES` / `TESS_BND` modes in `replay_pair`, `dump_solid` (per-wire
  edge ids), `audit_bin.rs` (HALFEDGE directed oracle — the authoritative winding
  oracle), `orient_scan.rs` / `fuse_orient.rs`, fillet instruments (`BK_FORCE_V2`,
  `BK_PIECES`, `BK_CORNER_TRACE`, `BK_TRIM_TRACE`, `BK_SPLIT_PREPASS`, `BK_NOTCH_TRACE`).

## Subsystem trap notes (crates without their own skill)

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
