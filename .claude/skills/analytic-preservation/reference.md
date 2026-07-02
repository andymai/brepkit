# Analytic Preservation: Reference

Deep catalog for the analytic-preservation skill. All symbols verified against the
current tree; cite symbol plus rg pattern, never line numbers.

## 1. Degradation probe sites and the census

Every degradation site carries a permanent `log::debug!(target: "brepkit_approx", ...)`
probe. The census example (`crates/operations/examples/approx_census.rs`) installs an
in-process logger filtering on that target and drains events per operation.

Probe sites (`rg -l 'brepkit_approx' crates/`):

| File | Path that degrades |
|---|---|
| `crates/operations/src/boolean/mod.rs` | Boolean mesh (co-refinement) fallback. The ONLY path that loses analytic surface types entirely. `rg -n 'analytic surface types will be lost'` |
| `crates/blend/src/fillet_builder.rs` | Newton-Raphson walker NURBS blend when analytic fast paths decline |
| `crates/blend/src/chamfer_builder.rs` | Chamfer v1 unsupported surface (instrumented hard error, no walker fallback) |
| `crates/offset/src/offset.rs` | Sampled-NURBS surface refit for `FaceSurface::Nurbs` input |
| `crates/operations/src/fillet/rolling_ball.rs` | Rolling-ball planar corner patch |
| `crates/operations/src/offset_face.rs` | Per-face offset degradation |
| `crates/operations/src/offset_trim.rs` | Grid-sampling offset trim |

Census sections run by `main()`: `boolean_matrix()`, `offset_matrix()`,
`nurbs_section()`, `revolve_matrix()` (prints a per-solid surface-type breakdown),
`blend_matrix()`, `remaining_paths()` (deliberately fires all seven paths).

Output per row: `<op> <name> <ms> faces=<n>` then either `exact analytic` (no probe
fired) or `FALLBACK xN: <deduped probe texts>`. Errors append `[ERR]` to the name with
the error as a pseudo-event.

The boolean gate that decides analytic vs fallback (`crates/operations/src/boolean/mod.rs`,
`rg -n 'open_shell_ok'`): the GFA result is accepted only if
`euler_ok && open_shell_ok && validate_boolean_result(topo, result).is_ok()`; otherwise
`mesh_boolean_fallback` runs. So a fallback is usually a symptom of a GFA correctness
failure upstream (see the boolean-debugging skill), not a reason to relax the gate.

## 2. Exact analytic-analytic intersection arms

File: `crates/math/src/analytic_intersection.rs`. Inventory
(`rg -n 'pub fn (exact_|intersect_)' crates/math/src/analytic_intersection.rs`):

| Symbol | Case | Notes |
|---|---|---|
| `exact_plane_analytic` | plane vs any, dispatch over `pub enum AnalyticSurface` | calls private `exact_plane_cylinder` / `exact_plane_sphere` / `exact_plane_cone` |
| `intersect_plane_analytic` | plane vs any, public dispatch over `AnalyticSurface` | thin dispatcher to the `intersect_plane_*` variants below; used by the offset engine (`crates/offset/src/inter3d.rs`) |
| `intersect_plane_cylinder` / `intersect_plane_sphere` / `intersect_plane_cone` | public plane-x variants | exact circles/ellipses/lines |
| `intersect_plane_torus` | plane vs torus | STILL a 128x128 grid march plus fit, not the exact quartic; the FF phase compensates by snapping arc endpoints to exact crossings |
| `intersect_line_torus` | line vs torus, real quartic roots | up to four `t` roots; the FF-phase section trimmer uses it to snap a plane-torus oval's endpoints to the exact box-edge crossings (caller in `crates/algo/src/pave_filler/phase_ff.rs`) |
| `exact_cone_cone` | coaxial only | defers (`None`) otherwise |
| `exact_cone_cylinder` | coaxial only | the gridfinity lip knife-edge case |
| `exact_sphere_cylinder` | coaxial only | template arm; test `exact_sphere_cylinder_non_coaxial_defers` proves deferral is deliberate |
| `intersect_analytic_analytic` / `_bounded` | generic marcher entry | returns fitted NURBS; the thing exact arms exist to bypass |

**Adding a new arm (procedure):**

1. Follow the `exact_sphere_cylinder` template: gate to the tractable configuration
   (coaxiality, perpendicularity, tangency class), return `Option<...>`, `None` defers
   to the marcher.
2. Add a deferral test alongside the exact-case test (pattern:
   `exact_sphere_cylinder_non_coaxial_defers`).
3. Wire the arm into `compute_raw_curves`
   (`rg -n 'fn compute_raw_curves' crates/algo/src/pave_filler/phase_ff.rs`) so it
   emits `RawCurve { curve: EdgeCurve::Circle(...), .. }`, matching the existing
   cone-cone / cone-cylinder / sphere-cylinder arms there.
4. Verify: census row for the new pair flips to `exact analytic`, AND run
   solid-verification (volume, free edges, manifold). Checkpoint: face count should be
   single digits, not hundreds.

Note: `crates/math/src/analytic_intersection.rs` is one of the ripple sites when
adding a `FaceSurface` variant (see CLAUDE.md, Ripple-Effect Checklists).

## 3. Doctrine items with code evidence

### 3a. Closed rims are `EdgeCurve::Circle`, never closed NURBS

Emission: the exact arms in `compute_raw_curves` (`crates/algo/src/pave_filler/phase_ff.rs`)
build `RawCurve { curve: EdgeCurve::Circle(circle), .. }`. Consumers in the same file
pattern-match the variant:

- Closed-circle split: `if let EdgeCurve::Circle(circle) = &raw.curve` leading to
  `closed_circle_boundary_crossings` and `emit_split_circle_arcs`.
- Seam adoption: the `adopted_seam` match keys on `EdgeCurve::Circle(c)`
  (`rg -n 'adopted_seam' crates/algo/src/pave_filler/phase_ff.rs`).
- `restrict_curves_to_faces` keeps closed loops whole ONLY for Circles:
  `!matches!(raw.curve, EdgeCurve::Circle(_))` gates the wrap-trim path. A closed NURBS
  in the same position gets wrap-trimmed and has produced out-of-domain evaluations.

Consequence: an exact rim that arrives as a closed NURBS silently loses seam adoption
and whole-loop handling. Emit the Circle.

### 3b. Prefer exact arms over the marcher

The marcher fragments a clean circle into micro-curve pieces that then need fitting,
trimming, and dedup, each a failure source. Section 2 is the procedure.

### 3c. Watertight seams: both operands split at the SAME exact crossing vertices

- `emit_exact_arc` (`rg -n 'fn emit_exact_arc' crates/algo/src/pave_filler/phase_ff.rs`)
  resolves endpoints through the shared-crossing registry first, so the adjacent
  face's arc ending at the same crossing reuses the same vertex.
- The plane-torus oval trim snaps arc endpoints to the exact box-edge/torus crossings
  so the box wall trimmed against the same edge shares those vertices.
- Lens arcs are emitted as ONE shared FF section so both consumers (the kept band and
  the wall sub-face) see the same midpoint vertex.

Symptom of violating this: free edges along the seam after the boolean, even though
both faces look correct in isolation. Check with the solid-verification skill.

### 3d. Splitter-side midpoint split for co-endpoint seam arcs

Problem: a diametric semicircle pair, or a chord plus arc, share BOTH endpoints.
`merge_duplicate_edges` (`rg -n 'fn merge_duplicate_edges' crates/algo/src/builder/builder_solid.rs`)
groups edges by quantized endpoint pair ONLY, so co-endpoint arcs collapse to one edge.

That endpoint-only behavior is load-bearing in BOTH directions:

- The gridfinity lip ring NEEDS a Line chord and a Circle arc with the same endpoints
  (deviating up to ~2.4 mm) to collapse to one edge.
- The torus lens NEEDS a Line and a co-endpoint arc to stay distinct.

Same local configuration, opposite required outcome. No merge-key discriminant can
exist: a type-only key and a deviation-threshold key were both tried and both
regressed gridfinity to non-manifold.

**Doctrine: control the geometry you EMIT (splitter-side); never make the shared merge
smarter.** Implementations in `phase_ff.rs`:

- `emit_split_circle_arcs` computes
  `n_sub = (arc_span / (PI * 0.999)).ceil()` (`rg -n 'PI \* 0\.999' crates/algo/src/pave_filler/phase_ff.rs`)
  so any span of pi or more gets a midpoint vertex and no two sub-arcs share both
  endpoints.
- Torus lens arcs are always split at their midpoint into two sub-arcs with a distinct
  middle vertex.

### 3e. Sequential booleans preserve analytic surfaces

Fixture: `crates/io/tests/gridfinity_wallcut_seq_inmem.rs`, a `Cut(Cut(body, c0), c1)`
on a captured real-world bin. Historically the second cut dropped a coincident
same-domain wall, produced free edges, and the next cut mesh-fell-back, losing the
analytic lip cones and corner cylinders. Assertions to copy for any new sequential
fixture: `free == 0`, `over == 0`, and `curved_count(...) >= N` with a message naming
the surfaces that must survive.

Capturing operands: the `serializeSolid` wasm binding exports live-tool solids into
faithful in-memory fixtures (pattern used by all `crates/io/tests/*_inmem.rs`).

Durable lesson: closing a curved boolean is three fixes, not one. The boolean fix, the
curved-band tessellation, and the measurement. The analytic split EXPOSES downstream
tessellation and volume gaps that the mesh fallback used to mask. Budget for all three.

## 4. Scoping filter table

Conceptual test: does the operation's output face lie on an INPUT surface (trimmed
patch, always closable) or on a NEW surface it constructs (no closed form)?

| Operation | Verdict | Evidence |
|---|---|---|
| Booleans on analytic solids | CHASE | Result faces are trimmed input patches; GFA plus exact arms |
| Revolve | CHASED, done | `revolution_band_surface` (`crates/operations/src/revolve.rs`): oblique line -> Cone, perpendicular line -> Plane cap, parallel line -> Cylinder, circular arc -> Torus band. Exact volume: `analytic_revolution_solid_volume` and `planar_cap_signed_volume` in `crates/operations/src/measure/volume.rs` (note: the volume helpers live in measure, not revolve). The volume gate is tight on purpose; a loose guard regressed unrelated boolean caps |
| Fillet/chamfer beyond analytic fast paths | DO NOT CHASE | `try_analytic_fillet` / `try_analytic_chamfer` (`crates/blend/src/analytic.rs`) decline torus pairs, cyl-cone, perpendicular cyl-cyl, non-coaxial cone-cone; the walker's blend surface has no closed form |
| Offset/shell of analytic input | Already exact | All five analytic types stay exact; only `FaceSurface::Nurbs` input degrades (`crates/offset/src/offset.rs` probe) |
| Offset of NURBS | DO NOT CHASE | General NURBS offset has no closed form |
| Sweep/loft side faces | DO NOT CHASE | NURBS by construction |
| Extrude of ellipse arcs | Type-system limit | There is no elliptical-cylinder `FaceSurface` variant; not a bug |

Narrowness rule: every merged exact arm is gated (coaxial-only, one specific
configuration) and defers otherwise. Ship the narrow case; leave the marcher as the
general fallback.

## 5. Terminal and open cases

### Perpendicular cylinder union render: TERMINAL

The exact Steinmetz seam is two ellipses touching at two points. On each cylinder wall
the hole boundary is a figure-eight pinching to a point, so the true union boundary is
genuinely non-manifold at the touch points (odd Euler count is correct, not a bug).
The shipped B-Rep is manifold only because its marched-NURBS seam loops sit about 0.11
apart and dodge the touch. Pursuing exact curves here makes the topology WORSE.
Confirmed over many measured attempts. Missing primitives that would reopen it: a
face-split at a boundary self-touch pinch on a periodic wall, or a periodic-seam-aware
crossing-holes tessellator. The shipped state (analytic faces plus exact closed-form
volume, render deferred) stands. `exact_cylinder_cylinder` does NOT exist in the
codebase; the prototype was reverted.

### Plane through sphere across the seam: SOLVED, reuse the pattern

Why it was hard: the sphere's seam wire is a chord polygon inscribed in the seam
circle. Exact section arcs land off the chords by the sagitta, so chord-crossing tests
find nothing. Fix (merged, PR #1006 as a git-history pointer):

- `sphere_seam_plane_crossings` (`rg -n 'fn sphere_seam_plane_crossings' crates/algo/src/pave_filler/phase_ff.rs`):
  Newell-fit plane intersected with the exact seam circle, facet-independent.
- `split_noseam_by_arrangement` (`rg -n 'fn split_noseam_by_arrangement' crates/algo/src/builder/face_splitter/special_cases.rs`):
  reconstructs the seam as its exact circle, splits it at the crossings, and traces
  the seam-arcs plus section-arcs arrangement in UV.

Portable lesson: never test exact curves against discretized boundary chords.
Reconstruct the exact boundary curve and intersect analytically.

### Genuinely open (low priority, fine to leave)

- `intersect_plane_torus` is a grid march plus fit, not the exact quartic.
- `crates/io/tests/dovetail_cornerclip_intersect_inmem.rs` is ignored: coincident-wall
  plus analytic-corner intersect drops the boundary.
- Revolve follow-ups are over-segmentation only (pointed-cone apex periodic merge,
  annulus-cap merge, partial-turn torus); results stay analytic and exact.
- Some older comments in `phase_ff.rs` describe mesh fallbacks that later fixes
  closed. Verify against a census run before trusting any "falls back" comment.

## 6. Glossary

- **Analytic surface/curve**: exact parametric primitive
  (`FaceSurface::Plane/Cylinder/Cone/Sphere/Torus` in `crates/topology/src/face.rs`,
  `EdgeCurve::Line/Circle/Ellipse` in `crates/topology/src/edge.rs`) vs the `Nurbs`
  variants.
- **GFA**: the native boolean engine (`crates/algo/src/gfa.rs`): intersect, split,
  classify, reassemble.
- **PaveFiller / FF phase**: GFA's intersection orchestrator
  (`crates/algo/src/pave_filler/`); `phase_ff.rs` computes surface-surface section
  curves. Nearly all rim and seam doctrine lives there.
- **Mesh fallback**: the boolean's last resort; tessellate both operands and
  co-refine triangles. All-planar result, analytic types lost.
- **SD (same-domain)**: detection that faces from opposite operands lie on the same
  surface region; mis-detection drops or duplicates walls.
- **Seam edge**: the u=0 equivalent to u=2pi boundary of a closed periodic face.
- **Marcher**: the generic numeric surface-surface intersector; returns fitted NURBS.
- **Free edge / watertight**: a boundary edge used by exactly one face; watertight
  means zero free edges. The cheap correctness proxy fixtures assert.
- **Splitter-side split**: giving an emitted section arc its own midpoint vertex so
  the endpoint-keyed duplicate-edge merge cannot collapse co-endpoint arcs.
- **The reference kernel**: the incumbent C++ CAD kernel brepkit benchmarks against.
  Head-to-head harness lives in the sibling repo `~/Git/brepjs` (see the
  parity-benchmarking skill).
