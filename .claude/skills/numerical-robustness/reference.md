# Numerical Robustness Reference

Companion catalog to SKILL.md. All paths and symbols verified against the current tree; refresh line positions with the given `rg` patterns, not remembered line numbers.

## §1 Tolerance and exact predicates

### The Tolerance struct

`crates/math/src/tolerance.rs` (find: `rg -n 'pub fn' crates/math/src/tolerance.rs`):

| API | Semantics | Use for |
|---|---|---|
| `Tolerance::new()` / `Default` | linear 1e-7, angular 1e-12, relative 1e-10 | the standard tolerance everywhere |
| `loose()` | 1e-4 / 1e-8 / 1e-6 | visualization only |
| `tight()` | 1e-10 / 1e-15 / 1e-14 | high-precision internal steps |
| `approx_eq(a, b)` | `abs(a-b) <= max(linear, relative * max(abs(a), abs(b)))` | coordinates; scale-aware, stays meaningful at large magnitudes |
| `approx_eq_abs(a, b)` | `abs(a-b) <= linear` | non-coordinate values, e.g. parameters in `[0,1]` where relative scaling is wrong |
| `parametric(deriv_mag)` | `linear / deriv_mag`, clamped to `[1e-15, 0.1]` | converting linear tolerance into a curve/surface parameter tolerance |
| `linear_sq()` | squared linear | distance-squared comparisons without sqrt |

Rule (also in CLAUDE.md, Key Patterns): never compare floats with `==`.

Pitfall: `approx_eq` on parameter values silently loosens the comparison when the parameter is large (relative term dominates). Use `approx_eq_abs` for parameters, or `parametric()` when the parameter tolerance should track a derivative magnitude.

### Exact predicates: when tolerance is the wrong tool

Tolerance answers "are these the same within manufacturing precision." Predicates answer "which side" with a guaranteed-correct sign. Any sign or topology decision where an epsilon can produce mutually inconsistent answers must use a predicate:

- Triangle orientation and in-circle tests during constrained Delaunay triangulation.
- Point-in-polygon and winding.
- Segment crossing tests.

Tolerating a near-zero determinant in these contexts produces non-convergent or self-contradicting triangulations: point A is left of line BC by one call and right of it by the next.

Where they live:

- `crates/math/src/predicates.rs`: `orient2d`, `orientation2d` (returns the `Orientation` enum), `in_circle`, `winding_number`, `point_in_polygon`, `orient3d`, `orientation3d`, `insphere`, plus simulation-of-simplicity variants `orient2d_sos`, `orient3d_sos` for degeneracy breaking. Backed by the `robust` crate.
- `crates/math/src/filtered.rs`: `filtered_orient2d`, `filtered_orient3d`, `filtered_in_circle`, `segment_intersection`. Fast float path with automatic fallback to exact arithmetic when the sign is ambiguous (Shewchuk-style adaptive precision); the vast majority of calls resolve in the fast path. Currently no production code calls these: the CDT calls the exact `predicates::{orient2d, in_circle}` and layers its own local fast path on top (`fast_in_circle` in `crates/math/src/cdt/mod.rs`, a float determinant with an error bound that falls back to exact `in_circle`), and mesh booleans (`crates/operations/src/mesh_boolean.rs`) call the exact `predicates::orient3d` directly. Treat `filtered.rs` as the drop-in replacement when a predicate call shows up hot in a profile. Beware name collisions: `cdt/mod.rs`, `polygon_boolean.rs`, and `polygon_offset.rs` each define their own private `segment_intersection_*` helpers that are unrelated to the filtered one.

Decision rule: computing a distance or a merged position, use `Tolerance`. Deciding a branch that must be globally consistent (orientation, containment, crossing), use a predicate.

## §2 Quantization-bucket fragility

### The scheme

Two independent quantizers exist, both named `quantize_point`, both `(coord * scale).round() as i64` triples (find both: `rg -n 'fn quantize_point' crates/algo/src/builder/`):

1. `crates/algo/src/builder/same_domain.rs`: feeds `compute_edge_set_quantized`, the same-domain face-matching edge-set key. Pairs are canonicalized (smaller vertex first) and the list is `sort_unstable()`ed: a sorted canonical form, not an order-dependent hash. Intentionally outer-wire-only (SD candidates share the outer boundary but may differ in holes).
2. `crates/algo/src/builder/builder_solid.rs`: feeds `merge_duplicate_edges`, keyed on quantized endpoint pairs at `const MERGE_TOL: f64 = 1e-7`.

### The failure mode

The pavefiller can place a vertex a few ULPs short of an exact pre-existing vertex (e.g. -11.999999 vs the exact -12.0). The two points quantize to different cells at `MERGE_TOL`, the duplicate-edge merge never unifies the two faces' partitions, the shared boundary stays open, free edges appear, and the boolean falls back to mesh co-refinement. The in-code comment near the top of `build_solid` in `builder_solid.rs` documents exactly this (find: `rg -n 'different cells at MERGE_TOL' crates/algo/src/builder/builder_solid.rs`).

Empirical signature: a knife-edge, not a smooth threshold. In one diagnosed case, translating the operand by 7e-14 or 2e-13 passed while 1e-13 failed. Non-monotonic pass/fail under tiny nudges = bucket straddle.

### The proven fix shape: weld below the bucket

`weld_coincident_vertices` (`builder_solid.rs`, find: `rg -n 'fn weld_coincident_vertices' crates/algo/src/builder/`) runs before edge splitting and merging with `snap = MERGE_TOL * 10.0`. Vertices within the snap radius are collapsed to one canonical vertex in deterministic vertex-index order, so by the time quantization keys are computed, no near-duplicates exist to straddle cells.

What NOT to do: widen `MERGE_TOL`. That relocates the cell boundaries; some other input will straddle the new ones. The snap radius must be strictly larger than the bucket so every intra-bucket-distance pair is unified before keying.

### Hash-order fragility

A hash accumulated over quantized floats in iteration order changes value when the iteration order changes, even for the identical point set. Always build a sorted canonical form first (collect, canonicalize each element, `sort_unstable`, then compare or hash). `compute_edge_set_quantized` is the in-repo template.

### The merge-key lesson (endpoint-pair keys are load-bearing)

`merge_duplicate_edges` keys edges by quantized endpoint pair only. Two workloads pull in opposite directions:

- One (a rounded-rectangle lip ring) requires a Line chord and a Circle arc with the same endpoints to collapse to one edge, despite deviating by millimeters mid-span.
- Another (a torus-minus-box lens) requires a Line and a co-endpoint arc to stay distinct, because a real face lies between them.

No local key discriminant (curve type, midpoint sample) separates the cases; the distinction is global (is there face area between the edges?). The surviving fix shape is splitter-side geometry control: give each co-endpoint seam arc a midpoint vertex so no two distinct edges share both endpoints, and the unchanged endpoint-only merge cannot collapse them. In-repo instance: `emit_split_circle_arcs` in `crates/algo/src/pave_filler/phase_ff.rs` divides arc span by `PI * 0.999`, so a diametric semicircle (span exactly pi) gets two sub-arcs and a midpoint vertex. Lesson: when a shared key mis-merges, change what you feed it, not the key.

### SD false-merge guard

Because the edge-set key is outer-wire-only, two sphere bands sharing the same equator polygon once matched falsely. Guards: `distinct_curved_regions` (far-apart interior surface samples mean distinct regions) and, for planar faces, `planar_faces_overlap` (whole-wire containment, conservative by design). Both in `same_domain.rs`.

## §3 Closed and periodic geometry

### Endpoint-only computations collapse on closed curves

`EdgeCurve::domain_with_endpoints` (`crates/topology/src/edge.rs`, find: `rg -n 'CLOSED_EPS' crates/topology/src/edge.rs`): if `(start - end).length() < 1e-9` the edge is treated as the full closed curve; otherwise both endpoints are projected onto the curve and the CCW span from start is taken via `rem_euclid(TAU)`.

Consequence: for a full circle or a closed boundary, start == end, so any computation that looks only at endpoint vertices sees zero extent. Shipped collapses, all fixed by sampling along the curve:

- A face AABB built from endpoint vertices collapsed a cylinder's closed circular edge to a line at the seam, and a full torus boundary to a single point. Current `compute_face_bbox` (`phase_ff.rs`) samples 9 points along every edge curve AND unions a surface-aware box for spheres (`sphere_region_axis` decides hemisphere extent) and full tori (gated by `face_boundary_all_degenerate`). A sphere or torus face bulges beyond its boundary edges, so even a boundary-sampled bbox underestimates; the surface term is required.
- `face_boundary_all_degenerate` distinguishes a degenerate `Line(v0, v0)` seam (zero sampled extent = full torus) from a closed Circle edge (real sampled loop = trimmed patch). Endpoint inspection alone cannot tell them apart.

Rule: winding, extent, AABB, and degeneracy tests must sample along the curve.

### Degenerate seam edges exist by construction

- `make_torus` (`crates/operations/src/primitives.rs`) builds the full torus as 1 vertex, 2 seam edges `Edge::new(v0, v0, EdgeCurve::Line)`, 1 face, wired as the fundamental polygon a, b, a-inverse, b-inverse (genus 1, V-E+F = 0). Code that assumes every edge has distinct endpoints or nonzero length breaks here.
- Cylinder and cone seams are real Lines between distinct vertices.
- `make_sphere` is 2 hemisphere faces joined by a chord-discretized equator: N vertices on the z=0 circle joined by `EdgeCurve::Line` edges. This is the chord seam of §4.

### Periodic-curve copy hazard

Copying a periodic curve (Circle, Ellipse) onto a NEW edge requires the stored start/end vertex order to match the curve's parameterization. If they disagree, `domain_with_endpoints` recovers the complementary sub-arc (the major arc instead of the minor, or vice versa). Forward edges hide the bug because stored order equals wire order; a reversed edge in the wire exposes it.

Fixed site: `reverse_edge_curve` in `crates/operations/src/extrude.rs` flips the circle/ellipse frame (normal and v-axis) so projection negates theta, and for NURBS reverses control points, weights, and mirrors knots. Guard tests: `extrude_half_circle_reversed_edge_volume` and `extrude_half_ellipse_reversed_edge_volume` in `crates/operations/src/extrude/tests.rs`.

Sweep-family exposure varies; check the curve type AND the endpoint pattern:

- `sweep.rs` and `pipe.rs` emit only `EdgeCurve::Line` edges: no hazard.
- `loft.rs` copies profile Circles onto closed seam edges (`Edge::new(seam_v, seam_v, EdgeCurve::Circle(..))`, find: `rg -n 'seam_v, seam_v' crates/operations/src/loft.rs`). Start == end, so `domain_with_endpoints` takes the full curve and the arc-order ambiguity cannot arise.
- `revolve.rs` constructs its own Circle and NURBS arc curves. Its closed rings (`Edge::new(vid, vid, ..)`) are safe for the same full-curve reason, but it also creates Circle edges between DISTINCT vertices (find: `rg -n 'v_bot, v_top' crates/operations/src/revolve.rs`). Those are exactly the hazard pattern: stored vertex order must stay consistent with the curve frame. Audit them when touched.

Rule: any edge carrying a periodic curve between distinct vertices needs a vertex-order versus parameterization audit; a closed edge (start == end) does not.

### Native-parameter trap

`evaluate_with_endpoints` (`edge.rs`) takes the curve's NATIVE parameter: radians for Circle/Ellipse, knot values for NURBS. Sampling over an assumed `[0,1]` once made a fix pass for the wrong reason (a wrong range happened to reject the bad cases). Always obtain the range from `domain_with_endpoints(start, end)` first. Related: `NurbsCurve::evaluate` (`crates/math/src/nurbs/curve.rs`) does not clamp; out-of-domain evaluation extrapolates silently, so guard t against the knot domain.

## §4 Chord versus analytic: the sagitta gap

Sagitta: the chord-to-arc midpoint deviation, `r*(1 - cos(theta/2))`. Defined and used in `crates/math/src/chord.rs`, which computes segment counts from a max deflection plus an angular cap (`DEFAULT_ANGULAR_TOL = 0.35`, about 20 degrees).

### The case study (plane x sphere)

The sphere's equator seam is stored as chords (§3). A cutting plane meets the true sphere in an exact circle. That circle bulges outside the inscribed chord polygon by the sagitta (0.05 units in the diagnosed case, far above tolerance), so segment-intersection tests against the chords found zero crossings and the split never fired (`closed_circle_boundary_crossings` in `phase_ff.rs` returned 0).

Fixing crossing detection alone was not enough: the next layer (splitting boundary chords at the crossing point) hit the same gap, because the crossing lies on the true equator, off the chord. Four attempts confirmed: every 3D-based layer downstream fights the chord seam.

Shipped fix: compute crossings against the analytic carrier, not the facets. `sphere_seam_plane_crossings` (`phase_ff.rs`) intersects the cut circle with a Newell-fit seam plane, which is facet-count independent.

### The durable architecture lesson

Do such arrangements in UV space, where the seam is exact: on a sphere the equator is the clean line v = 0, cut arcs are pcurves, and the arrangement is degeneracy-free. When you must stay in 3D, intersect against the analytic curve or surface, never its discretization.

Corroborating in-repo guard for the inverse direction (chord near arc, not crossing it): the `chord_break_on_arc` closure in `crates/algo/src/builder/face_splitter/mod.rs` rejects "phantom breaks" where a chord segment passes near a convex boundary arc; a break registers only if the crossing genuinely lies on the arc within tolerance.

## §5 Nondeterminism

### Signature and cause

Results identical within a process but varying across processes. Cause: std `HashMap` uses a per-process random seed; iteration order differs per process, and if that order feeds any branch (which face is the representative, which intersection is processed first), outputs diverge.

### Fix shape

Sort HashMap-derived iteration before it drives decisions:

- Integer/id keys: `sort_unstable_by_key`. In-repo template: `same_domain.rs`, where the comment "Sort outputs deterministically" precedes `pairs.sort_unstable_by_key(|p| (p.idx_a, p.idx_b))` (find: `rg -n 'Sort outputs deterministically' crates/algo/`).
- Float keys: `total_cmp`, e.g. `hits.sort_by(|a, b| a.0.total_cmp(&b.0))` in `phase_ff.rs`. Never sort floats with `partial_cmp().unwrap()` (NaN, and the lint denies unwrap anyway).

This class recurs; the git history is rich with instances (search `git log --oneline --grep=deterministic` for the trail, e.g. deterministic GFA iteration, deterministic same-domain merge ordering, deterministic vertex welding, order-independent coincident-face selection).

### Bisect procedure

1. Reproduce across FRESH processes (each reseeds):
   ```bash
   for i in $(seq 20); do
     cargo test -p brepkit-operations --release the_failing_test -- --exact 2>&1 | tail -1
   done
   ```
   Checkpoint: a mix of `ok` and `FAILED` lines confirms process-seed nondeterminism. All-pass or all-fail means the bug is input-shaped, not seed-shaped: go to §2.
2. Locate candidate HashMaps on the failing code path: `rg -n 'HashMap' <suspect files>`, focusing on `.values()`, `.iter()`, `.keys()`, `.drain()` whose results feed loops with side effects or early returns.
3. Instrument: `eprintln!` the iteration order per run. Two runs with different orders AND different outcomes pins the site.
4. Sort at that site, rerun step 1. Checkpoint: 20/20 identical outcomes.

Note `IndexMap` or `BTreeMap` are alternatives when insertion or key order is the natural canonical order, but a one-line sort at the consumption site is the usual minimal fix.

## §6 Prove the suspect path fires

Canonical incident: a plausible static diagnosis blamed a collinear-vertex edge-split helper for handling only Line edges. The proposed arc-handling fix was correct-looking code, but the path fired ZERO times on the failing case; the real cause was elsewhere. Hours saved by a counter, or lost without one.

Procedure before trusting any root-cause claim:

1. Add a hit counter or `eprintln!` at the suspect site (or gate the candidate fix behind an env-var flag).
2. Run the exact failing input. Checkpoint: the counter is nonzero. If zero, the diagnosis is wrong regardless of how plausible it reads; return to evidence gathering.
3. Toggle the fix (flag on/off). Checkpoint: the outcome flips with the flag. Green-with-flag-off means something else changed; you have an accidental pass.
4. Confirm the mechanism, not just the outcome. One shipped fix "worked" because a wrong parameter range rejected phantom breaks for the wrong reason; it was caught in review only because the reviewer traced the parameter semantics.

Supporting tactics (see the debugging-doctrine skill for the full discipline):

- Dump literal data at the boundary between "generated" and "consumed": in one case dumping section edges pre- and post-dedup proved a missing edge was never generated (absent pre-dedup), eliminating the dedup as a suspect in one step.
- Vary ONE variable at a time (one operand nudge, one flag, one tolerance).
- Verify geometry with the ray-cast classifier `brepkit_check::classify::classify_point` (`crates/check/src/classify/mod.rs`). Do not use tessellated volume as a correctness oracle (it once read 1.4% high and nearly masked an entirely missing cut), and do not use the winding classifier (`classify_point_winding`) on faceted or non-analytic solids; ray-cast is what the GFA builder itself trusts.
- Face count is the reliable mesh-fallback tell: an analytic boolean yields roughly 3 to 80 faces with curved types present; a mesh fallback yields hundreds to thousands of planar facets. The approximation census (`cargo run --release --example approx_census -p brepkit-operations`) reports which operations degrade analytic to approximate.

## Glossary

- **GFA**: the general boolean engine in `crates/algo` (entry `gfa.rs`): pavefiller (intersection) plus builder (face splitting and assembly).
- **Pavefiller**: the intersection orchestrator (`crates/algo/src/pave_filler/`); runs interference phases pairwise by dimension: VV, VE, EE, VF, EF, FF. The FF phase (`phase_ff.rs`) computes face-face section curves and hosts most seam and closed-curve handling.
- **Pave / PaveBlock**: a split point on an edge, and the sub-segment between paves.
- **SD (same-domain)**: detection of coincident or overlapping faces from the two operands (`builder/same_domain.rs`) so booleans keep one representative.
- **Section curve**: the intersection curve between two faces emitted by FF.
- **Seam edge**: the artificial edge closing a periodic surface: cylinder generatrix line, torus fundamental-polygon `Line(v0, v0)` pair, the sphere's chord equator.
- **PCurve**: a curve in a surface's UV parameter space; UV-space arrangements dodge 3D chord-vs-analytic gaps.
- **Mesh fallback**: when the analytic B-Rep boolean fails its validation gate (`crates/operations/src/boolean/mod.rs`), the op reroutes through tessellated co-refinement (`mesh_boolean.rs`).
- **Free edge**: an edge bounding exactly one face; any free edge means an open, non-watertight shell.
- **Sagitta**: chord-to-arc midpoint deviation `r*(1 - cos(theta/2))` (`crates/math/src/chord.rs`).
- **ULP**: unit in the last place, the spacing between adjacent representable floats at a given magnitude; about 2e-16 relative for f64. "A few ULPs apart" means as close as two distinct doubles can get.
- **Ray-cast vs winding classifier**: the two point-in-solid classifiers in `crates/check/src/classify/`; ray-cast (`classify_point`) is the trustworthy one for faceted or non-analytic solids.
