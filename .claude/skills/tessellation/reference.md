# Tessellation reference

Deep catalog backing SKILL.md. All symbols verified against main; re-verify with the rg patterns given (line numbers rot, symbols mostly do not).

## Mesher inventory

All in `crates/operations/src/tessellate/nonplanar.rs`. Find current set:

```bash
rg -n 'pub\(super\) fn tessellate_' crates/operations/src/tessellate/nonplanar.rs
```

| Mesher | Handles | Contract |
|---|---|---|
| `tessellate_revolution_band_shared` | Cylinder/Cone lateral face: a simple two-rim full-revolution band. No inner wires, exactly two closed rim-circle edges, everything else seam lines, matching shared-vertex counts on the two rims. | Reuses shared rim vertex ids from `edge_global_indices`; watertight by construction. Returns `Ok(false)` for anything else. |
| `tessellate_torus_notch_band` | Torus minus box notch: a kept toroidal patch wrapping the tube angle v fully, bounded by two v-wrapping seam-arc loops at the ends of a ring-angle (u) span; swept along u. | Boundary loops use shared wall vertices. Returns `Ok(false)` for any other torus face. |
| `tessellate_latitude_band_shared` | Sphere/Torus band between two full-revolution boundaries, exactly one inner wire. Two modes: constant-v latitude band (bored sphere), and a varying-v scalloped collar (box-intersect-sphere: scalloped floor plus latitude-cap ceiling). | Boundary rows reuse shared rim global ids; interior rows are fresh surface-evaluated points. Collar helpers: `collect_var_v_ring`, `build_collar_row`, `emit_aligned_quad_strip`, `stitch_rings`. Returns `Ok(false)` otherwise. |
| `tessellate_nonplanar_cdt` | Everything with a genuine simple UV boundary polygon: partial-revolution patches, NURBS faces, non-degenerate holed faces. | UV-space constrained Delaunay. Fallible; caller checks error or empty output. |
| `tessellate_nonplanar_snap` | Last resort. Meshes the face independently, snaps boundary vertices to the shared pool by 1e-6 proximity. | Crack-prone (issue #696). Never prefer it; it exists so nothing returns an empty mesh. |

Planar faces take a separate path (`crates/operations/src/tessellate/planar.rs`); this file covers nonplanar only.

## Dispatch chain

In `tessellate_face_with_shared_edges` in `crates/operations/src/tessellate/solid.rs` (find it with `rg -n 'fn tessellate_face_with_shared_edges' crates/operations/src/tessellate/solid.rs`):

```bash
rg -n 'handled_notch|handled_band|is_standard_rect|tessellate_revolution_band_shared' crates/operations/src/tessellate/solid.rs
```

- **Cylinder/Cone, "standard rect" outer wire** (at most 4 edges, all `EdgeCurve::Line` or `EdgeCurve::Circle`): try `tessellate_revolution_band_shared`; on `Ok(false)` fall to snap.
- **Cylinder/Cone, non-standard wire**: CDT; on error or zero triangles emitted, truncate the partial output back to the saved lengths and fall to snap.
- **Sphere/Torus**: `tessellate_torus_notch_band` first (torus only), then `tessellate_latitude_band_shared`, then CDT, then snap. Same truncate-on-failure discipline.

The truncation bookkeeping matters: CDT may have appended positions/normals/indices before failing. Any new fallible mesher slotted into the chain must save `positions.len()`, `normals.len()`, `indices.len()` (and `point_to_global` size where used) and truncate on failure, or the merged mesh gets orphan vertices and partial fans.

## Shared edge pool

`tessellate_solid` samples every edge once into `edge_global_indices: DetHashMap<usize, Vec<u32>>` mapping edge index to the ordered global vertex ids along that edge. Adjacent faces stitch because both reference the same ids. A structured mesher's job is to build its boundary rows FROM these ids, not from its own re-sampling of the same circle. If your mesher evaluates the surface to place a boundary vertex that an edge already owns, you have reinvented the snap path.

Entry points (`crates/operations/src/tessellate/solid.rs`):

```rust
pub fn tessellate_solid(topo, solid, deflection) -> Result<TriangleMesh, OperationsError>
pub fn tessellate_solid_with_tolerance(topo, solid, deflection, angular_tol) -> ...
pub fn tessellate_solid_for_boolean(...)   // keeps the circle curvature floor for co-refinement
pub fn tessellate_solid_grouped_with_tolerance(...)  // per-face groups, for render/face-id buffers
```

## Watertightness API

`crates/operations/src/tessellate/mesh_ops.rs`, re-exported at `brepkit_operations::tessellate`:

```rust
pub fn is_watertight(mesh: &TriangleMesh) -> bool
pub fn boundary_edge_count(mesh: &TriangleMesh) -> usize
pub fn non_manifold_edge_count(mesh: &TriangleMesh) -> usize
```

- `boundary_edge_count`: directed half-edges (a,b) lacking an opposite (b,a).
- `non_manifold_edge_count`: undirected edges referenced by 3 or more triangles. Edges used exactly once are NOT counted here; they are boundary edges, counted separately by `boundary_edge_count`. Always assert both.
- `is_watertight`: both counts zero.

### Reading a nonzero boundary count

- **Large and scattered**: a face failed to mesh (empty output swallowed) or a whole face's boundary missed the pool. Diff the per-face grouped output.
- **Small and stable across deflections, localized at cap-rim/seam corners**: structural. Two meshers disagree about vertices at a corner where a periodic seam meets a rim. Only shared-vertex-by-construction closes it; parameter tuning relocates it.
- **Exactly proportional to a ring's segment count**: an entire rim failed to weld (off-by-one segment count between rim and pool). Check who sampled that circle and with what count.

### Orientation check

No single-call API. Pair watertightness with signed volume: the divergence-theorem integral over the mesh (see how `crates/operations/src/measure/volume.rs` consumes tessellation) must match the analytic volume in sign and magnitude. Sign wrong: global winding flipped. Magnitude high: a hole skinned over. Magnitude low: a face missing or an inverted run canceling volume.

## Closed-edge sampling detail

`crates/operations/src/tessellate/edge_sampling.rs`:

```bash
rg -n 'is_closed' crates/operations/src/tessellate/edge_sampling.rs
```

`circle_param_range` returns the full `(0, TAU)` range when `edge.is_closed()`; `sample_edge` does the equivalent for closed Ellipse edges. Without this, start == end makes the param range zero and the edge contributes one point.

The parallel lesson boolean-side: `crates/algo/src/pave_filler/phase_ff.rs` (`sphere_region_axis`, `face_boundary_all_degenerate`) samples several points along each boundary curve instead of trusting endpoints, because a degenerate seam `Line(v0, v0)` has zero endpoint extent yet spans the full period, and a closed Circle is a real loop. Anywhere an edge can be closed or degenerate: sample along the curve.

## orient_triangle_run mechanics

`fn orient_triangle_run` in `nonplanar.rs`. Contract: the run appended from `idx_start` onward is already wound coherently by construction; the function only decides whether that single orientation needs a global flip. It scans for the largest-area triangle (best-conditioned normal), projects its centroid to (u,v), compares the triangle's geometric normal against the surface outward normal there, and if they disagree swaps indices t+1/t+2 for every triangle in the run.

Why not per-triangle: thin stitch triangles bridging a clustered ring to an evenly sampled ring have near-degenerate normals; per-triangle correction flips neighbors inconsistently and produces a non-manifold fan. `emit_aligned_quad_strip` deliberately emits raw with no winding correction for exactly this reason; the caller orients the whole strip afterwards.

If a whole face comes out inverted, suspect the flip decision itself: the largest triangle can still be a sliver on pathological faces, or the centroid projection can land across a seam. Check the (u,v) the centroid projects to.

## Crossing seam holes (known-terminal, do not grind)

The shape: a full-revolution periodic wall (u = 0 identified with u = 2pi) whose two hole loops cross each other, meeting in a figure-eight that pinches at two points. Canonical producer: the union of two equal-radius perpendicular cylinders (Steinmetz configuration); the lateral wall of either cylinder carries both intersection loops and they cross.

Why every current path fails:

1. Two crossing hole loops are invalid hole topology for any flat CDT-with-holes; the loops are not disjoint simple polygons.
2. The wall is periodic, so flat-strip CDT (unwrap u, weld the strip edges) still cracks at the seam corners against separately meshed caps (the doctrine failure from SKILL.md).
3. The torus-notch arc trick does not transfer: that band is bounded, non-periodic in the swept direction, and its loops do not cross.
4. Chasing exact intersection curves makes topology worse near the pinch, not better: as the loops approach the true crossing the mesh becomes genuinely non-manifold at the pinch points.

What a real fix requires (bounded scope, neither exists today):

- A **face-split-at-pinch primitive**: split a face at a boundary self-intersection vertex into simple-loop lobe faces on a periodic wall. This is a geometric/topological operation on the B-Rep, upstream of tessellation.
- Or a **periodic-seam-aware holed-band tessellator** that natively resolves two crossing loops on the unrolled periodic domain.

Do not confuse with `crates/heal/src/upgrade/split_self_intersecting_wires.rs`: that splits figure-eight WIRES (same-vertex-twice pinch artifacts from the mesh-fallback assembler) into simple sub-cycles. It rearranges wire topology only, creates or removes no edges, leaves outer wires alone, and does not perform the geometric face split above.

Practical stance: the analytic B-Rep and exact volume for such shapes can be correct while render meshing stays deferred (see git history around PR #1008). If asked to "just make it render", scope the ask to one of the two bounded fixes or decline.

## GPU compute mesher (brepkit-render)

`crates/render/src/compute_mesh.rs`. Cylinder only at time of writing; verify current coverage:

```bash
rg -n 'Descriptor|pub fn render_' crates/render/src/compute_mesh.rs
```

Key symbols:

- `CylinderDescriptor`: packed analytic surface params (axis frame, radius, axial range) uploaded to the GPU. Extracted from a `FaceSurface::Cylinder` face by `extract_cylinder_descriptor` (errors on non-cylindrical or degenerate-axial faces).
- `TessFactor` and `screen_space_tess_factor`: per-frame `n_u` from projected pixel radius using the chord-error bound `r * (1 - cos(pi / n_u)) <= target_px`; `n_v = 1` because a cylinder lateral face is ruled.
- `render_cylinder_compute_offscreen` and `render_cylinder_compute_screen_lod`: end-to-end paths; the WGSL kernel writes vertices at the render pipeline's vertex stride so one buffer binds as STORAGE then VERTEX.

Doctrine consequence: the GPU path consumes analytic surface PARAMETERS, not triangles. A face degraded to NURBS or mesh by an upstream operation is invisible to this path forever. When choosing between an analytic-preserving and an approximating implementation of an operation, the GPU mesher is a second consumer arguing for analytic (see the **analytic-preservation** skill). Audit current degradation points:

```bash
cargo run --release --example approx_census -p brepkit-operations
```

For verifying rendered output on the live display, see the **render-verify** skill.

## Glossary

- **Seam**: the u = 0 identified with u = 2pi closure line on a periodic surface; in UV the surface is a rectangle whose left and right edges are the same 3D line.
- **Band**: a face on a periodic surface bounded by two full-revolution loops (rims). Latitude band: boundaries at constant v. Notch band: boundaries wrapping v, swept in u.
- **Shared edge pool**: `edge_global_indices`, the once-per-edge sampled vertex ids that all faces must reference on their boundaries.
- **Snap path**: legacy fallback that meshes a face independently and welds by proximity; crack-prone.
- **bd=0**: `boundary_edge_count(mesh) == 0`, the primary watertightness regression signal.
- **Deflection**: max chord deviation (sag) driving sampling density, paired with an angular tolerance (`DEFAULT_ANGULAR_TOL` in `brepkit_math::chord`).
- **Mesh fallback**: when a boolean's analytic result fails its gates, the operation re-runs as a tessellate-then-co-refine mesh boolean; output is hundreds of planar faces. Face count is the tell (a handful analytic vs hundreds planar). See **boolean-debugging**.
