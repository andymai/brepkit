# solid-verification: reference

Symbol catalog for the verification ladder. All paths and symbols verified against the current tree; if something does not resolve, re-locate it with the given `rg` pattern rather than trusting a line number.

## Ladder detail

### Rung 1: entity counts and surface census

- `brepkit_topology::explorer::solid_entity_counts(topo, solid) -> (faces, edges, vertices)`. Locate: `rg -n 'pub fn solid_entity_counts' crates/topology/src/explorer.rs`.
- `brepkit_topology::explorer::solid_faces(topo, solid) -> Result<Vec<FaceId>, _>` flattens outer plus inner shells. Always use this over walking `outer_shell()` (see CLAUDE.md, Walking faces in a solid).
- Surface census: iterate `solid_faces`, call `FaceSurface::type_tag()` (an inherent method on the enum in `crates/topology/src/face.rs`). Edge equivalent: `EdgeCurve::type_tag()` in `crates/topology/src/edge.rs`.
- Approximation census tool: `cargo run --release --example approx_census -p brepkit-operations` (source: `crates/operations/examples/approx_census.rs`). It installs a logger capturing `brepkit_approx`-target debug probes and reports, per operation, whether the result stayed analytic or which approximation path fired. Probe sites: `rg -n 'brepkit_approx' crates/operations/src/`.
- Mesh-fallback tell: a clean analytic boolean of primitives yields roughly 3 to 80 faces with quadric surface types present. A mesh fallback yields hundreds to thousands of faces, all `plane`. Face count and census are the ONLY reliable tell; triangle count and validity both look normal after a fallback.
- WASM: `getEntityCounts(solid) -> [f, e, v]`, `getSurfaceType(face)`, `getEdgeCurveType(edge)` in `crates/wasm/src/bindings/query.rs`.
- Does not prove: representation quality is not correctness. An all-analytic, no-probe result can still be missing a carved feature (recorded precedent: PR #997).

### Rung 2: validation reports (two validators, do not conflate)

**(a) operations-crate validator, prefer for boolean results.** `crates/operations/src/validate.rs`:
- `validate_solid(topo, solid)` returns the full issue list.
- `validate_solid_with_options(topo, solid, &ValidationOptions { tolerance_scale })` (scale clamped internally to [0.1, 1000]). The bare-scale form `validateSolidWithOptions(solid, tolerance_scale)` is the WASM binding only.
- `validate_solid_relaxed(topo, solid)` skips Euler, boundary-edge, non-manifold, and shell-connectivity checks for assembled geometry that legitimately does not share edges.
- `euler_characteristic(topo, solid) -> i64`.
- Its Euler check is loop-aware: V-E+F = 2(1-g) + L with L the count of inner wire loops; it errors only when the implied genus is negative or non-integral. Its edge checks come from `explorer::edge_to_face_map`: 0 faces per edge is an orphan (Error), 1 is a boundary edge ("shell is not closed", Error), 3 or more is non-manifold (Error).

**(b) check-crate validator, richer per-check report.** `brepkit_check::validate::validate_solid(topo, solid, &ValidateOptions::default()) -> ValidationReport` in `crates/check/src/validate/mod.rs`. Options: `ValidateOptions { tolerance_scale, disabled_checks: HashSet<CheckId> }`. Report API: `is_valid()`, `error_count()`, `warning_count()`. `CheckId` variants (in `crates/check/src/validate/checks.rs`): `VertexOnCurve`, `VertexOnSurface`, `EdgeNoCurve3D`, `EdgeSameParameter`, `EdgeRangeValid`, `EdgeDegenerate`, `WireEmpty`, `WireNotConnected`, `WireClosure3D`, `WireRedundantEdge`, `WireSelfIntersection`, `FaceNoSurface`, `FaceOrientationConsistency`, `ShellEmpty`, `ShellConnected`, `ShellClosed`, `ShellOrientationConsistent`, `SolidEulerCharacteristic`, `SolidDuplicateFaces`. Severities: Info, Warning, Error.
- Caveat: its `check_euler` (`crates/check/src/validate/solid.rs`) requires V-E+F == 2 exactly (Warning). It does not correct for genus or inner loops, so a legitimate solid with a through-hole warns here. Disable via `disabled_checks` or use validator (a).
- WASM: `validateSolid(solid) -> u32` error count (routes to the operations validator, count only), `validateSolidRelaxed`, `validateSolidWithOptions(solid, tolerance_scale)` in `crates/wasm/src/bindings/measure.rs`.
- Does not prove: all checks are topological or local-geometric. None compares against intent. A watertight, manifold, Euler-consistent solid can be the wrong shape.

### Rung 3: free edges and Euler (B-Rep watertightness)

- Programmatic: validator (a) errors quoted above; or check-crate `ShellClosed` ("shell has {n} free (boundary) edges", `crates/check/src/validate/shell.rs`).
- To get the actual open loops for inspection: `brepkit_heal::analysis::free_bounds::find_free_bounds(topo, shell_id) -> Result<Vec<Vec<EdgeId>>, _>`. Per shell: for hollow solids run it on the outer shell and every inner shell.
- Raw map: `explorer::edge_to_face_map(topo, solid)`; expect every edge to map to exactly 2 faces.
- JS: `edgeToFaceMap(solid)` binding in `crates/wasm/src/bindings/query.rs`; count 1-face entries.
- Does not prove: closed and manifold does not mean correctly classified. A boolean can keep the wrong face set and still assemble a perfectly closed manifold. Note a spur (a face fan sharing one edge 3+ ways) does surface here as non-manifold.

### Rung 4: mesh watertightness

- `brepkit_operations::tessellate::tessellate_solid(topo, solid, deflection) -> Result<TriangleMesh, _>` (`crates/operations/src/tessellate/solid.rs`).
- From `crates/operations/src/tessellate/mesh_ops.rs`:
  - `boundary_edge_count(&mesh)`: directed half-edges without an opposite. Expect 0.
  - `non_manifold_edge_count(&mesh)`: undirected edges referenced by 3+ triangles. Expect 0.
  - `is_watertight(&mesh)`: both zero.
- Test-suite idiom: see `tessellate_solid_box_watertight` in `crates/operations/src/tessellate/tests.rs`.
- Does not prove: mesh closure does not prove B-Rep closure (the tessellator stitches shared boundary vertices and can close over small B-Rep defects). And a watertight mesh of the wrong solid is still wrong.

### Rung 5: point classification

**check crate (`crates/check/src/classify/mod.rs`), recommended in Rust:**
- `classify_point(topo, solid, point, &ClassifyOptions::default()) -> Result<PointClassification, _>`. Analytic ray casting against face surfaces (`crates/check/src/classify/ray_surface.rs`) with UV boundary containment, three fixed irrational ray directions, majority vote with early exit, up to `max_recovery_attempts` perturbed recovery rays. `ClassifyOptions { tolerance: 1e-6, max_recovery_attempts: 10 }`. Returns `Inside | Outside | OnBoundary`. This matches what the boolean builder itself uses (`crates/algo/src/classifier/ray_cast.rs`).
- `classify_point_winding`: generalized winding number, threshold 0.5.
- `classify_point_robust`: winding first, ray-cast fallback only when winding lands in (0.4, 0.6).
- The trap: on faceted, stepped, boolean-result, or NURBS-heavy solids, winding lands confidently in the wrong bucket, so `robust` never falls back and both return Outside for interior points. This cost multiple debugging sessions chasing a wrong theory. Rule: ray-cast `classify_point` for anything you are verifying.
- Code caveat: all three walk `solid.outer_shell()` faces only. Points inside a cavity bounded by an inner shell are not tested against the cavity faces.

**operations crate (`crates/operations/src/classify.rs`), tessellation-based:**
- Same-named `classify_point(topo, solid, point, deflection, tolerance)`: ray casting against tessellated faces, so deflection-dependent.
- The WASM `classifyPoint` binding routes here with a hardcoded deflection of 0.1 (`crates/wasm/src/bindings/measure.rs`), returning `"inside" | "outside" | "boundary"`. `classifyPointWinding` and `classifyPointRobust` bindings live in `crates/wasm/src/bindings/operations.rs`; the same winding caveat applies.

Probe selection: sample the intent, not the space. Centers of every carved pocket (expect Outside), midpoints of every kept wall (expect Inside), pairs just inside and just outside each cut face. Sparse random probes miss thin slivers.

### Rung 6: volume and area

- `brepkit_operations::measure::solid_volume(topo, solid, deflection) -> Result<f64, _>` (`crates/operations/src/measure/volume.rs`).
- `solid_surface_area(topo, solid, deflection)` (`crates/operations/src/measure/area.rs`): sums per-face areas, analytic formulas where the surface type allows, tessellation fallback otherwise.
- WASM: `volume(solid, deflection)` in `crates/wasm/src/bindings/measure.rs`. The deflection clamp lives in the kernel, so preview-tuned caller deflections still produce accurate volumes.

## Volume paths

`solid_volume` tries these in order (read the function body in `crates/operations/src/measure/volume.rs`; locate with `rg -n 'pub fn solid_volume' crates/operations/src/measure/volume.rs`):

| # | Path | Symbol | Fires when |
|---|---|---|---|
| 1 | Closed form | `try_analytic_solid_volume` | Pure primitive: sphere, cylinder, cone/frustum, torus. Not box (all-planar solids fall through and are exact via tessellation). Bails on any NURBS face or any face with inner wires |
| 2 | Guarded Gauss | `analytic_faces_solid_volume` | All-analytic solid matching narrow structural guards (e.g. bored sphere with constant-v outer wire); uses `brepkit_check::properties::face_integrator::integrate_face` per face |
| 3 | Revolution | `analytic_revolution_solid_volume` | Fully analytic surface-of-revolution solid: one shared axis, concentric circular caps, no NURBS, no inner wires. Deliberately narrow so it never fires on boolean results with arc-bounded planar caps |
| 4-5 | Gated mesh | shape guards + `signed_volume_from_mesh` | Specific boolean shapes (scalloped-sphere collar, torus notch band), gated on `mesh_boundary_edge_count == 0`; falls through rather than return a leaky volume |
| 6 | Direct faces | `solid_volume_from_faces` | All-planar-triangular solids (mesh imports) |
| 7 | Per-face tess | `volume_from_direct_face_tessellation` | Faces with inner wires or reversed non-planar faces |
| 8 | Whole-solid tess | `tessellate_solid` + `signed_volume_from_mesh`, fallback `volume_from_per_face_tessellation` | Everything else. Signed tetrahedra: abs(sum of v0 dot (v1 cross v2)) / 6 |

Mental model: an analytic tier (1-3), a direct/per-face tier (6-7), a tessellation tier (4-5, 8).

### The deflection clamp

`volume_tessellation_deflection(topo, solid, requested)` computes the bbox diagonal and returns `requested.min((diag * 5e-5).max(1e-9))`. It refines coarse requests, never coarsens fine ones. The motivation (see the code comments nearby): inscribed meshes under-count curved faces by roughly 1-2% at preview deflections.

Convergence-test consequence: two coarse requests (say 0.1 and 0.01) may both clamp to the same effective deflection and trivially agree. To genuinely test convergence, request deflections finer than `bbox_diag * 5e-5`, or scale requests off the bbox.

### Why the check-crate Gauss integrator is not a general oracle

`brepkit_check::properties::solid_volume` (`crates/check/src/properties/mod.rs`, default `gauss_order: 5`) sums `face_integrator::integrate_face` over all faces with no structural guards. It mishandles trimmed periodic faces (a half-cylinder lateral integrates as the full period) and is sensitive to face orientation errors that the tessellation path's abs() tolerates. Recorded findings from the PR #959 investigation: half-disc prism 54.08 vs true 39.27; wrong results on box-sphere booleans. The engine only trusts `integrate_face` behind the narrow guards of paths 2 and 3. Known latent issue feeding this: `wire_polygon` in `crates/check/src/util.rs` samples closed circle/ellipse/NURBS edges but reduces a NON-closed arc edge to its endpoint vertex, so open arcs become chords and corrupt `check::properties` area/volume for arc-bounded planar faces.

### Divergence heuristics (recorded findings, not code facts)

- Good geometry: inscribed-mesh volume converges to the truth from below as deflection refines.
- Volume that moves as deflection refines, especially upward, signals broken geometry (self-intersection, collapsed seam, doubled faces). The PR #997 seam-collapse result read about 1.4% high and got worse with refinement.
- Volume can be plausible while the intended cut never happened. Volume never signs off a boolean alone; pair with rung 5 probes inside the supposed cut.

## Quick command crib

```bash
cargo run --release --example approx_census -p brepkit-operations
rg -n 'pub fn solid_volume' crates/operations/src/measure/volume.rs
rg -n 'pub fn classify_point' crates/check/src/classify/mod.rs crates/operations/src/classify.rs
rg -n 'pub fn (is_watertight|boundary_edge_count|non_manifold_edge_count)' crates/operations/src/tessellate/mesh_ops.rs
rg -n 'pub fn validate_solid' crates/operations/src/validate.rs crates/check/src/validate/mod.rs
```

For head-to-head numeric comparison against the reference kernel, use the brepjs harness (see the parity-benchmarking skill); do the comparison at the JS binding layer, not by re-deriving expected values by hand.

## Glossary

- **GFA**: brepkit's boolean engine (`crates/algo`): pave-filler intersection phases plus a builder that splits faces, classifies pieces, and assembles the result.
- **Mesh fallback**: when the analytic boolean fails, meshes are co-refined instead; detected by face count and census, not triangle count.
- **Census**: the `approx_census` example; per-operation analytic-vs-approximation report driven by `brepkit_approx` log probes.
- **Deflection**: max chord deviation for tessellation; smaller is finer.
- **Free (boundary) edge**: a B-Rep edge referenced by exactly one face, meaning an open shell. Mesh analogue: a half-edge with no opposite.
- **Inscribed-mesh undercount**: tessellation vertices lie on curved surfaces, so mesh volume of good geometry converges from below; upward drift under refinement means broken geometry.
- **Inner shells**: cavity shells of a hollow solid; anything walking only `outer_shell()` silently skips them.
