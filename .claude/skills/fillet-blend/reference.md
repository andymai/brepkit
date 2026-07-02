# Fillet and Blend reference

Deep catalog for the fillet-blend skill. All paths and symbols verified against the working tree. Prefer symbol names and `rg` anchors over line numbers, which drift.

## The full engine map

There are three fillet code paths and two chamfer paths. "v1 vs v2" is a simplification: the production `fillet` binding chains three engines.

### Fillet engines

- **v1 rolling-ball** `crates/operations/src/fillet/rolling_ball.rs`, `pub fn fillet_rolling_ball`. Emits real rounded blend faces: NURBS walls for curved neighbors, `FaceSpec::CylindricalFace` for straight edges. `#[deprecated(since = "2.44.0", note = "Use brepkit_blend::fillet_builder::FilletBuilder (via blend_ops::fillet_v2) instead.")]`, yet still the primary production path. G1 chain propagation is internal; `fillet_rolling_ball_propagate_g1` (`fillet/g1_chain.rs`) just delegates.
- **v1 flat-bevel** `crates/operations/src/fillet/mod.rs`, `pub fn fillet`. `#[deprecated(since = "0.8.0", note = "Use fillet_rolling_ball for true rounded fillets")]`. Replaces each edge with a flat bevel; planar neighbors only. Last-resort fallback.
- **v1 variable** `fillet/mod.rs`, `pub fn fillet_variable` with `FilletRadiusLaw` (Constant / Linear / SCurve).
- **v2 walking (blend)** `crates/blend/` wrapped by `crates/operations/src/blend_ops.rs`: `fillet_v2`, `chamfer_v2`, `chamfer_distance_angle` construct `FilletBuilder` / `ChamferBuilder`.

### Chamfer engines

- **flat-bevel chamfer** `crates/operations/src/chamfer.rs`, `pub fn chamfer` → `chamfer_core` → `FaceSpec` assembly (`assemble_solid_mixed`). Planar-only. NOT the blend engine and NOT the same code as the v1 fillet bevel. If a `chamfer` (no V2) bug is "refuses NURBS faces," it is this engine.
- **v2 chamfer** `blend_ops::chamfer_v2` and `chamfer_distance_angle` in `crates/blend`.

### The dispatcher

`crates/wasm/src/helpers.rs`, `pub fn try_fillet` (carries `#[allow(deprecated)]`). Tries engines in preference order and accepts the first whose outer shell passes `validate_shell_closed`:

1. `fillet::fillet_rolling_ball`
2. `blend_ops::fillet_v2`
3. `fillet::fillet`

If `filter_filletable_edges` drops all edges, or no engine yields a closed shell, it returns `Ok(solid_id)` unchanged. That unchanged return is Layer A of the silent no-op trap.

### Public wasm bindings (`crates/wasm/src/bindings/operations.rs`)

| JS name | Rust fn | Engine |
|---|---|---|
| `fillet` | `fillet_solid` | `try_fillet` chain (rolling-ball → v2 → bevel) |
| `filletWithEvolution` | `fillet_with_evolution` | `try_fillet` (same chain) |
| `filletVariable` | `fillet_variable` | v1 `fillet::fillet_variable` |
| `chamfer` | `chamfer_solid` → `chamfer::chamfer` | flat-bevel chamfer (planar-only) |
| `filletV2` | `fillet_v2` | v2 `blend_ops::fillet_v2` |
| `chamferV2` | `chamfer_v2` | v2 `blend_ops::chamfer_v2` |
| `chamferDistanceAngle` | `chamfer_distance_angle` | v2 `blend_ops::chamfer_distance_angle` |

`executeBatch` (`bindings/batch.rs`) mirrors this: op `"fillet"` → `try_fillet`, `"chamfer"` → `chamfer::chamfer`, `"filletVariable"` → `fillet::fillet_variable`.

## The silent no-op trap, both layers

- **Layer A, `try_fillet`.** Returns the input `solid_id` unchanged when all edges are filtered out, and again when all three engines fail the closed-shell check. No error.
- **Layer B, `fillet_solid`.** In the `else` branch after `try_fillet` errs, `filter_planar_edges` is retried; if that set is empty the binding sets `solid = solid_id` (the original) and returns `Ok(solid_id_to_u32(solid))` as success.

Both `validateSolid` and Euler pass because the input was valid. Detection: assert `(F,E,V)` and volume changed, per Step 2 of the SKILL. This is the concrete case behind solid-verification's "a passing check that proves nothing."

## v2 module roles (`crates/blend/src/`)

| File | Role |
|---|---|
| `radius_law.rs` | Radius as a function of spine arc-length (constant, linear, s-curve, custom). |
| `spine.rs` | Edge chain + arc-length parameterization (Line = linear; Circle/Ellipse/NURBS via projection). |
| `section.rs` | Cross-section: contact points, center, radius. |
| `stripe.rs` | Fillet band: contact curves + pcurves. |
| `blend_func.rs` | Blend constraint functions the walker solves. |
| `walker.rs` | Newton-Raphson walker. `WalkerConfig::max_newton_iters` default 20, `min_step` failure floor. `newton_solve` returns `None` on non-convergence. |
| `analytic.rs` | Analytic fast paths. `try_analytic_fillet` / `try_analytic_chamfer` route to typed closed-form arms (`plane_plane_fillet`, `plane_cylinder_fillet`, `plane_cone_fillet`, `plane_sphere_fillet`, `cylinder_cylinder_fillet`, `cone_cone_coaxial_fillet`, plus chamfer variants). Pairs with no arm (anything × torus, cyl × cone, perpendicular cyl × cyl, non-coaxial cone × cone) fall to the walker → NURBS blend. |
| `corner.rs`, `spherical_triangle.rs` | Vertex/corner blend where several stripes meet. |
| `trimmer.rs` | Trims neighbor faces along contact curves. `trim_face`, `trim_face_general`, private `split_edge_at`. |
| `fillet_builder.rs`, `chamfer_builder.rs` | Orchestration: spine → stripe (analytic or walker) → trim → corner → assemble. `fillet_builder.rs` has a closed-rim path with a fallback (`rg "closed-rim assembly failed"`). |

## Known open bug classes

Frame these as symptom signatures, not solved recipes.

### (a) Closed rounded-rect rim, all peak edges → free edges. Engine: v1 rolling-ball.

`crates/operations/src/fillet/rolling_ball.rs`. Filleting all peak-rim edges of a rounded-rectangle lip in one call produces free edges around the closed loop. The single lip-peak fillet was fixed by emitting cylindrical fillet faces as `FaceSpec::CylindricalFace` (preserve arcs, do not flatten to chords). The all-edges closed-loop case is still open. Signature: single-edge fillets are fine; the full-rim fillet leaves free edges.

### (b) Latent trimmer cap-face edge-sharing risk. Engine: v2 blend.

This is a code-level risk in `crates/blend/src/trimmer.rs`, not a currently-failing repro.

`split_edge_at` (called from `trim_face`) splits the trimmed face's boundary edge into two new edges and rebuilds only that face's wire. It does not update a neighbor (cap) face that also referenced the original edge. If both faces shared that edge, they can stop sharing it: each side becomes a boundary edge and the shell goes non-manifold and unclosed with the Euler count dropping. When walking or reordering `crates/blend` trim code, this is the failure it can produce.

The boolean pipeline reconciles this class of problem with `refine_boundary_edges` (`crates/operations/src/boolean/assembly.rs`); the blend path has no equivalent (blend may only depend on math + topology). There is no `reconcile.rs` in the tree, so treat a blend-side reconciliation pass as an approach to design, not existing code.

The gridfinity D5 lip is the closest real exercise of a filleted closed rim: `crates/wasm/src/bindings/gridfinity_tests.rs::gridfinity_d5_box_with_filleted_lip`. It is a LIVE, non-ignored `#[test]` that PASSES, and it guards the fix: it fillets a peak rim edge via the default `fillet` op (the `try_fillet` chain, so rolling-ball first, not the pure v2 path) and asserts the filleted lip stays watertight and genus-0 (`lip_euler >= 2`, `lip_val <= 2`), then fuses it onto the box. The test comment attributes the closure to the fillet's arc-runout closure plus arc-preserving reassembly. Do not hunt for an open non-manifold d5 bug: the test asserts the opposite. Use it as the regression guard when touching the fillet chain, and re-run it after any trimmer change to catch a regression back into the edge-sharing failure described above.

## Scope reminders

- The blend WALL of a curved-neighbor fillet is a NURBS surface with no closed form. Do not chase exact-analytic recovery of it.
- Straight-edge cylindrical fillet walls ARE analytic (`FaceSpec::CylindricalFace`, `crates/operations/src/boolean/types.rs`, assembled in `boolean/assembly.rs`); a degrade to NURBS there is a regression.
- Fillet topology and contact curves must be watertight. The open rim bug (a) is a topology defect and is in scope; risk (b) is a latent trimmer hazard, not a live failure.

## Cross-references

- `debugging-doctrine`: instrument at the symptom layer, vary one variable, start from the smallest repro.
- `solid-verification`: the no-op trap is its poster child; before/after (F,E,V) plus volume.
- `boolean-debugging`: fillets feed booleans; `refine_boundary_edges` is the reconciliation the blend path lacks.
- `testing`: regression fixtures. The d5 lip test is a live passing guard, not an ignored ready-repro.
