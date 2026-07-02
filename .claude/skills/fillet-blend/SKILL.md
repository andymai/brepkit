---
name: fillet-blend
description: Diagnose fillet and chamfer bugs by locating which engine owns them, debug the v2 walking blend, and avoid the silent no-op success in the public fillet binding. Use when a fillet or chamfer leaves free edges, produces an open or non-manifold shell, silently changes nothing, degrades a cylindrical wall to NURBS, or when deciding whether to touch the deprecated v1 fillet code.
---

# Fillet and Blend

The full engine map, module roles, and open-bug catalog live in [reference.md](reference.md).

## When to use

You are working on `fillet`, `chamfer`, `filletV2`, `chamferV2`, `chamferDistanceAngle`, or `filletVariable` (wasm), or on `crates/operations/src/fillet/`, `crates/operations/src/chamfer.rs`, `crates/operations/src/blend_ops.rs`, or `crates/blend/`. The result leaves free edges, opens the shell, silently returns the input, or you must decide whether the deprecated v1 code is safe to change.

## Quick reference

| Symptom | First move | Detail |
|---|---|---|
| `fillet` "succeeded" but shape looks unchanged | Compare (F,E,V) and volume before/after; equal = silent no-op | Step 2 |
| Bug in default `fillet` / batch `fillet` | It chains THREE engines; find which one produced the result | Step 1 |
| Bug only in `filletV2`/`chamferV2` | Pure `crates/blend` walking engine | Step 4 |
| `chamfer` (no V2) rejects curved faces | Flat-bevel engine in `chamfer.rs` is planar-only, not blend | reference.md |
| Full closed rim fillet leaves free edges | Known open bug (rolling-ball); single-edge case works | reference.md |
| Filleted lip goes non-manifold at cap planes | Latent v2-trimmer edge-sharing risk (not the passing d5 guard) | reference.md |
| Straight-edge fillet wall came out as NURBS | Regression: it should be a `CylindricalFace` | Step 5 |
| Want to delete/reorder v1 fillet code | Do NOT. It backs the live public API | Step 3 |

## Step 1: Identify the engine (there are three fillet paths, not two)

The default `fillet` binding does NOT map to one engine. `fillet_solid` (`crates/wasm/src/bindings/operations.rs`) calls `try_fillet` (`crates/wasm/src/helpers.rs`), which tries three engines in order and accepts the first whose outer shell passes `validate_shell_closed`:

1. `fillet::fillet_rolling_ball` (v1 rolling-ball, real rounded blend faces)
2. `blend_ops::fillet_v2` (v2 walking engine, `crates/blend`)
3. `fillet::fillet` (v1 flat-bevel, last resort, planar neighbors only)

Map from a bug report to code:

| JS binding | Engine |
|---|---|
| `fillet`, `filletWithEvolution`, batch `fillet` | `try_fillet` chain: rolling-ball → v2 → bevel |
| `filletV2` | pure v2 `blend_ops::fillet_v2` |
| `chamferV2`, `chamferDistanceAngle` | pure v2 `blend_ops::chamfer_v2` / `chamfer_distance_angle` |
| `chamfer` | flat-bevel engine `chamfer::chamfer` (planar-only, separate code) |
| `filletVariable` | v1 `fillet::fillet_variable` |

Checkpoint: for a default `fillet` bug, instrument `try_fillet` to log which of the three branches returned. Do not debug `crates/blend` for a bug that the rolling-ball branch actually produced.

## Step 2: The silent no-op trap (verify this FIRST on any "it did nothing" report)

`fillet` can return the ORIGINAL solid as a successful result when every edge fails. Two layers cause it:

- `try_fillet` returns `Ok(solid_id)` unchanged when `filter_filletable_edges` drops all edges, and again at its final line when no engine produced a closed shell.
- `fillet_solid` (`operations.rs`, in the `try_fillet` else-branch) returns `solid_id` when `try_fillet` errs and no planar edges remain, then `Ok(solid_id_to_u32(solid))` reports success.

It slips through sign-off because no error is thrown and `validateSolid` / `isClosed` / Euler all pass: the INPUT was already a valid solid. "Ran without error" plus "solid verifies" both green-light a no-op. This is the poster case for the solid-verification skill.

Detection recipe. A real fillet MUST change topology and volume. Assert change, do not assert success:

```rust
let before = brepkit_topology::explorer::solid_entity_counts(&topo, solid)?; // (F,E,V)
// ... fillet ...
let after = brepkit_topology::explorer::solid_entity_counts(&topo, solid)?;
assert_ne!(before, after, "fillet was a silent no-op");
```

- Identical `(F,E,V)` means no blend faces/edges/vertices were added: no-op.
- Identical volume means no material moved. A convex edge rounds material away (volume drops); a concave edge adds it.
- In wasm, if the returned handle equals the input handle AND counts are unchanged, treat it as failure, not success.

## Step 3: Deprecation entanglement (what is safe to touch)

Both v1 fillet functions carry `#[deprecated]` AND are still wired into the live public API:

- `fillet_rolling_ball` (`#[deprecated(since = "2.44.0")]`) is engine #1 in `try_fillet`.
- `fillet::fillet` (`#[deprecated(since = "0.8.0")]`) is engine #3 in `try_fillet`.

Internal callers suppress the warning with `#[allow(deprecated)]` (the attribute lives on `try_fillet` in `helpers.rs`, plus benches, tests, and a re-export in `fillet/mod.rs`).

Removing or auto-migrating v1 is a BEHAVIOR and public-API change, not safe cleanup: it changes what the actively-used rolling-ball fillet ships to JS. This was left for a product decision.

- Safe: improving `crates/blend`, adding tests, fixing bugs inside an engine, editing comments.
- Not safe without a product decision: deleting either deprecated fn, reordering or removing engines in `try_fillet`, changing which engine a binding resolves to.

## Step 4: Debugging the v2 walking engine (`crates/blend`)

Module roles: `radius_law.rs` (radius vs arc-length), `spine.rs` (edge chain + arc-length parameterization), `section.rs`/`stripe.rs` (cross-section and fillet band), `blend_func.rs` (constraint functions), `walker.rs` (Newton-Raphson; `newton_solve` returns `None` on non-convergence, `max_newton_iters` default 20), `analytic.rs` (fast paths: `try_analytic_fillet` dispatches to `plane_plane_fillet`, `plane_cylinder_fillet`, `cylinder_cylinder_fillet`, etc.; unsupported pairs fall to the walker → NURBS wall), `corner.rs` + `spherical_triangle.rs` (vertex blend where stripes meet), `trimmer.rs` (trims neighbor faces along contact curves), `fillet_builder.rs`/`chamfer_builder.rs` (orchestration, plus a dedicated closed-rim path with a fallback).

Failure modes: walker divergence (`newton_solve` returns `None`), trimmer `split_edge_at` leaving a shared cap-face boundary edge unsplit so two faces stop sharing an edge (latent risk, see reference.md (b)), closed-rim fillets emitting free edges.

Instrument at the symptom layer (see debugging-doctrine). Dump `(F,E,V,euler)` and `validate_shell_closed` before and after; count and locate boundary edges (they cluster at the geometric feature); log which `try_analytic_*` arm fired versus the walker fallback; for divergence log `newton_solve` failures with the spine parameter and initial guess. Start from the smallest repro (one edge on a box) and add edges until it breaks.

## Step 5: Scope. What is inherently approximate versus what must be exact

- The blend WALL of a curved-neighbor fillet or chamfer is a NURBS surface with no closed form. Do NOT chase exact-analytic recovery of it (the analytic-preservation skill is for ops that RE-CREATE analytic types, not ops that invent a blend surface).
- Fillet TOPOLOGY and contact curves must still be watertight and correct. The open bugs in reference.md are topology defects and are always in scope.
- Cylindrical fillet walls of STRAIGHT edges ARE analytic and must stay so: the rolling-ball engine emits them as `FaceSpec::CylindricalFace` (defined in `crates/operations/src/boolean/types.rs`), and the blend `analytic.rs` fast paths emit exact typed surfaces for supported pairs. A straight-edge cylindrical wall degrading to NURBS is a regression, not "inherently approximate."

Fillets frequently feed booleans (a lip is Cut(outer, inner) then filleted), so a fillet that leaves free edges will fail the downstream boolean. See boolean-debugging.

Related skills: debugging-doctrine (the method), solid-verification (the no-op trap, before/after F,E,V and volume), boolean-debugging (fillets feed booleans), testing (regression fixtures; the d5 lip test is a live passing guard).
