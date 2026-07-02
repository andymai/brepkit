---
name: tessellation
description: Use when meshing B-Rep faces or debugging tessellation output in crates/operations/src/tessellate/. Triggers: mesh cracks at shared face boundaries, boundary_edge_count > 0, non-manifold edges, a hole on a sphere/torus skinned over (mesh area too large), flipped triangle normals, adding or changing a face mesher, or choosing between CDT and a structured band mesher on a periodic surface (cylinder, cone, sphere, torus).
---

# Tessellation: watertight meshing of B-Rep faces

## When to use

You are producing or debugging a `TriangleMesh` from a solid: cracks between faces, `boundary_edge_count(mesh) > 0`, wrong mesh area or volume, flipped normals, or you are adding a mesher path. All CPU meshing lives in `crates/operations/src/tessellate/`. For verifying the overall solid (volume, Euler, validity) see the **solid-verification** skill; for booleans producing bad faces to mesh, see **boolean-debugging**; for keeping faces analytic in the first place, see **analytic-preservation**.

## Quick reference

```rust
use brepkit_operations::tessellate::{
    tessellate_solid, is_watertight, boundary_edge_count, non_manifold_edge_count,
};

let mesh = tessellate_solid(&topo, solid, 0.1)?;
assert_eq!(boundary_edge_count(&mesh), 0);
assert!(is_watertight(&mesh));
```

`is_watertight` == boundary edges 0 AND non-manifold edges 0. Defined in `crates/operations/src/tessellate/mesh_ops.rs`, re-exported from `tessellate/mod.rs`. `TriangleMesh { positions, normals, indices }` is in `tessellate/mod.rs`. Deflection is the max chord sag; the default angular tolerance is `brepkit_math::chord::DEFAULT_ANGULAR_TOL`.

Locate the meshers and the dispatch:

```bash
rg -n 'pub\(super\) fn tessellate_' crates/operations/src/tessellate/nonplanar.rs
rg -n 'handled_notch|handled_band|tessellate_revolution_band_shared' crates/operations/src/tessellate/solid.rs
```

Expected mesher set (verify, do not assume): `tessellate_revolution_band_shared`, `tessellate_torus_notch_band`, `tessellate_latitude_band_shared`, `tessellate_nonplanar_cdt`, `tessellate_nonplanar_snap`. Full inventory and dispatch order: see [reference.md](reference.md), section "Mesher inventory".

## Core doctrine: shared vertices by construction

Independently triangulated adjacent faces crack at their shared boundary. `tessellate_solid` samples every edge exactly once into a shared pool (`edge_global_indices`); every face's boundary triangles must reference those shared global ids. Two ways a face can fail this:

1. **The snap path** (`tessellate_nonplanar_snap`) meshes the face independently and reconciles rim vertices to the pool by 1e-6 proximity. When independent rim sampling and shared-edge sampling diverge by one segment (a radius/deflection dependent off-by-one), rim vertices land at different angles, miss the snap, and become near-coincident duplicates: a crack. This is issue #696 (drilled magnet hole); the rationale is in the doc comment of `tessellate_revolution_band_shared` in `nonplanar.rs`.
2. **Free CDT on a periodic wall.** Unwrapping u to a flat strip and welding u=0 to u=2pi still leaves the wall CDT free to disagree with a separately triangulated cap disc exactly at the cap-rim/seam corner. CDT cannot weld to a mesh it did not build. Same crack class as #696.

**Rule: periodic-wall bands get STRUCTURED meshers that consume the shared rim vertex ids directly.** Each structured mesher is deliberately narrow, checks its exact shape, and returns `Ok(false)` to defer to the fallback chain otherwise. When adding one, copy that contract: never guess, always defer.

CDT (`tessellate_nonplanar_cdt`) is correct for faces whose UV boundary is a genuine simple polygon: non-periodic patches, partial-revolution faces, faces with inner wires that do not wrap the seam.

## Verify a change (procedure)

1. Build the shape, tessellate at two deflections (e.g. 0.1 and 0.5). Sampling-density bugs are deflection dependent; one value can pass by luck.
2. Check `boundary_edge_count(&mesh) == 0` and `non_manifold_edge_count(&mesh) == 0` at both. A small stable nonzero count (say 8) localized at seam/rim corners means a shared-vertex violation, not "almost done tuning". See [reference.md](reference.md), "Reading a nonzero boundary count".
3. Check orientation via signed volume: integrate the mesh (divergence theorem, as `measure/volume.rs` does) and compare against the analytic volume. Sign flip = global winding wrong; magnitude too large = a hole skinned over; too small = a face missing or inverted run.
4. All tessellation tests live in `crates/operations/src/tessellate/tests.rs` (included via `mod tests;` in `tessellate/mod.rs`); add one there that asserts bd=0 for your exact shape class.

## Symptom to cause

| Symptom | Likely cause | Where to look |
|---|---|---|
| Cracks at a hole rim or cap boundary | Snap path proximity miss (off-by-one rim sampling) | Did dispatch fall through to `tessellate_nonplanar_snap`? Instrument `solid.rs` dispatch |
| Mesh area/volume too LARGE on a bored sphere/torus | Constant-v band degenerated in UV, CDT skinned the hole | Should have hit `tessellate_latitude_band_shared`; check its early-return conditions |
| Stable small nonzero boundary count at seam corners | CDT wall vs separately meshed cap disagreement | Needs a structured shared-vertex mesher, not CDT tuning |
| Zero-extent face, empty triangulation on a closed edge | Endpoint-derived range on a closed edge (start == end) | `edge_sampling.rs` `is_closed()` handling; sample along the curve |
| Scattered flipped triangles inside one face, thin slivers | Per-triangle normal flipping on stitch triangles | Orient the whole run once: `orient_triangle_run` in `nonplanar.rs` |
| Whole face wound inward | Run-level flip decision wrong (degenerate largest triangle) | `orient_triangle_run` normal comparison at the centroid |
| Non-manifold edges after boolean | Upstream face topology (figure-eight wires), not the mesher | See **boolean-debugging**; also `crates/heal/src/upgrade/split_self_intersecting_wires.rs` |

## Degenerate-UV traps

- **Constant-v bands give CDT nothing to bound.** A sphere/torus band between two constant-v full-revolution loops projects each boundary to a zero-area back-and-forth segment in UV; CDT then fills the removed cap. The dispatch comment in `solid.rs` (search `degenerates in UV`) documents this. Structured band meshers exist because of it.
- **Closed edges: start == end.** Any quantity derived from a closed edge's endpoints (param range, winding, extent) collapses to zero. `circle_param_range` and `sample_edge` in `edge_sampling.rs` special-case `edge.is_closed()` to return the full period. Rule: whenever an edge can be closed, sample points along the curve; never derive from endpoints. The boolean engine learned the same lesson (`face_boundary_all_degenerate` in `crates/algo/src/pave_filler/phase_ff.rs` samples along curves).

## Triangle orientation: orient runs, not triangles

Structured meshers emit a coherently wound run, then make ONE flip decision for the whole run via `orient_triangle_run` (`nonplanar.rs`): take the largest-area triangle (most reliable normal), compare its geometric normal to the surface outward normal at its centroid, flip the entire run if they disagree. Per-triangle flipping is unstable on the thin stitch triangles that bridge unevenly sampled rings; it flips neighbors inconsistently and breaks the 2-manifold. If you write a new stitching routine, emit raw, then orient the run.

## Known-terminal case: crossing seam holes

A full-revolution periodic wall carrying two hole loops that CROSS each other (figure-eight, pinching at two points; the equal-radius perpendicular cylinder-union wall) is not triangulable by any current path. Two crossing hole loops are invalid hole topology for flat CDT, and the wall is periodic so flat-strip CDT cracks at the seam. Do not grind this blind. Bounded fixes only; details and what a real fix requires are in [reference.md](reference.md), "Crossing seam holes". Note `split_self_intersecting_wires.rs` in heal is a wire-topology cleanup and is NOT the missing geometric face split.

## GPU path: keep faces analytic

CPU tessellation is not the only consumer of surface parameters. `crates/render/src/compute_mesh.rs` meshes cylinders on the GPU from packed analytic descriptors at screen-space LOD (`CylinderDescriptor`, `screen_space_tess_factor`). Cylinder only at time of writing; verify with `rg -n 'Descriptor' crates/render/src/compute_mesh.rs` before claiming more. Any operation that degrades an analytic face to NURBS or mesh kills this path for that face. See the **analytic-preservation** skill; audit degradations with `cargo run --release --example approx_census -p brepkit-operations`.

## Anti-patterns

- Do NOT conclude "CDT just needs denser sampling" when boundary edges sit at seam or rim corners. That is a structural shared-vertex problem; density changes move the cracks, they do not close them.
- Do NOT snap vertices by proximity as a fix. Snapping is the disease (#696), not the cure.
- Do NOT widen a structured mesher's acceptance heuristically. If the shape check is not exact, return `Ok(false)` and let the chain fall through; a wrong structured mesh is worse than a snap-path crack.
- Do NOT fix winding per-triangle. Orient the run.
- Do NOT derive anything from a closed edge's endpoints.
- Do NOT tessellate inside L0-L2 crates to work around a hard face. Tessellation lives in operations (L3); core stays analytic (see CLAUDE.md, layer rules).
- Do NOT treat one passing deflection value as proof; sampling off-by-ones are deflection dependent.
