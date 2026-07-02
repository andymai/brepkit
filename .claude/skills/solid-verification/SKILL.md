---
name: solid-verification
description: Ground-truth verification of any B-Rep result in brepkit. Use when checking whether a solid is watertight, manifold, correctly classified, or has the right volume; when signing off a boolean, offset, fillet, or other geometry change; when a volume looks wrong or disagrees across deflections; or when deciding whether a "passing" check actually proves the geometry is correct.
---

# Solid Verification

## When to use

You produced or modified a solid (boolean result, primitive, sweep, import, healed shape) and need to answer: is this geometry actually correct? This skill gives the verification ladder, what each rung proves and does not prove, and the traps that have produced wrong sign-offs before. For debugging a failing boolean itself, see the boolean-debugging skill. For mesh quality questions, see the tessellation skill.

## The verification ladder (cheapest to strongest)

Run rungs in order. Each rung's "does NOT prove" column is the point. Full API detail per rung: [reference.md](reference.md), section "Ladder detail".

| Rung | Question | API (Rust) | Expect | A pass does NOT prove |
|---|---|---|---|---|
| 1. Census | Right face count and surface types? | `explorer::solid_entity_counts`, `FaceSurface::type_tag()` over `explorer::solid_faces`; `cargo run --release --example approx_census -p brepkit-operations` | Tens of faces, quadric types present for curved solids | Geometry is correct. An "exact analytic" result can still have an uncarved feature |
| 2. Validation | Topologically sane? | `brepkit_operations::validate::validate_solid` (loop-aware Euler) or `brepkit_check::validate::validate_solid` (report with `CheckId`s) | 0 errors | The shape matches intent. A closed manifold of the WRONG shape passes |
| 3. Free edges | B-Rep closed and manifold? | Errors from `validate_solid` ("boundary edge(s)", "non-manifold"), or `heal::analysis::free_bounds::find_free_bounds` for the loops | 0 free, 0 non-manifold | Faces were classified correctly by the boolean |
| 4. Mesh watertight | Tessellation closed? | `tessellate_solid` then `mesh_ops::boundary_edge_count` and `non_manifold_edge_count` | Both 0 | The B-Rep is closed (the tessellator can stitch over small B-Rep gaps) |
| 5. Classification | Is material where intended? | `brepkit_check::classify::classify_point` (ray-cast) at intent-encoding probe points | Inside carved features: `Outside`. Inside kept material: `Inside` | Anything between probes. A thin missing sliver escapes sparse probes |
| 6. Volume/area | Quantity right? | `measure::solid_volume(topo, solid, deflection)` at two deflections, vs closed form when one exists | Agreement within ~1e-4 relative; converges as deflection refines | The feature was carved. Volume can read plausible while a cut never happened |

## Procedure: verifying a boolean result

1. Count faces and census the surface types (rung 1). Checkpoint: a clean analytic boolean of primitives has roughly 3 to 80 faces with cylinder/sphere/cone/torus types surviving. Hundreds to thousands of all-plane faces means the mesh fallback fired: the result is a co-refined triangle soup, not analytic B-Rep. Triangle counts and validity checks BOTH mask this; face census is the only reliable tell. If the fallback fired, stop and see the boolean-debugging skill.
2. Validate (rungs 2 and 3). Checkpoint: `brepkit_operations::validate::validate_solid` returns an empty issue list, or only issues you can explain. Free edges must be 0. Note the check-crate `SolidEulerCharacteristic` check demands V-E+F == 2 exactly and will warn on any solid with a through-hole; the operations-crate validator applies the genus and inner-loop correction, so prefer it for boolean results. For assembled compounds that legitimately do not share edges, use `validate_solid_relaxed`.
3. Mesh watertightness (rung 4). `is_watertight(&mesh)` from `operations/src/tessellate/mesh_ops.rs`, or the two counts separately. Checkpoint: both 0 at the deflection you ship with.
4. Ray-cast spot checks (rung 5). Pick probes AT the feature being verified: pocket centers, wall midpoints, points just inside and just outside each cut face. Use `brepkit_check::classify::classify_point` with `ClassifyOptions::default()`. NEVER the winding classifier here (see below).
5. Volume convergence (rung 6). Compute `solid_volume` at two deflections FINER than `bbox_diagonal * 5e-5` (coarser requests are clamped to that value and will trivially "agree"). Checkpoint: agreement within ~1e-4 relative, and match against a closed form if one exists. If volume moves as deflection refines, especially upward, the geometry is bad: inscribed meshes of good geometry converge to the truth from below.

## Volume traps

Details and the full path list: [reference.md](reference.md), section "Volume paths".

- `solid_volume` clamps the tessellation deflection: `volume_tessellation_deflection` in `crates/operations/src/measure/volume.rs` returns `requested.min((bbox_diag * 5e-5).max(1e-9))`. It refines coarse preview deflections but never coarsens a fine request. Consequence: callers passing preview-tuned deflections still get accurate volume, and two coarse requests are not an independent convergence test.
- There are three tiers of volume path inside `solid_volume`: analytic (`try_analytic_solid_volume` for pure sphere/cylinder/cone/torus, plus `analytic_faces_solid_volume` and `analytic_revolution_solid_volume` behind narrow structural guards), direct per-face (`solid_volume_from_faces`, `volume_from_direct_face_tessellation`), and whole-solid tessellation with signed tetrahedra. Which path fired matters when a volume looks wrong.
- The check-crate `brepkit_check::properties::solid_volume` (naive Gauss sum over all faces via `face_integrator::integrate_face`) is NOT a safe exact path for general or boolean solids. It gets trimmed periodic faces and orientation issues wrong; the engine itself only uses it behind narrow guards. Recorded example (PR #959 investigation): a half-disc prism integrated 54.08 against a true 39.27. Do not reach for it as an independent oracle.
- Never sign off a boolean on volume alone. Recorded precedent (PR #997): a seam-collapse bug produced a result whose slot was never carved, the census said "exact analytic", and the volume read plausible (about 1.4% high, diverging further with refinement). Ray-cast classification inside the supposed cut is the ground truth.

## Point classification: which classifier

- Rust ground truth: `brepkit_check::classify::classify_point` (analytic ray casting, majority vote over fixed ray directions, perturbed recovery rays). Returns `Inside | Outside | OnBoundary`.
- The winding-number variants `classify_point_winding` and `classify_point_robust` return confidently WRONG answers (Outside for interior points) on faceted, stepped, boolean-result, and NURBS-heavy solids, and `robust` never falls back because the winding value is not near 0.5. This has caused multiple wrong diagnoses. Rule: winding only for clean analytic primitives, ray-cast for everything you are actually verifying.
- Caveat: all check-crate classifiers walk `outer_shell()` faces only. A probe inside a cavity bounded by an inner shell is not tested against the cavity faces.
- JS side: the `classifyPoint` binding routes to the operations-crate tessellation-based ray caster with a fixed deflection of 0.1. Fine for spot checks; in Rust prefer the check-crate analytic version.

## Watertightness: two distinct questions

- B-Rep watertight: every edge maps to exactly 2 faces. Check via `validate_solid` errors, or get the actual open loops from `find_free_bounds`.
- Mesh watertight: `boundary_edge_count(&mesh) == 0 && non_manifold_edge_count(&mesh) == 0`.
- They can disagree in both directions: the tessellator can stitch a closed mesh over a leaky B-Rep, and a closed B-Rep can tessellate with per-face seam boundary edges. Check both when it matters; `solid_volume` itself gates some paths on a watertight mesh and falls through rather than return a leaky volume.

## Verification bar for shipping a geometry change

1. Suite green: `cargo test --workspace`, `cargo clippy --all-targets -- -D warnings`, `./scripts/check-boundaries.sh` (see CLAUDE.md, Commands).
2. Census unchanged or improved: `cargo run --release --example approx_census -p brepkit-operations`. No new approximation probes fire; face counts stay analytic-sized. Re-run this after every boolean-engine change; prior results go stale.
3. Watertight both ways: B-Rep free edges 0 and non-manifold 0; mesh boundary edges 0.
4. Ray-cast spot checks with `brepkit_check::classify::classify_point` at points that encode the intent of the change.
5. Volume agreement within tolerance at two deflections finer than the clamp, plus closed-form match where available. Divergence under refinement means bad geometry.

## Anti-patterns: what NOT to conclude

- "Volume matches, so the cut happened." No. Verify with classification probes inside the cut.
- "The mesh is watertight, so the B-Rep is closed." No. Check free edges separately.
- "Validation passed, so the boolean is correct." No. Validation is topological; it cannot see a wrong-but-closed face selection.
- "Winding says Outside, so the point is outside." Not on faceted or NURBS solids. Re-check with the ray-cast classifier before theorizing.
- "Triangle count looks normal, so no mesh fallback." Triangle count masks the fallback. Only the face census tells.
- "V-E+F != 2, so the solid is broken." Not if it has a through-hole or inner loops; use the operations-crate Euler check.
- "Two volumes at 0.1 and 0.01 deflection agree, so it converged." Both were likely clamped to the same effective deflection. Use requests finer than `bbox_diag * 5e-5`.
- "I walked `outer_shell()` and saw all the faces." Hollow solids have inner shells; use `explorer::solid_faces` (see CLAUDE.md, Walking faces in a solid).

## Related skills

boolean-debugging (when a rung fails on a boolean result), tessellation (mesh quality and deflection), analytic-preservation (keeping results analytic), parity-benchmarking (comparing against the reference kernel via the brepjs harness), testing (where verification assertions belong).
