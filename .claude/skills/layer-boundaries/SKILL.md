---
name: layer-boundaries
description: Use when adding a workspace dependency between brepkit crates, deciding which crate new code belongs in, adding a variant to EdgeCurve or FaceSurface (or a new analytic surface type), fixing a CI "boundaries" job failure or a VIOLATION from scripts/check-boundaries.sh, or hitting "two different versions of crate brepkit-topology" errors.
---

# Layer Boundaries

brepkit is a strict layered DAG (L0 math up to L4 wasm/render). This skill covers: keeping the DAG intact, the safe procedure for adding `EdgeCurve`/`FaceSurface` variants, and where new code goes. The dependency table itself lives in CLAUDE.md, "Layer dependency rules". Do not restate it; check against it.

## Quick reference

```bash
./scripts/check-boundaries.sh          # verify crate deps; exit 0 = "✅ All crate boundaries valid."
rg -l 'EdgeCurve::' crates/*/src/      # all files matching EdgeCurve variants (~110 files)
rg -l 'FaceSurface::' crates/*/src/    # all files matching FaceSurface variants (~130 files)
rg -n '_ =>' crates/io/src/step/writer.rs crates/io/src/iges/writer.rs crates/operations/src/offset_face.rs
```

| Symptom | Cause | Fix |
|---|---|---|
| CI `boundaries` job fails / `VIOLATION: crates/X depends on brepkit-Y` | Cargo dep added against the layer table | Remove the dep; move the code to a layer where the dep is legal, or invert the dependency |
| `use brepkit_x::...` fails to resolve in crate Y | Y's layer may not depend on x | Same as above; do NOT add the dep to "make it compile" |
| "two different versions of crate `brepkit-topology`" type errors | A higher-layer crate was added as a dev-dep of topology | Revert; use topology's `test-utils` feature instead (see below) |
| Hundreds of `non-exhaustive patterns` errors after adding an enum variant | Expected; matches are exhaustive by design | Work through them; this IS the checklist (see procedure below) |
| New analytic surface pair silently returns "no intersection" | `_ => Ok(None)` fallback in `try_algebraic_intersection` | Add the pair arm deliberately; the compiler will not flag this one (see reference.md, AnalyticSurface) |

## The boundary check

`scripts/check-boundaries.sh` reads only the `[dependencies]` section of each `crates/*/Cargo.toml` and compares against a per-crate allowlist. Dev-dependencies are deliberately exempt. It does not inspect `use` paths; those are enforced transitively because a `use brepkit_x::` without the Cargo dep will not compile.

When it runs: CI only, as the `boundaries` job gated into `ci-pass` (`.github/workflows/ci.yml`). The pre-push hook delegates all validation to CI, so run the script manually before pushing:

```bash
./scripts/check-boundaries.sh
```

Expect `✅ All crate boundaries valid.` and exit 0. On failure it prints one `VIOLATION:` line per bad dep and exits 1.

Nuances:
- The script's allowlist is slightly looser than the CLAUDE.md table: it permits `brepkit-geometry` for `algo`/`blend`, and a direct `brepkit-blend` dep for `wasm` (the table allows blend only transitively, via operations). None of these are used today. Treat the CLAUDE.md table as intent, the script as the floor. Passing the script but violating the table still gets flagged in review.
- No crate may depend on `brepkit-render`. It is an L4 leaf.

## Adding an EdgeCurve or FaceSurface variant

Full procedure with checkpoints: see [reference.md](reference.md), "Variant-add procedure". Summary:

1. Both enums live in topology: `EdgeCurve` in `crates/topology/src/edge.rs`, `FaceSurface` in `crates/topology/src/face.rs`.
2. All production match arms are exhaustive (no `_ =>` wildcards). The compiler is the authoritative checklist: add the variant, run `cargo build --workspace`, and fix every flagged site. Do not hand-maintain a file list.
3. The delegate methods absorb most call sites. They are inherent methods on the enums themselves (not in `math/src/traits.rs`, which holds the `ParametricCurve`/`ParametricSurface` traits for concrete geometry types). Call sites using `EdgeCurve::evaluate_with_endpoints`, `FaceSurface::evaluate`, `as_analytic`, etc. need no change; only the delegate impl gets a new arm.
4. After the build is green, rescan the three files historically prone to re-introduced wildcards (`rg -n '_ =>'` on them): `io/src/step/writer.rs`, `io/src/iges/writer.rs`, `operations/src/offset_face.rs`. Wildcards inside `#[cfg(test)]` blocks are acceptable; production wildcards are not.
5. If the new surface is analytic, also extend `AnalyticSurface` in `crates/math/src/analytic_intersection.rs` and check its one compiler-invisible fallback. Details in reference.md.

## The dev-dependency cycle trap

Never add `brepkit-operations` (or any crate above topology) to `[dev-dependencies]` of `brepkit-topology`. Cargo then builds two feature-resolved versions of topology and every `Id<T>` and `Topology` value becomes a "two different versions of crate" type mismatch. The boundary script will NOT catch this: it ignores dev-deps on purpose.

The sanctioned pattern: shape-building test helpers go in `topology/src/test_utils.rs` behind the `test-utils` feature (`#[cfg(feature = "test-utils")]` in `topology/src/lib.rs`). Higher crates enable it in their dev-deps:

```toml
[dev-dependencies]
brepkit-topology = { workspace = true, features = ["test-utils"] }
```

algo, blend, heal, check, offset, operations, and io all already do this. Copy that line, do not invent a new mechanism.

## No tessellation in core (L0 to L2)

Core computation (booleans, classification, intersection, distance, volume, healing) must use exact geometry: analytic formulas and parameter-space algorithms on the B-Rep. Tessellating to triangles and computing on the mesh is a review-rejectable bug in L0-L2, not a style issue. Reasons: meshes lose accuracy, sequential operations explode face counts, and surface type information is destroyed. Production B-Rep kernels, including the reference kernel, never tessellate for computation.

Where tessellation legally lives: `crates/operations/src/tessellate/` (L3, for export and preview) and `crates/render/` (L4, GPU display meshing). If an algorithm seems to need a mesh below L3, the design is wrong; find the analytic or parameter-space formulation, or raise it as a blocker. See the tessellation and analytic-preservation skills.

## Where new code goes

Put code in the lowest layer whose allowed deps suffice, except tessellation and operation orchestration, which never go below L3.

| Kind of change | Location |
|---|---|
| Vector/matrix/NURBS/predicate/quadrature primitive | L0 `crates/math/src/` (no workspace deps) |
| New analytic curve or surface type | L0 `math/src/curves.rs` / `surfaces.rs`, then the variant-add procedure above |
| Sampling, extrema, analytic-NURBS conversion | L1 `crates/geometry/src/` |
| B-Rep entity, arena, explorer, adjacency | L1 `crates/topology/src/` |
| Boolean/classification engine internals | L2 `crates/algo/src/` |
| Fillet/chamfer engine | L2 `crates/blend/src/` |
| Healing, validation, properties, distance, solid offset, 2D constraints | L2 `heal` / `check` / `offset` / `sketch` |
| User-facing modeling op, measure, tessellation | L3 `crates/operations/src/` (CLAUDE.md Recipe 3; see add-operation skill) |
| New file format | L3 `crates/io/src/<format>/` (CLAUDE.md Recipe 2) |
| GPU rendering or display meshing | L4 `crates/render/src/` (leaf) |
| JS-facing API | L4 `crates/wasm/src/bindings/` (CLAUDE.md Recipe 4; see wasm-bindings skill) |

For the exact file within a crate, use the CLAUDE.md Module Map.

## Anti-patterns

- Do NOT add a Cargo dep to silence an unresolved `use`. The unresolved import is the boundary working as intended.
- Do NOT conclude "the script passed, so the layering is fine". The script is the floor; check the CLAUDE.md table for intent.
- Do NOT add `_ =>` arms to make a variant-add compile faster. Silencing the compiler discards the checklist and creates silent misbehavior for the new variant.
- Do NOT move an algorithm up a layer just because a convenient helper lives there. Extract or reimplement the helper at the lower layer instead.
- Do NOT assume "no compile errors in math" means the analytic intersection work is done: `try_algebraic_intersection` ends in `_ => Ok(None)` and swallows new surface pairs silently.
