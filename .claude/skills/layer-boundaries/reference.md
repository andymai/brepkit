# Layer Boundaries: Reference

Deep detail for the layer-boundaries skill. All paths and symbols verified against the current tree; when in doubt, re-run the rg commands rather than trusting counts.

## Variant-add procedure (EdgeCurve / FaceSurface)

### Current state

- `EdgeCurve` in `crates/topology/src/edge.rs` (`rg -n 'pub enum EdgeCurve'`). Variants: `Line`, `NurbsCurve(NurbsCurve)`, `Circle(Circle3D)`, `Ellipse(Ellipse3D)`.
- `FaceSurface` in `crates/topology/src/face.rs` (`rg -n 'pub enum FaceSurface'`). Variants: `Plane { normal, d }`, `Nurbs(NurbsSurface)`, `Cylinder(CylindricalSurface)`, `Cone(ConicalSurface)`, `Sphere(SphericalSurface)`, `Torus(ToroidalSurface)`.

### Delegate methods (where NOT to fan out)

The delegates are inherent methods on the enums, in the same files as the enum definitions. CLAUDE.md's pointer to `math/src/traits.rs` for these delegates is stale: that file holds the `ParametricCurve`/`ParametricSurface` traits and their impls for concrete geometry types, and never mentions `EdgeCurve` or `FaceSurface`.

- `EdgeCurve` delegates (`crates/topology/src/edge.rs`): `evaluate_with_endpoints`, `tangent_with_endpoints`, `domain_with_endpoints`, `type_tag`. Locate: `rg -n 'fn evaluate_with_endpoints' crates/topology/src/edge.rs`.
- `FaceSurface` delegates (`crates/topology/src/face.rs`): `evaluate`, `normal`, `project_point`, `estimate_radius`, `type_tag`, `is_planar`, `is_analytic`, `as_analytic`. Locate: `rg -n 'fn as_analytic' crates/topology/src/face.rs`.

Every call site that goes through a delegate is done once the delegate impl has the new arm. Only direct `match`/`if let` sites on the enum need per-site work.

### Procedure with checkpoints

1. Define the geometry type first. New analytic curve: `crates/math/src/curves.rs`. New analytic surface: `crates/math/src/surfaces.rs`. Implement the relevant trait from `math/src/traits.rs` (`ParametricCurve` or `ParametricSurface`) so sampling and extrema code can use it.
   - Checkpoint: `cargo build -p brepkit-math` is green before touching topology.
2. Add the enum variant in `topology/src/edge.rs` or `topology/src/face.rs` and add its arm to every delegate method in the same file.
   - Checkpoint: `cargo build -p brepkit-topology` is green.
3. Enumerate the blast radius (informational; the compiler is the real list):
   ```bash
   rg -l 'EdgeCurve::' crates/*/src/    # ~110 files at time of writing
   rg -l 'FaceSurface::' crates/*/src/  # ~130 files
   ```
   Counts drift upward over time; trust the command output, not any recorded number.
4. `cargo build --workspace` and fix every `non-exhaustive patterns` error. Expect many. Do not add `_ =>` arms to shortcut this; each site needs a real decision (support the variant, or return a typed "unsupported" error).
   High-traffic direct-match sites, in rough priority order:
   - `operations/src/tessellate/`, `operations/src/transform.rs`, `operations/src/copy.rs`, `operations/src/section.rs`, `operations/src/boolean/`
   - EdgeCurve extra: `operations/src/measure/edge_length.rs`, `operations/src/fill_face.rs`
   - FaceSurface extra: `operations/src/distance.rs`, `operations/src/feature_recognition.rs`, `operations/src/offset_face.rs`
   - `io/src/step/{reader,writer}.rs`, `io/src/iges/{reader,writer}.rs`
   - `wasm/src/bindings/{query,batch,tessellate,nurbs}.rs`
   - Checkpoint: `cargo build --workspace` green, then `cargo clippy --all-targets -- -D warnings`.
5. Wildcard rescan. Three files have historically re-grown `_ =>` arms:
   ```bash
   rg -n '_ =>' crates/io/src/step/writer.rs crates/io/src/iges/writer.rs crates/operations/src/offset_face.rs
   ```
   Expected output today: hits only inside `#[cfg(test)]` code in `offset_face.rs` (`_ => panic!("expected planar surface")` style assertions). Any hit on a production path means the new variant is being silently dropped or mis-serialized; replace the wildcard with explicit arms.
6. Tests: round-trip the new variant through the writers you touched (STEP at minimum), and through `transform`/`copy` (a copied or transformed solid must preserve the variant, not degrade to NURBS; see the analytic-preservation skill).
7. `./scripts/check-boundaries.sh` if any Cargo.toml changed, then `cargo test --workspace`.

### If the new surface is analytic: AnalyticSurface

`AnalyticSurface<'a>` in `crates/math/src/analytic_intersection.rs` (`rg -n 'pub enum AnalyticSurface'`) holds borrowed refs (`Cylinder(&CylindricalSurface)`, `Cone`, `Sphere`, `Torus`). Adding a variant there fans out to:

- Dispatch sites within `analytic_intersection.rs` itself: `exact_plane_analytic`, `intersect_plane_analytic`, `sample_plane_analytic`, `try_algebraic_intersection`, `project_analytic`, `surface_closures`, and a `matches!` in `is_u_periodic`. The compiler flags most of these. Old docs say "4 match sites"; the real number is larger and grows, so let the build enumerate them.
- COMPILER-INVISIBLE GOTCHA: `try_algebraic_intersection` does a pairwise `match (a, b)` ending in `_ => Ok(None)`, a legitimate unhandled-pair fallback. A new analytic variant compiles clean there and silently reports "no algebraic intersection" for every pair involving it, so intersections fall back to slower or failing paths. Add the pairs you can solve in closed form deliberately, and record the ones you cannot.
- Consumers outside math (find with `rg -l 'AnalyticSurface' crates/*/src/`): `topology/src/face.rs` (`as_analytic` delegate), `algo/src/pave_filler/phase_ff.rs`, `offset/src/inter3d.rs`, `wasm/src/bindings/query.rs`, plus tests in `operations/src/boolean/tests.rs`.

## check-boundaries.sh internals

- Extracts only `[dependencies]` per crate: `sed -n '/^\[dependencies\]/,/^\[/p'`, then greps each `brepkit-*` name against the crate's allowlist.
- `[dev-dependencies]` are exempt by design (integration tests may reach upward). Consequence: the script cannot catch the topology dev-dep cycle trap; that one only shows up as "two different versions of crate" build errors.
- Output shapes:
  - Pass: `Checking crate boundary rules...` then `✅ All crate boundaries valid.`, exit 0.
  - Fail: one `VIOLATION: crates/<crate> depends on <dep> (not allowed)` per offense, then `❌ Boundary check failed.`, exit 1.
  - `SKIP: crates/<name>/Cargo.toml not found` means a crate was renamed or removed; update the script's `check_deps` calls.
- Allowlist vs CLAUDE.md table: the script permits `brepkit-geometry` for `algo` and `blend`, and a direct `brepkit-blend` dep for `wasm` (the table allows blend only transitively, via operations; wasm's other deps, including offset and sketch, are allowed by both). None of the extra allowances are used in the actual Cargo.tomls today. If you find yourself needing one, it passes CI, but confirm the layering intent in review first; the CLAUDE.md table is the source of truth for intent.
- Enforcement points: CI job `boundaries` in `.github/workflows/ci.yml`, gated into `ci-pass`. The `.husky/pre-push` hook runs nothing (it delegates to CI), and `.husky/pre-commit` runs fmt + clippy + taplo + machete only. So a boundary violation surfaces at PR time unless you run the script yourself.

## Symptom-to-cause: boundary and cycle failures

| Symptom | Cause | Action |
|---|---|---|
| `error[E0432]: unresolved import brepkit_x` in crate Y | Missing Cargo dep, possibly because the layer forbids it | Check the CLAUDE.md table. If forbidden: move the code down or restructure. If allowed: add the dep and re-run the script |
| Script prints VIOLATION for a dep you believe is fine | The dep string appears in `[dependencies]` (even transitively via a table entry you edited) | Read the crate's Cargo.toml `[dependencies]` block; the script only sees that section |
| `expected struct brepkit_topology::Topology, found struct brepkit_topology::Topology` | Two feature-resolved builds of topology, usually from an upward dev-dep on topology | Remove the offending dev-dep; move the helper into `topology/src/test_utils.rs` behind `test-utils` |
| A crate compiles locally but the `boundaries` CI job fails | You added the dep but never ran the script | `./scripts/check-boundaries.sh` locally, fix, push |
| Reviewer flags mesh/triangle code in algo, check, math, etc. | Tessellation-based shortcut below L3 | Reformulate analytically; tessellation belongs in `operations/src/tessellate/` or `render/`. Current ground truth: `rg -in 'tessellat' crates/{math,geometry,topology,algo,blend,heal,check,offset,sketch}/src/` matches roughly 15 files, and every hit is a comment or docstring (e.g. "Instead of tessellating a face" in `math/src/quadrature.rs`). Any hit on executable code is a violation. Keep it that way |

## Glossary for newcomers

- B-Rep: boundary representation. Solids as faces/edges/vertices with exact analytic or NURBS geometry, not triangles.
- GFA: brepkit's general fuse algorithm, the boolean engine (`crates/algo/src/gfa.rs`).
- PCurve: a 2D curve in a face's UV parameter space representing an edge on that surface.
- Analytic geometry: exact closed-form types (plane, cylinder, cone, sphere, torus; line, circle, ellipse), as opposed to NURBS approximation.
- Arena / `Id<T>`: topology entities live in a central arena (`topology/src/arena.rs`) and are referenced by typed handles.
- Ripple effect: adding an enum variant forces updates at every exhaustive match site; the compiler enumerates them for you.
