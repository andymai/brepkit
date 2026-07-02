---
name: add-operation
description: Use when adding a new modeling operation to crates/operations (a new pub fn taking &mut Topology and returning a SolidId or similar), when extending an existing operation with a new code path, or when an operation's tests need to prove correctness before shipping. Covers implementation traps, error handling, test placement, wasm exposure, and the verification bar.
---

# Adding an Operation to brepkit

## When to use

You are creating or substantially extending an operation in `crates/operations/src/` (extrude/revolve/sweep class, a measure, a transform, a new primitive). CLAUDE.md Recipe 3 gives the file scaffolding. This skill adds what Recipe 3 omits: the verification bar, where tests go, and the correctness traps that produce compiles-but-wrong geometry.

## Quick reference

| Step | Command / API | Expect |
|------|---------------|--------|
| Scaffold | CLAUDE.md Recipe 3 | file, fn, `lib.rs` module, wasm binding, batch dispatch |
| Measure | `crate::measure::solid_volume(&topo, solid, 0.001)` | relative error < 1% vs closed form |
| Validate | `crate::validate::validate_solid(&topo, solid)` | `report.is_valid()` true |
| Watertight | `tessellate_solid_with_tolerance` + `tessellate::is_watertight` | 0 boundary + 0 non-manifold edges (index-based); also re-check with the positional-weld helper from `tessellate_watertight.rs` |
| Census | `cargo run --release --example approx_census -p brepkit-operations` | no new `brepkit_approx` probe fires for your op |
| Gate | `cargo clippy --all-targets -- -D warnings && cargo fmt --all && ./scripts/check-boundaries.sh` | clean |

## Procedure

1. **Scaffold per CLAUDE.md Recipe 3.** Signature `pub fn op_name(topo: &mut Topology, ...) -> Result<SolidId, OperationsError>`.
2. **Walk faces correctly.** Any solid-scoped loop over faces must use `brepkit_topology::explorer::solid_faces` (there is also `solid_edges`). See CLAUDE.md "Walking faces in a solid" for the exception rule for per-shell operations. Iterating only `outer_shell()` compiles, passes on simple boxes, and silently skips cavity faces on hollow solids.
3. **Respect the borrow pattern and error rules.** Snapshot-then-allocate and closure return type annotations: see CLAUDE.md Common Pitfalls. The workspace denies `unwrap_used`, `panic`, and `unsafe_code` in production code; test modules opt out with `#![allow(clippy::unwrap_used, clippy::expect_used)]`.
4. **If your op copies edges that may carry periodic curves** (`EdgeCurve::Circle` or `Ellipse`), read reference.md "Periodic edge copies" before writing the copy loop. Getting this wrong recovers the complementary arc (minor instead of major) and only reversed edges expose it.
5. **Write tests** (placement below). Every geometry-producing op needs at least: a volume-vs-closed-form test, a `validate_solid` test, and a watertight-tessellation test.
6. **Run the verification bar** (reference.md "Verification bar" has the exact APIs and checkpoint expectations). If the op touches analytic geometry, run the approx census; a fallback probe firing for your op means you degraded analytic surfaces to NURBS or mesh, see the analytic-preservation skill.
7. **Expose to wasm** per CLAUDE.md Recipe 4; details in the wasm-bindings skill (binding module choice, `batch_*` companion, contract tests via `execute_batch()`).
8. **Run the gate commands** from the quick reference before pushing.

## Where tests go

CLAUDE.md's Testing section reads as if golden and integration tests live at the repo root. They do not; the root dirs hold data and docs only.

| Kind | Location | Pattern |
|------|----------|---------|
| Unit tests | `crates/operations/src/<op>/tests.rs`, declared via `mod tests;` at the bottom of `<op>.rs` | `extrude.rs` + `extrude/tests.rs` |
| Crate integration | `crates/operations/tests/*.rs` | `tessellate_watertight.rs`, `boolean_invariants.rs`, `proptest_operations.rs` |
| Golden | `crates/operations/tests/golden_regression.rs`, data in `tests/golden/data/` | regenerate: `UPDATE_GOLDEN=1 cargo test --workspace golden` |
| Wasm contract | wasm crate, via `execute_batch()` only | see wasm-bindings skill |

Do not add standalone test files under the root `tests/integration/` or `tests/golden/`; both READMEs there confirm tests are crate-level.

## Pitfalls: what NOT to conclude

| Observation | Wrong conclusion | Reality |
|-------------|------------------|---------|
| Volume matches closed form | Geometry is correct | Volume can read high on arc-edged results and a nearly-unmodified solid can pass. Also classify interior/exterior probe points with `crate::classify::classify_point` (ray-cast). Never trust the winding classifier for faceted or NURBS solids. |
| Census reports exact analytic | Op is correct | It only proves no approximation fallback fired, not that the geometry is right. A wrong-but-analytic result passes the census. |
| Mesh indices share vertices | Mesh is watertight | Watertightness is geometric: weld vertices by quantized position first, then check edge sharing. Use `tessellate::is_watertight` / `boundary_edge_count`, or copy the `boundary_edges` helper from `crates/operations/tests/tessellate_watertight.rs`. |
| Box test passes | Face walk is complete | Boxes have no inner shells. Add a hollow-solid case (shell_op or boolean-cut cavity) if the op walks faces. |
| Forward-edge arc test passes | Periodic edge copy is correct | Only a reversed `OrientedEdge` swaps stored vertex order vs curve parameterization. Add a reversed-half-circle test (see reference.md). |
| Volume of an off-origin primitive is wrong | The measure is broken | Primitives place their base at z = 0, not centered. Check your expected value first. |
| Cone surface points land off the cone | ConicalSurface is broken | `half_angle` is measured from the radial plane, not from the axis. See reference.md "Cone conventions". |

## Sibling skills

- **wasm-bindings**: binding module, `js_name`, input validation helpers, batch dispatch, contract tests.
- **testing**: fixtures, proptest, golden data regeneration, test layout in depth.
- **solid-verification**: the full verification bar in depth (measure, validate, watertight, classifiers, census).
- **analytic-preservation**: keeping results as typed analytic surfaces instead of NURBS.
- **layer-boundaries**: which crates your op may `use`; `./scripts/check-boundaries.sh` enforces it.

## Reference

Detailed APIs, checkpoints, and the periodic-edge treatment: [reference.md](reference.md).
