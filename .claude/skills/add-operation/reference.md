# add-operation reference

Verified against the current repo. Locate symbols with the given `rg` patterns; line numbers rot, symbols do not.

## Verification bar

Run checks 1-3 for every geometry-producing op, check 4 when the op touches analytic geometry, and the gate (5) before pushing. The solid-verification skill covers each in depth; this is the operational summary.

### 1. Measure

```rust
let vol = crate::measure::solid_volume(&topo, solid, 0.001)?;
```

`rg -n 'pub fn solid_volume' crates/operations/src/measure/volume.rs`

- Picks its own path: exact closed form for recognized primitives, exact per-face Gauss quadrature when every face is analytic, tessellation otherwise. Do not route around it to another integrator; call it and let it choose.
- Deflection is internally clamped relative to the bbox diagonal, so a coarse deflection cannot silently undercount. Volume can still read high on arc-edged results.
- Test pattern (from `crates/operations/src/extrude/tests.rs`): assert relative error under 1% against a hand-computed closed form. Compute the closed form for the base-at-z=0 placement, not a centered one.
- Volume agreement is necessary, not sufficient. Also probe points that must be inside and outside:

```rust
use crate::classify::{classify_point, PointClassification};
let c = classify_point(&topo, solid, probe, 0.01, 1e-6)?;
assert_eq!(c, PointClassification::Inside);
```

This is the ray-cast classifier. The winding-number classifier exists both in `brepkit-check` (`classify/winding.rs`) and as `classify_point_winding` / `classify_point_robust` in the same `crates/operations/src/classify.rs`; none of these are valid verification oracles for faceted or NURBS solids. Use `classify_point` only.

### 2. Validate

```rust
let report = crate::validate::validate_solid(&topo, solid)?;
assert!(report.is_valid(), "{:?}", report.issues);
```

`rg -n 'pub fn validate_solid' crates/operations/src/validate.rs`

- Checks include Euler-Poincare consistency, closed shells, edge/face consistency, connectivity. `is_valid()` means no `Severity::Error` issue.
- For ops that produce NURBS faces (fillet, shell, offset class), the default tolerances can false-positive. Use `validate_solid_with_options` with `ValidationOptions { tolerance_scale: 10.0, .. }` (the scale is clamped to [0.1, 1000]). There is also `validate_solid_relaxed`.

### 3. Tessellate and check watertightness

```rust
use crate::tessellate::{tessellate_solid_with_tolerance, is_watertight, boundary_edge_count, non_manifold_edge_count};
```

`rg -n 'pub fn tessellate_solid' crates/operations/src/tessellate/solid.rs` and `rg -n 'pub fn is_watertight' crates/operations/src/tessellate/mesh_ops.rs`

- Watertight means: after welding vertices by quantized position, every undirected edge is shared by exactly two triangles.
- `is_watertight` operates on the `TriangleMesh` as built; the gold-standard integration test `crates/operations/tests/tessellate_watertight.rs` additionally welds by position at 1e-6 quantization before counting, which catches meshes that are watertight geometrically but not by index, and vice versa. Copy its `boundary_edges` helper for a new op's watertight test.
- A boundary edge count above zero means a hole; an STL consumer (slicer) will reject the export.

### 4. Approx census (only if the op touches analytic geometry)

```
cargo run --release --example approx_census -p brepkit-operations
```

- Installs a logger that captures `brepkit_approx` debug probes and prints per-op rows: exact analytic vs which approximation fallback fired, plus wall clock and face count. Probe-site catalog: analytic-preservation skill, reference section 1.
- Bar: an op that re-creates existing analytic surface types (extrude of a circle should make a Cylinder, revolve of a line should make a Cone, and so on) must not light a probe. If it does, you converted analytic geometry to NURBS or fell back to the mesh boolean; see the analytic-preservation skill.
- Do NOT conclude correctness from a clean census. It proves only that no fallback fired. A documented near-miss: a slot cut read 1.4% high on volume and almost passed while the slot had not actually been carved. Volume + census + classify_point probes together are the minimum.
- If you add a new operation with fallback paths, add a census scenario for it in `crates/operations/examples/approx_census.rs`.

### 5. Gate

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test -p brepkit-operations
./scripts/check-boundaries.sh
```

Pre-push runs the full test suite and cargo-deny; do not skip hooks.

## Periodic edge copies (extrude/revolve/loft/sweep class)

The trap: `EdgeCurve::Circle` and `EdgeCurve::Ellipse` are periodic. An edge stores start and end vertices; the parameter span is recovered by `EdgeCurve::domain_with_endpoints(start, end)` (`rg -n 'pub fn domain_with_endpoints' crates/topology/src/edge.rs`). Its contract:

- start and end coincident (within ~1e-9): full period `[0, 2*PI]`.
- otherwise: project both points onto the curve and take the CCW span from start.

So if a new edge's stored vertex order is swapped relative to the curve's parameterization, the CCW rule recovers the complementary sub-arc: you asked for the 90-degree arc and got the 270-degree one. Everything still compiles, validates, and often tessellates without holes.

Why it hides: for a forward `OrientedEdge`, stored order equals wire order and the span comes out right. Only a reversed edge in the source wire swaps the endpoints. A test suite with only forward arcs passes forever.

The fixed instance to copy from: `extrude_wire_vertices_with` in `crates/operations/src/extrude.rs` (`rg -n 'reverse_edge_curve' crates/operations/src/extrude.rs`). It builds the top edge's curve via `translate_edge_curve`, then:

```rust
if !oriented[i].is_forward() {
    top_curve = reverse_edge_curve(&top_curve)?;
}
```

`reverse_edge_curve` flips a circle/ellipse's normal and v-axis so projection maps theta to minus theta, and reverses a NURBS curve's control points, weights, and mirrored knots.

Rule for a new op: if it copies an edge and preserves its curve type, either store the vertices in the order matching the copied curve's parameterization, or reverse the curve for reversed source edges as extrude does. Ops that rebuild connector edges as `EdgeCurve::Line` or as fresh arcs from vertex positions do not hit this.

Required guard test: extrude (or apply your op to) a half-circle profile where the arc edge enters the wire reversed, and assert volume within 1% of the closed form. Live examples: `extrude_half_circle_reversed_edge_volume` and `extrude_half_ellipse_reversed_edge_volume` in `crates/operations/src/extrude/tests.rs`.

## Cone conventions

`rg -n 'radial plane' crates/math/src/surfaces.rs` and `rg -n 'atan2' crates/operations/src/primitives.rs`

- `ConicalSurface` in `crates/math/src/surfaces.rs`: `P(u,v) = apex + v*(cos(half_angle)*radial(u) + sin(half_angle)*axis)`. `half_angle` is measured from the radial plane to the generator, NOT from the axis. The constructor rejects values outside `(0, PI/2)`.
- `make_cone` in `crates/operations/src/primitives.rs`: pointed cone `half_angle = height.atan2(r_big)`; frustum `(axial_to_apex + height).atan2(r_big)` with `axial_to_apex = r_small * height / (r_big - r_small)`. The surface axis points from apex toward the base.
- Degenerate half-angles (at or beyond the `(0, PI/2)` bounds) fall back to revolving a trapezoid profile.
- If you construct boundary vertices for a face on an analytic surface, they must lie ON that surface. The all-analytic volume fast path integrates the surface, not the vertices, so off-surface vertices produce a plausible-looking volume that masks the placement error.

## Primitive placement

All primitives in `crates/operations/src/primitives.rs` place their base at z = 0: `make_cylinder` extends from z = 0 to z = height; `make_cone` has `bottom_radius` at z = 0. Nothing is centered on the origin. Wrong expected volumes and bboxes in tests usually trace to assuming centered placement.

## OperationsError shape

`rg -n 'error\(transparent\)' crates/operations/src/lib.rs`

`OperationsError` has multiple `#[error(transparent)] #[from]` variants (topology, math, and others). Two consequences:

- `?` works directly on lower-layer results; do not hand-wrap.
- Closures that return a `Result` need an explicit annotation, `|x| -> Result<_, OperationsError> { ... }`, because inference cannot pick among the `From` impls (CLAUDE.md Common Pitfalls has the snippet).

## Symptom-to-cause table

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| Volume ~3x expected on an arc profile, or 1/3 | Complementary periodic arc recovered | Reversed-edge handling; see "Periodic edge copies" |
| Op correct on solid box, wrong on hollow solid | Face walk used `outer_shell()` only | Switch to `explorer::solid_faces` |
| Volume slightly high but classify says probe points are Outside where material should be | Op did not actually modify the solid, or a face was dropped | classify_point probes + face count before/after |
| `validate_solid` errors only on NURBS-faced results | Default tolerance too tight for approximated faces | `validate_solid_with_options`, tolerance_scale ~10 |
| Watertight by `is_watertight` but a slicer sees holes | Index-shared but positionally split vertices, or the reverse | Positional-weld check from `tessellate_watertight.rs` |
| Census shows a fallback for an op that should be analytic | Surface type degraded during the op | analytic-preservation skill |
| Cone face normal or point evaluation off by a complement | half_angle treated as angle from axis | "Cone conventions" above |
| Borrow-checker fight when copying entities | Reading and allocating in one expression | Snapshot-then-allocate, CLAUDE.md Common Pitfalls |
| Volume off by a translation-dependent amount | Assumed centered primitive | Base is at z = 0 |
