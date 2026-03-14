# brepkit — Project Guidelines

brepkit is the B-Rep modeling engine behind brepjs. It provides the computational
backend (geometry, booleans, tessellation, I/O) while brepjs provides the
developer-facing TypeScript API.

## Architecture

Strict layered Cargo workspace. Each layer depends only on layers below it.

```
L3: brepkit-wasm        → JS bindings (wasm-bindgen)
L2: brepkit-io          → STEP, 3MF, STL, IGES, OBJ, PLY, glTF import/export
L2: brepkit-operations  → Booleans, fillets, extrusions, tessellation
L1: brepkit-topology    → B-Rep data structures (arena-based)
L0: brepkit-math        → Vectors, matrices, NURBS, predicates
```

### Layer dependency rules

Enforced by `scripts/check-boundaries.sh` — run before pushing:

| Crate | Allowed deps |
|-------|-------------|
| `math` | *(none — no workspace deps)* |
| `topology` | `math` |
| `operations` | `math`, `topology` |
| `io` | `math`, `topology`, `operations` |
| `wasm` | all crates |

The script checks `[dependencies]` in each `Cargo.toml`. A violation fails the pre-push hook.

**Allowed `use` paths per crate:**
- `math/src/**` → only `std`, external crates
- `topology/src/**` → `brepkit_math::*`
- `operations/src/**` → `brepkit_math::*`, `brepkit_topology::*`
- `io/src/**` → `brepkit_math::*`, `brepkit_topology::*`, `brepkit_operations::*`
- `wasm/src/**` → all `brepkit_*`

## Module Map

Quick reference — find the right file for any task:

### L0: math (`crates/math/src/`)
| Task | File(s) |
|------|---------|
| Points, vectors, matrices | `vec.rs`, `mat.rs` |
| Planes | `plane.rs` |
| NURBS curve evaluation/manipulation | `nurbs/curve.rs` |
| NURBS surface evaluation | `nurbs/surface.rs` |
| NURBS knot insertion/removal | `nurbs/knot_ops.rs` |
| Bezier decomposition | `nurbs/decompose.rs` |
| Bezier clipping intersection | `nurbs/bezier_clip.rs` |
| Curve/surface fitting (LSPIA) | `nurbs/fitting.rs`, `nurbs/surface_fitting.rs` |
| Point projection onto curves | `nurbs/projection.rs` |
| Self-intersection detection | `nurbs/self_intersection.rs` |
| 3D curves (Line, Circle, Parabola, Hyperbola) | `curves.rs` |
| 2D curves (Line2D, Circle2D, Ellipse2D) | `curves2d.rs` |
| Analytic surfaces (Cylinder, Cone, Sphere, Torus) | `surfaces.rs` |
| Surface-surface intersection | `nurbs/intersection.rs` |
| Analytic-analytic intersection | `analytic_intersection.rs` |
| AABB / bounding boxes | `aabb.rs` |
| BVH (bounding volume hierarchy) | `bvh.rs` |
| CDT (constrained Delaunay) | `cdt.rs` |
| Convex hull | `convex_hull.rs` |
| Filtered exact predicates | `filtered.rs` |
| Float tolerance | `tolerance.rs` |
| Geometric predicates (orient2d/3d) | `predicates.rs` |
| Ray-triangle intersection | `ray_triangle.rs` |
| 2D polygon offset | `polygon_offset.rs` |
| SIMD batch operations | `simd.rs` |
| Parametric geometry traits | `traits.rs` |

### L1: topology (`crates/topology/src/`)
| Task | File(s) |
|------|---------|
| Arena & typed `Id<T>` handles | `arena.rs` |
| `Topology` struct (the arena owner) | `topology.rs` |
| Vertex, Edge, Wire, Face, Shell, Solid | `vertex.rs`, `edge.rs`, `wire.rs`, `face.rs`, `shell.rs`, `solid.rs` |
| Compound, CompSolid | `compound.rs`, `compsolid.rs` |
| Adjacency graph (half-edge) | `graph.rs` |
| Builder helpers | `builder.rs` |
| Shape explorer (iterate children) | `explorer.rs` |
| PCurve registry | `pcurve.rs` |
| Topology validation | `validation.rs` |
| Test utilities (`test-utils` feature) | `test_utils.rs` |

### L2: operations (`crates/operations/src/`)
| Task | File(s) |
|------|---------|
| Primitives (box, cylinder, cone, sphere, torus) | `primitives.rs` |
| Boolean (union/cut/intersect) | `boolean/` (mod, types, precompute, intersect, split, classify, assembly, analytic, compound, fragments, tests), `nurbs_boolean.rs` |
| Mesh boolean (co-refinement) | `mesh_boolean.rs` |
| Extrude, Revolve, Sweep, Loft, Pipe | `extrude.rs`, `revolve.rs`, `sweep.rs`, `loft.rs`, `pipe.rs` |
| Helical sweep | `helix.rs` |
| Chamfer, Fillet | `chamfer.rs`, `fillet.rs` |
| Shell (hollow solid) | `shell_op.rs` |
| Draft (taper faces) | `draft.rs` |
| Section / Split | `section.rs`, `split.rs` |
| Transform, Mirror, Copy | `transform.rs`, `mirror.rs`, `copy.rs` |
| Measure (bbox, area, volume, CoM) | `measure.rs` |
| Distance queries | `distance.rs` |
| Tessellation | `tessellate.rs` |
| Point classification (in/on/out solid) | `classify.rs` |
| Offset face / solid | `offset_face.rs`, `offset_solid.rs`, `offset_trim.rs` |
| Offset wire | `offset_wire.rs` |
| Face filling (Coons patch) | `fill_face.rs` |
| Shape healing | `heal.rs` |
| Defeaturing | `defeature.rs` |
| Feature recognition | `feature_recognition.rs` |
| Pattern (linear/circular) | `pattern.rs` |
| Sewing | `sew.rs` |
| Thickening | `thicken.rs` |
| Untrimming | `untrim.rs` |
| Validation | `validate.rs` |
| Assembly management | `assembly.rs` |
| 2D sketch constraint solver | `sketch.rs` |
| Evolution tracking | `evolution.rs` |
| Compound operations | `compound_ops.rs` |

### L2: io (`crates/io/src/`)
| Task | File(s) |
|------|---------|
| STEP read/write | `step/reader.rs`, `step/writer.rs` |
| IGES read/write | `iges/reader.rs`, `iges/writer.rs` |
| STL read/write/import | `stl/reader.rs`, `stl/writer.rs`, `stl/import.rs` |
| 3MF read/write | `threemf/reader.rs`, `threemf/writer.rs` |
| OBJ read/write | `obj/reader.rs`, `obj/writer.rs` |
| PLY read/write | `ply/reader.rs`, `ply/writer.rs` |
| glTF read/write | `gltf/reader.rs`, `gltf/writer.rs` |

### L3: wasm (`crates/wasm/src/`)
| Task | File(s) |
|------|---------|
| `BrepKernel` struct, constructor, private helpers | `kernel.rs` |
| Error types, validation newtypes (`Positive`, `Finite`, `CoordArray3`) | `error.rs` |
| Shape type wrappers (`JsMesh`, `JsEdgeLines`, `JsPoint3`, `JsVec3`) | `shapes.rs` |
| Entity handle resolution (`resolve_*`) & ID converters | `handles.rs` |
| Shared free functions, constants (`TOL`), 2D polygon helpers | `helpers.rs` |
| Checkpoint & sketch state structs | `state.rs` |
| Typed result structs (`GroupedMeshResult`, `UvMeshResult`, tsify) | `types.rs` |
| **Binding modules** (`bindings/`) | |
| Primitive solid creation | `bindings/primitives.rs` |
| Shape construction (vertices, edges, wires, faces) | `bindings/shapes.rs` |
| Modeling operations (extrude, revolve, sweep, loft, fillet, etc.) | `bindings/operations.rs` |
| Boolean operations (fuse, cut, intersect) | `bindings/booleans.rs` |
| Transform, copy, mirror, pattern | `bindings/transforms.rs` |
| Topology query, edge/surface evaluation | `bindings/query.rs` |
| Measurement (volume, area, bbox, distances) | `bindings/measure.rs` |
| Tessellation & wireframe | `bindings/tessellate.rs` |
| File I/O import/export (`#[cfg(feature = "io")]`) | `bindings/io.rs` |
| Shape healing, validation, feature recognition | `bindings/heal.rs` |
| Checkpoint / restore | `bindings/checkpoint.rs` |
| 2D sketch constraint solver | `bindings/sketch.rs` |
| Assembly management | `bindings/assembly.rs` |
| 2D polygon operations | `bindings/polygon2d.rs` |
| NURBS curve/surface manipulation | `bindings/nurbs.rs` |
| Batch execution & dispatch | `bindings/batch.rs` |
| **Proc macro crate** (`crates/wasm-macros/`) | |
| `#[wasm_binding]` attribute (panic safety) | `wasm-macros/src/lib.rs` |

## Ripple-Effect Checklists

**These enums appear in `match` arms across many files. Adding a variant requires updating
every match site or the code won't compile (unless a `_ =>` wildcard swallows it silently).**

**Delegate methods reduce ripple scope:** `FaceSurface` has delegate methods (`evaluate`,
`normal`, `project_point`, `estimate_radius`, `type_tag`, `is_planar`, `is_analytic`,
`as_analytic`) and `EdgeCurve` has delegates (`evaluate_with_endpoints`,
`tangent_with_endpoints`, `domain_with_endpoints`, `type_tag`) — see `math/src/traits.rs`.
Call sites using these delegates need no update when adding new variants (only the delegate
impl needs the new arm). The files below still use direct match arms.

### Adding an `EdgeCurve` variant

`EdgeCurve` is defined in `topology/src/edge.rs`. Current variants: `Line`, `NurbsCurve`, `Circle`, `Ellipse`.

Update these files (16+ match sites across 5 crates):

- [ ] `operations/src/tessellate.rs` — sample edge to polyline points
- [ ] `operations/src/transform.rs` — rebuild/transform curve geometry
- [ ] `operations/src/copy.rs` — deep-clone curve data
- [ ] `operations/src/measure.rs` — edge arc-length formula
- [ ] `operations/src/boolean/` — sample edge curve to points (3 sites across sub-modules)
- [ ] `io/src/step/writer.rs` — write as STEP entity
- [ ] `io/src/iges/writer.rs` — write as IGES entity
- [ ] `wasm/src/bindings/query.rs` — type tag, param range, evaluate, edge geometry query
- [ ] `wasm/src/bindings/batch.rs` — batch dispatch match arms
- [ ] `wasm/src/bindings/tessellate.rs` — tessellation dispatch
- [ ] `wasm/src/bindings/nurbs.rs` — NURBS data extraction

Also check (may use wildcards that silently swallow):
- [ ] `io/src/step/reader.rs` — reconstruct from STEP entities
- [ ] `io/src/iges/reader.rs` — reconstruct from IGES entities
- [ ] `operations/src/section.rs` — edge-plane intersection
- [ ] `operations/src/fill_face.rs` — boundary edge sampling

### Adding a `FaceSurface` variant

`FaceSurface` is defined in `topology/src/face.rs`. Current variants: `Plane`, `Nurbs`, `Cylinder`, `Cone`, `Sphere`, `Torus`.

Update these files (22+ match sites across 7+ files):

- [ ] `operations/src/tessellate.rs` — dispatch tessellation strategy
- [ ] `operations/src/transform.rs` — transform surface geometry
- [ ] `operations/src/copy.rs` — deep-clone surface data
- [ ] `operations/src/section.rs` — find intersection segments
- [ ] `operations/src/distance.rs` — point-to-face distance
- [ ] `operations/src/feature_recognition.rs` — classify surface type (2 sites)
- [ ] `operations/src/boolean/` — extract `AnalyticSurface` (sites across sub-modules)
- [ ] `operations/src/offset_face.rs` — offset surface geometry
- [ ] `io/src/step/writer.rs` — write as STEP entity
- [ ] `io/src/iges/writer.rs` — write as IGES entity
- [ ] `wasm/src/bindings/query.rs` — type tag, analytic params, evaluate, domain, project, surface data
- [ ] `wasm/src/bindings/batch.rs` — batch dispatch match arms
- [ ] `wasm/src/bindings/tessellate.rs` — tessellation dispatch
- [ ] `wasm/src/bindings/nurbs.rs` — NURBS extract

Also update if the surface is analytic:
- [ ] `math/src/analytic_intersection.rs` — `AnalyticSurface` enum (4 match sites)
- [ ] `math/src/surfaces.rs` — surface definition

Files that reference `FaceSurface` but typically use pattern matching safely:
- `operations/src/nurbs_boolean.rs`, `split.rs`, `untrim.rs`, `validate.rs`
- `topology/src/builder.rs`, `graph.rs`, `pcurve.rs`, `validation.rs`

## Common Pitfalls

### Borrow checker: "snapshot then allocate"
When copying topology entities, you cannot borrow the arena immutably (to read) and
mutably (to write) at the same time. Read all needed data into local variables first,
then allocate new entities:
```rust
// ✅ Correct: snapshot first
let pos = topo.vertex(vid).position;
let new_vid = topo.add_vertex(pos);

// ❌ Wrong: simultaneous borrow
let new_vid = topo.add_vertex(topo.vertex(vid).position);
```

### Closure return type annotations
When `OperationsError` has multiple `From` impls, closures need explicit return type:
```rust
// ✅ Correct
let f = |x| -> Result<_, OperationsError> { ... };

// ❌ Wrong: compiler can't infer which From impl
let f = |x| { ... };
```

### Lint allows in tests
```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    // ...
}
```

### Complex functions
Add `#[allow(clippy::too_many_lines)]` above complex CAD operations rather than
artificially splitting them.

### WASM binding blocks
Multiple `#[wasm_bindgen] impl BrepKernel` blocks are needed — one for public
JS-exposed methods, one for private helpers. This is a wasm-bindgen requirement.

### Wildcard match arms
As of v1.3.2, all `EdgeCurve` and `FaceSurface` match arms use exhaustive
patterns — no production `_ =>` wildcards remain. When adding a new variant,
the compiler will flag every match site. Still worth a manual scan of these
files since `_ =>` could be re-introduced:
- `io/src/step/writer.rs`
- `io/src/iges/writer.rs`
- `operations/src/offset_face.rs`

### Dev-dependency cycles
Never add `brepkit-operations` as a dev-dependency of `brepkit-topology` — this
creates a "two versions" error. Use the `test-utils` feature flag instead.

## Cookbook: Common Agent Tasks

### Recipe 1: Add a new primitive

Pattern: see `primitives.rs` (`make_box`, `make_cylinder`, etc.)

1. **Create the function** in `operations/src/primitives.rs`:
   ```rust
   pub fn make_thing(topo: &mut Topology, params...) -> Result<SolidId, OperationsError> {
       // Create vertices → edges → wires → faces → shell → solid
   }
   ```
2. **Add WASM binding** in `wasm/src/kernel.rs`:
   - Add method to `impl BrepKernel` with `#[wasm_bindgen]`
   - Use `validate_positive` / `validate_finite` from `error.rs`
3. **Add test** in the same file or a dedicated test module
4. **Add to batch dispatch** in `kernel.rs` `dispatch_op` if applicable

### Recipe 2: Add a new IO format

Pattern: see `obj/` module (simplest), `step/` (most complex)

1. **Create module** `io/src/format/mod.rs`, `reader.rs`, `writer.rs`
2. **Add module** to `io/src/lib.rs`
3. **Writer signature**: `pub fn write_format(topo: &Topology, solid_id: SolidId) -> Result<Vec<u8>, IoError>`
4. **Reader signature**: `pub fn read_format(topo: &mut Topology, data: &[u8]) -> Result<SolidId, IoError>`
5. **Add WASM bindings** `importFormat` / `exportFormat` in `bindings/io.rs`
6. **Add to `executeBatch` dispatch** in `bindings/batch.rs` if commonly used

### Recipe 3: Add a new operation

Pattern: see `extrude.rs` (basic), `boolean.rs` (complex)

1. **Create file** `operations/src/op_name.rs`
2. **Define function**: `pub fn op_name(topo: &mut Topology, ...) -> Result<SolidId, OperationsError>`
3. **Add module** to `operations/src/lib.rs` with `pub mod op_name;`
4. **Add WASM binding** in the appropriate `bindings/` module
5. **Add to batch dispatch** in `bindings/batch.rs` if applicable
6. **Add test** — create a known shape, apply operation, verify with `measure`

### Recipe 4: Add a new WASM binding

Pick the appropriate `bindings/` module (or create a new one). Each module adds
methods to `BrepKernel` via a separate `#[wasm_bindgen] impl` block.

```rust
// In bindings/my_domain.rs:
use wasm_bindgen::prelude::*;
use crate::kernel::BrepKernel;
use crate::error::{WasmError, validate_positive};
use crate::handles::solid_id_to_u32;

#[wasm_bindgen]
impl BrepKernel {
    #[wasm_bindgen(js_name = "myOperation")]
    pub fn my_operation(&mut self, param: f64) -> Result<u32, JsError> {
        validate_positive(param, "param")?;
        let result = some_operation(&mut self.topo, param)?;
        Ok(solid_id_to_u32(result))
    }
}
```

Key points:
- Use `js_name` for camelCase JS naming
- Validate inputs with helpers from `error.rs`
- Return entity IDs as `u32` (auto-converted to JS number)
- Errors use `?` operator (WasmError → JsError via blanket `From` impl)
- Add a `batch_*` companion fn if the op should be in `executeBatch`
- Add contract tests using `execute_batch()` (not direct method calls,
  since `JsError` can't be constructed on non-wasm targets)

## Commands

```bash
cargo build --workspace                    # Build all
cargo test --workspace                     # Test all
cargo clippy --all-targets -- -D warnings  # Lint
cargo fmt --all                            # Format
cargo build -p brepkit-wasm --target wasm32-unknown-unknown  # WASM
./scripts/check-boundaries.sh              # Verify layer deps
```

## Key Patterns

### Error handling
- `thiserror` for typed error enums per crate (`MathError`, `TopologyError`, `OperationsError`, `IoError`)
- Never `unwrap()`, `expect()`, or `panic!()` — return `Result`
- Use `#[error(transparent)]` for error propagation across crate boundaries

### Topology
- Arena-based allocation with typed `Id<T>` handles
- All entities owned by the arena, referenced by ID
- Half-edge / winged-edge adjacency via `graph` module

### Tolerance
- `Tolerance` struct with `linear` (1e-7) and `angular` (1e-12) defaults
- Compare floats via `tolerance.approx_eq(a, b)`, never `==`

### Types
- `Point3` (position) vs `Vec3` (direction) — separate newtypes
- `Mat4` for affine transforms
- NURBS curves/surfaces as the native geometry representation

## Lints

Workspace-level strict lints:
- `unsafe_code = "deny"` — no unsafe
- `unwrap_used = "deny"` — no unwrap
- `panic = "deny"` — no panic
- `clippy::pedantic`, `clippy::nursery` = warn
- `missing_docs` = warn

## Testing

- Unit tests in each module
- `proptest` for property-based testing
- Golden file tests in `tests/golden/`
- Integration tests in `tests/integration/`
- `criterion` benchmarks in `benches/`

## Git Conventions

- Conventional commits enforced by commitlint
- Pre-commit: fmt + clippy (parallel) → test
- Pre-push: full test + cargo-deny
- Branch: `main` is the primary branch
