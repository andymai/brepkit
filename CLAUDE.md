# brepkit — Project Guidelines

brepkit is the B-Rep modeling engine behind brepjs. It provides the computational
backend (geometry, booleans, tessellation, I/O) while brepjs provides the
developer-facing TypeScript API.

## Architecture

Strict layered Cargo workspace. Each layer depends only on layers below it.

```
L4: brepkit-wasm        → JS bindings (wasm-bindgen)
L3: brepkit-io          → STEP, 3MF, STL, IGES, OBJ, PLY, glTF import/export
L3: brepkit-operations  → Booleans, fillets, extrusions, tessellation
L2: brepkit-algo        → GFA boolean engine, classification, intersection
L2: brepkit-blend       → Walking-based fillet and chamfer engine
L2: brepkit-check       → Classification, validation, properties, distance
L2: brepkit-heal        → Shape healing (analysis, fixing, upgrading)
L2: brepkit-offset      → Solid offset engine (global face-face intersection)
L2: brepkit-sketch      → 2D parametric constraint solver (GCS)
L1: brepkit-topology    → B-Rep data structures (arena-based)
L1: brepkit-geometry    → Curve sampling, extrema, geometry conversion
L0: brepkit-math        → Vectors, matrices, NURBS, predicates
```

### Layer dependency rules

Enforced by `scripts/check-boundaries.sh` — run before pushing:

| Crate | Allowed deps |
|-------|-------------|
| `math` | *(none — no workspace deps)* |
| `geometry` | `math` |
| `topology` | `math` |
| `algo` | `math`, `topology` |
| `blend` | `math`, `topology` |
| `heal` | `math`, `topology`, `geometry` |
| `check` | `math`, `topology`, `geometry` |
| `offset` | `math`, `topology`, `geometry` |
| `sketch` | *(none — no workspace deps)* |
| `operations` | `math`, `topology`, `algo`, `blend`, `heal`, `check`, `geometry`, `offset`, `sketch` |
| `io` | `math`, `topology`, `operations` |
| `wasm` | all crates (incl. `blend`, `check`, `heal`) |

The script checks `[dependencies]` in each `Cargo.toml`. A violation fails the pre-push hook.

**Allowed `use` paths per crate:**
- `math/src/**` → only `std`, external crates
- `geometry/src/**` → `brepkit_math::*`
- `topology/src/**` → `brepkit_math::*`
- `algo/src/**` → `brepkit_math::*`, `brepkit_topology::*`
- `blend/src/**` → `brepkit_math::*`, `brepkit_topology::*`
- `heal/src/**` → `brepkit_math::*`, `brepkit_topology::*`, `brepkit_geometry::*`
- `check/src/**` → `brepkit_math::*`, `brepkit_topology::*`, `brepkit_geometry::*`
- `offset/src/**` → `brepkit_math::*`, `brepkit_topology::*`, `brepkit_geometry::*`
- `sketch/src/**` → only `std`, external crates
- `operations/src/**` → `brepkit_math::*`, `brepkit_topology::*`, `brepkit_geometry::*`, `brepkit_algo::*`, `brepkit_blend::*`, `brepkit_heal::*`, `brepkit_check::*`, `brepkit_offset::*`, `brepkit_sketch::*`
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
| 3D curves (Line, Circle, Ellipse, Parabola, Hyperbola) | `curves.rs` |
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
| NURBS basis function evaluation | `nurbs/basis.rs` |
| Surface evaluator (power-basis cache) | `nurbs/evaluator.rs` |
| Polynomial power-basis (Horner evaluation) | `nurbs/power_basis.rs` |
| Orthonormal reference frame | `frame.rs` |
| Oriented bounding box (PCA + SAT) | `obb.rs` |
| Chord deviation arc discretization | `chord.rs` |
| Gauss-Legendre quadrature | `quadrature.rs` |

### L1: geometry (`crates/geometry/src/`)
| Task | File(s) |
|------|---------|
| Uniform curve sampling | `sampling/uniform.rs` |
| Deflection-adaptive sampling | `sampling/deflection.rs` |
| Arc-length-uniform sampling | `sampling/arc_length.rs` |
| Curvature-adaptive sampling (NURBS) | `sampling/curvature.rs` |
| Surface grid sampling | `sampling/surface.rs` |
| Point-to-curve projection | `extrema/point_curve.rs` |
| Curve-to-curve distance | `extrema/curve_curve.rs` |
| Point-to-surface distance (analytic) | `extrema/point_surface.rs` |
| Segment-segment distance | `extrema/segment.rs` |
| Lipschitz global optimizer | `extrema/lipschitz.rs` |
| Circle/Ellipse/Line → NURBS | `convert/curve_to_nurbs.rs` |
| Analytic surfaces → NURBS | `convert/surface_to_nurbs.rs` |
| NURBS → analytic curve recognition | `convert/recognize_curve.rs` |
| NURBS → analytic surface recognition | `convert/recognize_surface.rs` |
| Error types | `error.rs` |

### L1: topology (`crates/topology/src/`)
| Task | File(s) |
|------|---------|
| Arena & typed `Id<T>` handles | `arena.rs` |
| `Topology` struct (the arena owner) | `topology.rs` |
| Vertex, Edge, Wire, Face, Shell, Solid | `vertex.rs`, `edge.rs`, `wire.rs`, `face.rs`, `shell.rs`, `solid.rs` |
| Compound, CompSolid | `compound.rs`, `compsolid.rs` |
| Adjacency index (edge-to-face, face neighbors) | `adjacency.rs` |
| Builder helpers | `builder.rs` |
| Shape explorer (iterate children) | `explorer.rs` |
| PCurve registry | `pcurve.rs` |
| Topology validation | `validation.rs` |
| Test utilities (`test-utils` feature) | `test_utils.rs` |

### L2: algo (`crates/algo/src/`)
| Task | File(s) |
|------|---------|
| GFA entry point | `gfa.rs` |
| Boolean operation selection | `bop.rs` |
| Error types | `error.rs` |
| GFA data structures (Pave, PaveBlock, GfaArena) | `ds/pave.rs`, `ds/arena.rs` |
| Intersection curve data | `ds/curve.rs` |
| Face classification state | `ds/face_info.rs` |
| PaveFiller orchestrator | `pave_filler/mod.rs` |
| Phases VV/VE/EE/VF/EF/FF | `pave_filler/phase_*.rs` |
| Pave block splitting + edge creation | `pave_filler/make_blocks.rs`, `make_split_edges.rs` |
| FaceInfo population | `pave_filler/fill_face_info.rs` |
| Builder (face splitting + assembly) | `builder/mod.rs`, `builder/assemble.rs` |
| Face splitting (UV-space) | `builder/face_splitter.rs`, `builder/classify_2d.rs` |
| PCurve computation | `builder/pcurve_compute.rs` |
| Plane frame (3D↔UV projection) | `builder/plane_frame.rs` |
| Wire loop reconstruction | `builder/wire_builder.rs` |
| Split types + face class | `builder/split_types.rs`, `builder/face_class.rs` |
| Face image population | `builder/fill_images.rs`, `builder/fill_images_faces.rs` |
| Same-domain face merging | `builder/same_domain.rs` |
| Interference indexing | `ds/interference.rs`, `ds/shape_index.rs` |
| Analytic classifier (7 variants) | `classifier/analytic.rs` |
| Ray-cast classifier | `classifier/ray_cast.rs` |

### L2: blend (`crates/blend/src/`)
| Task | File(s) |
|------|---------|
| Public API, `BlendError`, `BlendResult` | `lib.rs` |
| Radius law (constant, linear, S-curve, custom) | `radius_law.rs` |
| Spine (edge chain, arc-length parameterization) | `spine.rs` |
| Cross-section (contact points, center, radius) | `section.rs` |
| Stripe (fillet band, contact curves, PCurves) | `stripe.rs` |
| Blend constraint functions | `blend_func.rs` |
| Newton-Raphson walking engine | `walker.rs` |
| Analytic fast paths (plane-plane, plane-cyl, etc.) | `analytic.rs` |
| Fillet builder (orchestration) | `fillet_builder.rs` |
| Chamfer builder (orchestration) | `chamfer_builder.rs` |
| Vertex blend / corner solver | `corner.rs` |
| Face trimming along contact curves | `trimmer.rs` |
| Shared builder utilities | `builder_utils.rs` |

### L2: heal (`crates/heal/src/`)
| Task | File(s) |
|------|---------|
| Public API, `HealError`, `Status` | `lib.rs`, `error.rs`, `status.rs` |
| Healing context (tolerance, reshape, messages) | `context.rs` |
| Entity replacement/removal tracking | `reshape.rs` |
| Edge analysis (3D/PCurve deviation, degeneracy) | `analysis/edge.rs` |
| Wire analysis (ordering, closure, gaps, small edges) | `analysis/wire.rs` |
| Surface analysis (singularity, closure, equivalence) | `analysis/surface.rs` |
| Face analysis (small face, degenerate face) | `analysis/face.rs` |
| Shell analysis (manifold, free edges, orientation) | `analysis/shell.rs` |
| Curve analysis (length, continuity) | `analysis/curve.rs` |
| Free boundary detection | `analysis/free_bounds.rs` |
| Wire edge ordering | `analysis/wire_order.rs` |
| Tolerance statistics | `analysis/tolerance.rs` |
| Entity counting | `analysis/contents.rs` |
| Fix config (tri-state FixMode per fix type) | `fix/config.rs` |
| Fix orchestrator (shape → solid → shell → face → wire → edge) | `fix/mod.rs` |
| Edge fixing (SameParameter, vertex tolerance) | `fix/edge.rs` |
| Wire fixing (30+ fixes: reorder, gaps, small, degenerate) | `fix/wire.rs` |
| Face fixing (orientation, small area, seam insertion) | `fix/face.rs` |
| Shell fixing (orientation consistency) | `fix/shell.rs` |
| Solid fixing (top-level orchestrator) | `fix/solid.rs` |
| Small face removal | `fix/small_face.rs` |
| Wireframe repair | `fix/wireframe.rs` |
| Split over-connected vertices | `fix/split_vertex.rs` |
| Unify same-domain faces (merge coplanar/co-cylindrical) | `upgrade/unify_same_domain.rs` |
| Curve splitting at continuity breaks | `upgrade/split_curve.rs` |
| Surface splitting | `upgrade/split_surface.rs` |
| NURBS → Bezier conversion | `upgrade/convert_to_bezier.rs` |
| Remove internal wires | `upgrade/remove_internal_wires.rs` |
| Shell sewing (stitch free edges) | `upgrade/shell_sewing.rs` |
| 3D → 2D PCurve projection | `construct/project_curve.rs` |
| Curve type conversions | `construct/convert_curve.rs` |
| Surface type conversions | `construct/convert_surface.rs` |
| Convert all to B-Spline | `custom/convert_to_bspline.rs` |
| B-Spline degree/continuity restriction | `custom/bspline_restriction.rs` |
| Recognize NURBS as elementary surfaces | `custom/convert_to_elementary.rs` |
| HealOperator trait | `pipeline/operator.rs` |
| Operator registry | `pipeline/registry.rs` |
| Configurable pipeline executor | `pipeline/process.rs` |
| 13 built-in operators | `pipeline/builtin.rs` |

### L2: check (`crates/check/src/`)
| Task | File(s) |
|------|---------|
| Public API, `CheckError` | `lib.rs`, `error.rs` |
| Shared utilities (face polygon, AABB, edge sampling) | `util.rs` |
| Ray-surface intersection (all types + solvers) | `classify/ray_surface.rs` |
| UV boundary polygon, containment tests | `classify/boundary.rs` |
| Point-in-solid classification (ray casting) | `classify/mod.rs` |
| Winding number classifier | `classify/winding.rs` |
| CheckId enum, ValidationReport, severity | `validate/checks.rs` |
| Wire topological checks | `validate/wire.rs` |
| Shell topological checks | `validate/shell.rs` |
| Solid checks (Euler, duplicate faces) | `validate/solid.rs` |
| Vertex geometric checks | `validate/vertex.rs` |
| Edge geometric checks | `validate/edge.rs` |
| Face geometric checks | `validate/face.rs` |
| Validation orchestrator | `validate/mod.rs` |
| GProps accumulator (Huygens' theorem) | `properties/accumulator.rs` |
| Closed-form formulas (box, sphere, cylinder, etc.) | `properties/analytic.rs` |
| AABB computation | `properties/bbox.rs` |
| Face Gauss integration | `properties/face_integrator.rs` |
| Properties orchestrator (volume, area, CoM) | `properties/mod.rs` |
| Point-to-surface distance (all analytic types) | `distance/analytic.rs` |
| Edge-to-edge distance | `distance/edge.rs` |
| Point-to-solid, solid-to-solid distance | `distance/mod.rs` |

### L2: offset (`crates/offset/src/`)
| Task | File(s) |
|------|---------|
| Public API (offset_solid, thick_solid) | `lib.rs` |
| Error types | `error.rs` |
| Central data structures (OffsetData, EdgeClass, etc.) | `data.rs` |
| Edge/vertex convexity classification | `analyse.rs` |
| Per-face surface offset (analytic + NURBS) | `offset.rs` |
| 3D face-face intersection | `inter3d.rs` |
| 2D edge splitting from intersections | `inter2d.rs` |
| Wire loop reconstruction | `loops.rs` |
| Arc joints (pipe + sphere cap) | `arc_joint.rs` |
| Shell assembly + solid creation | `assemble.rs` |
| Self-intersection removal (BOP-based) | `self_int.rs` |

### L2: sketch (`crates/sketch/src/`)
| Task | File(s) |
|------|---------|
| GCS system (CRUD, solve orchestration) | `gcs/system.rs` |
| Constraint types (10 variants, Jacobians) | `gcs/constraint.rs` |
| Entity arena (Points, Lines, Circles) | `gcs/entity.rs` |
| DogLeg trust-region solver | `gcs/solver.rs` |
| QR factorization (rank detection) | `gcs/qr.rs` |
| DOF analysis | `gcs/dof.rs` |
| Error types | `lib.rs` (`SketchError`) |

### L3: operations (`crates/operations/src/`)
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
| Blend v2 wrappers (fillet/chamfer) | `blend_ops.rs` |
| Offset v2 (delegates to brepkit-offset) | `offset_v2.rs` |
| Shared winding utilities | `winding.rs` |

### L3: io (`crates/io/src/`)
| Task | File(s) |
|------|---------|
| STEP read/write | `step/reader.rs`, `step/writer.rs` |
| IGES read/write | `iges/reader.rs`, `iges/writer.rs` |
| STL read/write/import | `stl/reader.rs`, `stl/writer.rs`, `stl/import.rs` |
| 3MF read/write | `threemf/reader.rs`, `threemf/writer.rs` |
| OBJ read/write | `obj/reader.rs`, `obj/writer.rs` |
| PLY read/write | `ply/reader.rs`, `ply/writer.rs` |
| glTF read/write | `gltf/reader.rs`, `gltf/writer.rs` |

### L4: wasm (`crates/wasm/src/`)
| Task | File(s) |
|------|---------|
| `BrepKernel` struct, constructor, private helpers | `kernel.rs` |
| Error types, validation helpers (`validate_positive`, `validate_finite`) | `error.rs` |
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
| Gridfinity integration tests | `bindings/gridfinity_tests.rs` |
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

### Profiling

The `[profile.profiling]` block in `Cargo.toml` inherits from release with debug symbols and no LTO for fast builds with full symbol resolution.

```bash
cargo flamegraph --profile profiling \     # Flamegraph a single benchmark
  --bench cad_operations \
  -p brepkit-operations \
  -o /tmp/flamegraph.svg \
  -- --bench "bench_name_filter"
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
