# brepkit-blend: OCCT-Parity Fillet/Chamfer Engine

**Date:** 2026-03-19
**Status:** Approved
**Scope:** New `brepkit-blend` crate at L1.5, replacing current fillet/chamfer in operations

## Problem

brepkit's fillet engine uses a "sample cross-sections → fit NURBS" approach that works
for planar faces but cannot trim curved faces, produces flat polygon vertex blends instead
of true surface blends, and lacks distance-angle chamfer support. OCCT's BRepFilletAPI is
the industry reference with 30+ years of refinement across 6 packages (~15K LOC).

## Goal

Replace the current fillet/chamfer implementation with an OCCT-style walking-based engine
that handles all surface combinations, produces proper vertex blends, supports distance-angle
chamfer, and provides detailed failure diagnostics.

## Architecture

### Layer Position

New crate `brepkit-blend` at L1.5, same level as `brepkit-algo`.

**Dependency rules:**

| Crate | Allowed deps |
|-------|-------------|
| `brepkit-blend` | `brepkit-math`, `brepkit-topology` |
| `brepkit-operations` | `brepkit-math`, `brepkit-topology`, `brepkit-algo`, `brepkit-blend` |

`brepkit-blend` does NOT depend on `brepkit-algo` or `brepkit-operations`.

### Crate Structure (flattened)

```
crates/blend/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API, BlendError, BlendResult, re-exports
│   ├── spine.rs            # Spine: edge contour / guideline with parameter mapping
│   ├── stripe.rs           # Stripe: fillet band + SurfData + CommonPoint
│   ├── section.rs          # CircSection: cross-section at parameter t
│   ├── blend_func.rs       # BlendFunction trait + ConstRad + EvolRad + Chamfer impls
│   ├── walker.rs           # Newton-Raphson walking engine + NURBS approximation
│   ├── analytic.rs         # Analytic fast paths (plane-plane, plane-cyl, plane-cone, cyl-cyl)
│   ├── fillet_builder.rs   # FilletBuilder: orchestrates fillet computation
│   ├── chamfer_builder.rs  # ChamferBuilder: orchestrates chamfer computation
│   ├── corner.rs           # Vertex blend / corner solver (sphere/torus + Coons patch)
│   ├── trimmer.rs          # Face trimming along contact curves + PCurve computation
│   └── radius_law.rs       # RadiusLaw: Constant, Linear, SCurve, Custom
```

## Core Algorithm: Walking Engine

### Blend Constraint Equation

At spine parameter `t`, the fillet surface satisfies a system of 4 equations in 4 unknowns
`(u1, v1, u2, v2)`:

1. Contact point `P1 = S1(u1, v1)` lies on Surface 1
2. Contact point `P2 = S2(u2, v2)` lies on Surface 2
3. Ball center `C = P1 + R·N1 = P2 + R·N2` (equidistant from both surfaces along normals)
4. `C` lies on the normal plane to the spine at parameter `t`

For constant-radius fillet, `R` is fixed. For variable-radius, `R = law(t)`.

### BlendFunction Trait

```rust
pub trait BlendFunction {
    /// Evaluate constraint residual F(u1,v1,u2,v2,t) -> [f64; 4]
    fn value(&self, params: &BlendParams, t: f64) -> [f64; 4];

    /// Evaluate 4x4 Jacobian dF/d(u1,v1,u2,v2)
    fn jacobian(&self, params: &BlendParams, t: f64) -> [[f64; 4]; 4];

    /// Extract cross-section from a converged solution
    fn section(&self, params: &BlendParams, t: f64) -> CircSection;

    /// Check if solution exits the surface domain
    fn is_tangent_exit(&self, params: &BlendParams) -> bool;
}
```

**Implementations:**

| Struct | Use | Constraint |
|--------|-----|-----------|
| `ConstRadBlend` | Constant-radius fillet | Ball radius = R (fixed) |
| `EvolRadBlend` | Variable-radius fillet | Ball radius = law(t) |
| `ChamferBlend` | Two-distance chamfer | Contact distances d1, d2 from edge |
| `ChamferAngleBlend` | Distance-angle chamfer | Distance d from edge, angle α with S1 |

Each implementation provides analytic Jacobians for its specific constraint. The Jacobian
computation requires surface first derivatives `dS/du`, `dS/dv` and normals.

### Prerequisite: ParametricSurface Trait Extension

The current `ParametricSurface` trait in `brepkit-math` exposes only `evaluate`, `normal`,
and `project_point`. The walker's Jacobian requires partial derivatives `dS/du` and `dS/dv`.

**Required addition to `ParametricSurface`:**

```rust
/// Partial derivative with respect to u at (u, v)
fn partial_u(&self, u: f64, v: f64) -> Vec3;

/// Partial derivative with respect to v at (u, v)
fn partial_v(&self, u: f64, v: f64) -> Vec3;
```

This must be implemented for all 6 surface types:
- **Plane:** constant vectors (trivial)
- **Cylinder:** `(-r·sin(u), r·cos(u), 0)` and `(0, 0, 1)` in local frame
- **Cone:** similar trigonometric derivatives
- **Sphere:** standard spherical coordinate derivatives
- **Torus:** standard toric coordinate derivatives
- **NurbsSurface:** already has `derivatives()` method, wrap it

Estimated: ~150 LOC across `traits.rs` + surface implementations. This is a ripple-effect
change (all `ParametricSurface` implementors) but straightforward analytically.

### Walker

```rust
pub struct Walker<'a, F: BlendFunction> {
    func: &'a F,
    surf1: &'a dyn ParametricSurface,
    surf2: &'a dyn ParametricSurface,
    spine: &'a Spine,
    tol_3d: f64,      // 3D tolerance for convergence
    tol_2d: f64,      // 2D (parameter space) tolerance
    max_step: f64,     // Maximum step along spine
    min_step: f64,     // Minimum step (below this → failure)
}
```

**Walking algorithm:**

1. **Find start solution** at spine parameter `t0`:
   - Initial guess from surface normals at the edge midpoint
   - Newton-Raphson to converge: `params -= J^{-1} · F(params, t0)`
   - Typically 3-5 iterations to converge within `tol_3d`

2. **March forward** from `t0` to `t_end`:
   - Predict next point via linear extrapolation from last two solutions
   - Correct via Newton-Raphson at new `t`
   - **Adaptive step control:**
     - Accept if residual < `tol_3d` → try larger step next
     - Reject if diverges → halve step and retry
     - Fail if step < `min_step`
   - **Domain boundary check:** if solution exits surface UV domain, snap to boundary
   - **Collect:** `(t, u1, v1, u2, v2, P1, P2, C)` at each accepted step

3. **March backward** from `t0` to `t_start` (same algorithm, negative direction)

4. **NURBS approximation** of the blend surface:
   - U-direction: rational quadratic circular arc (exact, weights `[1, cos(α/2), 1]`)
   - V-direction: cubic B-spline fit through walked cross-sections
   - Result: degree (2, 3) NURBS surface

**Error conditions:**
- Newton divergence → `WalkingFailure`
- No initial solution → `StartSolutionFailure`
- Self-intersecting result → `TwistedSurface`

## Analytic Fast Paths

For common surface pairs, bypass walking entirely with closed-form solutions:

| Surface 1 | Surface 2 | Fillet Result | Method |
|-----------|-----------|---------------|--------|
| Plane | Plane | Cylinder | Offset planes intersect → cylinder axis + radius |
| Plane | Cylinder | Torus or Cylinder | Offset plane/cylinder → toric section |
| Plane | Cone | Torus | Offset plane/cone → toric section |
| Cylinder | Cylinder | Torus | Offset cylinders → toric intersection |

The `analytic.rs` module dispatches based on `(FaceSurface, FaceSurface)` variant pair.
Returns the same `StripeResult` as the walker, so the builder treats both paths identically.

These handle ~80% of real-world fillets and are 10-100x faster than walking.

Sphere and Torus surface pairs do not have analytic fast paths — they fall through to the
walking engine. These could be added as future optimizations but are less common in practice.

## Spine: Edge Contour

```rust
pub struct Spine {
    /// Ordered edge chain (may include multiple edges for G1 chains)
    edges: Vec<EdgeId>,
    /// Cumulative arc-length parameter mapping
    params: Vec<f64>,
    /// Total arc length
    length: f64,
    /// Whether the chain is closed (forms a loop)
    is_closed: bool,
}
```

**G1 chain propagation:** Starting from user-specified seed edges, the builder expands
through tangent-continuous neighbors (anti-parallel tangent threshold: `cos(170°) ≈ -0.985`).
Each tangent chain becomes one Spine.

## Stripe: Fillet Band

```rust
pub struct Stripe {
    pub spine: Spine,
    /// Blend surface (NURBS or analytic)
    pub surface: FaceSurface,
    /// Contact curve on face 1 (UV-space). NurbsCurve2D for walked paths,
    /// or Line2D/Circle2D for analytic fast paths.
    pub pcurve1: Curve2D,
    /// Contact curve on face 2 (UV-space). Same typing rationale.
    pub pcurve2: Curve2D,
    /// 3D contact curve on face 1
    pub contact1: NurbsCurve,
    /// 3D contact curve on face 2
    pub contact2: NurbsCurve,
    /// Face IDs for the two adjacent faces
    pub face1: FaceId,
    pub face2: FaceId,
}
```

## Vertex Blend / Corner Solver

At vertices where 2+ fillet stripes meet, we need corner patches.

**Classification:**

| Vertex type | # edges | Strategy |
|-------------|---------|----------|
| 2-edge corner | 2 | Intersect stripe boundaries, extend/trim |
| 3-edge symmetric | 3 | Spherical cap (exact) |
| 3-edge asymmetric | 3 | Coons patch interpolation |
| N-edge (N ≥ 4) | N | Iterative Coons with G1 matching |

**Full OCCT corner solver approach:**

1. Collect stripe boundary curves at the vertex
2. For sphere/torus cases (equal radii, orthogonal faces): exact analytic surface
3. For general cases:
   a. Compute boundary curves from stripe endpoints
   b. Build Coons patch interpolating between boundaries
   c. Project onto constraint (G1 tangent match with adjacent stripes)
   d. Fit as NURBS surface
4. Compute PCurves on all adjacent original faces

The corner solver is the most complex component (~800-1000 LOC) but produces smooth,
tangent-continuous corners instead of flat polygonal patches. N-edge corners (N ≥ 4) are
the hardest case and may require iterative refinement.

## Face Trimming

After computing blend surfaces and contact curves, original faces must be trimmed.

### Contact Curve Extraction

During walking, contact points `P1(u1,v1)` on each face are collected. These form a
polyline in UV space that gets fit as a NURBS 2D curve — the **contact PCurve**.

### Face Splitting Strategy

Three cases:

| Case | Description | Method |
|------|-------------|--------|
| Edge-to-edge | Contact curve connects two boundary edges | Split boundary edges at contact endpoints, partition wire |
| Edge-to-vertex | Contact hits existing vertex | Split one boundary edge |
| Closed loop | Full-circumference fillet | Insert as inner wire (hole) |

### Periodic Face Handling

For cylindrical/spherical/toroidal faces with periodic UV domains:

1. Unwrap periodic parameters to continuous range
2. Split boundary edges at contact endpoints using 3D→parameter projection
3. Build new wire loops with original boundary segments + contact curve
4. Create new Face with trimmed wire, same FaceSurface, new PCurves

### Shared UV Utilities (extracted to brepkit-topology)

The following utilities are moved from `brepkit-algo` to `brepkit-topology` so both
`brepkit-algo` and `brepkit-blend` can use them:

- `quantize_uv_periodic()` — normalize UV coordinates for periodic surfaces
- `fit_pcurve_from_3d_samples_on_surface()` — fit 2D NURBS PCurve from 3D sample points
- Wire loop construction utilities for UV-space face building

This is a clean refactor: these functions operate on topology primitives (Face, Edge, PCurve)
and belong at L1. Note: `fit_pcurve_from_3d_samples_on_surface` calls into `brepkit-math`
NURBS fitting (which topology already depends on), so no new dependencies are introduced.
The functions must be audited to confirm they don't transitively need anything from `algo`
or `operations` before extraction.

## Chamfer v2

The chamfer builder reuses the walking engine with chamfer-specific blend functions.

### Modes

| Mode | Parameters | Constraint |
|------|-----------|-----------|
| Two-distance | `d1, d2` | Contact on S1 at distance d1 from edge, on S2 at distance d2 |
| Distance-angle | `d, α` | Contact on S1 at distance d, chamfer plane angle α with S1 |

### Chamfer Surface

The surface between contact curves is a **ruled surface** (degree 1 in one direction):
- For planar faces: flat quad (same as current behavior)
- For curved faces: ruled NURBS surface

### Shared Infrastructure

ChamferBuilder reuses: Spine, Walker, Trimmer, Corner solver.
Only the blend function differs.

## Error Handling

```rust
pub enum BlendError {
    StartSolutionFailure { edge: EdgeId, t: f64 },
    WalkingFailure { edge: EdgeId, t: f64, residual: f64 },
    TwistedSurface { stripe_idx: usize },
    RadiusTooLarge { edge: EdgeId, max_radius: f64 },
    TrimmingFailure { face: FaceId },
    CornerFailure { vertex: VertexId },
    UnsupportedSurface { face: FaceId, surface_tag: SurfaceTypeTag },
}

pub struct BlendResult {
    pub solid: SolidId,
    pub succeeded: Vec<EdgeId>,
    pub failed: Vec<(EdgeId, BlendError)>,
    pub is_partial: bool,
}
```

Partial results: if some edges succeed and others fail, the builder returns a partial
result with `is_partial = true`. The user can inspect `failed` for diagnostics and decide
whether to accept the partial result.

## Public API (in brepkit-operations)

```rust
// New v2 API
pub fn fillet_v2(
    topo: &mut Topology, solid: SolidId, edges: &[EdgeId], radius: f64,
) -> Result<BlendResult, OperationsError>;

pub fn fillet_variable_v2(
    topo: &mut Topology, solid: SolidId, edge_laws: &[(EdgeId, RadiusLaw)],
) -> Result<BlendResult, OperationsError>;

pub fn chamfer_v2(
    topo: &mut Topology, solid: SolidId, edges: &[EdgeId], d1: f64, d2: f64,
) -> Result<BlendResult, OperationsError>;

pub fn chamfer_distance_angle(
    topo: &mut Topology, solid: SolidId, edges: &[EdgeId], distance: f64, angle: f64,
) -> Result<BlendResult, OperationsError>;
```

Old API (`fillet()`, `fillet_rolling_ball()`, `chamfer()`, `chamfer_asymmetric()`) kept as
deprecated wrappers, removed after v2 validation.

WASM bindings updated to expose new API with `_v2` suffix initially, then renamed after
old code removal.

## Scope Estimate

| Module | LOC | Complexity |
|--------|-----|------------|
| `spine.rs` | ~200 | Low |
| `stripe.rs` | ~300 | Medium |
| `section.rs` | ~100 | Low |
| `blend_func.rs` | ~600 | High |
| `walker.rs` | ~800 | High |
| `analytic.rs` | ~600 | Medium |
| `fillet_builder.rs` | ~500 | Medium |
| `chamfer_builder.rs` | ~300 | Medium |
| `corner.rs` | ~900 | High |
| `trimmer.rs` | ~500 | High |
| `radius_law.rs` | ~100 | Low |
| `lib.rs` | ~100 | Low |
| **Crate subtotal** | **~5,000** | |
| Topology UV extraction | ~300 | Medium |
| Operations wrappers + WASM | ~400 | Low |
| ParametricSurface extension | ~150 | Low |
| Tests | ~2,000 | Medium |
| **Grand total** | **~7,850** | |

## Infrastructure Updates Required

These changes outside `brepkit-blend` are prerequisites or co-requisites:

1. **`scripts/check-boundaries.sh`** — Add `brepkit-blend` to the checked crate list:
   ```
   check_deps "blend" "brepkit-math" "brepkit-topology"
   ```
   Update the `operations` line to include `"brepkit-blend"`.

2. **`CLAUDE.md`** — Update the layer dependency table and "Allowed `use` paths" section
   to include `brepkit-blend` at L1.5.

3. **`ParametricSurface` trait** — Add `partial_u` and `partial_v` methods (see prerequisite
   section above).

4. **UV utility extraction** — Move shared functions from `brepkit-algo` to `brepkit-topology`
   (audit dependencies first).

## Development Strategy

Single large PR on a feature branch. Build order:

1. ParametricSurface trait extension (partial_u, partial_v on all surfaces)
2. Crate scaffold + data structures (spine, stripe, section, radius_law)
3. Blend functions (ConstRad + Chamfer) + walker
4. Analytic fast paths (plane-plane, plane-cylinder, plane-cone, cyl-cyl)
5. UV utility extraction (topology crate)
6. Fillet builder + trimmer
7. Chamfer builder
8. Corner solver (vertex blends)
9. Operations wrappers + WASM bindings
10. Tests (port existing + new curved-surface cases)
11. Infrastructure updates (check-boundaries.sh, CLAUDE.md)
12. Delete old fillet.rs / chamfer.rs code

## OCCT Reference Files

Key OCCT source files for reference during implementation:

| brepkit module | OCCT reference |
|---------------|---------------|
| `spine.rs` | `src/ModelingAlgorithms/TKFillet/ChFiDS/ChFiDS_Spine.hxx` |
| `stripe.rs` | `src/ModelingAlgorithms/TKFillet/ChFiDS/ChFiDS_Stripe.hxx` |
| `blend_func.rs` | `src/ModelingAlgorithms/TKFillet/BlendFunc/BlendFunc_ConstRad.hxx` |
| `walker.rs` | `src/ModelingAlgorithms/TKFillet/BRepBlend/BRepBlend_Walking.hxx` |
| `analytic.rs` | `src/ModelingAlgorithms/TKFillet/ChFiKPart/ChFiKPart_ComputeData.hxx` |
| `fillet_builder.rs` | `src/ModelingAlgorithms/TKFillet/ChFi3d/ChFi3d_FilBuilder.hxx` |
| `chamfer_builder.rs` | `src/ModelingAlgorithms/TKFillet/ChFi3d/ChFi3d_ChBuilder.hxx` |
| `corner.rs` | `src/ModelingAlgorithms/TKFillet/ChFi3d/ChFi3d_Builder_6.cxx` |
| `trimmer.rs` | `src/ModelingAlgorithms/TKFillet/ChFi3d/ChFi3d_Builder_0.cxx` |

OCCT source is at `~/Git/OCCT/src/`.
