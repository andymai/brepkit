# brepkit-offset: Offset Engine Design

## Overview

A new L1.5 crate implementing production-grade solid offset, replacing the
current per-face-then-reassemble approach with a global face-face intersection
pipeline. Handles concave/convex geometry correctly, supports intersection and
arc joint types, and uses the GFA boolean engine for self-intersection removal.

## Motivation

brepkit's current offset (`operations/src/offset_solid.rs`) offsets faces
individually then reassembles. This fails on non-trivial geometry because:

1. Adjacent offset faces don't share edges — their boundaries are independently
   computed, leaving gaps or overlaps.
2. Concave edges (where offset faces collide) aren't detected or resolved.
3. No edge classification drives the pipeline — convex vs concave treatment
   is the same.
4. Self-intersection removal is per-face only (SSI-based), not global.

The reference implementation (OCCT `BRepOffset_MakeOffset`) uses a 16-phase
pipeline with global face-face intersection. This design adapts that approach
to Rust/brepkit idioms.

## Architecture

```
brepkit-offset (L1.5)
├── lib.rs          — Public API: offset_solid, thick_solid, OffsetOptions
├── error.rs        — OffsetError enum
├── data.rs         — OffsetData, OffsetFace, EdgeClass, VertexClass
├── analyse.rs      — Edge/vertex convexity classification via dihedral angles
├── offset.rs       — Per-face surface offset (analytic fast paths + NURBS)
├── inter3d.rs      — 3D face-face intersection between adjacent offset faces
├── inter2d.rs      — 2D edge splitting: project intersections onto faces
├── loops.rs        — Wire reconstruction from split edges
├── arc_joint.rs    — Pipe surfaces at convex edges + sphere caps at vertices
├── assemble.rs     — Shell assembly + wall generation + solid creation
└── self_int.rs     — BOP-based self-intersection removal via brepkit-algo
```

### Layer Dependencies

```
offset → math, topology, geometry, algo
```

Operations delegates to offset; WASM bindings call operations.

## Pipeline Phases

### Phase 1: Analyse (`analyse.rs`)

Classify every edge of the input solid as Convex, Concave, or Tangent by
computing the dihedral angle between adjacent faces.

**Algorithm:**
1. For each edge, find the two adjacent faces via the topology graph.
2. Sample face normals at multiple points along the edge.
3. Compute the dihedral angle: `θ = atan2(n1 × n2 · t, n1 · n2)` where
   `t` is the edge tangent direction.
4. Classify: Tangent if |θ - π| < tol_angle, Convex if θ < π, Concave if θ > π.

**Tolerance:** `tol_angle = 4 * arcsin(min(tol / (|offset| * 0.5), 1.0))`
(from OCCT — adapts classification sensitivity to offset magnitude).

Also classify vertices: a vertex is Convex if all adjacent edges are convex,
Concave if any adjacent edge is concave, Mixed otherwise.

```rust
pub enum EdgeClass {
    Tangent,
    Convex { angle: f64 },
    Concave { angle: f64 },
}

pub enum VertexClass {
    Convex,
    Concave,
    Mixed,
}
```

### Phase 2: Offset Faces (`offset.rs`)

Compute the offset surface for each non-excluded face.

**Analytic fast paths (exact):**
- Plane → Plane (shift d by offset distance)
- Cylinder → Cylinder (adjust radius)
- Cone → Cone (adjust radius, handle apex degeneration)
- Sphere → Sphere (adjust radius, handle collapse)
- Torus → Torus (adjust minor radius)

**NURBS fallback:**
- Sample surface normals on a grid
- Displace each point by `distance * normal`
- Refit via surface interpolation
- Run SSI-based self-intersection detection/trimming

**Per-face storage:**
```rust
pub struct OffsetFace {
    pub original: FaceId,
    pub surface: FaceSurface,
    pub distance: f64,
    pub status: OffsetStatus,
}
```

### Phase 3: Inter3d (`inter3d.rs`)

Compute 3D intersection curves between pairs of adjacent offset faces.

**Algorithm:**
1. Build adjacency map: for each edge in the original solid, the two adjacent
   faces' offset surfaces need to be intersected.
2. For concave edges: the offset faces converge — their intersection curve
   replaces the original edge.
3. For convex edges in intersection mode: extend the offset faces until they
   meet (if they can) or leave a gap for arc joint filling.
4. Use `analytic_intersection` for analytic-analytic pairs (exact).
5. Use `nurbs/intersection` for NURBS-involving pairs.

**Output:** For each edge, a list of intersection curves (may be multiple
if the intersection is non-trivial).

```rust
pub struct IntersectionResult {
    pub edge: EdgeId,
    pub face_a: FaceId,
    pub face_b: FaceId,
    pub curves: Vec<IntersectionCurve>,
}

pub struct IntersectionCurve {
    pub curve_3d: EdgeCurve,
    pub pcurve_a: Option<Curve2D>,  // on face_a's offset surface
    pub pcurve_b: Option<Curve2D>,  // on face_b's offset surface
}
```

### Phase 4: Inter2d (`inter2d.rs`)

Project intersection curves onto each face and split existing edges.

**Algorithm:**
1. For each offset face, collect all intersection curves that touch it.
2. Project each curve onto the face's parameter space (UV domain).
3. Find where projected curves cross existing face boundary edges.
4. Split edges at crossing points.
5. Create new vertices at crossing points.

**Key data structure:** Edge split map — tracks where each original edge
has been split and what new edges/vertices were created.

```rust
pub struct EdgeSplit {
    pub original: EdgeId,
    pub splits: Vec<SplitPoint>,
}

pub struct SplitPoint {
    pub parameter: f64,       // t along original edge
    pub vertex: VertexId,     // new vertex at split
    pub new_edges: [EdgeId; 2],  // edges before/after split
}
```

### Phase 5: Arc Joints (`arc_joint.rs`)

For `JointType::Arc`, create smooth transition surfaces at convex edges
and vertices.

**At convex edges:** Create a cylindrical pipe surface tangent to both
adjacent offset faces. The pipe axis follows the original edge, radius
equals the offset distance.

**At convex vertices:** Create a spherical cap centered at the original
vertex position, radius equals the offset distance.

This phase produces new faces that fill the gaps between diverging offset
faces. Each pipe/cap face connects to the adjacent offset faces via shared
edges computed in Phase 3/4.

### Phase 6: MakeLoops (`loops.rs`)

Reconstruct closed wires from the split edges on each offset face.

**Algorithm:**
1. For each offset face, collect all edges (original boundary edges, split
   fragments, and new intersection edges).
2. Build a local edge graph: vertex → outgoing edges.
3. Walk the graph to form closed loops (outer wire + optional inner wires).
4. Classify loops as outer or inner based on signed area in UV space.

This is conceptually similar to the wire builder in `brepkit-algo`'s
`builder/wire_builder.rs`.

### Phase 7: Assemble (`assemble.rs`)

Build the offset shell from trimmed faces + joint faces.

**For offset_solid:** Collect all offset faces (with reconstructed wires) +
any arc joint faces. Assemble into a closed shell, then wrap in a solid.

**For thick_solid:** Also generate "wall" faces connecting the original
face boundaries (at excluded faces) to the offset face boundaries. Wall
faces are ruled surfaces following the edge between original and offset
positions.

**Validation:** Check shell closure, orientation consistency, and Euler
characteristic before creating the solid.

### Phase 8: SI Removal (`self_int.rs`)

Use the GFA boolean engine to detect and remove self-intersecting regions
of the offset shell.

**Algorithm:**
1. Build a temporary solid from the offset shell.
2. Run `brepkit_algo::boolean(topo, offset_solid, offset_solid, Intersect)`.
3. The self-intersection of the solid with itself produces the valid
   (non-inverted) region.
4. If the boolean produces no result, the entire solid is inverted (error).

This leverages the existing GFA infrastructure rather than implementing
custom SI detection.

### Phase 9: Solidify

Final cleanup and solid creation:
1. Orient all shells consistently (outward normals).
2. Validate with `brepkit-check`.
3. Update vertex tolerances based on offset magnitude.
4. Return the final solid.

## Central Data Structure

```rust
pub struct OffsetData {
    // Phase 1 outputs
    pub edge_class: BTreeMap<EdgeId, EdgeClass>,
    pub vertex_class: BTreeMap<VertexId, VertexClass>,

    // Phase 2 outputs
    pub face_offset: BTreeMap<FaceId, OffsetFace>,

    // Phase 3 outputs
    pub intersections: Vec<IntersectionResult>,

    // Phase 4 outputs
    pub edge_splits: BTreeMap<EdgeId, EdgeSplit>,
    pub new_vertices: Vec<VertexId>,
    pub new_edges: Vec<EdgeId>,

    // Phase 5 outputs
    pub joint_faces: Vec<FaceId>,

    // Phase 6 outputs
    pub face_wires: BTreeMap<FaceId, Vec<WireId>>,

    // Configuration
    pub excluded_faces: HashSet<FaceId>,
    pub distance: f64,
    pub options: OffsetOptions,
}
```

## Public API

```rust
/// Offset all faces of a solid by a uniform distance.
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    options: OffsetOptions,
) -> Result<SolidId, OffsetError>;

/// Create a hollow solid by offsetting inward with excluded (open) faces.
pub fn thick_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    exclude: &[FaceId],
    options: OffsetOptions,
) -> Result<SolidId, OffsetError>;

pub struct OffsetOptions {
    pub joint: JointType,
    pub tolerance: Tolerance,
    pub remove_self_intersections: bool,
}

pub enum JointType {
    Intersection,
    Arc,
}
```

## Error Types

```rust
pub enum OffsetError {
    TopologyError(TopologyError),
    InvalidInput { reason: String },
    AnalysisFailed { edge: EdgeId, reason: String },
    IntersectionFailed { face_a: FaceId, face_b: FaceId, reason: String },
    SelfIntersection { reason: String },
    AssemblyFailed { reason: String },
    CollapsedSolid,
}
```

## Migration Strategy

1. Build `brepkit-offset` as a new crate alongside existing code.
2. Add `offset_solid_v2` and `thick_solid_v2` in operations that delegate
   to the new crate.
3. Add WASM bindings `offsetSolidV2` / `thickSolidV2`.
4. Once tested, switch default bindings and deprecate old code.
5. Eventually remove `offset_solid.rs`, `shell_op.rs`, `thicken.rs`,
   `offset_trim.rs` from operations.

## Test Plan

**Unit tests per phase:**
- Analyse: box (all convex), cylinder (mixed), sphere (all convex)
- Offset: each analytic surface type + NURBS
- Inter3d: plane-plane, plane-cylinder, cylinder-cylinder intersections
- Inter2d: edge splitting on known geometries
- MakeLoops: wire reconstruction on split faces
- Assemble: shell closure validation

**Integration tests:**
- Box offset outward/inward (exact volume check)
- Cylinder offset (radius ± d)
- Sphere offset (radius ± d)
- Fillet box offset (mixed analytic + tangent edges)
- Concave L-shape offset (edge collision resolution)
- Shell operation (box with top face removed)
- Self-intersecting offset (large inward offset on concave geometry)

**Regression tests:**
- All existing offset tests must pass via the v2 API
- Volume comparison with v1 for compatible cases
