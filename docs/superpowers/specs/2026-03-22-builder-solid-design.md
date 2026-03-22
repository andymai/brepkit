---
tags:
  - architecture
  - active
project: brepkit
created: 2026-03-22
---

# BuilderSolid + CommonBlock — Design Spec

> OCCT-style shell assembly and edge deduplication for the GFA boolean pipeline.
> Fixes 31+ failing tests (Categories A + E from [[brepkit-gfa-triage]]).

## Problem

The GFA pipeline produces correct face counts (algo-level tests pass) but faces have non-shared boundary edges. Adjacent faces from different input solids reference different edge entities for the same geometric edge. This creates non-manifold shells:

- 26 tests fail with "result should be manifold" (Category A)
- 5 tests fail with wrong Euler/topology for shelled/complex solids (Category E)
- 9 gridfinity tests fail as downstream consequences

The current `assemble_solid` in `algo/builder/assemble.rs` simply dumps all BOP-selected faces into one shell with no edge deduplication, manifold checking, or void handling.

## Solution

Two coordinated changes:

1. **CommonBlock infrastructure** in the PaveFiller — ensures overlapping edges from different input solids share a single split edge entity
2. **BuilderSolid** — OCCT-style 4-phase shell assembly with edge connectivity, dihedral angle selection, and Growth/Hole classification

## Part 1: CommonBlock Infrastructure

### Data Structures

New in `ds/pave.rs`:

```rust
/// Typed handle into GfaArena.common_blocks (follows arena Id<T> pattern).
pub type CommonBlockId = Id<CommonBlock>;

/// A group of geometrically coincident PaveBlocks that must share
/// a single split edge in the output topology.
pub struct CommonBlock {
    /// PaveBlocks representing the same geometric edge segment.
    /// First entry is the "representative" (canonical).
    pub pave_blocks: Vec<PaveBlockId>,
    /// Faces this common block spans (for EF: edge lies on face boundary).
    pub faces: Vec<FaceId>,
    /// The single split edge created for this group.
    /// Set by MakeSplitEdges; None until then.
    pub split_edge: Option<EdgeId>,
    /// Tolerance covering deviation across all grouped pave blocks.
    pub tolerance: f64,
}
```

New fields on `GfaArena`:

```rust
pub common_blocks: Vec<CommonBlock>,
/// Reverse map: PaveBlock → its CommonBlock (if any).
pub pb_to_cb: HashMap<PaveBlockId, CommonBlockId>,
```

New methods on `GfaArena`:

```rust
/// Follow the CommonBlock chain to find the canonical PaveBlock.
/// If pb has no CB, returns pb itself.
pub fn real_pave_block(&self, pb: PaveBlockId) -> PaveBlockId;

/// Create a new CommonBlock grouping the given PaveBlocks.
pub fn create_common_block(&mut self, pbs: Vec<PaveBlockId>, tol: f64) -> CommonBlockId;

/// Add a face reference to an existing CommonBlock.
pub fn add_face_to_cb(&mut self, cb: CommonBlockId, face: FaceId);
```

### Phase EE Enhancement (`pave_filler/phase_ee.rs`)

Add overlap detection as a **post-split phase** (after `make_blocks` creates child PaveBlocks from split edges). This follows OCCT's `ForceInterfEE` pattern:

1. After `make_blocks`, iterate leaf PaveBlocks (those with no children)
2. For each pair of leaf PBs from different original edges:
   - Check if their 3D endpoints are within tolerance of each other
   - Check if their curves are geometrically coincident (same line direction, same circle, etc.)
   - If both endpoints match and curves match → they represent the same geometric edge segment
3. Collect matching PBs into `overlap_groups: HashMap<PaveBlockId, Vec<PaveBlockId>>`
4. Build transitive closure (if PB₁≡PB₂ and PB₂≡PB₃, group {PB₁,PB₂,PB₃})
5. For each group: call `arena.create_common_block(group, max_tolerance)`

**Timing is critical:** CommonBlocks must be created on leaf PaveBlocks (post-split), not on original edges (pre-split), because overlap is a property of edge segments, not whole edges.

**Detection heuristic:** Two PaveBlocks overlap if:
- Their original edges lie on curves with matching geometry (same line direction, same circle, etc.)
- Their parameter ranges overlap by more than `tol.linear`
- Their 3D endpoints are within `tol.linear` of each other

### Phase EF Enhancement (`pave_filler/phase_ef.rs`)

After the existing EF intersection detection:

1. When an edge lies entirely on a face boundary (intersection type = Edge, not Vertex):
   - Find the PaveBlock for that edge segment
   - If PB already has a CommonBlock: `arena.add_face_to_cb(cb_id, face_id)`
   - Otherwise: create new CB with just that PB + the face reference

### MakeSplitEdges Enhancement (`pave_filler/make_split_edges.rs`)

Currently creates one split edge per PaveBlock. Enhanced:

1. Track processed CommonBlocks: `processed_cbs: HashSet<CommonBlockId>`
2. For each PaveBlock:
   - If PB has a CB and CB already processed → skip (edge already created)
   - If PB has a CB and CB not processed:
     - Use the canonical PB (first in CB.pave_blocks)
     - Create ONE split edge from canonical PB's range
     - Set `cb.split_edge = Some(edge_id)`
     - Mark CB as processed
   - If PB has no CB: create split edge as before

### fill_images_faces Integration

In `build_topology_face`, the edge cache already uses `pave_block_id` for cross-face sharing. Enhanced:

- When looking up the edge for a section edge with `pave_block_id`:
  - Check if that PB has a CommonBlock
  - If so, use `cb.split_edge` directly (guaranteed to be the shared edge)
  - This ensures all faces referencing the same CB use the exact same edge entity

## Part 2: BuilderSolid

### New file: `algo/builder/builder_solid.rs`

Replaces the logic in `assemble_solid`. The current `assemble.rs` becomes a thin wrapper that delegates to `builder_solid::build_solid()`.

### Algorithm

**Input:** List of `SelectedFace` from BOP (face_id + reversed flag).

**Output:** `SolidId` (or error if assembly fails).

#### Phase 1: PerformShapesToAvoid

Remove faces with structural defects that prevent shell formation.

```rust
fn perform_shapes_to_avoid(
    topo: &Topology,
    faces: &mut Vec<FaceId>,
) -> Vec<FaceId> // returns avoided faces
```

1. Build edge→face map: `HashMap<EdgeId, Vec<FaceId>>` from each face's wire edges. Since CommonBlocks ensure shared edges use the same `EdgeId`, this correctly identifies adjacency. For faces that predate CommonBlock processing (unsplit originals), fall back to vertex-pair matching: `HashMap<(VertexId, VertexId), Vec<FaceId>>` where key is `(min(v1,v2), max(v1,v2))`.
2. Find edges with only 1 face → mark that face for avoidance
3. **Iterate** until stable (removing a face may expose new free edges)
4. Return avoided faces (for potential internal shell construction)

#### Phase 2: PerformLoops — Shell Construction

```rust
fn perform_loops(
    topo: &Topology,
    faces: &[FaceId],
) -> Vec<Vec<FaceId>> // returns shells (connected components)
```

**Step 2a: Build adjacency**

Build the vertex-pair → faces map (same as Phase 1 but on remaining faces).

**Step 2b: Flood-fill connected components**

Standard connected-component algorithm:
- Start with an unvisited face
- BFS/DFS: for each edge (vertex pair) of the current face, find adjacent faces
- If vertex-pair has exactly 2 faces: both are neighbors (manifold edge)
- If vertex-pair has 3+ faces: **GetFaceOff** dihedral selection — only pick the best neighbor

**Step 2c: GetFaceOff — Dihedral Angle Selection**

```rust
fn get_face_off(
    topo: &Topology,
    edge_start: Point3,
    edge_end: Point3,
    current_face: FaceId,
    candidate_faces: &[FaceId],
) -> Option<FaceId>
```

At a non-manifold junction (3+ faces sharing an edge):
1. Compute edge tangent: `t = normalize(edge_end - edge_start)`
2. For the current face: compute face normal `n₁` at edge midpoint, then bi-normal `b₁ = t × n₁`
3. For each candidate face: compute `n_i`, `b_i = n_i × t`
4. Compute signed angle from `b₁` to `b_i` using `t` as reference axis
5. Handle special cases: angle ≈ 0 (same face → π, coplanar → 2π)
6. Select face with **smallest positive angle**

This naturally implements clockwise face traversal around the edge, which is the correct solid-oriented face ordering.

#### Phase 3: PerformAreas — Growth vs Hole Classification

```rust
fn perform_areas(
    topo: &Topology,
    shells: &[Vec<FaceId>],
) -> (Vec<Vec<FaceId>>, Vec<Vec<FaceId>>) // (growth, holes)
```

For each shell:
1. Build a temporary solid from the shell faces
2. Classify a point known to be "at infinity" (e.g., AABB max + large offset)
3. If the point is `Inside` → shell is a **Hole** (normals point inward)
4. If `Outside` → shell is **Growth** (outer shell)

Use `brepkit_algo::classifier::classify_point` for the containment test. Handle non-Inside/Outside results: treat `On`/`CoplanarSame`/`CoplanarOpposite` as `Outside` (a point at infinity on the boundary is pathological), treat `Unknown` as an error.

**Temporary solid lifecycle:** The temporary solids created for classification are added to the `Topology` arena (which never frees). This is acceptable overhead — typically 1-2 temporary solids per boolean. If needed, a lightweight signed-volume test could replace this in the future.

**Error handling:** If all shells are classified as Holes (no growth shell), return `AlgoError::AssemblyFailed("no outer shell found")`.

#### Phase 4: Assemble

```rust
fn assemble(
    topo: &mut Topology,
    growth_shells: Vec<Vec<FaceId>>,
    hole_shells: Vec<Vec<FaceId>>,
) -> Result<SolidId, AlgoError>
```

1. For each hole shell: find the smallest growth shell containing it (AABB pre-filter + point-in-solid)
2. Build the solid: largest growth shell = outer, nested holes = inner shells
3. Typically produces 1 solid (multi-solid results from non-overlapping geometries)

### Edge Orientation Handling

When building shells, edges must have consistent orientation:
- A manifold edge appears in exactly 2 faces with **opposite** orientations
- When traversing a shell, if face F₁ sees edge E as FORWARD, face F₂ must see it as REVERSED
- The BuilderSolid verifies this during connectivity — if both faces see the edge the same way, the edge is mis-oriented (face normal flip)

### Handling Reversed Faces (BOP Cut)

BOP marks B-faces as `reversed: true` for Cut operations. Before assembly:
1. Create reversed copies of marked faces (flip surface orientation)
2. This ensures normals point outward on the result solid

## File Structure

| File | Action | Description |
|------|--------|-------------|
| `ds/pave.rs` | Modify | Add `CommonBlock`, `CommonBlockId`, arena methods |
| `ds/arena.rs` | Modify | Add `common_blocks`, `pb_to_cb` fields |
| `pave_filler/phase_ee.rs` | Modify | Add edge overlap detection → CB creation |
| `pave_filler/phase_ef.rs` | Modify | Add edge-on-face detection → CB update |
| `pave_filler/make_split_edges.rs` | Modify | CB-aware: one edge per CB |
| `builder/builder_solid.rs` | **New** | Full BuilderSolid (4 phases) |
| `builder/assemble.rs` | Modify | Delegate to builder_solid |
| `builder/mod.rs` | Modify | Add `pub mod builder_solid;` |
| `builder/fill_images_faces.rs` | Modify | Use CB split_edge in edge cache |

## Testing Strategy

1. **Unit tests** for `CommonBlock` creation and `real_pave_block` traversal
2. **Unit tests** for `get_face_off` dihedral angle computation (coplanar, perpendicular, acute, and **3+ faces at a shared edge** — the primary non-manifold scenario)
3. **Unit tests** for `make_connexity_blocks` (manifold, non-manifold with 3+ faces sharing an edge, disconnected)
4. **Integration tests**: un-ignore Category A tests (26) as they pass
5. **Integration tests**: un-ignore Category E tests (5) as they pass
6. **Regression check**: all currently-passing tests must continue to pass

## Success Criteria

- All 26 Category A tests pass (non-manifold edge fix)
- At least 3/5 Category E tests pass (shelled/complex solids)
- All currently-passing tests continue to pass (0 regressions)
- `cargo clippy --all-targets -- -D warnings` clean
- `cargo build -p brepkit-wasm --target wasm32-unknown-unknown` succeeds
- `./scripts/check-boundaries.sh` passes

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Dihedral angle computation edge cases (parallel normals, degenerate edges) | Use OCCT's exact algorithm with AngleWithRef; add unit tests for all edge cases |
| Performance regression from vertex-pair map building | O(F·E) where F=faces, E=edges per face — negligible for typical solids (<100 faces) |
| CommonBlock creation false positives (matching edges that shouldn't match) | Use strict tolerance + curve geometry matching, not just parameter overlap |
| Regression on tangent-touch cases (PR #385 lesson) | BuilderSolid uses connectivity, not generic sewing — tangent-touch faces aren't in the selected set |
| Hole classification via classifier may be slow | Cache classifier per solid; typical booleans have 0-1 holes |

## OCCT Reference Files

| Concept | OCCT File | Key Function |
|---------|-----------|-------------|
| CommonBlock | `BOPDS/BOPDS_CommonBlock.hxx` | Data structure |
| EE overlap | `BOPAlgo_PaveFiller_3.cxx` | `ForceInterfEE` |
| EF edge-on-face | `BOPAlgo_PaveFiller_5.cxx` | `PerformEF` |
| MakeSplitEdges | `BOPAlgo_PaveFiller_7.cxx` | CB-aware edge creation |
| Shell assembly | `BOPAlgo_BuilderSolid.cxx` | `PerformLoops`, `PerformAreas` |
| Dihedral angle | `BOPTools_AlgoTools.cxx` | `GetFaceOff`, `AngleWithRef` |
| Connectivity | `BOPTools_AlgoTools.cxx` | `MakeConnexityBlocks` |
