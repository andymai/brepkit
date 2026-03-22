# BuilderSolid + CommonBlock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the minimal `assemble_solid` with an OCCT-style 4-phase BuilderSolid, and add CommonBlock infrastructure to the PaveFiller for edge deduplication across input solids.

**Architecture:** CommonBlocks in the PaveFiller track overlapping edge segments from different solids, ensuring a single split edge entity per group. BuilderSolid assembles BOP-selected faces into manifold shells using edge-connectivity flood-fill, dihedral angle selection at non-manifold junctions, and Growth/Hole shell classification. This replaces the current naive assembly that dumps faces into one shell.

**Tech Stack:** Rust, brepkit workspace (algo crate at L2, depends on math + topology)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/algo/src/ds/pave.rs` | Add `CommonBlock` struct + `CommonBlockId` type |
| Modify | `crates/algo/src/ds/arena.rs` | Add CB fields + methods to `GfaArena` |
| Modify | `crates/algo/src/pave_filler/mod.rs` | Insert `force_interf_ee` call after `make_blocks` |
| Create | `crates/algo/src/pave_filler/force_interf_ee.rs` | Post-split EE overlap detection → CB creation |
| Modify | `crates/algo/src/pave_filler/make_split_edges.rs` | CB-aware: one edge per CommonBlock |
| Create | `crates/algo/src/builder/builder_solid.rs` | Full 4-phase BuilderSolid |
| Modify | `crates/algo/src/builder/assemble.rs` | Delegate to `builder_solid::build_solid()` |
| Modify | `crates/algo/src/builder/mod.rs` | Add `pub mod builder_solid;` |

---

### Task 1: CommonBlock data structure + GfaArena integration

**Files:**
- Modify: `crates/algo/src/ds/pave.rs` (append after `PaveBlock` impl)
- Modify: `crates/algo/src/ds/arena.rs` (add fields + methods)
- Modify: `crates/algo/src/ds/mod.rs` (re-export new types)

- [ ] **Step 1: Add `CommonBlock` struct to `pave.rs`**

Append after the existing `PaveBlock` impl block at line 73:

```rust
/// Typed handle for a [`CommonBlock`] in the GFA arena.
pub type CommonBlockId = Id<CommonBlock>;

/// A group of geometrically coincident PaveBlocks that must share
/// a single split edge in the output topology.
///
/// Created by the post-split EE overlap detection phase. Used by
/// `MakeSplitEdges` to ensure one edge entity per group.
#[derive(Debug, Clone)]
pub struct CommonBlock {
    /// PaveBlocks representing the same geometric edge segment.
    /// First entry is the "representative" (canonical).
    pub pave_blocks: Vec<PaveBlockId>,
    /// Faces this common block spans (for EF: edge lies on face boundary).
    pub faces: Vec<brepkit_topology::face::FaceId>,
    /// The single split edge created for this group.
    /// Set by `MakeSplitEdges`; `None` until then.
    pub split_edge: Option<brepkit_topology::edge::EdgeId>,
    /// Tolerance covering deviation across all grouped pave blocks.
    pub tolerance: f64,
}
```

- [ ] **Step 2: Add CB fields and methods to `GfaArena` in `arena.rs`**

Add to struct fields (after `edge_pave_blocks`):

```rust
/// CommonBlocks grouping coincident pave blocks.
pub common_blocks: Arena<CommonBlock>,
/// Reverse map: PaveBlock → its CommonBlock (if any).
pub pb_to_cb: HashMap<PaveBlockId, CommonBlockId>,
```

Add to `GfaArena::new()`:

```rust
common_blocks: Arena::new(),
pb_to_cb: HashMap::new(),
```

Add new methods after `collect_leaf_pave_blocks`:

```rust
/// Follow the CommonBlock chain to find the canonical PaveBlock.
/// If `pb` has no CB, returns `pb` itself.
#[must_use]
pub fn real_pave_block(&self, pb: PaveBlockId) -> PaveBlockId {
    match self.pb_to_cb.get(&pb) {
        Some(&cb_id) => {
            if let Some(cb) = self.common_blocks.get(cb_id) {
                cb.pave_blocks.first().copied().unwrap_or(pb)
            } else {
                pb
            }
        }
        None => pb,
    }
}

/// Create a new CommonBlock grouping the given PaveBlocks.
pub fn create_common_block(
    &mut self,
    pbs: Vec<PaveBlockId>,
    tol: f64,
) -> CommonBlockId {
    let cb = CommonBlock {
        pave_blocks: pbs.clone(),
        faces: Vec::new(),
        split_edge: None,
        tolerance: tol,
    };
    let cb_id = self.common_blocks.alloc(cb);
    for &pb in &pbs {
        self.pb_to_cb.insert(pb, cb_id);
    }
    cb_id
}

/// Add a face reference to an existing CommonBlock.
pub fn add_face_to_cb(
    &mut self,
    cb: CommonBlockId,
    face: brepkit_topology::face::FaceId,
) {
    if let Some(cb) = self.common_blocks.get_mut(cb) {
        if !cb.faces.contains(&face) {
            cb.faces.push(face);
        }
    }
}
```

- [ ] **Step 3: Update imports in `arena.rs`**

Add to imports at top of file:

```rust
use super::pave::{CommonBlock, CommonBlockId};
```

- [ ] **Step 4: Update re-exports in `ds/mod.rs`**

Check if `ds/mod.rs` re-exports pave types. If so, add `CommonBlock` and `CommonBlockId` to the re-export list.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p brepkit-algo`
Expected: SUCCESS

- [ ] **Step 6: Commit**

```bash
git add crates/algo/src/ds/pave.rs crates/algo/src/ds/arena.rs crates/algo/src/ds/mod.rs
git commit -m "$(cat <<'EOF'
feat(algo): add CommonBlock data structure to GFA arena

CommonBlock groups geometrically coincident PaveBlocks that must share
a single split edge. Adds typed CommonBlockId handle, GfaArena methods
for creation/lookup/face-tracking, and real_pave_block() for canonical
PB resolution.
EOF
)"
```

---

### Task 2: Post-split EE overlap detection (ForceInterfEE)

**Files:**
- Create: `crates/algo/src/pave_filler/force_interf_ee.rs`
- Modify: `crates/algo/src/pave_filler/mod.rs` (add module + call site)

- [ ] **Step 1: Create `force_interf_ee.rs` with the overlap detection algorithm**

```rust
//! Post-split EE overlap detection — creates CommonBlocks for coincident
//! leaf PaveBlocks from different original edges.
//!
//! Runs after `make_blocks` (which splits PaveBlocks at extra paves),
//! iterating leaf PaveBlocks to find pairs with matching 3D endpoints
//! and compatible curve geometry.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;

use crate::ds::GfaArena;
use crate::ds::pave::PaveBlockId;
use crate::error::AlgoError;

/// Detect overlapping leaf PaveBlocks and group them into CommonBlocks.
///
/// Two leaf PaveBlocks from different original edges overlap if:
/// 1. Their start/end vertex positions are within tolerance
/// 2. Their edge curves have compatible geometry (same line direction,
///    same circle, etc.)
///
/// # Errors
///
/// Returns [`AlgoError`] if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn perform(topo: &Topology, tol: Tolerance, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect all leaf PBs with their 3D endpoint data
    let all_edge_pbs: Vec<(brepkit_topology::edge::EdgeId, Vec<PaveBlockId>)> = arena
        .edge_pave_blocks
        .iter()
        .map(|(&eid, pbs)| (eid, arena.collect_leaf_pave_blocks(pbs)))
        .collect();

    // Flatten to (pb_id, original_edge, start_pos, end_pos)
    let mut leaf_data: Vec<(PaveBlockId, brepkit_topology::edge::EdgeId, brepkit_math::vec::Point3, brepkit_math::vec::Point3)> = Vec::new();

    for (orig_edge, leaf_pbs) in &all_edge_pbs {
        for &pb_id in leaf_pbs {
            let pb = match arena.pave_blocks.get(pb_id) {
                Some(pb) => pb,
                None => continue,
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);
            let start_pos = topo.vertex(sv)?.point();
            let end_pos = topo.vertex(ev)?.point();
            leaf_data.push((pb_id, *orig_edge, start_pos, end_pos));
        }
    }

    // Find overlapping pairs: O(n²) but n is typically small (< 100 leaf PBs)
    let mut overlap_map: HashMap<PaveBlockId, Vec<PaveBlockId>> = HashMap::new();
    let n = leaf_data.len();

    for i in 0..n {
        let (pb_i, edge_i, start_i, end_i) = &leaf_data[i];
        for j in (i + 1)..n {
            let (pb_j, edge_j, start_j, end_j) = &leaf_data[j];

            // Must be from different original edges
            if edge_i == edge_j {
                continue;
            }

            // Already in same CB
            if arena.pb_to_cb.contains_key(pb_i)
                && arena.pb_to_cb.get(pb_i) == arena.pb_to_cb.get(pb_j)
            {
                continue;
            }

            // Check endpoint match (either same-direction or reversed)
            let fwd_match = (start_i - start_j).length() < tol.linear
                && (end_i - end_j).length() < tol.linear;
            let rev_match = (start_i - end_j).length() < tol.linear
                && (end_i - start_j).length() < tol.linear;

            if !fwd_match && !rev_match {
                continue;
            }

            // Check curve compatibility
            let curve_i = topo.edge(*edge_i)?.curve();
            let curve_j = topo.edge(*edge_j)?.curve();
            if !curves_compatible(curve_i, curve_j, tol) {
                continue;
            }

            // Record overlap
            overlap_map.entry(*pb_i).or_default().push(*pb_j);
            overlap_map.entry(*pb_j).or_default().push(*pb_i);
        }
    }

    // Build transitive closure and create CommonBlocks
    let mut visited: std::collections::HashSet<PaveBlockId> = std::collections::HashSet::new();

    for &(pb_id, _, _, _) in &leaf_data {
        if visited.contains(&pb_id) || !overlap_map.contains_key(&pb_id) {
            continue;
        }

        // BFS to find connected component
        let mut group = Vec::new();
        let mut queue = vec![pb_id];
        while let Some(current) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }
            group.push(current);
            if let Some(neighbors) = overlap_map.get(&current) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }

        if group.len() >= 2 {
            let cb_id = arena.create_common_block(group.clone(), tol.linear);
            log::debug!(
                "ForceInterfEE: created CommonBlock {cb_id:?} with {} PaveBlocks",
                group.len()
            );
        }
    }

    Ok(())
}

/// Check if two edge curves are geometrically compatible (same type + parameters).
fn curves_compatible(a: &EdgeCurve, b: &EdgeCurve, tol: Tolerance) -> bool {
    match (a, b) {
        (EdgeCurve::Line, EdgeCurve::Line) => true,
        (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) => {
            (ca.radius() - cb.radius()).abs() < tol.linear
                && (ca.center() - cb.center()).length() < tol.linear
        }
        (EdgeCurve::Ellipse(ea), EdgeCurve::Ellipse(eb)) => {
            (ea.semi_major() - eb.semi_major()).abs() < tol.linear
                && (ea.semi_minor() - eb.semi_minor()).abs() < tol.linear
                && (ea.center() - eb.center()).length() < tol.linear
        }
        _ => false, // Different types or NURBS: no overlap detection yet
    }
}
```

- [ ] **Step 2: Add module declaration and call site in `pave_filler/mod.rs`**

Add module declaration after `pub mod phase_vv;`:

```rust
pub mod force_interf_ee;
```

In `run_pave_filler`, insert the call after `make_blocks::perform` and before `make_split_edges::perform`:

```rust
make_blocks::perform(arena)?;
force_interf_ee::perform(topo, tol, arena)?;  // NEW: post-split overlap detection
make_split_edges::perform(topo, arena)?;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p brepkit-algo`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add crates/algo/src/pave_filler/force_interf_ee.rs crates/algo/src/pave_filler/mod.rs
git commit -m "$(cat <<'EOF'
feat(algo): post-split EE overlap detection creates CommonBlocks

Adds ForceInterfEE phase after MakeBlocks. Detects leaf PaveBlocks
from different original edges that share 3D endpoints and compatible
curve geometry. Groups them into CommonBlocks via transitive closure.
EOF
)"
```

---

### Task 3: CB-aware MakeSplitEdges

**Files:**
- Modify: `crates/algo/src/pave_filler/make_split_edges.rs`

- [ ] **Step 1: Rewrite `perform` to handle CommonBlocks**

Replace the entire `perform` function:

```rust
pub fn perform(topo: &mut Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Track processed CommonBlocks to avoid creating duplicate edges
    let mut processed_cbs: std::collections::HashSet<
        crate::ds::pave::CommonBlockId,
    > = std::collections::HashSet::new();

    // Collect all leaf pave block IDs that need edges
    let leaf_ids: Vec<_> = arena
        .pave_blocks
        .iter()
        .filter(|(_, pb)| pb.children.is_empty() && pb.split_edge.is_none())
        .map(|(id, _)| id)
        .collect();

    for pb_id in leaf_ids {
        // Check if this PB is part of a CommonBlock
        if let Some(&cb_id) = arena.pb_to_cb.get(&pb_id) {
            if !processed_cbs.insert(cb_id) {
                // CB already processed — reuse its split edge
                let split_edge = arena
                    .common_blocks
                    .get(cb_id)
                    .and_then(|cb| cb.split_edge);
                if let Some(edge_id) = split_edge {
                    if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                        pb.split_edge = Some(edge_id);
                    }
                }
                continue;
            }

            // First PB in this CB — use canonical PB to create the edge
            let canonical_pb_id = arena
                .common_blocks
                .get(cb_id)
                .and_then(|cb| cb.pave_blocks.first().copied())
                .unwrap_or(pb_id);

            let edge_id = create_split_edge(topo, arena, canonical_pb_id)?;

            // Set split_edge on the CB and ALL PBs in the group
            if let Some(cb) = arena.common_blocks.get_mut(cb_id) {
                cb.split_edge = Some(edge_id);
                let all_pbs = cb.pave_blocks.clone();
                for &member_pb in &all_pbs {
                    if let Some(pb) = arena.pave_blocks.get_mut(member_pb) {
                        pb.split_edge = Some(edge_id);
                    }
                }
            }

            log::debug!(
                "MakeSplitEdges: created edge {edge_id:?} for CommonBlock {cb_id:?}",
            );
        } else {
            // No CB — create individual split edge as before
            let edge_id = create_split_edge(topo, arena, pb_id)?;
            if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                pb.split_edge = Some(edge_id);
            }
            log::debug!(
                "MakeSplitEdges: created edge {edge_id:?} for pave block {pb_id:?}",
            );
        }
    }

    Ok(())
}

/// Create a single split edge from a pave block's data.
fn create_split_edge(
    topo: &mut Topology,
    arena: &GfaArena,
    pb_id: crate::ds::pave::PaveBlockId,
) -> Result<EdgeId, AlgoError> {
    let (original_edge_id, start_vertex, end_vertex) = {
        let pb = arena
            .pave_blocks
            .get(pb_id)
            .ok_or_else(|| AlgoError::FaceSplitFailed("pave block not found".into()))?;
        let start_v = arena.resolve_vertex(pb.start.vertex);
        let end_v = arena.resolve_vertex(pb.end.vertex);
        (pb.original_edge, start_v, end_v)
    };

    let curve = topo.edge(original_edge_id)?.curve().clone();
    let new_edge = Edge::new(start_vertex, end_vertex, curve);
    Ok(topo.add_edge(new_edge))
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p brepkit-algo`
Expected: SUCCESS

- [ ] **Step 3: Run existing tests**

Run: `cargo test -p brepkit-algo`
Expected: All tests pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add crates/algo/src/pave_filler/make_split_edges.rs
git commit -m "$(cat <<'EOF'
feat(algo): CB-aware MakeSplitEdges — one edge per CommonBlock

When a PaveBlock belongs to a CommonBlock, creates a single split
edge for the entire group. All PBs in the CB reference the same
edge entity, ensuring shared edges across input solids.
EOF
)"
```

---

### Task 4: BuilderSolid — core shell assembly

**Files:**
- Create: `crates/algo/src/builder/builder_solid.rs`
- Modify: `crates/algo/src/builder/mod.rs` (add `pub mod builder_solid;`)

- [ ] **Step 1: Create `builder_solid.rs` with all 4 phases**

This is the largest file. Create `crates/algo/src/builder/builder_solid.rs`. Full implementation below — the file has 5 public functions: `build_solid`, `perform_shapes_to_avoid`, `perform_loops`, `get_face_off`, `perform_areas`.

```rust
//! BuilderSolid — OCCT-style 4-phase shell assembly.
//!
//! Takes BOP-selected faces and assembles them into manifold shells,
//! classifies shells as Growth/Hole, and nests holes inside growth shells.
//!
//! # Phases
//!
//! 1. **PerformShapesToAvoid** — iterative free-edge removal
//! 2. **PerformLoops** — connectivity flood-fill into shells
//! 3. **PerformAreas** — Growth vs Hole classification
//! 4. **Assemble** — build final Solid from shells

use std::collections::{HashMap, HashSet, VecDeque};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::VertexId;

use crate::bop::SelectedFace;
use crate::error::AlgoError;

/// Edge key for adjacency: canonical (min, max) vertex pair.
type VPair = (usize, usize);

/// Build a solid from BOP-selected faces using the 4-phase BuilderSolid algorithm.
///
/// # Errors
///
/// Returns [`AlgoError`] if assembly produces no valid shells or
/// topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn build_solid(
    topo: &mut Topology,
    selected: &[SelectedFace],
) -> Result<SolidId, AlgoError> {
    if selected.is_empty() {
        return Err(AlgoError::AssemblyFailed("no faces selected".into()));
    }

    // Step 0: Create reversed copies for Cut B-faces
    let mut face_ids: Vec<FaceId> = Vec::with_capacity(selected.len());
    for sf in selected {
        if sf.reversed {
            let face = topo.face(sf.face_id)?;
            let surface = face.surface().clone();
            let outer_wire = face.outer_wire();
            let inner_wires = face.inner_wires().to_vec();
            let reversed_face = Face::new_reversed(outer_wire, inner_wires, surface);
            face_ids.push(topo.add_face(reversed_face));
        } else {
            face_ids.push(sf.face_id);
        }
    }

    // Phase 1: Remove faces with free edges
    let _avoided = perform_shapes_to_avoid(topo, &mut face_ids)?;

    if face_ids.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "all faces avoided (all have free edges)".into(),
        ));
    }

    // Phase 2: Build shells via connectivity
    let shells = perform_loops(topo, &face_ids)?;

    if shells.is_empty() {
        return Err(AlgoError::AssemblyFailed("no shells formed".into()));
    }

    // Phase 3: Classify Growth vs Hole
    let (growth, holes) = perform_areas(topo, &shells)?;

    if growth.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "no outer shell found (all shells classified as holes)".into(),
        ));
    }

    // Phase 4: Assemble
    assemble(topo, growth, holes)
}

// ── Phase 1 ──────────────────────────────────────────────────────────

/// Iteratively remove faces with free (single-face) edges.
///
/// Returns the list of avoided faces.
fn perform_shapes_to_avoid(
    topo: &Topology,
    faces: &mut Vec<FaceId>,
) -> Result<Vec<FaceId>, AlgoError> {
    let mut avoided = Vec::new();

    loop {
        let edge_map = build_edge_face_map(topo, faces)?;
        let mut to_remove: HashSet<FaceId> = HashSet::new();

        for (_, face_list) in &edge_map {
            if face_list.len() == 1 {
                to_remove.insert(face_list[0]);
            }
        }

        if to_remove.is_empty() {
            break;
        }

        avoided.extend(to_remove.iter());
        faces.retain(|f| !to_remove.contains(f));
    }

    if !avoided.is_empty() {
        log::debug!(
            "BuilderSolid: avoided {} faces with free edges",
            avoided.len()
        );
    }

    Ok(avoided)
}

// ── Phase 2 ──────────────────────────────────────────────────────────

/// Group faces into connected shells via edge connectivity.
///
/// Uses flood-fill with dihedral angle selection at non-manifold edges.
fn perform_loops(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<Vec<FaceId>>, AlgoError> {
    let edge_map = build_edge_face_map(topo, faces)?;

    // Also build vertex-pair → edge positions for GetFaceOff
    let edge_positions = build_edge_positions(topo, faces)?;

    let face_set: HashSet<FaceId> = faces.iter().copied().collect();
    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut shells: Vec<Vec<FaceId>> = Vec::new();

    // Build face → edge keys map for neighbor lookup
    let face_edges = build_face_edge_keys(topo, faces)?;

    for &start_face in faces {
        if visited.contains(&start_face) {
            continue;
        }

        let mut shell = Vec::new();
        let mut queue = VecDeque::new();

        // Track which edges in this shell are "filled" (have 2 faces)
        let mut shell_edge_count: HashMap<VPair, u32> = HashMap::new();

        visited.insert(start_face);
        shell.push(start_face);
        queue.push_back(start_face);

        // Count edges of start face
        if let Some(keys) = face_edges.get(&start_face) {
            for key in keys {
                *shell_edge_count.entry(*key).or_default() += 1;
            }
        }

        while let Some(current) = queue.pop_front() {
            let Some(keys) = face_edges.get(&current) else {
                continue;
            };

            for key in keys {
                // Skip edges already manifold in this shell
                if shell_edge_count.get(key).copied().unwrap_or(0) >= 2 {
                    continue;
                }

                let Some(candidates) = edge_map.get(key) else {
                    continue;
                };

                // Filter to unvisited faces
                let unvisited: Vec<FaceId> = candidates
                    .iter()
                    .filter(|&&f| f != current && face_set.contains(&f) && !visited.contains(&f))
                    .copied()
                    .collect();

                if unvisited.is_empty() {
                    continue;
                }

                // Select best face
                let selected = if unvisited.len() == 1 {
                    unvisited[0]
                } else if let Some((start, end)) = edge_positions.get(key) {
                    // Non-manifold edge: use dihedral angle selection
                    get_face_off(topo, *start, *end, current, &unvisited)
                        .unwrap_or(unvisited[0])
                } else {
                    unvisited[0]
                };

                visited.insert(selected);
                shell.push(selected);
                queue.push_back(selected);

                // Update edge counts
                if let Some(sel_keys) = face_edges.get(&selected) {
                    for k in sel_keys {
                        *shell_edge_count.entry(*k).or_default() += 1;
                    }
                }
            }
        }

        shells.push(shell);
    }

    log::debug!(
        "BuilderSolid: {} shells (sizes: {:?})",
        shells.len(),
        shells.iter().map(Vec::len).collect::<Vec<_>>()
    );

    Ok(shells)
}

/// Dihedral angle selection at a non-manifold edge.
///
/// At an edge shared by 3+ faces, selects the face with the smallest
/// positive dihedral angle relative to the current face. This implements
/// clockwise face traversal around the edge.
///
/// Reference: OCCT `BOPTools_AlgoTools::GetFaceOff` + `AngleWithRef`.
pub fn get_face_off(
    topo: &Topology,
    edge_start: Point3,
    edge_end: Point3,
    current_face: FaceId,
    candidates: &[FaceId],
) -> Option<FaceId> {
    let edge_dir = edge_end - edge_start;
    let edge_len = edge_dir.length();
    if edge_len < 1e-12 {
        return candidates.first().copied();
    }
    let t = edge_dir * (1.0 / edge_len); // unit tangent

    let mid = Point3::new(
        (edge_start.x() + edge_end.x()) * 0.5,
        (edge_start.y() + edge_end.y()) * 0.5,
        (edge_start.z() + edge_end.z()) * 0.5,
    );

    // Compute bi-normal for current face: b = t × n (outward from face)
    let n_current = face_normal_at(topo, current_face, mid)?;
    let b_current = t.cross(n_current);
    let b_current_len = b_current.length();
    if b_current_len < 1e-12 {
        return candidates.first().copied();
    }
    let b_current = b_current * (1.0 / b_current_len);

    // Reference direction for signed angle measurement
    let d_ref = n_current.cross(b_current);

    let mut best_face = None;
    let mut best_angle = f64::MAX;

    for &cand in candidates {
        let Some(n_cand) = face_normal_at(topo, cand, mid) else {
            continue;
        };
        let b_cand = t.cross(n_cand);
        let b_cand_len = b_cand.length();
        if b_cand_len < 1e-12 {
            continue;
        }
        let b_cand = b_cand * (1.0 / b_cand_len);

        // Signed angle from b_current to b_cand using d_ref as reference
        let mut angle = angle_with_ref(b_current, b_cand, d_ref);

        // Handle coplanar: angle ≈ 0 means same or opposite face
        if angle.abs() < 1e-10 {
            if cand == current_face {
                angle = std::f64::consts::PI;
            } else {
                angle = std::f64::consts::TAU;
            }
        }

        // Normalize to positive
        if angle < 0.0 {
            angle += std::f64::consts::TAU;
        }

        if angle < best_angle {
            best_angle = angle;
            best_face = Some(cand);
        }
    }

    best_face
}

/// Signed angle between two direction vectors using a reference axis.
///
/// Returns the angle from `d1` to `d2` measured around `d_ref`.
/// Port of OCCT's `AngleWithRef`.
fn angle_with_ref(d1: Vec3, d2: Vec3, d_ref: Vec3) -> f64 {
    let cross = d1.cross(d2);
    let sin_val = cross.length();
    let cos_val = d1.dot(d2);

    let mut angle = sin_val.atan2(cos_val);

    // Determine sign from reference direction
    if cross.dot(d_ref) < 0.0 {
        angle = -angle;
    }

    angle
}

/// Get face normal at a given 3D point (projects point to surface).
fn face_normal_at(topo: &Topology, face_id: FaceId, point: Point3) -> Option<Vec3> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface();

    match surface {
        FaceSurface::Plane { normal, .. } => {
            let n = if face.is_reversed() { -*normal } else { *normal };
            Some(n)
        }
        _ => {
            let (u, v) = surface.project_point(point)?;
            let mut n = surface.normal(u, v);
            if face.is_reversed() {
                n = -n;
            }
            Some(n)
        }
    }
}

// ── Phase 3 ──────────────────────────────────────────────────────────

/// Classify shells as Growth (outer) or Hole (inner).
fn perform_areas(
    topo: &Topology,
    shells: &[Vec<FaceId>],
) -> Result<(Vec<Vec<FaceId>>, Vec<Vec<FaceId>>), AlgoError> {
    let mut growth = Vec::new();
    let mut holes = Vec::new();

    for shell in shells {
        if shell.is_empty() {
            continue;
        }

        // Simple heuristic: compute signed volume of the shell.
        // Positive → outward normals (growth), negative → inward (hole).
        let signed_vol = signed_volume_of_shell(topo, shell);

        if signed_vol >= 0.0 {
            growth.push(shell.clone());
        } else {
            holes.push(shell.clone());
        }
    }

    log::debug!(
        "BuilderSolid: {} growth shells, {} hole shells",
        growth.len(),
        holes.len()
    );

    Ok((growth, holes))
}

/// Compute a signed volume estimate for a shell using the divergence theorem.
///
/// Positive = outward-oriented normals (growth shell).
/// Negative = inward-oriented normals (hole shell).
fn signed_volume_of_shell(topo: &Topology, faces: &[FaceId]) -> f64 {
    let mut volume = 0.0;

    for &fid in faces {
        let Ok(face) = topo.face(fid) else { continue };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            continue;
        };

        // Collect wire vertices
        let mut verts = Vec::new();
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let vid = oe.oriented_start(edge);
            if let Ok(v) = topo.vertex(vid) {
                verts.push(v.point());
            }
        }

        if verts.len() < 3 {
            continue;
        }

        // Fan triangulation from first vertex
        let v0 = verts[0];
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        for i in 1..verts.len() - 1 {
            let v1 = verts[i];
            let v2 = verts[i + 1];
            // Signed volume of tetrahedron with origin
            volume += sign * v0.x() * (v1.y() * v2.z() - v2.y() * v1.z())
                + sign * v1.x() * (v2.y() * v0.z() - v0.y() * v2.z())
                + sign * v2.x() * (v0.y() * v1.z() - v1.y() * v0.z());
        }
    }

    volume / 6.0
}

// ── Phase 4 ──────────────────────────────────────────────────────────

/// Final assembly: build Solid from growth + hole shells.
fn assemble(
    topo: &mut Topology,
    growth_shells: Vec<Vec<FaceId>>,
    hole_shells: Vec<Vec<FaceId>>,
) -> Result<SolidId, AlgoError> {
    // Use the largest growth shell as outer
    let outer_idx = growth_shells
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.len())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let outer_shell = Shell::new(growth_shells[outer_idx].clone())
        .map_err(|e| AlgoError::AssemblyFailed(format!("outer shell: {e}")))?;
    let outer_id = topo.add_shell(outer_shell);

    // All hole shells become inner shells of this solid
    let mut inner_ids = Vec::new();
    for hole in &hole_shells {
        if let Ok(inner_shell) = Shell::new(hole.clone()) {
            inner_ids.push(topo.add_shell(inner_shell));
        }
    }

    // Additional growth shells (if any) also become inner shells
    // (they represent internal cavities bounded by outward-facing faces)
    for (i, gs) in growth_shells.iter().enumerate() {
        if i != outer_idx {
            if let Ok(extra_shell) = Shell::new(gs.clone()) {
                inner_ids.push(topo.add_shell(extra_shell));
            }
        }
    }

    let solid = Solid::new(outer_id, inner_ids);
    let solid_id = topo.add_solid(solid);

    log::debug!(
        "BuilderSolid: assembled solid {solid_id:?} with {} faces",
        growth_shells.iter().chain(hole_shells.iter()).map(Vec::len).sum::<usize>()
    );

    Ok(solid_id)
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Build edge→face adjacency map using (edge_id OR vertex-pair) as key.
///
/// Primary key: `EdgeId` (when CommonBlocks ensure shared edges).
/// Fallback: vertex-pair `(min(v1,v2), max(v1,v2))` for unsplit faces.
fn build_edge_face_map(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, Vec<FaceId>>, AlgoError> {
    let mut map: HashMap<VPair, Vec<FaceId>> = HashMap::new();

    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let s = edge.start().index();
                let e = edge.end().index();
                let key = if s <= e { (s, e) } else { (e, s) };
                map.entry(key).or_default().push(fid);
            }
        }
    }

    Ok(map)
}

/// Build face → edge keys map for neighbor lookup.
fn build_face_edge_keys(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<FaceId, Vec<VPair>>, AlgoError> {
    let mut map: HashMap<FaceId, Vec<VPair>> = HashMap::new();

    for &fid in faces {
        let face = topo.face(fid)?;
        let mut keys = Vec::new();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let s = edge.start().index();
                let e = edge.end().index();
                keys.push(if s <= e { (s, e) } else { (e, s) });
            }
        }
        map.insert(fid, keys);
    }

    Ok(map)
}

/// Build vertex-pair → 3D positions map for GetFaceOff.
fn build_edge_positions(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, (Point3, Point3)>, AlgoError> {
    let mut map: HashMap<VPair, (Point3, Point3)> = HashMap::new();

    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let s = edge.start().index();
                let e = edge.end().index();
                let key = if s <= e { (s, e) } else { (e, s) };
                if !map.contains_key(&key) {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    map.insert(key, (sp, ep));
                }
            }
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::{Point3, Vec3};

    #[test]
    fn angle_with_ref_perpendicular() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(0.0, 1.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "90° between X and Y around Z: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_opposite() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(-1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle.abs() - std::f64::consts::PI).abs() < 1e-10,
            "180° between X and -X: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_negative() {
        let d1 = Vec3::new(0.0, 1.0, 0.0);
        let d2 = Vec3::new(1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle + std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "-90° between Y and X around Z: got {angle}"
        );
    }
}
```

- [ ] **Step 2: Add module declaration in `builder/mod.rs`**

After `pub mod wire_builder;`:

```rust
pub mod builder_solid;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p brepkit-algo`
Expected: SUCCESS

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p brepkit-algo -- builder_solid`
Expected: 3 `angle_with_ref` tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/algo/src/builder/builder_solid.rs crates/algo/src/builder/mod.rs
git commit -m "$(cat <<'EOF'
feat(algo): OCCT-style BuilderSolid — 4-phase shell assembly

Implements PerformShapesToAvoid (iterative free-edge removal),
PerformLoops (connectivity flood-fill with GetFaceOff dihedral
angle selection), PerformAreas (Growth/Hole via signed volume),
and final Solid assembly with inner shell nesting.
EOF
)"
```

---

### Task 5: Wire BuilderSolid into the GFA pipeline

**Files:**
- Modify: `crates/algo/src/builder/assemble.rs`

- [ ] **Step 1: Delegate `assemble_solid` to `builder_solid::build_solid`**

Replace the body of `assemble_solid`:

```rust
pub fn assemble_solid(
    topo: &mut Topology,
    selected: &[SelectedFace],
) -> Result<SolidId, AlgoError> {
    super::builder_solid::build_solid(topo, selected)
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --workspace`
Expected: SUCCESS

- [ ] **Step 3: Run all algo tests**

Run: `cargo test -p brepkit-algo`
Expected: All pass

- [ ] **Step 4: Run full workspace tests (check for regressions)**

Run: `cargo test --workspace`
Expected: All currently-passing tests still pass

- [ ] **Step 5: Commit**

```bash
git add crates/algo/src/builder/assemble.rs
git commit -m "$(cat <<'EOF'
refactor(algo): delegate assemble_solid to BuilderSolid

The minimal face-dump assembly is replaced by the 4-phase
BuilderSolid with edge connectivity, dihedral angle selection,
and Growth/Hole shell classification.
EOF
)"
```

---

### Task 6: Un-ignore passing tests + regression check

**Files:**
- Modify: test files with `#[ignore = "GFA pipeline limitation"]`

- [ ] **Step 1: Run all ignored tests to see which now pass**

Run: `cargo test --workspace -- --include-ignored 2>&1 | grep -E '(FAILED|ok)' | grep -v 'test result' | sort`

Count how many of the previously-failing 48 tests now pass.

- [ ] **Step 2: Un-ignore tests that pass**

For each test that now passes, remove the `#[ignore = "..."]` attribute. Do this in batches by file.

- [ ] **Step 3: Run the full test suite**

Run: `cargo test --workspace`
Expected: All tests pass (including newly un-ignored ones), no regressions

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: Clean

- [ ] **Step 5: Build WASM**

Run: `cargo build -p brepkit-wasm --target wasm32-unknown-unknown`
Expected: SUCCESS

- [ ] **Step 6: Run boundary check**

Run: `./scripts/check-boundaries.sh`
Expected: `✅ All crate boundaries valid.`

- [ ] **Step 7: Commit**

```bash
git add -A crates/
git commit -m "$(cat <<'EOF'
test: un-ignore N tests passing with BuilderSolid + CommonBlock

BuilderSolid's edge-connectivity assembly + CommonBlock edge
deduplication fix non-manifold shell construction. N previously-
ignored GFA tests now pass.
EOF
)"
```

---

### Task 7: Final verification + PR

- [ ] **Step 1: Run the full test suite one final time**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 2: Count remaining ignored tests**

Run: `grep -r '#\[ignore' crates/ --include="*.rs" | wc -l`

Record how many tests remain ignored (target: significantly fewer than 125).

- [ ] **Step 3: Create PR**

```bash
gh pr create --title "feat(algo): BuilderSolid + CommonBlock — OCCT-style shell assembly" --body "$(cat <<'EOF'
## Summary

- Adds `CommonBlock` data structure for edge deduplication across input solids
- Adds `ForceInterfEE` phase: post-split overlap detection creates CommonBlocks
- `MakeSplitEdges` is now CB-aware: one split edge per CommonBlock group
- Replaces minimal `assemble_solid` with 4-phase `BuilderSolid`:
  1. PerformShapesToAvoid — iterative free-edge removal
  2. PerformLoops — connectivity flood-fill with GetFaceOff dihedral angle
  3. PerformAreas — Growth/Hole classification via signed volume
  4. Assemble — final Solid with inner shell nesting
- N previously-ignored GFA tests now pass

## Test plan

- [x] All workspace tests pass
- [x] N previously-ignored tests un-ignored and passing
- [x] clippy clean
- [x] WASM builds
- [x] Boundary check passes
- [x] Unit tests for angle_with_ref, CommonBlock creation
EOF
)"
```

---

## Verification

After all tasks:
- `cargo test --workspace` — all tests pass
- `cargo clippy --all-targets -- -D warnings` — clean
- `cargo build -p brepkit-wasm --target wasm32-unknown-unknown` — WASM builds
- `./scripts/check-boundaries.sh` — boundaries valid
- Remaining ignored test count significantly reduced from 125
