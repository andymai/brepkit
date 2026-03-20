# brepkit-offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new `brepkit-offset` crate at L1.5 implementing production-grade solid offset with global face-face intersection, dihedral angle edge analysis, intersection + arc joint types, and BOP-based self-intersection removal.

**Architecture:** A 9-phase pipeline: analyse edges → offset face surfaces → 3D face-face intersection → 2D edge splitting → arc joints → wire reconstruction → shell assembly → self-intersection removal → solidify. Central `OffsetData` struct tracks all intermediate state. Parallel API alongside existing offset code (v2 pattern).

**Tech Stack:** Rust, brepkit workspace crates (math, topology, geometry, algo), thiserror, log

**Spec:** `docs/superpowers/specs/2026-03-20-brepkit-offset-design.md`

---

## File Structure

```
crates/offset/
├── Cargo.toml
└── src/
    ├── lib.rs          — Public API: offset_solid, thick_solid, OffsetOptions, JointType
    ├── error.rs        — OffsetError enum with thiserror derives
    ├── data.rs         — OffsetData, OffsetFace, EdgeClass, VertexClass, OffsetStatus
    ├── analyse.rs      — Edge/vertex classification via dihedral angles
    ├── offset.rs       — Per-face surface offset (analytic fast paths + NURBS)
    ├── inter3d.rs      — 3D surface-surface intersection between adjacent offset faces
    ├── inter2d.rs      — 2D edge splitting: project intersections onto faces, split edges
    ├── loops.rs        — Wire reconstruction from split edges on each offset face
    ├── arc_joint.rs    — Pipe surfaces at convex edges + sphere caps at vertices
    ├── assemble.rs     — Shell assembly, wall generation (thick_solid), solid creation
    └── self_int.rs     — BOP-based self-intersection removal via brepkit-algo
```

Also modified:
- `Cargo.toml` (workspace root) — add `brepkit-offset` workspace member + dependency
- `scripts/check-boundaries.sh` — add offset boundary rule
- `crates/operations/Cargo.toml` — add `brepkit-offset` dependency
- `crates/operations/src/lib.rs` — add offset_v2 module
- `crates/operations/src/offset_v2.rs` — thin delegation to brepkit-offset
- `CLAUDE.md` — update layer table and module map

---

### Task 1: Crate Scaffold + Error Types + Data Structures

Create the new crate with Cargo.toml, error types, and central data structures. This is the foundation everything else builds on.

**Files:**
- Create: `crates/offset/Cargo.toml`
- Create: `crates/offset/src/lib.rs`
- Create: `crates/offset/src/error.rs`
- Create: `crates/offset/src/data.rs`
- Modify: `Cargo.toml` (workspace root, add workspace member + dep)
- Modify: `scripts/check-boundaries.sh` (add offset boundary rule)

- [ ] **Step 1: Create `crates/offset/Cargo.toml`**

```toml
[package]
name = "brepkit-offset"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Solid offset engine for brepkit"

[dependencies]
brepkit-math.workspace = true
brepkit-topology.workspace = true
brepkit-geometry.workspace = true
brepkit-algo.workspace = true
thiserror.workspace = true
log.workspace = true

[dev-dependencies]
brepkit-topology = { workspace = true, features = ["test-utils"] }

[lints]
workspace = true
```

- [ ] **Step 2: Create `crates/offset/src/error.rs`**

Define the error enum. Pattern: follow `crates/algo/src/error.rs` and `crates/blend/src/lib.rs`.

```rust
//! Offset operation error types.

use brepkit_topology::TopologyError;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;

/// Errors that can occur during offset operations.
#[derive(Debug, thiserror::Error)]
pub enum OffsetError {
    /// A topology entity was not found or is invalid.
    #[error(transparent)]
    Topology(#[from] TopologyError),

    /// Invalid input parameters.
    #[error("invalid offset input: {reason}")]
    InvalidInput {
        /// Description of what is invalid.
        reason: String,
    },

    /// Edge analysis failed for a specific edge.
    #[error("edge analysis failed for edge {edge:?}: {reason}")]
    AnalysisFailed {
        /// The edge that failed analysis.
        edge: EdgeId,
        /// What went wrong.
        reason: String,
    },

    /// Face-face intersection failed.
    #[error("intersection failed between faces {face_a:?} and {face_b:?}: {reason}")]
    IntersectionFailed {
        /// First face.
        face_a: FaceId,
        /// Second face.
        face_b: FaceId,
        /// What went wrong.
        reason: String,
    },

    /// Self-intersection could not be resolved.
    #[error("self-intersection removal failed: {reason}")]
    SelfIntersection {
        /// What went wrong.
        reason: String,
    },

    /// Shell assembly failed.
    #[error("offset shell assembly failed: {reason}")]
    AssemblyFailed {
        /// What went wrong.
        reason: String,
    },

    /// The offset caused the solid to collapse to zero or negative volume.
    #[error("offset distance causes solid to collapse")]
    CollapsedSolid,

    /// Math error during geometric computation.
    #[error("math error: {0}")]
    Math(#[from] brepkit_math::MathError),

    /// Algorithm error during boolean operation.
    #[error("algorithm error: {0}")]
    Algo(#[from] brepkit_algo::error::AlgoError),
}
```

- [ ] **Step 3: Create `crates/offset/src/data.rs`**

Central data structures used across all phases. Pattern: similar to `crates/algo/src/ds/arena.rs`.

```rust
//! Central data structures for the offset pipeline.

use std::collections::{BTreeMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

/// Edge convexity classification based on dihedral angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeClass {
    /// Adjacent faces are tangent (dihedral ≈ π). Smooth transition.
    Tangent,
    /// Faces diverge after offset (dihedral < π). Gap needs filling.
    Convex {
        /// The dihedral angle in radians.
        angle: f64,
    },
    /// Faces converge after offset (dihedral > π). Intersection needed.
    Concave {
        /// The dihedral angle in radians.
        angle: f64,
    },
}

/// Vertex classification derived from adjacent edge classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexClass {
    /// All adjacent edges are convex.
    Convex,
    /// At least one adjacent edge is concave.
    Concave,
    /// Mixed convex and concave adjacent edges.
    Mixed,
}

/// Status of an individual face offset computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetStatus {
    /// Offset computed successfully.
    Done,
    /// Face was excluded (for thick_solid).
    Excluded,
    /// Offset failed for this face.
    Failed,
}

/// Result of offsetting a single face's surface.
#[derive(Debug, Clone)]
pub struct OffsetFace {
    /// The original face ID.
    pub original: FaceId,
    /// The offset surface.
    pub surface: FaceSurface,
    /// Distance this face was offset.
    pub distance: f64,
    /// Computation status.
    pub status: OffsetStatus,
}

/// A 3D intersection curve between two offset faces.
#[derive(Debug, Clone)]
pub struct FaceIntersection {
    /// Edge in the original solid where these faces met.
    pub original_edge: EdgeId,
    /// First face.
    pub face_a: FaceId,
    /// Second face.
    pub face_b: FaceId,
    /// 3D intersection curve points (sampled).
    pub curve_points: Vec<Point3>,
    /// New edge(s) created from this intersection.
    pub new_edges: Vec<EdgeId>,
}

/// A point where an edge was split by an intersection curve.
#[derive(Debug, Clone)]
pub struct SplitPoint {
    /// Parameter along the original edge.
    pub parameter: f64,
    /// New vertex created at the split.
    pub vertex: VertexId,
}

/// Record of how an edge was split.
#[derive(Debug, Clone)]
pub struct EdgeSplitRecord {
    /// The original edge that was split.
    pub original: EdgeId,
    /// Split points in parameter order.
    pub splits: Vec<SplitPoint>,
    /// New edges created by splitting (splits.len() + 1 edges).
    pub new_edges: Vec<EdgeId>,
}

/// Joint type for edge treatment during offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JointType {
    /// Sharp edges: extend offset faces until they intersect.
    #[default]
    Intersection,
    /// Smooth edges: cylindrical pipe at edges + sphere caps at vertices.
    Arc,
}

/// Configuration options for offset operations.
#[derive(Debug, Clone)]
pub struct OffsetOptions {
    /// How to handle edges where offset faces diverge.
    pub joint: JointType,
    /// Geometric tolerance for the operation.
    pub tolerance: Tolerance,
    /// Whether to detect and remove self-intersections in the result.
    pub remove_self_intersections: bool,
}

impl Default for OffsetOptions {
    fn default() -> Self {
        Self {
            joint: JointType::Intersection,
            tolerance: Tolerance::default(),
            remove_self_intersections: true,
        }
    }
}

/// Central data structure accumulating state across all offset phases.
///
/// Each phase writes its outputs here; subsequent phases read them.
/// This is analogous to OCCT's combination of `BRepAlgo_AsDes` +
/// `DataMapOfShapeOffset`.
#[derive(Debug)]
pub struct OffsetData {
    // ── Configuration ────────────────────────────────────────────
    /// Offset distance (positive = outward, negative = inward).
    pub distance: f64,
    /// Options controlling joint type, tolerance, SI removal.
    pub options: OffsetOptions,
    /// Faces excluded from offset (for thick_solid).
    pub excluded_faces: HashSet<FaceId>,

    // ── Phase 1: Analyse ─────────────────────────────────────────
    /// Per-edge convexity classification. Keyed by edge arena index
    /// (matching `edge_to_face_map` return type from topology::explorer).
    pub edge_class: BTreeMap<usize, EdgeClass>,
    /// Per-vertex classification (derived from adjacent edges).
    pub vertex_class: BTreeMap<VertexId, VertexClass>,

    // ── Phase 2: Offset Faces ────────────────────────────────────
    /// Per-face offset result.
    pub face_offset: BTreeMap<FaceId, OffsetFace>,

    // ── Phase 3: Inter3d ─────────────────────────────────────────
    /// 3D intersection results between adjacent offset faces.
    pub intersections: Vec<FaceIntersection>,

    // ── Phase 4: Inter2d ─────────────────────────────────────────
    /// How each edge was split by intersection curves. Keyed by edge arena index.
    pub edge_splits: BTreeMap<usize, EdgeSplitRecord>,

    // ── Phase 5: Arc Joints ──────────────────────────────────────
    /// New faces created for arc joints (pipes + sphere caps).
    pub joint_faces: Vec<FaceId>,

    // ── Phase 6: MakeLoops ───────────────────────────────────────
    /// Reconstructed wires for each offset face.
    pub face_wires: BTreeMap<FaceId, Vec<WireId>>,
}

impl OffsetData {
    /// Create a new `OffsetData` for the given distance and options.
    pub fn new(distance: f64, options: OffsetOptions) -> Self {
        Self {
            distance,
            options,
            excluded_faces: HashSet::new(),
            edge_class: BTreeMap::new(),
            vertex_class: BTreeMap::new(),
            face_offset: BTreeMap::new(),
            intersections: Vec::new(),
            edge_splits: BTreeMap::new(),
            joint_faces: Vec::new(),
            face_wires: BTreeMap::new(),
        }
    }
}
```

- [ ] **Step 4: Create `crates/offset/src/lib.rs`**

Stub public API that exposes the modules and placeholder entry points.

```rust
//! # brepkit-offset
//!
//! Solid offset engine for brepkit.
//!
//! This is layer L1.5, depending on `brepkit-math`, `brepkit-topology`,
//! `brepkit-geometry`, and `brepkit-algo`. The `brepkit-operations` crate
//! delegates offset, shell, and thicken operations to this crate.
//!
//! # Architecture
//!
//! A 9-phase pipeline:
//!
//! 1. **Analyse** — classify edges as convex/concave/tangent
//! 2. **Offset** — compute offset surface for each face
//! 3. **Inter3d** — 3D face-face intersection between offset surfaces
//! 4. **Inter2d** — project intersections onto faces, split edges
//! 5. **Arc joints** — (optional) pipe + sphere cap at convex edges
//! 6. **MakeLoops** — reconstruct closed wires from split edges
//! 7. **Assemble** — build shell from trimmed faces + joints
//! 8. **SI removal** — remove self-intersections via GFA boolean
//! 9. **Solidify** — orient, validate, return solid

pub mod data;
pub mod error;

pub mod analyse;
pub mod arc_joint;
pub mod assemble;
pub mod inter2d;
pub mod inter3d;
pub mod loops;
pub mod offset;
pub mod self_int;

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

pub use data::{JointType, OffsetOptions};
pub use error::OffsetError;

/// Offset all faces of a solid by a uniform distance.
///
/// Positive `distance` offsets outward (solid grows); negative offsets inward.
///
/// # Errors
///
/// Returns [`OffsetError`] if any phase of the pipeline fails.
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    options: OffsetOptions,
) -> Result<SolidId, OffsetError> {
    thick_solid(topo, solid, distance, &[], options)
}

/// Create a hollow solid by offsetting with excluded (open) faces.
///
/// Excluded faces are removed, and the remaining faces are offset by `distance`.
/// Wall faces are generated to connect original boundaries at excluded faces
/// to the offset boundaries.
///
/// # Errors
///
/// Returns [`OffsetError`] if any phase of the pipeline fails.
pub fn thick_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    exclude: &[FaceId],
    options: OffsetOptions,
) -> Result<SolidId, OffsetError> {
    let tol = options.tolerance;

    if tol.approx_eq(distance, 0.0) {
        return Err(OffsetError::InvalidInput {
            reason: "offset distance must be non-zero".into(),
        });
    }

    // Build pipeline data.
    let mut data = data::OffsetData::new(distance, options);
    data.excluded_faces = exclude.iter().copied().collect();

    // Phase 1: Analyse edges.
    analyse::analyse_edges(topo, solid, &mut data)?;

    // Phase 2: Offset each face's surface.
    offset::offset_faces(topo, solid, &mut data)?;

    // Phase 3: 3D face-face intersection.
    inter3d::compute_intersections(topo, solid, &mut data)?;

    // Phase 4: 2D edge splitting.
    inter2d::split_edges(topo, &mut data)?;

    // Phase 5: Arc joints (if requested).
    if data.options.joint == JointType::Arc {
        arc_joint::create_joints(topo, solid, &mut data)?;
    }

    // Phase 6: Reconstruct wires.
    loops::make_loops(topo, &mut data)?;

    // Phase 7: Assemble shell + solid.
    let result = assemble::assemble(topo, solid, &mut data)?;

    // Phase 8: Self-intersection removal.
    let result = if data.options.remove_self_intersections {
        self_int::remove_self_intersections(topo, result, &data)?
    } else {
        result
    };

    Ok(result)
}
```

- [ ] **Step 5: Add workspace dependency in root `Cargo.toml`**

Add `brepkit-offset` to the `[workspace.dependencies]` section (after `brepkit-geometry`):

```toml
brepkit-offset = { path = "crates/offset" }
```

- [ ] **Step 6: Update `scripts/check-boundaries.sh`**

Add the offset crate boundary rule. Insert after the `check_deps "check"` line:

```bash
check_deps "offset"    "brepkit-math" "brepkit-topology" "brepkit-geometry" "brepkit-algo"
```

Also add `brepkit-offset` to the operations allowed deps:

```bash
check_deps "operations" "brepkit-math" "brepkit-topology" "brepkit-algo" "brepkit-blend" "brepkit-heal" "brepkit-check" "brepkit-geometry" "brepkit-offset"
```

And add `brepkit-offset` to the wasm allowed deps line.

- [ ] **Step 7: Create stub modules**

Create empty stub files for all remaining modules so the crate compiles:

Each file should contain a doc comment and a placeholder function that returns `todo!()`. For example, `analyse.rs`:

```rust
//! Edge and vertex convexity classification.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;
use crate::error::OffsetError;

/// Classify every edge of the solid as Convex, Concave, or Tangent.
///
/// Populates `data.edge_class` and `data.vertex_class`.
pub fn analyse_edges(
    _topo: &Topology,
    _solid: SolidId,
    _data: &mut OffsetData,
) -> Result<(), OffsetError> {
    todo!("Phase 1: edge analysis")
}
```

Create similar stubs for: `offset.rs` (`offset_faces`), `inter3d.rs` (`compute_intersections`), `inter2d.rs` (`split_edges`), `arc_joint.rs` (`create_joints`), `loops.rs` (`make_loops`), `assemble.rs` (`assemble`), `self_int.rs` (`remove_self_intersections`).

- [ ] **Step 8: Verify crate compiles and boundary check passes**

Run: `cargo build -p brepkit-offset 2>&1 | tail -5`
Expected: Compiles (with `todo!()` warnings)

Run: `./scripts/check-boundaries.sh`
Expected: `✅ All crate boundaries valid.`

- [ ] **Step 9: Commit**

```bash
git add crates/offset/ Cargo.toml scripts/check-boundaries.sh
git commit -m "feat(offset): scaffold brepkit-offset crate with data structures and error types"
```

---

### Task 2: Phase 1 — Edge Analysis

Implement dihedral angle computation and edge classification. This is the foundation that drives the entire pipeline — every subsequent phase reads from `data.edge_class`.

**Files:**
- Create/modify: `crates/offset/src/analyse.rs`
- Reference: `crates/topology/src/explorer.rs` (for `edge_to_face_map`)
- Reference: `crates/topology/src/adjacency.rs` (for `faces_for_edge`)
- Reference: `crates/math/src/traits.rs` (for `ParametricCurve`, `ParametricSurface` evaluate/normal)

- [ ] **Step 1: Write failing tests for edge classification**

In `analyse.rs`, add a `#[cfg(test)]` module with these tests:

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;

    use super::*;
    use crate::data::{EdgeClass, OffsetData, OffsetOptions};

    /// Helper: analyse a primitive and return the OffsetData.
    fn analyse_primitive(topo: &mut Topology, solid: SolidId) -> OffsetData {
        let mut data = OffsetData::new(1.0, OffsetOptions::default());
        analyse_edges(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_all_edges_convex() {
        // A box has 12 edges, all with dihedral angle π/2 (convex).
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let data = analyse_primitive(&mut topo, solid);

        assert_eq!(data.edge_class.len(), 12, "box should have 12 classified edges");
        for (_eid, class) in &data.edge_class {
            assert!(
                matches!(class, EdgeClass::Convex { .. }),
                "all box edges should be convex, got {class:?}"
            );
        }
    }

    #[test]
    fn box_edge_angle_is_half_pi() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let data = analyse_primitive(&mut topo, solid);

        for (_eid, class) in &data.edge_class {
            if let EdgeClass::Convex { angle } = class {
                let err = (angle - std::f64::consts::FRAC_PI_2).abs();
                assert!(err < 0.1, "box dihedral should be ≈ π/2, got {angle:.4}");
            }
        }
    }

    #[test]
    fn vertex_classification_box() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let data = analyse_primitive(&mut topo, solid);

        assert_eq!(data.vertex_class.len(), 8, "box should have 8 classified vertices");
        for (_vid, class) in &data.vertex_class {
            assert_eq!(
                *class,
                crate::data::VertexClass::Convex,
                "all box vertices should be convex"
            );
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p brepkit-offset -- analyse 2>&1 | tail -10`
Expected: FAIL (currently `todo!()` panic)

- [ ] **Step 3: Implement `analyse_edges`**

Replace the stub in `analyse.rs` with the full implementation. The algorithm:

1. Get the solid's outer shell and collect all faces.
2. Build an `edge_to_face_map` using `brepkit_topology::explorer::edge_to_face_map`.
3. For each edge that has exactly 2 adjacent faces, compute the dihedral angle:
   - Sample the edge at its midpoint (t = 0.5).
   - Evaluate each face's surface normal at the edge midpoint.
   - Compute dihedral angle: `θ = atan2((n1 × n2) · t, n1 · n2)` where `t` is the edge tangent.
   - Classify: if |θ - π| < tol_angle → Tangent; if θ < π → Convex; if θ > π → Concave.
4. For edges with ≠ 2 faces (boundary or non-manifold), skip classification.
5. Classify vertices based on their adjacent edge classes.

Key references:
- `crates/topology/src/explorer.rs:140` — `edge_to_face_map(topo, solid_id)` returns `BTreeMap<EdgeId, SmallVec<[FaceId; 2]>>`
- `crates/topology/src/edge.rs` — `EdgeCurve` variants, evaluate via `edge.evaluate_with_endpoints(t, topo)`
- `crates/math/src/traits.rs` — `ParametricSurface::normal(u, v)`, `ParametricCurve::evaluate(t)`
- Face surface normal: use the `FaceSurface` delegate method `normal(u, v)` or extract from `FaceSurface::Plane { normal, .. }` directly.

For getting the UV coordinates of an edge midpoint on a face's surface, use `project_point_to_surface` from geometry, or for planar faces just use the plane normal directly.

```rust
//! Edge and vertex convexity classification.
//!
//! Computes the dihedral angle at every edge of the input solid and
//! classifies each edge as Convex (faces diverge, angle < π),
//! Concave (faces converge, angle > π), or Tangent (smooth, angle ≈ π).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::edge_to_face_map;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::data::{EdgeClass, OffsetData, VertexClass};
use crate::error::OffsetError;

/// Classify every edge of the solid as Convex, Concave, or Tangent.
///
/// Populates `data.edge_class` and `data.vertex_class`.
pub fn analyse_edges(
    topo: &Topology,
    solid: SolidId,
    data: &mut OffsetData,
) -> Result<(), OffsetError> {
    // edge_to_face_map returns HashMap<usize, SmallVec<[FaceId; 2]>>
    // where the key is the edge's arena index, not EdgeId.
    let edge_faces = edge_to_face_map(topo, solid)?;
    let tol = data.options.tolerance;

    // Tolerance angle adapts to offset magnitude (from OCCT).
    let tol_angle = compute_tol_angle(tol, data.distance);

    // Phase 1a: classify edges.
    let mut vertex_edges: BTreeMap<VertexId, Vec<usize>> = BTreeMap::new();

    for (edge_idx, faces) in &edge_faces {
        if faces.len() != 2 {
            // Skip boundary or non-manifold edges.
            continue;
        }

        // Convert usize index to EdgeId via the topology helper.
        let edge_id = topo.edge_id_from_index(*edge_idx)
            .ok_or_else(|| OffsetError::InvalidInput {
                reason: format!("edge index {edge_idx} not found in arena"),
            })?;

        let class = classify_edge(topo, edge_id, faces[0], faces[1], tol_angle)?;
        data.edge_class.insert(*edge_idx, class);

        // Track vertex-edge adjacency for vertex classification.
        let edge = topo.edge(edge_id)?;
        vertex_edges.entry(edge.start()).or_default().push(*edge_idx);
        vertex_edges.entry(edge.end()).or_default().push(*edge_idx);
    }

    // Phase 1b: classify vertices from their adjacent edges.
    for (vid, edges) in &vertex_edges {
        let class = classify_vertex(edges, &data.edge_class);
        data.vertex_class.insert(*vid, class);
    }

    Ok(())
}

/// Compute the adaptive tolerance angle for edge classification.
///
/// From OCCT: `tol_angle = 4 * arcsin(min(tol / (|offset| * 0.5), 1.0))`.
/// This widens the tangent classification band for small offsets (where
/// geometric noise has more relative impact).
fn compute_tol_angle(tol: Tolerance, distance: f64) -> f64 {
    let ratio = (tol.linear / (distance.abs() * 0.5)).min(1.0);
    4.0 * ratio.asin()
}

/// Classify a single edge by computing its dihedral angle.
fn classify_edge(
    topo: &Topology,
    edge_id: EdgeId,
    face_a: FaceId,
    face_b: FaceId,
    tol_angle: f64,
) -> Result<EdgeClass, OffsetError> {
    // Get edge midpoint in 3D.
    let edge = topo.edge(edge_id)?;
    let p_start = topo.vertex(edge.start())?.point();
    let p_end = topo.vertex(edge.end())?.point();
    let mid = p_start.midpoint(p_end);

    // Compute edge tangent direction.
    let tangent = (p_end - p_start).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    // Get face normals at the edge midpoint.
    let n_a = face_normal_at_point(topo, face_a, mid)?;
    let n_b = face_normal_at_point(topo, face_b, mid)?;

    // Dihedral angle: angle between the two face normals, measured
    // in the plane perpendicular to the edge tangent.
    // θ = atan2((n_a × n_b) · t, n_a · n_b)
    let cross = n_a.cross(n_b);
    let sin_theta = cross.dot(tangent);
    let cos_theta = n_a.dot(n_b);
    let dihedral = sin_theta.atan2(cos_theta);

    // Convert to [0, 2π] range (dihedral angle between outward normals).
    // For a convex edge (like a box edge), normals point away from each
    // other: cos_theta ≈ 0, dihedral ≈ π/2.
    // The "opening angle" is π - dihedral for our sign convention.
    let angle = PI - dihedral;

    // Classify based on the opening angle.
    if (angle - PI).abs() < tol_angle {
        Ok(EdgeClass::Tangent)
    } else if angle < PI {
        Ok(EdgeClass::Convex { angle })
    } else {
        Ok(EdgeClass::Concave { angle })
    }
}

/// Get the outward face normal at a 3D point.
///
/// For planar faces, returns the plane normal directly.
/// For curved faces, projects the point onto the surface and evaluates.
fn face_normal_at_point(
    topo: &Topology,
    face: FaceId,
    point: brepkit_math::vec::Point3,
) -> Result<Vec3, OffsetError> {
    let face_data = topo.face(face)?;
    let reversed = face_data.is_reversed();

    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Cylinder(cyl) => {
            let (u, v) = cyl.project_point(point);
            cyl.normal(u, v)
        }
        FaceSurface::Cone(cone) => {
            let (u, v) = cone.project_point(point);
            cone.normal(u, v)
        }
        FaceSurface::Sphere(sphere) => {
            let (u, v) = sphere.project_point(point);
            sphere.normal(u, v)
        }
        FaceSurface::Torus(torus) => {
            let (u, v) = torus.project_point(point);
            torus.normal(u, v)
        }
        FaceSurface::Nurbs(nurbs) => {
            // Project point onto NURBS surface using geometry crate.
            // point_to_surface requires (point, &impl ParametricSurface, u_range, v_range).
            let (u_min, u_max) = nurbs.domain_u();
            let (v_min, v_max) = nurbs.domain_v();
            let proj = brepkit_geometry::extrema::point_surface::point_to_surface(
                point, nurbs, (u_min, u_max), (v_min, v_max),
            );
            nurbs.normal(proj.u, proj.v).map_err(|e| OffsetError::InvalidInput {
                reason: format!("NURBS normal failed: {e}"),
            })?
        }
    };

    // Respect face orientation.
    Ok(if reversed { -normal } else { normal })
}

/// Classify a vertex based on its adjacent edge classifications.
fn classify_vertex(
    edge_indices: &[usize],
    edge_class: &BTreeMap<usize, EdgeClass>,
) -> VertexClass {
    let mut has_convex = false;
    let mut has_concave = false;

    for idx in edge_indices {
        match edge_class.get(idx) {
            Some(EdgeClass::Convex { .. }) => has_convex = true,
            Some(EdgeClass::Concave { .. }) => has_concave = true,
            _ => {}
        }
    }

    match (has_convex, has_concave) {
        (true, true) => VertexClass::Mixed,
        (false, true) => VertexClass::Concave,
        _ => VertexClass::Convex,
    }
}
```

**Important:** The `face_normal_at_point` function needs to handle the `project_point` method on analytic surfaces. Check that `CylindricalSurface`, `ConicalSurface`, `SphericalSurface`, and `ToroidalSurface` all have a `project_point` method in `crates/math/src/surfaces.rs`. If not, you'll need to implement it (project the 3D point to the nearest surface point and return (u, v) parameters). Similarly, check `point_to_surface` in `crates/geometry/src/extrema/point_surface.rs` for the NURBS path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p brepkit-offset -- analyse 2>&1 | tail -15`
Expected: All 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/analyse.rs
git commit -m "feat(offset): implement Phase 1 edge analysis with dihedral angle classification"
```

---

### Task 3: Phase 2 — Face Surface Offset

Implement per-face surface offset with analytic fast paths for plane, cylinder, cone, sphere, torus, and NURBS fallback.

**Files:**
- Create/modify: `crates/offset/src/offset.rs`
- Reference: `crates/operations/src/offset_face.rs` (existing approach, can borrow patterns)
- Reference: `crates/math/src/surfaces.rs` (analytic surface constructors)

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;

    use super::*;
    use crate::data::{OffsetData, OffsetOptions, OffsetStatus};

    fn make_offset_data(dist: f64) -> OffsetData {
        OffsetData::new(dist, OffsetOptions::default())
    }

    #[test]
    fn offset_box_faces_are_planes() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let mut data = make_offset_data(0.5);

        // Run analyse first (Phase 1 prerequisite).
        crate::analyse::analyse_edges(&topo, solid, &mut data).unwrap();
        offset_faces(&mut topo, solid, &mut data).unwrap();

        assert_eq!(data.face_offset.len(), 6, "box has 6 faces");
        for (_fid, of) in &data.face_offset {
            assert_eq!(of.status, OffsetStatus::Done);
            assert!(
                matches!(of.surface, FaceSurface::Plane { .. }),
                "offset of plane should be plane"
            );
        }
    }

    #[test]
    fn offset_plane_shifts_d() {
        // A plane n·x=d offset by +0.5 should become n·x = d+0.5.
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let mut data = make_offset_data(0.5);
        crate::analyse::analyse_edges(&topo, solid, &mut data).unwrap();
        offset_faces(&mut topo, solid, &mut data).unwrap();

        // Check at least one face's d value shifted.
        let any_shifted = data.face_offset.values().any(|of| {
            if let FaceSurface::Plane { d, .. } = &of.surface {
                // Original box has d values of 0 and 1. Offset by 0.5 should produce 0.5 and 1.5.
                let tol = Tolerance::new();
                tol.approx_eq(*d, 0.5) || tol.approx_eq(*d, 1.5)
                    || tol.approx_eq(*d, -0.5) // negative-facing planes
            } else {
                false
            }
        });
        assert!(any_shifted, "at least one plane should have shifted d");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p brepkit-offset -- offset 2>&1 | tail -10`
Expected: FAIL

- [ ] **Step 3: Implement `offset_faces`**

The `offset_faces` function iterates over all faces of the solid. For each face:
- If excluded → mark `OffsetStatus::Excluded`, skip.
- Match on `FaceSurface` variant:
  - `Plane { normal, d }` → `Plane { normal, d: d + distance }` (exact)
  - `Cylinder(cyl)` → new `CylindricalSurface` with `radius ± distance` (exact, check for collapse)
  - `Cone(cone)` → adjust apex distance / half-angle (exact)
  - `Sphere(sphere)` → new `SphericalSurface` with `radius ± distance` (exact, check for collapse)
  - `Torus(torus)` → adjust minor radius (exact, check for self-intersection)
  - `Nurbs(nurbs)` → sample-and-refit approach (from existing `offset_face.rs`)

Pattern: follow `crates/operations/src/offset_face.rs` for the analytic fast paths.

**Key:** The face's `is_reversed()` flag affects the offset direction. If reversed, distance should be negated for that face's surface offset.

- [ ] **Step 4: Run tests**

Run: `cargo test -p brepkit-offset -- offset 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/offset.rs
git commit -m "feat(offset): implement Phase 2 face surface offset with analytic fast paths"
```

---

### Task 4: Phase 3 — 3D Face-Face Intersection

Compute intersection curves between pairs of adjacent offset faces. This is the core algorithm that makes offset work correctly.

**Files:**
- Create/modify: `crates/offset/src/inter3d.rs`
- Reference: `crates/math/src/analytic_intersection.rs` (`intersect_analytic_analytic`, `exact_plane_analytic`)
- Reference: `crates/math/src/nurbs/intersection.rs` (`intersect_nurbs_nurbs`)
- Reference: `crates/topology/src/explorer.rs` (`edge_to_face_map`)

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;

    use super::*;
    use crate::data::{OffsetData, OffsetOptions};

    fn run_phases_1_2_3(topo: &mut Topology, solid: SolidId, distance: f64) -> OffsetData {
        let mut data = OffsetData::new(distance, OffsetOptions::default());
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::offset_faces(topo, solid, &mut data).unwrap();
        compute_intersections(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_offset_produces_12_intersections() {
        // A box has 12 edges. Each edge produces one face-face intersection.
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let data = run_phases_1_2_3(&mut topo, solid, 0.5);

        assert_eq!(
            data.intersections.len(), 12,
            "box offset should produce 12 face-face intersections (one per edge)"
        );
    }

    #[test]
    fn box_intersection_curves_are_nonempty() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let data = run_phases_1_2_3(&mut topo, solid, 0.5);

        for intersection in &data.intersections {
            assert!(
                !intersection.curve_points.is_empty(),
                "intersection for edge {:?} should have points",
                intersection.original_edge
            );
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `compute_intersections`**

Algorithm:
1. Get `edge_to_face_map` from the topology.
2. For each edge with 2 adjacent faces (that both have offset surfaces in `data.face_offset`):
   a. Get the two offset surfaces.
   b. Dispatch to the appropriate intersection algorithm:
      - Plane-Plane → line intersection (exact via cross product of normals)
      - Plane-Analytic → `exact_plane_analytic` from math crate
      - Analytic-Analytic → `intersect_analytic_analytic` from math crate
      - NURBS-anything → `intersect_nurbs_nurbs` from math crate (convert analytic to NURBS first if needed)
   c. Store intersection curve points in `data.intersections`.

**Key detail:** For plane-plane intersection, the result is an infinite line. We need to sample it within the face bounds. Use the face's bounding box to clip the line to a reasonable segment.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/inter3d.rs
git commit -m "feat(offset): implement Phase 3 face-face intersection for offset surfaces"
```

---

### Task 5: Phase 4 — 2D Edge Splitting

Project intersection curves onto faces and split edges where they cross boundaries.

**Files:**
- Create/modify: `crates/offset/src/inter2d.rs`
- Reference: `crates/math/src/nurbs/projection.rs` (point projection)
- Reference: `crates/topology/src/edge.rs` (edge construction)

- [ ] **Step 1: Write failing tests**

Test that after Phase 4, edges that were at convex/concave junctions have been split.

- [ ] **Step 2: Run tests to verify failure**

- [ ] **Step 3: Implement `split_edges`**

Algorithm:
1. For each intersection result (from Phase 3), determine which original edges the intersection curve crosses.
2. For each crossing: compute the parameter value along the edge, create a new vertex at the crossing point.
3. Split the edge at each crossing point into sub-edges.
4. Store results in `data.edge_splits`.

For the initial implementation (all-planar offset), the intersection curves from Phase 3 are lines. The crossing computation simplifies to line-line intersection in the face's UV plane.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/inter2d.rs
git commit -m "feat(offset): implement Phase 4 edge splitting from intersection curves"
```

---

### Task 6: Phase 5 — Arc Joints

Create cylindrical pipe surfaces at convex edges and spherical caps at convex vertices for `JointType::Arc`.

**Files:**
- Create/modify: `crates/offset/src/arc_joint.rs`
- Reference: `crates/blend/src/analytic.rs` (analytic fillet creation)
- Reference: `crates/math/src/surfaces.rs` (CylindricalSurface, SphericalSurface)

- [ ] **Step 1: Write failing tests**

Test that arc joint mode produces additional faces at convex edges of a box.

- [ ] **Step 2: Run tests to verify failure**

- [ ] **Step 3: Implement `create_joints`**

Algorithm:
1. For each convex edge in `data.edge_class`:
   - Create a cylindrical pipe surface whose axis follows the edge, radius = |distance|.
   - The pipe is tangent to both adjacent offset faces.
   - Create a face for the pipe, bounded by the edge endpoints and the contact curves on the adjacent faces.
2. For each convex vertex in `data.vertex_class`:
   - Create a spherical cap centered at the original vertex, radius = |distance|.
   - The cap connects to all adjacent pipe faces.
3. Add new faces to `data.joint_faces`.

**Note:** For `JointType::Intersection`, this phase is skipped entirely. The initial integration tests should use intersection mode. Arc mode is tested separately.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/arc_joint.rs
git commit -m "feat(offset): implement Phase 5 arc joints (pipe + sphere cap)"
```

---

### Task 7: Phase 6 — Wire Reconstruction

Reconstruct closed wires from the split edges on each offset face.

**Files:**
- Create/modify: `crates/offset/src/loops.rs`
- Reference: `crates/algo/src/builder/wire_builder.rs` (similar wire reconstruction)
- Reference: `crates/topology/src/wire.rs` (Wire, OrientedEdge)

- [ ] **Step 1: Write failing tests**

Test that after Phase 6, each offset face has at least one closed wire.

- [ ] **Step 2: Run tests to verify failure**

- [ ] **Step 3: Implement `make_loops`**

Algorithm:
1. For each non-excluded face in `data.face_offset`:
   - Collect all edges belonging to this face: original boundary edges (possibly split by Phase 4) + new intersection edges (from Phase 3).
   - Build a vertex → outgoing-edge graph.
   - Walk the graph to form closed loops.
   - Classify each loop as outer or inner based on signed area in UV space.
   - Create Wire topology entities.
2. Store results in `data.face_wires`.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/loops.rs
git commit -m "feat(offset): implement Phase 6 wire reconstruction from split edges"
```

---

### Task 8: Phase 7 — Shell Assembly + Phase 9 Solidify

Assemble offset faces into a shell and create the final solid. Also handles wall generation for `thick_solid`.

**Files:**
- Create/modify: `crates/offset/src/assemble.rs`
- Reference: `crates/topology/src/builder.rs` (face/shell/solid construction)
- Reference: `crates/operations/src/boolean/assembly.rs` (shell assembly pattern)

- [ ] **Step 1: Write failing integration tests**

These are the end-to-end tests that validate the full pipeline:

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn offset_box_outward_volume() {
        // Box 2³ offset by 0.5 → 3³ = 27.0
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo) // 1x1x1 cube;
        let result = crate::offset_solid(&mut topo, solid, 0.5, Default::default()).unwrap();

        // Measure volume (need to add brepkit-check or brepkit-operations as dev-dep).
        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6, "offset box should have 6 faces");
    }

    #[test]
    fn offset_box_inward_volume() {
        // Unit cube offset inward by -0.1 → 0.8³ = 0.512
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube(&mut topo);
        let result = crate::offset_solid(&mut topo, solid, -0.1, Default::default()).unwrap();

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6);
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

- [ ] **Step 3: Implement `assemble`**

Algorithm:
1. For each offset face: create a new `Face` entity with the offset surface and reconstructed wires (from Phase 6).
2. Collect all offset faces + any arc joint faces (from Phase 5).
3. For `thick_solid` with excluded faces: generate "wall" faces that connect original face boundaries to offset boundaries at each excluded face edge.
4. Create a `Shell` from all faces.
5. Create a `Solid` from the shell.
6. Validate: check that the shell is closed (no boundary edges).

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/assemble.rs
git commit -m "feat(offset): implement Phase 7 shell assembly and solidification"
```

---

### Task 9: Phase 8 — Self-Intersection Removal

Use the GFA boolean engine to remove self-intersecting regions.

**Files:**
- Create/modify: `crates/offset/src/self_int.rs`
- Reference: `crates/algo/src/gfa.rs` (`boolean` function)
- Reference: `crates/algo/src/bop.rs` (`BooleanOp::Intersect`)

- [ ] **Step 1: Write failing test**

Test with a concave L-shaped solid where inward offset causes self-intersection.

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement `remove_self_intersections`**

Algorithm (two-stage approach):

**Stage 1: Detection.** Sample the offset shell's faces and check for normal
inversion — faces whose outward normal points inward (dot product with original
face normal < 0) indicate self-intersection. Also check for zero-volume regions
using the signed-tetrahedra method.

**Stage 2: Removal.** If self-intersections are detected:
1. Build a bounding box for the offset solid.
2. Create a slightly-larger box that fully contains the offset.
3. Use `algo::gfa::boolean(topo, Intersect, offset_solid, bounding_box)`.
   The intersection with a known-good convex solid clips away inverted regions.
4. If the boolean fails, fall back to face-by-face normal filtering:
   remove faces whose normals are inverted relative to the original solid.

Note: `boolean(Intersect, A, A)` is an identity operation (A ∩ A = A), so we
cannot directly intersect the solid with itself. The bounding-box intersection
approach exploits the GFA engine's ability to split faces at self-crossing
boundaries while the convex tool ensures correct classification.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add crates/offset/src/self_int.rs
git commit -m "feat(offset): implement Phase 8 BOP-based self-intersection removal"
```

---

### Task 10: Operations Integration + WASM Bindings

Wire the new offset engine into the operations crate and add WASM bindings.

**Files:**
- Modify: `crates/operations/Cargo.toml` (add brepkit-offset dep)
- Create: `crates/operations/src/offset_v2.rs` (thin delegation)
- Modify: `crates/operations/src/lib.rs` (add offset_v2 module)
- Modify: `crates/wasm/Cargo.toml` (add brepkit-offset dep)
- Modify: `crates/wasm/src/bindings/operations.rs` (add offsetSolidV2, thickSolidV2)
- Modify: `crates/wasm/src/bindings/batch.rs` (add batch dispatch)

- [ ] **Step 1: Add brepkit-offset to operations Cargo.toml**

```toml
brepkit-offset.workspace = true
```

- [ ] **Step 2: Create `crates/operations/src/offset_v2.rs`**

Thin delegation layer:

```rust
//! V2 offset operations delegating to brepkit-offset.

use brepkit_offset::{JointType, OffsetOptions};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Offset all faces of a solid (V2 pipeline).
pub fn offset_solid_v2(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    brepkit_offset::offset_solid(topo, solid, distance, OffsetOptions::default())
        .map_err(|e| OperationsError::InvalidInput {
            reason: format!("offset v2: {e}"),
        })
}

/// Shell (hollow solid) operation (V2 pipeline).
pub fn shell_v2(
    topo: &mut Topology,
    solid: SolidId,
    thickness: f64,
    exclude: &[FaceId],
) -> Result<SolidId, OperationsError> {
    brepkit_offset::thick_solid(
        topo, solid, thickness, exclude,
        OffsetOptions::default(),
    )
    .map_err(|e| OperationsError::InvalidInput {
        reason: format!("shell v2: {e}"),
    })
}

/// Offset with arc joints (V2 pipeline).
pub fn offset_solid_arc_v2(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    let options = OffsetOptions {
        joint: JointType::Arc,
        ..Default::default()
    };
    brepkit_offset::offset_solid(topo, solid, distance, options)
        .map_err(|e| OperationsError::InvalidInput {
            reason: format!("offset v2 arc: {e}"),
        })
}
```

- [ ] **Step 3: Add module to operations lib.rs**

Add `pub mod offset_v2;` to `crates/operations/src/lib.rs`.

- [ ] **Step 4: Add WASM bindings**

In `crates/wasm/src/bindings/operations.rs`, add:

```rust
#[wasm_bindgen(js_name = "offsetSolidV2")]
pub fn offset_solid_v2(&mut self, solid: u32, distance: f64) -> Result<u32, JsError> {
    let sid = self.resolve_solid(solid)?;
    let result = brepkit_operations::offset_v2::offset_solid_v2(&mut self.topo, sid, distance)?;
    Ok(solid_id_to_u32(result))
}

#[wasm_bindgen(js_name = "shellV2")]
pub fn shell_v2(&mut self, solid: u32, thickness: f64, exclude_faces: &[u32]) -> Result<u32, JsError> {
    let sid = self.resolve_solid(solid)?;
    let exclude: Vec<_> = exclude_faces.iter().map(|&f| face_id_from_u32(f)).collect();
    let result = brepkit_operations::offset_v2::shell_v2(&mut self.topo, sid, thickness, &exclude)?;
    Ok(solid_id_to_u32(result))
}
```

- [ ] **Step 5: Run full test suite**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass (new + existing).

Run: `cargo build -p brepkit-wasm --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: WASM build succeeds.

- [ ] **Step 6: Commit**

```bash
git add crates/operations/ crates/wasm/
git commit -m "feat(operations,wasm): integrate brepkit-offset v2 with WASM bindings"
```

---

### Task 11: Integration Tests + Volume Validation

Add comprehensive integration tests that validate the full offset pipeline against known volumes.

**Files:**
- Create: `crates/offset/tests/integration.rs`

- [ ] **Step 1: Write integration tests**

Create a dedicated integration test file. These tests need `brepkit-operations` for `make_box`, `make_cylinder`, `make_sphere`, and `solid_volume`. Add to dev-dependencies if needed, or use `brepkit-topology::test_utils` for simple shapes.

Tests to write:
1. Box 2³ offset +0.5 → volume = 3³ = 27.0
2. Box 4³ offset -0.5 → volume = 3³ = 27.0
3. Box 3×5×7 offset +1.0 → volume = 5×7×9 = 315.0
4. Offset zero distance → error
5. Offset collapse (box 1³ offset -0.6) → error
6. Thick solid: box with top face removed → hollow box
7. Arc joint: box offset +0.5 with arc mode → more faces than intersection mode

- [ ] **Step 2: Run tests**

Run: `cargo test -p brepkit-offset 2>&1 | tail -20`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add crates/offset/tests/
git commit -m "test(offset): add integration tests with volume validation"
```

---

### Task 12: Documentation + CLAUDE.md Update

Update project documentation to reflect the new crate.

**Files:**
- Modify: `CLAUDE.md` (add offset to layer table, module map, boundary rules, cookbook)

- [ ] **Step 1: Update CLAUDE.md layer table**

Add `offset` row:
```
| `offset`  | `math`, `topology`, `geometry`, `algo` |
```

Update `operations` row to include `offset`.

- [ ] **Step 2: Add offset to Module Map**

Add a new `### L1.5: offset` section listing all files.

- [ ] **Step 3: Update operations dependency in CLAUDE.md**

Add `brepkit-offset` to the operations allowed deps.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add brepkit-offset to CLAUDE.md layer table and module map"
```

---

### Task 13: Final Verification

Run the full CI suite to verify everything works together.

- [ ] **Step 1: Run boundary check**

Run: `./scripts/check-boundaries.sh`
Expected: `✅ All crate boundaries valid.`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -10`
Expected: No warnings.

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 4: Run fmt**

Run: `cargo fmt --all --check`
Expected: No formatting issues.

- [ ] **Step 5: Build WASM**

Run: `cargo build -p brepkit-wasm --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: Build succeeds.
