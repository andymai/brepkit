# Analytic Face CDT — Watertight Tessellation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make post-boolean analytic face meshes (cylinder, cone, sphere, torus) watertight by replacing snap-based stitching with CDT tessellation using shared boundary vertices and direct UV assignment for degenerate seams.

**Architecture:** Three-layer approach: (1) tolerance-based grid merge in Phase 3 to unify vertices from separately-sampled edges tracing the same curve, (2) Phase 3b edge refinement to augment circle edges with vertices from co-located edges, (3) CDT tessellation for analytic faces with direct UV seam assignment, plus a weld post-process safety net. Falls back to snap-based tessellation if CDT fails for any face.

**Tech Stack:** Rust, `brepkit-math` CDT module, `brepkit_math::tolerance::Tolerance`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/operations/src/tessellate.rs` | Modify | All tessellate_solid changes: point_merge_key, Phase 3b refinement, analytic CDT dispatch, seam UV handling, weld_boundary_vertices |
| `crates/math/src/cdt.rs` | Modify | Collinear vertex splitting in insert_constraint |

All changes are within existing files — no new files created.

## Chunk 1: Foundation — Grid Merge + CDT Collinear Splitting

---

### Task 1: Switch vertex dedup from bit-exact to tolerance-based grid

**Files:**
- Modify: `crates/operations/src/tessellate.rs:2608-2642` (Phase 3 vertex pool)

The current code uses `(pt.x().to_bits(), pt.y().to_bits(), pt.z().to_bits())` for vertex dedup. This only merges bit-identical vertices. After boolean ops, adjacent faces often have separate edge entities tracing the same curve — their independently-sampled vertices differ by floating-point noise. We need tolerance-based merging.

- [ ] **Step 1: Write the failing test**

Add a test that creates a boolean-cut cylinder, tessellates it, and checks for watertightness. This will fail because the current snap-based tessellation doesn't guarantee watertight meshes for post-boolean analytic faces.

```rust
#[test]
fn tessellate_boolean_cut_cylinder_watertight() {
    let mut topo = Topology::new();
    let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 4.0)
        .expect("cylinder");
    let tool = crate::primitives::make_box(&mut topo, 3.0, 3.0, 3.0)
        .expect("box");
    // Move tool so it cuts through the cylinder
    let tool = crate::transform::translate(&mut topo, tool, Vec3::new(-0.5, -0.5, 0.5))
        .expect("translate");
    let cut = crate::boolean::boolean_cut(&mut topo, cyl, tool)
        .expect("boolean cut");
    let mesh = super::tessellate_solid(&topo, cut, 0.1)
        .expect("tessellate");
    assert!(
        mesh.positions.len() > 10,
        "mesh should have vertices, got {}",
        mesh.positions.len()
    );
    let boundary = super::boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "mesh should be watertight (0 boundary edges), got {boundary}"
    );
}
```

Location: add to `mod tests` block near end of `tessellate.rs`, alongside existing `tessellate_solid_box_watertight`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brepkit-operations -- tessellate::tests::tessellate_boolean_cut_cylinder_watertight`
Expected: FAIL — boundary_edge_count > 0

- [ ] **Step 3: Add `point_merge_key` function and switch Phase 3 to use it**

Add the helper function above `tessellate_solid`:

```rust
/// Compute a tolerance-based grid key for vertex deduplication.
///
/// Vertices within the same grid cell merge to a single global ID.
/// Grid size uses `Tolerance::default().linear` (1e-7) which handles
/// boolean results where adjacent faces have separate edge entities
/// for the same geometric curve.
#[inline]
fn point_merge_key(pt: Point3, grid: f64) -> (i64, i64, i64) {
    (
        (pt.x() / grid).round() as i64,
        (pt.y() / grid).round() as i64,
        (pt.z() / grid).round() as i64,
    )
}
```

Then modify Phase 3 (lines 2617-2642):

Change the `point_to_global` type from `HashMap<(u64, u64, u64), u32>` to `HashMap<(i64, i64, i64), u32>`.

Change the key computation from:
```rust
let key = (pt.x().to_bits(), pt.y().to_bits(), pt.z().to_bits());
```
to:
```rust
let tol = brepkit_math::tolerance::Tolerance::default();
let key = point_merge_key(pt, tol.linear);
```

Note: `tol` should be created once before the loop, not per-vertex.

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `cargo test -p brepkit-operations -- tessellate`
Expected: All existing tessellation tests pass. The grid merge (1e-7) should be value-compatible with bit-exact for well-behaved geometry.

- [ ] **Step 5: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "refactor(tessellate): switch vertex dedup to tolerance-based grid merge

Replace bit-exact to_bits() vertex deduplication with a 1e-7 spatial
grid (Tolerance::default().linear). This merges vertices from separate
edge entities that trace the same curve — common after boolean ops
where cylinder lateral and cap faces have independent edges for the
same circle arc."
```

---

### Task 2: CDT collinear vertex splitting

**Files:**
- Modify: `crates/math/src/cdt.rs:292-357` (`insert_constraint`)

When CDT inserts a constraint edge, intermediate vertices may lie exactly on the constraint line. The current `recover_edge` can't flip through these — it exhausts iterations and silently fails. We need `insert_constraint` to detect collinear vertices and recursively split the constraint through them.

- [ ] **Step 1: Write the failing test**

Add a test in `cdt.rs` that inserts points collinear with a constraint:

```rust
#[test]
fn insert_constraint_through_collinear_vertices() {
    let bounds = (Point2::new(-1.0, -1.0), Point2::new(11.0, 2.0));
    let mut cdt = Cdt::with_capacity(bounds, 10);
    // Insert a row of points along y=0.5
    let v0 = cdt.insert_point(Point2::new(0.0, 0.5)).expect("v0");
    let v1 = cdt.insert_point(Point2::new(2.0, 0.5)).expect("v1");
    let v2 = cdt.insert_point(Point2::new(5.0, 0.5)).expect("v2");
    let v3 = cdt.insert_point(Point2::new(8.0, 0.5)).expect("v3");
    let v4 = cdt.insert_point(Point2::new(10.0, 0.5)).expect("v4");
    // Insert some off-line points so Delaunay doesn't trivially produce the edge
    let _ = cdt.insert_point(Point2::new(3.0, 1.5)).expect("off1");
    let _ = cdt.insert_point(Point2::new(6.0, -0.5)).expect("off2");
    // Constraint from v0 to v4 must split through v1, v2, v3
    cdt.insert_constraint(v0, v4).expect("constraint should succeed");
    let tris = cdt.triangles();
    // Check that constraint edges exist: v0-v1, v1-v2, v2-v3, v3-v4
    let has_edge = |a: usize, b: usize| -> bool {
        tris.iter().any(|&(i, j, k)| {
            (i == a && j == b) || (j == a && k == b) || (k == a && i == b)
            || (i == b && j == a) || (j == b && k == a) || (k == b && i == a)
        })
    };
    assert!(has_edge(v0, v1), "missing edge v0-v1");
    assert!(has_edge(v1, v2), "missing edge v1-v2");
    assert!(has_edge(v2, v3), "missing edge v2-v3");
    assert!(has_edge(v3, v4), "missing edge v3-v4");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brepkit-math -- cdt::tests::insert_constraint_through_collinear_vertices`
Expected: FAIL — `recover_edge` fails or constraint edges missing

- [ ] **Step 3: Implement collinear splitting in `insert_constraint`**

In `cdt.rs`, modify `insert_constraint` (after the duplicate/existing checks, before `recover_edge`). Insert collinear detection between lines 299 and 354:

```rust
// Check for existing vertices that lie on the segment v0→v1.
// If found, split the constraint into sub-segments through them.
let p0 = self.vertices[v0];
let p1 = self.vertices[v1];
let seg_len_sq = (p1.x() - p0.x()) * (p1.x() - p0.x())
    + (p1.y() - p0.y()) * (p1.y() - p0.y());

if seg_len_sq > 1e-20 {
    let mut collinear: Vec<(usize, f64)> = Vec::new();
    let sc = self.super_count;
    for (vi, &pt) in self.vertices.iter().enumerate() {
        if vi == v0 || vi == v1 || vi < sc {
            continue;
        }
        let dx = p1.x() - p0.x();
        let dy = p1.y() - p0.y();
        let t = ((pt.x() - p0.x()) * dx + (pt.y() - p0.y()) * dy) / seg_len_sq;
        if t > 1e-6 && t < 1.0 - 1e-6 {
            let proj_x = p0.x() + t * dx;
            let proj_y = p0.y() + t * dy;
            let dist_sq = (pt.x() - proj_x) * (pt.x() - proj_x)
                + (pt.y() - proj_y) * (pt.y() - proj_y);
            if dist_sq < 1e-12 * seg_len_sq {
                collinear.push((vi, t));
            }
        }
    }

    if !collinear.is_empty() {
        collinear.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut prev = v0;
        for &(vi, _) in &collinear {
            self.insert_constraint(prev, vi)?;
            prev = vi;
        }
        self.insert_constraint(prev, v1)?;
        return Ok(());
    }
}
```

Note: No `eprintln!` debug output — clean as we go.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p brepkit-math -- cdt::tests::insert_constraint_through_collinear_vertices`
Expected: PASS

- [ ] **Step 5: Run all CDT tests**

Run: `cargo test -p brepkit-math -- cdt::tests`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/math/src/cdt.rs
git commit -m "fix(math): CDT insert_constraint splits through collinear vertices

When a constraint edge has intermediate vertices lying exactly on its
line, recover_edge cannot flip through them. Detect collinear vertices
by projecting each CDT vertex onto the constraint segment, then
recursively split the constraint into sub-segments. This fixes CDT
failures for analytic face tessellation where seam/arc boundary
vertices are collinear."
```

---

## Chunk 2: Phase 3b Edge Refinement

---

### Task 3: Circle edge curve refinement

**Files:**
- Modify: `crates/operations/src/tessellate.rs` (add Phase 3b between Phase 3 and Phase 4)

After boolean ops, a cylinder's full-circle edge and a cap's short-arc edges trace the same geometric circle but have different vertex sets. Phase 3b projects ALL global vertices onto each circle edge and inserts those that lie on the curve, so all edges tracing the same circle get the union of their vertex sets.

- [ ] **Step 1: Add `circle_param_range` helper**

Add before `tessellate_solid`:

```rust
/// Returns the parameter range `(t_start, t_end)` for a circle edge.
///
/// For closed edges (same start/end vertex): `(0, 2π)`.
/// For open edges: projects start/end vertices onto the circle.
fn circle_param_range(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    circle: &brepkit_math::curves::Circle,
) -> Result<(f64, f64), crate::OperationsError> {
    let start_pt = topo.vertex(edge.start())?.point();
    let end_pt = topo.vertex(edge.end())?.point();
    if edge.start() == edge.end() {
        return Ok((0.0, std::f64::consts::TAU));
    }
    let t0 = circle.project(start_pt);
    let mut t1 = circle.project(end_pt);
    if t1 <= t0 {
        t1 += std::f64::consts::TAU;
    }
    Ok((t0, t1))
}
```

- [ ] **Step 2: Add Phase 3b edge refinement block**

Insert after the Phase 3 `edge_global_indices` loop (after line 2642 on main) and before Phase 4 (line 2644 on main). The block iterates circle edges, projects all global vertices onto each circle, and inserts those that lie within tolerance and parameter range:

```rust
// Phase 3b: Edge curve refinement.
//
// Boolean operations produce solids where adjacent faces have separate
// edge entities for the same geometric curve. These edges are sampled
// independently with different vertex sets. Project ALL global vertices
// onto each circle edge and insert those on-curve to unify vertex sets.
{
    use brepkit_topology::edge::EdgeCurve;
    let tol = brepkit_math::tolerance::Tolerance::default();

    for &edge_idx in &edge_indices {
        let edge_id = match topo.edges.id_from_index(edge_idx) {
            Some(id) => id,
            None => continue,
        };
        let edge_data = match topo.edge(edge_id) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if let EdgeCurve::Circle(circle) = edge_data.curve() {
            let (t_start, t_end) = match circle_param_range(topo, edge_data, circle) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let gids = match edge_global_indices.get(&edge_idx) {
                Some(g) => g,
                None => continue,
            };
            let existing: std::collections::HashSet<u32> = gids.iter().copied().collect();

            let mut insertions: Vec<(f64, u32)> = Vec::new();
            #[allow(clippy::cast_possible_truncation)]
            for (vi, &pos) in merged.positions.iter().enumerate() {
                let gid = vi as u32;
                if existing.contains(&gid) {
                    continue;
                }
                let t = circle.project(pos);
                let proj = circle.evaluate(t);
                let dist = (pos - proj).length();
                if dist > tol.linear * 10.0 {
                    continue;
                }
                // Normalize t into edge's parameter range.
                let mut t_adj = t;
                if t_end > t_start {
                    while t_adj < t_start - 1e-10 {
                        t_adj += std::f64::consts::TAU;
                    }
                    while t_adj > t_end + std::f64::consts::TAU - 1e-10 {
                        t_adj -= std::f64::consts::TAU;
                    }
                }
                if t_adj >= t_start - 1e-10 && t_adj <= t_end + 1e-10 {
                    let range = t_end - t_start;
                    let frac = (t_adj - t_start) / range;
                    if frac > 0.001 && frac < 0.999 {
                        insertions.push((t_adj, gid));
                    }
                }
            }

            if !insertions.is_empty() {
                insertions.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                // Deduplicate by parameter proximity.
                let mut deduped: Vec<(f64, u32)> = Vec::new();
                for (t, gid) in insertions {
                    if deduped.last().map_or(true, |&(lt, _)| (t - lt).abs() > 1e-8) {
                        deduped.push((t, gid));
                    }
                }

                let old_gids = edge_global_indices.remove(&edge_idx).unwrap_or_default();
                let n = old_gids.len();
                let old_params: Vec<f64> = old_gids
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        t_start + (t_end - t_start) * (i as f64) / ((n - 1).max(1) as f64)
                    })
                    .collect();

                let mut all: Vec<(f64, u32)> = old_params.into_iter().zip(old_gids).collect();
                all.extend(deduped);
                all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut seen = std::collections::HashSet::new();
                let merged_gids: Vec<u32> = all
                    .into_iter()
                    .filter(|&(_, gid)| seen.insert(gid))
                    .map(|(_, gid)| gid)
                    .collect();

                edge_global_indices.insert(edge_idx, merged_gids);
            }
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p brepkit-operations -- tessellate`
Expected: All existing tests pass. The cylinder watertight test may still fail (CDT dispatch not yet changed).

- [ ] **Step 4: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "feat(tessellate): Phase 3b circle edge refinement

Project all global vertices onto each circle edge; insert those that
lie on-curve within tolerance. This ensures co-located circle edges
from boolean results share the same vertex set, which is required for
CDT boundary matching on analytic faces."
```

---

## Chunk 3: CDT for Analytic Faces + Seam Handling

---

### Task 4: Route cylinder/cone/sphere/torus through CDT with fallback

**Files:**
- Modify: `crates/operations/src/tessellate.rs:3034-3092` (face dispatch)

Currently, cylinder/cone faces go directly to snap-based stitching (lines 3034-3049). Sphere/torus already try CDT with rollback. Change all four analytic surface types to use the CDT path with rollback fallback.

- [ ] **Step 1: Modify the face dispatch**

Replace lines 3034-3049 (the cylinder/cone snap-only branch) to merge it with the sphere/torus CDT-with-rollback branch. The result should be a single branch for all four analytic types:

```rust
} else {
    // For all analytic faces (cylinder, cone, sphere, torus):
    // use CDT-based tessellation with exact boundary constraints.
    // Falls back to snap-based stitching if CDT fails or produces
    // zero triangles.
    let pos_save = merged.positions.len();
    let nrm_save = merged.normals.len();
    let idx_save = merged.indices.len();
    let ptg_count_save = point_to_global.len();

    let cdt_ok = tessellate_nonplanar_cdt(
        topo,
        face_id,
        face_data,
        deflection,
        edge_global_indices,
        merged,
        point_to_global,
    );
    let cdt_produced_tris = cdt_ok.is_ok() && merged.indices.len() > idx_save;
    if !cdt_produced_tris {
        merged.positions.truncate(pos_save);
        merged.normals.truncate(nrm_save);
        merged.indices.truncate(idx_save);
        if point_to_global.len() > ptg_count_save {
            point_to_global.retain(|_, v| (*v as usize) < pos_save);
        }

        tessellate_nonplanar_snap(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        )?;
    }
}
```

This removes the separate cylinder/cone snap-only path and the separate sphere/torus CDT path, replacing both with a unified CDT-with-fallback for all non-NURBS non-planar faces.

- [ ] **Step 2: Run tests**

Run: `cargo test -p brepkit-operations -- tessellate`
Expected: All existing tests pass (CDT may fail for full-revolution faces but falls back to snap).

- [ ] **Step 3: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "refactor(tessellate): route all analytic faces through CDT with fallback

Cylinder and cone faces now attempt CDT tessellation (like sphere and
torus) instead of going directly to snap-based stitching. Falls back
to snap if CDT fails or produces zero triangles. This is a prerequisite
for watertight analytic face tessellation."
```

---

### Task 5: Seam UV handling — direct UV assignment for full-revolution faces

**Files:**
- Modify: `crates/operations/src/tessellate.rs` (inside `tessellate_nonplanar_cdt`)

This is the key fix. Full-revolution cylinder/cone faces have a degenerate seam: the same edge appears twice in the wire (forward + backward). Instead of projecting seam points to UV (which creates a diagonal) and nudging (which breaks collinear detection), assign UV coordinates directly:

- Forward seam: all points get `u = u_max` (right edge of UV rectangle)
- Backward seam: all points get `u = u_min` (left edge of UV rectangle)
- v coordinate: determined empirically from 3D→UV projection of run endpoints

This turns the degenerate slit into a proper rectangular UV boundary.

**Prerequisite:** The `boundary_3d` tuple must be extended to carry the `is_forward` flag from the wire traversal. Change the type from `Vec<(Point3, u32, EdgeId)>` to `Vec<(Point3, u32, EdgeId, bool)>` where the fourth element is `oe.is_forward()`. Update all push sites in the boundary collection loop to pass this flag through.

- [ ] **Step 1: Extend `boundary_3d` to carry `is_forward` flag**

In `tessellate_nonplanar_cdt`, find all `boundary_3d.push(...)` calls and change them:

```rust
// Before: boundary_3d.push((pt, gid, edge_id_local));
// After:  boundary_3d.push((pt, gid, edge_id_local, is_fwd));
```

Where `is_fwd` comes from `oe.is_forward()` captured at the start of the wire traversal loop. Update destructuring patterns throughout the function (e.g., `(pt, gid, edge_id, _)` where the flag isn't needed).

- [ ] **Step 2: Add seam detection and direct UV assignment**

In `tessellate_nonplanar_cdt`, after projecting boundary points to UV (the `boundary_uv` computation) and after the existing seam unwrapping code, add seam handling.

First, detect seam edges (edges appearing twice in the wire):

```rust
// Detect degenerate seam edges for full-revolution faces.
let mut wire_edge_counts: HashMap<usize, usize> = HashMap::new();
for oe in wire.edges() {
    *wire_edge_counts.entry(oe.edge().index()).or_default() += 1;
}
let seam_edge_indices: std::collections::HashSet<usize> = wire_edge_counts
    .iter()
    .filter(|&(_, &c)| c > 1)
    .map(|(&idx, _)| idx)
    .collect();
```

Then, if seam edges exist, directly assign UV for seam-edge boundary points. Use the `is_forward` flag to determine which run gets `u_max` vs `u_min`, and project run endpoints to determine v-direction empirically:

```rust
if !seam_edge_indices.is_empty() {
    // Compute UV bounds from non-seam boundary points.
    let (u_min_bnd, u_max_bnd, v_min_bnd, v_max_bnd) = {
        let non_seam_uvs: Vec<_> = boundary_uv.iter()
            .enumerate()
            .filter(|(i, _)| !seam_edge_indices.contains(&boundary_3d[*i].2.index()))
            .map(|(_, uv)| *uv)
            .collect();
        if non_seam_uvs.is_empty() {
            let u_range: Vec<f64> = boundary_uv.iter().map(|p| p.0).collect();
            let v_range: Vec<f64> = boundary_uv.iter().map(|p| p.1).collect();
            (
                u_range.iter().copied().fold(f64::INFINITY, f64::min),
                u_range.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                v_range.iter().copied().fold(f64::INFINITY, f64::min),
                v_range.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (
                non_seam_uvs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
                non_seam_uvs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max),
                non_seam_uvs.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
                non_seam_uvs.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max),
            )
        }
    };

    // Identify contiguous runs of seam-edge boundary points, tracking
    // whether each run is a forward or backward traversal.
    struct SeamRun {
        indices: Vec<usize>,
        is_forward: bool,
    }
    let mut seam_runs: Vec<SeamRun> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut current_fwd: Option<bool> = None;
    for i in 0..n_boundary {
        let (_, _, edge_id, is_fwd) = boundary_3d[i];
        if seam_edge_indices.contains(&edge_id.index()) {
            current_indices.push(i);
            if current_fwd.is_none() {
                current_fwd = Some(is_fwd);
            }
        } else if !current_indices.is_empty() {
            seam_runs.push(SeamRun {
                indices: std::mem::take(&mut current_indices),
                is_forward: current_fwd.unwrap_or(true),
            });
            current_fwd = None;
        }
    }
    if !current_indices.is_empty() {
        // Check if this run wraps around to the start
        if !seam_runs.is_empty()
            && seam_edge_indices.contains(&boundary_3d[0].2.index())
        {
            current_indices.extend(seam_runs.remove(0).indices);
        }
        seam_runs.push(SeamRun {
            indices: current_indices,
            is_forward: current_fwd.unwrap_or(true),
        });
    }

    // Assign UV for each seam run.
    // Forward traversal → u_max (right edge of UV rectangle).
    // Backward traversal → u_min (left edge of UV rectangle).
    // v-direction determined empirically: project first and last 3D
    // points of the run to get their v parameters, then interpolate
    // in the detected direction (handles flipped surface orientations).
    for run in &seam_runs {
        let u_assign = if run.is_forward { u_max_bnd } else { u_min_bnd };
        let n_pts = run.indices.len();

        // Determine v-direction from 3D endpoint projection.
        let v_first = boundary_uv[run.indices[0]].1;
        let v_last = boundary_uv[*run.indices.last().unwrap_or(&0)].1;
        let (v_start, v_end) = if (v_first - v_min_bnd).abs() < (v_first - v_max_bnd).abs() {
            // First point is closer to v_min → ascending
            (v_min_bnd, v_max_bnd)
        } else {
            // First point is closer to v_max → descending
            (v_max_bnd, v_min_bnd)
        };

        for (k, &i) in run.indices.iter().enumerate() {
            let t = if n_pts > 1 {
                k as f64 / (n_pts - 1) as f64
            } else {
                0.5
            };
            let v = v_start + t * (v_end - v_start);
            boundary_uv[i] = (u_assign, v);
        }
    }
}
```

**Key design decisions addressing review feedback:**
- **`is_forward` flag** (Issue 1): Each seam run uses the `is_forward` flag from the `OrientedEdge` captured during boundary collection, not iteration order. Forward → `u_max`, backward → `u_min`.
- **Empirical v-direction** (Issue 2): The v interpolation direction is determined by projecting the run's first 3D point and checking which v bound it's closest to. This handles flipped surface orientations correctly.

**Important:** This block must go AFTER the existing u-seam unwrapping code (which handles non-degenerate seam crossings for partial-revolution faces) but BEFORE the CDT point insertion.

- [ ] **Step 2: Run the cylinder watertight test**

Run: `cargo test -p brepkit-operations -- tessellate::tests::tessellate_boolean_cut_cylinder_watertight`
Expected: Should now pass (or get closer — may need weld pass for remaining mismatches).

- [ ] **Step 3: Add cone watertight test**

```rust
#[test]
fn tessellate_boolean_cut_cone_watertight() {
    let mut topo = Topology::new();
    let cone = crate::primitives::make_cone(&mut topo, 1.5, 0.5, 4.0)
        .expect("cone");
    let tool = crate::primitives::make_box(&mut topo, 3.0, 3.0, 3.0)
        .expect("box");
    let tool = crate::transform::translate(&mut topo, tool, Vec3::new(-0.5, -0.5, 0.5))
        .expect("translate");
    let cut = crate::boolean::boolean_cut(&mut topo, cone, tool)
        .expect("boolean cut");
    let mesh = super::tessellate_solid(&topo, cut, 0.1)
        .expect("tessellate");
    let boundary = super::boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "cone mesh should be watertight, got {boundary} boundary edges"
    );
}
```

- [ ] **Step 4: Run both tests**

Run: `cargo test -p brepkit-operations -- tessellate::tests::tessellate_boolean_cut_c`
Expected: Both pass (or close — weld may be needed).

- [ ] **Step 5: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "feat(tessellate): direct UV assignment for full-revolution analytic face seams

Full-revolution cylinder/cone faces have a degenerate seam where the
same edge appears twice (forward + backward). Instead of projecting
to UV (creating a diagonal) and nudging (breaking CDT constraints),
assign UV coordinates directly: forward seam → u_max, backward seam
→ u_min, v interpolated linearly. This produces a clean rectangular
UV boundary that CDT handles correctly."
```

---

## Chunk 4: Weld Post-Process + Additional Tests

---

### Task 6: Add `weld_boundary_vertices` post-process

**Files:**
- Modify: `crates/operations/src/tessellate.rs` (add function + call from `tessellate_solid`)

The weld pass is a safety net: after all face tessellations, find remaining boundary vertices that are spatially close and merge them via union-find index rewriting.

- [ ] **Step 1: Add `weld_boundary_vertices` function**

Add after `tessellate_solid` (near `boundary_edge_count`):

```rust
/// Weld nearby boundary vertices to close gaps from non-shared edges.
///
/// After tessellation, boundary edges (edges with only 1 adjacent triangle)
/// may exist where adjacent faces used separate edge entities for the same
/// geometric curve. This function finds pairs of nearby boundary vertices
/// and merges them by rewriting index references.
fn weld_boundary_vertices(mesh: &mut TriangleMesh, deflection: f64) {
    let n_verts = mesh.positions.len();
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return;
    }

    // Build edge → face count and identify boundary vertices.
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3];
        let i1 = mesh.indices[t * 3 + 1];
        let i2 = mesh.indices[t * 3 + 2];
        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_default() += 1;
        }
    }

    let boundary_verts: std::collections::HashSet<u32> = edge_count
        .iter()
        .filter(|&(_, &c)| c == 1)
        .flat_map(|(&(a, b), _)| [a, b])
        .collect();

    if boundary_verts.is_empty() {
        return;
    }

    // Spatial grid for O(V) expected merge.
    let tol = deflection.max(1e-6);
    let cell = tol * 2.0;

    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for &vi in &boundary_verts {
        let p = mesh.positions[vi as usize];
        let key = (
            (p.x() / cell).floor() as i64,
            (p.y() / cell).floor() as i64,
            (p.z() / cell).floor() as i64,
        );
        grid.entry(key).or_default().push(vi);
    }

    // Union-find.
    let mut parent: Vec<u32> = (0..n_verts as u32).collect();

    #[inline]
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    for &vi in &boundary_verts {
        let p = mesh.positions[vi as usize];
        let gx = (p.x() / cell).floor() as i64;
        let gy = (p.y() / cell).floor() as i64;
        let gz = (p.z() / cell).floor() as i64;
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    if let Some(neighbors) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &vj in neighbors {
                            if vj <= vi {
                                continue;
                            }
                            let q = mesh.positions[vj as usize];
                            if (p - q).length() < tol {
                                let ra = find(&mut parent, vi);
                                let rb = find(&mut parent, vj);
                                if ra != rb {
                                    if ra < rb {
                                        parent[rb as usize] = ra;
                                    } else {
                                        parent[ra as usize] = rb;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Compress paths and rewrite indices.
    let mut any_merged = false;
    for idx in &mut mesh.indices {
        let canonical = find(&mut parent, *idx);
        if canonical != *idx {
            *idx = canonical;
            any_merged = true;
        }
    }

    // Remove degenerate triangles.
    if any_merged {
        let mut new_indices = Vec::with_capacity(mesh.indices.len());
        for t in 0..mesh.indices.len() / 3 {
            let i0 = mesh.indices[t * 3];
            let i1 = mesh.indices[t * 3 + 1];
            let i2 = mesh.indices[t * 3 + 2];
            if i0 != i1 && i1 != i2 && i2 != i0 {
                new_indices.push(i0);
                new_indices.push(i1);
                new_indices.push(i2);
            }
        }
        mesh.indices = new_indices;
    }
}
```

- [ ] **Step 2: Call weld from tessellate_solid**

Add the call just before the final `Ok(merged)` in `tessellate_solid`, after Phase 5 (normals):

```rust
// Phase 6: Vertex welding — close remaining boundary gaps from
// independently-sampled edges that trace the same curve.
weld_boundary_vertices(&mut merged, deflection);
```

- [ ] **Step 3: Run all tessellation tests**

Run: `cargo test -p brepkit-operations -- tessellate`
Expected: All pass including cylinder/cone watertight tests.

- [ ] **Step 4: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "feat(tessellate): Phase 6 weld_boundary_vertices post-process

After all face tessellations, find boundary vertices (edges with only
1 adjacent triangle) that are spatially close and merge them via
union-find index rewriting. This is a safety net for remaining boundary
mismatches that grid merge and edge refinement don't catch. Uses a
spatial hash grid for O(V) expected complexity."
```

---

### Task 7: Sphere and torus watertight tests

**Files:**
- Modify: `crates/operations/src/tessellate.rs` (add tests)

- [ ] **Step 1: Add sphere boolean cut watertight test**

```rust
#[test]
fn tessellate_boolean_cut_sphere_watertight() {
    let mut topo = Topology::new();
    let sphere = crate::primitives::make_sphere(&mut topo, 2.0)
        .expect("sphere");
    let tool = crate::primitives::make_box(&mut topo, 3.0, 3.0, 3.0)
        .expect("box");
    let tool = crate::transform::translate(&mut topo, tool, Vec3::new(0.5, 0.5, 0.5))
        .expect("translate");
    let cut = crate::boolean::boolean_cut(&mut topo, sphere, tool)
        .expect("boolean cut");
    let mesh = super::tessellate_solid(&topo, cut, 0.1)
        .expect("tessellate");
    let boundary = super::boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "sphere mesh should be watertight, got {boundary} boundary edges"
    );
}
```

- [ ] **Step 2: Add torus boolean cut watertight test**

```rust
#[test]
fn tessellate_boolean_cut_torus_watertight() {
    let mut topo = Topology::new();
    let torus = crate::primitives::make_torus(&mut topo, 2.0, 0.5)
        .expect("torus");
    let tool = crate::primitives::make_box(&mut topo, 5.0, 5.0, 1.0)
        .expect("box");
    let tool = crate::transform::translate(&mut topo, tool, Vec3::new(-2.5, -2.5, -0.5))
        .expect("translate");
    let cut = crate::boolean::boolean_cut(&mut topo, torus, tool)
        .expect("boolean cut");
    let mesh = super::tessellate_solid(&topo, cut, 0.1)
        .expect("tessellate");
    let boundary = super::boundary_edge_count(&mesh);
    assert_eq!(
        boundary, 0,
        "torus mesh should be watertight, got {boundary} boundary edges"
    );
}
```

- [ ] **Step 3: Run all four watertight tests**

Run: `cargo test -p brepkit-operations -- tessellate::tests::tessellate_boolean_cut`
Expected: All 4 pass (cylinder, cone, sphere, torus).

- [ ] **Step 4: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass. No regressions.

- [ ] **Step 5: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: No warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/operations/src/tessellate.rs
git commit -m "test(tessellate): add sphere/torus boolean cut watertight tests

Verify that post-boolean sphere and torus meshes have zero boundary
edges, matching the cylinder and cone watertight tests. All four
analytic surface types now have watertight tessellation coverage."
```

---

## Execution Notes

### Key risks
1. **Phase 3b performance**: Projecting ALL global vertices onto each circle edge is O(V × E_circle). For large meshes this could be slow. If needed, add a spatial pre-filter using the circle's bounding box to skip vertices far from the circle.
2. **Tolerance propagation**: Using `tol.linear` for the grid means if Tolerance defaults ever change, the merge grid changes too. This is intentional per user's request.
3. **Sphere poles**: At sphere poles, all u values map to the same 3D point. The CDT path may struggle with this UV degeneracy. The fallback-to-snap safety net handles this case — if the sphere watertight test (Task 7) fails, the sphere may need pole-specific UV handling similar to the existing `AnalyticKind::SpherePole` triangle fan approach. Defer to a follow-up if needed.
4. **Grid cell boundaries**: The `point_merge_key` round-to-grid approach can miss vertices that straddle a cell boundary (separated by < 1e-7 but in different cells). The weld post-process (Task 6) catches these stragglers.

### Testing strategy
- TDD: each task writes the test first, verifies failure, implements, verifies pass
- Watertight assertion: `boundary_edge_count(&mesh) == 0`
- Full regression: `cargo test --workspace` at the end of each chunk
- Clippy: `cargo clippy --all-targets -- -D warnings` before final commit

### Branch strategy
- Create `feat/analytic-cdt-watertight` from main (after PRs #205/#206 merge)
- Single PR with all work
- Clean commit history (one per task)
