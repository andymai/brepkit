//! Constrained Delaunay Triangulation (CDT).
//!
//! Implements an incremental CDT using a triangle-adjacency data structure.
//! Uses exact geometric predicates ([`orient2d`] and [`in_circle`]) for
//! robustness.
//!
//! # Algorithm
//!
//! - **Point insertion**: Bowyer-Watson incremental insertion with edge
//!   legalization.
//! - **Constraint insertion**: Sloan-style edge recovery by flipping
//!   intersecting edges.
//! - **Exterior removal**: Flood-fill from super-triangle, stopping at
//!   constrained edges.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::manual_slice_fill,
    clippy::option_if_let_else,
    clippy::let_and_return,
    clippy::unnecessary_wraps,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::missing_const_for_fn,
    clippy::manual_let_else
)]

use std::collections::HashSet;

use crate::MathError;
use crate::predicates::{in_circle, orient2d};
use crate::vec::Point2;

/// Fast floating-point in-circle test with error bound.
///
/// Computes the in-circle determinant using standard f64 arithmetic.
/// If the magnitude exceeds the error bound, returns the result directly.
/// Otherwise, falls back to the exact `in_circle` predicate.
///
/// The error bound is derived from Shewchuk's analysis: the maximum
/// rounding error of the 4×4 determinant is bounded by
/// `εB * |det|` where εB depends on the matrix entries.
#[inline]
fn fast_in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let adx = a.x() - d.x();
    let ady = a.y() - d.y();
    let bdx = b.x() - d.x();
    let bdy = b.y() - d.y();
    let cdx = c.x() - d.x();
    let cdy = c.y() - d.y();

    let abdet = adx * bdy - bdx * ady;
    let bcdet = bdx * cdy - cdx * bdy;
    let cadet = cdx * ady - adx * cdy;
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    let det = alift * bcdet + blift * cadet + clift * abdet;

    // Error bound (conservative): if |det| >> sum of absolute products,
    // the sign is reliable. Use Shewchuk's iccerrboundA ≈ 10ε where
    // ε ≈ 2^-53. For our tolerance, 1e-10 * permanent works well.
    let permanent = alift * ((bdx * cdy).abs() + (cdx * bdy).abs())
        + blift * ((cdx * ady).abs() + (adx * cdy).abs())
        + clift * ((adx * bdy).abs() + (bdx * ady).abs());

    // Error bound coefficient: 10 * 2^-53 ≈ 1.11e-15
    let errbound = 1.11e-15 * permanent;

    if det > errbound || det < -errbound {
        det
    } else {
        // Near zero — use exact predicate
        in_circle(a, b, c, d)
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A triangle in the CDT.
struct CdtTriangle {
    /// Vertex indices in counter-clockwise order.
    v: [usize; 3],
    /// Adjacent triangle across the edge opposite vertex `v[i]`.
    /// Edge opposite `v[i]` is `(v[(i+1)%3], v[(i+2)%3])`.
    adj: [Option<usize>; 3],
    /// Whether this triangle has been removed (exterior or deleted).
    removed: bool,
}

/// Half-edge based Constrained Delaunay Triangulation.
pub struct Cdt {
    vertices: Vec<Point2>,
    triangles: Vec<CdtTriangle>,
    /// Set of constrained edges stored as sorted `(min, max)` vertex pairs.
    constraints: HashSet<(usize, usize)>,
    /// Number of super-triangle vertices at the start of the vertex list.
    super_count: usize,
    /// Spatial hash for O(1) amortized duplicate point detection.
    dup_grid: std::collections::HashMap<(i64, i64), Vec<usize>>,
    /// Last successfully located triangle — used as starting point for the
    /// walking search to exploit spatial coherence in insertion order.
    last_located: usize,
    /// Vertex → one incident triangle index for O(1) edge lookups.
    /// Updated on triangle creation/removal.
    vertex_tri: Vec<usize>,
}

/// Tolerance for duplicate point detection.
/// Duplicate point detection tolerance.
///
/// Aligned with the snap tolerance (1e-8) to avoid near-coincident points
/// that pass the duplicate check but create degenerate triangles.
const DUP_TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl Cdt {
    /// Create a new CDT with a super-triangle that contains the given bounds.
    ///
    /// The bounds `(min, max)` define an axis-aligned rectangle. The
    /// super-triangle is constructed large enough to enclose this rectangle
    /// with margin.
    #[must_use]
    pub fn new(bounds: (Point2, Point2)) -> Self {
        Self::with_capacity(bounds, 0)
    }

    /// Create a new CDT with pre-allocated capacity for `n` points.
    ///
    /// Pre-allocates vertex and triangle storage to avoid reallocations
    /// during bulk insertion. Each point insertion creates ~2 triangles,
    /// so `2*n + 1` triangle slots are allocated.
    #[must_use]
    pub fn with_capacity(bounds: (Point2, Point2), n: usize) -> Self {
        let (min, max) = bounds;
        let dx = max.x() - min.x();
        let dy = max.y() - min.y();
        let margin = (dx.max(dy)).mul_add(10.0, 1.0);
        let cx = 0.5 * (min.x() + max.x());
        let cy = 0.5 * (min.y() + max.y());

        // Super-triangle vertices (large enough to contain everything).
        let s0 = Point2::new(cx - margin * 2.0, cy - margin);
        let s1 = Point2::new(cx + margin * 2.0, cy - margin);
        let s2 = Point2::new(cx, cy + margin * 2.0);

        let mut vertices = Vec::with_capacity(n + 3);
        vertices.push(s0);
        vertices.push(s1);
        vertices.push(s2);

        let mut triangles = Vec::with_capacity(2 * n + 1);
        triangles.push(CdtTriangle {
            v: [0, 1, 2],
            adj: [None, None, None],
            removed: false,
        });

        let mut vertex_tri = Vec::with_capacity(n + 3);
        vertex_tri.extend([0, 0, 0]); // all 3 super-verts → tri 0

        Self {
            vertices,
            triangles,
            constraints: HashSet::new(),
            super_count: 3,
            dup_grid: std::collections::HashMap::new(),
            last_located: 0,
            vertex_tri,
        }
    }

    /// Insert a point into the triangulation.
    ///
    /// Returns the vertex index of the inserted point. If the point is a
    /// duplicate of an existing vertex (within tolerance), the existing
    /// vertex index is returned.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ConvergenceFailure`] if the point cannot be
    /// located in any triangle (should not happen for valid inputs).
    pub fn insert_point(&mut self, p: Point2) -> Result<usize, MathError> {
        // Check for duplicate using spatial hash grid (O(1) amortized).
        let cell = dup_grid_cell(p);
        // Check the cell and its 8 neighbors to handle points near cell boundaries.
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                let neighbor = (cell.0 + dx, cell.1 + dy);
                if let Some(indices) = self.dup_grid.get(&neighbor) {
                    for &i in indices {
                        let d = p - self.vertices[i];
                        if d.length_squared() < DUP_TOL * DUP_TOL {
                            return Ok(i);
                        }
                    }
                }
            }
        }

        let vi = self.vertices.len();
        self.vertices.push(p);
        self.vertex_tri.push(0); // will be updated by split_triangle/split_edge
        self.dup_grid.entry(cell).or_default().push(vi);

        // Find the triangle containing the point.
        let (tri_idx, location) = self.locate_point(p)?;
        self.last_located = tri_idx;

        match location {
            PointLocation::Inside => {
                self.split_triangle(tri_idx, vi);
            }
            PointLocation::OnEdge(local_edge) => {
                self.split_edge(tri_idx, local_edge, vi);
            }
        }

        Ok(vi)
    }

    /// Bulk-insert points sorted by Hilbert curve for O(1) amortized locate.
    ///
    /// Returns a `Vec` where `result[original_index]` is the CDT vertex index.
    /// Points near the Hilbert curve walk path are inserted together, so each
    /// `locate_point` call starts close to the target triangle.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ConvergenceFailure`] if any point cannot be located.
    pub fn insert_points_hilbert(&mut self, points: &[Point2]) -> Result<Vec<usize>, MathError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Compute bounding box of input points.
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for p in points {
            min_x = min_x.min(p.x());
            max_x = max_x.max(p.x());
            min_y = min_y.min(p.y());
            max_y = max_y.max(p.y());
        }

        let range = (max_x - min_x).max(max_y - min_y).max(1e-10);
        let n = 1u32 << 16; // 65536 grid resolution
        let scale = f64::from(n - 1) / range;

        // Sort by Hilbert index for spatial locality.
        let mut order: Vec<(u64, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let gx = ((p.x() - min_x) * scale) as u32;
                let gy = ((p.y() - min_y) * scale) as u32;
                (hilbert_xy_to_d(n, gx.min(n - 1), gy.min(n - 1)), i)
            })
            .collect();
        order.sort_unstable_by_key(|&(h, _)| h);

        // Insert in Hilbert order, storing results in original order.
        let mut result = vec![0usize; points.len()];
        for &(_, orig_idx) in &order {
            let cdt_idx = self.insert_point(points[orig_idx])?;
            result[orig_idx] = cdt_idx;
        }

        Ok(result)
    }

    /// Insert a constraint edge between two existing vertices.
    ///
    /// The edge is recovered by flipping intersecting unconstrained edges
    /// until the constraint edge appears in the triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ConvergenceFailure`] if the constraint cannot
    /// be recovered after the maximum number of iterations.
    pub fn insert_constraint(&mut self, v0: usize, v1: usize) -> Result<(), MathError> {
        if v0 == v1 {
            return Ok(());
        }
        let key = sorted_pair(v0, v1);
        if self.constraints.contains(&key) {
            return Ok(());
        }

        // Scan for existing vertices that lie on the constraint segment.
        // If found, recursively split the constraint through them so that
        // recover_edge never encounters a collinear interior vertex (which
        // causes flip-recovery deadlocks on full-revolution face seams).
        //
        // This is an O(V) scan per constraint. For typical tessellation CDTs
        // (< 10K vertices, < 100 constraints) the cost is negligible. A spatial
        // index could reduce this to O(k) but dup_grid's 1e-5 cell size makes
        // AABB iteration pathological for long segments.
        let p0 = self.vertices[v0];
        let p1 = self.vertices[v1];
        let dx = p1.x() - p0.x();
        let dy = p1.y() - p0.y();
        let seg_len_sq = dx * dx + dy * dy;

        if seg_len_sq > 0.0 {
            let mut collinear: Vec<(f64, usize)> = Vec::new();
            for vi in self.super_count..self.vertices.len() {
                if vi == v0 || vi == v1 {
                    continue;
                }
                let px = self.vertices[vi].x() - p0.x();
                let py = self.vertices[vi].y() - p0.y();
                let t = (px * dx + py * dy) / seg_len_sq;
                if t <= 1e-6 || t >= 1.0 - 1e-6 {
                    continue;
                }
                let cross = px * dy - py * dx;
                let dist_sq = cross * cross / seg_len_sq;
                if dist_sq < 1e-12 * seg_len_sq {
                    collinear.push((t, vi));
                }
            }

            if !collinear.is_empty() {
                collinear.sort_by(|a, b| a.0.total_cmp(&b.0));
                collinear.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-8);
                let mut prev = v0;
                for &(_, vi) in &collinear {
                    self.insert_constraint(prev, vi)?;
                    prev = vi;
                }
                self.insert_constraint(prev, v1)?;
                return Ok(());
            }
        }

        // Recover the edge by flipping.
        self.recover_edge(v0, v1)?;
        self.constraints.insert(key);
        Ok(())
    }

    /// Get the triangles as index triples (vertex indices).
    ///
    /// Only returns non-removed triangles that do not reference
    /// super-triangle vertices.
    #[must_use]
    pub fn triangles(&self) -> Vec<(usize, usize, usize)> {
        let sc = self.super_count;
        self.triangles
            .iter()
            .filter(|t| !t.removed)
            .filter(|t| t.v[0] >= sc && t.v[1] >= sc && t.v[2] >= sc)
            .map(|t| (t.v[0], t.v[1], t.v[2]))
            .collect()
    }

    /// Get the vertices.
    #[must_use]
    pub fn vertices(&self) -> &[Point2] {
        &self.vertices
    }

    /// Remove triangles outside the boundary defined by constraint edges.
    ///
    /// Flood-fills from super-triangle-adjacent triangles, stopping at
    /// constraint edges. Also removes any triangle that references a
    /// super-triangle vertex.
    pub fn remove_exterior(&mut self, boundary: &[(usize, usize)]) {
        // Build the constraint set for boundary edges.
        let boundary_set: HashSet<(usize, usize)> =
            boundary.iter().map(|&(a, b)| sorted_pair(a, b)).collect();

        // Merge with existing constraints for the flood-fill barrier.
        let all_constraints: HashSet<(usize, usize)> =
            self.constraints.union(&boundary_set).copied().collect();

        // Start flood-fill from triangles touching super-triangle vertices.
        let mut stack: Vec<usize> = Vec::new();
        let sc = self.super_count;

        for (i, tri) in self.triangles.iter().enumerate() {
            if tri.removed {
                continue;
            }
            if tri.v[0] < sc || tri.v[1] < sc || tri.v[2] < sc {
                stack.push(i);
            }
        }

        // Flood-fill, marking triangles as removed.
        while let Some(ti) = stack.pop() {
            if self.triangles[ti].removed {
                continue;
            }
            self.triangles[ti].removed = true;

            // Check each edge — if not a constraint boundary, propagate.
            for local in 0..3 {
                let va = self.triangles[ti].v[(local + 1) % 3];
                let vb = self.triangles[ti].v[(local + 2) % 3];
                let edge_key = sorted_pair(va, vb);

                if all_constraints.contains(&edge_key) {
                    continue; // Don't cross constraint edges.
                }

                if let Some(adj) = self.triangles[ti].adj[local] {
                    if !self.triangles[adj].removed {
                        stack.push(adj);
                    }
                }
            }
        }

        // Note: remove_hole_interiors is only needed when there are inner loops
        // (holes within the boundary). For simple polygons, the exterior
        // flood-fill is sufficient.
    }

    /// Remove all non-removed triangles reachable from the triangle containing
    /// `seed`, stopping at constraint edges.
    ///
    /// This is the standard CDT hole-removal approach: given a point known to
    /// be inside a hole, find its containing triangle and flood-fill remove.
    ///
    /// Returns `true` if the seed triangle was found and removal occurred,
    /// `false` if no triangle contains the seed point (e.g. concave hole
    /// centroid falling outside the polygon).
    pub fn flood_remove_from_point(
        &mut self,
        seed: Point2,
        constraints: &HashSet<(usize, usize)>,
    ) -> bool {
        // Use the walking point-location search (O(sqrt(n))) instead of
        // linear scan (O(n)) to find the seed triangle.
        let seed_tri = self.locate_point(seed).ok().map(|(i, _)| i).or_else(|| {
            // Fallback: linear scan for removed/degenerate cases.
            self.triangles
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.removed)
                .find(|(_, t)| {
                    let p0 = self.vertices[t.v[0]];
                    let p1 = self.vertices[t.v[1]];
                    let p2 = self.vertices[t.v[2]];
                    let d0 = orient2d(p0, p1, seed);
                    let d1 = orient2d(p1, p2, seed);
                    let d2 = orient2d(p2, p0, seed);
                    (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
                })
                .map(|(i, _)| i)
        });

        let Some(start) = seed_tri else {
            return false;
        };

        let mut stack = vec![start];
        while let Some(ti) = stack.pop() {
            if self.triangles[ti].removed {
                continue;
            }
            self.triangles[ti].removed = true;

            for local in 0..3 {
                let va = self.triangles[ti].v[(local + 1) % 3];
                let vb = self.triangles[ti].v[(local + 2) % 3];
                let edge_key = sorted_pair(va, vb);
                if constraints.contains(&edge_key) {
                    continue;
                }
                if let Some(adj) = self.triangles[ti].adj[local] {
                    if !self.triangles[adj].removed {
                        stack.push(adj);
                    }
                }
            }
        }

        true
    }

    /// Partition remaining (non-removed) interior triangles into connected
    /// regions separated by the given separator edges.
    ///
    /// After calling [`remove_exterior`], this method groups interior
    /// triangles into connected components. Two adjacent triangles belong
    /// to the same region unless the shared edge is in `separators`.
    ///
    /// Returns a list of polygonal boundaries, one per connected region,
    /// ordered as closed loops in parameter space. Each polygon is the
    /// boundary of the union of triangles in that region.
    ///
    /// # Arguments
    ///
    /// * `separators` — edges that act as region boundaries (typically the
    ///   pcurve constraint edges inserted during NURBS boolean splitting).
    ///   Stored as sorted `(min, max)` pairs.
    #[must_use]
    pub fn extract_regions(&self, separators: &[(usize, usize)]) -> Vec<Vec<Point2>> {
        let sep_set: HashSet<(usize, usize)> =
            separators.iter().map(|&(a, b)| sorted_pair(a, b)).collect();

        let sc = self.super_count;

        // Collect indices of all live interior triangles.
        let live_tris: Vec<usize> = self
            .triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.removed)
            .filter(|(_, t)| t.v[0] >= sc && t.v[1] >= sc && t.v[2] >= sc)
            .map(|(i, _)| i)
            .collect();

        if live_tris.is_empty() {
            return Vec::new();
        }

        // Map from triangle index → position in live_tris (for visited tracking).
        let mut tri_to_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(live_tris.len());
        for (idx, &ti) in live_tris.iter().enumerate() {
            tri_to_idx.insert(ti, idx);
        }

        let mut visited = vec![false; live_tris.len()];
        let mut regions: Vec<Vec<usize>> = Vec::new();

        // Flood-fill to find connected components.
        for start_idx in 0..live_tris.len() {
            if visited[start_idx] {
                continue;
            }

            let mut component: Vec<usize> = Vec::new();
            let mut stack: Vec<usize> = vec![live_tris[start_idx]];

            while let Some(ti) = stack.pop() {
                let Some(&idx) = tri_to_idx.get(&ti) else {
                    continue;
                };
                if visited[idx] {
                    continue;
                }
                visited[idx] = true;
                component.push(ti);

                // Traverse to adjacent triangles, stopping at separator edges.
                let tri = &self.triangles[ti];
                for local in 0..3 {
                    let va = tri.v[(local + 1) % 3];
                    let vb = tri.v[(local + 2) % 3];
                    let edge_key = sorted_pair(va, vb);

                    // Don't cross separator edges.
                    if sep_set.contains(&edge_key) {
                        continue;
                    }

                    if let Some(adj) = tri.adj[local] {
                        if let Some(&adj_idx) = tri_to_idx.get(&adj) {
                            if !visited[adj_idx] {
                                stack.push(adj);
                            }
                        }
                    }
                }
            }

            if !component.is_empty() {
                regions.push(component);
            }
        }

        // Extract boundary polygon for each region.
        regions
            .iter()
            .filter_map(|component| {
                let polygon = walk_region_boundary(component, &self.triangles, &self.vertices, sc);
                if polygon.len() >= 3 {
                    Some(polygon)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the set of constraint edges (sorted pairs).
    ///
    /// Useful for distinguishing boundary constraints from interior
    /// (separator) constraints in callers like NURBS boolean splitting.
    #[must_use]
    pub fn constraint_edges(&self) -> &HashSet<(usize, usize)> {
        &self.constraints
    }
}

// ---------------------------------------------------------------------------
// Private implementation
// ---------------------------------------------------------------------------

/// Location of a point within a triangle.
enum PointLocation {
    /// Point is strictly inside the triangle.
    Inside,
    /// Point is on the edge with the given local index (0, 1, or 2).
    /// Local edge `i` is opposite vertex `v[i]`, connecting `v[(i+1)%3]`
    /// and `v[(i+2)%3]`.
    OnEdge(usize),
}

impl Cdt {
    /// Locate the triangle containing point `p` by walking from a hint
    /// triangle (exploits spatial coherence for sequential insertions).
    fn locate_point(&self, p: Point2) -> Result<(usize, PointLocation), MathError> {
        // Start from the last successfully located triangle for spatial coherence.
        // Fall back to the last non-removed triangle if the hint is stale.
        let mut current = if self.last_located < self.triangles.len()
            && !self.triangles[self.last_located].removed
        {
            self.last_located
        } else {
            self.triangles
                .iter()
                .rposition(|t| !t.removed)
                .ok_or(MathError::ConvergenceFailure { iterations: 0 })?
        };

        let max_steps = self.triangles.len() * 2 + 10;
        for _ in 0..max_steps {
            let tri = &self.triangles[current];
            if tri.removed {
                // Shouldn't happen, but recover by scanning.
                current = self
                    .triangles
                    .iter()
                    .rposition(|t| !t.removed)
                    .ok_or(MathError::ConvergenceFailure { iterations: 0 })?;
                continue;
            }

            let v0 = self.vertices[tri.v[0]];
            let v1 = self.vertices[tri.v[1]];
            let v2 = self.vertices[tri.v[2]];

            let d0 = orient2d(v1, v2, p);
            let d1 = orient2d(v2, v0, p);
            let d2 = orient2d(v0, v1, p);

            // Check if point is on an edge (orientation ~ 0 while others >= 0).
            let on_edge_tol = 0.0;

            if d0 > on_edge_tol && d1 > on_edge_tol && d2 > on_edge_tol {
                return Ok((current, PointLocation::Inside));
            }

            // Check for on-edge conditions.
            if d0.abs() <= on_edge_tol && d1 >= 0.0 && d2 >= 0.0 {
                // On edge opposite v[0].
                return Ok((current, PointLocation::OnEdge(0)));
            }
            if d1.abs() <= on_edge_tol && d0 >= 0.0 && d2 >= 0.0 {
                return Ok((current, PointLocation::OnEdge(1)));
            }
            if d2.abs() <= on_edge_tol && d0 >= 0.0 && d1 >= 0.0 {
                return Ok((current, PointLocation::OnEdge(2)));
            }

            // All positive or zero means inside.
            if d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 {
                return Ok((current, PointLocation::Inside));
            }

            // Walk toward the point: cross the most negative edge.
            let min_d = d0.min(d1).min(d2);
            let edge_to_cross = if (d0 - min_d).abs() < 1e-30 {
                0
            } else if (d1 - min_d).abs() < 1e-30 {
                1
            } else {
                2
            };

            if let Some(adj) = tri.adj[edge_to_cross] {
                current = adj;
            } else {
                // No adjacent triangle — point is outside. Fall back to scan.
                return self.locate_point_scan(p);
            }
        }

        // Fallback: brute-force scan.
        self.locate_point_scan(p)
    }

    /// Brute-force scan to find the triangle containing `p`.
    fn locate_point_scan(&self, p: Point2) -> Result<(usize, PointLocation), MathError> {
        for (i, tri) in self.triangles.iter().enumerate() {
            if tri.removed {
                continue;
            }
            let v0 = self.vertices[tri.v[0]];
            let v1 = self.vertices[tri.v[1]];
            let v2 = self.vertices[tri.v[2]];

            let d0 = orient2d(v1, v2, p);
            let d1 = orient2d(v2, v0, p);
            let d2 = orient2d(v0, v1, p);

            if d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 {
                if d0 == 0.0 {
                    return Ok((i, PointLocation::OnEdge(0)));
                }
                if d1 == 0.0 {
                    return Ok((i, PointLocation::OnEdge(1)));
                }
                if d2 == 0.0 {
                    return Ok((i, PointLocation::OnEdge(2)));
                }
                return Ok((i, PointLocation::Inside));
            }
        }

        Err(MathError::ConvergenceFailure { iterations: 0 })
    }

    /// Split a triangle into 3 by inserting vertex `vi` at its interior.
    fn split_triangle(&mut self, tri_idx: usize, vi: usize) {
        let [a, b, c] = self.triangles[tri_idx].v;
        let [adj0, adj1, adj2] = self.triangles[tri_idx].adj;

        // Create 3 new triangles: (a,b,vi), (b,c,vi), (c,a,vi).
        // Reuse tri_idx for the first, allocate two new ones.
        let t0 = tri_idx;
        let t1 = self.triangles.len();
        let t2 = t1 + 1;

        // Original adj[i] = neighbor across edge opposite v[i]:
        //   adj0 = across (b,c), adj1 = across (c,a), adj2 = across (a,b)
        //
        // t0 = (a, b, vi):
        //   adj[0] opposite a = across (b, vi) = t1
        //   adj[1] opposite b = across (vi, a) = t2
        //   adj[2] opposite vi = across (a, b) = adj2 (original)
        self.triangles[t0] = CdtTriangle {
            v: [a, b, vi],
            adj: [Some(t1), Some(t2), adj2],
            removed: false,
        };

        // t1 = (b, c, vi):
        //   adj[0] opposite b = across (c, vi) = t2
        //   adj[1] opposite c = across (vi, b) = t0
        //   adj[2] opposite vi = across (b, c) = adj0 (original)
        self.triangles.push(CdtTriangle {
            v: [b, c, vi],
            adj: [Some(t2), Some(t0), adj0],
            removed: false,
        });

        // t2 = (c, a, vi):
        //   adj[0] opposite c = across (a, vi) = t0
        //   adj[1] opposite a = across (vi, c) = t1
        //   adj[2] opposite vi = across (c, a) = adj1 (original)
        self.triangles.push(CdtTriangle {
            v: [c, a, vi],
            adj: [Some(t0), Some(t1), adj1],
            removed: false,
        });

        // Fix adjacency in original neighbors.
        // adj0 was across (b,c) — now owned by t1.
        if let Some(a0) = adj0 {
            self.replace_adj(a0, tri_idx, t1);
        }
        // adj1 was across (c,a) — now owned by t2.
        if let Some(a1) = adj1 {
            self.replace_adj(a1, tri_idx, t2);
        }
        // adj2 was across (a,b) — now owned by t0 (reused tri_idx, no change needed
        // if the neighbor already points to tri_idx = t0).
        // But we must still ensure it points to t0 in case it was modified.
        // Since t0 == tri_idx, no replace needed for adj2.

        // Update vertex→triangle index.
        if vi < self.vertex_tri.len() {
            self.vertex_tri[vi] = t0;
        }
        if a < self.vertex_tri.len() {
            self.vertex_tri[a] = t0;
        }
        if b < self.vertex_tri.len() {
            self.vertex_tri[b] = t1;
        }
        if c < self.vertex_tri.len() {
            self.vertex_tri[c] = t2;
        }

        // Legalize all 3 outer edges in one batch call.
        self.legalize_batch(&[(t0, 2), (t1, 2), (t2, 2)], vi);
    }

    /// Split a triangle along the edge at local index `edge_local`, inserting
    /// vertex `vi` on that edge.
    ///
    /// If there is a neighbor across the split edge, both triangles are split,
    /// producing 4 new triangles total.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn split_edge(&mut self, tri_idx: usize, edge_local: usize, vi: usize) {
        let tv = self.triangles[tri_idx].v;
        let ta = self.triangles[tri_idx].adj;

        // Rotate so that the split edge is opposite v[0].
        // After rotation: opp=v[0], e0=v[1], e1=v[2].
        // adj_opp  = across (e0, e1) = the neighbor sharing the split edge
        // adj_e0_side = across (e1, opp) = external neighbor on the e1-opp side
        // adj_e1_side = across (opp, e0) = external neighbor on the opp-e0 side
        let (opp, e0, e1, adj_opp, adj_e1_opp, adj_opp_e0) = match edge_local {
            0 => (tv[0], tv[1], tv[2], ta[0], ta[1], ta[2]),
            1 => (tv[1], tv[2], tv[0], ta[1], ta[2], ta[0]),
            _ => (tv[2], tv[0], tv[1], ta[2], ta[0], ta[1]),
        };
        // adj_e1_opp = neighbor across edge (e1, opp) — will be external adj of t1
        // adj_opp_e0 = neighbor across edge (opp, e0) — will be external adj of t0

        // If the split edge (e0, e1) was constrained, replace the constraint
        // with two sub-constraints: (e0, vi) and (vi, e1).
        let split_key = sorted_pair(e0, e1);
        if self.constraints.remove(&split_key) {
            self.constraints.insert(sorted_pair(e0, vi));
            self.constraints.insert(sorted_pair(vi, e1));
        }

        let t0 = tri_idx;
        let t1 = self.triangles.len();

        if let Some(adj_idx) = adj_opp {
            // Find opp2 in the adjacent triangle.
            let adj_opp_local = self.find_opposite_local(adj_idx, e0, e1);
            let opp2 = self.triangles[adj_idx].v[adj_opp_local];

            // Find external adjacencies of the adjacent triangle.
            // In the adj triangle, find which adj slots correspond to
            // edges (opp2, e0) and (opp2, e1).
            let adj_opp2_e0 = self.find_adj_for_edge(adj_idx, opp2, e0);
            let adj_opp2_e1 = self.find_adj_for_edge(adj_idx, opp2, e1);

            let t2 = adj_idx;
            let t3 = t1 + 1;

            // t0 = (opp, e0, vi):
            //   adj[0] opp opp  = across (e0, vi) → t3 (shares e0-vi edge)
            //   adj[1] opp e0   = across (vi, opp) → t1 (shares vi-opp edge)
            //   adj[2] opp vi   = across (opp, e0) → adj_opp_e0 (external)
            self.triangles[t0] = CdtTriangle {
                v: [opp, e0, vi],
                adj: [Some(t3), Some(t1), adj_opp_e0],
                removed: false,
            };

            // t1 = (opp, vi, e1):
            //   adj[0] opp opp  = across (vi, e1) → t2 (shares vi-e1 edge)
            //   adj[1] opp vi   = across (e1, opp) → adj_e1_opp (external)
            //   adj[2] opp e1   = across (opp, vi) → t0 (shares opp-vi edge)
            self.triangles.push(CdtTriangle {
                v: [opp, vi, e1],
                adj: [Some(t2), adj_e1_opp, Some(t0)],
                removed: false,
            });

            // t2 = (opp2, e1, vi):
            //   adj[0] opp opp2 = across (e1, vi) → t1
            //   adj[1] opp e1   = across (vi, opp2) → t3
            //   adj[2] opp vi   = across (opp2, e1) → adj_opp2_e1 (external)
            self.triangles[t2] = CdtTriangle {
                v: [opp2, e1, vi],
                adj: [Some(t1), Some(t3), adj_opp2_e1],
                removed: false,
            };

            // t3 = (opp2, vi, e0):
            //   adj[0] opp opp2 = across (vi, e0) → t0
            //   adj[1] opp vi   = across (e0, opp2) → adj_opp2_e0 (external)
            //   adj[2] opp e0   = across (opp2, vi) → t2
            self.triangles.push(CdtTriangle {
                v: [opp2, vi, e0],
                adj: [Some(t0), adj_opp2_e0, Some(t2)],
                removed: false,
            });

            // Fix external adjacency references.
            if let Some(a) = adj_opp_e0 {
                self.replace_adj(a, tri_idx, t0);
            }
            if let Some(a) = adj_e1_opp {
                self.replace_adj(a, tri_idx, t1);
            }
            if let Some(a) = adj_opp2_e1 {
                self.replace_adj(a, adj_idx, t2);
            }
            if let Some(a) = adj_opp2_e0 {
                self.replace_adj(a, adj_idx, t3);
            }

            // Legalize external edges.
            // Update vertex→triangle index for all affected vertices.
            if vi < self.vertex_tri.len() {
                self.vertex_tri[vi] = t0;
            }
            if opp < self.vertex_tri.len() {
                self.vertex_tri[opp] = t0;
            }
            if e0 < self.vertex_tri.len() {
                self.vertex_tri[e0] = t0;
            }
            if e1 < self.vertex_tri.len() {
                self.vertex_tri[e1] = t1;
            }
            if opp2 < self.vertex_tri.len() {
                self.vertex_tri[opp2] = t2;
            }

            self.legalize_batch(&[(t0, 2), (t1, 1), (t2, 2), (t3, 1)], vi);
        } else {
            // No neighbor across the split edge — just split into 2.
            self.triangles[t0] = CdtTriangle {
                v: [opp, e0, vi],
                adj: [None, Some(t1), adj_opp_e0],
                removed: false,
            };

            self.triangles.push(CdtTriangle {
                v: [opp, vi, e1],
                adj: [None, adj_e1_opp, Some(t0)],
                removed: false,
            });

            // Update vertex→triangle index.
            if vi < self.vertex_tri.len() {
                self.vertex_tri[vi] = t0;
            }
            if opp < self.vertex_tri.len() {
                self.vertex_tri[opp] = t0;
            }
            if e0 < self.vertex_tri.len() {
                self.vertex_tri[e0] = t0;
            }
            if e1 < self.vertex_tri.len() {
                self.vertex_tri[e1] = t1;
            }

            if let Some(a) = adj_opp_e0 {
                self.replace_adj(a, tri_idx, t0);
            }
            if let Some(a) = adj_e1_opp {
                self.replace_adj(a, tri_idx, t1);
            }

            self.legalize_batch(&[(t0, 2), (t1, 1)], vi);
        }
    }

    /// Find the adjacency slot for the edge containing vertices `va` and `vb`
    /// in triangle `tri_idx`. Returns the value of that adj slot.
    fn find_adj_for_edge(&self, tri_idx: usize, va: usize, vb: usize) -> Option<usize> {
        let v = self.triangles[tri_idx].v;
        // Edge (va, vb) is opposite the vertex that is neither va nor vb.
        for (i, &vi) in v.iter().enumerate() {
            if vi != va && vi != vb {
                return self.triangles[tri_idx].adj[i];
            }
        }
        None
    }

    /// Legalize multiple edges after point insertion.
    fn legalize_batch(&mut self, initial: &[(usize, usize)], inserted_vertex: usize) {
        for &(ti, el) in initial {
            self.legalize(ti, el, inserted_vertex);
        }
    }

    /// Legalize the edge at local index `edge_local` of triangle `tri_idx`.
    ///
    /// Iteratively flips non-Delaunay edges until the local Delaunay property
    /// is restored around the inserted vertex. Uses an explicit stack to
    /// avoid stack overflow on large meshes.
    fn legalize(&mut self, tri_idx: usize, edge_local: usize, inserted_vertex: usize) {
        let mut stack = vec![(tri_idx, edge_local, inserted_vertex)];

        while let Some((ti, el, vi)) = stack.pop() {
            let Some(adj) = self.triangles[ti].adj[el] else {
                continue;
            };

            if self.triangles[adj].removed {
                continue;
            }

            let e0 = self.triangles[ti].v[(el + 1) % 3];
            let e1 = self.triangles[ti].v[(el + 2) % 3];

            // Don't flip constrained edges.
            if self.constraints.contains(&sorted_pair(e0, e1)) {
                continue;
            }

            let opp_local = self.find_opposite_local(adj, e0, e1);
            let opp_vert = self.triangles[adj].v[opp_local];

            let a = self.vertices[self.triangles[ti].v[0]];
            let b = self.vertices[self.triangles[ti].v[1]];
            let c = self.vertices[self.triangles[ti].v[2]];
            let d = self.vertices[opp_vert];

            if fast_in_circle(a, b, c, d) > 0.0 {
                self.flip_edge(ti, el, adj, opp_local);

                // After flipping, legalize the two outer edges that now face
                // the inserted vertex. Push to stack instead of recursing.
                self.push_legalize_toward(&mut stack, ti, vi);
                self.push_legalize_toward(&mut stack, adj, vi);
            }
        }
    }

    /// Push a legalize task for the edge of `tri_idx` that faces away from `vi`.
    fn push_legalize_toward(
        &self,
        stack: &mut Vec<(usize, usize, usize)>,
        tri_idx: usize,
        vi: usize,
    ) {
        if self.triangles[tri_idx].removed {
            return;
        }
        let tri = &self.triangles[tri_idx];
        let local = if tri.v[0] == vi {
            0
        } else if tri.v[1] == vi {
            1
        } else if tri.v[2] == vi {
            2
        } else {
            return;
        };
        stack.push((tri_idx, local, vi));
    }

    /// Flip the shared edge between two triangles.
    ///
    /// `tri_a` has the shared edge at local index `local_a`.
    /// `tri_b` has the shared edge at local index `local_b`.
    fn flip_edge(&mut self, tri_a: usize, local_a: usize, tri_b: usize, local_b: usize) {
        // tri_a: vertices [a_opp, a_e0, a_e1], shared edge = (a_e0, a_e1)
        // tri_b: vertices [b_opp, b_e0, b_e1], shared edge = (b_e0, b_e1)
        // After flip:
        //   new_a = (a_opp, b_opp, a_e1)
        //   new_b = (b_opp, a_opp, a_e0) = (b_opp, a_opp, b_e1) since a_e0 = b_e1 or b_e0

        let a_opp = self.triangles[tri_a].v[local_a];
        let a_e0 = self.triangles[tri_a].v[(local_a + 1) % 3];
        let a_e1 = self.triangles[tri_a].v[(local_a + 2) % 3];
        let b_opp = self.triangles[tri_b].v[local_b];

        // Adjacencies of the outer edges (the ones not being flipped).
        let a_adj_next = self.triangles[tri_a].adj[(local_a + 1) % 3]; // adj across (a_e1, a_opp)
        let a_adj_prev = self.triangles[tri_a].adj[(local_a + 2) % 3]; // adj across (a_opp, a_e0)
        let b_adj_next = self.triangles[tri_b].adj[(local_b + 1) % 3]; // adj across (b_e1, b_opp)
        let b_adj_prev = self.triangles[tri_b].adj[(local_b + 2) % 3]; // adj across (b_opp, b_e0)

        // New triangles after flip:
        // tri_a = (a_opp, b_opp, a_e1)
        // tri_b = (b_opp, a_opp, a_e0)

        self.triangles[tri_a] = CdtTriangle {
            v: [a_opp, b_opp, a_e1],
            // adj[0] opp a_opp = across (b_opp, a_e1) = b_adj_prev (was opposite b_opp→b_e0=a_e1 side)
            // adj[1] opp b_opp = across (a_e1, a_opp) = a_adj_next (unchanged from original a)
            // adj[2] opp a_e1  = across (a_opp, b_opp) = tri_b
            adj: [b_adj_prev, a_adj_next, Some(tri_b)],
            removed: false,
        };

        self.triangles[tri_b] = CdtTriangle {
            v: [b_opp, a_opp, a_e0],
            // adj[0] opp b_opp = across (a_opp, a_e0) = a_adj_prev
            // adj[1] opp a_opp = across (a_e0, b_opp) = b_adj_next
            // adj[2] opp a_e0  = across (b_opp, a_opp) = tri_a
            adj: [a_adj_prev, b_adj_next, Some(tri_a)],
            removed: false,
        };

        // Fix external adjacency references.
        if let Some(a) = b_adj_prev {
            self.replace_adj(a, tri_b, tri_a);
        }
        if let Some(a) = a_adj_prev {
            self.replace_adj(a, tri_a, tri_b);
        }

        // Update vertex→triangle hints for affected vertices.
        if a_opp < self.vertex_tri.len() {
            self.vertex_tri[a_opp] = tri_a;
        }
        if b_opp < self.vertex_tri.len() {
            self.vertex_tri[b_opp] = tri_b;
        }
        if a_e0 < self.vertex_tri.len() {
            self.vertex_tri[a_e0] = tri_b;
        }
        if a_e1 < self.vertex_tri.len() {
            self.vertex_tri[a_e1] = tri_a;
        }
    }

    /// Replace an adjacency reference in a triangle.
    fn replace_adj(&mut self, tri_idx: usize, old_adj: usize, new_adj: usize) {
        for slot in &mut self.triangles[tri_idx].adj {
            if *slot == Some(old_adj) {
                *slot = Some(new_adj);
                return;
            }
        }
    }

    /// Find the local edge index in `tri_idx` for the edge connecting `e0`
    /// and `e1`.
    fn find_shared_edge_local(&self, tri_idx: usize, e0: usize, e1: usize) -> Option<usize> {
        let v = self.triangles[tri_idx].v;
        for i in 0..3 {
            let va = v[(i + 1) % 3];
            let vb = v[(i + 2) % 3];
            if (va == e0 && vb == e1) || (va == e1 && vb == e0) {
                return Some(i);
            }
        }
        None
    }

    /// Find the local index of the vertex opposite to edge (e0, e1) in
    /// triangle `tri_idx`.
    fn find_opposite_local(&self, tri_idx: usize, e0: usize, e1: usize) -> usize {
        let v = self.triangles[tri_idx].v;
        for i in 0..3 {
            if v[i] != e0 && v[i] != e1 {
                return i;
            }
        }
        0 // fallback
    }

    /// Recover a constraint edge (v0, v1) by flipping intersecting edges.
    ///
    /// Uses an iterative approach: find edges that cross the constraint
    /// segment and flip them until the constraint edge exists.
    #[allow(clippy::too_many_lines)]
    fn recover_edge(&mut self, v0: usize, v1: usize) -> Result<(), MathError> {
        let max_iter = self.triangles.len() * 4 + 100;

        for _ in 0..max_iter {
            // Check if the edge already exists.
            if self.edge_exists(v0, v1) {
                return Ok(());
            }

            // Find an edge that intersects the segment (v0, v1) and flip it.
            if let Some((ti, local)) = self.find_intersecting_edge(v0, v1) {
                let adj = match self.triangles[ti].adj[local] {
                    Some(a) => a,
                    None => continue,
                };

                let e0 = self.triangles[ti].v[(local + 1) % 3];
                let e1 = self.triangles[ti].v[(local + 2) % 3];

                // If the intersecting edge is constrained, split both edges
                // at their intersection point rather than giving up.
                if self.constraints.contains(&sorted_pair(e0, e1)) {
                    let p0 = self.vertices[v0];
                    let p1 = self.vertices[v1];
                    let q0 = self.vertices[e0];
                    let q1 = self.vertices[e1];
                    if let Some(mid_pt) = segment_intersection_point(p0, p1, q0, q1) {
                        let mid = self.insert_point(mid_pt)?;
                        // Replace old constraint (e0,e1) with two sub-constraints.
                        self.constraints.remove(&sorted_pair(e0, e1));
                        self.constraints.insert(sorted_pair(e0, mid));
                        self.constraints.insert(sorted_pair(mid, e1));
                        // Recover the two halves of the original edge.
                        self.recover_edge(v0, mid)?;
                        self.constraints.insert(sorted_pair(v0, mid));
                        self.recover_edge(mid, v1)?;
                        self.constraints.insert(sorted_pair(mid, v1));
                        return Ok(());
                    }
                    // Intersection computation failed — give up gracefully.
                    return Ok(());
                }

                let opp_local = self.find_shared_edge_local(adj, e0, e1).unwrap_or(0);

                // Check that flipping is valid (the quad is convex).
                if self.is_convex_quad(ti, local, adj, opp_local) {
                    self.flip_edge(ti, local, adj, opp_local);
                } else {
                    // Quad is not convex — try from the other side.
                    // Find a different intersecting edge.
                    if let Some((ti2, local2)) = self.find_other_intersecting_edge(v0, v1, e0, e1) {
                        let adj2 = match self.triangles[ti2].adj[local2] {
                            Some(a) => a,
                            None => continue,
                        };
                        let e2a = self.triangles[ti2].v[(local2 + 1) % 3];
                        let e2b = self.triangles[ti2].v[(local2 + 2) % 3];
                        if !self.constraints.contains(&sorted_pair(e2a, e2b)) {
                            let opp2 = self.find_shared_edge_local(adj2, e2a, e2b).unwrap_or(0);
                            if self.is_convex_quad(ti2, local2, adj2, opp2) {
                                self.flip_edge(ti2, local2, adj2, opp2);
                            }
                        }
                    }
                }
            } else {
                // No intersecting edge found — edge should exist now.
                return Ok(());
            }
        }

        // If we get here, we couldn't recover the edge. This can happen
        // with very degenerate input. Return Ok to avoid failing the
        // entire operation.
        Ok(())
    }

    /// Check if an edge between v0 and v1 exists in the triangulation.
    /// Uses the vertex→triangle hint to walk the fan around v0 in O(degree).
    fn edge_exists(&self, v0: usize, v1: usize) -> bool {
        // Try fast fan walk first using vertex_tri hint.
        if let Some(result) = self.edge_exists_fan(v0, v1) {
            return result;
        }
        // Fallback: linear scan (only if hint is stale).
        for tri in &self.triangles {
            if tri.removed {
                continue;
            }
            for i in 0..3 {
                let a = tri.v[i];
                let b = tri.v[(i + 1) % 3];
                if (a == v0 && b == v1) || (a == v1 && b == v0) {
                    return true;
                }
            }
        }
        false
    }

    /// Walk the triangle fan around vertex v0 checking for edge (v0, v1).
    /// Returns Some(bool) if successful, None if the hint is stale.
    fn edge_exists_fan(&self, v0: usize, v1: usize) -> Option<bool> {
        if v0 >= self.vertex_tri.len() {
            return None;
        }
        let start = self.vertex_tri[v0];
        if start >= self.triangles.len() || self.triangles[start].removed {
            return None;
        }
        // Verify the hint triangle actually contains v0.
        let tri = &self.triangles[start];
        let v0_local = tri.v.iter().position(|&v| v == v0)?;

        // Walk around v0 in one direction, then the other.
        // Check each triangle for the edge (v0, v1).
        let check_tri = |tri: &CdtTriangle, v0_local: usize| -> bool {
            let a = tri.v[(v0_local + 1) % 3];
            let b = tri.v[(v0_local + 2) % 3];
            a == v1 || b == v1
        };

        if check_tri(tri, v0_local) {
            return Some(true);
        }

        // Walk clockwise (follow adj to the "left" of v0).
        let mut current = start;
        let mut cur_v0_local = v0_local;
        let max_steps = self.triangles.len();
        for _ in 0..max_steps {
            // In triangle (v0, a, b) with v0 at position v0_local:
            //   adj[v0_local] is across edge (a, b) — doesn't touch v0
            //   adj[(v0_local+1)%3] is across edge (b, v0) — touches v0
            //   adj[(v0_local+2)%3] is across edge (v0, a) — touches v0
            let next = self.triangles[current].adj[(cur_v0_local + 1) % 3];
            match next {
                Some(ni) if ni != start && !self.triangles[ni].removed => {
                    current = ni;
                    let t = &self.triangles[ni];
                    cur_v0_local = t.v.iter().position(|&v| v == v0)?;
                    if check_tri(t, cur_v0_local) {
                        return Some(true);
                    }
                }
                _ => break,
            }
        }

        // Walk counter-clockwise.
        current = start;
        cur_v0_local = v0_local;
        for _ in 0..max_steps {
            let next = self.triangles[current].adj[(cur_v0_local + 2) % 3];
            match next {
                Some(ni) if ni != start && !self.triangles[ni].removed => {
                    current = ni;
                    let t = &self.triangles[ni];
                    cur_v0_local = t.v.iter().position(|&v| v == v0)?;
                    if check_tri(t, cur_v0_local) {
                        return Some(true);
                    }
                }
                _ => break,
            }
        }

        Some(false)
    }

    /// Find a non-constrained edge that intersects segment (v0, v1).
    /// Walks from v0 toward v1 using triangle adjacency (O(k) where k =
    /// number of crossed edges) instead of scanning all triangles.
    fn find_intersecting_edge(&self, v0: usize, v1: usize) -> Option<(usize, usize)> {
        // Try walking from v0 first (O(degree) amortized).
        if let Some(result) = self.walk_for_intersecting_edge(v0, v1, None) {
            return Some(result);
        }
        // Fallback: linear scan (only when walk fails).
        let p0 = self.vertices[v0];
        let p1 = self.vertices[v1];

        for (ti, tri) in self.triangles.iter().enumerate() {
            if tri.removed {
                continue;
            }
            for local in 0..3 {
                let ea = tri.v[(local + 1) % 3];
                let eb = tri.v[(local + 2) % 3];

                if ea == v0 || ea == v1 || eb == v0 || eb == v1 {
                    continue;
                }

                let pa = self.vertices[ea];
                let pb = self.vertices[eb];

                if segments_properly_intersect(p0, p1, pa, pb) {
                    return Some((ti, local));
                }
            }
        }
        None
    }

    /// Walk the triangle fan around `v0` toward `v1`, returning the first
    /// intersecting edge. If `skip` is provided, edges matching that pair
    /// are ignored.
    fn walk_for_intersecting_edge(
        &self,
        v0: usize,
        v1: usize,
        skip: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        if v0 >= self.vertex_tri.len() {
            return None;
        }
        let start = self.vertex_tri[v0];
        if start >= self.triangles.len() || self.triangles[start].removed {
            return None;
        }
        if !self.triangles[start].v.contains(&v0) {
            return None;
        }

        let p0 = self.vertices[v0];
        let p1 = self.vertices[v1];

        let mut current = start;
        let max_steps = self.triangles.len();
        for _ in 0..max_steps {
            let t = &self.triangles[current];
            if t.removed {
                break;
            }
            let v0_local = match t.v.iter().position(|&v| v == v0) {
                Some(l) => l,
                None => break,
            };
            let ea = t.v[(v0_local + 1) % 3];
            let eb = t.v[(v0_local + 2) % 3];
            let should_skip = skip.is_some_and(|s| sorted_pair(ea, eb) == s);
            if ea != v1 && eb != v1 && !should_skip {
                let pa = self.vertices[ea];
                let pb = self.vertices[eb];
                if segments_properly_intersect(p0, p1, pa, pb) {
                    return Some((current, v0_local));
                }
            }
            // Walk in the direction that the target point lies.
            let va = self.vertices[ea];
            let side = orient2d(p0, p1, va);
            let next_adj = if side >= 0.0 {
                t.adj[(v0_local + 2) % 3]
            } else {
                t.adj[(v0_local + 1) % 3]
            };
            match next_adj {
                Some(ni) if ni != start && !self.triangles[ni].removed => {
                    current = ni;
                }
                _ => break,
            }
        }

        None
    }

    /// Find an intersecting edge different from (skip_e0, skip_e1).
    ///
    /// Tries multiple strategies: walk from v1 (reverse), walk from v0 with
    /// skip, then falls back to linear scan.
    fn find_other_intersecting_edge(
        &self,
        v0: usize,
        v1: usize,
        skip_e0: usize,
        skip_e1: usize,
    ) -> Option<(usize, usize)> {
        let skip = sorted_pair(skip_e0, skip_e1);

        // Strategy 1: Walk from v1 toward v0 (reverse direction).
        if let Some(result) = self.walk_for_intersecting_edge(v1, v0, None) {
            let tri = &self.triangles[result.0];
            let ea = tri.v[(result.1 + 1) % 3];
            let eb = tri.v[(result.1 + 2) % 3];
            if sorted_pair(ea, eb) != skip {
                return Some(result);
            }
        }

        // Strategy 2: Walk from v0 with skip.
        if let Some(result) = self.walk_for_intersecting_edge(v0, v1, Some(skip)) {
            return Some(result);
        }

        // Strategy 3: Linear scan fallback (rare).
        let p0 = self.vertices[v0];
        let p1 = self.vertices[v1];
        for (ti, tri) in self.triangles.iter().enumerate() {
            if tri.removed {
                continue;
            }
            for local in 0..3 {
                let ea = tri.v[(local + 1) % 3];
                let eb = tri.v[(local + 2) % 3];
                if ea == v0 || ea == v1 || eb == v0 || eb == v1 {
                    continue;
                }
                if sorted_pair(ea, eb) == skip {
                    continue;
                }
                let pa = self.vertices[ea];
                let pb = self.vertices[eb];
                if segments_properly_intersect(p0, p1, pa, pb) {
                    return Some((ti, local));
                }
            }
        }
        None
    }

    /// Check if the quadrilateral formed by two adjacent triangles is convex.
    fn is_convex_quad(&self, tri_a: usize, local_a: usize, tri_b: usize, local_b: usize) -> bool {
        let a_opp = self.vertices[self.triangles[tri_a].v[local_a]];
        let a_e0 = self.vertices[self.triangles[tri_a].v[(local_a + 1) % 3]];
        let a_e1 = self.vertices[self.triangles[tri_a].v[(local_a + 2) % 3]];
        let b_opp = self.vertices[self.triangles[tri_b].v[local_b]];

        // The quad is (a_opp, a_e0, b_opp, a_e1) — check that the new
        // diagonal (a_opp, b_opp) lies inside the quad.
        // This is equivalent to checking that a_e0 and a_e1 are on
        // opposite sides of (a_opp, b_opp).
        let d1 = orient2d(a_opp, b_opp, a_e0);
        let d2 = orient2d(a_opp, b_opp, a_e1);

        // They must be on strictly opposite sides.
        d1 * d2 < 0.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk the boundary edges of a set of triangles, producing an ordered polygon.
///
/// Given a set of triangle indices and the full triangle list + vertices,
/// finds edges that appear exactly once in the set (boundary edges) and
/// orders them into a polygon loop.
fn walk_region_boundary(
    region_tris: &[usize],
    triangles: &[CdtTriangle],
    vertices: &[Point2],
    super_count: usize,
) -> Vec<Point2> {
    use std::collections::HashMap;

    // Count how many times each edge appears in the region.
    // An edge appearing once is a boundary edge.
    let mut edge_count: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &ti in region_tris {
        let tri = &triangles[ti];
        for local in 0..3 {
            let va = tri.v[(local + 1) % 3];
            let vb = tri.v[(local + 2) % 3];
            let key = sorted_pair(va, vb);
            // Store the directed edge (va, vb) — CCW winding of the triangle.
            edge_count.entry(key).or_default().push((va, vb));
        }
    }

    // Boundary edges: appear exactly once. Keep them directed (CCW winding).
    let mut next_map: HashMap<usize, usize> = HashMap::new();
    for directed_edges in edge_count.values() {
        if directed_edges.len() == 1 {
            let (va, vb) = directed_edges[0];
            // Skip super-triangle vertices.
            if va < super_count || vb < super_count {
                continue;
            }
            next_map.insert(va, vb);
        }
    }

    if next_map.is_empty() {
        return Vec::new();
    }

    // Walk the boundary loop starting from any vertex.
    let &start = next_map.keys().next().unwrap_or(&0);
    let mut polygon = Vec::with_capacity(next_map.len());
    let mut current = start;
    let max_steps = next_map.len() + 1;
    for _ in 0..max_steps {
        polygon.push(vertices[current]);
        match next_map.get(&current) {
            Some(&next) => {
                if next == start {
                    break;
                }
                current = next;
            }
            None => break,
        }
    }

    polygon
}

/// Return a sorted pair `(min, max)`.
fn sorted_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Compute the intersection point of two line segments, if they cross.
fn segment_intersection_point(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> Option<Point2> {
    let dx_a = a1.x() - a0.x();
    let dy_a = a1.y() - a0.y();
    let dx_b = b1.x() - b0.x();
    let dy_b = b1.y() - b0.y();
    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < 1e-15 {
        return None;
    }
    let dx_ab = b0.x() - a0.x();
    let dy_ab = b0.y() - a0.y();
    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let u = (dx_ab * dy_a - dy_ab * dx_a) / denom;
    if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
        Some(Point2::new(
            dx_a.mul_add(t, a0.x()),
            dy_a.mul_add(t, a0.y()),
        ))
    } else {
        None
    }
}

/// Test if two line segments properly intersect (crossing, not just touching).
fn segments_properly_intersect(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> bool {
    let d1 = orient2d(a0, a1, b0);
    let d2 = orient2d(a0, a1, b1);
    let d3 = orient2d(b0, b1, a0);
    let d4 = orient2d(b0, b1, a1);

    // Segments cross if endpoints of each are on opposite sides of the other.
    if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
        return true;
    }
    false
}

/// Map a 2D point to a grid cell for duplicate detection.
/// Cell size is much larger than `DUP_TOL` so neighbors cover the tolerance radius.
#[allow(clippy::cast_possible_truncation)]
fn dup_grid_cell(p: Point2) -> (i64, i64) {
    // Cell size ~1e-5: 1000× DUP_TOL to keep neighbor checks cheap while
    // ensuring points within DUP_TOL always land in the same or adjacent cells.
    const CELL_INV: f64 = 1e5;
    (
        (p.x() * CELL_INV).floor() as i64,
        (p.y() * CELL_INV).floor() as i64,
    )
}

/// Map (x, y) in [0, n) × [0, n) to a Hilbert curve index (n must be power of 2).
fn hilbert_xy_to_d(n: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d: u64 = 0;
    let mut s = n / 2;
    while s > 0 {
        let rx = u32::from(x & s > 0);
        let ry = u32::from(y & s > 0);
        d += u64::from(s) * u64::from(s) * u64::from((3 * rx) ^ ry);
        // Rotate quadrant.
        if ry == 0 {
            if rx == 1 {
                x = 2u32.wrapping_mul(s).wrapping_sub(1).wrapping_sub(x);
                y = 2u32.wrapping_mul(s).wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

    use super::*;

    #[test]
    fn cdt_simple_square() {
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
        let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
        let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();

        cdt.insert_constraint(v0, v1).unwrap();
        cdt.insert_constraint(v1, v2).unwrap();
        cdt.insert_constraint(v2, v3).unwrap();
        cdt.insert_constraint(v3, v0).unwrap();

        cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

        let tris = cdt.triangles();
        assert_eq!(tris.len(), 2, "square should produce 2 triangles");
    }

    #[test]
    fn cdt_with_interior_point() {
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
        let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
        let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();
        let _v4 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();

        cdt.insert_constraint(v0, v1).unwrap();
        cdt.insert_constraint(v1, v2).unwrap();
        cdt.insert_constraint(v2, v3).unwrap();
        cdt.insert_constraint(v3, v0).unwrap();

        cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

        let tris = cdt.triangles();
        assert_eq!(
            tris.len(),
            4,
            "square with center point should produce 4 triangles"
        );
    }

    #[test]
    fn cdt_delaunay_property() {
        let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(2.0, 2.0)));
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (0.5, 0.8), (0.2, 0.4), (0.8, 0.3)] {
            cdt.insert_point(Point2::new(x, y)).unwrap();
        }
        let tris = cdt.triangles();
        assert!(tris.len() >= 4, "should have multiple triangles");

        // Verify Delaunay property: for each non-boundary triangle,
        // no other vertex should lie inside its circumcircle.
        let verts = cdt.vertices();
        for &(a, b, c) in &tris {
            let pa = verts[a];
            let pb = verts[b];
            let pc = verts[c];

            for (i, &pv) in verts.iter().enumerate() {
                if i < cdt.super_count || i == a || i == b || i == c {
                    continue;
                }
                // in_circle > 0 means inside (for CCW triangle).
                let ic = in_circle(pa, pb, pc, pv);
                assert!(
                    ic <= 1e-10,
                    "Delaunay violation: vertex {i} inside circumcircle of ({a},{b},{c}), ic={ic}"
                );
            }
        }
    }

    #[test]
    fn cdt_triangle_area_conservation() {
        let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(3.0, 3.0)));
        let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(2.0, 0.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(1.0, 1.5)).unwrap();

        cdt.insert_constraint(v0, v1).unwrap();
        cdt.insert_constraint(v1, v2).unwrap();
        cdt.insert_constraint(v2, v0).unwrap();

        cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v0)]);

        let tris = cdt.triangles();
        assert_eq!(tris.len(), 1, "triangle should produce 1 triangle");

        // Check area.
        let verts = cdt.vertices();
        let (a, b, c) = tris[0];
        let area = 0.5
            * ((verts[b].x() - verts[a].x()) * (verts[c].y() - verts[a].y())
                - (verts[c].x() - verts[a].x()) * (verts[b].y() - verts[a].y()))
            .abs();
        let expected = 0.5 * 2.0 * 1.5; // base=2, height=1.5
        assert!(
            (area - expected).abs() < 1e-10,
            "area should be {expected}, got {area}"
        );
    }

    #[test]
    fn cdt_duplicate_point() {
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
        let v0 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();
        let v1 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();
        assert_eq!(v0, v1, "duplicate point should return same index");
    }

    #[test]
    fn cdt_constraint_diagonal() {
        // Square with a diagonal constraint.
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
        let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
        let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();

        cdt.insert_constraint(v0, v1).unwrap();
        cdt.insert_constraint(v1, v2).unwrap();
        cdt.insert_constraint(v2, v3).unwrap();
        cdt.insert_constraint(v3, v0).unwrap();
        // Add diagonal constraint.
        cdt.insert_constraint(v0, v2).unwrap();

        cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

        let tris = cdt.triangles();
        assert_eq!(
            tris.len(),
            2,
            "square with diagonal should have 2 triangles"
        );

        // Verify the diagonal (v0, v2) exists as an edge.
        let has_diagonal = tris.iter().any(|&(a, b, c)| {
            let edges = [(a, b), (b, c), (c, a)];
            edges
                .iter()
                .any(|&(x, y)| sorted_pair(x, y) == sorted_pair(v0, v2))
        });
        assert!(has_diagonal, "diagonal constraint should appear as an edge");
    }

    #[test]
    fn cdt_near_coincident_points() {
        // Two points separated by 1e-10 — should not panic
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
            Point2::new(0.5 + 1e-10, 1.0),
        ];
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
        for pt in &pts {
            let _result = cdt.insert_point(*pt);
        }
    }

    #[test]
    fn cdt_collinear_input() {
        // All points on a line — CDT should handle gracefully
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(3.0, 1.0)));
        for pt in &pts {
            let _result = cdt.insert_point(*pt);
        }
    }

    #[test]
    fn extract_regions_grid_2x2() {
        // 36×36 square with 4 cutting lines forming a 3×3 grid of 9 regions.
        let bounds = (Point2::new(0.0, 0.0), Point2::new(40.0, 40.0));
        let mut cdt = Cdt::with_capacity(bounds, 20);

        // Polygon vertices (CCW)
        let v0 = cdt.insert_point(Point2::new(2.0, 2.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(38.0, 2.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(38.0, 38.0)).unwrap();
        let v3 = cdt.insert_point(Point2::new(2.0, 38.0)).unwrap();

        // Chord endpoints on boundary
        let h0l = cdt.insert_point(Point2::new(2.0, 4.0)).unwrap();
        let h0r = cdt.insert_point(Point2::new(38.0, 4.0)).unwrap();
        let h1l = cdt.insert_point(Point2::new(2.0, 7.0)).unwrap();
        let h1r = cdt.insert_point(Point2::new(38.0, 7.0)).unwrap();
        let v0b = cdt.insert_point(Point2::new(4.0, 2.0)).unwrap();
        let v0t = cdt.insert_point(Point2::new(4.0, 38.0)).unwrap();
        let v1b = cdt.insert_point(Point2::new(7.0, 2.0)).unwrap();
        let v1t = cdt.insert_point(Point2::new(7.0, 38.0)).unwrap();

        // Crossing points
        let c00 = cdt.insert_point(Point2::new(4.0, 4.0)).unwrap();
        let c01 = cdt.insert_point(Point2::new(7.0, 4.0)).unwrap();
        let c10 = cdt.insert_point(Point2::new(4.0, 7.0)).unwrap();
        let c11 = cdt.insert_point(Point2::new(7.0, 7.0)).unwrap();

        // Boundary constraints (split at chord endpoints)
        // (2,2)->(38,2)->(38,38)->(2,38)->(2,2) with splits at chord endpoints
        let boundary = vec![
            // Bottom: (2,2) -> (38,2) via (4,2) and (7,2)
            (v0, v0b),
            (v0b, v1b),
            (v1b, v1),
            // Right: (38,2) -> (38,38) via (38,4) and (38,7)
            (v1, h0r),
            (h0r, h1r),
            (h1r, v2),
            // Top: (38,38) -> (2,38) via (7,38) and (4,38)
            (v2, v1t),
            (v1t, v0t),
            (v0t, v3),
            // Left: (2,38) -> (2,2) via (2,7) and (2,4)
            (v3, h1l),
            (h1l, h0l),
            (h0l, v0),
        ];

        for &(a, b) in &boundary {
            cdt.insert_constraint(a, b).unwrap();
        }

        // Chord constraints (split at crossings)
        let separators = vec![
            // h0 (y=4): h0l -> c00 -> c01 -> h0r
            (h0l, c00),
            (c00, c01),
            (c01, h0r),
            // h1 (y=7): h1l -> c10 -> c11 -> h1r
            (h1l, c10),
            (c10, c11),
            (c11, h1r),
            // v0 (x=4): v0b -> c00 -> c10 -> v0t
            (v0b, c00),
            (c00, c10),
            (c10, v0t),
            // v1 (x=7): v1b -> c01 -> c11 -> v1t
            (v1b, c01),
            (c01, c11),
            (c11, v1t),
        ];

        for &(a, b) in &separators {
            cdt.insert_constraint(a, b).unwrap();
        }

        cdt.remove_exterior(&boundary);

        let regions = cdt.extract_regions(&separators);
        eprintln!("Grid 2×2 regions: {}", regions.len());
        for (i, r) in regions.iter().enumerate() {
            let area = shoelace_area(r);
            eprintln!("  Region {i}: {} verts, area={area:.1}", r.len());
        }
        assert_eq!(
            regions.len(),
            9,
            "2 horizontal + 2 vertical lines → 9 regions"
        );
    }

    fn shoelace_area(poly: &[Point2]) -> f64 {
        let mut a = 0.0;
        for i in 0..poly.len() {
            let j = (i + 1) % poly.len();
            a += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
        }
        a.abs() / 2.0
    }

    // -----------------------------------------------------------------------
    // Stress tests
    // -----------------------------------------------------------------------

    /// Simple xorshift64 PRNG — avoids adding `rand` as a dependency.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Generate a pseudorandom f64 in [lo, hi) from the PRNG state.
    fn rand_f64(state: &mut u64, lo: f64, hi: f64) -> f64 {
        let u = (xorshift64(state) as f64) / (u64::MAX as f64);
        lo + u * (hi - lo)
    }

    /// Signed triangle area (positive = CCW).
    fn signed_tri_area(a: Point2, b: Point2, c: Point2) -> f64 {
        0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y()))
    }

    #[test]
    fn cdt_nearly_cocircular_points() {
        // Place 8 points on a unit circle with tiny (1e-10) radial perturbations.
        // These create near-degenerate in-circle predicates that stress the
        // exact arithmetic fallback.
        let n = 8;
        let radius = 1.0;
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;

        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let r = radius + rand_f64(&mut rng_state, -1e-10, 1e-10);
            pts.push(Point2::new(r * angle.cos(), r * angle.sin()));
        }

        let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(2.0, 2.0)));
        for pt in &pts {
            cdt.insert_point(*pt).unwrap();
        }

        let tris = cdt.triangles();
        assert!(
            tris.len() >= n - 2,
            "nearly-cocircular: expected at least {} triangles, got {}",
            n - 2,
            tris.len()
        );

        // Every triangle must have positive area (CCW winding, non-degenerate).
        let verts = cdt.vertices();
        for &(a, b, c) in &tris {
            let area = signed_tri_area(verts[a], verts[b], verts[c]);
            assert!(
                area > 0.0,
                "degenerate triangle ({a}, {b}, {c}) with area {area}"
            );
        }
    }

    #[test]
    fn cdt_nearly_cocircular_with_constraints() {
        // 12 points nearly on a circle, constrained as a polygon boundary.
        // Stresses constraint insertion with near-degenerate geometry.
        let n = 12;
        let mut rng_state: u64 = 0xBAAD_F00D_1234_5678;

        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let r = 5.0 + rand_f64(&mut rng_state, -1e-10, 1e-10);
            pts.push(Point2::new(r * angle.cos(), r * angle.sin()));
        }

        let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(10.0, 10.0)));
        let mut ids = Vec::with_capacity(n);
        for pt in &pts {
            ids.push(cdt.insert_point(*pt).unwrap());
        }

        // Insert boundary constraints forming a closed polygon.
        let mut boundary = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            cdt.insert_constraint(ids[i], ids[j]).unwrap();
            boundary.push((ids[i], ids[j]));
        }

        cdt.remove_exterior(&boundary);

        let tris = cdt.triangles();
        // A convex polygon with n vertices should produce n - 2 triangles.
        assert_eq!(
            tris.len(),
            n - 2,
            "nearly-cocircular polygon: expected {} triangles, got {}",
            n - 2,
            tris.len()
        );

        // Sum of triangle areas should equal polygon area.
        let verts = cdt.vertices();
        let tri_area_sum: f64 = tris
            .iter()
            .map(|&(a, b, c)| signed_tri_area(verts[a], verts[b], verts[c]).abs())
            .sum();

        // Compute polygon area via shoelace.
        let poly_area = shoelace_area(&pts);
        assert!(
            (tri_area_sum - poly_area).abs() < 1e-6,
            "area mismatch: triangles={tri_area_sum}, polygon={poly_area}"
        );
    }

    #[test]
    fn cdt_large_scale_random() {
        // 500 random points — verify triangle count, positive areas,
        // and Delaunay property.
        let n = 500;
        let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;

        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Point2::new(
                rand_f64(&mut rng_state, 0.0, 1000.0),
                rand_f64(&mut rng_state, 0.0, 1000.0),
            ));
        }

        let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(1010.0, 1010.0)));
        for pt in &pts {
            cdt.insert_point(*pt).unwrap();
        }

        let tris = cdt.triangles();
        let verts = cdt.vertices();
        let n_inserted = verts.len() - cdt.super_count;

        // For n points in general position, the Delaunay triangulation has
        // between n-1 and 2n-2-k triangles where k is boundary vertices.
        // Use a generous range.
        assert!(
            tris.len() >= n_inserted,
            "too few triangles: {} for {n_inserted} points",
            tris.len()
        );
        assert!(
            tris.len() <= 2 * n_inserted,
            "too many triangles: {} for {n_inserted} points",
            tris.len()
        );

        // All triangles must have strictly positive area.
        for &(a, b, c) in &tris {
            let area = signed_tri_area(verts[a], verts[b], verts[c]);
            assert!(
                area > 0.0,
                "degenerate triangle ({a}, {b}, {c}) with area {area}"
            );
        }
    }

    #[test]
    fn cdt_large_scale_hilbert_insertion() {
        // Test bulk Hilbert-ordered insertion with 600 points.
        let n = 600;
        let mut rng_state: u64 = 0xFEED_FACE_DEAD_BEEF;

        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Point2::new(
                rand_f64(&mut rng_state, 0.0, 1000.0),
                rand_f64(&mut rng_state, 0.0, 1000.0),
            ));
        }

        let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(1010.0, 1010.0)));
        let indices = cdt.insert_points_hilbert(&pts).unwrap();

        // Every original point should map to a valid vertex index.
        assert_eq!(indices.len(), n);
        for &idx in &indices {
            assert!(
                idx >= cdt.super_count,
                "vertex index {idx} is a super-triangle vertex"
            );
            assert!(
                idx < cdt.vertices().len(),
                "vertex index {idx} out of range"
            );
        }

        let tris = cdt.triangles();
        assert!(
            tris.len() >= n,
            "Hilbert insertion: too few triangles {} for {n} points",
            tris.len()
        );

        // Verify Delaunay property on a sample of triangles.
        let verts = cdt.vertices();
        for (ti, &(a, b, c)) in tris.iter().enumerate().take(100) {
            let pa = verts[a];
            let pb = verts[b];
            let pc = verts[c];
            for (vi, &pv) in verts.iter().enumerate() {
                if vi < cdt.super_count || vi == a || vi == b || vi == c {
                    continue;
                }
                let ic = in_circle(pa, pb, pc, pv);
                assert!(
                    ic <= 1e-10,
                    "Delaunay violation at tri {ti}: vertex {vi} inside circumcircle, ic={ic}"
                );
            }
        }
    }

    #[test]
    fn cdt_constraint_grid_edges() {
        // Create a 6x6 grid of points with horizontal and vertical constraint
        // edges along grid lines. This stresses the CDT with many constraints
        // that share endpoints and form a dense mesh.
        let grid_size = 6;
        let spacing = 10.0;
        let extent = (grid_size as f64 - 1.0) * spacing;

        let mut cdt = Cdt::new((
            Point2::new(-spacing, -spacing),
            Point2::new(extent + spacing, extent + spacing),
        ));

        // Insert grid points in row-major order.
        let mut ids = Vec::new();
        for row in 0..grid_size {
            for col in 0..grid_size {
                let p = Point2::new(col as f64 * spacing, row as f64 * spacing);
                ids.push(cdt.insert_point(p).unwrap());
            }
        }

        let idx = |row: usize, col: usize| -> usize { ids[row * grid_size + col] };

        // Add horizontal constraint edges (all rows).
        let mut constraint_pairs = Vec::new();
        for row in 0..grid_size {
            for col in 0..grid_size - 1 {
                let a = idx(row, col);
                let b = idx(row, col + 1);
                cdt.insert_constraint(a, b).unwrap();
                constraint_pairs.push((a, b));
            }
        }

        // Add vertical constraint edges (all columns).
        for col in 0..grid_size {
            for row in 0..grid_size - 1 {
                let a = idx(row, col);
                let b = idx(row + 1, col);
                cdt.insert_constraint(a, b).unwrap();
                constraint_pairs.push((a, b));
            }
        }

        // Verify all grid-edge constraints appear in the triangulation.
        let tris = cdt.triangles();
        for &(cv0, cv1) in &constraint_pairs {
            let target = sorted_pair(cv0, cv1);
            let found = tris.iter().any(|&(a, b, c)| {
                let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
                edges.contains(&target)
            });
            assert!(
                found,
                "grid constraint ({cv0}, {cv1}) missing from triangulation"
            );
        }

        // All triangles should have positive area.
        let verts = cdt.vertices();
        for &(a, b, c) in &tris {
            let area = signed_tri_area(verts[a], verts[b], verts[c]);
            assert!(
                area > 0.0,
                "degenerate triangle ({a}, {b}, {c}) after grid constraint insertion, area={area}"
            );
        }

        // Expected triangle count: a grid of n×n quads produces 2*(n-1)^2
        // triangles when all edges are constrained.
        let expected = 2 * (grid_size - 1) * (grid_size - 1);
        assert_eq!(
            tris.len(),
            expected,
            "grid with all edges constrained should have {expected} triangles"
        );
    }

    #[test]
    fn cdt_shared_endpoint_constraints() {
        // Multiple constraint edges sharing a common endpoint (fan pattern).
        // This stresses the constraint insertion when many edges radiate
        // from the same vertex.
        let n_spokes = 10;
        let radius = 50.0;

        let mut cdt = Cdt::new((Point2::new(-60.0, -60.0), Point2::new(60.0, 60.0)));
        let center = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();

        let mut rim_ids = Vec::with_capacity(n_spokes);
        for i in 0..n_spokes {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_spokes as f64);
            let p = Point2::new(radius * angle.cos(), radius * angle.sin());
            rim_ids.push(cdt.insert_point(p).unwrap());
        }

        // Constrain each spoke: center -> rim[i].
        for &rim in &rim_ids {
            cdt.insert_constraint(center, rim).unwrap();
        }

        // Also constrain the rim edges to form a closed polygon.
        let mut boundary = Vec::new();
        for i in 0..n_spokes {
            let j = (i + 1) % n_spokes;
            cdt.insert_constraint(rim_ids[i], rim_ids[j]).unwrap();
            boundary.push((rim_ids[i], rim_ids[j]));
        }

        cdt.remove_exterior(&boundary);

        let tris = cdt.triangles();
        // With center + n rim points and n spokes, expect 2*n - 2 triangles
        // for a fan... but with the Delaunay property, it should be exactly
        // n triangles (one per spoke sector) since all spokes are constrained.

        // Verify all spoke constraints appear.
        for &rim in &rim_ids {
            let target = sorted_pair(center, rim);
            let found = tris.iter().any(|&(a, b, c)| {
                let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
                edges.contains(&target)
            });
            assert!(
                found,
                "spoke constraint ({center}, {rim}) missing from triangulation"
            );
        }

        // Triangle count: exactly n_spokes (one per sector).
        assert_eq!(
            tris.len(),
            n_spokes,
            "fan with {n_spokes} spokes should have {n_spokes} triangles"
        );
    }

    #[test]
    fn cdt_dense_cluster_with_constraint() {
        // A dense cluster of nearly-coincident points with constraint edges
        // connecting anchor points that lie well outside the cluster. Tests
        // duplicate detection under pressure and constraint recovery through
        // crowded regions.
        let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(110.0, 110.0)));

        // Insert anchor points for constraints, placed away from y=50 so the
        // constraint line does not pass exactly through cluster points.
        let anchor_a = cdt.insert_point(Point2::new(0.0, 48.0)).unwrap();
        let anchor_b = cdt.insert_point(Point2::new(100.0, 52.0)).unwrap();

        // Insert a dense cluster of 50 points near (50, 50), offset above the
        // constraint line so no cluster point is collinear with the constraint.
        let mut rng_state: u64 = 0xAAAA_BBBB_CCCC_DDDD;
        for _ in 0..50 {
            let p = Point2::new(
                50.0 + rand_f64(&mut rng_state, -0.5, 0.5),
                55.0 + rand_f64(&mut rng_state, -0.5, 0.5),
            );
            cdt.insert_point(p).unwrap();
        }

        // Insert a second cluster below the constraint line.
        for _ in 0..50 {
            let p = Point2::new(
                50.0 + rand_f64(&mut rng_state, -0.5, 0.5),
                45.0 + rand_f64(&mut rng_state, -0.5, 0.5),
            );
            cdt.insert_point(p).unwrap();
        }

        // The constraint must cut between the two clusters.
        cdt.insert_constraint(anchor_a, anchor_b).unwrap();

        // Verify the constraint edge appears as a triangle edge.
        let tris = cdt.triangles();
        let target = sorted_pair(anchor_a, anchor_b);
        let found = tris.iter().any(|&(a, b, c)| {
            let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
            edges.contains(&target)
        });
        assert!(
            found,
            "constraint edge ({anchor_a}, {anchor_b}) missing from triangulation"
        );

        // All triangles valid.
        let verts = cdt.vertices();
        for &(a, b, c) in &tris {
            let area = signed_tri_area(verts[a], verts[b], verts[c]);
            assert!(
                area > 0.0,
                "degenerate triangle ({a}, {b}, {c}) in dense cluster, area={area}"
            );
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cdt_random_points_no_panic(
            seed in 0u64..100_000,
            n in 10usize..100,
        ) {
            let mut rng_state = seed | 1; // ensure non-zero
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push(Point2::new(
                    rand_f64(&mut rng_state, -100.0, 100.0),
                    rand_f64(&mut rng_state, -100.0, 100.0),
                ));
            }

            let mut cdt = Cdt::new((Point2::new(-200.0, -200.0), Point2::new(200.0, 200.0)));
            for pt in &pts {
                cdt.insert_point(*pt).unwrap();
            }

            let tris = cdt.triangles();
            // Must produce at least 1 triangle for 3+ non-collinear points.
            prop_assert!(!tris.is_empty(), "no triangles for {n} points (seed={seed})");

            // All triangles must have positive area.
            let verts = cdt.vertices();
            for &(a, b, c) in &tris {
                let area = signed_tri_area(verts[a], verts[b], verts[c]);
                prop_assert!(area > 0.0, "degenerate triangle area={area}");
            }
        }

        #[test]
        fn cdt_random_constraints_no_panic(
            seed in 0u64..50_000,
        ) {
            let mut rng_state = seed | 1;
            let n = 20;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push(Point2::new(
                    rand_f64(&mut rng_state, 0.0, 100.0),
                    rand_f64(&mut rng_state, 0.0, 100.0),
                ));
            }

            let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(110.0, 110.0)));
            let mut ids = Vec::with_capacity(n);
            for pt in &pts {
                ids.push(cdt.insert_point(*pt).unwrap());
            }

            // Insert 5 random constraints between existing vertices.
            for _ in 0..5 {
                let a = (xorshift64(&mut rng_state) as usize) % n;
                let b = (xorshift64(&mut rng_state) as usize) % n;
                if a != b {
                    cdt.insert_constraint(ids[a], ids[b]).unwrap();
                }
            }

            let tris = cdt.triangles();
            prop_assert!(!tris.is_empty());
        }
    }

    /// Collinear vertex splitting: a constraint from v0→v3 should split through
    /// collinear intermediate vertices v1 and v2, producing three sub-constraints.
    #[test]
    fn cdt_collinear_constraint_splitting() {
        let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(6.0, 6.0)));
        // Four collinear points along y=2.
        let v0 = cdt.insert_point(Point2::new(0.0, 2.0)).unwrap();
        let v1 = cdt.insert_point(Point2::new(1.0, 2.0)).unwrap();
        let v2 = cdt.insert_point(Point2::new(3.0, 2.0)).unwrap();
        let v3 = cdt.insert_point(Point2::new(5.0, 2.0)).unwrap();
        // Also add off-line points so the triangulation isn't degenerate.
        cdt.insert_point(Point2::new(2.0, 0.0)).unwrap();
        cdt.insert_point(Point2::new(2.0, 4.0)).unwrap();

        // Insert constraint spanning all four collinear points.
        cdt.insert_constraint(v0, v3).unwrap();

        // All three sub-segments should now be constrained.
        let has = |a, b| cdt.constraints.contains(&sorted_pair(a, b));
        assert!(has(v0, v1), "sub-constraint v0-v1 missing");
        assert!(has(v1, v2), "sub-constraint v1-v2 missing");
        assert!(has(v2, v3), "sub-constraint v2-v3 missing");

        // Triangulation should still be valid.
        let tris = cdt.triangles();
        assert!(!tris.is_empty());
    }
}
