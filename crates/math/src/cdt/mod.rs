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

mod adjacency;
mod constraints;
mod insert;
mod locate;
#[cfg(test)]
mod tests;

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
            locate::PointLocation::Inside => {
                self.split_triangle(tri_idx, vi);
            }
            locate::PointLocation::OnEdge(local_edge) => {
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
    /// After calling [`Cdt::remove_exterior`], this method groups interior
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
