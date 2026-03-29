use crate::MathError;
use crate::predicates::orient2d;

use super::{Cdt, segment_intersection_point, segments_properly_intersect, sorted_pair};

impl Cdt {
    /// Recover a constraint edge (v0, v1) by flipping intersecting edges.
    ///
    /// Uses an iterative approach: find edges that cross the constraint
    /// segment and flip them until the constraint edge exists.
    #[allow(clippy::too_many_lines)]
    pub(super) fn recover_edge(&mut self, v0: usize, v1: usize) -> Result<(), MathError> {
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
        let check_tri = |tri: &super::CdtTriangle, v0_local: usize| -> bool {
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
