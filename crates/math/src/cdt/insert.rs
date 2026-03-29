use super::{Cdt, CdtTriangle, fast_in_circle, sorted_pair};

impl Cdt {
    /// Split a triangle into 3 by inserting vertex `vi` at its interior.
    pub(super) fn split_triangle(&mut self, tri_idx: usize, vi: usize) {
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
    pub(super) fn split_edge(&mut self, tri_idx: usize, edge_local: usize, vi: usize) {
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
    pub(super) fn find_adj_for_edge(&self, tri_idx: usize, va: usize, vb: usize) -> Option<usize> {
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
}
