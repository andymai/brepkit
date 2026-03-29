use super::{Cdt, CdtTriangle};

impl Cdt {
    /// Flip the shared edge between two triangles.
    ///
    /// `tri_a` has the shared edge at local index `local_a`.
    /// `tri_b` has the shared edge at local index `local_b`.
    pub(super) fn flip_edge(&mut self, tri_a: usize, local_a: usize, tri_b: usize, local_b: usize) {
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
    pub(super) fn replace_adj(&mut self, tri_idx: usize, old_adj: usize, new_adj: usize) {
        for slot in &mut self.triangles[tri_idx].adj {
            if *slot == Some(old_adj) {
                *slot = Some(new_adj);
                return;
            }
        }
    }

    /// Find the local edge index in `tri_idx` for the edge connecting `e0`
    /// and `e1`.
    pub(super) fn find_shared_edge_local(
        &self,
        tri_idx: usize,
        e0: usize,
        e1: usize,
    ) -> Option<usize> {
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
    pub(super) fn find_opposite_local(&self, tri_idx: usize, e0: usize, e1: usize) -> usize {
        let v = self.triangles[tri_idx].v;
        for i in 0..3 {
            if v[i] != e0 && v[i] != e1 {
                return i;
            }
        }
        0 // fallback
    }
}
