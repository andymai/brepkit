//! Split common vertex — split vertices shared by non-adjacent edges.
//!
//! In a manifold solid, each vertex is typically connected to a modest
//! number of edges (3+). Vertices connected to an unreasonably large
//! number of edges may indicate over-connected topology that should be
//! split into separate vertices. This module detects such vertices and
//! splits them into separate connected groups based on face adjacency.

use std::collections::{HashMap, HashSet, VecDeque};

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::{Vertex, VertexId};

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Maximum number of edges a vertex should be connected to before
/// it is flagged as over-connected.
const MAX_VERTEX_EDGES: usize = 20;

/// Split vertices that are shared by too many non-adjacent edges.
///
/// For each over-connected vertex (more than `MAX_VERTEX_EDGES` edge
/// connections), edges are grouped by face adjacency. If multiple
/// disconnected groups exist, the vertex is duplicated and edges in
/// each group (after the first) are reassigned to a fresh vertex at
/// the same position.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_split_common_vertex(
    topo: &mut Topology,
    solid_id: SolidId,
    ctx: &mut HealContext,
    _config: &FixConfig,
) -> Result<FixResult, HealError> {
    // Walk outer + inner (cavity) shells. Over-connection counting
    // is solid-scoped: a vertex can be over-connected via a mix of
    // outer-shell and inner-shell edges, and outer-shell-only would
    // miss those cases.
    let face_ids = solid_faces(topo, solid_id)?;

    // Count edges per vertex and collect edge-face associations.
    let mut vertex_edge_count: HashMap<usize, (VertexId, usize)> = HashMap::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;

                let start_vid = edge.start();
                vertex_edge_count
                    .entry(start_vid.index())
                    .or_insert_with(|| (start_vid, 0))
                    .1 += 1;

                let end_vid = edge.end();
                vertex_edge_count
                    .entry(end_vid.index())
                    .or_insert_with(|| (end_vid, 0))
                    .1 += 1;
            }
        }
    }

    // Detect over-connected vertices.
    let mut over_connected: Vec<VertexId> = Vec::new();
    for &(vid, count) in vertex_edge_count.values() {
        if count > MAX_VERTEX_EDGES {
            over_connected.push(vid);
        }
    }

    if over_connected.is_empty() {
        return Ok(FixResult::ok());
    }

    let mut total_splits = 0usize;

    for &vertex_id in &over_connected {
        let splits = split_vertex(topo, vertex_id, solid_id, ctx)?;
        total_splits += splits;
    }

    if total_splits == 0 {
        ctx.warn(format!(
            "detected {} over-connected vertices (>{MAX_VERTEX_EDGES} edges) but all are single-group — no splits needed",
            over_connected.len()
        ));
        return Ok(FixResult {
            status: Status::DONE5,
            actions_taken: 0,
        });
    }

    ctx.info(format!(
        "split {total_splits} over-connected vertices into separate groups"
    ));

    Ok(FixResult {
        status: Status::DONE5,
        actions_taken: total_splits,
    })
}

/// Split a single over-connected vertex into separate groups.
///
/// Groups edges by face adjacency: edges that share a face belong to the
/// same group. For each group after the first, a new vertex is created at
/// the same position and all edges in that group are updated.
///
/// Returns the number of new vertices created (0 if only one group).
fn split_vertex(
    topo: &mut Topology,
    vertex_id: VertexId,
    solid_id: SolidId,
    ctx: &mut HealContext,
) -> Result<usize, HealError> {
    // ── Step 1: Find all edges connected to this vertex ──────────
    // Walk outer + inner (cavity) shells (see top-level comment in
    // `fix_split_common_vertex`).
    let face_ids = solid_faces(topo, solid_id)?;

    // Build: edge_id -> set of face_ids that reference it
    // Also: collect edges that touch our vertex
    let mut edge_faces: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut vertex_edges: Vec<EdgeId> = Vec::new();
    let mut vertex_edges_set: HashSet<usize> = HashSet::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                let edge = topo.edge(eid)?;

                // Record face membership for this edge.
                edge_faces
                    .entry(eid.index())
                    .or_default()
                    .insert(fid.index());

                // Track edges touching our vertex.
                if (edge.start() == vertex_id || edge.end() == vertex_id)
                    && vertex_edges_set.insert(eid.index())
                {
                    vertex_edges.push(eid);
                }
            }
        }
    }

    if vertex_edges.len() <= 1 {
        return Ok(0);
    }

    // ── Step 2: Group edges by face adjacency ────────────────────
    // Two edges belong to the same group if they share at least one face.
    // Use union-find via BFS on the edge adjacency graph.
    let n = vertex_edges.len();
    let mut groups: Vec<i32> = vec![-1; n];
    let mut group_id = 0i32;

    for start_idx in 0..n {
        if groups[start_idx] >= 0 {
            continue;
        }
        groups[start_idx] = group_id;
        let mut queue = VecDeque::new();
        queue.push_back(start_idx);

        while let Some(current) = queue.pop_front() {
            let current_eid = vertex_edges[current];
            let current_faces = match edge_faces.get(&current_eid.index()) {
                Some(f) => f,
                None => continue,
            };

            for neighbor_idx in 0..n {
                if groups[neighbor_idx] >= 0 {
                    continue;
                }
                let neighbor_eid = vertex_edges[neighbor_idx];
                let neighbor_faces = match edge_faces.get(&neighbor_eid.index()) {
                    Some(f) => f,
                    None => continue,
                };

                // Check if they share any face.
                let shares_face = current_faces.iter().any(|f| neighbor_faces.contains(f));
                if shares_face {
                    groups[neighbor_idx] = group_id;
                    queue.push_back(neighbor_idx);
                }
            }
        }

        group_id += 1;
    }

    #[allow(clippy::cast_sign_loss)]
    let num_groups = group_id as usize;
    if num_groups <= 1 {
        return Ok(0);
    }

    // ── Step 3: Snapshot vertex data ─────────────────────────────
    let vertex_data = topo.vertex(vertex_id)?;
    let position = vertex_data.point();
    let vtx_tolerance = vertex_data.tolerance();

    // ── Step 4: Allocate new vertices and update edges ───────────
    // Group edges by their group index for batch vertex creation.
    let mut group_edges: HashMap<usize, Vec<(EdgeId, bool, bool)>> = HashMap::new();
    for (i, &eid) in vertex_edges.iter().enumerate() {
        #[allow(clippy::cast_sign_loss)]
        let g = groups[i] as usize;
        if g == 0 {
            continue;
        }
        let edge = topo.edge(eid)?;
        let is_start = edge.start() == vertex_id;
        let is_end = edge.end() == vertex_id;
        group_edges
            .entry(g)
            .or_default()
            .push((eid, is_start, is_end));
    }

    let mut new_vertices_created = 0usize;
    for edges in group_edges.values() {
        // Create a new vertex at the same position.
        let new_vid = topo.add_vertex(Vertex::new(position, vtx_tolerance));

        // Update all edges in this group to use the new vertex.
        for &(eid, update_start, update_end) in edges {
            let edge = topo.edge_mut(eid)?;
            if update_start {
                edge.set_start(new_vid);
            }
            if update_end {
                edge.set_end(new_vid);
            }
        }

        new_vertices_created += 1;
    }

    ctx.info(format!(
        "vertex (index {}) split into {} groups ({} new vertices)",
        vertex_id.index(),
        num_groups,
        new_vertices_created
    ));

    Ok(new_vertices_created)
}
