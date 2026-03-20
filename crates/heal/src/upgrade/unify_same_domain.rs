//! Unify same-domain faces — merge adjacent faces sharing the same
//! underlying surface.
//!
//! This is the most impactful healing operation in production.  After
//! boolean operations, a box may have 72 faces instead of 6 because
//! intersection curves split each original face.  `unify_same_domain`
//! detects adjacent faces on the same plane/cylinder/cone/sphere/torus
//! and merges them back, dramatically reducing face count.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::HealError;
use crate::analysis::surface::surfaces_equivalent;
use crate::status::Status;

/// Options for the unify-same-domain operation.
#[derive(Debug, Clone)]
pub struct UnifyOptions {
    /// Merge co-surface adjacent faces.
    pub unify_faces: bool,
    /// Merge collinear adjacent edges after face merge.
    pub unify_edges: bool,
    /// Linear tolerance for "same position" checks.
    pub linear_tolerance: f64,
    /// Angular tolerance for "same direction" checks.
    pub angular_tolerance: f64,
}

impl Default for UnifyOptions {
    fn default() -> Self {
        Self {
            unify_faces: true,
            unify_edges: true,
            linear_tolerance: 1e-7,
            angular_tolerance: 1e-12,
        }
    }
}

/// Result of the unify operation.
#[derive(Debug, Clone)]
pub struct UnifyResult {
    /// Number of faces that were merged away.
    pub faces_merged: usize,
    /// Number of edges that were merged.
    pub edges_merged: usize,
    /// Status flags.
    pub status: Status,
}

// ── Union-Find ──────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// Merge adjacent faces that share the same underlying surface.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail during merging.
#[allow(clippy::too_many_lines)]
pub fn unify_same_domain(
    topo: &mut Topology,
    solid_id: SolidId,
    options: &UnifyOptions,
) -> Result<(SolidId, UnifyResult), HealError> {
    if !options.unify_faces {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    let solid_data = topo.solid(solid_id)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();
    let n_faces = face_ids.len();

    if n_faces < 2 {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    // 1. Build edge → face adjacency.
    let mut edge_faces: HashMap<usize, Vec<usize>> = HashMap::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fi);
        }
        for &iw_id in face.inner_wires() {
            let iw = topo.wire(iw_id)?;
            for oe in iw.edges() {
                edge_faces.entry(oe.edge().index()).or_default().push(fi);
            }
        }
    }

    // 2. Snapshot face surfaces for comparison.
    let face_surfaces: Vec<_> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| f.surface().clone()))
        .collect::<Result<Vec<_>, _>>()?;

    // 3. Union-find: group adjacent faces with equivalent surfaces.
    let mut uf = UnionFind::new(n_faces);
    let tol = brepkit_math::tolerance::Tolerance {
        linear: options.linear_tolerance,
        angular: options.angular_tolerance,
        relative: 0.0,
    };

    for faces in edge_faces.values() {
        if faces.len() == 2 {
            let fi = faces[0];
            let fj = faces[1];
            if fi != fj && surfaces_equivalent(&face_surfaces[fi], &face_surfaces[fj], &tol) {
                uf.union(fi, fj);
            }
        }
    }

    // 4. Group faces by root.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_faces {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    let merge_groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();

    if merge_groups.is_empty() {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    // 5. Merge each group.
    let mut faces_to_remove: HashSet<FaceId> = HashSet::new();
    let mut new_faces_to_add: Vec<FaceId> = Vec::new();
    let mut total_merged = 0;

    for group in &merge_groups {
        let group_face_ids: Vec<FaceId> = group.iter().map(|&i| face_ids[i]).collect();

        // Count edge usage within this group.
        let mut edge_count: HashMap<usize, usize> = HashMap::new();
        let mut all_oes: Vec<OrientedEdge> = Vec::new();

        for &fid in &group_face_ids {
            let face = topo.face(fid)?;
            let wire = topo.wire(face.outer_wire())?;
            for oe in wire.edges() {
                *edge_count.entry(oe.edge().index()).or_insert(0) += 1;
                all_oes.push(*oe);
            }
            for &iw_id in face.inner_wires() {
                let iw = topo.wire(iw_id)?;
                for oe in iw.edges() {
                    *edge_count.entry(oe.edge().index()).or_insert(0) += 1;
                    all_oes.push(*oe);
                }
            }
        }

        // Boundary edges: appear exactly once in the group.
        let boundary_edges: Vec<OrientedEdge> = all_oes
            .into_iter()
            .filter(|oe| edge_count.get(&oe.edge().index()).copied().unwrap_or(0) == 1)
            .collect();

        if boundary_edges.is_empty() {
            continue;
        }

        // Build merged wire from boundary edges.
        let Ok(merged_wire) = Wire::new(boundary_edges, true) else {
            log::warn!(
                "unify_same_domain: failed to build merged wire for {} faces",
                group.len()
            );
            continue;
        };

        let new_wire_id = topo.add_wire(merged_wire);
        let surface = face_surfaces[group[0]].clone();
        let new_face = brepkit_topology::face::Face::new(new_wire_id, Vec::new(), surface);
        let new_face_id = topo.add_face(new_face);

        for &fid in &group_face_ids {
            faces_to_remove.insert(fid);
        }
        new_faces_to_add.push(new_face_id);
        total_merged += group.len() - 1;
    }

    // 6. Rebuild shell.
    if total_merged > 0 {
        let shell = topo.shell(shell_id)?;
        let current_faces: Vec<FaceId> = shell.faces().to_vec();

        let mut final_faces: Vec<FaceId> = current_faces
            .into_iter()
            .filter(|f| !faces_to_remove.contains(f))
            .collect();
        final_faces.extend(&new_faces_to_add);

        let new_shell = Shell::new(final_faces)?;
        let shell_mut = topo.shell_mut(shell_id)?;
        *shell_mut = new_shell;
    }

    // 7. Merge collinear adjacent edges in the newly created wires.
    let mut total_edges_merged = 0;
    if options.unify_edges && total_merged > 0 {
        for &fid in &new_faces_to_add {
            let outer_wire = topo.face(fid)?.outer_wire();
            let merged = merge_collinear_edges(topo, outer_wire, options.linear_tolerance)?;
            total_edges_merged += merged;
        }
    }

    let status = if total_merged > 0 || total_edges_merged > 0 {
        Status::DONE1
    } else {
        Status::OK
    };

    Ok((
        solid_id,
        UnifyResult {
            faces_merged: total_merged,
            edges_merged: total_edges_merged,
            status,
        },
    ))
}

/// Merge collinear adjacent Line edges in a wire.
///
/// Scans consecutive edge pairs for Line+Line edges on the same line
/// (same direction, shared vertex). When found, creates a single Line
/// edge from the start of the first to the end of the second and
/// replaces both edges.
///
/// Returns the number of edge merges performed.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
///
/// # Future work
///
/// TODO: Support Circle+Circle and NurbsCurve+NurbsCurve merging.
fn merge_collinear_edges(
    topo: &mut Topology,
    wire_id: WireId,
    linear_tolerance: f64,
) -> Result<usize, HealError> {
    // Snapshot edge data for collinearity checks.
    struct EdgeData {
        oe: OrientedEdge,
        start_vid: brepkit_topology::vertex::VertexId,
        end_vid: brepkit_topology::vertex::VertexId,
        start_pos: brepkit_math::vec::Point3,
        end_pos: brepkit_math::vec::Point3,
        is_line: bool,
    }

    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let is_closed = wire.is_closed();
    let n = edges_list.len();

    if n < 2 {
        return Ok(0);
    }

    let mut data = Vec::with_capacity(n);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let is_line = matches!(edge.curve(), EdgeCurve::Line);
        let start_vid = oe.oriented_start(edge);
        let end_vid = oe.oriented_end(edge);
        let start_pos = topo.vertex(start_vid)?.point();
        let end_pos = topo.vertex(end_vid)?.point();
        data.push(EdgeData {
            oe: *oe,
            start_vid,
            end_vid,
            start_pos,
            end_pos,
            is_line,
        });
    }

    // Build the merged edge list by scanning for collinear runs.
    let mut new_edges: Vec<OrientedEdge> = Vec::with_capacity(n);
    let mut merged_count = 0usize;
    let mut i = 0;

    while i < n {
        if !data[i].is_line {
            new_edges.push(data[i].oe);
            i += 1;
            continue;
        }

        // Start of a potential collinear run.
        let run_start_vid = data[i].start_vid;
        let mut run_end_vid = data[i].end_vid;
        let mut _run_end_pos = data[i].end_pos;

        // Compute the direction of the first edge in the run.
        let run_dir = data[i].end_pos - data[i].start_pos;
        let run_len = run_dir.length();
        let run_dir_norm = if run_len > linear_tolerance {
            if let Ok(d) = run_dir.normalize() {
                d
            } else {
                new_edges.push(data[i].oe);
                i += 1;
                continue;
            }
        } else {
            new_edges.push(data[i].oe);
            i += 1;
            continue;
        };

        let mut run_length = 1usize;
        let mut j = i + 1;

        while j < n {
            if !data[j].is_line {
                break;
            }

            // Check shared vertex.
            if data[j].start_vid != run_end_vid {
                break;
            }

            // Check collinearity: same direction.
            let next_dir = data[j].end_pos - data[j].start_pos;
            let next_len = next_dir.length();
            if next_len < linear_tolerance {
                break;
            }
            let next_norm = match next_dir.normalize() {
                Ok(d) => d,
                Err(_) => break,
            };

            let dot = run_dir_norm.dot(next_norm);
            if dot < 1.0 - linear_tolerance {
                break;
            }

            // Collinear — extend the run.
            run_end_vid = data[j].end_vid;
            _run_end_pos = data[j].end_pos;
            run_length += 1;
            j += 1;
        }

        if run_length == 1 {
            // No merge needed.
            new_edges.push(data[i].oe);
            i += 1;
        } else {
            // Create a merged Line edge from run_start to run_end.
            let new_edge = Edge::new(run_start_vid, run_end_vid, EdgeCurve::Line);
            let new_edge_id = topo.add_edge(new_edge);
            new_edges.push(OrientedEdge::new(new_edge_id, true));
            merged_count += run_length - 1;

            i = j;
        }
    }

    if merged_count == 0 {
        return Ok(0);
    }

    // Rebuild wire with merged edges.
    let new_wire = Wire::new(new_edges, is_closed)?;
    let wire_mut = topo.wire_mut(wire_id)?;
    *wire_mut = new_wire;

    Ok(merged_count)
}
