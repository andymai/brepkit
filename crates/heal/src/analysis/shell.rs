//! Shell analysis — manifold check, free edges, orientation consistency.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;

use crate::HealError;
use crate::status::Status;

/// Result of analyzing a shell.
#[derive(Debug, Clone)]
pub struct ShellAnalysis {
    /// Number of faces in the shell.
    pub face_count: usize,
    /// Edges used by exactly one face (free/boundary edges).
    pub boundary_edges: Vec<EdgeId>,
    /// Edges used by three or more faces (non-manifold edges).
    pub non_manifold_edges: Vec<EdgeId>,
    /// Whether all adjacent face pairs agree on edge direction.
    pub orientation_consistent: bool,
    /// Number of connected components (by edge adjacency).
    pub connected_components: usize,
    /// Outcome status flags.
    pub status: Status,
}

/// Edge usage record: which faces reference this edge and with what orientation.
struct EdgeUse {
    edge_id: EdgeId,
    faces: Vec<(FaceId, bool)>, // (face_id, is_forward)
}

/// Analyze a shell for manifoldness, boundary edges, orientation
/// consistency, and connectivity.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::too_many_lines)]
pub fn analyze_shell(topo: &Topology, shell_id: ShellId) -> Result<ShellAnalysis, HealError> {
    let shell = topo.shell(shell_id)?;
    let faces = shell.faces();
    let face_count = faces.len();

    // Build edge-to-face usage map.
    let mut edge_uses: HashMap<usize, EdgeUse> = HashMap::new();

    for &face_id in faces {
        let face = topo.face(face_id)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let idx = oe.edge().index();
                edge_uses
                    .entry(idx)
                    .or_insert_with(|| EdgeUse {
                        edge_id: oe.edge(),
                        faces: Vec::new(),
                    })
                    .faces
                    .push((face_id, oe.is_forward()));
            }
        }
    }

    // Classify edges.
    let mut boundary_edges = Vec::new();
    let mut non_manifold_edges = Vec::new();

    for eu in edge_uses.values() {
        match eu.faces.len() {
            0 => {} // Should not happen.
            1 => boundary_edges.push(eu.edge_id),
            2 => {} // Manifold edge — checked for orientation below.
            _ => non_manifold_edges.push(eu.edge_id),
        }
    }

    // Orientation consistency: for each manifold edge (exactly 2 faces),
    // the two face-uses should have opposite orientations (one forward,
    // one reverse). If both are forward or both reverse, the orientation
    // is inconsistent.
    let mut orientation_consistent = true;
    for eu in edge_uses.values() {
        if eu.faces.len() == 2 {
            let (_, fwd_a) = eu.faces[0];
            let (_, fwd_b) = eu.faces[1];
            // In a consistently oriented shell, adjacent faces traverse the
            // shared edge in opposite directions.
            if fwd_a == fwd_b {
                orientation_consistent = false;
                break;
            }
        }
    }

    // Connected components via union-find on face indices.
    let connected_components = count_connected_components(faces, &edge_uses);

    let mut status = Status::OK;
    if !boundary_edges.is_empty() {
        status = status.merge(Status::DONE1);
    }
    if !non_manifold_edges.is_empty() {
        status = status.merge(Status::DONE2);
    }
    if !orientation_consistent {
        status = status.merge(Status::DONE3);
    }
    if connected_components > 1 {
        status = status.merge(Status::DONE4);
    }

    Ok(ShellAnalysis {
        face_count,
        boundary_edges,
        non_manifold_edges,
        orientation_consistent,
        connected_components,
        status,
    })
}

/// Count connected components among faces using edge adjacency.
fn count_connected_components(faces: &[FaceId], edge_uses: &HashMap<usize, EdgeUse>) -> usize {
    // Find with path compression.
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    // Union.
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    if faces.is_empty() {
        return 0;
    }

    // Map face index (in the shell's face list) to a contiguous 0..N range.
    let mut face_to_idx: HashMap<usize, usize> = HashMap::new();
    for (i, fid) in faces.iter().enumerate() {
        face_to_idx.insert(fid.index(), i);
    }

    let n = faces.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // Connect faces that share an edge.
    for eu in edge_uses.values() {
        if eu.faces.len() >= 2 {
            if let Some(&idx_a) = face_to_idx.get(&eu.faces[0].0.index()) {
                for &(fid, _) in &eu.faces[1..] {
                    if let Some(&idx_b) = face_to_idx.get(&fid.index()) {
                        union(&mut parent, idx_a, idx_b);
                    }
                }
            }
        }
    }

    // Count distinct roots.
    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}
