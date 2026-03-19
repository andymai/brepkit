//! Shared helper functions for PaveFiller phases.
//!
//! Extracted from phase_ee, phase_ef, and phase_ve to eliminate
//! duplicated vertex-lookup and pave-insertion logic.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Pave};

/// Find a vertex near the given point among all pave block vertices.
///
/// Iterates every pave block in the arena and checks start/end vertices
/// (resolved through same-domain mapping) against `point`. Returns the
/// first vertex within `tol.linear`.
pub(super) fn find_nearby_pave_vertex(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    tol: Tolerance,
) -> Option<VertexId> {
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved) {
                        if (v.point() - point).length() <= tol.linear {
                            return Some(resolved);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Add a pave to the appropriate pave block of an edge.
///
/// Finds the pave block whose parameter range contains the pave's
/// parameter (with a small guard band) and adds the extra pave to it.
pub(super) fn add_pave_to_edge(arena: &mut GfaArena, edge_id: EdgeId, pave: Pave) {
    if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id) {
        let pb_ids_copy: Vec<_> = pb_ids.clone();
        for pb_id in pb_ids_copy {
            if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                let (start, end) = pb.parameter_range();
                if pave.parameter > start + 1e-10 && pave.parameter < end - 1e-10 {
                    pb.add_extra_pave(pave);
                }
            }
        }
    }
}
