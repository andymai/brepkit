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
/// Returns the resolved (same-domain canonical) vertex first encountered within
/// `tol.linear` of `point`, scanning pave blocks in `edge_pave_blocks` order
/// (ascending `EdgeId`, start-before-end). When the arena's spatial index is
/// available (built after Phase VV) the lookup is O(1) and returns the exact
/// same vertex; otherwise it falls back to the linear scan.
pub(super) fn find_nearby_pave_vertex(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    tol: Tolerance,
) -> Option<VertexId> {
    if let Some(index) = &arena.pave_vertex_index {
        return index.find_within(point, tol.linear);
    }
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    crate::perf::bump_pave_vertex_probe();
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved)
                        && (v.point() - point).length() <= tol.linear
                    {
                        return Some(resolved);
                    }
                }
            }
        }
    }
    None
}

/// Widened variant of [`find_nearby_pave_vertex`] for tangential contacts.
///
/// A grazing crossing's solved position is only accurate to
/// `sqrt(2 * r * residual)`, so the exact junction vertex can sit microns
/// outside the linear tolerance. This scans every pave-block endpoint within
/// `radius` and returns the nearest candidate that passes `accept` (the
/// caller checks genuine curve/surface incidence, which is what makes the
/// widened radius safe). The spatial index is deliberately not used: its
/// 3x3x3 cell stencil is exhaustive only for radius <= one tolerance cell.
pub(super) fn find_nearby_pave_vertex_widened(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    radius: f64,
    accept: impl Fn(Point3) -> bool,
) -> Option<VertexId> {
    let mut best: Option<(f64, VertexId)> = None;
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved) {
                        let d = (v.point() - point).length();
                        if d <= radius && best.is_none_or(|(bd, _)| d < bd) && accept(v.point()) {
                            best = Some((d, resolved));
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, v)| v)
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
