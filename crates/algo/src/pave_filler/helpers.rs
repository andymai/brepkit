//! Shared helper functions for PaveFiller phases.
//!
//! Extracted from phase_ee, phase_ef, and phase_ve to eliminate
//! duplicated vertex-lookup and pave-insertion logic.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
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

/// The nearest pave vertex whose tolerance ball, or a new vertex's ball of
/// `reach`, holds `p`. A vertex carried over from an inexact operand covers
/// the fit error of every point computed for its corner, so a crossing or a
/// section end found there is that vertex's.
pub(super) fn pave_vertex_within_tolerance(
    topo: &Topology,
    arena: &GfaArena,
    p: Point3,
    reach: f64,
    tol: Tolerance,
) -> Option<VertexId> {
    let mut best: Option<(f64, VertexId)> = None;
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            let ends = [pb.start.vertex, pb.end.vertex];
            for vid in ends
                .into_iter()
                .chain(pb.extra_paves.iter().map(|pave| pave.vertex))
            {
                let resolved = arena.resolve_vertex(vid);
                let Ok(v) = topo.vertex(resolved) else {
                    continue;
                };
                let ball = v.tolerance().max(reach);
                let d = (v.point() - p).length();
                if ball > tol.linear && d <= ball && best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, resolved));
                }
            }
        }
    }
    best.map(|(_, v)| v)
}

/// The gap below which two computations of one junction read as the same
/// point.
const WELD_BAND: f64 = 1e-5;

/// The vertex where a boundary edge of one face crosses the other face,
/// nearest `p` within ten times its tolerance or the weld band. A section of
/// two faces leaves each one through its boundary, so its end there is that
/// edge's crossing of the other face, however far the section's own end and
/// the crossing drift apart: on operands whose vertices sit off their faces,
/// or where the section's end was clipped against a curved boundary's
/// chords. The vertex's tolerance grows to hold `p`.
pub(super) fn ef_crossing_vertex(
    topo: &mut Topology,
    arena: &GfaArena,
    (fa, fb): (FaceId, FaceId),
    p: Point3,
) -> Option<VertexId> {
    let boundary = |f: FaceId| -> Vec<EdgeId> {
        topo.face(f).ok().map_or_else(Vec::new, |face| {
            std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .filter_map(|w| topo.wire(w).ok())
                .flat_map(|w| {
                    w.edges()
                        .iter()
                        .map(brepkit_topology::wire::OrientedEdge::edge)
                })
                .collect()
        })
    };
    let (edges_a, edges_b) = (boundary(fa), boundary(fb));
    let mut best: Option<(f64, VertexId)> = None;
    for i in &arena.interference.ef {
        let crate::ds::Interference::EF {
            edge,
            face,
            new_vertex: Some(v),
            ..
        } = i
        else {
            continue;
        };
        let across =
            (*face == fb && edges_a.contains(edge)) || (*face == fa && edges_b.contains(edge));
        if !across {
            continue;
        }
        let v = arena.resolve_vertex(*v);
        let Ok(vertex) = topo.vertex(v) else {
            continue;
        };
        let d = (vertex.point() - p).length();
        if d <= (10.0 * vertex.tolerance()).max(WELD_BAND) && best.is_none_or(|(b, _)| d < b) {
            best = Some((d, v));
        }
    }
    let (d, v) = best?;
    if let Ok(vertex) = topo.vertex_mut(v)
        && vertex.tolerance() < d
    {
        vertex.set_tolerance(d);
    }
    Some(v)
}

/// Widened variant of [`find_nearby_pave_vertex`] for tangential contacts.
///
/// A grazing crossing's solved position is only accurate to
/// `sqrt(2 * r * residual)`, so the exact junction vertex can sit microns
/// outside the linear tolerance. This scans every pave-block endpoint within
/// `radius` and returns the nearest candidate that passes `accept` (the
/// caller checks genuine curve/surface incidence, which is what makes the
/// widened radius safe). If the accepted candidates span more than one
/// distinct position (beyond `tol_linear` of each other), the contact is
/// ambiguous — two different junctions inside the window — and `None` is
/// returned so the caller keeps the solved point rather than merging
/// distinct junctions. The spatial index is deliberately not used: its
/// 3x3x3 cell stencil is exhaustive only for radius <= one tolerance cell.
pub(super) fn find_nearby_pave_vertex_widened(
    topo: &Topology,
    arena: &GfaArena,
    point: Point3,
    radius: f64,
    tol_linear: f64,
    accept: impl Fn(Point3) -> bool,
) -> Option<VertexId> {
    let mut best: Option<(f64, Point3, VertexId)> = None;
    let mut ambiguous = false;
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                for vid in [pb.start.vertex, pb.end.vertex] {
                    let resolved = arena.resolve_vertex(vid);
                    if let Ok(v) = topo.vertex(resolved) {
                        let d = (v.point() - point).length();
                        if d <= radius && accept(v.point()) {
                            match &best {
                                Some((bd, bp, _)) => {
                                    if (v.point() - *bp).length() > tol_linear {
                                        ambiguous = true;
                                    }
                                    if d < *bd {
                                        best = Some((d, v.point(), resolved));
                                    }
                                }
                                None => best = Some((d, v.point(), resolved)),
                            }
                        }
                    }
                }
            }
        }
    }
    if ambiguous {
        return None;
    }
    best.map(|(_, _, v)| v)
}

/// Add a pave to the appropriate pave block of an edge.
///
/// Finds the pave block whose parameter range contains the pave's
/// parameter (with a small guard band) and adds the extra pave to it.
pub(super) fn add_pave_to_edge(arena: &mut GfaArena, edge_id: EdgeId, pave: Pave) {
    if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id) {
        let pb_ids_copy: Vec<_> = pb_ids.clone();
        for pb_id in pb_ids_copy {
            if let Some(pb) = arena.pave_blocks.get_mut(pb_id)
                && pb.spans_interior(pave.parameter)
            {
                pb.add_extra_pave(pave);
            }
        }
    }
}
