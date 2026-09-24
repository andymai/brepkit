//! Finalize pave block structure by splitting at extra paves.
//!
//! For each edge's pave blocks, if a pave block has accumulated `extra_paves`
//! from intersection phases (VE, EE, EF), this phase splits it into child
//! pave blocks and updates `edge_pave_blocks` to reference the leaves.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::vertex::Vertex;

use crate::ds::{GfaArena, Pave};
use crate::error::AlgoError;

/// Split all pave blocks at their extra paves.
///
/// After this, each pave block represents a contiguous edge segment
/// with no pending intersection points. The `edge_pave_blocks` map
/// is updated to reference leaf (unsplit) pave blocks only.
///
/// A closed edge paved at exactly one point away from its seam would split
/// into two complementary arcs sharing both endpoint vertices. The assembler
/// keys duplicate edges on their endpoint pair, so the two would collapse
/// into one and every face around the rim would lose its partner (a barrel
/// whose seam vertex lies on a bracket plane that also crosses the rim once
/// more). Both arcs are paved at their middles, so no two pieces of one
/// edge share both endpoints, and neither does a piece and a section chord
/// across the rim (a rod halved by a plane through its seam); the planar
/// face splitter applies the same rule to the rims it re-splits itself.
///
/// # Errors
///
/// Returns [`AlgoError`] if arena operations fail.
#[allow(clippy::unnecessary_wraps)] // Result kept for API consistency with other phases
pub fn perform(topo: &mut Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect edges that need processing (can't iterate and mutate simultaneously)
    let edges: Vec<_> = arena.edge_pave_blocks.keys().copied().collect();

    for edge_id in edges {
        let pb_ids: Vec<_> = arena
            .edge_pave_blocks
            .get(&edge_id)
            .cloned()
            .unwrap_or_default();

        let mut new_pb_ids = Vec::new();
        for pb_id in pb_ids {
            let has_extras = arena
                .pave_blocks
                .get(pb_id)
                .is_some_and(|pb| !pb.extra_paves.is_empty());

            if has_extras {
                // Snapshot all data we need before mutating
                let (original_edge, mut start, mut end, mut extra_paves) = {
                    let pb = match arena.pave_blocks.get(pb_id) {
                        Some(pb) => pb,
                        None => continue,
                    };
                    (pb.original_edge, pb.start, pb.end, pb.extra_paves.clone())
                };
                let closed = arena.resolve_vertex(start.vertex) == arena.resolve_vertex(end.vertex);
                // A closed circle's block spans the curve's own period from
                // its angular origin, but its one vertex sits wherever the
                // seam was placed. Anchor the children at that seam so their
                // vertex order follows the walk around the rim; otherwise a
                // child between the origin and the seam reads backwards and
                // its edge covers the complementary arc.
                if closed && let Some(seam) = closed_seam_parameter(topo, original_edge) {
                    use std::f64::consts::TAU;
                    start.parameter = seam;
                    end.parameter = seam + TAU;
                    for pave in &mut extra_paves {
                        pave.parameter = seam + (pave.parameter - seam).rem_euclid(TAU);
                    }
                }

                let mut sorted_paves = extra_paves;
                sorted_paves.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
                sorted_paves.dedup_by(|a, b| (a.parameter - b.parameter).abs() < 1e-10);
                let interior: Vec<Pave> = sorted_paves
                    .iter()
                    .copied()
                    .filter(|pave| {
                        (pave.parameter - start.parameter).abs() >= 1e-10
                            && (pave.parameter - end.parameter).abs() >= 1e-10
                    })
                    .collect();
                if closed {
                    let mids = midpoint_paves_of_both_arcs(
                        topo,
                        arena,
                        original_edge,
                        start,
                        end,
                        &interior,
                    );
                    if !mids.is_empty() {
                        sorted_paves.extend(mids);
                        sorted_paves.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
                    }
                }

                let mut prev_pave = start;
                let mut children = Vec::new();

                for pave in &sorted_paves {
                    // Skip paves that coincide with the boundaries
                    if (pave.parameter - start.parameter).abs() < 1e-10
                        || (pave.parameter - end.parameter).abs() < 1e-10
                    {
                        continue;
                    }

                    let child = crate::ds::PaveBlock::new(original_edge, prev_pave, *pave);
                    let child_id = arena.pave_blocks.alloc(child);
                    children.push(child_id);
                    prev_pave = *pave;
                }

                let last_child = crate::ds::PaveBlock::new(original_edge, prev_pave, end);
                let last_id = arena.pave_blocks.alloc(last_child);
                children.push(last_id);

                if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                    pb.children.clone_from(&children);
                }

                log::debug!(
                    "MakeBlocks: edge {edge_id:?} pave block {pb_id:?} split into {} children",
                    children.len(),
                );

                new_pb_ids.extend(children);
            } else {
                // No extras — keep the original pave block as a leaf
                new_pb_ids.push(pb_id);
            }
        }

        arena.edge_pave_blocks.insert(edge_id, new_pb_ids);
    }

    Ok(())
}

/// The angular parameter of a closed circle or ellipse edge's seam vertex.
fn closed_seam_parameter(topo: &Topology, edge_id: brepkit_topology::edge::EdgeId) -> Option<f64> {
    let edge = topo.edge(edge_id).ok()?;
    let sp = topo.vertex(edge.start()).ok()?.point();
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Circle(c) => Some(c.project(sp)),
        brepkit_topology::edge::EdgeCurve::Ellipse(e) => Some(e.project(sp)),
        brepkit_topology::edge::EdgeCurve::Line
        | brepkit_topology::edge::EdgeCurve::NurbsCurve(_) => None,
    }
}

/// For a closed edge whose interior paves sit at exactly one position away
/// from the seam vertex: a pave at the middle of the longer of the two arcs
/// that position and the seam bound, with a fresh vertex on the curve.
/// Parameters are the block's own (seam-anchored for circles and ellipses).
fn midpoint_paves_of_both_arcs(
    topo: &mut Topology,
    arena: &GfaArena,
    edge_id: brepkit_topology::edge::EdgeId,
    start: Pave,
    end: Pave,
    interior: &[Pave],
) -> Vec<Pave> {
    let Ok(edge) = topo.edge(edge_id) else {
        return Vec::new();
    };
    let Ok(seam) = topo.vertex(edge.start()) else {
        return Vec::new();
    };
    let (sp, vtol) = (seam.point(), seam.tolerance());
    let Ok(ep) = topo
        .vertex(edge.end())
        .map(brepkit_topology::vertex::Vertex::point)
    else {
        return Vec::new();
    };
    let mut distinct: Option<(Point3, f64)> = None;
    for pave in interior {
        let Ok(p) = topo
            .vertex(arena.resolve_vertex(pave.vertex))
            .map(brepkit_topology::vertex::Vertex::point)
        else {
            return Vec::new();
        };
        if (p - sp).length() <= vtol {
            continue;
        }
        match distinct {
            None => distinct = Some((p, pave.parameter)),
            Some((q, _)) if (q - p).length() <= vtol => {}
            Some(_) => return Vec::new(),
        }
    }
    let Some((p, p_param)) = distinct else {
        return Vec::new();
    };
    let curve = edge.curve().clone();
    let mut mids = Vec::with_capacity(2);
    for t in [
        f64::midpoint(start.parameter, p_param),
        f64::midpoint(p_param, end.parameter),
    ] {
        let point = curve.evaluate_with_endpoints(t, sp, ep);
        if (point - sp).length() <= vtol || (point - p).length() <= vtol {
            return Vec::new();
        }
        let vid = topo.add_vertex(Vertex::new(point, vtol));
        log::debug!("MakeBlocks: closed edge {edge_id:?} paved at an arc's middle t={t:.6}");
        mids.push(Pave::new(vid, t));
    }
    mids
}
