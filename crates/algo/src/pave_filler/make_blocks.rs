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

/// Two paves of one edge mark the same point: equal parameters, one
/// vertex, or two vertices within the linear tolerance.
///
/// Separate interferences put their own vertices on an edge where it crosses
/// two faces' shared boundary, and their parameters agree only to the curve
/// projection's rounding (5e-14 natively, 1.3e-10 in wasm on a fillet
/// material's corner): a parameter test alone leaves a zero-length block
/// between them on one platform and not the other.
fn same_pave(topo: &Topology, arena: &GfaArena, a: Pave, b: Pave) -> bool {
    if (a.parameter - b.parameter).abs() < 1e-10 {
        return true;
    }
    let (va, vb) = (
        arena.resolve_vertex(a.vertex),
        arena.resolve_vertex(b.vertex),
    );
    va == vb
        || matches!(
            (topo.vertex(va), topo.vertex(vb)),
            (Ok(p), Ok(q)) if (p.point() - q.point()).length()
                <= brepkit_math::tolerance::Tolerance::new().linear
        )
}

/// A block's paves as MakeBlocks splits it at them.
struct BlockPaves {
    start: Pave,
    end: Pave,
    closed: bool,
    descending: bool,
    /// The extra paves in the order the block is walked, one of each group
    /// that coincides.
    paves: Vec<Pave>,
    /// Those of `paves` that are not at either end.
    interior: Vec<Pave>,
}

/// The paves MakeBlocks splits `pb` at: a closed block's parameters wrapped
/// round its seam, sorted along the block, and one kept of each group that
/// coincides (the first along the block).
fn block_paves(topo: &Topology, arena: &GfaArena, pb: &crate::ds::PaveBlock) -> BlockPaves {
    let (mut start, mut end, mut extra_paves) = (pb.start, pb.end, pb.extra_paves.clone());
    let closed = arena.resolve_vertex(start.vertex) == arena.resolve_vertex(end.vertex);
    // A closed circle's block spans the curve's own period from its angular
    // origin, but its one vertex sits wherever the seam was placed. Anchor
    // the children at that seam so their vertex order follows the walk
    // around the rim; otherwise a child between the origin and the seam
    // reads backwards and its edge covers the complementary arc.
    if closed && let Some(seam) = super::helpers::closed_seam_parameter(topo, pb.original_edge) {
        use std::f64::consts::TAU;
        start.parameter = seam;
        end.parameter = seam + TAU;
        for pave in &mut extra_paves {
            pave.parameter = seam + (pave.parameter - seam).rem_euclid(TAU);
        }
    }
    let descending = start.parameter > end.parameter;
    extra_paves.sort_by(|a, b| {
        if descending {
            b.parameter.total_cmp(&a.parameter)
        } else {
            a.parameter.total_cmp(&b.parameter)
        }
    });
    extra_paves.dedup_by(|a, b| same_pave(topo, arena, *a, *b));
    let interior: Vec<Pave> = extra_paves
        .iter()
        .copied()
        .filter(|&pave| !same_pave(topo, arena, pave, start) && !same_pave(topo, arena, pave, end))
        .collect();
    BlockPaves {
        start,
        end,
        closed,
        descending,
        paves: extra_paves,
        interior,
    }
}

/// The vertices a block's pieces end at once MakeBlocks splits it: its two
/// ends, then the extra paves it keeps, in the order it walks them.
pub(super) fn piece_vertices(
    topo: &Topology,
    arena: &GfaArena,
    pb: &crate::ds::PaveBlock,
) -> Vec<brepkit_topology::vertex::VertexId> {
    let paves = block_paves(topo, arena, pb);
    [paves.start.vertex, paves.end.vertex]
        .into_iter()
        .chain(paves.interior.iter().map(|pave| pave.vertex))
        .collect()
}

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
                let Some(pb) = arena.pave_blocks.get(pb_id) else {
                    continue;
                };
                let original_edge = pb.original_edge;
                let BlockPaves {
                    start,
                    end,
                    closed,
                    descending,
                    paves: mut sorted_paves,
                    interior,
                } = block_paves(topo, arena, pb);
                let along = |a: &Pave, b: &Pave| {
                    if descending {
                        b.parameter.total_cmp(&a.parameter)
                    } else {
                        a.parameter.total_cmp(&b.parameter)
                    }
                };
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
                        sorted_paves.sort_by(along);
                    }
                }

                let mut prev_pave = start;
                let mut children = Vec::new();

                for pave in &sorted_paves {
                    // Skip paves that coincide with the boundaries
                    if same_pave(topo, arena, *pave, start) || same_pave(topo, arena, *pave, end) {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::ds::PaveBlock;
    use brepkit_topology::edge::{Edge, EdgeCurve};

    #[test]
    fn two_paves_at_one_point_split_an_edge_once() {
        let mut topo = Topology::new();
        let at = |topo: &mut Topology, x: f64| {
            topo.add_vertex(Vertex::new(Point3::new(x, 0.0, 0.0), 1e-7))
        };
        let (v0, v1) = (at(&mut topo, 0.0), at(&mut topo, 1.0));
        // Two interferences' vertices on one crossing, 1e-9 apart, so their
        // parameters differ by more than 1e-10.
        let (a, b) = (at(&mut topo, 0.5), at(&mut topo, 0.5 + 1e-9));
        let edge = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let mut arena = GfaArena::new();
        let mut pb = PaveBlock::new(edge, Pave::new(v0, 0.0), Pave::new(v1, 1.0));
        pb.extra_paves = vec![Pave::new(a, 0.5), Pave::new(b, 0.5 + 1e-9)];
        let id = arena.pave_blocks.alloc(pb);
        arena.edge_pave_blocks.insert(edge, vec![id]);
        perform(&mut topo, &mut arena).unwrap();
        assert_eq!(arena.pave_blocks.get(id).unwrap().children.len(), 2);
    }

    /// A rim whose vertex is a quarter turn before the origin of its
    /// circle's parameter, as a boolean leaves it. A crossing at the origin
    /// is a point inside the rim's block; a crossing at the vertex is not.
    #[test]
    fn a_rim_takes_a_pave_at_its_circles_origin_and_none_at_its_vertex() {
        use crate::pave_filler::helpers::add_pave_to_edge;
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::{FRAC_PI_2, TAU};

        let mut topo = Topology::new();
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let at =
            |topo: &mut Topology, t: f64| topo.add_vertex(Vertex::new(circle.evaluate(t), 1e-7));
        let seam = at(&mut topo, -FRAC_PI_2);
        let (origin, quarter) = (at(&mut topo, 0.0), at(&mut topo, FRAC_PI_2));
        let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
        let mut arena = GfaArena::new();
        let id = arena.init_edge_pave_block(edge, seam, 0.0, seam, TAU);
        let paves = |arena: &GfaArena| arena.pave_blocks.get(id).unwrap().extra_paves.len();

        add_pave_to_edge(&mut arena, &topo, edge, Pave::new(seam, 3.0 * FRAC_PI_2));
        add_pave_to_edge(&mut arena, &topo, edge, Pave::new(seam, -FRAC_PI_2));
        assert_eq!(paves(&arena), 0, "the vertex is the block's end");
        add_pave_to_edge(&mut arena, &topo, edge, Pave::new(origin, 0.0));
        add_pave_to_edge(&mut arena, &topo, edge, Pave::new(origin, TAU));
        add_pave_to_edge(&mut arena, &topo, edge, Pave::new(quarter, FRAC_PI_2));
        assert_eq!(paves(&arena), 3, "the origin is inside the block");

        perform(&mut topo, &mut arena).unwrap();
        // From the vertex: a quarter turn to the origin, a quarter turn on,
        // and the rest of the turn.
        let children = arena.pave_blocks.get(id).unwrap().children.clone();
        let spans: Vec<f64> = children
            .iter()
            .map(|&c| {
                let (a, b) = arena.pave_blocks.get(c).unwrap().parameter_range();
                b - a
            })
            .collect();
        assert_eq!(spans.len(), 3, "{spans:?}");
        for (span, want) in spans.iter().zip([FRAC_PI_2, FRAC_PI_2, 2.0 * FRAC_PI_2]) {
            assert!((span - want).abs() < 1e-12, "{spans:?}");
        }
    }
}
