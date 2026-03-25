//! Link section PaveBlocks with existing boundary PaveBlocks.
//!
//! After ForceInterfEE has grouped boundary PBs into CommonBlocks,
//! this pass checks each FF section PB against boundary PBs. When a
//! section PB has the same resolved vertex endpoints as a boundary PB
//! (and compatible curve geometry), it is added to the boundary PB's
//! CommonBlock — or a new CB is created for the pair.
//!
//! This implements the reference implementation's `IsExistingPaveBlock`
//! pattern: section edges that coincide with face boundary edges are
//! linked so `MakeSplitEdges` creates one shared edge entity.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;

use crate::ds::{GfaArena, PaveBlockId};
use crate::error::AlgoError;

/// Quantized 3D position pair for endpoint matching.
type QPair = ((i64, i64, i64), (i64, i64, i64));

/// Link section PBs to coincident boundary PBs via CommonBlocks.
///
/// For each leaf section PB (from `arena.curves`), resolves its vertex
/// endpoints and searches for a boundary PB with matching resolved
/// endpoints and compatible curve geometry. If found, links them in a
/// CommonBlock so `MakeSplitEdges` creates a shared edge.
///
/// # Errors
///
/// Returns [`AlgoError`] if topology lookups fail.
pub fn perform(topo: &Topology, tol: Tolerance, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect resolved endpoints for all boundary leaf PBs.
    // Key: (min_pos, max_pos) quantized at tolerance, Value: list of PB IDs.
    let scale = 1.0 / tol.linear;
    let qpt = |p: brepkit_math::vec::Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // Build index of boundary PBs by quantized endpoint position pair.
    let mut boundary_index: std::collections::HashMap<QPair, Vec<PaveBlockId>> =
        std::collections::HashMap::new();

    let all_edge_pbs: Vec<Vec<PaveBlockId>> = arena
        .edge_pave_blocks
        .values()
        .map(|pbs| arena.collect_leaf_pave_blocks(pbs))
        .collect();

    for leaf_pbs in &all_edge_pbs {
        for &pb_id in leaf_pbs {
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);
            let Ok(sp) = topo.vertex(sv).map(brepkit_topology::vertex::Vertex::point) else {
                continue;
            };
            let Ok(ep) = topo.vertex(ev).map(brepkit_topology::vertex::Vertex::point) else {
                continue;
            };
            let qs = qpt(sp);
            let qe = qpt(ep);
            let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
            boundary_index.entry(key).or_default().push(pb_id);
        }
    }

    // For each section PB, check if a boundary PB has matching endpoints.
    let mut linked = 0_usize;

    // Collect section PB IDs upfront to avoid borrowing arena.curves while mutating arena.
    let section_pb_ids: Vec<PaveBlockId> = arena
        .curves
        .iter()
        .flat_map(|c| c.pave_blocks.iter().copied())
        .collect();

    for root_pb_id in &section_pb_ids {
        let leaves = arena.collect_leaf_pave_blocks(&[*root_pb_id]);
        for section_pb_id in leaves {
            // Already in a CB — skip
            if arena.pb_to_cb.contains_key(&section_pb_id) {
                continue;
            }

            let Some(section_pb) = arena.pave_blocks.get(section_pb_id) else {
                continue;
            };
            let sv = arena.resolve_vertex(section_pb.start.vertex);
            let ev = arena.resolve_vertex(section_pb.end.vertex);
            let Ok(sp) = topo.vertex(sv).map(brepkit_topology::vertex::Vertex::point) else {
                continue;
            };
            let Ok(ep) = topo.vertex(ev).map(brepkit_topology::vertex::Vertex::point) else {
                continue;
            };
            let qs = qpt(sp);
            let qe = qpt(ep);
            let key = if qs <= qe { (qs, qe) } else { (qe, qs) };

            let Some(candidates) = boundary_index.get(&key) else {
                log::trace!("link_existing: no boundary PB at position");
                continue;
            };

            // Check curve compatibility with each candidate
            let section_curve = topo.edge(section_pb.original_edge)?.curve().clone();

            for &boundary_pb_id in candidates {
                // Already in same CB
                if arena.pb_to_cb.get(&boundary_pb_id) == arena.pb_to_cb.get(&section_pb_id)
                    && arena.pb_to_cb.contains_key(&section_pb_id)
                {
                    continue;
                }

                let Some(boundary_pb) = arena.pave_blocks.get(boundary_pb_id) else {
                    continue;
                };
                let boundary_curve = topo.edge(boundary_pb.original_edge)?.curve();

                if !curves_compatible(&section_curve, boundary_curve, tol) {
                    continue;
                }

                // Link: add section PB to boundary PB's CB, or create new CB
                if let Some(&cb_id) = arena.pb_to_cb.get(&boundary_pb_id) {
                    // Add section PB to existing CB
                    if let Some(cb) = arena.common_blocks.get_mut(cb_id) {
                        cb.pave_blocks.push(section_pb_id);
                    }
                    arena.pb_to_cb.insert(section_pb_id, cb_id);
                } else {
                    // Create new CB for the pair
                    arena.create_common_block(vec![boundary_pb_id, section_pb_id], tol.linear);
                }

                linked += 1;
                log::debug!(
                    "link_existing: linked section PB {section_pb_id:?} with boundary PB \
                     {boundary_pb_id:?}"
                );
                break; // One match is sufficient
            }
        }
    }

    if linked > 0 {
        log::debug!(
            "link_existing: linked {linked} section PBs with boundary PBs ({} section total, {} boundary groups)",
            section_pb_ids.len(),
            boundary_index.len()
        );
    }

    Ok(())
}

/// Check if two edge curves are geometrically compatible.
fn curves_compatible(a: &EdgeCurve, b: &EdgeCurve, tol: Tolerance) -> bool {
    match (a, b) {
        (EdgeCurve::Line, EdgeCurve::Line) => true,
        (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) => {
            (ca.radius() - cb.radius()).abs() < tol.linear
                && (ca.center() - cb.center()).length() < tol.linear
                && ca.normal().dot(cb.normal()).abs() > 1.0 - tol.angular
        }
        (EdgeCurve::Ellipse(ea), EdgeCurve::Ellipse(eb)) => {
            (ea.semi_major() - eb.semi_major()).abs() < tol.linear
                && (ea.semi_minor() - eb.semi_minor()).abs() < tol.linear
                && (ea.center() - eb.center()).length() < tol.linear
                && ea.normal().dot(eb.normal()).abs() > 1.0 - tol.angular
        }
        (EdgeCurve::NurbsCurve(_), EdgeCurve::NurbsCurve(_)) => false,
        (
            EdgeCurve::Line,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_),
        )
        | (
            EdgeCurve::Circle(_),
            EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_),
        )
        | (
            EdgeCurve::Ellipse(_),
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::NurbsCurve(_),
        )
        | (
            EdgeCurve::NurbsCurve(_),
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_),
        ) => false,
    }
}
