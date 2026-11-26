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
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Pave, PaveBlock, PaveBlockId};
use crate::error::AlgoError;

/// Quantized 3D position pair for endpoint matching.
type QPair = ((i64, i64, i64), (i64, i64, i64));

/// Quantized circle geometry (center, radius, axis up to sign) for
/// matching full closed blocks whose seam vertices differ.
type QCircle = ((i64, i64, i64), i64, (i64, i64, i64));

/// Quantization scale for unit axis directions.
const AXIS_SCALE: f64 = 1.0e7;

fn circle_key(
    c: &brepkit_math::curves::Circle3D,
    qpt: impl Fn(brepkit_math::vec::Point3) -> (i64, i64, i64),
    linear_scale: f64,
) -> QCircle {
    let n = c.normal();
    // Canonicalize axis sign so opposite-facing but coincident circles match.
    let flip = match (n.x().abs() > 0.5, n.y().abs() > 0.5) {
        (true, _) => n.x() < 0.0,
        (false, true) => n.y() < 0.0,
        (false, false) => n.z() < 0.0,
    };
    let n = if flip { -n } else { n };
    #[allow(clippy::cast_possible_truncation)]
    let qaxis = (
        (n.x() * AXIS_SCALE).round() as i64,
        (n.y() * AXIS_SCALE).round() as i64,
        (n.z() * AXIS_SCALE).round() as i64,
    );
    #[allow(clippy::cast_possible_truncation)]
    let qr = (c.radius() * linear_scale).round() as i64;
    (qpt(c.center()), qr, qaxis)
}

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
#[allow(clippy::unnecessary_wraps)] // Signature matches other PaveFiller passes
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

    let mut boundary_index: std::collections::HashMap<QPair, Vec<PaveBlockId>> =
        std::collections::HashMap::new();

    // Secondary index for full closed circle blocks (start == end vertex):
    // their endpoint is an arbitrary seam vertex, so endpoint-pair keys
    // cannot match across differently-seamed but coincident circles.
    let mut closed_index: std::collections::HashMap<QCircle, Vec<PaveBlockId>> =
        std::collections::HashMap::new();

    // Open boundary leaf PBs by their resolved end vertices, for sections
    // that lie along a boundary edge split where the section is not.
    let mut boundary_by_vertex: std::collections::HashMap<VertexId, Vec<PaveBlockId>> =
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
            if sv != ev {
                boundary_by_vertex.entry(sv).or_default().push(pb_id);
                boundary_by_vertex.entry(ev).or_default().push(pb_id);
            }

            if sv == ev
                && let Ok(edge) = topo.edge(pb.original_edge)
                && let EdgeCurve::Circle(c) = edge.curve()
            {
                closed_index
                    .entry(circle_key(c, qpt, scale))
                    .or_default()
                    .push(pb_id);
            }
        }
    }

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

            // Check curve compatibility with each candidate.
            // Use graceful skip (not `?`) for edge lookups — consistent with
            // vertex lookups above. A stale original_edge should skip the PB,
            // not abort the entire linking pass.
            let Ok(section_edge) = topo.edge(section_pb.original_edge) else {
                continue;
            };
            let section_curve = section_edge.curve().clone();

            let mut linked_this = false;
            if let Some(candidates) = boundary_index.get(&key) {
                for &boundary_pb_id in candidates {
                    if try_link(
                        topo,
                        tol,
                        arena,
                        section_pb_id,
                        &section_curve,
                        boundary_pb_id,
                    ) {
                        linked += 1;
                        linked_this = true;
                        break; // One match is sufficient
                    }
                }
            } else {
                log::trace!("link_existing: no boundary PB at position");
            }

            if !linked_this && sv != ev {
                let pieces = link_along_split_arc(
                    topo,
                    tol,
                    arena,
                    &boundary_by_vertex,
                    section_pb_id,
                    &section_curve,
                );
                linked += pieces;
                linked_this = pieces > 0;
            }
            if !linked_this && sv != ev {
                let pieces = link_along_split_spline(
                    topo,
                    tol,
                    arena,
                    &boundary_by_vertex,
                    section_pb_id,
                    &section_curve,
                );
                linked += pieces;
                linked_this = pieces > 0;
            }

            // Fallback for full closed circles: the endpoint-pair key uses
            // the seam vertex, which is arbitrary, so coincident circles
            // with different seams never share a key. Match on quantized
            // circle geometry instead.
            if !linked_this
                && sv == ev
                && let EdgeCurve::Circle(c) = &section_curve
                && let Some(candidates) = closed_index.get(&circle_key(c, qpt, scale))
            {
                for &boundary_pb_id in candidates {
                    if try_link(
                        topo,
                        tol,
                        arena,
                        section_pb_id,
                        &section_curve,
                        boundary_pb_id,
                    ) {
                        linked += 1;
                        break;
                    }
                }
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

/// Most pieces of one boundary edge [`link_along_split_spline`] walks.
const MOST_SPLINE_PIECES: usize = 64;

/// Link a B-spline section that runs along a B-spline boundary edge the pave
/// filler split where the section is not (an edge that weaves about a
/// surface the section lies on, split where it crosses it): the edge's
/// pieces from the section's start to its end split the section where their
/// vertices project onto it, and each piece of the section joins the edge's
/// piece beside it. Every pair must be proved one curve within the
/// section's tolerance and the edge's
/// ([`super::force_interf_ee::spans_lie_on_one_curve`]), and each inner
/// vertex must project inside the section, in the order the edge runs, and
/// within that sum of it. Returns how many pieces were linked: none unless
/// every piece was.
fn link_along_split_spline(
    topo: &Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
    boundary_by_vertex: &std::collections::HashMap<VertexId, Vec<PaveBlockId>>,
    section_pb_id: PaveBlockId,
    section_curve: &EdgeCurve,
) -> usize {
    use brepkit_topology::edge::EdgeId;
    let EdgeCurve::NurbsCurve(curve) = section_curve else {
        return 0;
    };
    let Some(section) = arena.pave_blocks.get(section_pb_id) else {
        return 0;
    };
    let (start, end, section_edge) = (section.start, section.end, section.original_edge);
    let (sv, ev) = (
        arena.resolve_vertex(start.vertex),
        arena.resolve_vertex(end.vertex),
    );
    let Ok(section_tol) = topo
        .edge(section_edge)
        .map(brepkit_topology::edge::Edge::tolerance)
    else {
        return 0;
    };
    let rising = start.parameter < end.parameter;
    let mut edges: Vec<EdgeId> = Vec::new();
    for &pb_id in boundary_by_vertex.get(&sv).into_iter().flatten() {
        if let Some(pb) = arena.pave_blocks.get(pb_id)
            && topo
                .edge(pb.original_edge)
                .is_ok_and(|e| matches!(e.curve(), EdgeCurve::NurbsCurve(_)))
            && !edges.contains(&pb.original_edge)
        {
            edges.push(pb.original_edge);
        }
    }
    for edge_id in edges {
        let Ok(edge) = topo.edge(edge_id) else {
            continue;
        };
        let within = section_tol.unwrap_or(0.0).max(tol.linear) + edge.tolerance().unwrap_or(0.0);
        // The edge's pieces from the section's start vertex to its end.
        let mut chain: Vec<(PaveBlockId, VertexId)> = Vec::new();
        let mut at = sv;
        while at != ev && chain.len() < MOST_SPLINE_PIECES {
            let next = boundary_by_vertex.get(&at).and_then(|pbs| {
                pbs.iter().find_map(|&pb_id| {
                    let pb = arena.pave_blocks.get(pb_id)?;
                    if pb.original_edge != edge_id || chain.iter().any(|&(c, _)| c == pb_id) {
                        return None;
                    }
                    let (a, b) = (
                        arena.resolve_vertex(pb.start.vertex),
                        arena.resolve_vertex(pb.end.vertex),
                    );
                    Some((pb_id, if a == at { b } else { a }))
                })
            });
            let Some(step) = next else {
                break;
            };
            chain.push(step);
            at = step.1;
        }
        if at != ev || chain.len() < 2 {
            continue;
        }
        // The section's parameter at each inner vertex, in the edge's order,
        // each strictly past the last the way the section runs.
        let mut cuts: Vec<Pave> = Vec::with_capacity(chain.len() - 1);
        let mut last = start.parameter;
        let mut placed = true;
        for &(_, v) in &chain[..chain.len() - 1] {
            let found = topo.vertex(v).ok().and_then(|x| {
                brepkit_math::nurbs::projection::project_point_to_curve(curve, x.point(), 1e-12)
                    .ok()
            });
            let Some(found) = found else {
                placed = false;
                break;
            };
            let t = found.parameter;
            let onward = if rising {
                t > last && t < end.parameter
            } else {
                t < last && t > end.parameter
            };
            if !onward || found.distance.is_nan() || found.distance > within {
                placed = false;
                break;
            }
            cuts.push(Pave::new(v, t));
            last = t;
        }
        if !placed {
            continue;
        }
        let mut pieces: Vec<(Pave, Pave, PaveBlockId)> = Vec::with_capacity(chain.len());
        let mut from = start;
        for (pave, &(partner, _)) in cuts.into_iter().chain(std::iter::once(end)).zip(&chain) {
            let Some(theirs) = arena.pave_blocks.get(partner) else {
                break;
            };
            let one = super::force_interf_ee::spans_lie_on_one_curve(
                topo,
                (section_edge, (from.parameter, pave.parameter)),
                (
                    theirs.original_edge,
                    (theirs.start.parameter, theirs.end.parameter),
                ),
                within,
            )
            .unwrap_or(false);
            if !one {
                break;
            }
            pieces.push((from, pave, partner));
            from = pave;
        }
        if pieces.len() != chain.len() {
            continue;
        }
        let children: Vec<PaveBlockId> = pieces
            .iter()
            .map(|&(a, b, _)| arena.pave_blocks.alloc(PaveBlock::new(section_edge, a, b)))
            .collect();
        if let Some(pb) = arena.pave_blocks.get_mut(section_pb_id) {
            pb.children.clone_from(&children);
        }
        let mut linked = 0;
        for (child, &(_, _, partner)) in children.into_iter().zip(&pieces) {
            if try_link(topo, tol, arena, child, section_curve, partner) {
                linked += 1;
            }
        }
        return linked;
    }
    0
}

/// Link a section arc that runs along a boundary arc the pave filler split
/// where the section is not (a rim shared by two coincident walls, split at
/// a vertex of one): the boundary pieces from the section's start to its end
/// on the same circle split the section at their vertices, and each piece
/// of it joins its boundary piece. Without this the section stays one edge
/// beside the rim's pieces, and the two bound a sliver of no area. Returns
/// how many pieces were linked: none unless every piece was.
fn link_along_split_arc(
    topo: &Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
    boundary_by_vertex: &std::collections::HashMap<VertexId, Vec<PaveBlockId>>,
    section_pb_id: PaveBlockId,
    section_curve: &EdgeCurve,
) -> usize {
    use std::f64::consts::TAU;
    const MAX_PIECES: usize = 8;
    let EdgeCurve::Circle(circle) = section_curve else {
        return 0;
    };
    let Some(section) = arena.pave_blocks.get(section_pb_id) else {
        return 0;
    };
    let (start, end, original_edge) = (section.start, section.end, section.original_edge);
    let (t0, t1) = (
        start.parameter.min(end.parameter),
        start.parameter.max(end.parameter),
    );
    if t1 - t0 >= TAU - 1e-9 {
        return 0;
    }
    let (sv, ev) = (
        arena.resolve_vertex(start.vertex),
        arena.resolve_vertex(end.vertex),
    );
    // The section's own parameter of a point on its circle, strictly inside
    // its span.
    let inside = |p: brepkit_math::vec::Point3| -> Option<f64> {
        let t = t0 + (circle.project(p) - t0).rem_euclid(TAU);
        ((circle.evaluate(t) - p).length() <= tol.linear * 100.0 && t > t0 + 1e-9 && t < t1 - 1e-9)
            .then_some(t)
    };
    let point = |v: VertexId| {
        topo.vertex(v)
            .ok()
            .map(brepkit_topology::vertex::Vertex::point)
    };
    // The section's parameter at a vertex of the walk.
    let along = |v: VertexId| {
        if v == sv {
            Some(start.parameter)
        } else if v == ev {
            Some(end.parameter)
        } else {
            point(v).and_then(inside)
        }
    };
    // Walk boundary pieces on this circle from the start vertex, each one
    // lying inside the section's span and able to join the section's piece
    // beside it, until the end vertex. A piece that cannot join (a copy on a
    // turned circle) is passed over for the next.
    let mut chain: Vec<(PaveBlockId, VertexId)> = Vec::new();
    let mut at = sv;
    while at != ev && chain.len() < MAX_PIECES {
        let next = boundary_by_vertex.get(&at).and_then(|pbs| {
            pbs.iter().find_map(|&pb_id| {
                if chain.iter().any(|&(c, _)| c == pb_id) {
                    return None;
                }
                let pb = arena.pave_blocks.get(pb_id)?;
                let edge = topo.edge(pb.original_edge).ok()?;
                let EdgeCurve::Circle(own) = edge.curve() else {
                    return None;
                };
                let (a, b) = (
                    arena.resolve_vertex(pb.start.vertex),
                    arena.resolve_vertex(pb.end.vertex),
                );
                let other = if a == at { b } else { a };
                let mid = own.evaluate(f64::midpoint(pb.start.parameter, pb.end.parameter));
                inside(mid)?;
                let span = (along(at)?, along(other)?);
                can_link(topo, tol, arena, section_curve, span, pb_id).then_some((pb_id, other))
            })
        });
        let Some((pb_id, other)) = next else {
            return 0;
        };
        chain.push((pb_id, other));
        at = other;
    }
    if at != ev || chain.len() < 2 {
        return 0;
    }
    // Split the section at the chain's inner vertices, in the order the
    // section runs.
    let mut cuts: Vec<Pave> = Vec::with_capacity(chain.len() - 1);
    for &(_, v) in &chain[..chain.len() - 1] {
        let Some(t) = point(v).and_then(inside) else {
            return 0;
        };
        cuts.push(Pave::new(v, t));
    }
    if start.parameter > end.parameter {
        cuts.sort_by(|a, b| b.parameter.total_cmp(&a.parameter));
    } else {
        cuts.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
    }
    // Pair every piece with its rim piece before the section is split: a
    // piece left without one would stay beside the rim as the sliver this
    // function exists to prevent.
    let mut pieces: Vec<(Pave, Pave, PaveBlockId)> = Vec::with_capacity(cuts.len() + 1);
    let mut from = start;
    for pave in cuts.into_iter().chain(std::iter::once(end)) {
        let mut ours = [
            arena.resolve_vertex(from.vertex),
            arena.resolve_vertex(pave.vertex),
        ];
        ours.sort_by_key(|v| v.index());
        let partner = chain.iter().find_map(|&(pb_id, _)| {
            let b = arena.pave_blocks.get(pb_id)?;
            let mut theirs = [
                arena.resolve_vertex(b.start.vertex),
                arena.resolve_vertex(b.end.vertex),
            ];
            theirs.sort_by_key(|v| v.index());
            let span = (from.parameter, pave.parameter);
            (theirs == ours && can_link(topo, tol, arena, section_curve, span, pb_id))
                .then_some(pb_id)
        });
        let Some(partner) = partner else {
            return 0;
        };
        pieces.push((from, pave, partner));
        from = pave;
    }
    let children: Vec<PaveBlockId> = pieces
        .iter()
        .map(|&(a, b, _)| arena.pave_blocks.alloc(PaveBlock::new(original_edge, a, b)))
        .collect();
    if let Some(pb) = arena.pave_blocks.get_mut(section_pb_id) {
        pb.children.clone_from(&children);
    }
    let mut linked = 0;
    for (child, &(_, _, partner)) in children.into_iter().zip(&pieces) {
        if try_link(topo, tol, arena, child, section_curve, partner) {
            linked += 1;
        }
    }
    linked
}

/// Whether [`try_link`] can link a new pave block over `span` of
/// `section_curve`, in no common block yet, with `boundary_pb_id`: the
/// curves pass the gate, and the two blocks are [`one_arc`].
///
/// [`one_arc`]: super::force_interf_ee::one_arc
fn can_link(
    topo: &Topology,
    tol: Tolerance,
    arena: &GfaArena,
    section_curve: &EdgeCurve,
    span: (f64, f64),
    boundary_pb_id: PaveBlockId,
) -> bool {
    arena.pave_blocks.get(boundary_pb_id).is_some_and(|pb| {
        topo.edge(pb.original_edge).is_ok_and(|edge| {
            curves_compatible(section_curve, edge.curve(), tol)
                && super::force_interf_ee::one_arc(
                    section_curve,
                    span,
                    edge.curve(),
                    pb.parameter_range(),
                    tol,
                )
        })
    })
}

/// Attempt to link a section PB with a boundary PB into a CommonBlock.
///
/// Returns `true` if the pair was linked. Skips self-pairs, pairs already
/// sharing a CB, and geometrically incompatible curves.
fn try_link(
    topo: &Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
    section_pb_id: PaveBlockId,
    section_curve: &EdgeCurve,
    boundary_pb_id: PaveBlockId,
) -> bool {
    // Self-match guard: coplanar FF section PBs can appear in both
    // arena.curves and arena.edge_pave_blocks. Skip self-pairing.
    if boundary_pb_id == section_pb_id {
        return false;
    }

    if arena.pb_to_cb.get(&boundary_pb_id) == arena.pb_to_cb.get(&section_pb_id)
        && arena.pb_to_cb.contains_key(&section_pb_id)
    {
        return false;
    }

    let Some(boundary_pb) = arena.pave_blocks.get(boundary_pb_id) else {
        return false;
    };

    let Ok(boundary_edge) = topo.edge(boundary_pb.original_edge) else {
        return false;
    };
    let boundary_curve = boundary_edge.curve();

    let within = if let (EdgeCurve::NurbsCurve(_), EdgeCurve::NurbsCurve(_)) =
        (section_curve, boundary_curve)
    {
        // A B-spline section and a B-spline boundary edge are both fits: they
        // are one edge where they are proved one curve within the section's
        // proved distance from its faces and the edge's own tolerance.
        let section_tol = arena
            .pave_blocks
            .get(section_pb_id)
            .and_then(|pb| topo.edge(pb.original_edge).ok())
            .and_then(brepkit_topology::edge::Edge::tolerance);
        let within =
            section_tol.unwrap_or(0.0).max(tol.linear) + boundary_edge.tolerance().unwrap_or(0.0);
        let proved = super::force_interf_ee::block_lies_on_block(
            topo,
            arena,
            section_pb_id,
            boundary_pb_id,
            within,
        );
        if !proved.unwrap_or(false) {
            return false;
        }
        within
    } else {
        if !curves_compatible(section_curve, boundary_curve, tol) {
            return false;
        }
        // On one circle, two blocks with the same ends cover one arc or the
        // two arcs between those ends.
        let boundary_span = boundary_pb.parameter_range();
        let Some(section_span) = arena
            .pave_blocks
            .get(section_pb_id)
            .map(PaveBlock::parameter_range)
        else {
            return false;
        };
        if !super::force_interf_ee::one_arc(
            section_curve,
            section_span,
            boundary_curve,
            boundary_span,
            tol,
        ) {
            return false;
        }
        tol.linear
    };

    if let Some(&cb_id) = arena.pb_to_cb.get(&boundary_pb_id) {
        if let Some(cb) = arena.common_blocks.get_mut(cb_id) {
            cb.pave_blocks.push(section_pb_id);
            cb.tolerance = cb.tolerance.max(within);
        }
        arena.pb_to_cb.insert(section_pb_id, cb_id);
    } else {
        arena.create_common_block(vec![boundary_pb_id, section_pb_id], within);
    }

    log::debug!(
        "link_existing: linked section PB {section_pb_id:?} with boundary PB {boundary_pb_id:?}"
    );
    true
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::aabb::Aabb3;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use crate::ds::{GfaArena, IntersectionCurveDS};

    /// A cubic edge with a tolerance of 2e-4, split at two vertices on it,
    /// and a section from its start to its end with a tolerance of 1e-4 that
    /// bows `bow` off it at its middle. Within the 3e-4 the two allow, the
    /// section is cut where the edge's vertices project onto it, and each
    /// piece joins the edge's piece beside it. Past that, the section stays
    /// whole and joins nothing. Upright, turned and mirrored.
    #[test]
    fn a_b_spline_section_along_a_split_b_spline_edge_links_piece_by_piece() {
        use crate::ds::{Pave, PaveBlock};
        use brepkit_math::nurbs::curve::NurbsCurve;
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let o = Point3::new(3.0, -2.0, 5.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let poses: [(&str, Point3, [Vec3; 3]); 3] = [
            ("upright", Point3::new(0.0, 0.0, 0.0), [x, y, z]),
            ("turned", o, [turn(x), turn(y), turn(z)]),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                [mirror(turn(x)), mirror(turn(y)), mirror(turn(z))],
            ),
        ];
        for (name, origin, [u, v, w]) in poses {
            let at = |a: f64, b: f64, c: f64| origin + u * a + v * b + w * c;
            // The middle Bezier points carry three quarters of their lift to
            // the cubic's middle.
            let cubic = |lift: f64| {
                NurbsCurve::new(
                    3,
                    vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    vec![
                        at(0.0, 0.0, 0.0),
                        at(3.0, 1.0, lift / 0.75),
                        at(7.0, -1.0, lift / 0.75),
                        at(10.0, 0.0, 0.0),
                    ],
                    vec![1.0; 4],
                )
                .unwrap()
            };
            for (bow, linked) in [(1e-4, true), (1e-3, false)] {
                let (along, beside) = (cubic(0.0), cubic(bow));
                let mut topo = Topology::default();
                let mut vertex = |t: f64| topo.add_vertex(Vertex::new(along.evaluate(t), 1e-7));
                let (a, m1, m2, b) = (vertex(0.0), vertex(0.3), vertex(0.6), vertex(1.0));
                let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(along.clone())));
                topo.edge_mut(edge).unwrap().set_tolerance(Some(2e-4));
                let section = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(beside.clone())));
                topo.edge_mut(section).unwrap().set_tolerance(Some(1e-4));
                let wire = topo
                    .add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
                let face = topo.add_face(Face::new(
                    wire,
                    vec![],
                    FaceSurface::Plane {
                        normal: w,
                        d: w.dot(Vec3::new(origin.x(), origin.y(), origin.z())),
                    },
                ));
                let mut arena = GfaArena::new();
                let root = arena.init_edge_pave_block(edge, a, 0.0, b, 1.0);
                let parts: Vec<_> = [(a, 0.0, m1, 0.3), (m1, 0.3, m2, 0.6), (m2, 0.6, b, 1.0)]
                    .iter()
                    .map(|&(s, ts, e, te)| {
                        arena.pave_blocks.alloc(PaveBlock::new(
                            edge,
                            Pave::new(s, ts),
                            Pave::new(e, te),
                        ))
                    })
                    .collect();
                arena
                    .pave_blocks
                    .get_mut(root)
                    .unwrap()
                    .children
                    .clone_from(&parts);
                let section_pb = arena.pave_blocks.alloc(PaveBlock::new(
                    section,
                    Pave::new(a, 0.0),
                    Pave::new(b, 1.0),
                ));
                arena.curves.push(IntersectionCurveDS {
                    curve: EdgeCurve::NurbsCurve(beside.clone()),
                    face_a: face,
                    face_b: face,
                    bbox: Aabb3::from_points(beside.control_points().iter().copied()),
                    pave_blocks: vec![section_pb],
                    t_range: (0.0, 1.0),
                });

                super::perform(&topo, Tolerance::default(), &mut arena).unwrap();

                let leaves = arena.collect_leaf_pave_blocks(&[section_pb]);
                if linked {
                    assert_eq!(leaves.len(), 3, "{name}: the section's pieces");
                    for (leaf, part) in leaves.iter().zip(&parts) {
                        let cb = arena
                            .pb_to_cb
                            .get(leaf)
                            .and_then(|&c| arena.common_blocks.get(c));
                        assert!(
                            cb.is_some_and(|cb| cb.pave_blocks.contains(part)),
                            "{name}: a piece apart from the edge's"
                        );
                    }
                } else {
                    assert!(
                        leaves == vec![section_pb] && !arena.pb_to_cb.contains_key(&section_pb),
                        "{name}: a section {bow:e} off the edge is linked"
                    );
                }
            }
        }
    }

    /// Two copies of a cubic edge, each with a tolerance of 2e-4 and split
    /// at two vertices on it, their pieces already one common block each, and
    /// a section 1e-4 off them with a tolerance of 1e-4. Each piece of the
    /// section joins a block within the 3e-4 the two allow: the block, and
    /// the one edge made for it, then hold that 3e-4. Upright, turned and
    /// mirrored.
    #[test]
    fn a_section_joining_a_common_block_raises_its_tolerance() {
        use crate::ds::{Pave, PaveBlock};
        use brepkit_math::nurbs::curve::NurbsCurve;
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let o = Point3::new(3.0, -2.0, 5.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let poses: [(&str, Point3, [Vec3; 3]); 3] = [
            ("upright", Point3::new(0.0, 0.0, 0.0), [x, y, z]),
            ("turned", o, [turn(x), turn(y), turn(z)]),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                [mirror(turn(x)), mirror(turn(y)), mirror(turn(z))],
            ),
        ];
        for (name, origin, [u, v, w]) in poses {
            let at = |a: f64, b: f64, c: f64| origin + u * a + v * b + w * c;
            let cubic = |lift: f64| {
                NurbsCurve::new(
                    3,
                    vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    vec![
                        at(0.0, 0.0, 0.0),
                        at(3.0, 1.0, lift / 0.75),
                        at(7.0, -1.0, lift / 0.75),
                        at(10.0, 0.0, 0.0),
                    ],
                    vec![1.0; 4],
                )
                .unwrap()
            };
            let (along, beside) = (cubic(0.0), cubic(1e-4));
            let mut topo = Topology::default();
            let mut vertex = |t: f64| topo.add_vertex(Vertex::new(along.evaluate(t), 1e-7));
            let (a, m1, m2, b) = (vertex(0.0), vertex(0.3), vertex(0.6), vertex(1.0));
            let mut arena = GfaArena::new();
            let copy = |topo: &mut Topology, arena: &mut GfaArena| {
                let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(along.clone())));
                topo.edge_mut(edge).unwrap().set_tolerance(Some(2e-4));
                let root = arena.init_edge_pave_block(edge, a, 0.0, b, 1.0);
                let parts: Vec<_> = [(a, 0.0, m1, 0.3), (m1, 0.3, m2, 0.6), (m2, 0.6, b, 1.0)]
                    .iter()
                    .map(|&(s, ts, e, te)| {
                        arena.pave_blocks.alloc(PaveBlock::new(
                            edge,
                            Pave::new(s, ts),
                            Pave::new(e, te),
                        ))
                    })
                    .collect();
                arena
                    .pave_blocks
                    .get_mut(root)
                    .unwrap()
                    .children
                    .clone_from(&parts);
                parts
            };
            let first = copy(&mut topo, &mut arena);
            let second = copy(&mut topo, &mut arena);
            for (&p, &q) in first.iter().zip(&second) {
                arena.create_common_block(vec![p, q], 1e-7);
            }
            let section = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(beside.clone())));
            topo.edge_mut(section).unwrap().set_tolerance(Some(1e-4));
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: w,
                    d: w.dot(Vec3::new(origin.x(), origin.y(), origin.z())),
                },
            ));
            let section_pb = arena.pave_blocks.alloc(PaveBlock::new(
                section,
                Pave::new(a, 0.0),
                Pave::new(b, 1.0),
            ));
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::NurbsCurve(beside.clone()),
                face_a: face,
                face_b: face,
                bbox: Aabb3::from_points(beside.control_points().iter().copied()),
                pave_blocks: vec![section_pb],
                t_range: (0.0, 1.0),
            });

            super::perform(&topo, Tolerance::default(), &mut arena).unwrap();
            crate::pave_filler::make_split_edges::perform(&mut topo, &mut arena).unwrap();

            let leaves = arena.collect_leaf_pave_blocks(&[section_pb]);
            assert_eq!(leaves.len(), 3, "{name}: the section's pieces");
            for (leaf, part) in leaves.iter().zip(&first) {
                let cb = arena
                    .pb_to_cb
                    .get(leaf)
                    .and_then(|&c| arena.common_blocks.get(c))
                    .unwrap();
                assert!(cb.pave_blocks.contains(part), "{name}: a piece apart");
                assert!(
                    cb.tolerance >= 3e-4,
                    "{name}: the block holds {}",
                    cb.tolerance
                );
                let held = cb
                    .split_edge
                    .and_then(|e| topo.edge(e).unwrap().tolerance());
                assert!(
                    held.is_some_and(|t| t >= 3e-4),
                    "{name}: the block's edge holds {held:?}"
                );
            }
        }
    }

    #[test]
    fn a_section_along_a_rim_is_split_only_when_every_piece_links() {
        // Two coincident rims split at different vertices: one piece runs
        // from the start at angle 0 to angle 2, the other's pieces from 2
        // back to 1 and from 1 to the end at 3. The walk along them from
        // the section's start reaches its end, but the section's pieces
        // (0 to 1, 1 to 2, 2 to 3) match only one of the three rim pieces.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let mut topo = Topology::default();
        let v: Vec<_> = [0.0, 1.0, 2.0, 3.0]
            .iter()
            .map(|&t| topo.add_vertex(Vertex::new(circle.evaluate(t), 1e-7)))
            .collect();
        let rim = |topo: &mut Topology, a: usize, b: usize| {
            topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Circle(circle.clone())))
        };
        let e02 = rim(&mut topo, 0, 2);
        let e21 = rim(&mut topo, 1, 2);
        let e13 = rim(&mut topo, 1, 3);
        let section = rim(&mut topo, 0, 3);
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let mut arena = GfaArena::new();
        arena.init_edge_pave_block(e02, v[0], 0.0, v[2], 2.0);
        arena.init_edge_pave_block(e21, v[1], 1.0, v[2], 2.0);
        arena.init_edge_pave_block(e13, v[1], 1.0, v[3], 3.0);
        let section_pb = arena.pave_blocks.alloc(crate::ds::PaveBlock::new(
            section,
            crate::ds::Pave::new(v[0], 0.0),
            crate::ds::Pave::new(v[3], 3.0),
        ));
        arena.curves.push(IntersectionCurveDS {
            curve: EdgeCurve::Circle(circle),
            face_a: face,
            face_b: face,
            bbox: Aabb3::from_points([Point3::new(-1.0, -1.0, 0.0), Point3::new(1.0, 1.0, 0.0)]),
            pave_blocks: vec![section_pb],
            t_range: (0.0, 3.0),
        });

        super::perform(&topo, Tolerance::default(), &mut arena).unwrap();

        let leaves = arena.collect_leaf_pave_blocks(&[section_pb]);
        let linked = leaves
            .iter()
            .filter(|pb| arena.pb_to_cb.contains_key(pb))
            .count();
        assert!(
            leaves == vec![section_pb] && linked == 0,
            "the section is left in {} pieces, {linked} linked",
            leaves.len()
        );
    }

    /// A section along a rim from angle 0 to 2, and two rim pieces from 0 to
    /// 1 and from 1 to 2. The second lies on the rim's circle turned a
    /// millionth of a radian about the x axis: close enough for the circles'
    /// gate, but no bound holds its arc on the section's. The section stays
    /// whole, with no piece linked.
    #[test]
    fn a_section_along_a_rim_stays_whole_when_a_piece_has_no_bound() {
        let centre = Point3::new(0.0, 0.0, 0.0);
        let circle = Circle3D::with_axes(
            centre,
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let (sin, cos) = 1e-6f64.sin_cos();
        let tilted = Circle3D::with_axes(
            centre,
            Vec3::new(0.0, -sin, cos),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, cos, sin),
        )
        .unwrap();
        let mut topo = Topology::default();
        let v: Vec<_> = [0.0, 1.0, 2.0]
            .iter()
            .map(|&t| topo.add_vertex(Vertex::new(circle.evaluate(t), 1e-7)))
            .collect();
        let e01 = topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Circle(circle.clone())));
        let e12 = topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Circle(tilted)));
        let section = topo.add_edge(Edge::new(v[0], v[2], EdgeCurve::Circle(circle.clone())));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let mut arena = GfaArena::new();
        arena.init_edge_pave_block(e01, v[0], 0.0, v[1], 1.0);
        arena.init_edge_pave_block(e12, v[1], 1.0, v[2], 2.0);
        let section_pb = arena.pave_blocks.alloc(crate::ds::PaveBlock::new(
            section,
            crate::ds::Pave::new(v[0], 0.0),
            crate::ds::Pave::new(v[2], 2.0),
        ));
        arena.curves.push(IntersectionCurveDS {
            curve: EdgeCurve::Circle(circle),
            face_a: face,
            face_b: face,
            bbox: Aabb3::from_points([Point3::new(-1.0, -1.0, 0.0), Point3::new(1.0, 1.0, 0.0)]),
            pave_blocks: vec![section_pb],
            t_range: (0.0, 2.0),
        });

        super::perform(&topo, Tolerance::default(), &mut arena).unwrap();

        let leaves = arena.collect_leaf_pave_blocks(&[section_pb]);
        let linked = leaves
            .iter()
            .filter(|pb| arena.pb_to_cb.contains_key(pb))
            .count();
        assert!(
            leaves == vec![section_pb] && linked == 0,
            "the section is left in {} pieces, {linked} linked",
            leaves.len()
        );
    }

    /// A section on a circle from angle 0 to 1, and two boundary edges
    /// between its ends: first the rest of the circle, then the arc. Run 20
    /// times, each with fresh hash seeds, the section joins the arc's block
    /// every time and never the rest's.
    #[test]
    fn a_section_joins_the_arc_and_not_the_rest_on_every_run() {
        let circle = Circle3D::with_axes(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        for run in 0..20 {
            let mut topo = Topology::default();
            let (p, q) = (
                topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7)),
                topo.add_vertex(Vertex::new(circle.evaluate(1.0), 1e-7)),
            );
            let edge = |topo: &mut Topology, a, b| {
                topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle.clone())))
            };
            let rest = edge(&mut topo, q, p);
            let arc = edge(&mut topo, p, q);
            let section = edge(&mut topo, p, q);
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 0.0,
                },
            ));
            let mut arena = GfaArena::new();
            arena.init_edge_pave_block(rest, q, 1.0, p, std::f64::consts::TAU);
            arena.init_edge_pave_block(arc, p, 0.0, q, 1.0);
            let section_pb = arena.pave_blocks.alloc(crate::ds::PaveBlock::new(
                section,
                crate::ds::Pave::new(p, 0.0),
                crate::ds::Pave::new(q, 1.0),
            ));
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::Circle(circle.clone()),
                face_a: face,
                face_b: face,
                bbox: Aabb3::from_points([
                    Point3::new(-1.0, -1.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                ]),
                pave_blocks: vec![section_pb],
                t_range: (0.0, 1.0),
            });

            super::perform(&topo, Tolerance::default(), &mut arena).unwrap();

            let block = |edge| arena.edge_pave_blocks[&edge][0];
            let joined = arena.pb_to_cb.get(&section_pb);
            assert!(
                joined.is_some() && joined == arena.pb_to_cb.get(&block(arc)),
                "run {run}: the section joins no block of the arc"
            );
            assert!(
                !arena.pb_to_cb.contains_key(&block(rest)),
                "run {run}: the rest joins a block"
            );
        }
    }

    /// A section along a rim from angle 0 to 2, a rim piece from 0 to 1, and
    /// two pieces from 1 to 2: one on the rim's circle and one on that circle
    /// turned a millionth of a radian about the x axis, which no bound holds
    /// on the section. With either piece made first, the walk passes over
    /// the turned piece, and the section's two pieces join the rim's pieces
    /// on its circle.
    #[test]
    fn a_section_along_a_rim_takes_the_piece_the_bound_holds() {
        let centre = Point3::new(0.0, 0.0, 0.0);
        let circle = Circle3D::with_axes(
            centre,
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let (sin, cos) = 1e-6f64.sin_cos();
        let tilted = Circle3D::with_axes(
            centre,
            Vec3::new(0.0, -sin, cos),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, cos, sin),
        )
        .unwrap();
        for tilted_first in [true, false] {
            let mut topo = Topology::default();
            let v: Vec<_> = [0.0, 1.0, 2.0]
                .iter()
                .map(|&t| topo.add_vertex(Vertex::new(circle.evaluate(t), 1e-7)))
                .collect();
            let e01 = topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Circle(circle.clone())));
            let second = |topo: &mut Topology, c: &Circle3D| {
                topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Circle(c.clone())))
            };
            let (turned, exact) = if tilted_first {
                let turned = second(&mut topo, &tilted);
                (turned, second(&mut topo, &circle))
            } else {
                let exact = second(&mut topo, &circle);
                (second(&mut topo, &tilted), exact)
            };
            let section = topo.add_edge(Edge::new(v[0], v[2], EdgeCurve::Circle(circle.clone())));
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 0.0,
                },
            ));
            let mut arena = GfaArena::new();
            arena.init_edge_pave_block(e01, v[0], 0.0, v[1], 1.0);
            arena.init_edge_pave_block(turned, v[1], 1.0, v[2], 2.0);
            arena.init_edge_pave_block(exact, v[1], 1.0, v[2], 2.0);
            let section_pb = arena.pave_blocks.alloc(crate::ds::PaveBlock::new(
                section,
                crate::ds::Pave::new(v[0], 0.0),
                crate::ds::Pave::new(v[2], 2.0),
            ));
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::Circle(circle.clone()),
                face_a: face,
                face_b: face,
                bbox: Aabb3::from_points([
                    Point3::new(-1.0, -1.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                ]),
                pave_blocks: vec![section_pb],
                t_range: (0.0, 2.0),
            });

            super::perform(&topo, Tolerance::default(), &mut arena).unwrap();

            let block = |edge| arena.edge_pave_blocks[&edge][0];
            let leaves = arena.collect_leaf_pave_blocks(&[section_pb]);
            let joined: Vec<_> = leaves.iter().map(|pb| arena.pb_to_cb.get(pb)).collect();
            let want = [block(e01), block(exact)].map(|pb| arena.pb_to_cb.get(&pb));
            assert!(
                leaves.len() == 2 && want.iter().all(Option::is_some) && joined == want,
                "tilted first: {tilted_first}: {} pieces, joined {joined:?}, want {want:?}",
                leaves.len()
            );
            assert!(
                !arena.pb_to_cb.contains_key(&block(turned)),
                "tilted first: {tilted_first}: the turned piece joins a block"
            );
        }
    }
}
