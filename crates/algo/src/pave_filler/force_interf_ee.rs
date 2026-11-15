//! Post-split EE overlap detection — creates CommonBlocks for coincident
//! leaf PaveBlocks from different original edges.
//!
//! Runs after `make_blocks` (which splits PaveBlocks at extra paves),
//! iterating leaf PaveBlocks to find pairs with matching 3D endpoints
//! and compatible curve geometry.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};

use crate::ds::{GfaArena, PaveBlockId};
use crate::error::AlgoError;

/// Detect overlapping leaf PaveBlocks and group them into CommonBlocks.
///
/// Two leaf PaveBlocks from different original edges overlap if:
/// 1. Their start/end vertex positions are within tolerance
/// 2. Their edge curves have compatible geometry (same line direction,
///    same circle, etc.)
///
/// # Errors
///
/// Returns [`AlgoError`] if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn perform(topo: &Topology, tol: Tolerance, arena: &mut GfaArena) -> Result<(), AlgoError> {
    let all_edge_pbs: Vec<(EdgeId, Vec<PaveBlockId>)> = arena
        .edge_pave_blocks
        .iter()
        .map(|(&eid, pbs)| (eid, arena.collect_leaf_pave_blocks(pbs)))
        .collect();

    let mut leaf_data: Vec<(
        PaveBlockId,
        EdgeId,
        brepkit_math::vec::Point3,
        brepkit_math::vec::Point3,
    )> = Vec::new();

    for (orig_edge, leaf_pbs) in &all_edge_pbs {
        for &pb_id in leaf_pbs {
            let pb = match arena.pave_blocks.get(pb_id) {
                Some(pb) => pb,
                None => continue,
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);
            let start_pos = topo.vertex(sv)?.point();
            let end_pos = topo.vertex(ev)?.point();
            // A zero-length line (a whole torus's collapsed seams) shares no
            // geometry with anything; merging two would leave one edge
            // standing for both of the torus's seams.
            if (start_pos - end_pos).length() < tol.linear
                && matches!(
                    topo.edge(*orig_edge)?.curve(),
                    brepkit_topology::edge::EdgeCurve::Line
                )
            {
                continue;
            }
            leaf_data.push((pb_id, *orig_edge, start_pos, end_pos));
        }
    }

    // Find overlapping pairs. A naive scan is O(n²) over leaf PaveBlocks,
    // which explodes on solids with many edges (a shelled, lip-fused bin can
    // reach thousands of leaf blocks). Two blocks can only overlap if BOTH
    // endpoints coincide within tolerance, so we spatially hash each block by
    // the quantized cell of its (unordered) endpoint pair and only compare
    // blocks that share a candidate cell — collapsing the scan to ~O(n).
    let mut overlap_map: HashMap<PaveBlockId, Vec<PaveBlockId>> = HashMap::new();
    let n = leaf_data.len();

    // Cell size large enough that two endpoints within `tol.linear` of each
    // other never straddle the gap between non-adjacent cells once we probe
    // the immediate neighborhood. Quantizing the midpoint gives one key per
    // block; matching blocks have midpoints within `tol.linear`, so probing
    // the 3×3×3 neighbor cells of a block's midpoint covers every true match.
    let cell = (tol.linear * 4.0).max(f64::MIN_POSITIVE);
    let key = |p: brepkit_math::vec::Point3| -> (i64, i64, i64) {
        (
            (p.x() / cell).floor() as i64,
            (p.y() / cell).floor() as i64,
            (p.z() / cell).floor() as i64,
        )
    };
    let midpoint = |a: brepkit_math::vec::Point3, b: brepkit_math::vec::Point3| {
        brepkit_math::vec::Point3::new(
            f64::midpoint(a.x(), b.x()),
            f64::midpoint(a.y(), b.y()),
            f64::midpoint(a.z(), b.z()),
        )
    };

    // Bucket each leaf block by its midpoint cell.
    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, (_, _, start, end)) in leaf_data.iter().enumerate() {
        buckets
            .entry(key(midpoint(*start, *end)))
            .or_default()
            .push(i);
    }

    // For each block, gather candidate partners from its own cell plus the
    // 3×3×3 neighborhood (each block lives in exactly one bucket, so a pair is
    // visited once via `j > i`), and run the exact same fwd/rev endpoint +
    // curve-compatibility test as the naive scan.
    for i in 0..n {
        let (pb_i, edge_i, start_i, end_i) = leaf_data[i];
        let mid_i = midpoint(start_i, end_i);
        let (kx, ky, kz) = key(mid_i);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(cands) = buckets.get(&(kx + dx, ky + dy, kz + dz)) else {
                        continue;
                    };
                    for &j in cands {
                        if j <= i {
                            continue;
                        }
                        let (pb_j, edge_j, start_j, end_j) = leaf_data[j];

                        if edge_i == edge_j {
                            continue;
                        }
                        if arena.pb_to_cb.contains_key(&pb_i)
                            && arena.pb_to_cb.get(&pb_i) == arena.pb_to_cb.get(&pb_j)
                        {
                            continue;
                        }

                        let fwd_match = (start_i - start_j).length() < tol.linear
                            && (end_i - end_j).length() < tol.linear;
                        let rev_match = (start_i - end_j).length() < tol.linear
                            && (end_i - start_j).length() < tol.linear;
                        if !fwd_match && !rev_match {
                            continue;
                        }

                        let curve_i = topo.edge(edge_i)?.curve();
                        let curve_j = topo.edge(edge_j)?.curve();
                        let compatible = match curves_compatible(curve_i, curve_j, tol) {
                            Some(compatible) => compatible,
                            None => block_lies_on_block(topo, arena, pb_i, pb_j, tol)?,
                        };
                        if !compatible {
                            continue;
                        }

                        overlap_map.entry(pb_i).or_default().push(pb_j);
                        overlap_map.entry(pb_j).or_default().push(pb_i);
                    }
                }
            }
        }
    }

    // Build transitive closure and create CommonBlocks
    let mut visited: HashSet<PaveBlockId> = HashSet::new();

    for &(pb_id, _, _, _) in &leaf_data {
        if visited.contains(&pb_id) || !overlap_map.contains_key(&pb_id) {
            continue;
        }

        // BFS to find connected component
        let mut group = Vec::new();
        let mut queue = vec![pb_id];
        while let Some(current) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }
            group.push(current);
            if let Some(neighbors) = overlap_map.get(&current) {
                for &nb in neighbors {
                    if !visited.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }

        if group.len() >= 2 {
            let cb_id = arena.create_common_block(group.clone(), tol.linear);
            log::debug!(
                "ForceInterfEE: created CommonBlock {cb_id:?} with {} PaveBlocks",
                group.len()
            );
        }
    }

    Ok(())
}

/// Whether two edge curves of one type are the same curve, from their
/// parameters. `None` for a pair that has no such test: the blocks then
/// decide, in [`block_lies_on_block`].
fn curves_compatible(a: &EdgeCurve, b: &EdgeCurve, tol: Tolerance) -> Option<bool> {
    // Exhaustive match — no wildcards per CLAUDE.md convention.
    // Adding a new EdgeCurve variant will require updating this match.
    match (a, b) {
        (EdgeCurve::Line, EdgeCurve::Line) => Some(true),
        (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) => Some(
            (ca.radius() - cb.radius()).abs() < tol.linear
                && (ca.center() - cb.center()).length() < tol.linear
                && ca.normal().dot(cb.normal()).abs() > 1.0 - tol.angular,
        ),
        (EdgeCurve::Ellipse(ea), EdgeCurve::Ellipse(eb)) => Some(
            (ea.semi_major() - eb.semi_major()).abs() < tol.linear
                && (ea.semi_minor() - eb.semi_minor()).abs() < tol.linear
                && (ea.center() - eb.center()).length() < tol.linear
                && ea.normal().dot(eb.normal()).abs() > 1.0 - tol.angular,
        ),
        // A B-spline can be the same curve as another B-spline with other
        // poles, or as a line, a circle or an ellipse.
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
            EdgeCurve::Line
            | EdgeCurve::Circle(_)
            | EdgeCurve::Ellipse(_)
            | EdgeCurve::NurbsCurve(_),
        ) => None,
    }
}

/// Whether the parts of their edges' curves that two blocks cover are one
/// curve, within tolerance. The two blocks share their ends already.
///
/// Each block is read at the places of [`block_places`], and each place
/// must be within tolerance of its foot on the other block.
fn block_lies_on_block(
    topo: &Topology,
    arena: &GfaArena,
    pb_i: PaveBlockId,
    pb_j: PaveBlockId,
    tol: Tolerance,
) -> Result<bool, AlgoError> {
    let (Some(block_i), Some(block_j)) = (arena.pave_blocks.get(pb_i), arena.pave_blocks.get(pb_j))
    else {
        return Ok(false);
    };
    let read = |block: &crate::ds::PaveBlock| -> Result<BlockCurve<'_>, AlgoError> {
        let edge = topo.edge(block.original_edge)?;
        let ends = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let range = (block.start.parameter, block.end.parameter);
        let places = block_places(edge.curve(), range)
            .into_iter()
            .map(|t| (t, edge.curve().evaluate_with_endpoints(t, ends.0, ends.1)))
            .collect();
        Ok(BlockCurve {
            curve: edge.curve(),
            ends,
            range,
            places,
        })
    };
    let (i, j) = (read(block_i)?, read(block_j)?);
    let lies_on = |from: &BlockCurve<'_>, on: &BlockCurve<'_>| {
        from.places.iter().all(|&(_, p)| {
            super::helpers::foot_on_curve(p, on.curve, on.ends, on.range, &on.places).1
                <= tol.linear
        })
    };
    Ok(lies_on(&i, &j) && lies_on(&j, &i))
}

/// The part of an edge's curve that a pave block covers.
struct BlockCurve<'a> {
    /// The edge's curve.
    curve: &'a EdgeCurve,
    /// The edge's vertex positions.
    ends: (brepkit_math::vec::Point3, brepkit_math::vec::Point3),
    /// The block's parameters.
    range: (f64, f64),
    /// The places of [`block_places`], each with its parameter.
    places: Vec<(f64, brepkit_math::vec::Point3)>,
}

/// The parameters at which a block's part of its curve is read, in rising
/// order: 17 or more, the two ends included.
///
/// A B-spline is a different polynomial between each two knots, so a narrow
/// span can leave the rest of the curve and come back between two places
/// that are spread evenly. Each span between the knots inside the block
/// gets its own places, two more than the degree.
fn block_places(curve: &EdgeCurve, range: (f64, f64)) -> Vec<f64> {
    const LEAST: usize = 16;
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    let mut breaks = vec![lo];
    let per_span = if let EdgeCurve::NurbsCurve(nurbs) = curve {
        for &knot in nurbs.knots() {
            if knot > lo && knot < hi && breaks.last().is_some_and(|&last| knot > last) {
                breaks.push(knot);
            }
        }
        (nurbs.degree() + 3).max(LEAST.div_ceil(breaks.len()))
    } else {
        LEAST
    };
    breaks.push(hi);
    let mut places = Vec::with_capacity((breaks.len() - 1) * per_span + 1);
    for span in breaks.windows(2) {
        for k in 0..per_span {
            #[allow(clippy::cast_precision_loss)]
            places.push(span[0] + (span[1] - span[0]) * (k as f64 / per_span as f64));
        }
    }
    places.push(hi);
    places
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::Edge;
    use brepkit_topology::vertex::Vertex;

    fn cubic(poles: [Point3; 4]) -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            poles.to_vec(),
            vec![1.0; 4],
        )
        .unwrap()
    }

    /// An edge with its own two vertices and one pave block over all of it.
    fn whole_edge(
        topo: &mut Topology,
        arena: &mut GfaArena,
        start: Point3,
        end: Point3,
        curve: EdgeCurve,
    ) -> EdgeId {
        let tol = Tolerance::default().linear;
        let from = topo.add_vertex(Vertex::new(start, tol));
        let to = topo.add_vertex(Vertex::new(end, tol));
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        let edge = topo.add_edge(Edge::new(from, to, curve));
        arena.init_edge_pave_block(edge, from, t0, to, t1);
        edge
    }

    fn common_blocks(topo: &Topology, arena: &mut GfaArena) -> usize {
        perform(topo, Tolerance::default(), arena).unwrap();
        arena.common_blocks.iter().count()
    }

    /// Two edges on one cubic share their block, in the same direction and
    /// in opposite directions. A different cubic between the same two ends
    /// does not.
    #[test]
    fn blocks_on_one_b_spline_curve_are_common() {
        let poles = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.5),
            Point3::new(3.0, 2.0, -0.5),
            Point3::new(4.0, 0.0, 0.0),
        ];
        let mut back = poles;
        back.reverse();
        let mut lifted = poles;
        lifted[1] = Point3::new(1.0, 2.5, 0.5);
        for (other, common) in [(poles, 1), (back, 1), (lifted, 0)] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            whole_edge(
                &mut topo,
                &mut arena,
                poles[0],
                poles[3],
                EdgeCurve::NurbsCurve(cubic(poles)),
            );
            whole_edge(
                &mut topo,
                &mut arena,
                other[0],
                other[3],
                EdgeCurve::NurbsCurve(cubic(other)),
            );
            assert_eq!(common_blocks(&topo, &mut arena), common, "{other:?}");
        }
    }

    /// A degree-1 B-spline along a line, but for one narrow span where it
    /// goes a unit off the line and back. Places spread evenly over the
    /// block all lie on the line. The blocks are not common.
    #[test]
    fn a_narrow_span_off_the_other_curve_keeps_the_blocks_apart() {
        let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let spike = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.27, 0.275, 0.28, 1.0, 1.0],
            vec![
                at(0.0, 0.0),
                at(0.27, 0.0),
                at(0.275, 1.0),
                at(0.28, 0.0),
                at(1.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap();
        let straight = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![at(0.0, 0.0), at(1.0, 0.0)],
            vec![1.0; 2],
        )
        .unwrap();
        for other in [EdgeCurve::Line, EdgeCurve::NurbsCurve(straight)] {
            for spike_first in [true, false] {
                let mut topo = Topology::new();
                let mut arena = GfaArena::new();
                let mut curves = [EdgeCurve::NurbsCurve(spike.clone()), other.clone()];
                if !spike_first {
                    curves.reverse();
                }
                for curve in curves {
                    whole_edge(&mut topo, &mut arena, at(0.0, 0.0), at(1.0, 0.0), curve);
                }
                assert_eq!(common_blocks(&topo, &mut arena), 0);
            }
        }
    }

    /// A quarter circle and the same quarter as a rational quadratic share
    /// their block. The three quarters of the circle between the same two
    /// ends do not.
    #[test]
    fn an_arc_and_the_same_arc_as_a_b_spline_are_common() {
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let centre = Point3::new(1.0, -2.0, 0.5);
        let (start, end) = (centre + x * 2.0, centre + y * 2.0);
        let quarter = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![start, centre + x * 2.0 + y * 2.0, end],
            vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        )
        .unwrap();
        let circle = Circle3D::new_with_ref(centre, z, 2.0, x).unwrap();
        // From `end` round to `start`, the circle edge is the other three
        // quarters.
        for (from, to, common) in [(start, end, 1), (end, start, 0)] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            whole_edge(
                &mut topo,
                &mut arena,
                from,
                to,
                EdgeCurve::Circle(circle.clone()),
            );
            whole_edge(
                &mut topo,
                &mut arena,
                start,
                end,
                EdgeCurve::NurbsCurve(quarter.clone()),
            );
            assert_eq!(common_blocks(&topo, &mut arena), common);
        }
    }
}
