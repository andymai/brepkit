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
/// 2. They cover one span of one curve (one line, or one arc of one
///    circle: an arc and the rest of its circle share their ends)
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
                        let span = |pb: PaveBlockId| {
                            arena
                                .pave_blocks
                                .get(pb)
                                .map(crate::ds::PaveBlock::parameter_range)
                        };
                        let (Some(span_i), Some(span_j)) = (span(pb_i), span(pb_j)) else {
                            continue;
                        };
                        let compatible =
                            match curves_compatible(curve_i, span_i, curve_j, span_j, tol) {
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

/// Whether two blocks, over `span_a` of edge curve `a` and `span_b` of `b`
/// (each from its start's parameter to its end's), whose ends meet, are one
/// curve over one span, from the curves' parameters and the spans. `None`
/// for a pair that has no such test: the blocks then decide, in
/// [`block_lies_on_block`].
///
/// Two blocks on one circle or ellipse with the same ends cover either one
/// arc or the two arcs between those ends, so they are one only where
/// [`one_arc`] says so.
fn curves_compatible(
    a: &EdgeCurve,
    span_a: (f64, f64),
    b: &EdgeCurve,
    span_b: (f64, f64),
    tol: Tolerance,
) -> Option<bool> {
    // Exhaustive match — no wildcards per CLAUDE.md convention.
    // Adding a new EdgeCurve variant will require updating this match.
    match (a, b) {
        (EdgeCurve::Line, EdgeCurve::Line) => Some(true),
        (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) => Some(
            (ca.radius() - cb.radius()).abs() < tol.linear
                && (ca.center() - cb.center()).length() < tol.linear
                && ca.normal().dot(cb.normal()).abs() > 1.0 - tol.angular
                && one_arc(a, span_a, b, span_b, tol),
        ),
        (EdgeCurve::Ellipse(ea), EdgeCurve::Ellipse(eb)) => Some(
            (ea.semi_major() - eb.semi_major()).abs() < tol.linear
                && (ea.semi_minor() - eb.semi_minor()).abs() < tol.linear
                && (ea.center() - eb.center()).length() < tol.linear
                && ea.normal().dot(eb.normal()).abs() > 1.0 - tol.angular
                && one_arc(a, span_a, b, span_b, tol),
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

/// How far apart, in linear tolerances, [`arcs_apart`] may hold two blocks
/// that are one arc. Their ends meet within one tolerance, and the gates of
/// [`curves_compatible`] hold the curves' centres and sizes within one each,
/// so the bound reads under the ends' tolerance and twice the curves' offset.
const ARC_BAND: f64 = 6.0;

/// Whether the block over `span_a` of edge curve `a` and the block over
/// `span_b` of `b`, on circles or ellipses taken for one curve, are one arc:
/// where they are [`one_block_as_stored`], or [`arcs_apart`] holds them
/// within [`ARC_BAND`] tolerances along their whole length. Blocks on other
/// curves pass.
///
/// [`arcs_apart`] reads one curve onto the other, so its two readings of a
/// pair differ by rounding and by the curves' sizes. Each bounds the same
/// distance, so the lesser stands, and the answer does not hang on which
/// block comes first.
pub(super) fn one_arc(
    a: &EdgeCurve,
    span_a: (f64, f64),
    b: &EdgeCurve,
    span_b: (f64, f64),
    tol: Tolerance,
) -> bool {
    let conic = |c: &EdgeCurve| matches!(c, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_));
    if !(conic(a) && conic(b)) || one_block_as_stored(a, span_a, b, span_b) {
        return true;
    }
    let bound = match (
        arcs_apart(a, span_a, b, span_b),
        arcs_apart(b, span_b, a, span_a),
    ) {
        (Some(ab), Some(ba)) => Some(ab.min(ba)),
        (ab, ba) => ab.or(ba),
    };
    bound.is_some_and(|d| d <= ARC_BAND * tol.linear)
}

/// Whether `a` over `span_a` and `b` over `span_b` are one circle or one
/// ellipse over one span, either way round, in finite numbers alike to the
/// bit (a zero of either sign counts as one). Such copies are one block with
/// no bound to read: a bound can fail where the curve is large, far off, or
/// stored on axes no bound reads.
///
/// One symmetry of the stored frame is read exactly too: with the normal and
/// the second axis negated, the curve at `-t` is the first at `t` (cosine is
/// even and sine odd to the bit), so `b`'s span `(s, e)` is `a`'s `(-e, -s)`.
/// The other symmetries shift the parameter by a part of a turn, which
/// rounds, and are left to the bound.
fn one_block_as_stored(
    a: &EdgeCurve,
    span_a: (f64, f64),
    b: &EdgeCurve,
    span_b: (f64, f64),
) -> bool {
    use brepkit_math::vec::Vec3;
    let xyz = |v: Vec3| [v.x(), v.y(), v.z()];
    let data = |c: &EdgeCurve, reversed: bool| -> Option<Vec<f64>> {
        let (centre, normal, sizes, u, v) = match c {
            EdgeCurve::Circle(c) => (
                c.center(),
                c.normal(),
                vec![c.radius()],
                c.u_axis(),
                c.v_axis(),
            ),
            EdgeCurve::Ellipse(e) => (
                e.center(),
                e.normal(),
                vec![e.semi_major(), e.semi_minor()],
                e.u_axis(),
                e.v_axis(),
            ),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
        };
        let (normal, v) = if reversed { (-normal, -v) } else { (normal, v) };
        let mut all = vec![centre.x(), centre.y(), centre.z()];
        all.extend(xyz(normal));
        all.extend(sizes);
        all.extend(xyz(u));
        all.extend(xyz(v));
        Some(all)
    };
    let alike = |x: &[f64], y: &[f64]| {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|(&p, &q)| p.is_finite() && (p + 0.0).to_bits() == (q + 0.0).to_bits())
    };
    let same_kind = matches!(
        (a, b),
        (EdgeCurve::Circle(_), EdgeCurve::Circle(_))
            | (EdgeCurve::Ellipse(_), EdgeCurve::Ellipse(_))
    );
    let (Some(da), Some(db), Some(back)) = (data(a, false), data(b, false), data(b, true)) else {
        return false;
    };
    let ours = [span_a.0, span_a.1];
    let one_span = |(s, e): (f64, f64)| alike(&ours, &[s, e]) || alike(&ours, &[e, s]);
    same_kind
        && ((alike(&da, &db) && one_span(span_b))
            || (alike(&da, &back) && one_span((-span_b.1, -span_b.0))))
}

/// A bound on how far apart the arc of circle or ellipse `a` over `span_a`
/// and the arc of `b` over `span_b` lie along their whole length, each read
/// on its form with unit, square axes ([`unit_form`]), that form's gap added.
///
/// Two whole turns are each the whole curve, so `a`'s is read from the
/// parameter of `b`'s start, found from the start's offset from `b`'s centre:
/// a point far from the origin rounds to more than a tolerance. A span is a
/// whole turn where its sweep reads a turn within the rounding of its ends;
/// the piece a sweep may lack, times the curve's size, is added. An arc
/// short of a turn by more keeps its own start. `None` for other curves, or
/// where no bound holds.
fn arcs_apart(a: &EdgeCurve, span_a: (f64, f64), b: &EdgeCurve, span_b: (f64, f64)) -> Option<f64> {
    use crate::builder::face_splitter::{circle_arcs_apart, ellipse_arcs_apart};
    use crate::builder::pcurve_compute::unit_conic_arc_gap;
    use brepkit_math::madd::Madd;
    use std::f64::consts::TAU;
    let (unit_a, gap_a) = unit_form(a)?;
    let (unit_b, gap_b) = unit_form(b)?;
    let sweep = |(t0, t1): (f64, f64)| (t0, t1 - t0);
    // A span stored as `(t0, t0 + TAU)` reads its sweep a turn within an
    // ulp or two of its larger end.
    let lacks = |(t0, t1): (f64, f64)| TAU - (t1 - t0).abs();
    let whole = |span: (f64, f64)| {
        lacks(span) <= 4.0 * f64::EPSILON * span.0.abs().max(span.1.abs()).max(TAU)
    };
    let relocate = whole(span_a) && whole(span_b);
    let short = if relocate {
        lacks(span_a).max(0.0) + lacks(span_b).max(0.0)
    } else {
        0.0
    };
    let (a0, a_sweep) = sweep(span_a);
    let from = |nearest: f64| {
        if relocate {
            (nearest, a_sweep)
        } else {
            (a0, a_sweep)
        }
    };
    let (sin, cos) = span_b.0.sin_cos();
    let bound = if let (EdgeCurve::Circle(p), EdgeCurve::Circle(q)) = (&*unit_a, &*unit_b) {
        let offset = q.u_axis() * cos + q.v_axis() * sin;
        let nearest = offset.dot(p.v_axis()).atan2(offset.dot(p.u_axis()));
        let size = p.radius().max(q.radius());
        size.madd(short, circle_arcs_apart(p, from(nearest), q, sweep(span_b)))
    } else if let (EdgeCurve::Ellipse(p), EdgeCurve::Ellipse(q)) = (&*unit_a, &*unit_b) {
        let offset = q.u_axis() * (q.semi_major() * cos) + q.v_axis() * (q.semi_minor() * sin);
        let nearest = (offset.dot(p.v_axis()) / p.semi_minor())
            .atan2(offset.dot(p.u_axis()) / p.semi_major());
        let size = p.semi_major().max(q.semi_major());
        size.madd(
            short,
            ellipse_arcs_apart(p, from(nearest), q, sweep(span_b)),
        )
    } else {
        return None;
    };
    let bound = bound + unit_conic_arc_gap(gap_a) + unit_conic_arc_gap(gap_b);
    (!bound.is_nan()).then_some(bound)
}

/// [`unit_conic`] of `curve`. An ellipse whose minor semi-axis reads longer
/// than its major on unit axes (a round ellipse on a second axis a hair over
/// a unit) has none; it is read as the ellipse with its minor semi-axis cut
/// to just under its major's reading, the cut times its second axis's length
/// added to the gap, as the two lie that far apart at each parameter.
///
/// [`unit_conic`]: crate::builder::pcurve_compute::unit_conic
fn unit_form(curve: &EdgeCurve) -> Option<(std::borrow::Cow<'_, EdgeCurve>, f64)> {
    use crate::builder::pcurve_compute::unit_conic;
    if let Some(form) = unit_conic(curve) {
        return Some(form);
    }
    let EdgeCurve::Ellipse(e) = curve else {
        return None;
    };
    let (lu, lv) = (e.u_axis().length(), e.v_axis().length());
    let minor = e.semi_major() * (lu / lv) * (1.0 - 8.0 * f64::EPSILON);
    if !(minor > 0.0 && minor < e.semi_minor()) {
        return None;
    }
    let cut = brepkit_math::curves::Ellipse3D::with_axes(
        e.center(),
        e.normal(),
        e.semi_major(),
        minor,
        e.u_axis(),
        e.v_axis(),
    )
    .ok()?;
    let moved = (e.semi_minor() - minor) * lv * (1.0 + 4.0 * f64::EPSILON);
    let cut = EdgeCurve::Ellipse(cut);
    let (form, gap) = unit_conic(&cut)?;
    Some((std::borrow::Cow::Owned(form.into_owned()), gap + moved))
}

/// A block read for [`block_lies_on_block`]: its curve, its span on that
/// curve, its two ends, and the widest gap at a join of the pieces of a cut
/// B-spline.
type BlockRead = Option<(
    EdgeCurve,
    (f64, f64),
    brepkit_math::vec::Point3,
    brepkit_math::vec::Point3,
    f64,
)>;

/// Whether the parts of their edges' curves that two blocks cover are one
/// curve, within tolerance. The two blocks share their ends already.
///
/// Each block is read over its own parameters, never back from its ends: a
/// B-spline cut to the block's span, so neither a short block nor half of a
/// closed curve reads as some other trim, and a circle or an ellipse over
/// the block's span, so a short arc whose ends lie close does not read as
/// a whole turn. The two must be proved one curve along their whole length
/// ([`crate::builder::face_splitter::conic_spans_one_curve`] where either
/// is a conic, else [`crate::builder::face_splitter::curves_one_curve`]):
/// two B-splines that agree at any number of places may part between them.
/// A pair no bound holds together is two curves. Where the pieces of a cut
/// B-spline part at a join, twice the widest gap counts against the
/// tolerance: the cut moves the curve by up to that gap.
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
    let read = |block: &crate::ds::PaveBlock| -> Result<BlockRead, AlgoError> {
        let edge = topo.edge(block.original_edge)?;
        let ends = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (from, to) = (block.start.parameter, block.end.parameter);
        let (from, to) = if from <= to { (from, to) } else { (to, from) };
        if let EdgeCurve::NurbsCurve(curve) = edge.curve() {
            let Some((piece, gap)) = b_spline_between(curve, from, to) else {
                return Ok(None);
            };
            let (d0, d1) = piece.domain();
            let (a, b) = (piece.evaluate(d0), piece.evaluate(d1));
            return Ok(Some((EdgeCurve::NurbsCurve(piece), (d0, d1), a, b, gap)));
        }
        let at = |t: f64| edge.curve().evaluate_with_endpoints(t, ends.0, ends.1);
        Ok(Some((
            edge.curve().clone(),
            (from, to),
            at(from),
            at(to),
            0.0,
        )))
    };
    let (Some(i), Some(j)) = (read(block_i)?, read(block_j)?) else {
        return Ok(false);
    };
    let room = tol.linear - 2.0 * (i.4 + j.4);
    if room.is_nan() || room <= 0.0 {
        return Ok(false);
    }
    if let Some(one) =
        crate::builder::face_splitter::conic_spans_one_curve((&i.0, i.1), (&j.0, j.1), room)
    {
        return Ok(one);
    }
    Ok(crate::builder::face_splitter::curves_one_curve(
        (&i.0, i.2, i.3),
        (&j.0, j.2, j.3),
        room,
    ))
}

/// B-spline `curve` cut to `[from, to]`, or `curve` itself where that is its
/// whole domain, with the widest gap at a join of its pieces; `None` where a
/// piece fails, the span is empty, or the cut crosses a break of the curve.
///
/// The cut is read span by span: the Bezier piece of each knot span the
/// block covers, from the curve's own de Boor points about its first control
/// point, joined end to start at knots of multiplicity one less than the
/// degree's. No knot is inserted, so no knot close to a cut merges with it
/// and drops a stretch between. At a knot of full multiplicity two pieces
/// need not meet; where they part by more than the rounding of the curve's
/// own size, the cut is refused, as joining them would change the curve.
/// A narrower gap is closed: the later piece starts where the earlier ends,
/// which moves the curve by up to the gap.
fn b_spline_between(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    from: f64,
    to: f64,
) -> Option<(brepkit_math::nurbs::curve::NurbsCurve, f64)> {
    use brepkit_math::vec::{Point3, Vec3};
    let (d0, d1) = curve.domain();
    if from.is_nan() || to.is_nan() || from < d0 || to > d1 || from >= to {
        return None;
    }
    if from.to_bits() == d0.to_bits() && to.to_bits() == d1.to_bits() {
        return Some((curve.clone(), 0.0));
    }
    let p = curve.degree();
    let &first = curve.control_points().first()?;
    let offset = Vec3::new(first.x(), first.y(), first.z());
    let local = brepkit_math::nurbs::curve::NurbsCurve::new(
        p,
        curve.knots().to_vec(),
        curve.control_points().iter().map(|&q| q - offset).collect(),
        curve.weights().to_vec(),
    )
    .ok()?;
    let size = local
        .control_points()
        .iter()
        .map(|q| q.x().abs().max(q.y().abs()).max(q.z().abs()))
        .fold(0.0f64, f64::max);
    let mut breaks = vec![from];
    for &k in curve.knots() {
        if k > from && k < to && breaks.last().is_some_and(|&last| k > last) {
            breaks.push(k);
        }
    }
    breaks.push(to);
    let (mut points, mut weights, mut knots) =
        (Vec::<Point3>::new(), Vec::<f64>::new(), vec![from; p + 1]);
    let mut widest = 0.0f64;
    for (n, span) in breaks.windows(2).enumerate() {
        let piece =
            brepkit_math::nurbs::decompose::curve_bezier_between(&local, span[0], span[1]).ok()?;
        // One scale of all the weights of a piece keeps its shape. It brings
        // the first weight of the piece to the last weight kept, so the one
        // pole at the join stands for both pieces.
        let factor = match weights.last() {
            Some(&last) if n > 0 => last / piece.first()?[3],
            _ => 1.0,
        };
        if !factor.is_finite() || factor <= 0.0 {
            return None;
        }
        for (k, q) in piece.iter().enumerate() {
            let weight = q[3] * factor;
            if !weight.is_finite() || weight <= 0.0 {
                return None;
            }
            let point = Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]);
            // A piece starts where the last ended, and that point is kept
            // once; a piece that starts elsewhere is a break.
            if n > 0 && k == 0 {
                let &end = points.last()?;
                let gap = crate::builder::pcurve_compute::bound_length(point - end);
                if gap.is_nan() || gap > 64.0 * f64::EPSILON * size {
                    return None;
                }
                widest = widest.max(gap);
                continue;
            }
            points.push(point);
            weights.push(weight);
        }
        let joint = if n + 2 == breaks.len() { p + 1 } else { p };
        knots.extend(std::iter::repeat_n(span[1], joint));
    }
    let cut = brepkit_math::nurbs::curve::NurbsCurve::new(
        p,
        knots,
        points.into_iter().map(|q| q + offset).collect(),
        weights,
    )
    .ok()?;
    Some((cut, widest))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::f64::consts::TAU;

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

    /// A quadratic B-spline over 32 even spans whose control points weave
    /// 0.2 either side of the line from (0, 0) to (32, 0), and the polyline
    /// through its points at its 33 knots: the two agree at each knot and
    /// part by 0.1 at each span's middle, so they do not share their block.
    /// A copy of the spline 2e-8 aside does.
    #[test]
    fn b_splines_that_agree_at_their_knots_and_part_between_are_not_common() {
        let mut points = vec![Point3::new(0.0, 0.0, 0.0)];
        for i in 1..=32_i32 {
            let side = if i % 2 == 0 { 0.2 } else { -0.2 };
            points.push(Point3::new(f64::from(i) - 0.5, side, 0.0));
        }
        points.push(Point3::new(32.0, 0.0, 0.0));
        let mut knots = vec![0.0, 0.0];
        knots.extend((0..=32).map(f64::from));
        knots.extend([32.0, 32.0]);
        let weave = NurbsCurve::new(2, knots.clone(), points.clone(), vec![1.0; 34]).unwrap();
        let at_knots: Vec<Point3> = (0..=32).map(|k| weave.evaluate(f64::from(k))).collect();
        let mut line_knots = vec![0.0];
        line_knots.extend((0..=32).map(f64::from));
        line_knots.push(32.0);
        let polyline = NurbsCurve::new(1, line_knots, at_knots, vec![1.0; 33]).unwrap();
        let aside = NurbsCurve::new(
            2,
            knots,
            points
                .iter()
                .map(|&p| p + Vec3::new(0.0, 0.0, 2e-8))
                .collect(),
            vec![1.0; 34],
        )
        .unwrap();
        let (start, end) = (Point3::new(0.0, 0.0, 0.0), Point3::new(32.0, 0.0, 0.0));
        for (other, common) in [(polyline, 0), (aside, 1)] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            whole_edge(
                &mut topo,
                &mut arena,
                start,
                end,
                EdgeCurve::NurbsCurve(weave.clone()),
            );
            whole_edge(
                &mut topo,
                &mut arena,
                start,
                end,
                EdgeCurve::NurbsCurve(other),
            );
            assert_eq!(common_blocks(&topo, &mut arena), common);
        }
    }

    /// An edge from `curve`'s point at `t0` to its point at `t1`, with its
    /// own two vertices and one pave block over `(t0, t1)`.
    fn span_edge(
        topo: &mut Topology,
        arena: &mut GfaArena,
        curve: &EdgeCurve,
        (t0, t1): (f64, f64),
    ) -> EdgeId {
        let tol = Tolerance::default().linear;
        let at = |t: f64| match curve {
            EdgeCurve::NurbsCurve(n) => n.evaluate(t),
            EdgeCurve::Circle(c) => c.evaluate(t),
            EdgeCurve::Ellipse(e) => e.evaluate(t),
            EdgeCurve::Line => Point3::new(t, 0.0, 0.0),
        };
        let from = topo.add_vertex(Vertex::new(at(t0), tol));
        let to = topo.add_vertex(Vertex::new(at(t1), tol));
        let edge = topo.add_edge(Edge::new(from, to, curve.clone()));
        arena.init_edge_pave_block(edge, from, t0, to, t1);
        edge
    }

    /// A closed cubic through (2, 0) round a square of half side 2: its
    /// halves over [0, 0.5] and [0.5, 1] share their two ends but are
    /// opposite sides of the loop, so they share no block.
    #[test]
    fn opposite_halves_of_a_closed_b_spline_are_not_common() {
        let loop_curve = EdgeCurve::NurbsCurve(
            NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
                vec![
                    Point3::new(2.0, 0.0, 0.0),
                    Point3::new(2.0, 2.0, 0.0),
                    Point3::new(-2.0, 2.0, 0.0),
                    Point3::new(-2.0, 0.0, 0.0),
                    Point3::new(-2.0, -2.0, 0.0),
                    Point3::new(2.0, -2.0, 0.0),
                    Point3::new(2.0, 0.0, 0.0),
                ],
                vec![1.0; 7],
            )
            .unwrap(),
        );
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        span_edge(&mut topo, &mut arena, &loop_curve, (0.0, 0.5));
        span_edge(&mut topo, &mut arena, &loop_curve, (0.5, 1.0));
        assert_eq!(common_blocks(&topo, &mut arena), 0);
    }

    /// Two cubics on eight even spans that differ only in their last
    /// control point, so they are one curve over their first spans: blocks
    /// of each over [0.3, 0.3 + 5e-7], a millionth of the domain, share
    /// their block, and blocks over their last span do not.
    #[test]
    fn short_blocks_of_curves_that_agree_there_are_common() {
        let poles: Vec<Point3> = (0..11)
            .map(|i| Point3::new(f64::from(i), (0.7 * f64::from(i)).sin(), 0.0))
            .collect();
        let mut moved = poles.clone();
        moved[10] = Point3::new(10.0, 3.0, 0.0);
        let mut knots = vec![0.0; 4];
        knots.extend((1..8).map(|k| f64::from(k) / 8.0));
        knots.extend([1.0; 4]);
        let cubic = |p: &[Point3]| {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(3, knots.clone(), p.to_vec(), vec![1.0; 11]).unwrap(),
            )
        };
        for (span, common) in [((0.3, 0.3 + 5e-7), 1), ((0.9, 0.95), 0)] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            span_edge(&mut topo, &mut arena, &cubic(&poles), span);
            span_edge(&mut topo, &mut arena, &cubic(&moved), span);
            assert_eq!(common_blocks(&topo, &mut arena), common, "{span:?}");
        }
    }

    /// The circle of radius 2 about (1, -2, 0.5) as three quarter turns
    /// from (3, -2, 0.5) to (1, -4, 0.5), and as three rational quadratic
    /// quarters, and as a whole turn against four: each pair shares its
    /// block.
    #[test]
    fn a_major_arc_and_a_whole_circle_share_their_blocks_with_their_b_splines() {
        let centre = Point3::new(1.0, -2.0, 0.5);
        let (x, z) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let circle = Circle3D::new_with_ref(centre, z, 2.0, x).unwrap();
        let at = |k: f64| circle.evaluate(k * std::f64::consts::FRAC_PI_2);
        let corner = |k: f64| {
            let (a, b) = (at(k) - centre, at(k + 1.0) - centre);
            centre + a + b
        };
        let quarters = |n: u32| {
            let mut points = vec![at(0.0)];
            let mut weights = vec![1.0];
            let mut knots = vec![0.0; 3];
            for k in 0..n {
                let k = f64::from(k);
                points.extend([corner(k), at(k + 1.0)]);
                weights.extend([std::f64::consts::FRAC_1_SQRT_2, 1.0]);
                knots.extend([k + 1.0, k + 1.0]);
            }
            knots.push(f64::from(n));
            NurbsCurve::new(2, knots, points, weights).unwrap()
        };
        for (turns, end) in [(3, at(3.0)), (4, at(0.0))] {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            whole_edge(
                &mut topo,
                &mut arena,
                at(0.0),
                end,
                EdgeCurve::Circle(circle.clone()),
            );
            whole_edge(
                &mut topo,
                &mut arena,
                at(0.0),
                end,
                EdgeCurve::NurbsCurve(quarters(turns)),
            );
            assert_eq!(common_blocks(&topo, &mut arena), 1, "{turns} quarters");
        }
    }

    /// A block of the unit circle over [0, 5e-10] and a block of the same
    /// circle as an ellipse of equal semi-axes over [5e-10, 2 pi]: their
    /// ends lie 5e-10 apart, but one is a sliver of arc and the other the
    /// rest of the turn, so they share no block.
    #[test]
    fn a_short_arc_and_the_rest_of_its_circle_are_not_common() {
        use brepkit_math::curves::Ellipse3D;
        let (o, x, y, z) = (
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let circle = EdgeCurve::Circle(Circle3D::new_with_ref(o, z, 1.0, x).unwrap());
        let round = EdgeCurve::Ellipse(Ellipse3D::with_axes(o, z, 1.0, 1.0, x, y).unwrap());
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        span_edge(&mut topo, &mut arena, &circle, (0.0, 5e-10));
        span_edge(&mut topo, &mut arena, &round, (5e-10, TAU));
        assert_eq!(common_blocks(&topo, &mut arena), 0);
    }

    /// Two quadratics on knots [a, a, a, 0.5, 0.5, 1, 1, 1], `a` 2^-48
    /// short of 0.5, one with the `y` control coordinates 224, -16, 0, 0, 0
    /// and one flat: over blocks from 2^-51 short of 0.5 to 1 they meet at
    /// both ends, but the first dips to y = -1 between its block's start and
    /// 0.5, so they share no block.
    #[test]
    fn blocks_that_part_just_past_a_knot_are_not_common() {
        let a = 0.5 - 2f64.powi(-48);
        let from = 0.5 - 2f64.powi(-51);
        let knots = vec![a, a, a, 0.5, 0.5, 1.0, 1.0, 1.0];
        let quadratic = |ys: [f64; 5]| {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    knots.clone(),
                    ys.iter()
                        .enumerate()
                        .map(|(i, &y)| Point3::new(f64::from(u8::try_from(i).unwrap()), y, 0.0))
                        .collect(),
                    vec![1.0; 5],
                )
                .unwrap(),
            )
        };
        let (dipped, flat) = (
            quadratic([224.0, -16.0, 0.0, 0.0, 0.0]),
            quadratic([0.0; 5]),
        );
        let at = |c: &EdgeCurve, t: f64| match c {
            EdgeCurve::NurbsCurve(n) => n.evaluate(t),
            _ => unreachable!("a quadratic"),
        };
        assert!((at(&dipped, from) - at(&flat, from)).length() < 1e-9);
        let low = (0..=64)
            .map(|k| at(&dipped, from + (0.5 - from) * f64::from(k) / 64.0).y())
            .fold(f64::INFINITY, f64::min);
        assert!(low < -0.9, "the dip reaches {low}");
        let mut topo = Topology::new();
        let mut arena = GfaArena::new();
        span_edge(&mut topo, &mut arena, &dipped, (from, 1.0));
        span_edge(&mut topo, &mut arena, &flat, (from, 1.0));
        assert_eq!(common_blocks(&topo, &mut arena), 0);
    }

    /// A linear spline on knots [0, 0, 1, 1, 2, 3, 3] through (0, 0), (1, 0),
    /// (1, 1), (2, 0) and (3, 0) breaks at its knot 1, its piece over [1, 2]
    /// starting at (1, 1): over the block [0.5, 2.5] it reaches y = 0.75 at
    /// 1.25, so it shares no block with the segment from (0.5, 0) to
    /// (2.5, 0), upright, turned and mirrored.
    #[test]
    fn a_block_across_a_break_of_its_spline_is_not_a_segment() {
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let o = Point3::new(3.0, -2.0, 5.0);
        let poses: [(&str, Point3, Vec3, Vec3); 3] = [
            (
                "upright",
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                "turned",
                o,
                turn(Vec3::new(1.0, 0.0, 0.0)),
                turn(Vec3::new(0.0, 1.0, 0.0)),
            ),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                {
                    let u = turn(Vec3::new(1.0, 0.0, 0.0));
                    Vec3::new(-u.x(), u.y(), u.z())
                },
                {
                    let v = turn(Vec3::new(0.0, 1.0, 0.0));
                    Vec3::new(-v.x(), v.y(), v.z())
                },
            ),
        ];
        for (name, origin, u, v) in poses {
            let at = |x: f64, y: f64| origin + u * x + v * y;
            let broken = EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0],
                    vec![
                        at(0.0, 0.0),
                        at(1.0, 0.0),
                        at(1.0, 1.0),
                        at(2.0, 0.0),
                        at(3.0, 0.0),
                    ],
                    vec![1.0; 5],
                )
                .unwrap(),
            );
            let segment = EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    1,
                    vec![0.5, 0.5, 2.5, 2.5],
                    vec![at(0.5, 0.0), at(2.5, 0.0)],
                    vec![1.0; 2],
                )
                .unwrap(),
            );
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            span_edge(&mut topo, &mut arena, &broken, (0.5, 2.5));
            span_edge(&mut topo, &mut arena, &segment, (0.5, 2.5));
            assert_eq!(common_blocks(&topo, &mut arena), 0, "{name}");
        }
    }

    /// A polyline of degree one whose last pole lies 1e9 off, so the joins
    /// of its pieces may part by up to the rounding of that size, cut over
    /// [0.5, 1.5] across a full knot at 1 where its pieces part by `gap`,
    /// against the polyline of one piece fewer through the cut's own poles.
    /// Pieces that part by 1.5e-7 put the two that far apart at the join,
    /// past the tolerance of 1e-7, so they share no block. Pieces that part
    /// by 1e-12 share it. Upright, turned and mirrored.
    #[test]
    fn a_join_gap_of_a_cut_b_spline_counts_against_the_tolerance() {
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let o = Point3::new(3.0, -2.0, 5.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let poses: [(&str, Point3, Vec3, Vec3); 3] = [
            ("upright", Point3::new(0.0, 0.0, 0.0), x, y),
            ("turned", o, turn(x), turn(y)),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                mirror(turn(x)),
                mirror(turn(y)),
            ),
        ];
        for (name, origin, u, v) in poses {
            let at = |a: f64, b: f64| origin + u * a + v * b;
            for (gap, common) in [(1.5e-7, 0), (1e-12, 1)] {
                let parted = NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0],
                    vec![
                        at(0.0, 0.0),
                        at(1.0, 0.0),
                        at(1.0, gap),
                        at(2.0, gap),
                        at(1e9, 0.0),
                    ],
                    vec![1.0; 5],
                )
                .unwrap();
                let joined = NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
                    vec![at(0.0, 0.0), at(1.0, 0.0), at(2.0, 2.0 * gap), at(1e9, 0.0)],
                    vec![1.0; 4],
                )
                .unwrap();
                let mut topo = Topology::new();
                let mut arena = GfaArena::new();
                span_edge(
                    &mut topo,
                    &mut arena,
                    &EdgeCurve::NurbsCurve(parted),
                    (0.5, 1.5),
                );
                span_edge(
                    &mut topo,
                    &mut arena,
                    &EdgeCurve::NurbsCurve(joined),
                    (0.5, 1.5),
                );
                assert_eq!(
                    common_blocks(&topo, &mut arena),
                    common,
                    "{name}, gap {gap}"
                );
            }
        }
    }

    /// The unit half circle as two rational quadratic quarters on knots
    /// [0, 0, 0, 1, 1, 1, 2, 2, 2], the second quarter's weights three times
    /// the first's, so the quarters meet at (0, 1) with weights 1 and 3.
    /// Over [0.5, 1.5] it shares its block with its own copy. It shares no
    /// block with the spline that joins its two pieces end to start and
    /// keeps weight 1 at the join, which bends the second piece off the
    /// circle. Upright, turned and mirrored.
    #[test]
    fn a_block_across_a_join_of_unequal_weights_keeps_its_shape() {
        use brepkit_math::nurbs::decompose::curve_bezier_between;
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let o = Point3::new(3.0, -2.0, 5.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let poses: [(&str, Point3, Vec3, Vec3); 3] = [
            ("upright", Point3::new(0.0, 0.0, 0.0), x, y),
            ("turned", o, turn(x), turn(y)),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                mirror(turn(x)),
                mirror(turn(y)),
            ),
        ];
        let half = std::f64::consts::FRAC_1_SQRT_2;
        for (name, origin, u, v) in poses {
            let at = |a: f64, b: f64| origin + u * a + v * b;
            let round = NurbsCurve::new(
                2,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                vec![
                    at(1.0, 0.0),
                    at(1.0, 1.0),
                    at(0.0, 1.0),
                    at(0.0, 1.0),
                    at(-1.0, 1.0),
                    at(-1.0, 0.0),
                ],
                vec![1.0, half, 1.0, 3.0, 3.0 * half, 3.0],
            )
            .unwrap();
            let pieces = [
                curve_bezier_between(&round, 0.5, 1.0).unwrap(),
                curve_bezier_between(&round, 1.0, 1.5).unwrap(),
            ];
            let homogeneous: Vec<[f64; 4]> = pieces[0]
                .iter()
                .chain(pieces[1].iter().skip(1))
                .copied()
                .collect();
            let bent = NurbsCurve::new(
                2,
                vec![0.5, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 1.5],
                homogeneous
                    .iter()
                    .map(|q| Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]))
                    .collect(),
                homogeneous.iter().map(|q| q[3]).collect(),
            )
            .unwrap();
            assert!(
                (bent.evaluate(1.25) - round.evaluate(1.25)).length() > 0.01,
                "{name}: the end-to-start join bends the second piece"
            );
            let round = EdgeCurve::NurbsCurve(round);
            let bent = EdgeCurve::NurbsCurve(bent);
            for (other, common) in [(&round, 1), (&bent, 0)] {
                let mut topo = Topology::new();
                let mut arena = GfaArena::new();
                span_edge(&mut topo, &mut arena, &round, (0.5, 1.5));
                span_edge(&mut topo, &mut arena, other, (0.5, 1.5));
                assert_eq!(common_blocks(&topo, &mut arena), common, "{name}");
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

    /// The frames the arcs below lie in: upright, turned about a slant axis
    /// and moved, and that turn mirrored in the plane `x = 0`. Each is its
    /// origin and two square unit axes.
    fn frames() -> [(&'static str, Point3, Vec3, Vec3); 3] {
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let (u, v) = (
            turn(Vec3::new(1.0, 0.0, 0.0)),
            turn(Vec3::new(0.0, 1.0, 0.0)),
        );
        let o = Point3::new(3.0, -2.0, 5.0);
        [
            (
                "upright",
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            ("turned", o, u, v),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                mirror(u),
                mirror(v),
            ),
        ]
    }

    /// Two whole edges between one pair of points on one circle of radius 2,
    /// in each frame: the arc of a sixth of a turn and the rest of the
    /// circle, stored from the arc's end or from its start about the reversed
    /// normal, and the two halves of the circle, are two curves. The arc and
    /// its copy, stored either way round, are one.
    #[test]
    fn an_arc_and_the_rest_of_its_circle_are_not_common() {
        use std::f64::consts::PI;
        for (frame, o, u, v) in frames() {
            let normal = u.cross(v);
            let centre = o + u * 5.0 + v * 5.0;
            let up = Circle3D::with_axes(centre, normal, 2.0, u, v).unwrap();
            let down = Circle3D::with_axes(centre, -normal, 2.0, u, -v).unwrap();
            let at = |t: f64| up.evaluate(t);
            let (p, q, opposite) = (at(0.2), at(0.2 + PI / 3.0), at(0.2 + PI));
            let cases = [
                ("the rest from the arc's end", (p, q, &up), (q, p, &up), 0),
                (
                    "the rest about the reversed normal",
                    (p, q, &up),
                    (p, q, &down),
                    0,
                ),
                ("the two halves", (p, opposite, &up), (opposite, p, &up), 0),
                ("a copy", (p, q, &up), (p, q, &up), 1),
                (
                    "a copy about the reversed normal",
                    (p, q, &up),
                    (q, p, &down),
                    1,
                ),
            ];
            for (what, first, second, common) in cases {
                let mut topo = Topology::new();
                let mut arena = GfaArena::new();
                for (from, to, circle) in [first, second] {
                    whole_edge(
                        &mut topo,
                        &mut arena,
                        from,
                        to,
                        EdgeCurve::Circle(circle.clone()),
                    );
                }
                assert_eq!(common_blocks(&topo, &mut arena), common, "{frame}: {what}");
            }
        }
    }

    /// An ellipse of semi-axes 3 and 1 in each frame, and the same ellipse
    /// turned a quarter in its plane: they cross at `(s, s)` and `(-s, s)`,
    /// `s = 3 / sqrt(10)`, and between those points the first runs through
    /// `(0, 1)` and the second through `(0, 3)`. Each arc and the first's
    /// rest are two curves; the first's arc and its copy are one.
    #[test]
    fn arcs_of_two_ellipses_or_of_one_ellipse_and_its_rest_are_not_common() {
        use brepkit_math::curves::Ellipse3D;
        for (frame, o, u, v) in frames() {
            let normal = u.cross(v);
            let first = Ellipse3D::with_axes(o, normal, 3.0, 1.0, u, v).unwrap();
            let turned = Ellipse3D::with_axes(o, normal, 3.0, 1.0, v, -u).unwrap();
            let s = 3.0 / 10.0f64.sqrt();
            let (p, q) = (o + (u + v) * s, o + (v - u) * s);
            for e in [&first, &turned] {
                for x in [p, q] {
                    assert!((e.evaluate(e.project(x)) - x).length() < 1e-12);
                }
            }
            let cases = [
                ("the turned ellipse", (p, q, &first), (p, q, &turned), 0),
                ("the rest", (p, q, &first), (q, p, &first), 0),
                ("a copy", (p, q, &first), (p, q, &first), 1),
            ];
            for (what, a, b, common) in cases {
                let mut topo = Topology::new();
                let mut arena = GfaArena::new();
                for (from, to, ellipse) in [a, b] {
                    whole_edge(
                        &mut topo,
                        &mut arena,
                        from,
                        to,
                        EdgeCurve::Ellipse(ellipse.clone()),
                    );
                }
                assert_eq!(common_blocks(&topo, &mut arena), common, "{frame}: {what}");
            }
        }
    }

    /// Whether blocks over `span_a` of `a` and `span_b` of `b` are one, by
    /// [`curves_compatible`] at the default tolerance.
    fn one(a: &EdgeCurve, span_a: (f64, f64), b: &EdgeCurve, span_b: (f64, f64)) -> bool {
        curves_compatible(a, span_a, b, span_b, Tolerance::default()) == Some(true)
    }

    /// An ellipse of semi-axes 1 and 1 whose second axis is a hair longer
    /// than a unit, `(1 + 2^-50) Y`: on unit axes its minor semi-axis reads
    /// longer than its major. Two copies over one span, either way round,
    /// are one block; the arc and the rest are not.
    #[test]
    fn copies_of_an_ellipse_no_bound_reads_share_their_block() {
        use brepkit_math::curves::Ellipse3D;
        let ellipse = EdgeCurve::Ellipse(
            Ellipse3D::with_axes(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                1.0,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0 + 2.0f64.powi(-50), 0.0),
            )
            .unwrap(),
        );
        assert!(one(&ellipse, (0.0, 1.0), &ellipse.clone(), (0.0, 1.0)));
        assert!(one(&ellipse, (0.0, 1.0), &ellipse.clone(), (1.0, 0.0)));
        assert!(!one(&ellipse, (0.0, 1.0), &ellipse.clone(), (1.0, TAU)));
    }

    /// Circles of radius 1e9: two copies over the span (0, 1), either way
    /// round, are one block. The arc and the rest of the circle, stored
    /// about either normal, are not.
    #[test]
    fn copies_of_a_huge_circle_share_their_block() {
        let centre = Point3::new(0.0, 0.0, 0.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let up = EdgeCurve::Circle(Circle3D::with_axes(centre, z, 1e9, x, y).unwrap());
        let down = EdgeCurve::Circle(Circle3D::with_axes(centre, -z, 1e9, x, -y).unwrap());
        assert!(one(&up, (0.0, 1.0), &up.clone(), (0.0, 1.0)));
        assert!(one(&up, (0.0, 1.0), &up.clone(), (1.0, 0.0)));
        assert!(!one(&up, (0.0, 1.0), &up.clone(), (1.0, TAU)));
        assert!(!one(&up, (0.0, 1.0), &down, (0.0, TAU - 1.0)));
    }

    /// A circle of radius 1 about (1e10, 1e10, 0) on the axes (0.6, 0.8, 0)
    /// and (-0.8, 0.6, 0). Two copies over a whole turn are one block, from
    /// one seam or from seams a radian apart; the arc and the rest are not.
    #[test]
    fn copies_of_a_circle_far_from_the_origin_share_their_block() {
        let circle = EdgeCurve::Circle(
            Circle3D::with_axes(
                Point3::new(1e10, 1e10, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                Vec3::new(0.6, 0.8, 0.0),
                Vec3::new(-0.8, 0.6, 0.0),
            )
            .unwrap(),
        );
        assert!(one(&circle, (0.0, TAU), &circle.clone(), (0.0, TAU)));
        assert!(one(&circle, (0.0, TAU), &circle.clone(), (1.0, 1.0 + TAU)));
        assert!(!one(&circle, (0.0, 1.0), &circle.clone(), (1.0, TAU)));
    }

    /// Ellipses of semi-axes 2 and 2 on the axes (X, Y) and (Y, -X): over
    /// (0, pi / 2) and (-pi / 2, 0) both run the quarter from 2 X to 2 Y, so
    /// they are one block. The second over (0, 3 pi / 2) is the rest.
    #[test]
    fn round_ellipses_on_turned_axes_share_their_block() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::FRAC_PI_2;
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let centre = Point3::new(0.0, 0.0, 0.0);
        let first = EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, z, 2.0, 2.0, x, y).unwrap());
        let turned = EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, z, 2.0, 2.0, y, -x).unwrap());
        assert!(one(&first, (0.0, FRAC_PI_2), &turned, (-FRAC_PI_2, 0.0)));
        assert!(!one(
            &first,
            (0.0, FRAC_PI_2),
            &turned,
            (0.0, 3.0 * FRAC_PI_2)
        ));
    }

    /// The ellipse of [`copies_of_an_ellipse_no_bound_reads_share_their_block`]
    /// and the circle of radius 1e9, each stored again with its normal and
    /// second axis negated: the copy at `-t` is the curve at `t`, so the
    /// copy over (-1, 0) traces the curve's arc over (0, 1) backwards, and
    /// they are one block. The copy over (0, 2 pi - 1) is the rest.
    #[test]
    fn copies_stored_about_the_reversed_normal_share_their_block() {
        use brepkit_math::curves::Ellipse3D;
        let centre = Point3::new(0.0, 0.0, 0.0);
        let (x, z) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let y = Vec3::new(0.0, 1.0 + 2.0f64.powi(-50), 0.0);
        let ellipse = |n: Vec3, v: Vec3| {
            EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, n, 1.0, 1.0, x, v).unwrap())
        };
        let unit_y = Vec3::new(0.0, 1.0, 0.0);
        let circle = |n: Vec3, v: Vec3| {
            EdgeCurve::Circle(Circle3D::with_axes(centre, n, 1e9, x, v).unwrap())
        };
        for (what, up, down) in [
            ("ellipse", ellipse(z, y), ellipse(-z, -y)),
            ("circle", circle(z, unit_y), circle(-z, -unit_y)),
        ] {
            let at = |c: &EdgeCurve, t: f64| match c {
                EdgeCurve::Circle(c) => c.evaluate(t),
                EdgeCurve::Ellipse(e) => e.evaluate(t),
                EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
            };
            for t in [0.0, 0.3, 1.0, 2.5, -4.0] {
                let (p, q) = (at(&up, t), at(&down, -t));
                let bits = |p: Point3| [p.x(), p.y(), p.z()].map(|x| (x + 0.0).to_bits());
                assert_eq!(bits(p), bits(q), "{what} at {t}");
            }
            assert!(one(&up, (0.0, 1.0), &down, (-1.0, 0.0)), "{what}");
            assert!(one(&up, (0.0, 1.0), &down, (0.0, -1.0)), "{what}");
            assert!(!one(&up, (0.0, 1.0), &down, (0.0, TAU - 1.0)), "{what}");
            assert!(!one(&up, (0.0, 1.0), &down, (1.0, 0.0)), "{what}");
        }
    }

    /// A circle of radius 1e4 on the axes (X, Y), and the same circle on
    /// (X, -Y) about the reversed normal. Over (0, 2 pi - d) and (d, 2 pi),
    /// `d = 5e-10`, they run one arc short of a whole turn, opposite ways,
    /// and are one block. The first and the second over (0, d), the rest of
    /// the turn, are not.
    #[test]
    fn an_arc_just_short_of_a_whole_turn_keeps_its_own_start() {
        let centre = Point3::new(0.0, 0.0, 0.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let up = EdgeCurve::Circle(Circle3D::with_axes(centre, z, 1e4, x, y).unwrap());
        let down = EdgeCurve::Circle(Circle3D::with_axes(centre, -z, 1e4, x, -y).unwrap());
        let d = 5e-10;
        assert!(one(&up, (0.0, TAU - d), &down, (d, TAU)));
        assert!(!one(&up, (0.0, TAU - d), &down, (0.0, d)));
    }

    /// Ellipses of semi-axes 2 and 1, and 1 and 1 on a second axis a hair
    /// over a unit, about a centre near the origin and one 1e4 off, in each
    /// frame. An arc over a radian from each of 20 starts, and the same arc
    /// stored about the reversed normal (normal and second axis negated) from
    /// its end, with each span read from its ends as the pave filler reads
    /// it, are one block. The arc and the rest of the ellipse are not.
    #[test]
    fn an_ellipse_arc_and_its_copy_about_the_reversed_normal_share_their_block() {
        use brepkit_math::curves::Ellipse3D;
        for (frame, o, u, v) in frames() {
            let n = u.cross(v);
            for (major, minor, second) in [(2.0, 1.0, v), (1.0, 1.0, v * (1.0 + 2.0f64.powi(-50)))]
            {
                for shift in [0.0, 1e4] {
                    let centre = o + u * shift + v * (0.5 * shift);
                    let up = Ellipse3D::with_axes(centre, n, major, minor, u, second).unwrap();
                    let down = Ellipse3D::with_axes(centre, -n, major, minor, u, -second).unwrap();
                    for k in 0..20 {
                        let a = 0.1 + 0.3 * f64::from(k);
                        let (p, q) = (up.evaluate(a), up.evaluate(a + 1.0));
                        for (rest, common) in [(false, 1), (true, 0)] {
                            let mut topo = Topology::new();
                            let mut arena = GfaArena::new();
                            let (from, to) = if rest { (p, q) } else { (q, p) };
                            whole_edge(&mut topo, &mut arena, p, q, EdgeCurve::Ellipse(up.clone()));
                            whole_edge(
                                &mut topo,
                                &mut arena,
                                from,
                                to,
                                EdgeCurve::Ellipse(down.clone()),
                            );
                            assert_eq!(
                                common_blocks(&topo, &mut arena),
                                common,
                                "{frame}, semi-axes {major} and {minor}, centre off {shift}, start {a}, rest {rest}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Circles of radius 1 and 1 + 9e-8 about one centre on one frame, over
    /// (0, 1) and (0, 1 + g): read from the first, the bound is
    /// `9e-8 + g`, and from the second `9e-8 + (1 + 9e-8) g`. At the `g`
    /// where the second reading first passes the band, the first is under
    /// it. Each reading bounds one Hausdorff distance, so the blocks are one
    /// either way round.
    #[test]
    fn the_arc_bound_reads_alike_either_way_round() {
        let centre = Point3::new(0.0, 0.0, 0.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let small = EdgeCurve::Circle(Circle3D::with_axes(centre, z, 1.0, x, y).unwrap());
        let large = EdgeCurve::Circle(Circle3D::with_axes(centre, z, 1.0 + 9e-8, x, y).unwrap());
        let band = ARC_BAND * Tolerance::default().linear;
        let read = |from: &EdgeCurve, f: (f64, f64), to: &EdgeCurve, t: (f64, f64)| {
            arcs_apart(from, f, to, t).unwrap()
        };
        // The least g whose reading from the larger circle passes the band.
        let (mut lo, mut hi) = (0.0_f64, 1e-6_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if read(&large, (0.0, 1.0 + mid), &small, (0.0, 1.0)) > band {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let g = hi;
        assert!(read(&small, (0.0, 1.0), &large, (0.0, 1.0 + g)) <= band);
        assert!(read(&large, (0.0, 1.0 + g), &small, (0.0, 1.0)) > band);
        assert!(one(&small, (0.0, 1.0), &large, (0.0, 1.0 + g)));
        assert!(one(&large, (0.0, 1.0 + g), &small, (0.0, 1.0)));
    }

    /// Four copies of an arc of a circle (stored from its start, from its end,
    /// and both about the reversed normal) and two of the rest, run 20 times,
    /// each with fresh hash seeds: every run groups the arc's four copies in
    /// one common block and the rest's two in another.
    #[test]
    fn copies_of_an_arc_group_alike_on_every_run() {
        use std::f64::consts::PI;
        let centre = Point3::new(1.0, -2.0, 0.5);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let up = Circle3D::with_axes(centre, z, 2.0, x, y).unwrap();
        let down = Circle3D::with_axes(centre, -z, 2.0, x, -y).unwrap();
        let (p, q) = (up.evaluate(0.3), up.evaluate(0.3 + PI / 3.0));
        let edges = [
            (p, q, &up),
            (q, p, &down),
            (p, q, &up),
            (q, p, &up),
            (q, p, &down),
            (p, q, &down),
        ];
        for run in 0..20 {
            let mut topo = Topology::new();
            let mut arena = GfaArena::new();
            let ids: Vec<EdgeId> = edges
                .iter()
                .map(|&(from, to, circle)| {
                    whole_edge(
                        &mut topo,
                        &mut arena,
                        from,
                        to,
                        EdgeCurve::Circle(circle.clone()),
                    )
                })
                .collect();
            perform(&topo, Tolerance::default(), &mut arena).unwrap();
            let mut groups: Vec<Vec<usize>> = arena
                .common_blocks
                .iter()
                .map(|(_, cb)| {
                    let mut members: Vec<usize> = cb
                        .pave_blocks
                        .iter()
                        .filter_map(|&pb| arena.pave_blocks.get(pb))
                        .filter_map(|pb| ids.iter().position(|&e| e == pb.original_edge))
                        .collect();
                    members.sort_unstable();
                    members
                })
                .collect();
            groups.sort();
            assert_eq!(groups, vec![vec![0, 1, 2, 4], vec![3, 5]], "run {run}");
        }
    }
}
