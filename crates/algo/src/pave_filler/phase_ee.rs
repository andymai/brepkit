//! Phase EE: Edge-edge intersection detection.
//!
//! Finds intersection points and overlapping segments between edges
//! from the two solids. Creates new vertices at crossing points and
//! adds extra paves to both edges.

use crate::ds::{GfaArena, Interference, Pave};
use brepkit_math::madd::Madd;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use super::helpers::{add_pave_to_edge, find_nearby_pave_vertex, foot_on_curve};
use crate::error::AlgoError;

/// Detect edge-edge intersections between the two solids.
///
/// For each `(ea, eb)` pair where `ea` belongs to `solid_a` and `eb` to
/// `solid_b`, find intersection points. When a crossing coincides with
/// an existing vertex, add paves to both edges. When no existing vertex
/// is near, record the interference for the later `MakeSplitEdges` phase.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let edges_a = brepkit_topology::explorer::solid_edges(topo, solid_a)?;
    let edges_b = brepkit_topology::explorer::solid_edges(topo, solid_b)?;
    check_edge_edge_pairs(topo, &edges_a, &edges_b, tol, arena)
}

/// Find where each edge of `edges_a` meets each of `edges_b`, paving both.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub(super) fn check_edge_edge_pairs(
    topo: &mut Topology,
    edges_a: &[EdgeId],
    edges_b: &[EdgeId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // Collect edge data up front to avoid repeated lookups
    let data_a = collect_edge_data(topo, edges_a)?;
    let data_b = collect_edge_data(topo, edges_b)?;

    for (idx_a, (&ea_id, ea_data)) in edges_a.iter().zip(data_a.iter()).enumerate() {
        for (idx_b, (&eb_id, eb_data)) in edges_b.iter().zip(data_b.iter()).enumerate() {
            // Avoid duplicates when both edges are in both solids
            if ea_id == eb_id {
                continue;
            }

            if !aabbs_overlap(
                &ea_data.bbox_min,
                &ea_data.bbox_max,
                &eb_data.bbox_min,
                &eb_data.bbox_max,
                tol.linear,
            ) {
                continue;
            }

            let crossings = find_edge_edge_crossings(topo, ea_id, ea_data, eb_id, eb_data, tol)?;

            for (t_a, t_b, point) in crossings {
                let existing = find_nearby_pave_vertex(topo, arena, point, tol);

                let vertex_id = if let Some(vid) = existing {
                    vid
                } else {
                    // The crossing is no closer to either edge's faces than
                    // those edges' own ends are.
                    let ball = [ea_id, eb_id]
                        .into_iter()
                        .filter_map(|e| topo.edge(e).ok())
                        .flat_map(|e| [e.start(), e.end()])
                        .filter_map(|v| topo.vertex(v).ok())
                        .map(Vertex::tolerance)
                        .fold(tol.linear, f64::max);
                    super::helpers::pave_vertex_within_tolerance(topo, arena, point, ball, tol)
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(point, ball)))
                };

                add_pave_to_edge(arena, topo, ea_id, Pave::new(vertex_id, t_a));
                add_pave_to_edge(arena, topo, eb_id, Pave::new(vertex_id, t_b));

                arena.interference.ee.push(Interference::EE {
                    e1: ea_id,
                    e2: eb_id,
                    new_vertex: Some(vertex_id),
                    common_pave_block: None,
                });

                log::debug!(
                    "EE: edges {ea_id:?}[{idx_a}] and {eb_id:?}[{idx_b}] cross at \
                     t_a={t_a:.6}, t_b={t_b:.6}",
                );
            }
        }
    }

    Ok(())
}

/// Pre-computed edge data for fast intersection checks.
struct EdgeData {
    /// Start vertex position.
    start_pos: Point3,
    /// End vertex position.
    end_pos: Point3,
    /// Start of parameter domain.
    t0: f64,
    /// End of parameter domain.
    t1: f64,
    /// Axis-aligned bounding box minimum corner.
    bbox_min: Point3,
    /// Axis-aligned bounding box maximum corner.
    bbox_max: Point3,
    /// The edge's sample points and chord sags, made for its first pair
    /// that samples them and kept for the others.
    sampled: std::cell::OnceCell<SampledEdgeChords>,
}

/// The [`SAMPLE_CHORDS`] chords of an edge: their `SAMPLE_CHORDS + 1` end
/// points, evenly spaced in the edge's parameter, and the span of the curve
/// over each chord ([`ChordSpan`]).
struct SampledEdgeChords {
    pts: Vec<SegmentEndpoint>,
    spans: Vec<ChordSpan>,
}

/// How many chords a sampled edge is read along.
const SAMPLE_CHORDS: usize = 32;

impl SampledEdgeChords {
    fn new(curve: &EdgeCurve, data: &EdgeData) -> Self {
        let n = SAMPLE_CHORDS;
        let pts: Vec<SegmentEndpoint> = (0..=n)
            .map(|i| {
                let t = data.t0 + (data.t1 - data.t0) * (i as f64 / n as f64);
                let pos = curve.evaluate_with_endpoints(t, data.start_pos, data.end_pos);
                SegmentEndpoint { t, pos }
            })
            .collect();
        let spans = pts
            .windows(2)
            .map(|w| ChordSpan::of(curve, w[0], w[1]))
            .collect();
        Self { pts, spans }
    }
}

/// Collect pre-computed data for a set of edges.
fn collect_edge_data(topo: &Topology, edges: &[EdgeId]) -> Result<Vec<EdgeData>, AlgoError> {
    let mut data = Vec::with_capacity(edges.len());
    for &eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        let span = super::helpers::edge_span_box(edge.curve(), start_pos, end_pos, 16);

        data.push(EdgeData {
            start_pos,
            end_pos,
            t0,
            t1,
            bbox_min: span.min,
            bbox_max: span.max,
            sampled: std::cell::OnceCell::new(),
        });
    }
    Ok(data)
}

/// Check if two AABBs overlap with tolerance padding.
fn aabbs_overlap(min_a: &Point3, max_a: &Point3, min_b: &Point3, max_b: &Point3, tol: f64) -> bool {
    min_a.x() <= max_b.x() + tol
        && max_a.x() >= min_b.x() - tol
        && min_a.y() <= max_b.y() + tol
        && max_a.y() >= min_b.y() - tol
        && min_a.z() <= max_b.z() + tol
        && max_a.z() >= min_b.z() - tol
}

/// Find crossing points between two edges.
///
/// Uses algebraic line-line intersection when both edges are lines,
/// and segment-pair sampling otherwise.
#[allow(clippy::too_many_lines)]
fn find_edge_edge_crossings(
    topo: &Topology,
    ea_id: EdgeId,
    ea: &EdgeData,
    eb_id: EdgeId,
    eb: &EdgeData,
    tol: Tolerance,
) -> Result<Vec<(f64, f64, Point3)>, AlgoError> {
    let edge_a = topo.edge(ea_id)?;
    let edge_b = topo.edge(eb_id)?;

    if matches!(edge_a.curve(), EdgeCurve::Line) && matches!(edge_b.curve(), EdgeCurve::Line) {
        return Ok(line_line_intersection(ea, eb, tol));
    }

    // Coincident circles (same center/axis/radius): the arcs overlap along
    // a shared curve, not at isolated points. Like collinear parallel lines
    // (which return no crossings above), the overlap is handled by VE paves
    // at arc endpoints plus ForceInterfEE common blocks — sampling here
    // would emit a spurious crossing at every sample pair along the arc.
    if let (EdgeCurve::Circle(ca), EdgeCurve::Circle(cb)) = (edge_a.curve(), edge_b.curve())
        && (ca.radius() - cb.radius()).abs() < tol.linear
        && (ca.center() - cb.center()).length() < tol.linear
        && ca.normal().dot(cb.normal()).abs() > 1.0 - tol.angular
    {
        return Ok(Vec::new());
    }

    // Exact line-vs-circle: a line edge crossing an arc edge is extremely
    // common (every analytic corner arc of a body vs every straight tool
    // edge). The closed-form segment-circle solve replaces the 32×32 = 1024
    // segment-pair samples below with at most two candidate roots, which is
    // the dominant EE cost on solids with many cylindrical-corner arcs.
    if let EdgeCurve::Circle(circle) = edge_b.curve()
        && matches!(edge_a.curve(), EdgeCurve::Line)
    {
        return Ok(line_circle_intersection(ea, eb, circle, tol, false));
    }
    if let EdgeCurve::Circle(circle) = edge_a.curve()
        && matches!(edge_b.curve(), EdgeCurve::Line)
    {
        return Ok(line_circle_intersection(eb, ea, circle, tol, true));
    }

    let n = SAMPLE_CHORDS;
    let mut crossings = Vec::new();

    let sampled_a = ea
        .sampled
        .get_or_init(|| SampledEdgeChords::new(edge_a.curve(), ea));
    let sampled_b = eb
        .sampled
        .get_or_init(|| SampledEdgeChords::new(edge_b.curve(), eb));
    let (pts_a, spans_a) = (sampled_a.pts.as_slice(), sampled_a.spans.as_slice());
    let (pts_b, spans_b) = (sampled_b.pts.as_slice(), sampled_b.spans.as_slice());

    let domain_a = (ea.t0.min(ea.t1) - tol.linear)..=(ea.t0.max(ea.t1) + tol.linear);
    let domain_b = (eb.t0.min(eb.t1) - tol.linear)..=(eb.t0.max(eb.t1) + tol.linear);

    // Two sample chords meet at a point the curves miss by the chords' sag;
    // that point disagrees with the exact crossing the EF and FF phases
    // compute on the same curves, and the two copies of one vertex then
    // bound a sliver edge.
    let mut matched: Vec<bool> = Vec::new();
    let mut read = |crossings: &mut Vec<(f64, f64, Point3)>, hit: &ChordApproach| {
        let refined = refine_on_curves(
            (edge_a.curve(), ea),
            (edge_b.curve(), eb),
            (hit.t_a, hit.t_b),
            tol.linear,
        );
        let (t_a, t_b, pt) = refined.unwrap_or((hit.t_a, hit.t_b, hit.point));
        let is_dup = crossings
            .iter()
            .any(|&(ct_a, ct_b, _): &(f64, f64, Point3)| {
                (t_a - ct_a).abs() < 1e-6 && (t_b - ct_b).abs() < 1e-6
            });
        if !is_dup {
            crossings.push((t_a, t_b, pt));
            matched.push(refined.is_some());
        }
    };

    let mut alongside: Vec<(usize, usize)> = Vec::new();
    let mut alongside_hits: Vec<ChordApproach> = Vec::new();
    let mut crossed: Vec<Option<f64>> = vec![None; n * n];
    let mut near: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (chord_a, chord_b) = ([&pts_a[i], &pts_a[i + 1]], [&pts_b[j], &pts_b[j + 1]]);
            // A chord whose curve has no bound on its sag is read by the
            // chords alone.
            let reach = Some(spans_a[i].sag + spans_b[j].sag).filter(|r| r.is_finite());
            // Quick distance check: a point the two curves share lies
            // within a chord's length and its sag of that chord's ends.
            let min_dist = (pts_a[i].pos - pts_b[j].pos)
                .length()
                .min((pts_a[i].pos - pts_b[j + 1].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j + 1].pos).length());

            let seg_len_a = (pts_a[i + 1].pos - pts_a[i].pos).length();
            let seg_len_b = (pts_b[j + 1].pos - pts_b[j].pos).length();

            if min_dist > seg_len_a + seg_len_b + reach.unwrap_or(0.0) + tol.linear {
                continue;
            }

            if let Some(hit) = closest_segment_pair(chord_a, chord_b, tol.linear)
                && domain_a.contains(&hit.t_a)
                && domain_b.contains(&hit.t_b)
            {
                if hit.parallel {
                    alongside.push((i, j));
                    alongside_hits.push(hit);
                } else {
                    crossed[i * n + j] = Some(hit.t_a);
                    read(&mut crossings, &hit);
                    // The two spans can hold another crossing beside it.
                    if reach.is_some_and(|r| r > 0.0) {
                        near.push((i, j));
                    }
                }
            } else if let Some(reach) = reach.filter(|&r| r > 0.0) {
                let (_, _, gap) = segment_distance(
                    (chord_a[0].pos, chord_a[1].pos),
                    (chord_b[0].pos, chord_b[1].pos),
                );
                if gap <= reach + tol.linear {
                    near.push((i, j));
                }
            }
        }
    }

    // A chord stands off its curve by up to its sag, so where the two curves
    // meet between sample points, the chords there pass within their two
    // sags of each other. Where the curves are not in one plane, such chords
    // can miss each other by more than the tolerance. Gauss-Newton on the
    // curves, from the chords' closest points, reads such a crossing.
    //
    // The curves cross at most once over two spans whose tangent cones share
    // no line: the line through two crossings would lie in both. Two spans
    // whose cones share a line are split in four, each half with its own
    // chord and sag, and searched the same way, down to `SPLIT_DEPTH` and
    // within `SPLIT_BUDGET` for the pair of edges. A crossing where the
    // curves run along each other, and stay together a quarter span each
    // way, ends its search: the stretch along one curve is read whole
    // below. A shallow crossing parts from the other curve in that quarter.
    let along = |t_a: f64, t_b: f64| {
        let da = edge_a
            .curve()
            .tangent_with_endpoints(t_a, ea.start_pos, ea.end_pos);
        let db = edge_b
            .curve()
            .tangent_with_endpoints(t_b, eb.start_pos, eb.end_pos);
        let cross = da.cross(db);
        cross.dot(cross) <= ALONG_SINE * ALONG_SINE * da.dot(da) * db.dot(db)
    };
    let samples_b: Vec<(f64, Point3)> = pts_b.iter().map(|p| (p.t, p.pos)).collect();
    let (lo_b, hi_b) = (eb.t0.min(eb.t1), eb.t0.max(eb.t1));
    let shared = |t_a: f64, span: &ChordSpan| {
        let quarter = 0.25 * (span.to.t - span.from.t).abs();
        let (lo_a, hi_a) = (ea.t0.min(ea.t1), ea.t0.max(ea.t1));
        [t_a - quarter, t_a + quarter].into_iter().all(|t| {
            let p = edge_a.curve().evaluate_with_endpoints(
                t.clamp(lo_a, hi_a),
                ea.start_pos,
                ea.end_pos,
            );
            let ends = (eb.start_pos, eb.end_pos);
            foot_on_curve(p, edge_b.curve(), ends, (lo_b, hi_b), &samples_b).1 <= tol.linear
        })
    };
    let mut budget = SPLIT_BUDGET;
    let mut stack: Vec<(usize, ChordSpan, ChordSpan, usize)> = near
        .into_iter()
        .rev()
        .map(|(i, j)| (i * n + j, spans_a[i], spans_b[j], 0))
        .collect();
    while let Some((cell, a, b, depth)) = stack.pop() {
        let (s, t, gap) = segment_distance((a.from.pos, a.to.pos), (b.from.pos, b.to.pos));
        if gap > a.sag + b.sag + tol.linear {
            continue;
        }
        let split = |stack: &mut Vec<_>, budget: &mut usize| {
            *budget -= 1;
            for half_a in a.halves(edge_a.curve(), ea) {
                for half_b in b.halves(edge_b.curve(), eb) {
                    stack.push((cell, half_a, half_b, depth + 1));
                }
            }
        };
        let can_split = depth < SPLIT_DEPTH && budget > 0;
        // Chords more than half their sags apart mostly part once halved,
        // and halving costs less than a search that finds nothing.
        if can_split && gap > 0.5 * (a.sag + b.sag) + tol.linear {
            split(&mut stack, &mut budget);
            continue;
        }
        let isolated = cones_apart(a.cone, b.cone);
        if isolated
            && crossings
                .iter()
                .any(|&(t_a, t_b, _)| a.holds(t_a) && b.holds(t_b))
        {
            continue;
        }
        let seed = (
            s.madd(a.to.t - a.from.t, a.from.t),
            t.madd(b.to.t - b.from.t, b.from.t),
        );
        let ends = match refine((edge_a.curve(), ea), (edge_b.curve(), eb), seed, tol.linear) {
            Refined::Met(t_a, t_b, point) => {
                crossed[cell] = Some(t_a);
                let at = ChordApproach {
                    t_a,
                    t_b,
                    point,
                    gap: 0.0,
                    parallel: false,
                };
                read(&mut crossings, &at);
                along(t_a, t_b) && shared(t_a, &a)
            }
            Refined::Along => true,
            Refined::Apart => false,
        };
        if !isolated && !ends && can_split {
            split(&mut stack, &mut budget);
        }
    }

    // Chords all but parallel and within tolerance form ribbons on the grid
    // of chord pairs. A ribbon stops where an edge ends on the other edge or
    // where the edges part at a corner (a crossing read above, next to the
    // ribbon). Where it stops at both of its ends, by two different stops,
    // the edges overlap. The VE phase paves those edge ends and the corner
    // is read already; a crossing read along the ribbon would split both
    // edges where nothing meets them. Else the edges come together and part
    // again while both go on, and the ribbon holds one crossing, at its
    // closest place. Places are read along edge A, as parameters of A.
    let on = |p: &SegmentEndpoint, chord: [&SegmentEndpoint; 2]| {
        closest_segment_pair([p, p], chord, tol.linear)
    };
    // The stops next to cell `(i, j)`, each as `(which, place)`: the ends of
    // A and of B are 0 to 3, a corner is 4 past its cell's index.
    let stops_in = |(i, j): (usize, usize), stops: &mut Vec<(usize, f64)>| {
        let chord_a = [&pts_a[i], &pts_a[i + 1]];
        let chord_b = [&pts_b[j], &pts_b[j + 1]];
        if i == 0 && on(&pts_a[0], chord_b).is_some() {
            stops.push((0, pts_a[0].t));
        }
        if i + 1 == n && on(&pts_a[n], chord_b).is_some() {
            stops.push((1, pts_a[n].t));
        }
        if j == 0
            && let Some(hit) = on(&pts_b[0], chord_a)
        {
            stops.push((2, hit.t_b));
        }
        if j + 1 == n
            && let Some(hit) = on(&pts_b[n], chord_a)
        {
            stops.push((3, hit.t_b));
        }
        for ni in i.saturating_sub(1)..=(i + 1).min(n - 1) {
            for nj in j.saturating_sub(1)..=(j + 1).min(n - 1) {
                if let Some(at) = crossed[ni * n + nj] {
                    stops.push((4 + ni * n + nj, at));
                }
            }
        }
    };
    // A stop lies at or past the end of the ribbon it stops. Where that
    // end's closest place is the stop itself, the two differ by rounding.
    let slack = 1e-9 * (ea.t1 - ea.t0).abs();
    for ribbon in ribbons(&alongside, n) {
        let (lo, hi) = ribbon
            .iter()
            .map(|&k| alongside_hits[k].t_a)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
                (lo.min(t), hi.max(t))
            });
        let mut stops = Vec::new();
        for &k in &ribbon {
            stops_in(alongside[k], &mut stops);
        }
        let overlap = stops.iter().any(|&(low, at_low)| {
            at_low <= lo + slack
                && stops
                    .iter()
                    .any(|&(high, at_high)| high != low && at_high >= hi - slack)
        });
        if overlap {
            continue;
        }
        if let Some(&k) = ribbon
            .iter()
            .min_by(|&&p, &&q| alongside_hits[p].gap.total_cmp(&alongside_hits[q].gap))
        {
            read(&mut crossings, &alongside_hits[k]);
        }
    }

    drop_crossings_along_one_curve(
        &mut crossings,
        &matched,
        (edge_a.curve(), ea, pts_a),
        (edge_b.curve(), eb, pts_b),
        tol.linear,
    );

    Ok(crossings)
}

/// The splits of pairs of spans one pair of edges may take, and how many
/// times one first pair of spans may be halved.
const SPLIT_BUDGET: usize = 64;
const SPLIT_DEPTH: usize = 8;

/// The span of an edge's curve between two of its points: a bound on how
/// far the curve stands off the chord between them (its sag), and a cone
/// that holds the direction of each of its tangents.
#[derive(Clone, Copy)]
struct ChordSpan {
    from: SegmentEndpoint,
    to: SegmentEndpoint,
    sag: f64,
    /// The cone's axis, the chord made unit and turned the way the
    /// parameter rises, and its half angle. `None` when no such cone is
    /// narrower than a right angle.
    cone: Option<(Vec3, f64)>,
}

impl ChordSpan {
    /// The span of `curve` from `from` to `to`.
    ///
    /// A line is its chord. An arc of radius `r` over `d` radians, up to a
    /// half turn, stands off by at most `r (1 - cos(d / 2))`, at its middle,
    /// and a longer arc by at most its diameter; its tangent turns by `d`,
    /// and the chord halves that turn. An ellipse arc over `d` radians stands
    /// off by at most `d^2 / 8` times the largest second derivative of the
    /// curve, which is its major radius; its tangent turns one way, so it
    /// stays between its two end tangents. A B-spline between two knots lies
    /// in the hull of its Bezier poles when their weights are positive, so
    /// its farthest pole bounds its sag there, and each of its tangents lies
    /// in the cone of the differences of its poles, each later pole less an
    /// earlier one. A piece with a weight that is not positive has no
    /// bound, and its chord gets an infinite sag.
    fn of(curve: &EdgeCurve, from: SegmentEndpoint, to: SegmentEndpoint) -> Self {
        use std::f64::consts::{FRAC_PI_2, PI};
        let turn = (to.t - from.t).abs();
        let rising = if from.t <= to.t {
            to.pos - from.pos
        } else {
            from.pos - to.pos
        };
        let axis = rising.normalize().ok();
        let (sag, half) = match curve {
            EdgeCurve::Line => (0.0, Some(0.0)),
            EdgeCurve::Circle(circle) if turn <= PI => {
                let half = (0.25 * turn).sin();
                (2.0 * circle.radius() * half * half, Some(0.5 * turn))
            }
            EdgeCurve::Circle(circle) => (2.0 * circle.radius(), None),
            EdgeCurve::Ellipse(ellipse) => {
                let half = axis.filter(|_| turn <= PI).map(|axis| {
                    let (lo, hi) = (from.t.min(to.t), from.t.max(to.t));
                    angle_from(axis, ellipse.tangent(lo)).max(angle_from(axis, ellipse.tangent(hi)))
                });
                (ellipse.semi_major() * turn * turn / 8.0, half)
            }
            EdgeCurve::NurbsCurve(nurbs) => nurbs_span_bounds(nurbs, (from, to), axis),
        };
        let cone = axis.zip(half).filter(|&(_, half)| half < FRAC_PI_2);
        Self {
            from,
            to,
            sag,
            cone,
        }
    }

    /// Whether the edge parameter `t` lies in the span.
    fn holds(&self, t: f64) -> bool {
        t >= self.from.t.min(self.to.t) && t <= self.from.t.max(self.to.t)
    }

    /// The span's two halves about the middle of its parameters.
    fn halves(&self, curve: &EdgeCurve, data: &EdgeData) -> [Self; 2] {
        let t = f64::midpoint(self.from.t, self.to.t);
        let pos = curve.evaluate_with_endpoints(t, data.start_pos, data.end_pos);
        let mid = SegmentEndpoint { t, pos };
        [
            Self::of(curve, self.from, mid),
            Self::of(curve, mid, self.to),
        ]
    }
}

/// The angle from the unit `axis` to the direction of `v`.
fn angle_from(axis: Vec3, v: Vec3) -> f64 {
    axis.cross(v).length().atan2(axis.dot(v))
}

/// The sag and the tangent cone's half angle about `axis` of a B-spline
/// between two of its points, from the Bezier poles of each piece between
/// its knots (see [`ChordSpan::of`]).
fn nurbs_span_bounds(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    (from, to): (SegmentEndpoint, SegmentEndpoint),
    axis: Option<Vec3>,
) -> (f64, Option<f64>) {
    let (d0, d1) = curve.domain();
    let (lo, hi) = (from.t.min(to.t).max(d0), from.t.max(to.t).min(d1));
    if lo > hi {
        return (f64::INFINITY, None);
    }
    let mut cuts = vec![lo];
    cuts.extend(curve.knots().iter().copied().filter(|&k| k > lo && k < hi));
    cuts.push(hi);
    cuts.dedup();
    // The curve can jump at a knot repeated past its degree, and no cone of
    // tangents spans a jump.
    let degree = curve.degree();
    let jumps = curve
        .knots()
        // By value: -0 and 0 are one knot.
        .chunk_by(|a, b| (a - b).abs() <= 0.0)
        .any(|run| run[0] > lo && run[0] <= hi && run.len() > degree);
    let axis = axis.filter(|_| !jumps);
    let (mut sag, mut half) = (0.0_f64, axis.map(|_| 0.0_f64));
    let mut poles: Vec<Point3> = Vec::new();
    for piece in cuts.windows(2) {
        let Ok(homogeneous) =
            brepkit_math::nurbs::decompose::curve_bezier_between(curve, piece[0], piece[1])
        else {
            return (f64::INFINITY, None);
        };
        poles.clear();
        for [x, y, z, w] in homogeneous {
            if w.is_nan() || w <= 0.0 {
                return (f64::INFINITY, None);
            }
            let pole = Point3::new(x / w, y / w, z / w);
            let off = distance_to_segment(pole, (from.pos, to.pos));
            if !off.is_finite() {
                return (f64::INFINITY, None);
            }
            sag = sag.max(off);
            poles.push(pole);
        }
        if let (Some(axis), Some(half)) = (axis, half.as_mut()) {
            for (k, &p) in poles.iter().enumerate() {
                for &q in &poles[k + 1..] {
                    let d = q - p;
                    if d.length() > 0.0 {
                        *half = half.max(angle_from(axis, d));
                    }
                }
            }
        }
    }
    (sag, half)
}

/// Whether two tangent cones share no line: the lines of their axes part by
/// more than the two half angles.
fn cones_apart(a: Option<(Vec3, f64)>, b: Option<(Vec3, f64)>) -> bool {
    let (Some((axis_a, half_a)), Some((axis_b, half_b))) = (a, b) else {
        return false;
    };
    let apart = axis_a
        .cross(axis_b)
        .length()
        .atan2(axis_a.dot(axis_b).abs());
    apart > half_a + half_b
}

/// The closest points of the segments `a` and `b`, as fractions `s` and `t`
/// along them, and their distance. The fraction on one segment is clamped
/// and the other one read again from it, so the distance is the least of
/// any two points of the segments, also where one end of a segment is
/// nearest the inside of the other.
fn segment_distance(a: (Point3, Point3), b: (Point3, Point3)) -> (f64, f64, f64) {
    let (da, db, r) = (a.1 - a.0, b.1 - b.0, a.0 - b.0);
    let (aa, bb, f) = (da.dot(da), db.dot(db), db.dot(r));
    let (s, t) = if aa <= 0.0 && bb <= 0.0 {
        (0.0, 0.0)
    } else if aa <= 0.0 {
        (0.0, (f / bb).clamp(0.0, 1.0))
    } else {
        let c = da.dot(r);
        if bb <= 0.0 {
            ((-c / aa).clamp(0.0, 1.0), 0.0)
        } else {
            let ab = da.dot(db);
            let denom = aa.madd(bb, -(ab * ab));
            let s = if denom > 0.0 {
                (ab.madd(f, -(c * bb)) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t = ab.madd(s, f) / bb;
            if t < 0.0 {
                ((-c / aa).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((ab - c) / aa).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    let gap = ((a.0 + da * s) - (b.0 + db * t)).length();
    (s, t, gap)
}

/// The distance from `p` to the segment from `a` to `b`.
fn distance_to_segment(p: Point3, (a, b): (Point3, Point3)) -> f64 {
    let along = b - a;
    let length2 = along.dot(along);
    let s = if length2 > 0.0 {
        ((p - a).dot(along) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + along * s)).length()
}

/// One edge of a pair: its curve, its data and its sample points.
type SampledEdge<'a> = (&'a EdgeCurve, &'a EdgeData, &'a [SegmentEndpoint]);

/// The parameters of edge A at which a walk along it reads it, in rising
/// order, its two ends left out.
///
/// The marks are a sample chord apart. Between two knots a B-spline is one
/// piece, and a narrow piece can leave the other curve and come back
/// between two such marks. So each piece of A narrower than two chords gets
/// marks of its own, and so does each piece of B, between the feet of its
/// knots on A.
fn walk_marks(a: SampledEdge<'_>, b: SampledEdge<'_>) -> Vec<f64> {
    let (lo, hi) = (a.1.t0.min(a.1.t1), a.1.t0.max(a.1.t1));
    let span = hi - lo;
    let step = span / 32.0;
    let mut breaks = vec![lo, hi];
    if let EdgeCurve::NurbsCurve(nurbs) = a.0 {
        breaks.extend(nurbs.knots().iter().copied().filter(|&k| k > lo && k < hi));
    }
    if let EdgeCurve::NurbsCurve(nurbs) = b.0 {
        let (lo_b, hi_b) = (b.1.t0.min(b.1.t1), b.1.t0.max(b.1.t1));
        let samples_a: Vec<(f64, Point3)> = a.2.iter().map(|p| (p.t, p.pos)).collect();
        for &knot in nurbs.knots() {
            if knot > lo_b && knot < hi_b {
                let at = nurbs.evaluate(knot);
                let ends = (a.1.start_pos, a.1.end_pos);
                breaks.push(foot_on_curve(at, a.0, ends, (lo, hi), &samples_a).0);
            }
        }
    }
    breaks.sort_by(f64::total_cmp);
    let mut marks: Vec<f64> = (1..32).map(|k| lo + step * f64::from(k)).collect();
    for piece in breaks.windows(2) {
        let width = piece[1] - piece[0];
        if width > 1e-12 * span && width < 2.0 * step {
            marks.extend((1..4).map(|k| piece[0] + width * (f64::from(k) / 4.0)));
        }
        marks.push(piece[1]);
    }
    marks.retain(|&t| t > lo && t < hi);
    marks.sort_by(f64::total_cmp);
    marks.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * span);
    marks
}

/// How a stretch of two edges along each other ends.
#[derive(Clone, Copy, PartialEq)]
enum StretchEnd {
    /// One of the two edges ends there, on the other.
    EdgeEnd,
    /// The curves part there at a real angle: the sine of it.
    Corner(f64),
    /// The curves draw apart while both go on.
    Open,
    /// One edge leaves the curve that the two share, with no corner, at
    /// this parameter of A.
    Junction(f64),
}

/// A stretch of edge A within tolerance of edge B, as parameters of A.
struct Stretch {
    /// The two ends of the stretch.
    span: (f64, f64),
    /// How far past each end a crossing still counts as read in it.
    slack: (f64, f64),
    /// The stretch is an overlap: each end is an edge end, a corner or a
    /// junction.
    overlap: bool,
    /// The crossing read in the stretch where the curves are nearest, and
    /// their distance there.
    nearest: (usize, f64),
}

/// Below this sine of the angle between two curves at a crossing, the
/// crossing can be a place where the curves run along each other.
const ALONG_SINE: f64 = 1e-4;
/// From this sine of the angle at which two curves part, the place is a
/// corner.
const CORNER_SINE: f64 = 1e-3;
/// The part of a stretch that must stay at a thousandth of the tolerance
/// for the two edges to share a curve along it. A contact of order `n`
/// keeps `0.001^(1/n)` of its stretch: 0.03 for a tangency, 0.5 for order
/// 10. Edges that share a curve keep all of it but the last of the run-out
/// where one leaves.
const SHARED_PART: f64 = 0.8;

/// Drop the crossings read inside a stretch where the two edges lie on one
/// curve.
///
/// Sample chords of two edges on one curved line are chords of that curve
/// between different points. They meet at each sample point that the edges
/// share and cross between the others, at a real angle, although the curves
/// do not cross. So the decision is made on the curves.
///
/// A crossing where the curves meet at a real angle stays. From each other
/// crossing, edge A is followed both ways for as long as it stays within
/// `tol` of B. That stretch is an overlap when each of its ends is an end of
/// one of the edges, or a corner where the curves part at a real angle. The
/// crossings in an overlap are dropped: the VE phase paves the edge ends,
/// and a corner gets one crossing of its own.
///
/// Where an end of the stretch is open, the edges come together and part
/// while both go on. A tangency and a grazing crossing are such places.
/// Such a stretch is one contact, and keeps the one crossing where the
/// curves are nearest. It is as long as the tolerance makes it. Where the stretch is almost as long at a thousandth
/// of the tolerance ([`SHARED_PART`] of it or more), the edges share a curve
/// there, and one of them leaves
/// it with no corner. That end is a junction: it ends an overlap as a corner
/// does, and gets one crossing.
///
/// `matched[k]` tells that crossing `k` has the same point on both curves.
#[allow(clippy::too_many_lines)]
fn drop_crossings_along_one_curve(
    crossings: &mut Vec<(f64, f64, Point3)>,
    matched: &[bool],
    a: SampledEdge<'_>,
    b: SampledEdge<'_>,
    tol: f64,
) {
    let (lo, hi) = (a.1.t0.min(a.1.t1), a.1.t0.max(a.1.t1));
    let (lo_b, hi_b) = (b.1.t0.min(b.1.t1), b.1.t0.max(b.1.t1));
    let (span, span_b) = (hi - lo, hi_b - lo_b);
    if crossings.is_empty() || span <= 0.0 || span_b <= 0.0 {
        return;
    }
    let at_a = |t: f64| a.0.evaluate_with_endpoints(t, a.1.start_pos, a.1.end_pos);
    let samples_b: Vec<(f64, Point3)> = b.2.iter().map(|p| (p.t, p.pos)).collect();
    let foot = |t: f64| {
        foot_on_curve(
            at_a(t),
            b.0,
            (b.1.start_pos, b.1.end_pos),
            (lo_b, hi_b),
            &samples_b,
        )
    };
    let on_b = |t: f64| foot(t).1 <= tol;
    let at_b = |t: f64| b.0.evaluate_with_endpoints(t, b.1.start_pos, b.1.end_pos);
    // B ends at a place on it that is within tolerance of one of its ends.
    // A part of B's parameter span is not such a measure: on a long edge it
    // is far more than the tolerance.
    let ends_b = [at_b(lo_b), at_b(hi_b)];
    let b_ends_at = |t_b: f64| {
        let p = at_b(t_b);
        ends_b.iter().any(|&end| (p - end).length() <= tol)
    };

    // The crossings where the curves meet at a real angle.
    let real: Vec<bool> = crossings
        .iter()
        .zip(matched)
        .map(|(&(t_a, t_b, _), &matched)| {
            let t_b = if matched {
                t_b
            } else {
                let (foot_b, gap) = foot(t_a);
                if gap > tol {
                    // Not on B at all: nothing here says the edges overlap.
                    return true;
                }
                foot_b
            };
            let dir_a = a.0.tangent_with_endpoints(t_a, a.1.start_pos, a.1.end_pos);
            let dir_b = b.0.tangent_with_endpoints(t_b, b.1.start_pos, b.1.end_pos);
            let cross = dir_a.cross(dir_b);
            cross.dot(cross) > ALONG_SINE * ALONG_SINE * dir_a.dot(dir_a) * dir_b.dot(dir_b)
        })
        .collect();
    if real.iter().all(|&r| r) {
        return;
    }

    // A sample chord of A.
    let step = span / 32.0;
    let probe = span * 1e-5;
    // Two crossings can lie closer than a mark apart, with the curves
    // parting between them. A mark halfway between each two keeps the walk
    // from one from running on past the other.
    let mut marks = walk_marks(a, b);
    let mut places: Vec<f64> = crossings.iter().map(|c| c.0.clamp(lo, hi)).collect();
    places.sort_by(f64::total_cmp);
    marks.extend(places.windows(2).map(|w| f64::midpoint(w[0], w[1])));
    marks.retain(|&t| t > lo && t < hi);
    marks.sort_by(f64::total_cmp);
    marks.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * span);
    // Follow A from `from`, which is on B, towards `side` (-1 or 1) to the
    // end of the stretch, from one mark to the next.
    let follow = |from: f64, side: f64| -> (f64, StretchEnd) {
        let end = if side < 0.0 { lo } else { hi };
        let mut t = from;
        let off = loop {
            let next = if side > 0.0 {
                marks.get(marks.partition_point(|&m| m <= t)).copied()
            } else {
                marks
                    .partition_point(|&m| m < t)
                    .checked_sub(1)
                    .map(|i| marks[i])
            };
            let Some(next) = next else {
                if on_b(end) {
                    return (end, StretchEnd::EdgeEnd);
                }
                break end;
            };
            if !on_b(next) {
                break next;
            }
            t = next;
        };
        let (mut inside, mut outside) = (t, off);
        for _ in 0..26 {
            let mid = f64::midpoint(inside, outside);
            if on_b(mid) {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        let (t_b, gap) = foot(inside);
        let past = (inside + side * probe).clamp(lo, hi);
        let (t_b_past, gap_past) = foot(past);
        if b_ends_at(t_b) || b_ends_at(t_b_past) {
            return (inside, StretchEnd::EdgeEnd);
        }
        let run = (at_a(past) - at_a(inside)).length();
        let sine = if run > 0.0 {
            (gap_past - gap) / run
        } else {
            0.0
        };
        if sine > CORNER_SINE {
            (inside, StretchEnd::Corner(sine))
        } else {
            (inside, StretchEnd::Open)
        }
    };

    // The length of A between two of its parameters, along 16 chords.
    let arc = |from: f64, to: f64| -> f64 {
        let at = |i: usize| at_a(from + (to - from) * (i as f64 / 16.0));
        (0..16).map(|i| (at(i + 1) - at(i)).length()).sum()
    };
    let mut stretches: Vec<Stretch> = Vec::new();
    let mut corners: Vec<(f64, f64)> = Vec::new();
    let mut joins: Vec<f64> = Vec::new();
    let mut dropped = vec![false; crossings.len()];
    for (k, &(t_a, _, _)) in crossings.iter().enumerate() {
        if real[k] {
            continue;
        }
        let within = |s: &Stretch| t_a >= s.span.0 - s.slack.0 && t_a <= s.span.1 + s.slack.1;
        if let Some(known) = stretches.iter_mut().find(|s| within(s)) {
            // A stretch that is no overlap is one contact, and keeps the
            // one crossing where the curves are nearest.
            let gap = foot(t_a.clamp(lo, hi)).1;
            if known.overlap || gap >= known.nearest.1 {
                dropped[k] = true;
            } else {
                dropped[known.nearest.0] = true;
                known.nearest = (k, gap);
            }
            continue;
        }
        let t_a = t_a.clamp(lo, hi);
        if !on_b(t_a) {
            continue;
        }
        let (from, low) = follow(t_a, -1.0);
        let (to, high) = follow(t_a, 1.0);
        // A crossing at an angle has such a stretch too, as long as the
        // tolerance over the sine of the angle each side. An overlap that
        // ends at a corner is far longer.
        let length = arc(from, to);
        let (low, high) = junctions(&foot, &arc, tol, t_a, (from, low), (to, high));
        let stops = |end: StretchEnd| match end {
            StretchEnd::EdgeEnd | StretchEnd::Junction(_) => true,
            StretchEnd::Corner(sine) => length * sine > 50.0 * tol,
            StretchEnd::Open => false,
        };
        // Edges that only touch end to end have a stretch as well, about
        // the tolerance long. Their crossing stays: it joins the two ends.
        let overlap = stops(low) && stops(high) && length > 100.0 * tol;
        // A sample chord reads a crossing a part of a chord off its place
        // on the curve, so a crossing at an edge end can read just past it.
        let slack = |end: StretchEnd| match end {
            StretchEnd::EdgeEnd => step.min((to - from) / 4.0),
            StretchEnd::Corner(_) | StretchEnd::Open | StretchEnd::Junction(_) => 2.0 * probe,
        };
        if overlap {
            for (at, end, side) in [(from, low, -1.0), (to, high, 1.0)] {
                match end {
                    StretchEnd::Corner(_) => corners.push((at, side)),
                    StretchEnd::Junction(at) => joins.push(at),
                    StretchEnd::EdgeEnd | StretchEnd::Open => {}
                }
            }
        }
        dropped[k] = overlap;
        stretches.push(Stretch {
            span: (from, to),
            slack: (slack(low), slack(high)),
            overlap,
            nearest: (k, foot(t_a).1),
        });
    }

    let mut index = 0;
    crossings.retain(|_| {
        index += 1;
        !dropped[index - 1]
    });
    // One crossing at each corner, unless a crossing at a real angle is
    // read there already.
    for (at, side) in corners {
        if crossings
            .iter()
            .any(|&(t_a, _, _)| (t_a - at).abs() <= 4.0 * probe)
        {
            continue;
        }
        let past = (at + side * probe).clamp(lo, hi);
        let corner = refine_on_curves((a.0, a.1), (b.0, b.1), (past, foot(past).0), tol)
            .filter(|&(t_a, _, _)| (t_a - at).abs() <= 2.0 * probe)
            .unwrap_or_else(|| {
                let t_b = foot(at).0;
                let on_b = b.0.evaluate_with_endpoints(t_b, b.1.start_pos, b.1.end_pos);
                let on_a = at_a(at);
                (at, t_b, on_a + (on_b - on_a) * 0.5)
            });
        crossings.push(corner);
    }
    // Two junctions can be close: the two ends of a narrow bump. Only a
    // crossing at the same place, within tolerance, stands for a junction.
    for at in joins {
        let (t_b, on_a) = (foot(at).0, at_a(at));
        let point = on_a + (at_b(t_b) - on_a) * 0.5;
        if crossings.iter().any(|hit| (hit.2 - point).length() <= tol) {
            continue;
        }
        crossings.push((at, t_b, point));
    }
}

/// Turn the open ends of a stretch into junctions, where the two edges
/// share a curve along the stretch.
///
/// `foot` gives the foot on B of A's point at a parameter, and its distance.
/// `at` is a place in the stretch, and each end is its parameter with its
/// kind. An end stays open unless A lies on B to a thousandth of `tol`
/// from that end to the other, over [`SHARED_PART`] of the stretch or more.
/// `arc` gives the length of A between two parameters: a part of the
/// parameter span is no measure of a part of the curve.
fn junctions(
    foot: &impl Fn(f64) -> (f64, f64),
    arc: &impl Fn(f64, f64) -> f64,
    tol: f64,
    at: f64,
    low: (f64, StretchEnd),
    high: (f64, StretchEnd),
) -> (StretchEnd, StretchEnd) {
    let open = |end: StretchEnd| end == StretchEnd::Open;
    if !open(low.1) && !open(high.1) {
        return (low.1, high.1);
    }
    let on_curve = |t: f64| foot(t).1 <= tol * 1e-3;
    let Some(inside) = [at, f64::midpoint(low.0, high.0)]
        .into_iter()
        .find(|&t| on_curve(t))
    else {
        return (low.1, high.1);
    };
    // Where A leaves the common curve, between `inside` and an open end.
    let leaves = |end: f64| {
        let (mut on, mut off) = (inside, end);
        for _ in 0..26 {
            let mid = f64::midpoint(on, off);
            if on_curve(mid) {
                on = mid;
            } else {
                off = mid;
            }
        }
        on
    };
    let place = |end: (f64, StretchEnd)| if open(end.1) { leaves(end.0) } else { end.0 };
    let (from, to) = (place(low), place(high));
    let shared = arc(from, to) >= SHARED_PART * arc(low.0, high.0)
        && (1..8).all(|k| on_curve(from + (to - from) * (f64::from(k) / 8.0)));
    if !shared {
        return (low.1, high.1);
    }
    let kind = |end: (f64, StretchEnd), place: f64| {
        if open(end.1) {
            StretchEnd::Junction(place)
        } else {
            end.1
        }
    };
    (kind(low, from), kind(high, to))
}

/// Gauss-Newton on `Ca(ta) - Cb(tb)` from a sampled crossing, inside both
/// edges' domains. `None` when the curves do not meet within `tol` there.
fn refine_on_curves(
    curve_a: (&EdgeCurve, &EdgeData),
    curve_b: (&EdgeCurve, &EdgeData),
    seed: (f64, f64),
    tol: f64,
) -> Option<(f64, f64, Point3)> {
    match refine(curve_a, curve_b, seed, tol) {
        Refined::Met(ta, tb, point) => Some((ta, tb, point)),
        Refined::Along | Refined::Apart => None,
    }
}

/// How Gauss-Newton on two curves ended.
enum Refined {
    /// The curves meet within the tolerance at these parameters and point.
    Met(f64, f64, Point3),
    /// The search stopped where the curves lie within the tolerance and
    /// their tangents run along each other.
    Along,
    /// The curves do not meet near the seed.
    Apart,
}

/// [`refine_on_curves`], telling where it stops on parallel tangents
/// within the tolerance.
fn refine(
    (ca, ea): (&EdgeCurve, &EdgeData),
    (cb, eb): (&EdgeCurve, &EdgeData),
    (mut ta, mut tb): (f64, f64),
    tol: f64,
) -> Refined {
    let at_a = |t: f64| ca.evaluate_with_endpoints(t, ea.start_pos, ea.end_pos);
    let at_b = |t: f64| cb.evaluate_with_endpoints(t, eb.start_pos, eb.end_pos);
    let (lo_a, hi_a) = (ea.t0.min(ea.t1), ea.t0.max(ea.t1));
    let (lo_b, hi_b) = (eb.t0.min(eb.t1), eb.t0.max(eb.t1));
    let (ha, hb) = ((hi_a - lo_a) * 1e-7, (hi_b - lo_b) * 1e-7);
    if ha <= 0.0 || hb <= 0.0 {
        return Refined::Apart;
    }
    for _ in 0..30 {
        let r = at_a(ta) - at_b(tb);
        if r.length() <= tol * 1e-3 {
            break;
        }
        let da = (at_a((ta + ha).min(hi_a)) - at_a((ta - ha).max(lo_a)))
            * (1.0 / ((ta + ha).min(hi_a) - (ta - ha).max(lo_a)));
        let db = (at_b((tb + hb).min(hi_b)) - at_b((tb - hb).max(lo_b)))
            * (-1.0 / ((tb + hb).min(hi_b) - (tb - hb).max(lo_b)));
        let (a11, a12, a22) = (da.dot(da), da.dot(db), db.dot(db));
        let (g1, g2) = (da.dot(r), db.dot(r));
        let det = a11.madd(a22, -(a12 * a12));
        if det.abs() <= 1e-12 * a11 * a22 {
            return if r.length() <= tol {
                Refined::Along
            } else {
                Refined::Apart
            };
        }
        ta = (ta - (a22 * g1 - a12 * g2) / det).clamp(lo_a, hi_a);
        tb = (tb - (a11 * g2 - a12 * g1) / det).clamp(lo_b, hi_b);
    }
    let (pa, pb) = (at_a(ta), at_b(tb));
    if (pa - pb).length() <= tol {
        Refined::Met(ta, tb, pa + (pb - pa) * 0.5)
    } else {
        Refined::Apart
    }
}

/// Exact line-segment vs circular-arc intersection.
///
/// `line` is the straight edge, `arc` the circle edge's data, `circle` its
/// geometry. Returns `(t_a, t_b, point)` triples in the original edge order:
/// when `circle_is_a` the arc is edge A, otherwise the line is edge A.
fn line_circle_intersection(
    line: &EdgeData,
    arc: &EdgeData,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
    circle_is_a: bool,
) -> Vec<(f64, f64, Point3)> {
    let mut out = Vec::new();
    for (pt, angle) in circle.intersect_segment(line.start_pos, line.end_pos, tol.linear) {
        // Validate the hit lies within the arc's angular domain. The arc
        // parameter runs t0..t1 (radians); the solver returns [0, TAU). Test
        // the angle and its ±TAU shifts so a hit near the seam still matches.
        let lo = arc.t0.min(arc.t1) - tol.linear;
        let hi = arc.t0.max(arc.t1) + tol.linear;
        let in_arc = [
            angle,
            angle + std::f64::consts::TAU,
            angle - std::f64::consts::TAU,
        ]
        .iter()
        .find(|&&a| a >= lo && a <= hi)
        .copied();
        let Some(t_arc) = in_arc else {
            continue;
        };

        // Line parameter from the foot of the point on the segment.
        let dir = line.end_pos - line.start_pos;
        let len_sq = dir.length_squared();
        if len_sq < tol.linear * tol.linear {
            continue;
        }
        let s = ((pt - line.start_pos).dot(dir) / len_sq).clamp(0.0, 1.0);
        let t_line = s.madd(line.t1 - line.t0, line.t0);

        let triple = if circle_is_a {
            (t_arc, t_line, pt)
        } else {
            (t_line, t_arc, pt)
        };
        let is_dup = out.iter().any(|&(ca, cb, _): &(f64, f64, Point3)| {
            (triple.0 - ca).abs() < 1e-6 && (triple.1 - cb).abs() < 1e-6
        });
        if !is_dup {
            out.push(triple);
        }
    }
    out
}

/// Algebraic line-line intersection.
///
/// Computes the closest approach between two line segments. If the
/// segments are within tolerance at that point, returns the crossing.
fn line_line_intersection(ea: &EdgeData, eb: &EdgeData, tol: Tolerance) -> Vec<(f64, f64, Point3)> {
    let da = ea.end_pos - ea.start_pos;
    let db = eb.end_pos - eb.start_pos;
    let w = ea.start_pos - eb.start_pos;

    let a = da.dot(da);
    let b = da.dot(db);
    let c = db.dot(db);
    let d = da.dot(w);
    let e = db.dot(w);

    let denom = a.madd(c, -(b * b));

    // Parallel lines — 1e-20 checks for mathematical degeneracy
    // (near-zero determinant), not geometric tolerance.
    if denom.abs() < 1e-20 {
        return Vec::new();
    }

    let s = b.madd(e, -(c * d)) / denom;
    let t = a.madd(e, -(b * d)) / denom;

    // Check if within edge domains [0, 1] for lines
    let range = -tol.linear..=1.0 + tol.linear;
    if !range.contains(&s) || !range.contains(&t) {
        return Vec::new();
    }

    let pt_a = ea.start_pos + da * s;
    let pt_b = eb.start_pos + db * t;
    let dist = (pt_a - pt_b).length();

    if dist <= tol.linear {
        let midpoint = Point3::new(
            f64::midpoint(pt_a.x(), pt_b.x()),
            f64::midpoint(pt_a.y(), pt_b.y()),
            f64::midpoint(pt_a.z(), pt_b.z()),
        );
        // Map s,t from [0,1] to actual edge parameter domains
        let param_a = s.madd(ea.t1 - ea.t0, ea.t0);
        let param_b = t.madd(eb.t1 - eb.t0, eb.t0);
        vec![(param_a, param_b, midpoint)]
    } else {
        Vec::new()
    }
}

/// A parameterized sample point on an edge segment.
#[derive(Clone, Copy)]
struct SegmentEndpoint {
    /// Parameter value on the edge curve.
    t: f64,
    /// 3D position at this parameter.
    pos: Point3,
}

/// The closest approach of two segments.
struct ChordApproach {
    /// Parameter on the first segment's edge.
    t_a: f64,
    /// Parameter on the second segment's edge.
    t_b: f64,
    /// Midway between the two closest points.
    point: Point3,
    /// Distance between the two closest points.
    gap: f64,
    /// The segments are all but parallel. Alone, such a pair cannot tell a
    /// crossing from two edges that run along each other.
    parallel: bool,
}

/// Find closest approach between two line segments.
///
/// Returns it if the distance is within tolerance.
#[allow(clippy::similar_names)]
fn closest_segment_pair(
    seg_a: [&SegmentEndpoint; 2],
    seg_b: [&SegmentEndpoint; 2],
    tol: f64,
) -> Option<ChordApproach> {
    let da = seg_a[1].pos - seg_a[0].pos;
    let db = seg_b[1].pos - seg_b[0].pos;
    let w = seg_a[0].pos - seg_b[0].pos;

    let a = da.dot(da);
    let b = da.dot(db);
    let c = db.dot(db);
    let d = da.dot(w);
    let e = db.dot(w);

    let denom = a.madd(c, -(b * b));
    let parallel = a > 0.0 && c > 0.0 && denom <= 1e-12 * a * c;

    // A segment of no length is a point, and the closest pair is that point
    // and its foot on the other segment (`w` runs from `b`'s start to `a`'s).
    let (s, t) = if a <= 0.0 && c <= 0.0 {
        (0.5, 0.5)
    } else if a <= 0.0 {
        (0.0, (e / c).clamp(0.0, 1.0))
    } else if c <= 0.0 {
        ((-d / a).clamp(0.0, 1.0), 0.0)
    } else if parallel {
        // `a c - b^2` is `a c sin^2` of the angle between the segments, and
        // for segments all but parallel it is lost in rounding: `s` and `t`
        // would be noise. `b` is read against `a`'s line instead: its foot
        // on the line is at `s = (b t - d) / a`, and its standoff from the
        // line, `start + drift t`, is least at `-start.drift / drift^2`.
        let off_line = |q: Vec3| q - da * (q.dot(da) / a);
        let start = off_line(w * -1.0);
        let drift = off_line(db);
        let drift2 = drift.dot(drift);
        // The part of `b` whose foot lies on segment `a`.
        let (t0, t1) = (d / b, (a + d) / b);
        let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));
        let size = [seg_a[0], seg_a[1], seg_b[0], seg_b[1]]
            .iter()
            .map(|p| p.pos.x().abs().max(p.pos.y().abs()).max(p.pos.z().abs()))
            .fold(0.0, f64::max);
        let t = if lo > hi {
            // No overlap: the end of `b` that faces `a`.
            if t0.max(t1) < 0.0 { 0.0 } else { 1.0 }
        } else if drift2 <= (64.0 * f64::EPSILON * size).min(0.5 * tol).powi(2) {
            // A standoff that changes by rounding alone is the same all
            // along the overlap: its middle reads it whichever way it rounds.
            // Far from the origin rounding moves it more, but never past
            // half the tolerance here: a crossing whose standoff changes
            // more is read where it is, not at the middle, where it could
            // stand out of tolerance.
            f64::midpoint(lo, hi)
        } else {
            (-start.dot(drift) / drift2).clamp(lo, hi)
        };
        (((b * t - d) / a).clamp(0.0, 1.0), t)
    } else {
        let s = (b.madd(e, -(c * d)) / denom).clamp(0.0, 1.0);
        let t = (a.madd(e, -(b * d)) / denom).clamp(0.0, 1.0);
        (s, t)
    };

    let pt_a = seg_a[0].pos + da * s;
    let pt_b = seg_b[0].pos + db * t;
    let gap = (pt_a - pt_b).length();

    (gap <= tol).then(|| ChordApproach {
        t_a: s.madd(seg_a[1].t - seg_a[0].t, seg_a[0].t),
        t_b: t.madd(seg_b[1].t - seg_b[0].t, seg_b[0].t),
        point: Point3::new(
            f64::midpoint(pt_a.x(), pt_b.x()),
            f64::midpoint(pt_a.y(), pt_b.y()),
            f64::midpoint(pt_a.z(), pt_b.z()),
        ),
        gap,
        parallel,
    })
}

/// Group cells of an `n` by `n` grid into sets joined through their eight
/// neighbours, as indices into `cells`.
fn ribbons(cells: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
    let mut slot = vec![None; n * n];
    for (k, &(i, j)) in cells.iter().enumerate() {
        slot[i * n + j] = Some(k);
    }
    let mut seen = vec![false; cells.len()];
    let mut out = Vec::new();
    for first in 0..cells.len() {
        if seen[first] {
            continue;
        }
        seen[first] = true;
        let mut ribbon = vec![first];
        let mut next = 0;
        while let Some(&k) = ribbon.get(next) {
            next += 1;
            let (i, j) = cells[k];
            for ni in i.saturating_sub(1)..=(i + 1).min(n - 1) {
                for nj in j.saturating_sub(1)..=(j + 1).min(n - 1) {
                    if let Some(m) = slot[ni * n + nj]
                        && !seen[m]
                    {
                        seen[m] = true;
                        ribbon.push(m);
                    }
                }
            }
        }
        out.push(ribbon);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::Vec3;

    /// Chords in random directions, each with its size, from 0.1 to 100: a
    /// chord is up to that long and up to 50 times that far from the origin.
    fn random_chords(count: usize) -> Vec<(Point3, Point3, f64)> {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..count)
            .map(|_| {
                let scale = 10f64.powf(3.0 * rand() - 1.0);
                let from = Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5) * (100.0 * scale);
                let dir = Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5) * scale;
                let to = from + dir;
                (
                    Point3::new(from.x(), from.y(), from.z()),
                    Point3::new(to.x(), to.y(), to.z()),
                    scale,
                )
            })
            .collect()
    }

    /// Each coordinate one rounding step down, up, or kept, by turns.
    fn nudged(p: Point3, turn: usize) -> Point3 {
        let step = |x: f64, k: usize| match (turn + k) % 3 {
            0 => x.next_down(),
            1 => x,
            _ => x.next_up(),
        };
        Point3::new(step(p.x(), 0), step(p.y(), 1), step(p.z(), 2))
    }

    /// Two edges on one curve, running opposite ways, sample to chords that
    /// lie on each other: `a c - b^2` is zero or rounding noise there. Read
    /// at their middles (when it came out exactly zero) or at noise, the
    /// answer turned on how the products rounded. Chords a rounding step
    /// apart are read at the same place.
    #[test]
    fn chords_lying_on_each_other_are_read_alike_a_rounding_step_apart() {
        let end = |t: f64, pos: Point3| SegmentEndpoint { t, pos };
        for (turn, (from, to, scale)) in random_chords(20_000).into_iter().enumerate() {
            let a = [end(0.0, from), end(1.0, to)];
            let b = [end(5.0, to), end(6.0, from)];
            let moved = [end(5.0, nudged(to, turn)), end(6.0, nudged(from, turn + 1))];
            let tol = 1e-7 * scale.max(1.0);
            let read = |b: &[SegmentEndpoint; 2]| {
                closest_segment_pair([&a[0], &a[1]], [&b[0], &b[1]], tol)
                    .expect("the chords lie on each other")
            };
            let (x, y) = (read(&b), read(&moved));
            assert!(x.parallel && y.parallel);
            assert!(
                (x.t_a - y.t_a).abs() < 1e-9 && (x.t_b - y.t_b).abs() < 1e-9,
                "read at {}, {} and at {}, {}",
                x.t_a,
                x.t_b,
                y.t_a,
                y.t_b
            );
        }
    }

    /// The same two edges on one line, running opposite ways, the ends of
    /// one a rounding step off the other's: they overlap, and nothing inside
    /// them is a crossing.
    #[test]
    fn edges_on_one_line_running_opposite_ways_do_not_cross_inside() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        for (turn, (from, to, _)) in random_chords(400).into_iter().enumerate() {
            let mut topo = Topology::new();
            let line = add_edge(&mut topo, from, to, EdgeCurve::Line);
            let (start, end) = (nudged(to, turn), nudged(from, turn + 1));
            let nurbs =
                NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2])
                    .unwrap();
            let spline = add_edge(&mut topo, start, end, EdgeCurve::NurbsCurve(nurbs));
            for (ea, eb) in [(line, spline), (spline, line)] {
                let hits = crossings_inside(&topo, ea, eb);
                assert!(hits.is_empty(), "{from:?}..{to:?}: {hits:?}");
            }
        }
    }

    /// Two edges meeting end to end meet only there. At a real angle that
    /// is one crossing; all but parallel or in one line, the VV phase joins
    /// the ends, and nothing inside either edge is a crossing.
    #[test]
    fn edges_meeting_end_to_end_meet_only_at_that_end() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let touch = Point3::new(1.0, 0.0, 0.0);
        for angle in [1.0_f64, 1e-3, 1e-7, 0.0] {
            let far = Point3::new(1.0 + angle.cos(), angle.sin(), 0.0);
            for (from, to) in [(touch, far), (far, touch)] {
                let mut topo = Topology::new();
                let line = add_edge(
                    &mut topo,
                    Point3::new(0.0, 0.0, 0.0),
                    touch,
                    EdgeCurve::Line,
                );
                let nurbs =
                    NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![from, to], vec![1.0; 2])
                        .unwrap();
                let spline = add_edge(&mut topo, from, to, EdgeCurve::NurbsCurve(nurbs));
                for (ea, eb) in [(line, spline), (spline, line)] {
                    let hits = crossings(&topo, ea, eb);
                    if angle >= 1e-3 {
                        assert_eq!(hits.len(), 1, "angle {angle}: {hits:?}");
                        assert!(
                            (hits[0].2 - touch).length() < 1e-9,
                            "angle {angle}: {hits:?}"
                        );
                    } else {
                        assert!(hits.len() <= 1, "angle {angle}: {hits:?}");
                    }
                    let inside = crossings_inside(&topo, ea, eb);
                    assert!(inside.is_empty(), "angle {angle}: {inside:?}");
                }
            }
        }
    }

    /// A chord of no length (a curve standing still between two samples)
    /// meets the other chord at its foot on it, not at that chord's middle.
    #[test]
    fn a_point_chord_meets_the_other_chord_at_its_foot() {
        let end = |t: f64, x: f64, y: f64| SegmentEndpoint {
            t,
            pos: Point3::new(x, y, 1.0),
        };
        let chord = [end(5.0, 1.0, 1.0), end(6.0, 3.0, 1.0)];
        let on = [end(0.0, 1.4, 1.0), end(1.0, 1.4, 1.0)];
        let off = [end(0.0, 1.4, 1.0 + 1e-6), end(1.0, 1.4, 1.0 + 1e-6)];
        let tol = 1e-7;

        let hit = closest_segment_pair([&on[0], &on[1]], [&chord[0], &chord[1]], tol)
            .expect("the point lies on the chord");
        let (tp, tc, p) = (hit.t_a, hit.t_b, hit.point);
        assert!(tp.abs() < 1e-12 && (tc - 5.2).abs() < 1e-12, "{tp}, {tc}");
        assert!((p - on[0].pos).length() < 1e-12, "{p:?}");
        let hit = closest_segment_pair([&chord[0], &chord[1]], [&on[0], &on[1]], tol)
            .expect("the point lies on the chord");
        let (tc, tp) = (hit.t_a, hit.t_b);
        assert!((tc - 5.2).abs() < 1e-12 && tp.abs() < 1e-12, "{tc}, {tp}");

        assert!(closest_segment_pair([&off[0], &off[1]], [&chord[0], &chord[1]], tol).is_none());
        assert!(closest_segment_pair([&chord[0], &chord[1]], [&off[0], &off[1]], tol).is_none());
        assert!(closest_segment_pair([&on[0], &on[1]], [&on[0], &on[1]], tol).is_some());
        assert!(closest_segment_pair([&on[0], &on[1]], [&off[0], &off[1]], tol).is_none());
    }

    fn line_data(start: Point3, end: Point3) -> EdgeData {
        EdgeData {
            start_pos: start,
            end_pos: end,
            t0: 0.0,
            t1: 1.0,
            bbox_min: start,
            bbox_max: end,
            sampled: std::cell::OnceCell::new(),
        }
    }

    fn arc_data(t0: f64, t1: f64) -> EdgeData {
        // Positions/bbox are unused by line_circle_intersection (only t0/t1 are).
        EdgeData {
            start_pos: Point3::new(0.0, 0.0, 0.0),
            end_pos: Point3::new(0.0, 0.0, 0.0),
            t0,
            t1,
            bbox_min: Point3::new(0.0, 0.0, 0.0),
            bbox_max: Point3::new(0.0, 0.0, 0.0),
            sampled: std::cell::OnceCell::new(),
        }
    }

    /// A scoop's cubic end profile crossing an envelope's straight edge in
    /// the plane they share: the crossing lies on both curves, not where two
    /// of their sample chords meet (0.008 along the edge from it).
    #[test]
    fn a_cubic_crossing_a_line_is_found_on_the_cubic() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_topology::edge::Edge;

        let mut topo = Topology::new();
        let tol = Tolerance::default();
        let profile = [
            Point3::new(44.03, -5.116_666_666_666_667, 2.25),
            Point3::new(44.03, -12.702_471_043_387_55, -1.230_890_664_590_062),
            Point3::new(44.03, -21.630_890_664_590_066, 7.697_528_956_612_448),
            Point3::new(44.03, -18.15, 15.283_333_333_333_331),
        ];
        let cubic = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            profile.to_vec(),
            vec![1.0; 4],
        )
        .unwrap();
        let mut edge = |a: Point3, b: Point3, curve: EdgeCurve| {
            let va = topo.add_vertex(Vertex::new(a, tol.linear));
            let vb = topo.add_vertex(Vertex::new(b, tol.linear));
            topo.add_edge(Edge::new(va, vb, curve))
        };
        let ea = edge(profile[0], profile[3], EdgeCurve::NurbsCurve(cubic));
        let eb = edge(
            Point3::new(44.03, -17.0, 6.0),
            Point3::new(44.03, 17.0, 6.0),
            EdgeCurve::Line,
        );
        let da = collect_edge_data(&topo, &[ea]).unwrap();
        let db = collect_edge_data(&topo, &[eb]).unwrap();
        let hits = find_edge_edge_crossings(&topo, ea, &da[0], eb, &db[0], tol).unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        let p = hits[0].2;
        assert!((p.z() - 6.0).abs() < 1e-9, "off the line: {p:?}");
        assert!(
            (p.y() + 16.963_564_789_203).abs() < 1e-9,
            "off the cubic: {p:?}"
        );
    }

    fn add_edge(topo: &mut Topology, start: Point3, end: Point3, curve: EdgeCurve) -> EdgeId {
        let tol = Tolerance::default().linear;
        let va = topo.add_vertex(Vertex::new(start, tol));
        let vb = topo.add_vertex(Vertex::new(end, tol));
        topo.add_edge(brepkit_topology::edge::Edge::new(va, vb, curve))
    }

    fn crossings(topo: &Topology, ea: EdgeId, eb: EdgeId) -> Vec<(f64, f64, Point3)> {
        let data_a = collect_edge_data(topo, &[ea]).unwrap();
        let data_b = collect_edge_data(topo, &[eb]).unwrap();
        find_edge_edge_crossings(topo, ea, &data_a[0], eb, &data_b[0], Tolerance::default())
            .unwrap()
    }

    /// The crossings of two edges that lie strictly inside both of them.
    fn crossings_inside(topo: &Topology, ea: EdgeId, eb: EdgeId) -> Vec<(f64, f64, Point3)> {
        let data_a = collect_edge_data(topo, &[ea]).unwrap();
        let data_b = collect_edge_data(topo, &[eb]).unwrap();
        let inside = |t: f64, d: &EdgeData| {
            let (lo, hi) = (d.t0.min(d.t1), d.t0.max(d.t1));
            let margin = 1e-9 * (hi - lo);
            t > lo + margin && t < hi - margin
        };
        crossings(topo, ea, eb)
            .into_iter()
            .filter(|&(ta, tb, _)| inside(ta, &data_a[0]) && inside(tb, &data_b[0]))
            .collect()
    }

    /// Heal turns a line into a degree-1 NURBS, so a line edge can meet a
    /// NURBS edge along the same line, or a hair off it. The two overlap and
    /// do not cross: the VE phase paves the ends of the overlap, and a
    /// crossing inside it splits both edges where nothing meets them.
    #[test]
    fn a_line_and_a_nurbs_along_it_do_not_cross_inside_their_overlap() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        for (origin, dir) in [
            (Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Point3::new(1.3, -2.7, 0.4), Vec3::new(0.6, 0.48, 0.64)),
        ] {
            let side = dir.cross(Vec3::new(0.0, 0.0, 1.0)).normalize().unwrap();
            for apart in [0.0, 5e-8] {
                let at = |s: f64| origin + dir * s + side * apart;
                // Inside the line, past one of its ends, past both, and on
                // its span; each one both ways.
                for (lo, hi) in [
                    (2.0, 7.0),
                    (7.0, 2.0),
                    (3.0, 15.0),
                    (15.0, 3.0),
                    (-4.0, 13.0),
                    (13.0, -4.0),
                    (0.0, 10.0),
                    (10.0, 0.0),
                ] {
                    let mut topo = Topology::new();
                    let line = add_edge(&mut topo, origin, origin + dir * 10.0, EdgeCurve::Line);
                    let nurbs = NurbsCurve::new(
                        1,
                        vec![0.0, 0.0, 1.0, 1.0],
                        vec![at(lo), at(hi)],
                        vec![1.0; 2],
                    )
                    .unwrap();
                    let spline = add_edge(&mut topo, at(lo), at(hi), EdgeCurve::NurbsCurve(nurbs));
                    for (ea, eb) in [(line, spline), (spline, line)] {
                        let hits = crossings_inside(&topo, ea, eb);
                        assert!(hits.is_empty(), "{lo}..{hi} {apart} off {dir:?}: {hits:?}");
                    }
                }
            }
        }
    }

    /// A parabola crossing a line twice at 0.12 rad, 0.12 apart, in
    /// neighbouring chords of both: two crossings at a real angle stay two.
    #[test]
    fn two_crossings_in_neighbouring_chords_stay_two() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let height = |x: f64| (x - 1.62) * (x - 1.74);
        let poles = vec![
            Point3::new(0.0, height(0.0), 0.0),
            Point3::new(1.6, height(0.0) - 1.6 * 3.36, 0.0),
            Point3::new(3.2, height(3.2), 0.0),
        ];
        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.2, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let parabola = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles.clone(),
            vec![1.0; 3],
        )
        .unwrap();
        let curve = add_edge(
            &mut topo,
            poles[0],
            poles[2],
            EdgeCurve::NurbsCurve(parabola),
        );
        for (ea, eb) in [(line, curve), (curve, line)] {
            let mut xs: Vec<f64> = crossings(&topo, ea, eb).iter().map(|h| h.2.x()).collect();
            xs.sort_by(f64::total_cmp);
            assert_eq!(xs.len(), 2, "{xs:?}");
            assert!(
                (xs[0] - 1.62).abs() < 1e-9 && (xs[1] - 1.74).abs() < 1e-9,
                "{xs:?}"
            );
        }
    }

    /// A spline that runs along a line from x = 2 to x = 6 and leaves it
    /// there at a corner. The stretch along the line is an overlap: the VE
    /// phase paves the spline's end at x = 2, and the crossing at the corner,
    /// read at a real angle, bounds it at x = 6. Nothing else is a crossing.
    #[test]
    fn a_spline_leaving_a_line_at_a_corner_crosses_it_only_there() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        for (origin, dir) in [
            (Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Point3::new(1.3, -2.7, 0.4), Vec3::new(0.6, 0.48, 0.64)),
        ] {
            let side = dir.cross(Vec3::new(0.0, 0.0, 1.0)).normalize().unwrap();
            let corner = origin + dir * 6.0;
            let poles = vec![origin + dir * 2.0, corner, origin + dir * 8.0 + side * 3.0];
            let mut topo = Topology::new();
            let line = add_edge(&mut topo, origin, origin + dir * 10.0, EdgeCurve::Line);
            let nurbs = NurbsCurve::new(
                1,
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                poles.clone(),
                vec![1.0; 3],
            )
            .unwrap();
            let spline = add_edge(&mut topo, poles[0], poles[2], EdgeCurve::NurbsCurve(nurbs));
            for (ea, eb) in [(line, spline), (spline, line)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "along {dir:?}: {hits:?}");
                assert!(
                    (hits[0].2 - corner).length() < 1e-9,
                    "along {dir:?}: {hits:?}"
                );
            }
        }
    }

    /// A straight cubic crossing a line at 1e-7 rad stays within the linear
    /// tolerance of it for a unit each side, past several sample chords. A
    /// crossing read at the end of each of those chords minted a cluster of
    /// vertices. It is read once, where the cubic crosses the line.
    #[test]
    fn a_grazing_crossing_is_not_a_cluster() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let angle = 1e-7_f64;
        let at = |u: f64| Point3::new(1.65 + u * angle.cos(), u * angle.sin(), 0.0);
        // The cubic's samples level with the line's, then off them.
        for from in [-1.55, -1.62] {
            let mut topo = Topology::new();
            let line = add_edge(
                &mut topo,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(3.2, 0.0, 0.0),
                EdgeCurve::Line,
            );
            let poles: Vec<Point3> = (0..4)
                .map(|k| at(from + 3.2 * f64::from(k) / 3.0))
                .collect();
            let cubic = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                poles.clone(),
                vec![1.0; 4],
            )
            .unwrap();
            let curve = add_edge(&mut topo, poles[0], poles[3], EdgeCurve::NurbsCurve(cubic));
            let hits = crossings(&topo, line, curve);
            assert_eq!(hits.len(), 1, "from {from}: {hits:?}");
            let gap = (hits[0].2 - at(0.0)).length();
            assert!(gap < 1e-6, "from {from}: read {gap:e} from the crossing");
        }
    }

    /// Far from the origin, rounding moves a standoff by more than the
    /// tolerance allows a crossing to stand off. A crossing a fifth of the
    /// way along a chord of both edges, whose standoff changes by six
    /// tolerances along that chord, stands out of tolerance at the chord's
    /// middle and at both its ends. It is still read, once, where it is.
    #[test]
    fn a_shallow_crossing_far_from_the_origin_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let far = 1e8;
        let at = |x: f64| Point3::new(far + x, 1.92e-9 * (x - 60.0), 0.0);
        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(far - 5000.0, 0.0, 0.0),
            Point3::new(far + 5000.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let nurbs = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![at(-5000.0), at(5000.0)],
            vec![1.0; 2],
        )
        .unwrap();
        let spline = add_edge(
            &mut topo,
            at(-5000.0),
            at(5000.0),
            EdgeCurve::NurbsCurve(nurbs),
        );
        for (ea, eb) in [(line, spline), (spline, line)] {
            let hits = crossings(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            let gap = (hits[0].2 - at(60.0)).length();
            assert!(gap < 1e-3, "read {gap:e} from the crossing");
        }
    }

    /// Two long edges crossing at 5e-7 rad are all but parallel, yet stand
    /// 2.5e-3 apart at their ends: they cross at one place, and no end is
    /// near the other edge for the VE phase to pave.
    #[test]
    fn a_shallow_crossing_of_long_edges_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        // The crossing at a sample point of both edges, then inside a chord.
        for (from, to) in [(-5000.0, 5000.0), (-4100.0, 5300.0)] {
            let at = |x: f64| Point3::new(x, 5e-7 * x, 0.0);
            let mut topo = Topology::new();
            let line = add_edge(
                &mut topo,
                Point3::new(-5000.0, 0.0, 0.0),
                Point3::new(5000.0, 0.0, 0.0),
                EdgeCurve::Line,
            );
            let nurbs = NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![at(from), at(to)],
                vec![1.0; 2],
            )
            .unwrap();
            let spline = add_edge(&mut topo, at(from), at(to), EdgeCurve::NurbsCurve(nurbs));
            for (ea, eb) in [(line, spline), (spline, line)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{from}..{to}: {hits:?}");
                let gap = (hits[0].2 - at(0.0)).length();
                assert!(gap < 1e-6, "{from}..{to}: read {gap:e} from the crossing");
            }
        }
    }

    /// A spline crossing a line at 5e-7 rad, its two chords at the crossing
    /// only 0.001 long. Along each of them its standoff from the line changes
    /// by 5e-10: alone, such a chord pair looks like two edges that keep
    /// their distance, and only the whole ribbon shows the crossing.
    #[test]
    fn a_shallow_crossing_between_short_chords_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let at = |x: f64| Point3::new(x, 5e-7 * x, 0.0);
        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(-10.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let poles: Vec<Point3> = [-10.0, -0.001, 0.001, 10.0].into_iter().map(at).collect();
        let nurbs = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 15.0 / 32.0, 17.0 / 32.0, 1.0, 1.0],
            poles.clone(),
            vec![1.0; 4],
        )
        .unwrap();
        let spline = add_edge(&mut topo, poles[0], poles[3], EdgeCurve::NurbsCurve(nurbs));
        for (ea, eb) in [(line, spline), (spline, line)] {
            let hits = crossings(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            let gap = (hits[0].2 - at(0.0)).length();
            assert!(gap < 1e-6, "read {gap:e} from the crossing");
        }
    }

    /// A parabola that touches a line at x = 49 and stays within the linear
    /// tolerance of it for x in 39..59. The closest place on each sampled
    /// chord's line lies past the chord, so no one chord pair holds it, and
    /// both ends of the parabola are too far from the line for the VE phase.
    #[test]
    fn a_shallow_tangency_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let height = |x: f64| 1e-5 * (x / 100.0 - 0.49).powi(2);
        let poles = vec![
            Point3::new(0.0, height(0.0), 0.0),
            Point3::new(50.0, height(0.0) - 1e-5 * 0.49, 0.0),
            Point3::new(100.0, height(100.0), 0.0),
        ];
        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let parabola = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles.clone(),
            vec![1.0; 3],
        )
        .unwrap();
        let curve = add_edge(
            &mut topo,
            poles[0],
            poles[2],
            EdgeCurve::NurbsCurve(parabola),
        );
        let tol = Tolerance::default().linear;
        for (ea, eb) in [(line, curve), (curve, line)] {
            let hits = crossings(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            let p = hits[0].2;
            assert!((p.x() - 49.0).abs() <= 2.0, "read at {p:?}");
            assert!(p.y().hypot(p.z()) <= tol, "off the line: {p:?}");
            assert!(
                (p.y() - height(p.x())).hypot(p.z()) <= tol,
                "off the parabola: {p:?}"
            );
        }
    }

    /// A long line, its chords 100 long, and a short degree-1 spline from
    /// x = 40 to 60 at 1e-8 rad to it. The spline starts within tolerance of
    /// the line, crosses it at x = 45 and leaves the band near x = 55 while
    /// both go on. Its ribbon lies in one chord of the line and stops only at
    /// the spline's start, so it holds the crossing.
    #[test]
    fn a_shallow_crossing_inside_one_long_chord_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3200.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let (start, end) = (
            Point3::new(40.0, -5e-8, 0.0),
            Point3::new(60.0, 1.5e-7, 0.0),
        );
        let nurbs =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        let spline = add_edge(&mut topo, start, end, EdgeCurve::NurbsCurve(nurbs));
        for (ea, eb) in [(line, spline), (spline, line)] {
            let hits = crossings(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            let gap = (hits[0].2 - Point3::new(45.0, 0.0, 0.0)).length();
            assert!(gap < 1e-6, "read {gap:e} from the crossing");
        }
    }

    /// A long line and a short spline that start at one place, run within
    /// tolerance of each other, cross at x = 5 and part near x = 15 while
    /// both go on. Two edge ends stop the ribbon, both at its start; its
    /// other end is open, so it holds the crossing.
    #[test]
    fn edges_from_one_start_that_cross_further_on_cross_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3200.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let (start, end) = (Point3::new(0.0, -5e-8, 0.0), Point3::new(20.0, 1.5e-7, 0.0));
        let nurbs =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        let spline = add_edge(&mut topo, start, end, EdgeCurve::NurbsCurve(nurbs));
        for (ea, eb) in [(line, spline), (spline, line)] {
            let hits = crossings(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            let inside = crossings_inside(&topo, ea, eb);
            assert_eq!(inside.len(), 1, "{hits:?}");
            let gap = (inside[0].2 - Point3::new(5.0, 0.0, 0.0)).length();
            assert!(gap < 1e-6, "read {gap:e} from the crossing");
        }
    }

    /// A short spline inside one chord of a long line and within tolerance
    /// of it all along, at 3e-8 or from 1e-8 to 9e-8 off it, either way: the
    /// spline's two ends stop its ribbon at both of its ends, an overlap.
    #[test]
    fn a_short_edge_along_one_long_chord_does_not_cross_inside_it() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
        for (start, end) in [
            (at(40.0, 3e-8), at(60.0, 3e-8)),
            (at(40.0, 1e-8), at(60.0, 9e-8)),
            (at(60.0, 9e-8), at(40.0, 1e-8)),
        ] {
            let mut topo = Topology::new();
            let line = add_edge(&mut topo, at(0.0, 0.0), at(3200.0, 0.0), EdgeCurve::Line);
            let nurbs =
                NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2])
                    .unwrap();
            let spline = add_edge(&mut topo, start, end, EdgeCurve::NurbsCurve(nurbs));
            for (ea, eb) in [(line, spline), (spline, line)] {
                let hits = crossings_inside(&topo, ea, eb);
                assert!(hits.is_empty(), "{start:?}..{end:?}: {hits:?}");
            }
        }
    }

    /// A short spline that starts within tolerance of a long line, inside
    /// one of its chords, and leaves the band near x = 45 without crossing.
    /// Its ribbon stops at the spline's start only, so it is read once, at
    /// its closest place: that start, which the VE phase paves as well.
    /// Nothing inside both edges is a crossing.
    #[test]
    fn a_short_edge_leaving_a_long_one_from_its_start_meets_it_only_there() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let line = add_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3200.0, 0.0, 0.0),
            EdgeCurve::Line,
        );
        let (start, end) = (Point3::new(40.0, 5e-8, 0.0), Point3::new(60.0, 2.5e-7, 0.0));
        let nurbs =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        let spline = add_edge(&mut topo, start, end, EdgeCurve::NurbsCurve(nurbs));
        let tol = Tolerance::default().linear;
        for (ea, eb) in [(line, spline), (spline, line)] {
            let hits = crossings(&topo, ea, eb);
            assert!(hits.len() <= 1, "{hits:?}");
            assert!(
                hits.iter().all(|h| (h.2 - start).length() <= tol),
                "{hits:?}"
            );
            let inside = crossings_inside(&topo, ea, eb);
            assert!(inside.is_empty(), "{inside:?}");
        }
    }

    /// A cubic arch, in a plane or lifted out of it by `twist`.
    fn arch(twist: f64) -> brepkit_math::nurbs::curve::NurbsCurve {
        brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, twist),
                Point3::new(3.0, 2.0, -twist),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0; 4],
        )
        .unwrap()
    }

    /// An edge on `curve` from its place at `from` to its place at `to`.
    fn edge_on(
        topo: &mut Topology,
        curve: &brepkit_math::nurbs::curve::NurbsCurve,
        from: f64,
        to: f64,
    ) -> EdgeId {
        add_edge(
            topo,
            curve.evaluate(from),
            curve.evaluate(to),
            EdgeCurve::NurbsCurve(curve.clone()),
        )
    }

    /// The poles of the part of a cubic Bezier curve from `from` to `to`.
    fn bezier_part(poles: &[Point3], from: f64, to: f64) -> Vec<Point3> {
        let blossom = |us: [f64; 3]| {
            let mut level = poles.to_vec();
            for u in us {
                level = level.windows(2).map(|w| w[0] + (w[1] - w[0]) * u).collect();
            }
            level[0]
        };
        vec![
            blossom([from, from, from]),
            blossom([from, from, to]),
            blossom([from, to, to]),
            blossom([to, to, to]),
        ]
    }

    /// Two edges on one cubic, in opposite directions, share each sample
    /// point. Chords of the two edges meet there at a real angle, and each
    /// such place was a crossing: 31 inside the two edges.
    #[test]
    fn two_edges_on_one_cubic_running_opposite_ways_do_not_cross() {
        for twist in [0.0, 0.5] {
            let curve = arch(twist);
            let mut topo = Topology::new();
            let forward = edge_on(&mut topo, &curve, 0.0, 1.0);
            let twin = edge_on(&mut topo, &curve, 0.0, 1.0);
            let back = edge_on(&mut topo, &curve, 1.0, 0.0);
            for (ea, eb) in [
                (forward, back),
                (back, forward),
                (forward, twin),
                (twin, forward),
            ] {
                let hits = crossings(&topo, ea, eb);
                assert!(hits.is_empty(), "{} crossings: {hits:?}", hits.len());
            }
        }
    }

    /// An edge on a part of another edge's cubic, and two edges that share
    /// a part of one cubic: the sample points of the two differ, so their
    /// chords cross between them. The edges overlap, and nothing inside both
    /// is a crossing.
    #[test]
    fn edges_that_share_a_part_of_one_cubic_do_not_cross_inside_it() {
        for (twist, a, b) in [0.0, 0.5].into_iter().flat_map(|twist| {
            [
                ((0.0, 1.0), (0.23, 0.71)),
                ((0.0, 1.0), (0.71, 0.23)),
                ((0.0, 0.6), (0.37, 1.0)),
                ((0.0, 0.6), (1.0, 0.37)),
                ((0.6, 0.0), (0.37, 1.0)),
                ((0.0, 1.0), (0.4, 0.4 + 1.0 / 128.0)),
                ((0.0, 0.5), (0.0, 1.0)),
            ]
            .map(|(a, b)| (twist, a, b))
        }) {
            let curve = arch(twist);
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &curve, a.0, a.1);
            let eb = edge_on(&mut topo, &curve, b.0, b.1);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings_inside(&topo, ea, eb);
                assert!(hits.is_empty(), "{a:?} on {b:?}, twist {twist}: {hits:?}");
            }
        }
    }

    /// The same cubic as a degree-4 curve, and as a cubic over other knots:
    /// equal curves with different poles. An edge on a part of one overlaps
    /// an edge on the other.
    #[test]
    fn edges_on_equal_curves_with_different_poles_do_not_cross() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let curve = arch(0.0);
        let p = bezier_part(&bezier_part(&arch_poles(), 0.0, 1.0), 0.0, 1.0);
        let raised: Vec<Point3> = (0..=4)
            .map(|i| {
                let w = f64::from(i) / 4.0;
                let before = p[usize::try_from(i - 1).unwrap_or(0).min(3)];
                let here = p[usize::try_from(i).unwrap_or(0).min(3)];
                here + (before - here) * w
            })
            .collect();
        let quartic = NurbsCurve::new(
            4,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            raised,
            vec![1.0; 5],
        )
        .unwrap();
        let mut halves = bezier_part(&p, 0.0, 0.3);
        halves.extend(bezier_part(&p, 0.3, 1.0).into_iter().skip(1));
        let split = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0],
            halves,
            vec![1.0; 7],
        )
        .unwrap();
        for (other, from, to) in [
            (&quartic, 0.0, 1.0),
            (&quartic, 0.81, 0.17),
            (&split, 0.0, 5.0),
            (&split, 4.3, 0.9),
        ] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
            let eb = edge_on(&mut topo, other, from, to);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings_inside(&topo, ea, eb);
                assert!(hits.is_empty(), "{from}..{to}: {hits:?}");
            }
        }
    }

    fn arch_poles() -> Vec<Point3> {
        arch(0.0).control_points().to_vec()
    }

    /// A circle arc and an ellipse arc, each against the same arc as a
    /// rational quadratic: the edges overlap from 0.2 rad to a quarter turn.
    #[test]
    fn an_arc_and_the_same_arc_as_a_spline_do_not_cross_inside_their_overlap() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::nurbs::curve::NurbsCurve;

        let centre = Point3::new(1.0, -2.0, 0.5);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let quarter = |major: f64, minor: f64| {
            NurbsCurve::new(
                2,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![
                    centre + x * major,
                    centre + x * major + y * minor,
                    centre + y * minor,
                ],
                vec![1.0, w, 1.0],
            )
            .unwrap()
        };
        let circle = Circle3D::new_with_ref(centre, z, 2.0, x).unwrap();
        let ellipse = Ellipse3D::new_with_ref(centre, z, 3.0, 2.0, x).unwrap();
        let on = |major: f64, minor: f64, angle: f64| {
            centre + x * (major * angle.cos()) + y * (minor * angle.sin())
        };
        for (from, to) in [(0.2, 1.9), (0.0, std::f64::consts::FRAC_PI_2)] {
            for (arc, spline, major) in [
                (EdgeCurve::Circle(circle.clone()), quarter(2.0, 2.0), 2.0),
                (EdgeCurve::Ellipse(ellipse.clone()), quarter(3.0, 2.0), 3.0),
            ] {
                let mut topo = Topology::new();
                let ea = add_edge(&mut topo, on(major, 2.0, from), on(major, 2.0, to), arc);
                let eb = edge_on(&mut topo, &spline, 0.0, 1.0);
                let back = edge_on(&mut topo, &spline, 1.0, 0.0);
                for (ea, eb) in [(ea, eb), (eb, ea), (ea, back), (back, ea)] {
                    let hits = crossings_inside(&topo, ea, eb);
                    assert!(hits.is_empty(), "{from}..{to} r {major}: {hits:?}");
                }
            }
        }
    }

    /// A spline that runs along a cubic from 0.2 to 0.7 of it and leaves it
    /// there at a corner. The stretch along the cubic is an overlap, bounded
    /// by the spline's start and by the corner. The corner is the one
    /// crossing.
    #[test]
    fn a_spline_leaving_a_cubic_at_a_corner_crosses_it_only_there() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let curve = arch(0.0);
        let mut poles = bezier_part(&arch_poles(), 0.2, 0.7);
        let corner = poles[3];
        let away = Vec3::new(0.3, 1.0, 0.8);
        poles.extend((1..=3).map(|k| corner + away * f64::from(k)));
        let leaving = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            poles,
            vec![1.0; 7],
        )
        .unwrap();
        for (from, to) in [(0.0, 2.0), (2.0, 0.0)] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
            let eb = edge_on(&mut topo, &leaving, from, to);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{hits:?}");
                assert!((hits[0].2 - corner).length() < 1e-7, "{hits:?}");
            }
        }
    }

    /// A spline that runs along a cubic from 0.2 to 0.7 of it and leaves it
    /// there with no corner. The stretch along the cubic is an overlap, and
    /// the one crossing is where the spline leaves.
    #[test]
    fn a_spline_leaving_a_cubic_smoothly_meets_it_where_it_leaves() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let curve = arch(0.0);
        let mut poles = bezier_part(&arch_poles(), 0.2, 0.7);
        let leave = poles[3];
        let along = (poles[3] - poles[2]).normalize().unwrap();
        let up = Vec3::new(0.0, 1.0, 0.0);
        poles.extend([
            leave + along,
            leave + along * 2.0 + up * 0.7,
            leave + along * 3.0 + up * 2.0,
        ]);
        let leaving = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            poles,
            vec![1.0; 7],
        )
        .unwrap();
        let tol = Tolerance::default().linear;
        for (from, to) in [(0.0, 2.0), (2.0, 0.0)] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
            let eb = edge_on(&mut topo, &leaving, from, to);
            for (ea, eb, on_arch) in [(ea, eb, 0), (eb, ea, 1)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{hits:?}");
                let (t_arch, p) = ([hits[0].0, hits[0].1][on_arch], hits[0].2);
                // Where the spline is still on the cubic to a thousandth of
                // the tolerance: at the end of the shared stretch, or a
                // little to one side of it.
                assert!((t_arch - 0.7).abs() < 1e-3, "{hits:?}");
                assert!((p - curve.evaluate(t_arch)).length() <= tol, "{hits:?}");
                assert!((p - leave).length() < 5e-3, "{hits:?}");
            }
        }
    }

    /// A parabola that touches a line 2000 long, 0.001 from the line's
    /// start. The place where the parabola leaves the tolerance of the line
    /// is a small part of the line's parameter span from its start, and
    /// far from that start in space. The contact is one crossing.
    #[test]
    fn a_tangency_near_the_start_of_a_long_edge_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let parabola = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![at(-1.0, 1.0), at(0.0, -1.0), at(1.0, 1.0)],
            vec![1.0; 3],
        )
        .unwrap();
        let (start, end) = (at(-0.001, 0.0), at(1999.999, 0.0));
        let straight =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        for long in [EdgeCurve::Line, EdgeCurve::NurbsCurve(straight)] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &parabola, 0.0, 1.0);
            let eb = add_edge(&mut topo, start, end, long);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{hits:?}");
                assert!((hits[0].2 - at(0.0, 0.0)).length() < 1e-3, "{hits:?}");
            }
        }
    }

    /// A curve `y = x^10` touches the line `y = 0` at one place. It stays
    /// within the tolerance of the line for 0.2 each side, and within a
    /// thousandth of the tolerance for 0.1 each side: half of the stretch.
    /// The two do not share a curve, and the contact is one crossing.
    #[test]
    fn a_contact_of_high_order_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        // `(2u - 1)^10` in powers of `u`, then as Bezier ordinates.
        let binomial =
            |n: u32, k: u32| (0..k).fold(1.0, |c, i| c * f64::from(n - i) / f64::from(i + 1));
        let power: Vec<f64> = (0..=10)
            .map(|k| binomial(10, k) * f64::from(1u32 << k) * if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let poles: Vec<Point3> = (0..=10u32)
            .map(|i| {
                let y: f64 = (0..=i)
                    .map(|k| binomial(i, k) / binomial(10, k) * power[k as usize])
                    .sum();
                Point3::new(-1.0 + 0.2 * f64::from(i), y, 0.0)
            })
            .collect();
        let mut knots = vec![0.0; 11];
        knots.extend([1.0; 11]);
        let flat = NurbsCurve::new(10, knots, poles, vec![1.0; 11]).unwrap();
        assert!((flat.evaluate(0.75).y() - 0.5f64.powi(10)).abs() < 1e-12);

        let (start, end) = (Point3::new(-1.3, 0.0, 0.0), Point3::new(1.1, 0.0, 0.0));
        let straight =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        let tol = Tolerance::default().linear;
        for line in [EdgeCurve::Line, EdgeCurve::NurbsCurve(straight)] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &flat, 0.0, 1.0);
            let eb = add_edge(&mut topo, start, end, line);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{hits:?}");
                let p = hits[0].2;
                assert!(p.x().abs() < 0.2 && p.y().abs() <= tol, "{hits:?}");
            }
        }
    }

    /// The product of two polynomials given by their Bezier ordinates.
    fn bezier_product(a: &[f64], b: &[f64]) -> Vec<f64> {
        let binomial =
            |n: usize, k: usize| (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
        let (m, n) = (a.len() - 1, b.len() - 1);
        (0..=m + n)
            .map(|k| {
                (0..=m)
                    .filter(|&i| k >= i && k - i <= n)
                    .map(|i| binomial(m, i) * binomial(n, k - i) * a[i] * b[k - i])
                    .sum::<f64>()
                    / binomial(m + n, k)
            })
            .collect()
    }

    /// A parabola that touches a line, with most of its parameter span
    /// spent next to the place where they touch: `x = 1e-6 s / (1 - s^2 +
    /// 1e-6)` and `y = x^2` for `s` from -1 to 1. At a thousandth of the
    /// tolerance, the stretch keeps 0.95 of its parameter span and 0.03 of
    /// its length. The two do not share a curve, and the contact is one
    /// crossing.
    #[test]
    fn a_tangency_that_fills_its_parameter_span_is_read_once() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let s = [-1.0, 1.0];
        let w = [1e-6, 2.0 + 1e-6, 1e-6];
        let weights = bezier_product(&w, &w);
        let x: Vec<f64> = bezier_product(&bezier_product(&s, &w), &[1.0, 1.0])
            .iter()
            .map(|c| c * 1e-6)
            .collect();
        let y: Vec<f64> = bezier_product(&bezier_product(&s, &s), &[1.0, 1.0, 1.0])
            .iter()
            .map(|c| c * 1e-12)
            .collect();
        assert!(weights.iter().all(|&weight| weight > 0.0));
        let poles: Vec<Point3> = (0..5)
            .map(|i| Point3::new(x[i] / weights[i], y[i] / weights[i], 0.0))
            .collect();
        let mut knots = vec![0.0; 5];
        knots.extend([1.0; 5]);
        let parabola = NurbsCurve::new(4, knots, poles, weights).unwrap();
        let p = parabola.evaluate(0.875);
        assert!((p.y() - p.x() * p.x()).abs() < 1e-15 && p.x() > 1e-6);

        let (start, end) = (Point3::new(-1.2, 0.0, 0.0), Point3::new(1.3, 0.0, 0.0));
        let straight =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        for line in [EdgeCurve::Line, EdgeCurve::NurbsCurve(straight)] {
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &parabola, 0.0, 1.0);
            let eb = add_edge(&mut topo, start, end, line);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let hits = crossings(&topo, ea, eb);
                assert_eq!(hits.len(), 1, "{hits:?}");
                assert!((hits[0].2 - Point3::new(0.0, 0.0, 0.0)).length() < 4e-4);
            }
        }
    }

    /// A cubic and the same curve with a smooth bump over one narrow span:
    /// from 0.5 to 17/32 of it, between two sample points of both edges,
    /// which all agree, and from 0.5 to 0.50001 of it, 4.5e-5 long. The
    /// edges overlap up to the bump and after it, and each end of the bump
    /// is one crossing.
    #[test]
    fn a_bump_between_two_sample_points_ends_the_overlap_each_side() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let curve = arch(0.0);
        let raised = |part: Vec<Point3>| -> Vec<Point3> {
            (0..5usize)
                .map(|i| {
                    let w = i as f64 / 4.0;
                    let (before, here) = (part[i.saturating_sub(1)], part[i.min(3)]);
                    here + (before - here) * w
                })
                .collect()
        };
        let from = 0.5;
        for to in [17.0 / 32.0, 0.500_01] {
            let mut bump = raised(bezier_part(&arch_poles(), from, to));
            // `1.6 s^2 (1 - s)^2` along z: 0.1 at the middle of the span,
            // and no step in place or direction at its ends.
            bump[2] = bump[2] + Vec3::new(0.0, 0.0, 1.6 / 6.0);
            let mut poles = raised(bezier_part(&arch_poles(), 0.0, from));
            poles.extend(bump.into_iter().skip(1));
            poles.extend(
                raised(bezier_part(&arch_poles(), to, 1.0))
                    .into_iter()
                    .skip(1),
            );
            let mut knots = vec![0.0; 5];
            knots.extend([from; 4]);
            knots.extend([to; 4]);
            knots.extend([1.0; 5]);
            let bumped = NurbsCurve::new(4, knots, poles, vec![1.0; 13]).unwrap();
            assert!((bumped.evaluate(0.3) - curve.evaluate(0.3)).length() < 1e-12);
            assert!((bumped.evaluate(f64::midpoint(from, to)).z() - 0.1).abs() < 1e-12);

            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
            let eb = edge_on(&mut topo, &bumped, 0.0, 1.0);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let mut at: Vec<f64> = crossings(&topo, ea, eb).iter().map(|h| h.0).collect();
                at.sort_by(f64::total_cmp);
                assert_eq!(at.len(), 2, "to {to}: {at:?}");
                let width = to - from;
                assert!(
                    (at[0] - from).abs() < 0.1 * width && (at[1] - to).abs() < 0.1 * width,
                    "to {to}: {at:?}"
                );
            }
        }
    }

    /// Two edges that share both ends and cross between them, and an edge
    /// that starts on a cubic and crosses it further on: crossings of
    /// different curves stay, with stops on one side or on both.
    #[test]
    fn different_curves_between_shared_ends_still_cross() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let curve = arch(0.0);
        let p = arch_poles();
        // The arch with its two inner poles swapped in height: an S through
        // the same ends, crossing the arch once at its middle.
        let s_poles = vec![
            p[0],
            p[1] + Vec3::new(0.0, 1.5, 0.0),
            p[2] - Vec3::new(0.0, 1.5, 0.0),
            p[3],
        ];
        let s_curve = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            s_poles,
            vec![1.0; 4],
        )
        .unwrap();
        let mut topo = Topology::new();
        let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
        let eb = edge_on(&mut topo, &s_curve, 0.0, 1.0);
        for (ea, eb) in [(ea, eb), (eb, ea)] {
            let hits = crossings_inside(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            assert!(
                (hits[0].2 - curve.evaluate(0.5)).length() < 1e-9,
                "{hits:?}"
            );
        }

        // From a place on the arch, straight through it further on.
        let start = curve.evaluate(0.2);
        let through = curve.evaluate(0.6);
        let end = start + (through - start) * 1.5;
        let chord =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![start, end], vec![1.0; 2]).unwrap();
        let eb = edge_on(&mut topo, &chord, 0.0, 1.0);
        for (ea, eb) in [(ea, eb), (eb, ea)] {
            let hits = crossings_inside(&topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            assert!((hits[0].2 - through).length() < 1e-9, "{hits:?}");
        }
    }

    /// Segments whose closest points are one end of the first and a point
    /// inside the second: the distance is that of the end from the second
    /// segment, `sqrt(1.8)`, and the fractions are 1 and 0.2.
    #[test]
    fn segment_distance_reads_an_end_against_the_inside_of_the_other() {
        let p = Point3::new;
        let a = (p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0));
        let b = (p(2.0, -1.0, 0.0), p(3.0, 1.0, 0.0));
        let (s, t, gap) = segment_distance(a, b);
        assert!((gap - 1.8_f64.sqrt()).abs() < 1e-12, "{gap}");
        assert!(
            (s - 1.0).abs() < 1e-12 && (t - 0.2).abs() < 1e-12,
            "{s} {t}"
        );
        let (t, s, gap) = segment_distance(b, a);
        assert!((gap - 1.8_f64.sqrt()).abs() < 1e-12, "{gap}");
        assert!(
            (s - 1.0).abs() < 1e-12 && (t - 0.2).abs() < 1e-12,
            "{s} {t}"
        );
        let crossing = segment_distance(
            (p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0)),
            (p(1.0, -1.0, 0.5), p(1.0, 1.0, 0.5)),
        );
        assert!((crossing.2 - 0.5).abs() < 1e-12, "{crossing:?}");
    }

    /// A line edge, or a degree-1 B-spline on the same line, through the
    /// point of `curve` at 0.6 along `dir`, from `-1` to `1.3` times `dir`
    /// from it, and that point.
    fn lines_through(
        topo: &mut Topology,
        curve: &brepkit_math::nurbs::curve::NurbsCurve,
        dir: Vec3,
    ) -> ([EdgeId; 2], Point3) {
        let through = curve.evaluate(0.6);
        let (from, to) = (through - dir, through + dir * 1.3);
        let spline = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![from, to],
            vec![1.0; 2],
        )
        .unwrap();
        (
            [
                add_edge(topo, from, to, EdgeCurve::Line),
                add_edge(topo, from, to, EdgeCurve::NurbsCurve(spline)),
            ],
            through,
        )
    }

    /// One crossing of `ea` and `eb`, either way round, at `through`.
    fn crosses_once_at(topo: &Topology, ea: EdgeId, eb: EdgeId, through: Point3) {
        for (ea, eb) in [(ea, eb), (eb, ea)] {
            let hits = crossings(topo, ea, eb);
            assert_eq!(hits.len(), 1, "{hits:?}");
            assert!((hits[0].2 - through).length() < 1e-9, "{hits:?}");
        }
    }

    /// A cubic arch in the plane `z = 0`, and a line across that plane
    /// through the arch's point at 0.6, between two sample points. The
    /// chords of the arch stand off it there, and a line out of its plane
    /// passes them at a distance.
    #[test]
    fn a_line_across_the_plane_of_a_cubic_crosses_it_between_samples() {
        let curve = arch(0.0);
        let mut topo = Topology::new();
        let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
        let (lines, through) = lines_through(&mut topo, &curve, Vec3::new(0.3, 0.2, 1.0));
        for eb in lines {
            crosses_once_at(&topo, ea, eb, through);
        }
    }

    /// The arch with its inner poles 0.001 out of its plane, one up and one
    /// down, and a line through its point at 0.6 along `(1, 1, 0)`.
    #[test]
    fn a_line_across_a_cubic_a_hair_out_of_plane_crosses_it_between_samples() {
        let curve = arch(0.001);
        let mut topo = Topology::new();
        let ea = edge_on(&mut topo, &curve, 0.0, 1.0);
        let (lines, through) = lines_through(&mut topo, &curve, Vec3::new(1.0, 1.0, 0.0));
        for eb in lines {
            crosses_once_at(&topo, ea, eb, through);
        }
    }

    /// A cubic `(4 t, s y(t), s t y(t))` with `y(t) = (t - r1) (t - r2)`
    /// meets the x axis at `r1` and at `r2`, both inside its sample chord
    /// from 19/32 to 20/32, and leaves the plane of the axis between them. A
    /// line along the axis crosses it at both points: with `r1` 0.6 and 0.61,
    /// with `r1` at the sample point 19/32, where the chords meet, and with
    /// the scale `s` 0.02, where the curves meet at a sine of 6e-5.
    #[test]
    fn two_crossings_inside_one_pair_of_chords_are_both_read() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        // The Bezier poles of the cubic with these power coefficients.
        let poles = |a: [Vec3; 4]| {
            let at = |v: Vec3| Point3::new(v.x(), v.y(), v.z());
            vec![
                at(a[0]),
                at(a[0] + a[1] * (1.0 / 3.0)),
                at(a[0] + a[1] * (2.0 / 3.0) + a[2] * (1.0 / 3.0)),
                at(a[0] + a[1] + a[2] + a[3]),
            ]
        };
        let v = Vec3::new;
        for (r1, r2, start, scale) in [
            (0.6, 0.61, -1.0, 1.0),
            (19.0 / 32.0, 0.61, -1.02, 1.0),
            (0.6, 0.61, -1.0, 0.02),
        ] {
            let (c0, c1) = (scale * r1 * r2, -scale * (r1 + r2));
            let cubic = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                poles([
                    v(0.0, c0, 0.0),
                    v(4.0, c1, c0),
                    v(0.0, scale, c1),
                    v(0.0, 0.0, scale),
                ]),
                vec![1.0; 4],
            )
            .unwrap();
            let mut topo = Topology::new();
            let ea = edge_on(&mut topo, &cubic, 0.0, 1.0);
            let (from, to) = (
                Point3::new(start, 0.0, 0.0),
                Point3::new(start + 6.0, 0.0, 0.0),
            );
            let eb = add_edge(&mut topo, from, to, EdgeCurve::Line);
            for (ea, eb) in [(ea, eb), (eb, ea)] {
                let mut hits: Vec<f64> = crossings(&topo, ea, eb).iter().map(|h| h.2.x()).collect();
                hits.sort_by(f64::total_cmp);
                assert_eq!(hits.len(), 2, "{r1} {scale}: {hits:?}");
                // The curves meet at a sine of 0.003 times the scale or
                // less, so a point within the tolerance of both can stand
                // that far off along the axis.
                let off = 1e-7 / scale;
                assert!(
                    (hits[0] - 4.0 * r1).abs() < off && (hits[1] - 4.0 * r2).abs() < off,
                    "{r1} {scale}: {hits:?}"
                );
            }
        }
    }

    /// A polyline of degree 1 that jumps back at its knot 0.51, taken twice:
    /// from (0, 0, 0) to (1, 0, 0), then from (0.99, 0.01, 0) to (1.99,
    /// 0.01, 0). A line along y at x = 0.995 crosses both pieces inside the
    /// one sample chord from 0.5 to 0.53125.
    #[test]
    fn a_line_crosses_both_sides_of_a_jump_inside_one_chord() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let p = Point3::new;
        let polyline = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.51, 0.51, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(1.0, 0.0, 0.0),
                p(0.99, 0.01, 0.0),
                p(1.99, 0.01, 0.0),
            ],
            vec![1.0; 4],
        )
        .unwrap();
        let mut topo = Topology::new();
        let ea = edge_on(&mut topo, &polyline, 0.0, 1.0);
        let (from, to) = (p(0.995, -1.0, 0.0), p(0.995, 1.0, 0.0));
        let eb = add_edge(&mut topo, from, to, EdgeCurve::Line);
        for (ea, eb) in [(ea, eb), (eb, ea)] {
            let mut hits: Vec<f64> = crossings(&topo, ea, eb).iter().map(|h| h.2.y()).collect();
            hits.sort_by(f64::total_cmp);
            assert_eq!(hits.len(), 2, "{hits:?}");
            assert!(
                hits[0].abs() < 1e-9 && (hits[1] - 0.01).abs() < 1e-9,
                "{hits:?}"
            );
        }
    }

    /// Two whole circles at right angles that meet at one point, between
    /// the sample points of each.
    #[test]
    fn circles_at_right_angles_cross_between_samples() {
        let a = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let through = a.evaluate(0.1);
        let tangent = Vec3::new(-0.1_f64.sin(), 0.1_f64.cos(), 0.0);
        let (radial, up) = (
            Vec3::new(0.1_f64.cos(), 0.1_f64.sin(), 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        // B's point at 0 lies 0.3 rad round from the point it shares with A.
        let b = Circle3D::new_with_ref(
            through + up * 0.5,
            tangent,
            0.5,
            radial * 0.3_f64.sin() - up * 0.3_f64.cos(),
        )
        .unwrap();
        let mut topo = Topology::new();
        let (start_a, start_b) = (a.evaluate(0.0), b.evaluate(0.0));
        let ea = add_edge(&mut topo, start_a, start_a, EdgeCurve::Circle(a));
        let eb = add_edge(&mut topo, start_b, start_b, EdgeCurve::Circle(b));
        crosses_once_at(&topo, ea, eb, through);
    }

    #[test]
    fn line_crosses_full_circle_at_two_points() {
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let line = line_data(Point3::new(-2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
        let arc = arc_data(0.0, std::f64::consts::TAU);
        let tol = Tolerance::default();

        let mut hits = line_circle_intersection(&line, &arc, &circle, tol, false);
        hits.sort_by(|a, b| a.2.x().partial_cmp(&b.2.x()).unwrap());
        assert_eq!(hits.len(), 2, "line should cross full circle at 2 points");
        assert!((hits[0].2.x() - (-1.0)).abs() < 1e-6, "left hit at x=-1");
        assert!((hits[1].2.x() - 1.0).abs() < 1e-6, "right hit at x=1");
    }

    #[test]
    fn arc_domain_excludes_out_of_range_hit() {
        // Quarter arc [0, π/2] only contains the +X crossing (angle 0), not the
        // -X crossing (angle π).
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let line = line_data(Point3::new(-2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
        let arc = arc_data(0.0, std::f64::consts::FRAC_PI_2);
        let tol = Tolerance::default();

        let hits = line_circle_intersection(&line, &arc, &circle, tol, false);
        assert_eq!(hits.len(), 1, "only the in-domain crossing should survive");
        assert!((hits[0].2.x() - 1.0).abs() < 1e-6, "surviving hit at x=1");
    }

    #[test]
    fn order_is_preserved_when_circle_is_edge_a() {
        // When the circle is edge A, the triple must be (t_arc, t_line, point).
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let line = line_data(Point3::new(1.0, -2.0, 0.0), Point3::new(1.0, 2.0, 0.0));
        let arc = arc_data(0.0, std::f64::consts::TAU);
        let tol = Tolerance::default();

        // Line x=1 is tangent to the circle at (1,0,0), angle 0.
        let hits = line_circle_intersection(&line, &arc, &circle, tol, true);
        assert!(!hits.is_empty(), "tangent line should touch the circle");
        // t_arc (first slot) ≈ 0 (angle at (1,0,0)); t_line (second slot) ≈ 0.5
        // (midpoint of the y-segment).
        assert!(hits[0].0.abs() < 1e-4, "t_arc should be the angle ~0");
        assert!(
            (hits[0].1 - 0.5).abs() < 1e-4,
            "t_line should be ~0.5 (segment midpoint)"
        );
    }

    #[test]
    fn line_misses_circle() {
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        // Line well outside the circle radius.
        let line = line_data(Point3::new(-2.0, 5.0, 0.0), Point3::new(2.0, 5.0, 0.0));
        let arc = arc_data(0.0, std::f64::consts::TAU);
        let tol = Tolerance::default();

        let hits = line_circle_intersection(&line, &arc, &circle, tol, false);
        assert!(hits.is_empty(), "line far from circle should not intersect");
    }
}
