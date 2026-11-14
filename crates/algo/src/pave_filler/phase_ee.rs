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

use super::helpers::{add_pave_to_edge, find_nearby_pave_vertex};
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

    let n: usize = 32;
    let mut crossings = Vec::new();

    let pts_a: Vec<SegmentEndpoint> = (0..=n)
        .map(|i| {
            let t = ea.t0 + (ea.t1 - ea.t0) * (i as f64 / n as f64);
            let pos = edge_a
                .curve()
                .evaluate_with_endpoints(t, ea.start_pos, ea.end_pos);
            SegmentEndpoint { t, pos }
        })
        .collect();

    let pts_b: Vec<SegmentEndpoint> = (0..=n)
        .map(|i| {
            let t = eb.t0 + (eb.t1 - eb.t0) * (i as f64 / n as f64);
            let pos = edge_b
                .curve()
                .evaluate_with_endpoints(t, eb.start_pos, eb.end_pos);
            SegmentEndpoint { t, pos }
        })
        .collect();

    let domain_a = (ea.t0.min(ea.t1) - tol.linear)..=(ea.t0.max(ea.t1) + tol.linear);
    let domain_b = (eb.t0.min(eb.t1) - tol.linear)..=(eb.t0.max(eb.t1) + tol.linear);

    // Two sample chords meet at a point the curves miss by the chords' sag;
    // that point disagrees with the exact crossing the EF and FF phases
    // compute on the same curves, and the two copies of one vertex then
    // bound a sliver edge.
    let read = |crossings: &mut Vec<(f64, f64, Point3)>, hit: &ChordApproach| {
        let (t_a, t_b, pt) = refine_on_curves(
            (edge_a.curve(), ea),
            (edge_b.curve(), eb),
            (hit.t_a, hit.t_b),
            tol.linear,
        )
        .unwrap_or((hit.t_a, hit.t_b, hit.point));
        let is_dup = crossings
            .iter()
            .any(|&(ct_a, ct_b, _): &(f64, f64, Point3)| {
                (t_a - ct_a).abs() < 1e-6 && (t_b - ct_b).abs() < 1e-6
            });
        if !is_dup {
            crossings.push((t_a, t_b, pt));
        }
    };

    let mut alongside: Vec<(usize, usize)> = Vec::new();
    let mut alongside_hits: Vec<ChordApproach> = Vec::new();
    let mut crossed: Vec<Option<f64>> = vec![None; n * n];
    for i in 0..n {
        for j in 0..n {
            // Quick distance check: if minimum endpoint distance exceeds
            // the sum of segment lengths + tolerance, skip.
            let min_dist = (pts_a[i].pos - pts_b[j].pos)
                .length()
                .min((pts_a[i].pos - pts_b[j + 1].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j + 1].pos).length());

            let seg_len_a = (pts_a[i + 1].pos - pts_a[i].pos).length();
            let seg_len_b = (pts_b[j + 1].pos - pts_b[j].pos).length();

            if min_dist > seg_len_a + seg_len_b + tol.linear {
                continue;
            }

            if let Some(hit) = closest_segment_pair(
                [&pts_a[i], &pts_a[i + 1]],
                [&pts_b[j], &pts_b[j + 1]],
                tol.linear,
            ) && domain_a.contains(&hit.t_a)
                && domain_b.contains(&hit.t_b)
            {
                if hit.parallel {
                    alongside.push((i, j));
                    alongside_hits.push(hit);
                } else {
                    crossed[i * n + j] = Some(hit.t_a);
                    read(&mut crossings, &hit);
                }
            }
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

    Ok(crossings)
}

/// Gauss-Newton on `Ca(ta) - Cb(tb)` from a sampled crossing, inside both
/// edges' domains. `None` when the curves do not meet within `tol` there.
fn refine_on_curves(
    (ca, ea): (&EdgeCurve, &EdgeData),
    (cb, eb): (&EdgeCurve, &EdgeData),
    (mut ta, mut tb): (f64, f64),
    tol: f64,
) -> Option<(f64, f64, Point3)> {
    let at_a = |t: f64| ca.evaluate_with_endpoints(t, ea.start_pos, ea.end_pos);
    let at_b = |t: f64| cb.evaluate_with_endpoints(t, eb.start_pos, eb.end_pos);
    let (lo_a, hi_a) = (ea.t0.min(ea.t1), ea.t0.max(ea.t1));
    let (lo_b, hi_b) = (eb.t0.min(eb.t1), eb.t0.max(eb.t1));
    let (ha, hb) = ((hi_a - lo_a) * 1e-7, (hi_b - lo_b) * 1e-7);
    if ha <= 0.0 || hb <= 0.0 {
        return None;
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
            return None;
        }
        ta = (ta - (a22 * g1 - a12 * g2) / det).clamp(lo_a, hi_a);
        tb = (tb - (a11 * g2 - a12 * g1) / det).clamp(lo_b, hi_b);
    }
    let (pa, pb) = (at_a(ta), at_b(tb));
    ((pa - pb).length() <= tol).then(|| (ta, tb, pa + (pb - pa) * 0.5))
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
        } else if drift2 <= (64.0 * f64::EPSILON * size).powi(2) {
            // A standoff that changes by rounding alone is the same all
            // along the overlap: its middle reads it whichever way it rounds.
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
