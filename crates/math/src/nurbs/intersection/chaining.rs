//! Chaining intersection points into curves.

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::fitting::{chord_length_params, interpolate};
use crate::vec::Point3;

use super::surface_marching::near_existing_segment;
use super::{IntersectionCurve, IntersectionPoint};

/// Assemble the marcher's traced segments into ordered chains.
///
/// Each segment is already ordered along its curve, so its points are never
/// re-chained by proximity: a proximity threshold taken from their spacing
/// collapses where a march refines toward a boundary in short steps, and
/// once widened its nearest-neighbour walk no longer follows the curve. The
/// points of a later segment within `near` of an earlier one (a seed just
/// past that segment's end re-tracing it) are dropped, splitting it where
/// they fall; chains whose ends meet within `join` are then concatenated.
pub(super) fn chain_traced_segments(
    segments: Vec<Vec<IntersectionPoint>>,
    near: f64,
    join: f64,
) -> Vec<Vec<IntersectionPoint>> {
    let mut chains: Vec<Vec<IntersectionPoint>> = Vec::new();
    for segment in segments {
        let mut runs = Vec::new();
        let mut run: Vec<IntersectionPoint> = Vec::new();
        for p in segment {
            if near_existing_segment(&chains, &p, near) {
                if run.len() >= 2 {
                    runs.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            } else {
                run.push(p);
            }
        }
        if run.len() >= 2 {
            runs.push(run);
        }
        chains.extend(runs);
    }
    let meets = |a: Point3, b: Point3| (a - b).length() < join;
    'join: loop {
        for i in 0..chains.len() {
            for j in 0..chains.len() {
                if i == j {
                    continue;
                }
                let (first_i, last_i) = (chains[i][0].point, chains[i][chains[i].len() - 1].point);
                let (first_j, last_j) = (chains[j][0].point, chains[j][chains[j].len() - 1].point);
                // A closed loop joins nothing, on either side of the pair.
                if (first_i - last_i).length() < 1e-12 || (first_j - last_j).length() < 1e-12 {
                    continue;
                }
                let joined = if meets(last_i, first_j) {
                    Some((false, false))
                } else if meets(last_i, last_j) {
                    Some((false, true))
                } else if meets(first_i, first_j) {
                    Some((true, false))
                } else {
                    None
                };
                if let Some((flip_i, flip_j)) = joined {
                    let mut tail = chains.remove(j);
                    let i = if j < i { i - 1 } else { i };
                    if flip_j {
                        tail.reverse();
                    }
                    if flip_i {
                        chains[i].reverse();
                    }
                    chains[i].extend(tail);
                    continue 'join;
                }
            }
        }
        break;
    }
    chains
}

/// A section's two surfaces as a fit sees them, each closure started from
/// a chain point beside the point it is given.
pub(super) struct Section<'a> {
    /// How far a point lies from the farther surface.
    pub off: &'a dyn Fn(Point3, &IntersectionPoint) -> Option<f64>,
    /// The intersection point Newton reaches from a point.
    pub settle: &'a dyn Fn(Point3, &IntersectionPoint) -> Option<IntersectionPoint>,
}

/// Fit a NURBS curve through each ALREADY-ORDERED chain of points.
///
/// The proximity chainer above cannot serve callers that know connectivity
/// exactly (the plane x NURBS grid scan links crossings through shared
/// cells): its distance threshold cannot separate a sampling gap along one
/// branch from two genuinely-close branches, and no statistic fixes that.
/// Each fit holds `tolerance` against `section` (see [`fit_within`]).
pub(super) fn build_curves_from_chains(
    chains: &[Vec<IntersectionPoint>],
    tolerance: f64,
    section: &Section<'_>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let mut curves = Vec::with_capacity(chains.len());

    for chain in chains {
        // Points closer than 1e-6, or than a thousandth of a chain smaller
        // than a millimetre, are one point met twice.
        let span = chain
            .iter()
            .map(|p| (p.point - chain[0].point).length())
            .fold(0.0_f64, f64::max);
        let same = (1e-3 * span).min(1e-6);
        let mut deduped: Vec<IntersectionPoint> = Vec::new();
        for pt in chain {
            let is_dup = deduped
                .last()
                .is_some_and(|last: &IntersectionPoint| (last.point - pt.point).length() < same);
            if !is_dup {
                deduped.push(*pt);
            }
        }

        if deduped.len() < 2 {
            continue;
        }

        let curve = fit_within(&mut deduped, tolerance, section)?;
        curves.push(IntersectionCurve {
            curve,
            points: deduped,
        });
    }

    Ok(curves)
}

/// Most points a chain grows to while its fit is refined.
const MAX_FIT_POINTS: usize = 4096;

/// Most rounds of refinement for one fit; each halves the spacing where the
/// curve strays, which cuts a cubic's error about sixteenfold.
const MAX_FIT_ROUNDS: usize = 8;

/// Interpolate an ordered chain of intersection points, adding points until
/// the curve holds `tolerance` between each pair of neighbours.
///
/// The curve is measured at the quarter, half and three-quarter marks (in
/// its own parameter) between two neighbours: a span bending through an
/// inflection can pass its middle on the section and stray either side of
/// it. Where any mark strays past `tolerance`, the point the middle settles
/// to joins the chain between them, if it lands nearer each of them than
/// they are to each other. A marched chain is accurate at its points, but a
/// cubic through points a fixed step apart strays between them by the
/// fourth power of that step, so a split cuts the stray about sixteenfold.
/// The spline is global, though, and one split can leave a half straying
/// more than its whole did, so a span is left as it is only when two splits
/// have not halved its stray: the chain there is no smooth run of the
/// section (a crossing, a tangency, points out of order), and more points
/// cannot fix it. No round adds more points than `MAX_FIT_POINTS` leaves.
pub(super) fn fit_within(
    chain: &mut Vec<IntersectionPoint>,
    tolerance: f64,
    section: &Section<'_>,
) -> Result<NurbsCurve, MathError> {
    // The strays of the splits that made each span, the latest first; a
    // span left as it is holds zeros.
    let mut split_at = vec![[f64::INFINITY; 2]; chain.len() - 1];
    let mut round = 0;
    loop {
        let positions: Vec<Point3> = chain.iter().map(|p| p.point).collect();
        let degree = if positions.len() <= 3 { 1 } else { 3 };
        let curve = interpolate(&positions, degree)?;
        round += 1;
        if round > MAX_FIT_ROUNDS || chain.len() >= MAX_FIT_POINTS {
            return Ok(curve);
        }
        let mut budget = MAX_FIT_POINTS - chain.len();
        let params = chord_length_params(&positions);
        let mut grown = Vec::with_capacity(chain.len() * 2);
        let mut grown_at = Vec::with_capacity(split_at.len() * 2);
        for ((pair, t), &[last, before]) in chain.windows(2).zip(params.windows(2)).zip(&split_at) {
            grown.push(pair[0]);
            let at = |f: f64| curve.evaluate(t[0] + f * (t[1] - t[0]));
            let stray = if last > 0.0 {
                [(0.25, &pair[0]), (0.5, &pair[0]), (0.75, &pair[1])]
                    .into_iter()
                    .filter_map(|(f, near)| (section.off)(at(f), near))
                    .fold(None, |far: Option<f64>, off| {
                        Some(far.map_or(off, |far| far.max(off)))
                    })
            } else {
                None
            };
            let Some(stray) = stray.filter(|&stray| stray > tolerance) else {
                grown_at.push(if last > 0.0 {
                    [f64::INFINITY; 2]
                } else {
                    [0.0; 2]
                });
                continue;
            };
            let gap = (pair[1].point - pair[0].point).length();
            let split = (budget > 0 && stray <= 0.5 * before)
                .then(|| (section.settle)(at(0.5), &pair[0]))
                .flatten()
                .filter(|q| {
                    (q.point - pair[0].point).length() < gap
                        && (q.point - pair[1].point).length() < gap
                });
            if let Some(q) = split {
                grown.push(q);
                grown_at.extend([[stray, last], [stray, last]]);
                budget -= 1;
            } else {
                grown_at.push([0.0; 2]);
            }
        }
        grown.push(chain[chain.len() - 1]);
        if grown.len() == chain.len() {
            return Ok(curve);
        }
        *chain = grown;
        split_at = grown_at;
    }
}

/// Chain intersection points into connected components using proximity.
///
/// Points within `threshold` distance are considered connected. Returns
/// ordered chains (each chain is a connected component, ordered by
/// nearest-neighbor walk). Closed loops are detected when the last
/// point is within `threshold` of the first.
#[must_use]
pub fn chain_intersection_points(
    points: &[IntersectionPoint],
    threshold: f64,
) -> Vec<Vec<IntersectionPoint>> {
    if points.is_empty() {
        return Vec::new();
    }

    let n = points.len();
    let threshold_sq = threshold * threshold;

    // Build adjacency: for each point, find neighbors within threshold.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = points[i].point - points[j].point;
            if d.x().mul_add(d.x(), d.y().mul_add(d.y(), d.z() * d.z())) < threshold_sq {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // BFS to find connected components.
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            for &neighbor in &adj[idx] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(component);
    }

    // Order each component via nearest-neighbor walk.
    let mut chains = Vec::with_capacity(components.len());
    for comp in &components {
        if comp.is_empty() {
            continue;
        }

        // Find endpoint: a point with degree <= 1 in the adjacency (within component).
        let start_idx = comp
            .iter()
            .copied()
            .min_by_key(|&i| adj[i].iter().filter(|&&j| comp.contains(&j)).count())
            .unwrap_or(comp[0]);

        let mut chain = Vec::with_capacity(comp.len());
        let mut used = vec![false; n];
        let mut current = start_idx;
        used[current] = true;
        chain.push(points[current]);

        for _ in 1..comp.len() {
            // Find nearest unused point in the component.
            let mut best_dist = f64::MAX;
            let mut best_idx = None;
            for &idx in comp {
                if used[idx] {
                    continue;
                }
                let d = points[current].point - points[idx].point;
                let dist_sq = d.x().mul_add(d.x(), d.y().mul_add(d.y(), d.z() * d.z()));
                if dist_sq < best_dist {
                    best_dist = dist_sq;
                    best_idx = Some(idx);
                }
            }

            if let Some(next) = best_idx {
                used[next] = true;
                chain.push(points[next]);
                current = next;
            } else {
                break;
            }
        }

        chains.push(chain);
    }

    chains
}
