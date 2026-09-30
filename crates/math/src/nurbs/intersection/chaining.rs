//! Chaining intersection points into curves.

use crate::MathError;
use crate::nurbs::fitting::{approximate_lspia, chord_length_params, interpolate};
use crate::nurbs::projection::project_point_to_curve;
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
                if (first_i - last_i).length() < 1e-12 {
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

/// Fit a NURBS curve through each ALREADY-ORDERED chain of points.
///
/// The proximity chainer above cannot serve callers that know connectivity
/// exactly (the plane x NURBS grid scan links crossings through shared
/// cells): its distance threshold cannot separate a sampling gap along one
/// branch from two genuinely-close branches, and no statistic fixes that.
pub(super) fn build_curves_from_chains(
    chains: &[Vec<IntersectionPoint>],
) -> Result<Vec<IntersectionCurve>, MathError> {
    let mut curves = Vec::with_capacity(chains.len());

    for chain in chains {
        // Deduplicate closely spaced points within the chain.
        let mut deduped: Vec<IntersectionPoint> = Vec::new();
        for pt in chain {
            let is_dup = deduped
                .last()
                .is_some_and(|last: &IntersectionPoint| (last.point - pt.point).length() < 1e-6);
            if !is_dup {
                deduped.push(*pt);
            }
        }

        if deduped.len() < 2 {
            continue;
        }

        // Fit a NURBS curve through this chain's points.
        let positions: Vec<Point3> = deduped.iter().map(|p| p.point).collect();
        let degree = if positions.len() <= 3 {
            1
        } else {
            3.min(positions.len() - 1)
        };
        let curve = if positions.len() > 50 {
            let num_cps = (positions.len() / 3).max(degree + 1).min(positions.len());
            let fitted = approximate_lspia(&positions, degree, num_cps, 1e-6, 100)?;

            // Validate fit quality: re-evaluate residual at each sample point
            // using the same chord-length parameterisation used during fitting.
            // Use a relative threshold (residual / point-cloud diagonal) so the
            // check is scale-independent.  A relative residual > 1% warrants a
            // warning; the intersection curve may be geometrically inaccurate.
            let fit_params = chord_length_params(&positions);
            let mut max_residual = 0.0f64;
            let mut bbox_min = positions[0];
            let mut bbox_max = positions[0];
            for (i, &t) in fit_params.iter().enumerate() {
                let src = positions[i];
                // Nearest-point projection gives the true geometric residual.
                // Fall back to parametric evaluation only for degenerate curves.
                let d = if let Ok(proj) = project_point_to_curve(&fitted, src, 1e-6) {
                    proj.distance
                } else {
                    let pt = fitted.evaluate(t);
                    (pt.x() - src.x()).hypot((pt.y() - src.y()).hypot(pt.z() - src.z()))
                };
                max_residual = max_residual.max(d);
                bbox_min = Point3::new(
                    bbox_min.x().min(src.x()),
                    bbox_min.y().min(src.y()),
                    bbox_min.z().min(src.z()),
                );
                bbox_max = Point3::new(
                    bbox_max.x().max(src.x()),
                    bbox_max.y().max(src.y()),
                    bbox_max.z().max(src.z()),
                );
            }
            let diagonal = (bbox_max.x() - bbox_min.x())
                .hypot((bbox_max.y() - bbox_min.y()).hypot(bbox_max.z() - bbox_min.z()));
            let rel_residual = if diagonal > 1e-12 {
                max_residual / diagonal
            } else {
                max_residual
            };
            if rel_residual > 1e-2 {
                log::warn!(
                    "SSI: LSPIA fit relative residual {rel_residual:.2e} (abs={max_residual:.2e}) \
                     exceeds 1% of curve extent — intersection curve may be inaccurate \
                     (degree={degree}, num_cps={num_cps}, samples={})",
                    positions.len()
                );
            }
            fitted
        } else {
            interpolate(&positions, degree)?
        };

        curves.push(IntersectionCurve {
            curve,
            points: deduped,
        });
    }

    Ok(curves)
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
