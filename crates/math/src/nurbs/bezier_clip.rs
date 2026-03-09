//! Bezier clipping for curve-curve intersection (Sederberg-Nishita 1990).
//!
//! Decomposes NURBS curves into Bezier segments, then uses recursive
//! fat-line clipping to find all intersection points.

#![allow(clippy::similar_names, clippy::suspicious_operation_groupings)]

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::curve_to_bezier_segments;
use crate::vec::{Point3, Vec3};

/// Maximum recursion depth for Bezier clipping.
const MAX_DEPTH: usize = 50;

/// Maximum Newton refinement iterations.
const MAX_NEWTON: usize = 10;

/// Threshold ratio: if clipping removes less than 40%, subdivide instead.
const CLIP_THRESHOLD: f64 = 0.6;

/// A curve-curve intersection result.
#[derive(Debug, Clone, Copy)]
pub struct CurveCurveHit {
    /// Parameter on the first curve.
    pub u1: f64,
    /// Parameter on the second curve.
    pub u2: f64,
    /// The intersection point.
    pub point: Point3,
}

/// A coincident/overlapping interval between two curves.
#[derive(Debug, Clone, Copy)]
pub struct CurveCurveOverlap {
    /// Start parameter on the first curve.
    pub u1_start: f64,
    /// End parameter on the first curve.
    pub u1_end: f64,
    /// Start parameter on the second curve.
    pub u2_start: f64,
    /// End parameter on the second curve.
    pub u2_end: f64,
}

/// Complete result of a curve-curve intersection, including both point
/// intersections and overlapping (coincident) intervals.
#[derive(Debug, Clone)]
pub struct CurveCurveResult {
    /// Isolated intersection points.
    pub hits: Vec<CurveCurveHit>,
    /// Coincident curve intervals (shared sub-arcs).
    pub overlaps: Vec<CurveCurveOverlap>,
}

/// Find all intersections between two NURBS curves using Bezier clipping.
///
/// Returns intersection parameters and points, accurate to `tolerance`.
///
/// # Errors
///
/// Returns an error if curve decomposition fails.
pub fn curve_curve_intersect(
    curve1: &NurbsCurve,
    curve2: &NurbsCurve,
    tolerance: f64,
) -> Result<Vec<CurveCurveHit>, MathError> {
    let result = curve_curve_intersect_full(curve1, curve2, tolerance)?;
    Ok(result.hits)
}

/// Find all intersections between two NURBS curves, including overlapping
/// (coincident) intervals.
///
/// Use this instead of [`curve_curve_intersect`] when you need to detect
/// shared sub-arcs between curves.
///
/// # Errors
///
/// Returns an error if curve decomposition fails.
pub fn curve_curve_intersect_full(
    curve1: &NurbsCurve,
    curve2: &NurbsCurve,
    tolerance: f64,
) -> Result<CurveCurveResult, MathError> {
    let segments1 = curve_to_bezier_segments(curve1)?;
    let segments2 = curve_to_bezier_segments(curve2)?;

    let mut hits = Vec::new();
    let mut overlaps = Vec::new();

    for seg1 in &segments1 {
        let aabb1 = seg1.aabb();
        for seg2 in &segments2 {
            let aabb2 = seg2.aabb();
            if !aabb1.intersects(aabb2) {
                continue;
            }

            let (u1_lo, u1_hi) = seg1.domain();
            let (u2_lo, u2_hi) = seg2.domain();

            bezier_clip_recurse(
                seg1,
                seg2,
                u1_lo,
                u1_hi,
                u2_lo,
                u2_hi,
                tolerance,
                0,
                &mut hits,
                &mut overlaps,
            );
        }
    }

    merge_duplicate_hits(&mut hits, tolerance);
    merge_overlaps(&mut overlaps, tolerance);
    // Remove point hits that fall within an overlap interval.
    if !overlaps.is_empty() {
        hits.retain(|h| {
            !overlaps
                .iter()
                .any(|o| h.u1 >= o.u1_start - tolerance && h.u1 <= o.u1_end + tolerance)
        });
    }
    Ok(CurveCurveResult { hits, overlaps })
}

/// Signed distances of control points to the fat line defined by the first
/// and last control points of the curve.
///
/// Returns `(line_dir, d_values, d_min, d_max, ref_normal)` where `line_dir`
/// is the normalized baseline direction, `d_values` are the signed distances
/// for each control point, `d_min`/`d_max` bound the fat line, and
/// `ref_normal` is the reference normal used for consistent sign convention.
#[allow(clippy::type_complexity)]
fn fat_line(cps: &[Point3]) -> Option<(Vec3, Vec<f64>, f64, f64, Option<Vec3>)> {
    let n = cps.len();
    if n < 2 {
        return None;
    }
    let p0 = cps[0];
    let pn = cps[n - 1];
    let baseline = pn - p0;
    let baseline_len = baseline.length();
    if baseline_len < 1e-30 {
        return None;
    }
    let dir = Vec3::new(
        baseline.x() / baseline_len,
        baseline.y() / baseline_len,
        baseline.z() / baseline_len,
    );

    // Compute signed distance for each control point.
    // Use the cross product magnitude as the signed perpendicular distance.
    //
    // For consistent sign convention in 3D: establish a reference normal from
    // the first non-degenerate cross product, then project all subsequent
    // cross products onto it. This prevents sign flips for nearly-coplanar
    // curves where the dominant component can change due to floating-point noise.
    let ref_normal = cps
        .iter()
        .map(|&pi| (pi - p0).cross(dir))
        .find(|c| c.length() > 1e-20);

    let dists: Vec<f64> = cps
        .iter()
        .map(|&pi| {
            let v = pi - p0;
            let cross = v.cross(dir);
            let len = cross.length();
            let sign = match ref_normal {
                Some(ref_n) => {
                    if cross.dot(ref_n) >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                None => dominant_sign(cross),
            };
            len * sign
        })
        .collect();

    let d_min = dists.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    let d_max = dists
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    Some((dir, dists, d_min, d_max, ref_normal))
}

/// Return +1.0 or -1.0 based on the dominant component of the cross product.
fn dominant_sign(cross: Vec3) -> f64 {
    let ax = cross.x().abs();
    let ay = cross.y().abs();
    let az = cross.z().abs();
    let val = if ax >= ay && ax >= az {
        cross.x()
    } else if ay >= az {
        cross.y()
    } else {
        cross.z()
    };
    if val >= 0.0 { 1.0 } else { -1.0 }
}

/// Clip the parameter interval of `curve_b` against the fat line of
/// `curve_a`. Returns the new `(t_lo, t_hi)` interval for B, or `None`
/// if no intersection is possible.
#[allow(clippy::too_many_lines)]
fn clip_to_fat_line(
    cps_a: &[Point3],
    cps_b: &[Point3],
    t_b_lo: f64,
    t_b_hi: f64,
) -> Option<(f64, f64)> {
    let (dir, _dists_a, d_min, d_max, ref_normal_a) = fat_line(cps_a)?;

    let p0_a = cps_a[0];
    let n_b = cps_b.len();
    if n_b < 2 {
        return None;
    }

    // Compute signed distances of B's control points to A's fat line.
    // CRITICAL: use the SAME reference normal that fat_line used for A's own
    // control points. If we compute a different reference normal from B's
    // points, the sign convention can flip, making d_min/d_max bounds
    // inconsistent with B's distance values.
    //
    // When A's ref_normal is None (e.g. A is a straight line where all
    // points lie on the baseline), we fall back to dominant_sign per-point.
    // Do NOT try to establish a different reference from B's points — that
    // can produce a sign polarity opposite to what the convex hull clip
    // expects relative to d_min/d_max = 0.
    let ref_normal_b = ref_normal_a;

    #[allow(clippy::cast_precision_loss)]
    let dist_pts: Vec<(f64, f64)> = cps_b
        .iter()
        .enumerate()
        .map(|(i, &pi)| {
            let v = pi - p0_a;
            let cross = v.cross(dir);
            let len = cross.length();
            let sign = match ref_normal_b {
                Some(ref_n) => {
                    if cross.dot(ref_n) >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                None => dominant_sign(cross),
            };
            let t = t_b_lo + (t_b_hi - t_b_lo) * (i as f64) / ((n_b - 1) as f64);
            (t, len * sign)
        })
        .collect();

    // Find the parameter interval where the convex hull of dist_pts
    // intersects the band [d_min, d_max].
    convex_hull_clip(&dist_pts, d_min, d_max)
}

/// Clip the convex hull of a set of (t, d) points against the horizontal
/// band `[d_min, d_max]`. Returns the parameter interval `(t_lo, t_hi)`
/// where the convex hull lies within the band, or `None` if no overlap.
fn convex_hull_clip(pts: &[(f64, f64)], d_min: f64, d_max: f64) -> Option<(f64, f64)> {
    // Build upper and lower convex hulls of the (t, d) polygon.
    let upper = upper_hull(pts);
    let lower = lower_hull(pts);

    // Find the t-range where the hull is between d_min and d_max.
    // The upper hull must be >= d_min and the lower hull must be <= d_max
    // for there to be overlap.
    let t_min_from_upper = intersect_hull_with_line(&upper, d_min, true);
    let t_min_from_lower = intersect_hull_with_line(&lower, d_min, true);
    let t_max_from_upper = intersect_hull_with_line(&upper, d_max, false);
    let t_max_from_lower = intersect_hull_with_line(&lower, d_max, false);

    let t_lo = match (t_min_from_upper, t_min_from_lower) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return None,
    };

    let t_hi = match (t_max_from_upper, t_max_from_lower) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return None,
    };

    if t_lo > t_hi {
        return None;
    }

    Some((t_lo, t_hi))
}

/// Compute the upper convex hull of a set of (t, d) points sorted by t.
fn upper_hull(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut hull = Vec::with_capacity(pts.len());
    for &p in pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) >= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

/// Compute the lower convex hull of a set of (t, d) points sorted by t.
fn lower_hull(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut hull = Vec::with_capacity(pts.len());
    for &p in pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

/// 2D cross product for convex hull computation.
fn cross_2d(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
}

/// Find the first (or last) t where the hull crosses a horizontal line at `d`.
///
/// If `find_min` is true, returns the smallest t where the hull enters the
/// feasible region. If false, returns the largest t.
fn intersect_hull_with_line(hull: &[(f64, f64)], d: f64, find_min: bool) -> Option<f64> {
    if hull.is_empty() {
        return None;
    }

    let mut result: Option<f64> = None;

    // Check each edge of the hull for intersection with the line y = d.
    for window in hull.windows(2) {
        let (t0, d0) = window[0];
        let (t1, d1) = window[1];

        // Does this edge cross d?
        if (d0 - d) * (d1 - d) <= 0.0 {
            let dd = d1 - d0;
            let t = if dd.abs() < 1e-30 {
                if find_min { t0.min(t1) } else { t0.max(t1) }
            } else {
                t0 + (d - d0) * (t1 - t0) / dd
            };

            result =
                Some(result.map_or(t, |prev| if find_min { prev.min(t) } else { prev.max(t) }));
        }
    }

    // Also check individual hull points that lie exactly in the band.
    for &(t, di) in hull {
        if find_min && di >= d {
            result = Some(result.map_or(t, |prev| prev.min(t)));
        }
        if !find_min && di <= d {
            result = Some(result.map_or(t, |prev| prev.max(t)));
        }
    }

    result
}

/// Recursion depth at which we start checking for overlap instead of
/// continuing to subdivide fruitlessly.
const OVERLAP_CHECK_DEPTH: usize = 8;

/// Fat line thickness below which we consider the curve degenerate
/// (collinear control points). Triggers immediate overlap detection.
const DEGENERATE_FAT_LINE: f64 = 1e-12;

/// Number of samples for approximate Hausdorff distance check.
const HAUSDORFF_SAMPLES: usize = 5;

/// Recursive Bezier clipping core.
#[allow(clippy::too_many_arguments)]
fn bezier_clip_recurse(
    seg_a: &NurbsCurve,
    seg_b: &NurbsCurve,
    u_a_lo: f64,
    u_a_hi: f64,
    u_b_lo: f64,
    u_b_hi: f64,
    tolerance: f64,
    depth: usize,
    hits: &mut Vec<CurveCurveHit>,
    overlaps: &mut Vec<CurveCurveOverlap>,
) {
    // Base case: both intervals are small enough.
    let span_a = u_a_hi - u_a_lo;
    let span_b = u_b_hi - u_b_lo;

    if span_a < tolerance && span_b < tolerance {
        let u1_mid = 0.5 * (u_a_lo + u_a_hi);
        let u2_mid = 0.5 * (u_b_lo + u_b_hi);
        if let Some(hit) = newton_refine(seg_a, seg_b, u1_mid, u2_mid, tolerance) {
            hits.push(hit);
        } else {
            // Newton failed; use midpoint approximation.
            let pt = seg_a.evaluate(u1_mid);
            hits.push(CurveCurveHit {
                u1: u1_mid,
                u2: u2_mid,
                point: pt,
            });
        }
        return;
    }

    if depth >= MAX_DEPTH {
        // Before giving up, check for coincident overlap.
        if check_overlap(
            seg_a, seg_b, u_a_lo, u_a_hi, u_b_lo, u_b_hi, tolerance, overlaps,
        ) {
            return;
        }
        // Not coincident — report current best guess as a point hit.
        let u1_mid = 0.5 * (u_a_lo + u_a_hi);
        let u2_mid = 0.5 * (u_b_lo + u_b_hi);
        let pt = seg_a.evaluate(u1_mid);
        hits.push(CurveCurveHit {
            u1: u1_mid,
            u2: u2_mid,
            point: pt,
        });
        return;
    }

    // AABB check for early exit.
    let aabb_a = sub_aabb(seg_a, u_a_lo, u_a_hi);
    let aabb_b = sub_aabb(seg_b, u_b_lo, u_b_hi);
    if !aabb_a.intersects(aabb_b) {
        return;
    }

    let cps_a = seg_a.control_points();
    let cps_b = seg_b.control_points();

    // Early overlap detection: if both fat lines are degenerate (near-zero
    // thickness), the curves are collinear. Check for overlap immediately
    // instead of subdividing 2^30 times.
    if depth <= 2 {
        if let Some((_, _, d_min_a, d_max_a, _)) = fat_line(cps_a) {
            if (d_max_a - d_min_a) < DEGENERATE_FAT_LINE {
                if let Some((_, _, d_min_b, d_max_b, _)) = fat_line(cps_b) {
                    if (d_max_b - d_min_b) < DEGENERATE_FAT_LINE {
                        // Both curves are essentially straight lines — check overlap.
                        if check_overlap(
                            seg_a, seg_b, u_a_lo, u_a_hi, u_b_lo, u_b_hi, tolerance, overlaps,
                        ) {
                            return;
                        }
                    }
                }
            }
        }
    }

    // Try clipping B against A's fat line.
    if let Some((new_b_lo, new_b_hi)) = clip_to_fat_line(cps_a, cps_b, u_b_lo, u_b_hi) {
        let new_span_b = new_b_hi - new_b_lo;
        let ratio = if span_b > 1e-30 {
            new_span_b / span_b
        } else {
            1.0
        };

        if ratio < CLIP_THRESHOLD {
            // Good clip: recurse with swapped roles (clip A against B next).
            bezier_clip_recurse(
                seg_b,
                seg_a,
                new_b_lo,
                new_b_hi,
                u_a_lo,
                u_a_hi,
                tolerance,
                depth + 1,
                hits,
                overlaps,
            );
            return;
        }

        // Try clipping A against B's fat line.
        if let Some((new_a_lo, new_a_hi)) = clip_to_fat_line(cps_b, cps_a, u_a_lo, u_a_hi) {
            let new_span_a = new_a_hi - new_a_lo;
            let ratio_a = if span_a > 1e-30 {
                new_span_a / span_a
            } else {
                1.0
            };

            if ratio_a < CLIP_THRESHOLD {
                bezier_clip_recurse(
                    seg_a,
                    seg_b,
                    new_a_lo,
                    new_a_hi,
                    new_b_lo,
                    new_b_hi,
                    tolerance,
                    depth + 1,
                    hits,
                    overlaps,
                );
                return;
            }
        }

        // Neither clip was effective. At high depth, check for overlap before
        // subdividing further — coincident curves will never clip effectively.
        if depth >= OVERLAP_CHECK_DEPTH
            && check_overlap(
                seg_a, seg_b, u_a_lo, u_a_hi, new_b_lo, new_b_hi, tolerance, overlaps,
            )
        {
            return;
        }

        // Subdivide the longer interval.
        subdivide_and_recurse(
            seg_a, seg_b, u_a_lo, u_a_hi, new_b_lo, new_b_hi, tolerance, depth, hits, overlaps,
        );
    }
    // If clip_to_fat_line returned None, no intersection in this pair.
}

/// Check if two curve segments are coincident over the given parameter
/// intervals. Samples points on curve A and checks their distance to
/// curve B. If the maximum distance (approximate Hausdorff distance)
/// is below tolerance, emits an overlap and returns `true`.
#[allow(clippy::too_many_arguments)]
fn check_overlap(
    seg_a: &NurbsCurve,
    seg_b: &NurbsCurve,
    u_a_lo: f64,
    u_a_hi: f64,
    u_b_lo: f64,
    u_b_hi: f64,
    tolerance: f64,
    overlaps: &mut Vec<CurveCurveOverlap>,
) -> bool {
    let span_a = u_a_hi - u_a_lo;
    let span_b = u_b_hi - u_b_lo;

    // Don't classify tiny intervals as overlaps — those are point
    // intersections where both curves happen to be close near a crossing.
    // Overlap requires the curves to be coincident over a meaningful arc
    // length, so the 3D extent must exceed a multiple of tolerance.
    let pa_lo = seg_a.evaluate(u_a_lo);
    let pa_hi = seg_a.evaluate(u_a_hi);
    let arc_a = (pa_hi - pa_lo).length();
    if arc_a < tolerance * 50.0 && span_a < tolerance * 100.0 && span_b < tolerance * 100.0 {
        return false;
    }

    // Sample points on A and find closest points on B (symmetric Hausdorff).
    let mut max_dist = 0.0_f64;
    #[allow(clippy::cast_precision_loss)]
    for i in 0..=HAUSDORFF_SAMPLES {
        let t_a = u_a_lo + (u_a_hi - u_a_lo) * (i as f64) / (HAUSDORFF_SAMPLES as f64);
        let pa = seg_a.evaluate(t_a);

        let mut best_dist = f64::MAX;
        #[allow(clippy::cast_precision_loss)]
        for j in 0..=HAUSDORFF_SAMPLES {
            let t_b = u_b_lo + (u_b_hi - u_b_lo) * (j as f64) / (HAUSDORFF_SAMPLES as f64);
            let pb = seg_b.evaluate(t_b);
            best_dist = best_dist.min((pa - pb).length());
        }
        max_dist = max_dist.max(best_dist);
    }

    #[allow(clippy::cast_precision_loss)]
    for i in 0..=HAUSDORFF_SAMPLES {
        let t_b = u_b_lo + (u_b_hi - u_b_lo) * (i as f64) / (HAUSDORFF_SAMPLES as f64);
        let pb = seg_b.evaluate(t_b);

        let mut best_dist = f64::MAX;
        #[allow(clippy::cast_precision_loss)]
        for j in 0..=HAUSDORFF_SAMPLES {
            let t_a = u_a_lo + (u_a_hi - u_a_lo) * (j as f64) / (HAUSDORFF_SAMPLES as f64);
            let pa = seg_a.evaluate(t_a);
            best_dist = best_dist.min((pb - pa).length());
        }
        max_dist = max_dist.max(best_dist);
    }

    if max_dist < tolerance * 10.0 {
        overlaps.push(CurveCurveOverlap {
            u1_start: u_a_lo,
            u1_end: u_a_hi,
            u2_start: u_b_lo,
            u2_end: u_b_hi,
        });
        true
    } else {
        false
    }
}

/// Subdivide the longer curve at the midpoint and recurse on both halves.
#[allow(clippy::too_many_arguments)]
fn subdivide_and_recurse(
    seg_a: &NurbsCurve,
    seg_b: &NurbsCurve,
    u_a_lo: f64,
    u_a_hi: f64,
    u_b_lo: f64,
    u_b_hi: f64,
    tolerance: f64,
    depth: usize,
    hits: &mut Vec<CurveCurveHit>,
    overlaps: &mut Vec<CurveCurveOverlap>,
) {
    let span_a = u_a_hi - u_a_lo;
    let span_b = u_b_hi - u_b_lo;

    if span_a > span_b {
        let mid = 0.5 * (u_a_lo + u_a_hi);
        bezier_clip_recurse(
            seg_a,
            seg_b,
            u_a_lo,
            mid,
            u_b_lo,
            u_b_hi,
            tolerance,
            depth + 1,
            hits,
            overlaps,
        );
        bezier_clip_recurse(
            seg_a,
            seg_b,
            mid,
            u_a_hi,
            u_b_lo,
            u_b_hi,
            tolerance,
            depth + 1,
            hits,
            overlaps,
        );
    } else {
        let mid = 0.5 * (u_b_lo + u_b_hi);
        bezier_clip_recurse(
            seg_a,
            seg_b,
            u_a_lo,
            u_a_hi,
            u_b_lo,
            mid,
            tolerance,
            depth + 1,
            hits,
            overlaps,
        );
        bezier_clip_recurse(
            seg_a,
            seg_b,
            u_a_lo,
            u_a_hi,
            mid,
            u_b_hi,
            tolerance,
            depth + 1,
            hits,
            overlaps,
        );
    }
}

/// Compute a conservative AABB for a sub-interval of a curve by sampling.
/// For Bezier segments the full AABB is already tight from control points,
/// but for sub-intervals we sample densely.
fn sub_aabb(curve: &NurbsCurve, u_lo: f64, u_hi: f64) -> crate::aabb::Aabb3 {
    const N_SAMPLES: usize = 8;
    let mut pts = Vec::with_capacity(N_SAMPLES + 1);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..=N_SAMPLES {
        let t = u_lo + (u_hi - u_lo) * (i as f64) / (N_SAMPLES as f64);
        pts.push(curve.evaluate(t));
    }
    crate::aabb::Aabb3::from_points(pts)
}

/// Newton-Raphson refinement for a curve-curve intersection.
///
/// Given approximate parameters `(u1, u2)`, refine to find the exact
/// intersection. Uses a 2x2 least-squares projection from 3D.
fn newton_refine(
    curve_a: &NurbsCurve,
    curve_b: &NurbsCurve,
    mut u1: f64,
    mut u2: f64,
    tolerance: f64,
) -> Option<CurveCurveHit> {
    let (a_lo, a_hi) = curve_a.domain();
    let (b_lo, b_hi) = curve_b.domain();

    for _ in 0..MAX_NEWTON {
        let pa = curve_a.evaluate(u1);
        let pb = curve_b.evaluate(u2);
        let f = pa - pb; // Vec3

        if f.length() < tolerance {
            return Some(CurveCurveHit { u1, u2, point: pa });
        }

        let da = curve_a.derivatives(u1, 1);
        let db = curve_b.derivatives(u2, 1);
        let t1 = da[1]; // tangent of curve A
        let t2 = db[1]; // tangent of curve B

        // Solve the 3x2 system [t1 | -t2] * [du1, du2]^T = -f
        // via normal equations: J^T J delta = -J^T f
        let j11 = t1.dot(t1);
        let j12 = -t1.dot(t2);
        let j22 = t2.dot(t2);

        let r1 = -t1.dot(f);
        let r2 = t2.dot(f);

        let det = j11 * j22 - j12 * j12;
        if det.abs() < 1e-30 {
            return None; // Degenerate (parallel tangents at this point).
        }

        let du1 = (j22 * r1 - j12 * r2) / det;
        let du2 = (-j12).mul_add(r1, j11 * r2) / det;

        u1 = (u1 + du1).clamp(a_lo, a_hi);
        u2 = (u2 + du2).clamp(b_lo, b_hi);
    }

    // Check convergence after max iterations.
    let pa = curve_a.evaluate(u1);
    let pb = curve_b.evaluate(u2);
    if (pa - pb).length() < tolerance * 10.0 {
        Some(CurveCurveHit { u1, u2, point: pa })
    } else {
        None
    }
}

/// Merge duplicate intersection hits that are closer than `tolerance`.
fn merge_duplicate_hits(hits: &mut Vec<CurveCurveHit>, tolerance: f64) {
    if hits.len() <= 1 {
        return;
    }

    // Sort by u1, then merge nearby hits.
    hits.sort_by(|a, b| a.u1.partial_cmp(&b.u1).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged = Vec::with_capacity(hits.len());
    merged.push(hits[0]);

    for hit in hits.iter().skip(1) {
        if let Some(last) = merged.last() {
            if (hit.u1 - last.u1).abs() < tolerance && (hit.u2 - last.u2).abs() < tolerance {
                // Duplicate — skip it.
                continue;
            }
        }
        merged.push(*hit);
    }

    *hits = merged;
}

/// Merge adjacent or overlapping overlap intervals.
fn merge_overlaps(overlaps: &mut Vec<CurveCurveOverlap>, tolerance: f64) {
    if overlaps.len() <= 1 {
        return;
    }
    overlaps.sort_by(|a, b| {
        a.u1_start
            .partial_cmp(&b.u1_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut merged = Vec::with_capacity(overlaps.len());
    merged.push(overlaps[0]);

    for ov in overlaps.iter().skip(1) {
        if let Some(last) = merged.last_mut() {
            if ov.u1_start <= last.u1_end + tolerance {
                // Extend the existing interval.
                last.u1_end = last.u1_end.max(ov.u1_end);
                last.u2_start = last.u2_start.min(ov.u2_start);
                last.u2_end = last.u2_end.max(ov.u2_end);
            } else {
                merged.push(*ov);
            }
        }
    }

    *overlaps = merged;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Create a degree-1 NURBS line from p0 to p1.
    fn make_line(p0: Point3, p1: Point3) -> NurbsCurve {
        NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![p0, p1], vec![1.0, 1.0])
            .expect("valid line")
    }

    #[test]
    fn two_lines_one_intersection() {
        // Line 1: (0,0,0) to (2,2,0) — the diagonal
        // Line 2: (0,2,0) to (2,0,0) — the anti-diagonal
        // They cross at (1,1,0).
        let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 0.0));
        let c2 = make_line(Point3::new(0.0, 2.0, 0.0), Point3::new(2.0, 0.0, 0.0));

        let hits = curve_curve_intersect(&c1, &c2, 1e-10).expect("no error");
        assert_eq!(hits.len(), 1, "expected 1 hit, got {}", hits.len());

        let hit = &hits[0];
        assert!((hit.point.x() - 1.0).abs() < 1e-6, "x: {}", hit.point.x());
        assert!((hit.point.y() - 1.0).abs() < 1e-6, "y: {}", hit.point.y());
        assert!((hit.u1 - 0.5).abs() < 1e-6, "u1: {}", hit.u1);
        assert!((hit.u2 - 0.5).abs() < 1e-6, "u2: {}", hit.u2);
    }

    #[test]
    fn two_quarter_circles_intersection() {
        let w = std::f64::consts::FRAC_1_SQRT_2;

        // Arc 1: quarter circle centered at origin, from (1,0) to (0,1).
        let c1 = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, w, 1.0],
        )
        .expect("valid arc1");

        // Arc 2: quarter circle centered at (1, 1), from (1,0) to (0,1).
        // This arc has radius 1 centered at (1,1), sweeping from 270 deg to 180 deg.
        // Control points for rational quadratic: start=(1,0), mid=(0,0) with weight w, end=(0,1).
        let c2 = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, w, 1.0],
        )
        .expect("valid arc2");

        let hits = curve_curve_intersect(&c1, &c2, 1e-8).expect("no error");
        assert!(
            !hits.is_empty(),
            "expected at least one intersection between overlapping arcs"
        );

        // Verify each hit lies on both curves.
        for hit in &hits {
            let p1 = c1.evaluate(hit.u1);
            let p2 = c2.evaluate(hit.u2);
            let dist = (p1 - p2).length();
            assert!(
                dist < 1e-4,
                "hit not on both curves: dist={dist}, u1={}, u2={}",
                hit.u1,
                hit.u2
            );
        }
    }

    #[test]
    fn disjoint_curves_no_hits() {
        // Two lines far apart.
        let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        let c2 = make_line(Point3::new(0.0, 10.0, 0.0), Point3::new(1.0, 10.0, 0.0));

        let hits = curve_curve_intersect(&c1, &c2, 1e-10).expect("no error");
        assert!(hits.is_empty(), "expected no hits for disjoint curves");
    }

    #[test]
    fn parallel_lines_no_hits() {
        // Two parallel lines close but not touching.
        let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        let c2 = make_line(Point3::new(0.0, 0.1, 0.0), Point3::new(1.0, 0.1, 0.0));

        let hits = curve_curve_intersect(&c1, &c2, 1e-10).expect("no error");
        assert!(hits.is_empty(), "expected no hits for parallel lines");
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_known_intersection(u_param in 0.1f64..0.9) {
            // Build curve1 as a line from (0,0,0) to (2,0,0).
            let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
            let target = c1.evaluate(u_param);

            // Build curve2 as a line passing through that point vertically.
            let c2 = make_line(
                Point3::new(target.x(), -1.0, 0.0),
                Point3::new(target.x(), 1.0, 0.0),
            );

            let hits = curve_curve_intersect(&c1, &c2, 1e-8).expect("no error");
            prop_assert!(!hits.is_empty(), "expected hit near u={u_param}");

            // At least one hit should be near the target point.
            let near = hits.iter().any(|h| (h.point - target).length() < 1e-4);
            prop_assert!(near, "no hit near target {:?}, hits: {:?}", target, hits);
        }
    }

    #[test]
    fn overlapping_lines_detected() {
        // Two collinear lines that share the interval [0.5, 1.5] on x.
        // Line 1: (0,0,0) → (2,0,0)
        // Line 2: (1,0,0) → (3,0,0)
        let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0));
        let c2 = make_line(Point3::new(1.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));

        let result = curve_curve_intersect_full(&c1, &c2, 1e-8).expect("no error");
        assert!(
            !result.overlaps.is_empty(),
            "expected overlap, got {} hits and {} overlaps",
            result.hits.len(),
            result.overlaps.len()
        );

        // The overlap on c1 should span roughly [0.5, 1.0] (u-space).
        let ov = &result.overlaps[0];
        assert!(ov.u1_start < 0.55, "u1_start too high: {}", ov.u1_start);
        assert!(ov.u1_end > 0.95, "u1_end too low: {}", ov.u1_end);
    }

    #[test]
    fn identical_curves_full_overlap() {
        let c1 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0));
        let c2 = make_line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0));

        let result = curve_curve_intersect_full(&c1, &c2, 1e-8).expect("no error");
        assert!(
            !result.overlaps.is_empty(),
            "identical curves should produce overlap, got {} hits",
            result.hits.len()
        );
    }
}
