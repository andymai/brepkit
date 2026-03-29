//! SSI seed finding: subdivision-based and grid-based approaches, plus the
//! main `intersect_nurbs_nurbs` entry point.

use std::collections::VecDeque;

use crate::MathError;
use crate::aabb::Aabb3;
use crate::bvh::Bvh;
use crate::nurbs::decompose::{BezierPatch, surface_to_bezier_patches};
use crate::nurbs::fitting::interpolate;
use crate::nurbs::projection::project_point_to_surface;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::chaining::build_curves_from_points;
use super::surface_marching::{
    constrain_param, constrain_state, march_with_branches, near_existing_segment,
    surface_newton_step,
};
use super::{IntersectionCurve, IntersectionPoint, MAX_NEWTON_ITER, MAX_QUEUE_SIZE, MAX_SEGMENTS};

/// Intersect two NURBS surfaces.
///
/// Uses a subdivision + marching approach:
/// 1. Sample both surfaces on grids to find seed points where they are close
/// 2. Refine seeds with Newton iteration in (u1,v1,u2,v2) space
/// 3. March along the intersection curve from each seed
/// 4. Fit NURBS curves through the traced points
///
/// # Parameters
///
/// - `surface1`, `surface2`: The two NURBS surfaces
/// - `samples`: Grid resolution for seed finding (e.g., 20)
/// - `march_step`: Step size for marching. If `0.0`, an adaptive initial
///   step is computed from the parameter domain extents and control polygon
///   diagonals.
///
/// # Errors
///
/// Returns an error if NURBS evaluation or curve fitting fails.
#[allow(clippy::too_many_lines)]
pub fn intersect_nurbs_nurbs(
    surface1: &NurbsSurface,
    surface2: &NurbsSurface,
    samples: usize,
    march_step: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n = samples.max(5);
    let tolerance = 1e-6;

    // Compute adaptive initial step if not explicitly provided.
    let march_step = if march_step <= 0.0 || march_step > 1.0 {
        // Use parameter domain extent -- step is a fraction of the smallest
        // domain dimension. This ensures roughly consistent point density
        // regardless of surface parameterization scale.
        let (u1_min, u1_max) = surface1.domain_u();
        let (v1_min, v1_max) = surface1.domain_v();
        let (u2_min, u2_max) = surface2.domain_u();
        let (v2_min, v2_max) = surface2.domain_v();
        let extent1 = (u1_max - u1_min).min(v1_max - v1_min);
        let extent2 = (u2_max - u2_min).min(v2_max - v2_min);
        let min_extent = extent1.min(extent2);
        // Start with 1/50th of the smallest domain dimension -- the RKF45
        // and curvature adaptation will refine from there.
        (min_extent / 50.0).max(1e-4)
    } else {
        march_step
    };

    // Phase 1: Find seed points using Bezier subdivision (robust, can't miss branches).
    // Falls back to grid sampling if decomposition fails.
    let seeds = {
        let sub_seeds = find_ssi_seeds_subdivision(surface1, surface2, tolerance);
        if sub_seeds.is_empty() {
            find_ssi_seeds_grid(surface1, surface2, n, tolerance)
        } else {
            sub_seeds
        }
    };

    if seeds.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 2: March from each seed along the intersection curve.
    // Uses a work-queue to discover branch points during marching and
    // spawn additional traces from those branches. Segment-distance
    // dedup avoids redundant marching on already-traced curves.
    let dedup_dist = march_step * 0.5;
    let mut traced_segments: Vec<Vec<IntersectionPoint>> = Vec::new();
    let mut work_queue: VecDeque<IntersectionPoint> = seeds.into();

    while let Some(seed) = work_queue.pop_front() {
        if traced_segments.len() >= MAX_SEGMENTS {
            break;
        }

        // Segment-distance dedup: check if this seed is near any already-
        // traced polyline *segment*, not just individual points. This
        // prevents false positives when a seed is near the middle of a
        // traced curve but could reach a different branch.
        if near_existing_segment(&traced_segments, &seed, dedup_dist) {
            continue;
        }

        let (traced, branch_seeds) =
            march_with_branches(surface1, surface2, &seed, march_step, tolerance);

        if !traced.is_empty() {
            traced_segments.push(traced);
        }

        // Add branch seeds to the work queue (capped).
        for bs in branch_seeds {
            if work_queue.len() < MAX_QUEUE_SIZE {
                work_queue.push_back(bs);
            }
        }
    }

    let all_points: Vec<IntersectionPoint> = traced_segments.into_iter().flatten().collect();

    if all_points.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Build curves from collected points.
    let curves = build_curves_from_points(&all_points)?;

    // Phase 4: Validate fitted curves against both surfaces.
    // Reject or refit curves whose NURBS approximation deviates too far
    // from the actual intersection.
    // 10x relaxation: fitted NURBS curve may deviate slightly from sample points
    let validated = validate_intersection_curves(&curves, surface1, surface2, tolerance * 10.0);

    Ok(validated)
}

/// Validate intersection curves by checking that the fitted NURBS curve
/// matches the stored sample points within tolerance, and that the curve
/// actually lies on both input surfaces.
///
/// Two-phase validation:
/// 1. **Sample-point check**: verify the curve passes near its own sample
///    points. Since those sample points are known to lie on the
///    intersection, large deviations indicate a bad LSPIA fit.
/// 2. **Dual-surface check**: project evenly-spaced curve points onto
///    both surfaces and verify they are within tolerance.  Curves that
///    pass phase 1 but fail phase 2 are re-fitted via interpolation
///    (exact through sample points).
///
/// Curves that deviate too far are re-fitted via interpolation (exact
/// through sample points). If refitting fails, the curve is kept as-is.
#[allow(clippy::cast_precision_loss)]
fn validate_intersection_curves(
    curves: &[IntersectionCurve],
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    tolerance: f64,
) -> Vec<IntersectionCurve> {
    let mut result = Vec::with_capacity(curves.len());

    for ic in curves {
        if ic.points.len() < 2 {
            result.push(ic.clone());
            continue;
        }

        // Phase 1: Check deviation at a subset of stored sample points.
        // Evaluate the curve at evenly spaced parameters and compare
        // against the nearest sample point.
        let (t_min, t_max) = ic.curve.domain();
        let n_check = 5.min(ic.points.len());
        let step = ic.points.len() / n_check;

        let mut max_dev = 0.0_f64;
        for i in 0..n_check {
            let idx = (i * step).min(ic.points.len() - 1);
            let sample_pt = ic.points[idx].point;

            // Find the parameter closest to this sample point by
            // evaluating the curve at a proportional parameter.
            let t = t_min + (t_max - t_min) * idx as f64 / (ic.points.len() - 1).max(1) as f64;
            let curve_pt = ic.curve.evaluate(t.clamp(t_min, t_max));
            let dev = (curve_pt - sample_pt).length();
            max_dev = max_dev.max(dev);
        }

        // Use the refitted curve if Phase 1 failed, otherwise the original.
        let working_curve = if max_dev > tolerance {
            refit_from_samples(ic)
        } else {
            ic.clone()
        };

        // Phase 2: Dual-surface validation -- verify the curve lies on
        // both input surfaces by projecting evenly-spaced curve points.
        let (wt_min, wt_max) = working_curve.curve.domain();
        let n_surface_check = 5;
        let mut max_surface_dev = 0.0_f64;
        let mut successful_projections = 0_u32;
        for i in 0..n_surface_check {
            let t = wt_min + (wt_max - wt_min) * i as f64 / (n_surface_check - 1).max(1) as f64;
            let curve_pt = working_curve.curve.evaluate(t.clamp(wt_min, wt_max));

            // Project onto surface 1 and measure deviation.
            if let Ok(proj) = project_point_to_surface(s1, curve_pt, tolerance) {
                max_surface_dev = max_surface_dev.max(proj.distance);
                successful_projections += 1;
            }
            // Project onto surface 2 and measure deviation.
            if let Ok(proj) = project_point_to_surface(s2, curve_pt, tolerance) {
                max_surface_dev = max_surface_dev.max(proj.distance);
                successful_projections += 1;
            }
        }

        if successful_projections == 0 {
            // No projections succeeded -- can't validate, keep curve as-is.
            result.push(working_curve);
            continue;
        }

        if max_surface_dev > tolerance {
            log::warn!(
                "SSI: curve deviates {max_surface_dev:.2e} from surface(s) \
                 (tolerance={tolerance:.2e}), re-fitting from sample points"
            );
            let refit_curve = refit_from_samples(ic);
            result.push(refit_curve);
        } else {
            result.push(working_curve);
        }
    }

    result
}

/// Re-fit an intersection curve from its stored sample points via
/// interpolation.  Falls back to the original curve if interpolation
/// fails.
fn refit_from_samples(ic: &IntersectionCurve) -> IntersectionCurve {
    let positions: Vec<Point3> = ic.points.iter().map(|p| p.point).collect();
    let degree = if positions.len() <= 3 {
        1
    } else {
        3.min(positions.len() - 1)
    };
    if let Ok(refit) = interpolate(&positions, degree) {
        IntersectionCurve {
            curve: refit,
            points: ic.points.clone(),
        }
    } else {
        ic.clone()
    }
}

/// Find SSI seed points using recursive Bezier patch subdivision + BVH overlap.
///
/// This approach **cannot miss intersection branches** because it converges
/// on all regions where the two surfaces are close. Steps:
/// 1. Decompose both surfaces into Bezier patches
/// 2. Build a BVH over B's patch AABBs
/// 3. For each A-patch, find overlapping B-patches
/// 4. Small overlapping pairs -> seed from centroid + `refine_ssi_point`
/// 5. Large pairs -> subdivide and recurse (max depth limit)
#[allow(clippy::cast_precision_loss)]
pub(super) fn find_ssi_seeds_subdivision(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let patches_a = match surface_to_bezier_patches(s1) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let patches_b = match surface_to_bezier_patches(s2) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };

    // Build BVH over B's patches.
    let b_entries: Vec<(usize, Aabb3)> = patches_b
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.aabb()))
        .collect();
    let bvh = Bvh::build(&b_entries);

    // Collect candidate pairs by AABB overlap.
    let mut candidate_pairs: Vec<(BezierPatch, BezierPatch)> = Vec::new();
    for pa in &patches_a {
        let aabb_a = pa.aabb();
        let candidates = bvh.query_overlap(&aabb_a);
        for &b_idx in &candidates {
            candidate_pairs.push((pa.clone(), patches_b[b_idx].clone()));
        }
    }

    // Recursively subdivide overlapping pairs to find seeds.
    // 100x: patch diagonal threshold for Newton seeding -- patches this small
    // are close enough to attempt direct refinement
    let diag_threshold = tolerance * 100.0;
    let max_depth = 6;
    let mut seeds: Vec<IntersectionPoint> = Vec::new();

    subdivide_for_seeds(
        s1,
        s2,
        &candidate_pairs,
        diag_threshold,
        max_depth,
        0,
        tolerance,
        &mut seeds,
    );

    seeds
}

/// Recursive helper: subdivide overlapping Bezier patch pairs to find SSI seeds.
#[allow(clippy::too_many_arguments)]
fn subdivide_for_seeds(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    pairs: &[(BezierPatch, BezierPatch)],
    diag_threshold: f64,
    max_depth: usize,
    depth: usize,
    tolerance: f64,
    seeds: &mut Vec<IntersectionPoint>,
) {
    // Cap seed count: marching only needs a few seeds per intersection branch.
    // Near-tangential cases can generate thousands of subdivision candidates,
    // most of which converge to the same curve. 50 seeds is plenty.
    const MAX_SEEDS: usize = 50;

    for (pa, pb) in pairs {
        if seeds.len() >= MAX_SEEDS {
            return;
        }

        let diag_a = pa.diagonal();
        let diag_b = pb.diagonal();

        if diag_a < diag_threshold && diag_b < diag_threshold {
            // Both patches are small: try to find a seed from centroid parameters.
            // Skip Newton if patch centroids are far apart -- the AABB overlap
            // doesn't guarantee the surfaces actually intersect at this location.
            let p1 = s1.evaluate(pa.u_mid(), pa.v_mid());
            let p2 = s2.evaluate(pb.u_mid(), pb.v_mid());
            let centroid_dist = (p1 - p2).length();
            let max_diag = diag_a.max(diag_b);
            if centroid_dist > max_diag * 2.0 {
                continue;
            }

            let u1 = pa.u_mid();
            let v1 = pa.v_mid();
            let u2 = pb.u_mid();
            let v2 = pb.v_mid();

            if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
                // 100x dedup: multiple patches may converge to the same intersection
                let is_dup = seeds
                    .iter()
                    .any(|s| (s.point - refined.point).length() < tolerance * 100.0);
                if !is_dup {
                    seeds.push(refined);
                }
            }
            continue;
        }

        if depth >= max_depth {
            // Max depth: try seed anyway.
            let p1 = s1.evaluate(pa.u_mid(), pa.v_mid());
            let p2 = s2.evaluate(pb.u_mid(), pb.v_mid());
            let centroid_dist = (p1 - p2).length();
            let max_diag = diag_a.max(diag_b);
            if centroid_dist > max_diag * 2.0 {
                continue;
            }

            let u1 = pa.u_mid();
            let v1 = pa.v_mid();
            let u2 = pb.u_mid();
            let v2 = pb.v_mid();

            if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
                // 100x dedup: multiple patches may converge to the same intersection
                let is_dup = seeds
                    .iter()
                    .any(|s| (s.point - refined.point).length() < tolerance * 100.0);
                if !is_dup {
                    seeds.push(refined);
                }
            }
            continue;
        }

        // Subdivide the larger patch and check overlaps with the smaller one.
        if diag_a >= diag_b {
            // Subdivide patch A at its u or v midpoint (whichever span is larger).
            let sub_pairs = subdivide_patch_a_check_overlap(pa, pb);
            subdivide_for_seeds(
                s1,
                s2,
                &sub_pairs,
                diag_threshold,
                max_depth,
                depth + 1,
                tolerance,
                seeds,
            );
        } else {
            // Subdivide patch B.
            let sub_pairs = subdivide_patch_b_check_overlap(pa, pb);
            subdivide_for_seeds(
                s1,
                s2,
                &sub_pairs,
                diag_threshold,
                max_depth,
                depth + 1,
                tolerance,
                seeds,
            );
        }
    }
}

/// Subdivide patch A at its midpoint and return sub-pairs that overlap with B.
fn subdivide_patch_a_check_overlap(
    pa: &BezierPatch,
    pb: &BezierPatch,
) -> Vec<(BezierPatch, BezierPatch)> {
    let subs = subdivide_bezier_patch(pa);
    let aabb_b = pb.aabb();
    subs.into_iter()
        .filter(|sub| sub.aabb().intersects(aabb_b))
        .map(|sub| (sub, pb.clone()))
        .collect()
}

/// Subdivide patch B at its midpoint and return sub-pairs that overlap with A.
fn subdivide_patch_b_check_overlap(
    pa: &BezierPatch,
    pb: &BezierPatch,
) -> Vec<(BezierPatch, BezierPatch)> {
    let subs = subdivide_bezier_patch(pb);
    let aabb_a = pa.aabb();
    subs.into_iter()
        .filter(|sub| sub.aabb().intersects(aabb_a))
        .map(|sub| (pa.clone(), sub))
        .collect()
}

/// Subdivide a Bezier patch into 2 patches by splitting at the midpoint
/// of the larger parameter span (u or v).
fn subdivide_bezier_patch(patch: &BezierPatch) -> Vec<BezierPatch> {
    let u_span = patch.u_range.1 - patch.u_range.0;
    let v_span = patch.v_range.1 - patch.v_range.0;

    let (split_u, split_param) = if u_span >= v_span {
        (true, patch.u_mid())
    } else {
        (false, patch.v_mid())
    };

    // Insert the midpoint knot to full multiplicity and split.
    let surf = &patch.surface;
    let degree = if split_u {
        surf.degree_u()
    } else {
        surf.degree_v()
    };

    let refined = if split_u {
        crate::nurbs::knot_ops::surface_knot_insert_u(surf, split_param, degree)
    } else {
        crate::nurbs::knot_ops::surface_knot_insert_v(surf, split_param, degree)
    };

    let Ok(refined) = refined else {
        return vec![patch.clone()]; // Fallback: don't subdivide
    };

    // Extract two sub-patches from the refined surface.
    if split_u {
        extract_u_subpatches(&refined, patch, split_param)
    } else {
        extract_v_subpatches(&refined, patch, split_param)
    }
}

/// Extract two sub-patches from a surface that has been refined in u.
fn extract_u_subpatches(
    refined: &NurbsSurface,
    parent: &BezierPatch,
    u_split: f64,
) -> Vec<BezierPatch> {
    let pu = refined.degree_u();
    let pv = refined.degree_v();
    let cps = refined.control_points();
    let ws = refined.weights();
    let n_rows = cps.len();
    if n_rows == 0 || cps[0].is_empty() {
        return vec![parent.clone()];
    }

    // Find the split index: the row where the u-knot multiplicity reaches pu.
    let knots_u = refined.knots_u();
    let split_row = find_split_row(knots_u, u_split, pu, n_rows);

    if split_row == 0 || split_row >= n_rows {
        return vec![parent.clone()];
    }

    let mut result = Vec::with_capacity(2);

    // Left patch: rows 0..=split_row
    if split_row >= pu {
        let left_rows = split_row + 1;
        let left_cps: Vec<Vec<Point3>> = cps[..left_rows].to_vec();
        let left_ws: Vec<Vec<f64>> = ws[..left_rows].to_vec();
        let mut left_ku = vec![parent.u_range.0; pu + 1];
        left_ku.extend(std::iter::repeat_n(u_split, pu + 1));
        let left_kv = refined.knots_v().to_vec();

        if let Ok(s) = NurbsSurface::new(pu, pv, left_ku, left_kv, left_cps, left_ws) {
            result.push(BezierPatch {
                surface: s,
                u_range: (parent.u_range.0, u_split),
                v_range: parent.v_range,
            });
        }
    }

    // Right patch: rows split_row..n_rows
    let right_rows = n_rows - split_row;
    if right_rows > pu {
        let right_cps: Vec<Vec<Point3>> = cps[split_row..].to_vec();
        let right_ws: Vec<Vec<f64>> = ws[split_row..].to_vec();
        let mut right_ku = vec![u_split; pu + 1];
        right_ku.extend(std::iter::repeat_n(parent.u_range.1, pu + 1));
        let right_kv = refined.knots_v().to_vec();

        if let Ok(s) = NurbsSurface::new(pu, pv, right_ku, right_kv, right_cps, right_ws) {
            result.push(BezierPatch {
                surface: s,
                u_range: (u_split, parent.u_range.1),
                v_range: parent.v_range,
            });
        }
    }

    if result.is_empty() {
        vec![parent.clone()]
    } else {
        result
    }
}

/// Extract two sub-patches from a surface that has been refined in v.
fn extract_v_subpatches(
    refined: &NurbsSurface,
    parent: &BezierPatch,
    v_split: f64,
) -> Vec<BezierPatch> {
    let pu = refined.degree_u();
    let pv = refined.degree_v();
    let cps = refined.control_points();
    let ws = refined.weights();
    let n_rows = cps.len();
    let n_cols = if n_rows > 0 {
        cps[0].len()
    } else {
        return vec![parent.clone()];
    };

    // Find the split column index.
    let knots_v = refined.knots_v();
    let split_col = find_split_row(knots_v, v_split, pv, n_cols);

    if split_col == 0 || split_col >= n_cols {
        return vec![parent.clone()];
    }

    let mut result = Vec::with_capacity(2);

    // Bottom patch: cols 0..=split_col
    let left_cols = split_col + 1;
    if left_cols > pv {
        let left_cps: Vec<Vec<Point3>> = cps.iter().map(|row| row[..left_cols].to_vec()).collect();
        let left_ws: Vec<Vec<f64>> = ws.iter().map(|row| row[..left_cols].to_vec()).collect();
        let left_ku = refined.knots_u().to_vec();
        let mut left_kv = vec![parent.v_range.0; pv + 1];
        left_kv.extend(std::iter::repeat_n(v_split, pv + 1));

        if let Ok(s) = NurbsSurface::new(pu, pv, left_ku, left_kv, left_cps, left_ws) {
            result.push(BezierPatch {
                surface: s,
                u_range: parent.u_range,
                v_range: (parent.v_range.0, v_split),
            });
        }
    }

    // Top patch: cols split_col..n_cols
    let right_cols = n_cols - split_col;
    if right_cols > pv {
        let right_cps: Vec<Vec<Point3>> = cps.iter().map(|row| row[split_col..].to_vec()).collect();
        let right_ws: Vec<Vec<f64>> = ws.iter().map(|row| row[split_col..].to_vec()).collect();
        let right_ku = refined.knots_u().to_vec();
        let mut right_kv = vec![v_split; pv + 1];
        right_kv.extend(std::iter::repeat_n(parent.v_range.1, pv + 1));

        if let Ok(s) = NurbsSurface::new(pu, pv, right_ku, right_kv, right_cps, right_ws) {
            result.push(BezierPatch {
                surface: s,
                u_range: parent.u_range,
                v_range: (v_split, parent.v_range.1),
            });
        }
    }

    if result.is_empty() {
        vec![parent.clone()]
    } else {
        result
    }
}

/// Find the row/column index where a knot reaches full multiplicity.
fn find_split_row(knots: &[f64], split_val: f64, degree: usize, n_cps: usize) -> usize {
    // After inserting to multiplicity `degree`, the split point is where
    // knots[i] == split_val for `degree` consecutive entries.
    // The corresponding CP index is the last of those minus degree.
    let mut count = 0_usize;
    let mut last_idx = 0_usize;
    for (i, &k) in knots.iter().enumerate() {
        if (k - split_val).abs() < 1e-15 {
            count += 1;
            last_idx = i;
        }
    }

    if count >= degree {
        let split_cp = last_idx.saturating_sub(degree);
        split_cp.min(n_cps.saturating_sub(1))
    } else {
        0
    }
}

/// Find seed points for NURBS-NURBS intersection by grid sampling (fallback).
///
/// Strategy: sample both surfaces on an n x n grid and try Newton
/// refinement for all cell-center pairs whose 3D positions are within
/// a generous distance threshold.
#[allow(clippy::cast_precision_loss)]
pub(super) fn find_ssi_seeds_grid(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    n: usize,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let (u1_min, u1_max) = s1.domain_u();
    let (v1_min, v1_max) = s1.domain_v();
    let (u2_min, u2_max) = s2.domain_u();
    let (v2_min, v2_max) = s2.domain_v();

    #[allow(clippy::cast_precision_loss)]
    let u1_step = (u1_max - u1_min) / (n - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let v1_step = (v1_max - v1_min) / (n - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let u2_step = (u2_max - u2_min) / (n - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let v2_step = (v2_max - v2_min) / (n - 1) as f64;

    let mut seeds = Vec::new();

    // Sample both surfaces.
    let mut pts1: Vec<(f64, f64, Point3)> = Vec::with_capacity(n * n);
    let mut pts2: Vec<(f64, f64, Point3)> = Vec::with_capacity(n * n);

    for i in 0..n {
        let u1 = u1_min + i as f64 * u1_step;
        for j in 0..n {
            let v1 = v1_min + j as f64 * v1_step;
            pts1.push((u1, v1, s1.evaluate(u1, v1)));
        }
    }

    for i in 0..n {
        let u2 = u2_min + i as f64 * u2_step;
        for j in 0..n {
            let v2 = v2_min + j as f64 * v2_step;
            pts2.push((u2, v2, s2.evaluate(u2, v2)));
        }
    }

    // For each pair of sample points, if close enough, try Newton.
    // Use a generous threshold based on the diagonal of a grid cell (in 3D space).
    let diag1 = (s1.evaluate(u1_min, v1_min) - s1.evaluate(u1_max, v1_max)).length();
    let diag2 = (s2.evaluate(u2_min, v2_min) - s2.evaluate(u2_max, v2_max)).length();
    #[allow(clippy::cast_precision_loss)]
    let threshold = ((diag1.max(diag2) / n as f64) * 3.0).max(0.1);

    for &(u1, v1, p1) in &pts1 {
        for &(u2, v2, p2) in &pts2 {
            let dist = (p1 - p2).length();
            if dist < threshold {
                if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
                    // 100x dedup: multiple grid samples may converge to the same intersection
                    let dup = seeds.iter().any(|s: &IntersectionPoint| {
                        (s.point - refined.point).length() < tolerance * 100.0
                    });
                    if !dup {
                        seeds.push(refined);
                    }
                }
            }
        }
    }

    seeds
}

#[allow(clippy::similar_names)]
/// Refine an SSI point using coupled 4D Newton iteration.
///
/// Solves `S1(u1,v1) - S2(u2,v2) = 0` directly as a 3x4 system via
/// normal equations `JtJ*d = Jtr`, giving **quadratic convergence**.
/// This replaces the previous alternating-projection approach which had
/// only linear convergence and could fail near tangent intersections.
pub(super) fn refine_ssi_point(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1_guess: f64,
    v1_guess: f64,
    u2_guess: f64,
    v2_guess: f64,
    tolerance: f64,
) -> Option<IntersectionPoint> {
    let mut state = [u1_guess, v1_guess, u2_guess, v2_guess];
    let mut prev_residual = f64::MAX;

    for iteration in 0..MAX_NEWTON_ITER {
        let cstate = constrain_state(&state, s1, s2);
        let p1 = s1.evaluate(cstate[0], cstate[1]);
        let p2 = s2.evaluate(cstate[2], cstate[3]);
        let r = p1 - p2;
        let residual = r.length();

        if residual < tolerance {
            return Some(IntersectionPoint {
                point: p1,
                param1: (cstate[0], cstate[1]),
                param2: (cstate[2], cstate[3]),
            });
        }

        // Early bail-out: if residual isn't decreasing after initial iterations,
        // the surfaces likely don't intersect near this guess. Saves ~25 wasted
        // iterations per non-intersecting patch pair in near-tangential cases.
        if iteration >= 5 && residual > prev_residual * 0.5 {
            return None;
        }
        prev_residual = residual;

        // Build 3x4 Jacobian: J = [dS1/du1, dS1/dv1, -dS2/du2, -dS2/dv2]
        let d1 = s1.derivatives(cstate[0], cstate[1], 1);
        let d2 = s2.derivatives(cstate[2], cstate[3], 1);
        let j = [d1[1][0], d1[0][1], -d2[1][0], -d2[0][1]]; // 4 column vectors (Vec3)

        // Normal equations: JtJ (4x4) * d = -Jtr (4x1)
        // Use only the well-conditioned 4x4 system.
        let r_vec = Vec3::new(r.x(), r.y(), r.z());
        let jtr: [f64; 4] = std::array::from_fn(|i| -j[i].dot(r_vec));
        let jtj: [[f64; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|k| j[i].dot(j[k])));

        // Solve 4x4 system via Gaussian elimination with partial pivoting.
        // If singular (near a surface pole/apex), try Tikhonov regularization
        // before falling back to alternating projection.
        let delta = solve_4x4(jtj, jtr).or_else(|| {
            // Regularize: add lI to JtJ (Tikhonov / Levenberg-Marquardt).
            let trace = jtj[0][0] + jtj[1][1] + jtj[2][2] + jtj[3][3];
            let lambda = trace.max(1e-10) * 1e-4;
            let mut jtj_r = jtj;
            for i in 0..4 {
                jtj_r[i][i] += lambda;
            }
            solve_4x4(jtj_r, jtr)
        });

        if let Some(delta) = delta {
            state[0] += delta[0];
            state[1] += delta[1];
            state[2] += delta[2];
            state[3] += delta[3];
        } else {
            // Still singular after regularization -- fall back to alternating projection.
            let (du2, dv2) = surface_newton_step(s2, cstate[2], cstate[3], p1);
            state[2] += du2;
            state[3] += dv2;
            let p2_new = s2.evaluate(
                constrain_param(
                    state[2],
                    s2.domain_u().0,
                    s2.domain_u().1,
                    s2.is_periodic_u(),
                ),
                constrain_param(
                    state[3],
                    s2.domain_v().0,
                    s2.domain_v().1,
                    s2.is_periodic_v(),
                ),
            );
            let (du1, dv1) = surface_newton_step(s1, cstate[0], cstate[1], p2_new);
            state[0] += du1;
            state[1] += dv1;
        }
    }

    // Final check with relaxed tolerance.
    // 10x relaxation: seed points may be imprecise
    let cstate = constrain_state(&state, s1, s2);
    let p1 = s1.evaluate(cstate[0], cstate[1]);
    let p2 = s2.evaluate(cstate[2], cstate[3]);
    if (p1 - p2).length() < tolerance * 10.0 {
        Some(IntersectionPoint {
            point: p1,
            param1: (cstate[0], cstate[1]),
            param2: (cstate[2], cstate[3]),
        })
    } else {
        None
    }
}

/// Solve a 4x4 linear system via Gaussian elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular.
pub(super) fn solve_4x4(a: [[f64; 4]; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let mut m = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j];
        }
        m[i][4] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..4 {
        // Find pivot.
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..4 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != col {
            m.swap(col, max_row);
        }
        let pivot = m[col][col];
        for row in (col + 1)..4 {
            let factor = m[row][col] / pivot;
            for j in col..5 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution.
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        let mut sum = m[i][4];
        for j in (i + 1)..4 {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }

    Some(x)
}
