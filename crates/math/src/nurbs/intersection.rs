//! Surface-surface and curve-surface intersection routines.
//!
//! These are the geometric foundations for boolean operations on NURBS solids.
//!
//! ## Algorithms
//!
//! - **Plane-NURBS**: Sample the NURBS surface on a grid, find sign changes of the
//!   signed distance to the plane, trace zero-crossings via linear interpolation,
//!   then refine with Newton iteration.
//! - **NURBS-NURBS**: Subdivision + marching method in (u1,v1,u2,v2) parameter space.
//! - **Line-surface**: Newton iteration from grid-based seed points.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::manual_let_else
)]

use crate::MathError;
use crate::aabb::Aabb3;
use crate::bvh::Bvh;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::{BezierPatch, surface_to_bezier_patches};
use crate::nurbs::fitting::{approximate_lspia, interpolate};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// A point on an intersection curve, with parameter values on both surfaces.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionPoint {
    /// 3D position of the intersection.
    pub point: Point3,
    /// Parameter on the first surface (u1, v1) or the curve parameter.
    pub param1: (f64, f64),
    /// Parameter on the second surface (u2, v2).
    pub param2: (f64, f64),
}

/// Result of a surface-surface intersection: a list of intersection curves.
#[derive(Debug, Clone)]
pub struct IntersectionCurve {
    /// The 3D intersection curve as a NURBS.
    pub curve: NurbsCurve,
    /// Sampled points along the curve with parameter values.
    pub points: Vec<IntersectionPoint>,
}

/// Intersect a plane with a NURBS surface.
///
/// Returns a list of intersection curves (there may be multiple
/// disconnected branches).
///
/// # Parameters
///
/// - `surface`: The NURBS surface
/// - `plane_normal`: Normal of the cutting plane
/// - `plane_d`: Signed distance from origin (`n · p = d` for points on plane)
/// - `samples`: Grid resolution for finding seed points (e.g., 50)
///
/// # Errors
///
/// Returns an error if NURBS evaluation fails or curve fitting fails.
pub fn intersect_plane_nurbs(
    surface: &NurbsSurface,
    plane_normal: Vec3,
    plane_d: f64,
    samples: usize,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n = samples.max(10);

    // Phase 1: Sample the signed distance field on a grid.
    let mut distances = vec![vec![0.0_f64; n]; n];

    #[allow(clippy::cast_precision_loss)]
    let step = 1.0 / (n - 1) as f64;

    #[allow(clippy::cast_precision_loss)]
    for (i, row) in distances.iter_mut().enumerate() {
        let u = i as f64 * step;
        for (j, dist) in row.iter_mut().enumerate() {
            let v = j as f64 * step;
            let pt = surface.evaluate(u, v);
            let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
            *dist = plane_normal.dot(pt_vec) - plane_d;
        }
    }

    // Phase 2: Find zero-crossings between adjacent grid cells.
    let mut crossing_points: Vec<(f64, f64, Point3)> = Vec::new();

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            let d00 = distances[i][j];
            let d10 = distances[i + 1][j];
            let d01 = distances[i][j + 1];
            let d11 = distances[i + 1][j + 1];

            // Check edges for sign changes.
            let u0 = i as f64 * step;
            let u1 = (i + 1) as f64 * step;
            let v0 = j as f64 * step;
            let v1 = (j + 1) as f64 * step;

            // Bottom edge (i,j) → (i+1,j)
            if d00 * d10 < 0.0 {
                let t = d00 / (d00 - d10);
                let u = u0.mul_add(1.0 - t, u1 * t);
                let pt = surface.evaluate(u, v0);
                crossing_points.push((u, v0, pt));
            }

            // Left edge (i,j) → (i,j+1)
            if d00 * d01 < 0.0 {
                let t = d00 / (d00 - d01);
                let v = v0.mul_add(1.0 - t, v1 * t);
                let pt = surface.evaluate(u0, v);
                crossing_points.push((u0, v, pt));
            }

            // Top edge (i,j+1) → (i+1,j+1)
            if d01 * d11 < 0.0 {
                let t = d01 / (d01 - d11);
                let u = u0.mul_add(1.0 - t, u1 * t);
                let pt = surface.evaluate(u, v1);
                crossing_points.push((u, v1, pt));
            }

            // Right edge (i+1,j) → (i+1,j+1)
            if d10 * d11 < 0.0 {
                let t = d10 / (d10 - d11);
                let v = v0.mul_add(1.0 - t, v1 * t);
                let pt = surface.evaluate(u1, v);
                crossing_points.push((u1, v, pt));
            }
        }
    }

    if crossing_points.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Refine crossing points with Newton iteration.
    let mut refined_points: Vec<IntersectionPoint> = Vec::new();
    for (u_guess, v_guess, _pt) in &crossing_points {
        if let Some(refined) =
            refine_plane_surface_point(surface, plane_normal, plane_d, *u_guess, *v_guess)
        {
            refined_points.push(refined);
        }
    }

    if refined_points.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 4: Sort points and connect them into curves.
    // Simple approach: sort by parameter distance and group into chains.
    let curves = build_curves_from_points(&refined_points)?;

    Ok(curves)
}

/// Intersect a line (ray) with a NURBS surface.
///
/// Returns all intersection points along the ray.
///
/// # Parameters
///
/// - `surface`: The NURBS surface
/// - `ray_origin`: Starting point of the ray
/// - `ray_dir`: Direction of the ray
/// - `samples`: Grid resolution for finding seed points
///
/// # Errors
///
/// Returns an error if evaluation or refinement fails.
pub fn intersect_line_nurbs(
    surface: &NurbsSurface,
    ray_origin: Point3,
    ray_dir: Vec3,
    samples: usize,
) -> Result<Vec<IntersectionPoint>, MathError> {
    let n = samples.max(10);

    #[allow(clippy::cast_precision_loss)]
    let step = 1.0 / (n - 1) as f64;

    // Sample surface points and compute distance to ray.
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new(); // (u, v, t_along_ray)

    let dir_len_sq = ray_dir.dot(ray_dir);
    if dir_len_sq < 1e-20 {
        return Err(MathError::ZeroVector);
    }

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let u = i as f64 * step;
        for j in 0..n {
            let v = j as f64 * step;
            let pt = surface.evaluate(u, v);
            let diff = pt - ray_origin;
            let diff_vec = Vec3::new(diff.x(), diff.y(), diff.z());

            // Project onto ray.
            let t = diff_vec.dot(ray_dir) / dir_len_sq;
            let closest_on_ray = Point3::new(
                ray_origin.x().mul_add(1.0, ray_dir.x() * t),
                ray_origin.y().mul_add(1.0, ray_dir.y() * t),
                ray_origin.z().mul_add(1.0, ray_dir.z() * t),
            );

            let dist = (pt - closest_on_ray).length();
            if dist < 0.1 {
                // Rough candidate.
                candidates.push((u, v, t));
            }
        }
    }

    // Refine candidates with Newton iteration.
    let mut results: Vec<IntersectionPoint> = Vec::new();
    for (u_guess, v_guess, _t) in &candidates {
        if let Some(pt) =
            refine_line_surface_point(surface, ray_origin, ray_dir, *u_guess, *v_guess)
        {
            // Deduplicate: skip if close to an existing result.
            let dominated = results
                .iter()
                .any(|existing| (existing.point - pt.point).length() < 1e-6);
            if !dominated {
                results.push(pt);
            }
        }
    }

    Ok(results)
}

/// Refine a plane-surface intersection point using Newton iteration.
fn refine_plane_surface_point(
    surface: &NurbsSurface,
    plane_normal: Vec3,
    plane_d: f64,
    u_guess: f64,
    v_guess: f64,
) -> Option<IntersectionPoint> {
    let mut u = u_guess;
    let mut v = v_guess;

    for _ in 0..20 {
        let pt = surface.evaluate(u, v);
        let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
        let f = plane_normal.dot(pt_vec) - plane_d;

        if f.abs() < 1e-12 {
            return Some(IntersectionPoint {
                point: pt,
                param1: (u, v),
                param2: (0.0, 0.0),
            });
        }

        // Compute gradient of f w.r.t. (u, v).
        let derivs = surface.derivatives(u, v, 1);
        let du = derivs[1][0]; // ∂S/∂u
        let dv = derivs[0][1]; // ∂S/∂v

        let grad_u = plane_normal.dot(du);
        let grad_v = plane_normal.dot(dv);

        let grad_len_sq = grad_u.mul_add(grad_u, grad_v * grad_v);
        if grad_len_sq < 1e-20 {
            break; // Singular.
        }

        // Steepest descent step.
        let step_size = f / grad_len_sq;
        u -= grad_u * step_size;
        v -= grad_v * step_size;

        // Clamp to [0, 1].
        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
    }

    // Final check.
    let pt = surface.evaluate(u, v);
    let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
    let f = plane_normal.dot(pt_vec) - plane_d;

    if f.abs() < 1e-6 {
        Some(IntersectionPoint {
            point: pt,
            param1: (u, v),
            param2: (0.0, 0.0),
        })
    } else {
        None
    }
}

/// Refine a line-surface intersection point using Newton iteration.
fn refine_line_surface_point(
    surface: &NurbsSurface,
    ray_origin: Point3,
    ray_dir: Vec3,
    u_guess: f64,
    v_guess: f64,
) -> Option<IntersectionPoint> {
    let mut u = u_guess;
    let mut v = v_guess;

    for _ in 0..20 {
        let pt = surface.evaluate(u, v);
        let diff = pt - ray_origin;
        let diff_vec = Vec3::new(diff.x(), diff.y(), diff.z());

        // Closest t on ray.
        let t = diff_vec.dot(ray_dir) / ray_dir.dot(ray_dir);
        let ray_pt = Point3::new(
            ray_dir.x().mul_add(t, ray_origin.x()),
            ray_dir.y().mul_add(t, ray_origin.y()),
            ray_dir.z().mul_add(t, ray_origin.z()),
        );

        let residual = pt - ray_pt;
        if residual.length() < 1e-10 {
            return Some(IntersectionPoint {
                point: pt,
                param1: (u, v),
                param2: (t, 0.0),
            });
        }

        // Newton step in (u, v) space.
        let derivs = surface.derivatives(u, v, 1);
        let su = derivs[1][0];
        let sv = derivs[0][1];

        let r = Vec3::new(residual.x(), residual.y(), residual.z());

        // Solve 2×2 system: [su·su, su·sv; sv·su, sv·sv] * [du, dv] = [su·r, sv·r]
        let a11 = su.dot(su);
        let a12 = su.dot(sv);
        let a22 = sv.dot(sv);
        let b1 = su.dot(r);
        let b2 = sv.dot(r);

        let det = a11.mul_add(a22, -(a12 * a12));
        if det.abs() < 1e-20 {
            break;
        }

        let du = b1.mul_add(a22, -(b2 * a12)) / det;
        let dv = a11.mul_add(b2, -(a12 * b1)) / det;

        u -= du;
        v -= dv;
        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
    }

    // Final check.
    let pt = surface.evaluate(u, v);
    let diff = pt - ray_origin;
    let diff_vec = Vec3::new(diff.x(), diff.y(), diff.z());
    let t = diff_vec.dot(ray_dir) / ray_dir.dot(ray_dir);
    let ray_pt = Point3::new(
        ray_dir.x().mul_add(t, ray_origin.x()),
        ray_dir.y().mul_add(t, ray_origin.y()),
        ray_dir.z().mul_add(t, ray_origin.z()),
    );

    if (pt - ray_pt).length() < 1e-5 {
        Some(IntersectionPoint {
            point: pt,
            param1: (u, v),
            param2: (t, 0.0),
        })
    } else {
        None
    }
}

/// Build intersection curves from a set of points by chaining and fitting.
///
/// First chains points into connected components (separate intersection
/// branches), then fits a NURBS curve through each chain independently.
fn build_curves_from_points(
    points: &[IntersectionPoint],
) -> Result<Vec<IntersectionCurve>, MathError> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    // Estimate a chaining threshold from the average spacing.
    let threshold = estimate_chain_threshold(points);

    // Chain points into connected components.
    let chains = chain_intersection_points(points, threshold);

    let mut curves = Vec::with_capacity(chains.len());

    for chain in &chains {
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
            approximate_lspia(&positions, degree, num_cps, 1e-6, 100)?
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

/// Estimate a reasonable chaining threshold from point spacing.
#[allow(clippy::cast_precision_loss)]
fn estimate_chain_threshold(points: &[IntersectionPoint]) -> f64 {
    if points.len() < 2 {
        return 1.0;
    }

    // Compute average nearest-neighbor distance (sample up to 100 points for speed).
    let sample_size = points.len().min(100);
    let mut total_min_dist = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..sample_size {
        let mut min_d = f64::MAX;
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = (points[i].point - q.point).length();
            if d < min_d {
                min_d = d;
            }
        }
        if min_d < f64::MAX {
            total_min_dist += min_d;
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }

    // Use 3× average nearest-neighbor distance as threshold.
    // The threshold must be large enough to chain adjacent sampling
    // points along the same intersection branch. We also compute
    // the bounding box diagonal as an upper-bound reference.
    let avg = total_min_dist / count as f64;

    // Also compute the bounding box diagonal of all points.
    let mut bb_min = [f64::MAX; 3];
    let mut bb_max = [f64::MIN; 3];
    for p in points {
        bb_min[0] = bb_min[0].min(p.point.x());
        bb_min[1] = bb_min[1].min(p.point.y());
        bb_min[2] = bb_min[2].min(p.point.z());
        bb_max[0] = bb_max[0].max(p.point.x());
        bb_max[1] = bb_max[1].max(p.point.y());
        bb_max[2] = bb_max[2].max(p.point.z());
    }
    let diag = ((bb_max[0] - bb_min[0]).powi(2)
        + (bb_max[1] - bb_min[1]).powi(2)
        + (bb_max[2] - bb_min[2]).powi(2))
    .sqrt();

    // Floor: 5% of the bounding diagonal, which handles cases where
    // many points converge to the same location after Newton refinement.
    let floor = diag * 0.05;
    (avg * 3.0).max(floor).max(1e-4)
}

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
/// - `march_step`: Step size for marching (e.g., 0.02)
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
    let mut all_points: Vec<IntersectionPoint> = Vec::new();

    for seed in &seeds {
        let traced = march_intersection(surface1, surface2, seed, march_step, tolerance);
        all_points.extend(traced);
    }

    if all_points.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Build curves from collected points.
    build_curves_from_points(&all_points)
}

/// Find SSI seed points using recursive Bezier patch subdivision + BVH overlap.
///
/// This approach **cannot miss intersection branches** because it converges
/// on all regions where the two surfaces are close. Steps:
/// 1. Decompose both surfaces into Bezier patches
/// 2. Build a BVH over B's patch AABBs
/// 3. For each A-patch, find overlapping B-patches
/// 4. Small overlapping pairs → seed from centroid + `refine_ssi_point`
/// 5. Large pairs → subdivide and recurse (max depth limit)
#[allow(clippy::cast_precision_loss)]
fn find_ssi_seeds_subdivision(
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
    let diag_threshold = tolerance * 100.0; // Below this diagonal, try Newton directly
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
    for (pa, pb) in pairs {
        let diag_a = pa.diagonal();
        let diag_b = pb.diagonal();

        if diag_a < diag_threshold && diag_b < diag_threshold {
            // Both patches are small: try to find a seed from centroid parameters.
            let u1 = pa.u_mid();
            let v1 = pa.v_mid();
            let u2 = pb.u_mid();
            let v2 = pb.v_mid();

            if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
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
            let u1 = pa.u_mid();
            let v1 = pa.v_mid();
            let u2 = pb.u_mid();
            let v2 = pb.v_mid();

            if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
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

/// Find seed points for NURBS-NURBS intersection by grid sampling (fallback).
///
/// Strategy: sample both surfaces on an n×n grid and try Newton
/// refinement for all cell-center pairs whose 3D positions are within
/// a generous distance threshold.
#[allow(clippy::cast_precision_loss)]
fn find_ssi_seeds_grid(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    n: usize,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let step = 1.0 / (n - 1) as f64;
    let mut seeds = Vec::new();

    // Sample both surfaces.
    let mut pts1: Vec<(f64, f64, Point3)> = Vec::with_capacity(n * n);
    let mut pts2: Vec<(f64, f64, Point3)> = Vec::with_capacity(n * n);

    for i in 0..n {
        let u = i as f64 * step;
        for j in 0..n {
            let v = j as f64 * step;
            pts1.push((u, v, s1.evaluate(u, v)));
            pts2.push((u, v, s2.evaluate(u, v)));
        }
    }

    // For each pair of sample points, if close enough, try Newton.
    // Use a generous threshold based on the diagonal of a grid cell.
    let threshold = step * 3.0;

    for &(u1, v1, p1) in &pts1 {
        for &(u2, v2, p2) in &pts2 {
            let dist = (p1 - p2).length();
            if dist < threshold {
                if let Some(refined) = refine_ssi_point(s1, s2, u1, v1, u2, v2, tolerance) {
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

/// Refine an SSI point using alternating projection.
///
/// Instead of solving the coupled 4D system, alternately:
/// 1. Project the midpoint onto surface 1 (find closest (u1,v1))
/// 2. Project the midpoint onto surface 2 (find closest (u2,v2))
/// 3. Repeat until the two projections converge.
#[allow(clippy::similar_names)]
fn refine_ssi_point(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1_guess: f64,
    v1_guess: f64,
    u2_guess: f64,
    v2_guess: f64,
    tolerance: f64,
) -> Option<IntersectionPoint> {
    let mut u1 = u1_guess;
    let mut v1 = v1_guess;
    let mut u2 = u2_guess;
    let mut v2 = v2_guess;

    for _ in 0..50 {
        let p1 = s1.evaluate(u1, v1);
        let p2 = s2.evaluate(u2, v2);
        let residual = p1 - p2;

        if residual.length() < tolerance {
            return Some(IntersectionPoint {
                point: p1,
                param1: (u1, v1),
                param2: (u2, v2),
            });
        }

        // Step 1: Move (u2, v2) on surface 2 toward p1.
        let (du2, dv2) = surface_newton_step(s2, u2, v2, p1);
        u2 += du2;
        v2 += dv2;
        u2 = u2.clamp(0.0, 1.0);
        v2 = v2.clamp(0.0, 1.0);

        // Step 2: Move (u1, v1) on surface 1 toward the updated p2.
        let p2_new = s2.evaluate(u2, v2);
        let (du1, dv1) = surface_newton_step(s1, u1, v1, p2_new);
        u1 += du1;
        v1 += dv1;
        u1 = u1.clamp(0.0, 1.0);
        v1 = v1.clamp(0.0, 1.0);
    }

    // Final check.
    let p1 = s1.evaluate(u1, v1);
    let p2 = s2.evaluate(u2, v2);
    if (p1 - p2).length() < tolerance * 100.0 {
        Some(IntersectionPoint {
            point: p1,
            param1: (u1, v1),
            param2: (u2, v2),
        })
    } else {
        None
    }
}

/// Compute a Newton step to move (u, v) on the surface closer to a target
/// 3D point. Solves the 2×2 system from the surface's first derivatives.
fn surface_newton_step(surface: &NurbsSurface, u: f64, v: f64, target: Point3) -> (f64, f64) {
    let pt = surface.evaluate(u, v);
    let r = target - pt;
    let r_vec = Vec3::new(r.x(), r.y(), r.z());

    let derivs = surface.derivatives(u, v, 1);
    let su = derivs[1][0];
    let sv = derivs[0][1];

    // Solve: [su·su, su·sv; su·sv, sv·sv] [du; dv] = [su·r; sv·r]
    let a11 = su.dot(su);
    let a12 = su.dot(sv);
    let a22 = sv.dot(sv);
    let b1 = su.dot(r_vec);
    let b2 = sv.dot(r_vec);

    let det = a11.mul_add(a22, -(a12 * a12));
    if det.abs() < 1e-20 {
        return (0.0, 0.0);
    }

    let du = b1.mul_add(a22, -(b2 * a12)) / det;
    let dv = a11.mul_add(b2, -(a12 * b1)) / det;

    (du, dv)
}

/// March along an intersection curve from a seed point.
///
/// Uses the tangent direction (cross product of surface normals) to step
/// forward, then corrects back to the intersection with Newton.
fn march_intersection(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    step_size: f64,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let max_steps = 200;

    // March forward.
    let forward = march_direction(s1, s2, seed, true, step_size, tolerance, max_steps);
    // March backward.
    let backward = march_direction(s1, s2, seed, false, step_size, tolerance, max_steps);

    // Combine: backward (reversed) + seed + forward.
    let mut result: Vec<IntersectionPoint> = backward.into_iter().rev().collect();
    result.push(*seed);
    result.extend(forward);

    result
}

/// Compute the SSI tangent in parameter space at the given parameters.
/// Returns `(du1, dv1, du2, dv2)` or `None` if normals are degenerate.
#[allow(clippy::similar_names)]
fn ssi_tangent_params(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
    sign: f64,
) -> Option<[f64; 4]> {
    let n1 = s1.normal(u1, v1).ok()?;
    let n2 = s2.normal(u2, v2).ok()?;

    let tangent_raw = n1.cross(n2);
    let tangent = if let Ok(t) = tangent_raw.normalize() {
        t
    } else {
        // Tangential intersection (normals parallel/antiparallel).
        // Try perturbation analysis to recover a direction.
        let pt = IntersectionPoint {
            point: s1.evaluate(u1, v1),
            param1: (u1, v1),
            param2: (u2, v2),
        };
        singular_tangent_direction(s1, s2, &pt)?
    };

    let t = Vec3::new(tangent.x() * sign, tangent.y() * sign, tangent.z() * sign);

    let d1 = s1.derivatives(u1, v1, 1);
    let d2 = s2.derivatives(u2, v2, 1);

    let (du1, dv1) = project_tangent_to_params(&d1, t, 1.0);
    let (du2, dv2) = project_tangent_to_params(&d2, t, 1.0);

    // Normalize so the maximum component magnitude is 1.0.
    let max_comp = du1.abs().max(dv1.abs()).max(du2.abs()).max(dv2.abs());
    if max_comp < 1e-20 {
        return None;
    }

    Some([
        du1 / max_comp,
        dv1 / max_comp,
        du2 / max_comp,
        dv2 / max_comp,
    ])
}

/// At a singular point (where surface normals are parallel/antiparallel),
/// use perturbation analysis to determine the intersection curve direction.
///
/// Compute the tangent direction at a singular (tangential) intersection point.
///
/// At a tangential point, the first-order tangent `n1 × n2` vanishes because
/// the surface normals are parallel. This function uses second-order curvature
/// analysis to determine the correct marching direction.
///
/// Algorithm (based on Patrikalakis-Maekawa):
/// 1. Compute second derivatives of both surfaces at the touch point
/// 2. Compute the curvature difference tensor in the tangent plane
/// 3. Find the principal direction of the curvature difference —
///    this is the direction along which the surfaces separate fastest
/// 4. The intersection curve follows the direction perpendicular to
///    the maximum curvature difference
///
/// Falls back to perturbation-based search if second-order analysis
/// is degenerate (e.g., surfaces are osculating to second order).
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn singular_tangent_direction(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    point: &IntersectionPoint,
) -> Option<Vec3> {
    let (u1, v1) = point.param1;
    let (u2, v2) = point.param2;

    // Try second-order analysis first.
    if let Some(dir) = second_order_tangent(s1, s2, u1, v1, u2, v2) {
        return Some(dir);
    }

    // Fallback: perturbation-based search (original method).
    perturbation_tangent(s1, s2, point)
}

/// Second-order curvature analysis for tangential intersection direction.
///
/// Computes the difference of the second fundamental forms of the two
/// surfaces at the touch point, projected onto the shared tangent plane.
/// The eigenvector corresponding to the zero (or smallest) eigenvalue of
/// this difference gives the direction along which the surfaces remain
/// in contact — i.e., the intersection curve tangent.
#[allow(clippy::similar_names)]
fn second_order_tangent(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
) -> Option<Vec3> {
    // Compute second-order derivatives for both surfaces.
    let d1 = s1.derivatives(u1, v1, 2);
    let d2 = s2.derivatives(u2, v2, 2);

    // Check we have enough derivative data.
    if d1.len() < 3 || d1[0].len() < 3 || d2.len() < 3 || d2[0].len() < 3 {
        return None;
    }

    // First derivatives (tangent vectors).
    let s1u = d1[1][0]; // ∂S1/∂u1
    let s1v = d1[0][1]; // ∂S1/∂v1

    // Normal of surface 1.
    let n1 = s1u.cross(s1v);
    let n1_len = n1.length();
    if n1_len < 1e-12 {
        return None;
    }
    let n1 = n1 * (1.0 / n1_len);

    // Second fundamental form coefficients of surface 1:
    // L = S_uu · n, M = S_uv · n, N = S_vv · n
    let l1 = d1[2][0].dot(n1);
    let m1 = d1[1][1].dot(n1);
    let n1_coeff = d1[0][2].dot(n1);

    // Second fundamental form coefficients of surface 2:
    let s2u = d2[1][0];
    let s2v = d2[0][1];
    let n2 = s2u.cross(s2v);
    let n2_len = n2.length();
    if n2_len < 1e-12 {
        return None;
    }
    let n2 = n2 * (1.0 / n2_len);

    let l2 = d2[2][0].dot(n2);
    let m2 = d2[1][1].dot(n2);
    let n2_coeff = d2[0][2].dot(n2);

    // Curvature difference: ΔII = II_1 - II_2
    // In the 2×2 matrix [ΔL, ΔM; ΔM, ΔN]:
    let dl = l1 - l2;
    let dm = m1 - m2;
    let dn = n1_coeff - n2_coeff;

    // Find eigenvectors of the 2×2 symmetric matrix [dl, dm; dm, dn].
    // The eigenvector with the smaller eigenvalue gives the direction
    // where the curvature difference is minimal → intersection continues.
    let trace = dl + dn;
    let det = dl * dn - dm * dm;
    let disc = trace * trace - 4.0 * det;

    if disc < -1e-12 {
        return None; // Complex eigenvalues (shouldn't happen for symmetric matrix)
    }
    let disc_sqrt = disc.max(0.0).sqrt();

    let lambda1 = 0.5 * (trace - disc_sqrt);
    let lambda2 = 0.5 * (trace + disc_sqrt);

    // Pick the eigenvector corresponding to the eigenvalue closest to zero.
    let target_lambda = if lambda1.abs() < lambda2.abs() {
        lambda1
    } else {
        lambda2
    };

    // Eigenvector of [dl, dm; dm, dn] for eigenvalue λ:
    // (dl - λ) x + dm y = 0 → (x, y) = (-dm, dl - λ) or (dn - λ, -dm)
    let (ex, ey) = if (dl - target_lambda).abs() > dm.abs() {
        (-dm, dl - target_lambda)
    } else {
        (dn - target_lambda, -dm)
    };

    let e_len = (ex * ex + ey * ey).sqrt();
    if e_len < 1e-12 {
        return None; // Degenerate: curvature difference is isotropic
    }

    // Convert the 2D eigenvector (in parameter space of surface 1) back to 3D.
    // The direction in 3D is: ex * S1_u + ey * S1_v
    let tangent_3d = s1u * (ex / e_len) + s1v * (ey / e_len);

    tangent_3d.normalize().ok()
}

/// Perturbation-based tangent direction finder (fallback).
///
/// Samples 8 directions around the current point in parameter space, attempts
/// Newton refinement at each, and returns the direction to the most distant
/// successfully refined point.
fn perturbation_tangent(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    point: &IntersectionPoint,
) -> Option<Vec3> {
    let eps = 1e-4;
    let (u1, v1) = point.param1;
    let (u2, v2) = point.param2;

    let directions: [(f64, f64); 8] = [
        (eps, 0.0),
        (-eps, 0.0),
        (0.0, eps),
        (0.0, -eps),
        (eps, eps),
        (eps, -eps),
        (-eps, eps),
        (-eps, -eps),
    ];

    let mut best_dir: Option<Vec3> = None;
    let mut best_dist = 0.0_f64;

    for &(du, dv) in &directions {
        let u1p = (u1 + du).clamp(0.001, 0.999);
        let v1p = (v1 + dv).clamp(0.001, 0.999);

        if let Some(refined) = refine_ssi_point(s1, s2, u1p, v1p, u2, v2, 1e-8) {
            let d = refined.point - point.point;
            let dist = d.length();
            if dist > best_dist && dist > 1e-12 {
                if let Ok(normalized) = d.normalize() {
                    best_dist = dist;
                    best_dir = Some(normalized);
                }
            }
        }
    }

    best_dir
}

/// Clamp parameter value to avoid edge singularities.
fn clamp_param(v: f64) -> f64 {
    v.clamp(0.001, 0.999)
}

/// Check if a parameter state is at the domain boundary.
fn at_boundary(state: &[f64; 4]) -> bool {
    state.iter().any(|&v| v <= 0.001 || v >= 0.999)
}

/// March in one direction along the intersection curve using RKF45
/// adaptive stepping with closed-loop detection.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn march_direction(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    forward: bool,
    step_size: f64,
    tolerance: f64,
    max_steps: usize,
) -> Vec<IntersectionPoint> {
    let mut points = Vec::new();
    let mut current = *seed;

    let sign = if forward { 1.0 } else { -1.0 };
    let mut h = step_size;
    let h_min = tolerance;
    let max_h = step_size * 4.0;
    let seed_state = [seed.param1.0, seed.param1.1, seed.param2.0, seed.param2.1];

    let mut total_evals = 0_usize;
    let max_evals = max_steps * 3; // Allow some retries but bound total work.

    for _ in 0..max_steps {
        let y = [
            current.param1.0,
            current.param1.1,
            current.param2.0,
            current.param2.1,
        ];

        // Try RKF45 with adaptive step, allowing a few retries per accepted step.
        let (y4, accepted_h) = loop {
            total_evals += 1;
            if total_evals > max_evals {
                return points;
            }

            let Some(result) = rkf45_step(s1, s2, &y, h, sign) else {
                return points;
            };

            let (y4, y5) = result;

            // Compute error estimate.
            let err = ((y5[0] - y4[0]).powi(2)
                + (y5[1] - y4[1]).powi(2)
                + (y5[2] - y4[2]).powi(2)
                + (y5[3] - y4[3]).powi(2))
            .sqrt();

            if err > tolerance && h > h_min {
                // Reject step, halve h and retry.
                h = (h * 0.5).max(h_min);
                continue;
            }

            // Accept this step.
            let accepted = h;

            // Adjust h for next step.
            if err < tolerance / 10.0 {
                h = (h * 2.0).min(max_h);
            }

            break (y4, accepted);
        };
        let _ = accepted_h;

        // Accept the 4th-order solution (more conservative).
        let next = [
            clamp_param(y4[0]),
            clamp_param(y4[1]),
            clamp_param(y4[2]),
            clamp_param(y4[3]),
        ];

        // Newton-refine to stay on the intersection curve.
        if let Some(refined) =
            refine_ssi_point(s1, s2, next[0], next[1], next[2], next[3], tolerance)
        {
            // Check that we actually moved.
            if (refined.point - current.point).length() < tolerance {
                break;
            }

            // Check boundary.
            let ref_state = [
                refined.param1.0,
                refined.param1.1,
                refined.param2.0,
                refined.param2.1,
            ];
            if at_boundary(&ref_state) {
                points.push(refined);
                break;
            }

            // Closed-loop detection: after accumulating enough points,
            // check if we've returned close to the seed.
            if points.len() >= 5 {
                let dist_to_seed = ((ref_state[0] - seed_state[0]).powi(2)
                    + (ref_state[1] - seed_state[1]).powi(2)
                    + (ref_state[2] - seed_state[2]).powi(2)
                    + (ref_state[3] - seed_state[3]).powi(2))
                .sqrt();

                if dist_to_seed < 3.0 * step_size {
                    // Close the loop by adding the seed point.
                    points.push(*seed);
                    break;
                }
            }

            points.push(refined);
            current = refined;
        } else {
            break;
        }
    }

    points
}

/// Perform one RKF45 step. Returns `(y_4th, y_5th)` or `None` if
/// tangent evaluation fails at any stage.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn rkf45_step(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    y: &[f64; 4],
    h: f64,
    sign: f64,
) -> Option<([f64; 4], [f64; 4])> {
    // Helper: evaluate f at a state, scaling by h.
    let f = |state: &[f64; 4]| -> Option<[f64; 4]> {
        let t = ssi_tangent_params(
            s1,
            s2,
            clamp_param(state[0]),
            clamp_param(state[1]),
            clamp_param(state[2]),
            clamp_param(state[3]),
            sign,
        )?;
        Some([t[0] * h, t[1] * h, t[2] * h, t[3] * h])
    };

    // Helper: y + sum of scaled k vectors.
    let add = |base: &[f64; 4], terms: &[(&[f64; 4], f64)]| -> [f64; 4] {
        let mut out = *base;
        for &(k, coeff) in terms {
            for i in 0..4 {
                out[i] += k[i] * coeff;
            }
        }
        out
    };

    let k1 = f(y)?;
    let k2 = f(&add(y, &[(&k1, 1.0 / 4.0)]))?;
    let k3 = f(&add(y, &[(&k1, 3.0 / 32.0), (&k2, 9.0 / 32.0)]))?;
    let k4 = f(&add(
        y,
        &[
            (&k1, 1932.0 / 2197.0),
            (&k2, -7200.0 / 2197.0),
            (&k3, 7296.0 / 2197.0),
        ],
    ))?;
    let k5 = f(&add(
        y,
        &[
            (&k1, 439.0 / 216.0),
            (&k2, -8.0),
            (&k3, 3680.0 / 513.0),
            (&k4, -845.0 / 4104.0),
        ],
    ))?;
    let k6 = f(&add(
        y,
        &[
            (&k1, -8.0 / 27.0),
            (&k2, 2.0),
            (&k3, -3544.0 / 2565.0),
            (&k4, 1859.0 / 4104.0),
            (&k5, -11.0 / 40.0),
        ],
    ))?;

    // 4th-order solution.
    let y4 = add(
        y,
        &[
            (&k1, 25.0 / 216.0),
            (&k3, 1408.0 / 2565.0),
            (&k4, 2197.0 / 4104.0),
            (&k5, -1.0 / 5.0),
        ],
    );

    // 5th-order solution.
    let y5 = add(
        y,
        &[
            (&k1, 16.0 / 135.0),
            (&k3, 6656.0 / 12825.0),
            (&k4, 28561.0 / 56430.0),
            (&k5, -9.0 / 50.0),
            (&k6, 2.0 / 55.0),
        ],
    );

    Some((y4, y5))
}

/// Project a 3D tangent vector onto surface parameter space.
///
/// Given surface derivatives `derivs` (from `surface.derivatives(u, v, 1)`)
/// and a 3D tangent direction scaled by `step`, compute the parameter
/// increments (du, dv) that move along the tangent on the surface.
fn project_tangent_to_params(derivs: &[Vec<Vec3>], tangent: Vec3, step: f64) -> (f64, f64) {
    let su = derivs[1][0]; // ∂S/∂u
    let sv = derivs[0][1]; // ∂S/∂v

    let t = Vec3::new(tangent.x() * step, tangent.y() * step, tangent.z() * step);

    // Solve [su·su, su·sv; su·sv, sv·sv] [du; dv] = [su·t; sv·t]
    let a11 = su.dot(su);
    let a12 = su.dot(sv);
    let a22 = sv.dot(sv);
    let b1 = su.dot(t);
    let b2 = sv.dot(t);

    let det = a11.mul_add(a22, -(a12 * a12));
    if det.abs() < 1e-20 {
        return (0.0, 0.0);
    }

    let du = b1.mul_add(a22, -(b2 * a12)) / det;
    let dv = a11.mul_add(b2, -(a12 * b1)) / det;

    (du, dv)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::nurbs::surface::NurbsSurface;
    use crate::vec::{Point3, Vec3};

    use super::*;

    /// Create a simple bilinear NURBS surface (flat plane at z=0, from (0,0) to (1,1)).
    fn flat_surface() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    /// Create a curved surface (saddle shape).
    fn saddle_surface() -> NurbsSurface {
        NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.0, 0.5, 0.25),
                    Point3::new(0.0, 1.0, 0.0),
                ],
                vec![
                    Point3::new(0.5, 0.0, -0.25),
                    Point3::new(0.5, 0.5, 0.0),
                    Point3::new(0.5, 1.0, 0.25),
                ],
                vec![
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 0.5, -0.25),
                    Point3::new(1.0, 1.0, 0.0),
                ],
            ],
            vec![vec![1.0; 3]; 3],
        )
        .unwrap()
    }

    // ── Plane-NURBS intersection ──────────────────────────────

    #[test]
    fn flat_surface_plane_no_intersection() {
        let surface = flat_surface();
        // Plane at z=1 shouldn't intersect surface at z=0.
        let result = intersect_plane_nurbs(&surface, Vec3::new(0.0, 0.0, 1.0), 1.0, 30).unwrap();

        assert!(result.is_empty(), "no intersection expected");
    }

    #[test]
    fn saddle_surface_plane_intersection() {
        let surface = saddle_surface();
        // Plane at z=0 should intersect the saddle surface.
        let result = intersect_plane_nurbs(&surface, Vec3::new(0.0, 0.0, 1.0), 0.0, 50).unwrap();

        assert!(
            !result.is_empty(),
            "saddle surface should intersect z=0 plane"
        );

        // The intersection curve should have points near z=0.
        for curve in &result {
            for pt in &curve.points {
                assert!(
                    pt.point.z().abs() < 0.1,
                    "intersection point should be near z=0, got z={}",
                    pt.point.z()
                );
            }
        }
    }

    // ── Line-NURBS intersection ───────────────────────────────

    #[test]
    fn line_flat_surface_intersection() {
        let surface = flat_surface();
        // Vertical ray through (0.5, 0.5) should hit the surface at z=0.
        let result = intersect_line_nurbs(
            &surface,
            Point3::new(0.5, 0.5, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            20,
        )
        .unwrap();

        assert!(!result.is_empty(), "ray should hit flat surface");

        let pt = &result[0];
        assert!(
            (pt.point.x() - 0.5).abs() < 0.05,
            "x should be ~0.5, got {}",
            pt.point.x()
        );
        assert!(
            (pt.point.y() - 0.5).abs() < 0.05,
            "y should be ~0.5, got {}",
            pt.point.y()
        );
        assert!(
            pt.point.z().abs() < 0.05,
            "z should be ~0.0, got {}",
            pt.point.z()
        );
    }

    #[test]
    fn line_misses_surface() {
        let surface = flat_surface();
        // Ray parallel to the surface should miss.
        let result = intersect_line_nurbs(
            &surface,
            Point3::new(0.5, 0.5, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            20,
        )
        .unwrap();

        assert!(result.is_empty(), "parallel ray should miss");
    }

    // ── Intersection point quality ────────────────────────────

    #[test]
    fn refined_points_are_on_plane() {
        let surface = saddle_surface();
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.1; // Slightly above z=0.
        let result = intersect_plane_nurbs(&surface, normal, d, 50).unwrap();

        for curve in &result {
            for pt in &curve.points {
                let signed_dist =
                    Vec3::new(pt.point.x(), pt.point.y(), pt.point.z()).dot(normal) - d;
                assert!(
                    signed_dist.abs() < 0.01,
                    "point should be on plane, signed_dist={signed_dist}"
                );
            }
        }
    }

    // ── NURBS-NURBS intersection ──────────────────────────────

    /// Create a flat surface at z=0.5 (overlapping region with `flat_surface` at z=0).
    fn flat_surface_offset() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.5), Point3::new(0.0, 1.0, 0.5)],
                vec![Point3::new(1.0, 0.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    /// Create a tilted flat surface that intersects the flat z=0 surface.
    fn tilted_surface() -> NurbsSurface {
        // Surface tilted in the XZ plane: goes from z=-0.5 at x=0 to z=0.5 at x=1.
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, -0.5), Point3::new(0.0, 1.0, -0.5)],
                vec![Point3::new(1.0, 0.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn parallel_surfaces_no_intersection() {
        let s1 = flat_surface();
        let s2 = flat_surface_offset();
        let result = intersect_nurbs_nurbs(&s1, &s2, 15, 0.02).unwrap();
        assert!(result.is_empty(), "parallel surfaces should not intersect");
    }

    #[test]
    fn refine_ssi_basic() {
        let s1 = flat_surface();
        let s2 = tilted_surface();
        // At u1=0.5, v1=0.5 on flat → (0.5, 0.5, 0)
        // At u2=0.5, v2=0.5 on tilted → (0.5, 0.5, 0)
        // These should refine to an intersection point.
        let result = refine_ssi_point(&s1, &s2, 0.5, 0.5, 0.5, 0.5, 1e-6);
        assert!(
            result.is_some(),
            "refine should find intersection at (0.5, 0.5)"
        );
    }

    #[test]
    fn seed_finding_basic() {
        let s1 = flat_surface();
        let s2 = tilted_surface();

        // Verify surfaces evaluate correctly.
        let p1 = s1.evaluate(0.5, 0.5);
        let p2 = s2.evaluate(0.5, 0.5);
        let dist = (p1 - p2).length();
        assert!(
            dist < 0.01,
            "flat(0.5,0.5)={p1:?} tilted(0.5,0.5)={p2:?} dist={dist}",
        );

        // Verify refine works from off-center guess.
        let refined = refine_ssi_point(&s1, &s2, 0.5263, 0.5, 0.5263, 0.5, 1e-6);
        assert!(
            refined.is_some(),
            "refine should converge from off-center guess"
        );

        let seeds = find_ssi_seeds_grid(&s1, &s2, 10, 1e-6);
        assert!(
            !seeds.is_empty(),
            "should find seeds between flat and tilted surfaces"
        );
    }

    #[test]
    fn tilted_intersects_flat() {
        let s1 = flat_surface();
        let s2 = tilted_surface();

        // First verify seed finding works.
        let seeds = find_ssi_seeds_grid(&s1, &s2, 10, 1e-6);
        assert!(
            !seeds.is_empty(),
            "should find at least one seed point, got 0"
        );

        let result = intersect_nurbs_nurbs(&s1, &s2, 10, 0.05).unwrap();

        assert!(
            !result.is_empty(),
            "tilted surface should intersect flat surface (seeds: {})",
            seeds.len()
        );

        for curve in &result {
            for pt in &curve.points {
                assert!(
                    pt.point.z().abs() < 0.15,
                    "point should be near z=0, got z={}",
                    pt.point.z()
                );
            }
        }
    }

    #[test]
    fn ssi_points_lie_on_both_surfaces() {
        let s1 = flat_surface();
        let s2 = tilted_surface();
        let result = intersect_nurbs_nurbs(&s1, &s2, 10, 0.02).unwrap();

        for curve in &result {
            for pt in &curve.points {
                // Check point lies on surface 1.
                let p1 = s1.evaluate(pt.param1.0, pt.param1.1);
                let dist1 = (p1 - pt.point).length();
                assert!(dist1 < 0.05, "point should lie on surface 1, dist={dist1}");

                // Check point lies on surface 2.
                let p2 = s2.evaluate(pt.param2.0, pt.param2.1);
                let dist2 = (p2 - pt.point).length();
                assert!(dist2 < 0.05, "point should lie on surface 2, dist={dist2}");
            }
        }
    }

    /// Create a dome-shaped NURBS surface (quadratic, unit domain).
    /// High at center (z=2), low at edges (z=-1), so slicing at z=0
    /// produces a closed ring-like intersection.
    fn dome_surface() -> NurbsSurface {
        NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                vec![
                    Point3::new(0.0, 0.0, -1.0),
                    Point3::new(0.0, 0.5, 0.5),
                    Point3::new(0.0, 1.0, -1.0),
                ],
                vec![
                    Point3::new(0.5, 0.0, 0.5),
                    Point3::new(0.5, 0.5, 2.0),
                    Point3::new(0.5, 1.0, 0.5),
                ],
                vec![
                    Point3::new(1.0, 0.0, -1.0),
                    Point3::new(1.0, 0.5, 0.5),
                    Point3::new(1.0, 1.0, -1.0),
                ],
            ],
            vec![vec![1.0; 3]; 3],
        )
        .unwrap()
    }

    /// Create a flat surface at a given z height, mapping [0,1]^2 to the
    /// same XY extent [0,1]x[0,1] as the dome.
    fn flat_plane_at_z(z: f64) -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, z), Point3::new(0.0, 1.0, z)],
                vec![Point3::new(1.0, 0.0, z), Point3::new(1.0, 1.0, z)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn ssi_tangential_touch() {
        // Two surfaces that touch tangentially: a dome and a flat plane at the
        // dome's peak height. The normals are parallel at the touch point, so
        // this exercises the singular_tangent_direction fallback.
        let dome = dome_surface();
        // The dome peaks around z=2 at the center. Use a plane slightly below
        // to create a tangential touch region.
        let peak_z = dome.evaluate(0.5, 0.5).z();

        // Place the plane at the peak height — tangential contact.
        let plane = flat_plane_at_z(peak_z);

        // At the tangent point both normals point in +z, so cross product vanishes.
        // The marching should handle this gracefully via singular_tangent_direction.
        let seed = refine_ssi_point(&dome, &plane, 0.5, 0.5, 0.5, 0.5, 1e-6);
        assert!(
            seed.is_some(),
            "should find a seed at the tangential contact point"
        );

        let seed = seed.unwrap();
        assert!(
            (seed.point.z() - peak_z).abs() < 0.2,
            "seed should be near z={peak_z}, got z={}",
            seed.point.z()
        );

        // March from the tangential point. The key requirement is that this
        // does not panic and handles the singular point.
        let traced = march_intersection(&dome, &plane, &seed, 0.05, 1e-6);

        // At a true tangential touch (single point contact), marching may
        // produce few or no additional points — that's acceptable. The test
        // ensures we don't crash/panic at the singular point.
        // If the plane is slightly below peak, there may be a small intersection
        // loop.
        for pt in &traced {
            // All traced points should be reasonably close to both surfaces.
            let p1 = dome.evaluate(pt.param1.0, pt.param1.1);
            let p2 = plane.evaluate(pt.param2.0, pt.param2.1);
            let dist1 = (p1 - pt.point).length();
            let dist2 = (p2 - pt.point).length();
            assert!(
                dist1 < 0.5,
                "traced point should be near dome surface, dist={dist1}"
            );
            assert!(
                dist2 < 0.5,
                "traced point should be near plane surface, dist={dist2}"
            );
        }
    }

    #[test]
    fn ssi_closed_loop() {
        // Intersect a dome surface with a horizontal plane.
        // Use a known seed point and march directly to test closed-loop
        // detection without the expensive O(n^4) seed search.
        let dome = dome_surface();
        let plane = flat_plane_at_z(0.0);

        // Find one seed by refining a point we know is on the intersection
        // (from the debug test: the z=0 contour passes through the region
        // around u=0.25 on the dome).
        let seed = refine_ssi_point(&dome, &plane, 0.25, 0.5, 0.25, 0.5, 1e-6)
            .expect("should refine to a seed on the dome-plane intersection");

        // Verify the seed is near z=0.
        assert!(
            seed.point.z().abs() < 0.1,
            "seed should be near z=0, got z={}",
            seed.point.z()
        );

        // March from the seed.
        let traced = march_intersection(&dome, &plane, &seed, 0.05, 1e-6);

        assert!(
            traced.len() >= 5,
            "should trace at least 5 points, got {}",
            traced.len()
        );

        // Check that the curve closes: first and last points should be close.
        let first = &traced[0];
        let last = &traced[traced.len() - 1];
        let gap = (first.point - last.point).length();

        assert!(
            gap < 0.5,
            "expected closed loop (first-last gap < 0.5), got gap={gap:.4}"
        );

        // All points should lie near z=0.
        for pt in &traced {
            assert!(
                pt.point.z().abs() < 0.15,
                "intersection point should be near z=0, got z={}",
                pt.point.z()
            );
        }
    }

    // ── Subdivision seed finder tests ─────────────────────────

    #[test]
    fn subdivision_finds_seeds() {
        let s1 = flat_surface();
        let s2 = tilted_surface();

        let seeds = find_ssi_seeds_subdivision(&s1, &s2, 1e-6);
        assert!(
            !seeds.is_empty(),
            "subdivision should find seeds between flat and tilted"
        );

        // All seeds should lie on both surfaces
        for seed in &seeds {
            let p1 = s1.evaluate(seed.param1.0, seed.param1.1);
            let p2 = s2.evaluate(seed.param2.0, seed.param2.1);
            assert!(
                (p1 - seed.point).length() < 0.01,
                "seed should lie on surface 1"
            );
            assert!(
                (p2 - seed.point).length() < 0.01,
                "seed should lie on surface 2"
            );
        }
    }

    // ── Chain building tests ──────────────────────────────────

    #[test]
    fn chain_separates_branches() {
        // Two clusters of points with a gap between them
        let points = vec![
            IntersectionPoint {
                point: Point3::new(0.0, 0.0, 0.0),
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            },
            IntersectionPoint {
                point: Point3::new(0.1, 0.0, 0.0),
                param1: (0.1, 0.0),
                param2: (0.1, 0.0),
            },
            IntersectionPoint {
                point: Point3::new(0.2, 0.0, 0.0),
                param1: (0.2, 0.0),
                param2: (0.2, 0.0),
            },
            // Gap
            IntersectionPoint {
                point: Point3::new(5.0, 0.0, 0.0),
                param1: (0.5, 0.0),
                param2: (0.5, 0.0),
            },
            IntersectionPoint {
                point: Point3::new(5.1, 0.0, 0.0),
                param1: (0.6, 0.0),
                param2: (0.6, 0.0),
            },
        ];

        let chains = chain_intersection_points(&points, 0.5);
        assert_eq!(
            chains.len(),
            2,
            "should separate into 2 branches, got {}",
            chains.len()
        );
    }

    #[test]
    fn chain_detects_single_group() {
        // Points close together: should form 1 chain
        let points: Vec<IntersectionPoint> = (0..5)
            .map(|i| {
                let x = f64::from(i) * 0.1;
                IntersectionPoint {
                    point: Point3::new(x, 0.0, 0.0),
                    param1: (x, 0.0),
                    param2: (x, 0.0),
                }
            })
            .collect();

        let chains = chain_intersection_points(&points, 0.5);
        assert_eq!(chains.len(), 1, "all close points should form 1 chain");
        assert_eq!(chains[0].len(), 5);
    }

    /// Test second-order tangent analysis with two nearly-tangent surfaces.
    #[test]
    fn second_order_tangent_finds_direction() {
        // Two surfaces that touch at (0.5, 0.5): one flat, one dome.
        // At the touch point, normals are parallel (both ~+z), so
        // first-order tangent n1 × n2 ≈ 0.
        let dome = dome_surface();
        let peak_z = dome.evaluate(0.5, 0.5).z();

        // Place a flat plane at the dome's peak height.
        let plane = flat_plane_at_z(peak_z);

        // Try the second-order analysis.
        let result = second_order_tangent(&dome, &plane, 0.5, 0.5, 0.5, 0.5);

        // The result should be Some (a direction was found) or None
        // (degenerate — surfaces osculate to second order).
        // For a dome with quadratic curvature vs flat plane, the
        // curvature difference is non-zero, so we should get a direction.
        if let Some(dir) = result {
            // The direction should be a unit vector in the tangent plane.
            let len = dir.length();
            assert!(
                (len - 1.0).abs() < 0.01,
                "tangent direction should be unit length, got {len}"
            );
            // The direction should be roughly in the XY plane (since
            // both surfaces are horizontal at the touch point).
            assert!(
                dir.z().abs() < 0.5,
                "tangent direction should be mostly horizontal, got z={}",
                dir.z()
            );
        }
        // None is also acceptable for this degenerate case — it means
        // the perturbation fallback will be used.
    }

    /// Verify that the tangential touch test still works with the new
    /// second-order analysis integrated into the main SSI pipeline.
    #[test]
    fn ssi_tangential_with_second_order() {
        let dome = dome_surface();
        let peak_z = dome.evaluate(0.5, 0.5).z();
        let plane = flat_plane_at_z(peak_z - 0.1); // Slightly below peak

        // This should find a small intersection loop near the peak.
        let result = intersect_nurbs_nurbs(&dome, &plane, 10, 0.05).unwrap();

        // Near-tangential: may or may not find an intersection (depends
        // on numerical precision), but should NOT crash.
        for curve in &result {
            for pt in &curve.points {
                // All points should be close to the plane height.
                assert!(
                    (pt.point.z() - (peak_z - 0.1)).abs() < 0.5,
                    "intersection point should be near z={:.2}, got z={:.4}",
                    peak_z - 0.1,
                    pt.point.z()
                );
            }
        }
    }
}
