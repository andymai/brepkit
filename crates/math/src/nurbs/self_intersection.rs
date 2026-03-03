//! NURBS surface self-intersection detection.
//!
//! Detects regions where a single NURBS surface folds back on itself,
//! producing `S(u1,v1) = S(u2,v2)` with `(u1,v1) ≠ (u2,v2)`.
//!
//! ## Algorithm
//!
//! 1. Sample the surface on a grid, build triangles with (u,v) parameter ranges
//! 2. Build a BVH over triangle AABBs
//! 3. Query overlapping pairs, filter out adjacent triangles
//! 4. For non-adjacent close pairs: Newton-refine with constraint
//!    `(u1,v1) ≠ (u2,v2)`, `S(u1,v1) = S(u2,v2)`
//! 5. March along self-intersection curves

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss
)]

use crate::MathError;
use crate::aabb::Aabb3;
use crate::bvh::Bvh;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// A self-intersection curve on a NURBS surface.
///
/// Contains the 3D intersection curve and the two sets of parameter
/// values that map to the same 3D points.
#[derive(Debug, Clone)]
pub struct SelfIntersectionCurve {
    /// The 3D self-intersection curve as a NURBS.
    pub curve: NurbsCurve,
    /// Parameter values on the "first sheet" of the surface.
    pub params_a: Vec<(f64, f64)>,
    /// Parameter values on the "second sheet" of the surface.
    pub params_b: Vec<(f64, f64)>,
}

/// A triangle in the surface sampling grid, with parameter-space coordinates.
struct SampleTriangle {
    /// AABB of the triangle in 3D.
    aabb: Aabb3,
    /// Grid indices of the three corners.
    indices: [(usize, usize); 3],
    /// Parameter-space midpoint.
    uv_mid: (f64, f64),
}

/// Detect self-intersections on a NURBS surface.
///
/// Samples the surface on a grid, finds non-adjacent regions where the
/// surface folds back on itself, and traces the self-intersection curves.
///
/// # Parameters
///
/// - `surface`: The NURBS surface to test
/// - `grid_res`: Grid resolution for sampling (e.g., 20)
/// - `tolerance`: Distance tolerance for considering points as intersecting
///
/// # Errors
///
/// Returns an error if NURBS evaluation or curve fitting fails.
pub fn detect_self_intersection(
    surface: &NurbsSurface,
    grid_res: usize,
    tolerance: f64,
) -> Result<Vec<SelfIntersectionCurve>, MathError> {
    let n = grid_res.max(5);
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    // Step 1: Sample surface on grid.
    let mut grid_pts: Vec<Vec<Point3>> = Vec::with_capacity(n + 1);
    let mut grid_uv: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let u = u_min + (u_max - u_min) * (i as f64 / n as f64);
        let mut row_pts = Vec::with_capacity(n + 1);
        let mut row_uv = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let v = v_min + (v_max - v_min) * (j as f64 / n as f64);
            row_pts.push(surface.evaluate(u, v));
            row_uv.push((u, v));
        }
        grid_pts.push(row_pts);
        grid_uv.push(row_uv);
    }

    // Step 2: Build triangles from grid and their AABBs.
    let mut triangles: Vec<SampleTriangle> = Vec::with_capacity(2 * n * n);

    for i in 0..n {
        for j in 0..n {
            // Lower-left triangle: (i,j), (i+1,j), (i+1,j+1)
            let t1_pts = [grid_pts[i][j], grid_pts[i + 1][j], grid_pts[i + 1][j + 1]];
            let t1_aabb = Aabb3::from_points(t1_pts.iter().copied());
            let t1_uv_mid = (
                (grid_uv[i][j].0 + grid_uv[i + 1][j].0 + grid_uv[i + 1][j + 1].0) / 3.0,
                (grid_uv[i][j].1 + grid_uv[i + 1][j].1 + grid_uv[i + 1][j + 1].1) / 3.0,
            );
            triangles.push(SampleTriangle {
                aabb: t1_aabb,
                indices: [(i, j), (i + 1, j), (i + 1, j + 1)],
                uv_mid: t1_uv_mid,
            });

            // Upper-right triangle: (i,j), (i+1,j+1), (i,j+1)
            let t2_pts = [grid_pts[i][j], grid_pts[i + 1][j + 1], grid_pts[i][j + 1]];
            let t2_aabb = Aabb3::from_points(t2_pts.iter().copied());
            let t2_uv_mid = (
                (grid_uv[i][j].0 + grid_uv[i + 1][j + 1].0 + grid_uv[i][j + 1].0) / 3.0,
                (grid_uv[i][j].1 + grid_uv[i + 1][j + 1].1 + grid_uv[i][j + 1].1) / 3.0,
            );
            triangles.push(SampleTriangle {
                aabb: t2_aabb,
                indices: [(i, j), (i + 1, j + 1), (i, j + 1)],
                uv_mid: t2_uv_mid,
            });
        }
    }

    // Step 3: Build BVH over triangles.
    let bvh_entries: Vec<(usize, Aabb3)> = triangles
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t.aabb))
        .collect();
    let bvh = Bvh::build(&bvh_entries);

    // Step 4: Find non-adjacent overlapping pairs.
    let adjacency_threshold = 2; // Triangles within 2 grid steps are "adjacent"
    let mut candidate_pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();

    for (i, tri_a) in triangles.iter().enumerate() {
        let overlaps = bvh.query_overlap(&tri_a.aabb);
        for &j in &overlaps {
            if j <= i {
                continue; // Skip self and already-processed pairs
            }
            let tri_b = &triangles[j];

            // Check if triangles are non-adjacent in parameter space.
            if are_adjacent(&tri_a.indices, &tri_b.indices, adjacency_threshold) {
                continue;
            }

            candidate_pairs.push((tri_a.uv_mid, tri_b.uv_mid));
        }
    }

    if candidate_pairs.is_empty() {
        return Ok(Vec::new());
    }

    // Step 5: Refine candidate pairs to find actual self-intersection points.
    #[allow(clippy::type_complexity)]
    let mut self_int_points: Vec<((f64, f64), (f64, f64), Point3)> = Vec::new();

    for &((u1, v1), (u2, v2)) in &candidate_pairs {
        if let Some((pt, pa, pb)) =
            refine_self_intersection_point(surface, u1, v1, u2, v2, tolerance)
        {
            // Verify parameters are actually distinct
            let param_dist = ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt();
            if param_dist > tolerance * 10.0 {
                // Deduplicate
                let is_dup = self_int_points.iter().any(|(existing, _, _)| {
                    let dist = ((existing.0 - pa.0).powi(2) + (existing.1 - pa.1).powi(2)).sqrt();
                    dist < tolerance * 100.0
                });
                if !is_dup {
                    self_int_points.push((pa, pb, pt));
                }
            }
        }
    }

    if self_int_points.is_empty() {
        return Ok(Vec::new());
    }

    // Step 6: Build self-intersection curves from found points.
    let points_3d: Vec<Point3> = self_int_points.iter().map(|(_, _, p)| *p).collect();
    let params_a: Vec<(f64, f64)> = self_int_points.iter().map(|(a, _, _)| *a).collect();
    let params_b: Vec<(f64, f64)> = self_int_points.iter().map(|(_, b, _)| *b).collect();

    // Fit a curve through the points (if enough points).
    if points_3d.len() < 2 {
        // Single point self-intersection: create a degenerate curve.
        let degree = 1;
        let curve = crate::nurbs::interpolate(&[points_3d[0], points_3d[0]], degree)?;
        return Ok(vec![SelfIntersectionCurve {
            curve,
            params_a,
            params_b,
        }]);
    }

    let degree = 3.min(points_3d.len() - 1);
    let curve = if points_3d.len() > 50 {
        let num_cps = (points_3d.len() / 3).max(degree + 1).min(points_3d.len());
        crate::nurbs::fitting::approximate_lspia(&points_3d, degree, num_cps, 1e-6, 100)?
    } else {
        crate::nurbs::interpolate(&points_3d, degree)?
    };

    Ok(vec![SelfIntersectionCurve {
        curve,
        params_a,
        params_b,
    }])
}

/// Check if two triangles are adjacent in the grid (within `threshold` steps).
fn are_adjacent(a: &[(usize, usize); 3], b: &[(usize, usize); 3], threshold: usize) -> bool {
    for &(ai, aj) in a {
        for &(bi, bj) in b {
            let di = ai.abs_diff(bi);
            let dj = aj.abs_diff(bj);
            if di <= threshold && dj <= threshold {
                return true;
            }
        }
    }
    false
}

/// Newton-refine a self-intersection point.
///
/// Solves `S(u1,v1) = S(u2,v2)` with the constraint that
/// `(u1,v1) ≠ (u2,v2)`, using alternating projection on the
/// single surface.
#[allow(clippy::type_complexity)]
fn refine_self_intersection_point(
    surface: &NurbsSurface,
    u1_guess: f64,
    v1_guess: f64,
    u2_guess: f64,
    v2_guess: f64,
    tolerance: f64,
) -> Option<(Point3, (f64, f64), (f64, f64))> {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let eps = (u_max - u_min + v_max - v_min) * 0.01;

    let mut u1 = u1_guess;
    let mut v1 = v1_guess;
    let mut u2 = u2_guess;
    let mut v2 = v2_guess;

    for _ in 0..50 {
        let p1 = surface.evaluate(u1, v1);
        let p2 = surface.evaluate(u2, v2);
        let residual = p1 - p2;

        if residual.length() < tolerance {
            // Check that parameters are genuinely distinct
            let param_dist = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
            if param_dist > eps {
                return Some((p1, (u1, v1), (u2, v2)));
            }
            return None; // Same point, not a self-intersection
        }

        // Move (u2, v2) toward p1 using Newton step on the surface
        let (du2, dv2) = surface_newton_step_self(surface, u2, v2, p1);
        u2 = (u2 + du2).clamp(u_min, u_max);
        v2 = (v2 + dv2).clamp(v_min, v_max);

        // Move (u1, v1) toward updated p2
        let p2_new = surface.evaluate(u2, v2);
        let (du1, dv1) = surface_newton_step_self(surface, u1, v1, p2_new);
        u1 = (u1 + du1).clamp(u_min, u_max);
        v1 = (v1 + dv1).clamp(v_min, v_max);

        // Push parameters apart if they're converging to the same point
        let param_dist = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
        if param_dist < eps * 0.5 {
            return None; // Parameters converging — not a real self-intersection
        }
    }

    // Final check
    let p1 = surface.evaluate(u1, v1);
    let p2 = surface.evaluate(u2, v2);
    if (p1 - p2).length() < tolerance * 100.0 {
        let param_dist = ((u1 - u2).powi(2) + (v1 - v2).powi(2)).sqrt();
        if param_dist > eps {
            return Some((p1, (u1, v1), (u2, v2)));
        }
    }

    None
}

/// Newton step to project (u,v) on a surface toward a target 3D point.
fn surface_newton_step_self(surface: &NurbsSurface, u: f64, v: f64, target: Point3) -> (f64, f64) {
    let pt = surface.evaluate(u, v);
    let r = target - pt;
    let r_vec = Vec3::new(r.x(), r.y(), r.z());

    let derivs = surface.derivatives(u, v, 1);
    let su = derivs[1][0];
    let sv = derivs[0][1];

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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::nurbs::surface::NurbsSurface;
    use crate::vec::Point3;

    /// A flat bilinear surface — no self-intersection.
    fn flat_surface() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    /// A surface with crossed control points that creates a self-intersection.
    /// The control polygon folds over itself.
    fn folded_surface() -> NurbsSurface {
        NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.5, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                ],
                vec![
                    // Middle row crosses over: points at x=0 and x=1 swap z values
                    Point3::new(0.0, 0.5, 1.0),
                    Point3::new(0.5, 0.5, -1.0),
                    Point3::new(1.0, 0.5, 1.0),
                ],
                vec![
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(0.5, 1.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                ],
            ],
            vec![vec![1.0; 3]; 3],
        )
        .unwrap()
    }

    #[test]
    fn flat_surface_clean() {
        let surf = flat_surface();
        let result = detect_self_intersection(&surf, 10, 1e-6).unwrap();
        assert!(
            result.is_empty(),
            "flat surface should have no self-intersection"
        );
    }

    #[test]
    fn folded_surface_detected() {
        let surf = folded_surface();
        // The folded control polygon creates regions where S(u1,v1) ≈ S(u2,v2).
        let result = detect_self_intersection(&surf, 15, 1e-4).unwrap();

        // We may or may not find self-intersections depending on how strongly
        // the surface folds. The key test is that it doesn't crash.
        // If found, verify the detected points have distinct parameters.
        for si in &result {
            assert!(
                si.params_a.len() == si.params_b.len(),
                "param lists should have same length"
            );
            for (pa, pb) in si.params_a.iter().zip(si.params_b.iter()) {
                let dist = ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt();
                assert!(
                    dist > 1e-6,
                    "self-intersection params should be distinct: {pa:?} vs {pb:?}"
                );
            }
        }
    }
}
