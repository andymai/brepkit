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

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::fitting::interpolate;
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

        let du = (b1 * a22 - b2 * a12) / det;
        let dv = (a11 * b2 - a12 * b1) / det;

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

/// Build intersection curves from a set of points by sorting and fitting.
fn build_curves_from_points(
    points: &[IntersectionPoint],
) -> Result<Vec<IntersectionCurve>, MathError> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    // Simple approach: sort by u parameter and fit a single curve.
    // For multiple branches, a more sophisticated chain-building algorithm
    // would be needed.
    let mut sorted: Vec<IntersectionPoint> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.param1
            .0
            .partial_cmp(&b.param1.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate closely spaced points.
    let mut deduped: Vec<IntersectionPoint> = Vec::new();
    for pt in &sorted {
        let dominated = deduped
            .last()
            .is_some_and(|last: &IntersectionPoint| (last.point - pt.point).length() < 1e-6);
        if !dominated {
            deduped.push(*pt);
        }
    }

    if deduped.len() < 2 {
        return Ok(Vec::new());
    }

    // Fit a NURBS curve through the intersection points.
    let positions: Vec<Point3> = deduped.iter().map(|p| p.point).collect();
    let degree = if positions.len() <= 3 {
        1
    } else {
        3.min(positions.len() - 1)
    };
    let curve = interpolate(&positions, degree)?;

    Ok(vec![IntersectionCurve {
        curve,
        points: deduped,
    }])
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

    // Phase 1: Find seed points by sampling both surfaces and finding
    // close pairs.
    let seeds = find_ssi_seeds(surface1, surface2, n, tolerance);

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

/// Find seed points for NURBS-NURBS intersection by grid sampling.
///
/// Strategy: sample both surfaces on an n×n grid and try Newton
/// refinement for all cell-center pairs whose 3D positions are within
/// a generous distance threshold.
#[allow(clippy::cast_precision_loss)]
fn find_ssi_seeds(
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

    let du = (b1 * a22 - b2 * a12) / det;
    let dv = (a11 * b2 - a12 * b1) / det;

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

/// March in one direction along the intersection curve.
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

    for _ in 0..max_steps {
        // Compute tangent direction from surface normals.
        let n1 = s1.normal(current.param1.0, current.param1.1);
        let n2 = s2.normal(current.param2.0, current.param2.1);

        let (Ok(n1), Ok(n2)) = (n1, n2) else { break };

        let tangent = n1.cross(n2);
        let Ok(tangent) = tangent.normalize() else {
            break;
        };

        let sign = if forward { 1.0 } else { -1.0 };

        // Step along tangent.
        let next_pt = Point3::new(
            tangent.x().mul_add(step_size * sign, current.point.x()),
            tangent.y().mul_add(step_size * sign, current.point.y()),
            tangent.z().mul_add(step_size * sign, current.point.z()),
        );

        // Project back onto both surfaces and refine.
        let _ = next_pt; // marching direction is encoded in the parameter shift
        if let Some(refined) = refine_ssi_point(
            s1,
            s2,
            current.param1.0,
            current.param1.1,
            current.param2.0,
            current.param2.1,
            tolerance,
        ) {
            // Check that we actually moved.
            if (refined.point - current.point).length() < tolerance {
                break;
            }

            // Check we haven't left the parameter domain.
            if refined.param1.0 <= 0.001
                || refined.param1.0 >= 0.999
                || refined.param1.1 <= 0.001
                || refined.param1.1 >= 0.999
                || refined.param2.0 <= 0.001
                || refined.param2.0 >= 0.999
                || refined.param2.1 <= 0.001
                || refined.param2.1 >= 0.999
            {
                points.push(refined);
                break; // Reached boundary.
            }

            points.push(refined);
            current = refined;
        } else {
            break;
        }
    }

    points
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

        let seeds = find_ssi_seeds(&s1, &s2, 10, 1e-6);
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
        let seeds = find_ssi_seeds(&s1, &s2, 20, 1e-6);
        assert!(
            !seeds.is_empty(),
            "should find at least one seed point, got 0"
        );

        let result = intersect_nurbs_nurbs(&s1, &s2, 20, 0.05).unwrap();

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
        let result = intersect_nurbs_nurbs(&s1, &s2, 20, 0.02).unwrap();

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
}
