//! Line (ray) - NURBS surface intersection.

use crate::MathError;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::{IntersectionPoint, MAX_NEWTON_ITER};

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

    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    #[allow(clippy::cast_precision_loss)]
    let u_step = (u_max - u_min) / (n - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let v_step = (v_max - v_min) / (n - 1) as f64;

    // Sample surface points and compute distance to ray.
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new(); // (u, v, t_along_ray)

    let dir_len_sq = ray_dir.dot(ray_dir);
    if dir_len_sq < 1e-20 {
        return Err(MathError::ZeroVector);
    }

    // Compute an adaptive distance threshold based on the surface's 3D extent.
    // Use the diagonal of the surface bounding box divided by the grid resolution.
    let corner_00 = surface.evaluate(u_min, v_min);
    let corner_11 = surface.evaluate(u_max, v_max);
    #[allow(clippy::cast_precision_loss)]
    let candidate_threshold = ((corner_11 - corner_00).length() / n as f64).max(0.1);

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let u = u_min + i as f64 * u_step;
        for j in 0..n {
            let v = v_min + j as f64 * v_step;
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
            if dist < candidate_threshold {
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
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    for _ in 0..MAX_NEWTON_ITER {
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

        // Solve 2x2 system: [su*su, su*sv; sv*su, sv*sv] * [du, dv] = [su*r, sv*r]
        let a11 = su.dot(su);
        let a12 = su.dot(sv);
        let a22 = sv.dot(sv);
        let b1 = su.dot(r);
        let b2 = sv.dot(r);

        let det = a11.mul_add(a22, -(a12 * a12));
        // Relative singularity threshold -- catches surface poles/apex where
        // derivatives shrink to zero (making absolute 1e-20 too lenient).
        let (du, dv) = if det.abs() < (a11 + a22).max(1e-30) * 1e-12 {
            // Near-degenerate: Tikhonov regularization.
            let lambda = (a11 + a22).max(1e-10) * 1e-4;
            let a11r = a11 + lambda;
            let a22r = a22 + lambda;
            let det_r = a11r.mul_add(a22r, -(a12 * a12));
            if det_r.abs() < 1e-30 {
                // Truly singular -- step along the non-degenerate direction only.
                if a11 > a22 {
                    (b1 / a11.max(1e-30), 0.0)
                } else if a22 > 1e-30 {
                    (0.0, b2 / a22.max(1e-30))
                } else {
                    break;
                }
            } else {
                (
                    b1.mul_add(a22r, -(b2 * a12)) / det_r,
                    a11r.mul_add(b2, -(a12 * b1)) / det_r,
                )
            }
        } else {
            (
                b1.mul_add(a22, -(b2 * a12)) / det,
                a11.mul_add(b2, -(a12 * b1)) / det,
            )
        };

        u -= du;
        v -= dv;
        u = u.clamp(u_min, u_max);
        v = v.clamp(v_min, v_max);
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
