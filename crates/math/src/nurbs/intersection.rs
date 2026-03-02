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
}
