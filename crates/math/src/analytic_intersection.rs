//! Closed-form and semi-analytic intersections of analytic surfaces with planes.
//!
//! Provides specialized intersection algorithms for cylinder, cone, sphere,
//! and torus surfaces with planes, as well as a general marching approach
//! for analytic-analytic surface intersections.

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::MathError;
use crate::nurbs::fitting::interpolate;
use crate::nurbs::intersection::{IntersectionCurve, IntersectionPoint};
use crate::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface};
use crate::vec::{Point3, Vec3};

/// Reference to an analytic surface for intersection dispatch.
#[derive(Clone, Copy)]
pub enum AnalyticSurface<'a> {
    /// Cylindrical surface reference.
    Cylinder(&'a CylindricalSurface),
    /// Conical surface reference.
    Cone(&'a ConicalSurface),
    /// Spherical surface reference.
    Sphere(&'a SphericalSurface),
    /// Toroidal surface reference.
    Torus(&'a ToroidalSurface),
}

/// Compute `n . p` treating a `Point3` as a position vector.
fn dot_np(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// Intersect a plane with an analytic surface.
///
/// The plane is defined by `dot(normal, p) = d`.
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn intersect_plane_analytic(
    surface: AnalyticSurface<'_>,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => intersect_plane_cylinder(cyl, normal, d),
        AnalyticSurface::Cone(cone) => intersect_plane_cone(cone, normal, d),
        AnalyticSurface::Sphere(sphere) => intersect_plane_sphere(sphere, normal, d),
        AnalyticSurface::Torus(torus) => intersect_plane_torus(torus, normal, d),
    }
}

/// Sample points on the plane-analytic intersection without NURBS curve fitting.
///
/// Returns chains of ordered 3D sample points. Each chain is one connected
/// component of the intersection curve. This is much faster than
/// `intersect_plane_analytic` when only sample points are needed (e.g. for
/// boolean intersection segment generation).
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn sample_plane_analytic(
    surface: AnalyticSurface<'_>,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => sample_plane_cylinder(cyl, normal, d),
        AnalyticSurface::Cone(cone) => sample_plane_cone(cone, normal, d),
        AnalyticSurface::Sphere(sphere) => sample_plane_sphere(sphere, normal, d),
        AnalyticSurface::Torus(torus) => sample_plane_torus(torus, normal, d),
    }
}

/// Sample the plane-cylinder intersection as ordered 3D points.
#[allow(clippy::cast_precision_loss, clippy::unnecessary_wraps)]
fn sample_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let n_samples = 64_usize;
    let mut points = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        let base = cyl.evaluate(u, 0.0);
        let n_dot_axis = normal.dot(cyl.axis());
        let n_dot_base = dot_np(normal, base);

        if n_dot_axis.abs() < 1e-12 {
            if (n_dot_base - d).abs() < 1e-6 {
                points.push(base);
            }
        } else {
            let v = (d - n_dot_base) / n_dot_axis;
            if v.abs() <= 100.0 {
                points.push(cyl.evaluate(u, v));
            }
        }
    }

    if points.len() < 2 {
        Ok(vec![])
    } else {
        Ok(vec![points])
    }
}

/// Sample the plane-sphere intersection as ordered 3D points.
#[allow(clippy::cast_precision_loss)]
fn sample_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let h = dot_np(normal, sphere.center()) - d;
    let r = sphere.radius();

    if h.abs() > r - 1e-10 {
        return Ok(vec![]);
    }

    let circle_r = (r.mul_add(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.mul_add(-normal.x(), sphere.center().x()),
        h.mul_add(-normal.y(), sphere.center().y()),
        h.mul_add(-normal.z(), sphere.center().z()),
    );

    let candidate = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_dir = normal.cross(candidate).normalize()?;
    let v_dir = normal.cross(u_dir);

    let n_samples = 64_usize;
    let mut points = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let theta = TAU * (i as f64) / (n_samples as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        points.push(circle_center + u_dir * (circle_r * cos_t) + v_dir * (circle_r * sin_t));
    }

    Ok(vec![points])
}

/// Sample the plane-cone intersection as ordered 3D points.
#[allow(clippy::cast_precision_loss, clippy::unnecessary_wraps)]
fn sample_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let n_samples = 64_usize;
    let mut points = Vec::with_capacity(n_samples + 1);
    let half_angle = cone.half_angle();

    for i in 0..=n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        let base = cone.evaluate(u, 0.0);
        let dir_v = cone.evaluate(u, 1.0) - base;
        let n_dot_dir = normal.dot(Vec3::new(dir_v.x(), dir_v.y(), dir_v.z()));
        let n_dot_base = dot_np(normal, base);

        if n_dot_dir.abs() < 1e-12 {
            if (n_dot_base - d).abs() < 1e-6 * (1.0 + half_angle.tan().abs()) {
                points.push(base);
            }
        } else {
            let v = (d - n_dot_base) / n_dot_dir;
            if v.abs() <= 100.0 {
                points.push(base + dir_v * v);
            }
        }
    }

    if points.len() < 2 {
        Ok(vec![])
    } else {
        Ok(vec![points])
    }
}

/// Sample the plane-torus intersection as ordered 3D points.
///
/// Uses the same Newton-refined sampling grid as `intersect_plane_torus`
/// but skips NURBS curve fitting.
#[allow(clippy::cast_precision_loss)]
fn sample_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    // Delegate to the full version and extract just the points.
    let curves = intersect_plane_torus(torus, normal, d)?;
    Ok(curves
        .into_iter()
        .map(|c| c.points.into_iter().map(|p| p.point).collect())
        .collect())
}

/// Intersect a plane with a cylindrical surface.
///
/// For each `u` in `[0, 2pi)`, the cylinder point is linear in `v`,
/// so the plane equation `dot(normal, P(u,v)) = d` is linear in `v`
/// and can be solved directly.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..=n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        // P(u, v) = origin + r*(cos(u)*x + sin(u)*y) + v*axis
        // dot(normal, P) = d  =>  dot(normal, base(u)) + v * dot(normal, axis) = d
        let base = cyl.evaluate(u, 0.0);
        let n_dot_axis = normal.dot(cyl.axis());
        let n_dot_base = dot_np(normal, base);

        if n_dot_axis.abs() < 1e-12 {
            // Plane parallel to axis -- check if base is on plane.
            if (n_dot_base - d).abs() < 1e-6 {
                let pt = base;
                points_3d.push(pt);
                ipoints.push(IntersectionPoint {
                    point: pt,
                    param1: (u, 0.0),
                    param2: (0.0, 0.0),
                });
            }
        } else {
            let v = (d - n_dot_base) / n_dot_axis;
            // Only keep points within a reasonable v range.
            if v.abs() <= 100.0 {
                let pt = cyl.evaluate(u, v);
                points_3d.push(pt);
                ipoints.push(IntersectionPoint {
                    point: pt,
                    param1: (u, v),
                    param2: (0.0, 0.0),
                });
            }
        }
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a spherical surface.
///
/// The intersection of a plane with a sphere is a circle (or empty/point).
/// Computes the circle center, radius, and samples points on it.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let h = dot_np(normal, sphere.center()) - d;
    let r = sphere.radius();

    // No intersection if plane is too far from center.
    if h.abs() > r - 1e-10 {
        return Ok(vec![]);
    }

    let circle_r = (r.mul_add(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.mul_add(-normal.x(), sphere.center().x()),
        h.mul_add(-normal.y(), sphere.center().y()),
        h.mul_add(-normal.z(), sphere.center().z()),
    );

    // Build a local frame on the plane.
    let candidate = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_dir = normal.cross(candidate).normalize()?;
    let v_dir = normal.cross(u_dir);

    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..=n_samples {
        let theta = TAU * (i as f64) / (n_samples as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = circle_center + u_dir * (circle_r * cos_t) + v_dir * (circle_r * sin_t);
        points_3d.push(pt);
        ipoints.push(IntersectionPoint {
            point: pt,
            param1: (theta, 0.0),
            param2: (0.0, 0.0),
        });
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a conical surface.
///
/// Like a cylinder, the cone is linear along each generatrix, so the plane
/// equation is linear in `v` for each fixed `u`.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        // P(u, v) = apex + v * dir(u)
        // dot(normal, apex) + v * dot(normal, dir(u)) = d
        let apex = cone.apex();
        let n_dot_apex = dot_np(normal, apex);
        // dir(u) = P(u,1) - apex
        let p1 = cone.evaluate(u, 1.0);
        let dir = Vec3::new(p1.x() - apex.x(), p1.y() - apex.y(), p1.z() - apex.z());
        let n_dot_dir = normal.dot(dir);

        if n_dot_dir.abs() < 1e-12 {
            continue;
        }

        let v = (d - n_dot_apex) / n_dot_dir;
        if v > 0.0 && v < 100.0 {
            let pt = cone.evaluate(u, v);
            points_3d.push(pt);
            ipoints.push(IntersectionPoint {
                point: pt,
                param1: (u, v),
                param2: (0.0, 0.0),
            });
        }
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a toroidal surface.
///
/// The torus-plane intersection is a degree-4 curve with no simple closed form.
/// Uses grid sampling with sign-change detection and Newton refinement.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]
pub fn intersect_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_grid = 128_usize;

    // Signed distance to plane for a torus point.
    let sdf = |u: f64, v: f64| -> f64 { dot_np(normal, torus.evaluate(u, v)) - d };

    // Collect zero-crossing points by scanning edges of a (u,v) grid.
    let mut crossing_pts: Vec<(f64, f64, Point3)> = Vec::new();

    let du = TAU / (n_grid as f64);
    let dv = TAU / (n_grid as f64);

    for iu in 0..n_grid {
        for iv in 0..n_grid {
            let u0 = (iu as f64) * du;
            let v0 = (iv as f64) * dv;
            let u1 = u0 + du;
            let v1 = v0 + dv;

            let f00 = sdf(u0, v0);
            let f10 = sdf(u1, v0);
            let f01 = sdf(u0, v1);

            // Check horizontal edge (u0,v0)-(u1,v0).
            if f00 * f10 < 0.0 {
                let t = f00 / (f00 - f10);
                let u = t.mul_add(u1 - u0, u0);
                let (u_r, v_r) = newton_refine_torus(torus, normal, d, u, v0);
                crossing_pts.push((u_r, v_r, torus.evaluate(u_r, v_r)));
            }

            // Check vertical edge (u0,v0)-(u0,v1).
            if f00 * f01 < 0.0 {
                let t = f00 / (f00 - f01);
                let v = t.mul_add(v1 - v0, v0);
                let (u_r, v_r) = newton_refine_torus(torus, normal, d, u0, v);
                crossing_pts.push((u_r, v_r, torus.evaluate(u_r, v_r)));
            }
        }
    }

    if crossing_pts.is_empty() {
        return Ok(vec![]);
    }

    // Group nearby points into connected curves via greedy chaining.
    let mut used = vec![false; crossing_pts.len()];
    let mut curves = Vec::new();

    for start in 0..crossing_pts.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain = vec![start];

        loop {
            let last = chain[chain.len() - 1];
            let last_pt = crossing_pts[last].2;
            let mut best_idx = None;
            let mut best_dist = 1.0_f64;

            for (j, &is_used) in used.iter().enumerate() {
                if is_used {
                    continue;
                }
                let dist = (crossing_pts[j].2 - last_pt).length();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(j);
                }
            }

            if let Some(j) = best_idx {
                used[j] = true;
                chain.push(j);
            } else {
                break;
            }
        }

        if chain.len() >= 4 {
            let pts: Vec<Point3> = chain.iter().map(|&i| crossing_pts[i].2).collect();
            let ipts: Vec<IntersectionPoint> = chain
                .iter()
                .map(|&i| IntersectionPoint {
                    point: crossing_pts[i].2,
                    param1: (crossing_pts[i].0, crossing_pts[i].1),
                    param2: (0.0, 0.0),
                })
                .collect();

            if let Ok(curve) = interpolate(&pts, 3.min(pts.len() - 1)) {
                curves.push(IntersectionCurve {
                    curve,
                    points: ipts,
                });
            }
        }
    }

    Ok(curves)
}

/// Newton-refine a torus parameter to lie on the cutting plane.
fn newton_refine_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    mut u: f64,
    mut v: f64,
) -> (f64, f64) {
    let eps = 1e-6;
    for _ in 0..10 {
        let f = dot_np(normal, torus.evaluate(u, v)) - d;
        if f.abs() < 1e-12 {
            break;
        }
        // Numerical gradient via central differences.
        let fu = (dot_np(normal, torus.evaluate(u + eps, v))
            - dot_np(normal, torus.evaluate(u - eps, v)))
            / (2.0 * eps);
        let fv = (dot_np(normal, torus.evaluate(u, v + eps))
            - dot_np(normal, torus.evaluate(u, v - eps)))
            / (2.0 * eps);

        let grad_sq = fu.mul_add(fu, fv * fv);
        if grad_sq < 1e-20 {
            break;
        }
        let step = f / grad_sq;
        u -= step * fu;
        v -= step * fv;
    }
    (u, v)
}

/// Build intersection curves from a collection of ordered 3D points.
///
/// If there are enough points, fits a NURBS curve through them.
fn build_curves_from_points(
    points_3d: &[Point3],
    ipoints: Vec<IntersectionPoint>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    if points_3d.len() < 2 {
        return Ok(vec![]);
    }

    let degree = 3.min(points_3d.len() - 1);
    let curve = interpolate(points_3d, degree)?;
    Ok(vec![IntersectionCurve {
        curve,
        points: ipoints,
    }])
}

// -- Analytic-Analytic Intersection -------------------------------------------

/// Intersect two analytic surfaces using a general marching approach.
///
/// Seeds intersection points by sampling both parameter spaces on a grid,
/// then marches along the intersection curve using the cross product of
/// the two surface normals as the tangent direction.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::type_complexity
)]
pub fn intersect_analytic_analytic(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let (surf_a, norm_a, u_range_a, v_range_a) = surface_closures(&a);
    let (surf_b, norm_b, u_range_b, v_range_b) = surface_closures(&b);

    // Sample surface A on a grid and find closest points on surface B.
    let mut seeds: Vec<(Point3, (f64, f64), (f64, f64))> = Vec::new();
    let threshold = 1e-4;

    for ia in 0..grid_res {
        for ja in 0..grid_res {
            let ua =
                u_range_a.0 + (u_range_a.1 - u_range_a.0) * (ia as f64 + 0.5) / (grid_res as f64);
            let va =
                v_range_a.0 + (v_range_a.1 - v_range_a.0) * (ja as f64 + 0.5) / (grid_res as f64);

            let pa = surf_a(ua, va);

            let mut best_dist = f64::MAX;
            let mut best_ub = 0.0;
            let mut best_vb = 0.0;

            for ib in 0..grid_res {
                for jb in 0..grid_res {
                    let ub = u_range_b.0
                        + (u_range_b.1 - u_range_b.0) * (ib as f64 + 0.5) / (grid_res as f64);
                    let vb = v_range_b.0
                        + (v_range_b.1 - v_range_b.0) * (jb as f64 + 0.5) / (grid_res as f64);

                    let pb = surf_b(ub, vb);
                    let dist = (pa - pb).length();
                    if dist < best_dist {
                        best_dist = dist;
                        best_ub = ub;
                        best_vb = vb;
                    }
                }
            }

            if best_dist < threshold {
                seeds.push((pa, (ua, va), (best_ub, best_vb)));
            }
        }
    }

    if seeds.is_empty() {
        return Ok(vec![]);
    }

    // Deduplicate seeds.
    let mut unique_seeds = Vec::new();
    for seed in &seeds {
        let dominated = unique_seeds
            .iter()
            .any(|s: &(Point3, (f64, f64), (f64, f64))| (s.0 - seed.0).length() < threshold * 5.0);
        if !dominated {
            unique_seeds.push(*seed);
        }
    }

    // March from each seed.
    let mut curves = Vec::new();
    let mut used_seeds = vec![false; unique_seeds.len()];

    for si in 0..unique_seeds.len() {
        if used_seeds[si] {
            continue;
        }
        used_seeds[si] = true;

        let march_result = march_intersection(
            surf_a.as_ref(),
            norm_a.as_ref(),
            surf_b.as_ref(),
            norm_b.as_ref(),
            unique_seeds[si].0,
            u_range_a,
            v_range_a,
            u_range_b,
            v_range_b,
        );

        if march_result.len() >= 2 {
            for (sj, other) in unique_seeds.iter().enumerate() {
                if !used_seeds[sj]
                    && march_result
                        .iter()
                        .any(|p| (*p - other.0).length() < threshold * 10.0)
                {
                    used_seeds[sj] = true;
                }
            }

            let ipts: Vec<IntersectionPoint> = march_result
                .iter()
                .map(|&pt| IntersectionPoint {
                    point: pt,
                    param1: (0.0, 0.0),
                    param2: (0.0, 0.0),
                })
                .collect();

            let degree = 3.min(march_result.len() - 1);
            if let Ok(curve) = interpolate(&march_result, degree) {
                curves.push(IntersectionCurve {
                    curve,
                    points: ipts,
                });
            }
        }
    }

    Ok(curves)
}

/// March along the intersection of two surfaces from a seed point.
///
/// Uses the cross product of surface normals as the tangent direction
/// and projects back onto both surfaces via Newton iteration.
#[allow(clippy::too_many_arguments)]
fn march_intersection(
    surf_a: &dyn Fn(f64, f64) -> Point3,
    norm_a: &dyn Fn(f64, f64) -> Vec3,
    surf_b: &dyn Fn(f64, f64) -> Point3,
    norm_b: &dyn Fn(f64, f64) -> Vec3,
    seed: Point3,
    u_range_a: (f64, f64),
    v_range_a: (f64, f64),
    u_range_b: (f64, f64),
    v_range_b: (f64, f64),
) -> Vec<Point3> {
    let step_size: f64 = 0.02;
    let max_steps = 500;
    let mut result = vec![seed];

    // March in both directions.
    for direction in &[1.0_f64, -1.0] {
        let mut current = seed;
        for _ in 0..max_steps {
            let (ua, va) = project_to_surface(surf_a, current, u_range_a, v_range_a, 16);
            let (ub, vb) = project_to_surface(surf_b, current, u_range_b, v_range_b, 16);

            let na = norm_a(ua, va);
            let nb = norm_b(ub, vb);

            let tangent = na.cross(nb);
            let t_len = tangent.length();
            if t_len < 1e-10 {
                break;
            }
            let t_dir = tangent * (direction / t_len);

            let next = Point3::new(
                step_size.mul_add(t_dir.x(), current.x()),
                step_size.mul_add(t_dir.y(), current.y()),
                step_size.mul_add(t_dir.z(), current.z()),
            );

            let (ua2, va2) = project_to_surface(surf_a, next, u_range_a, v_range_a, 16);
            let (ub2, vb2) = project_to_surface(surf_b, next, u_range_b, v_range_b, 16);

            let pa = surf_a(ua2, va2);
            let pb = surf_b(ub2, vb2);
            let mid = Point3::new(
                (pa.x() + pb.x()) * 0.5,
                (pa.y() + pb.y()) * 0.5,
                (pa.z() + pb.z()) * 0.5,
            );

            let out_a = ua2 <= u_range_a.0
                || ua2 >= u_range_a.1
                || va2 <= v_range_a.0
                || va2 >= v_range_a.1;
            let out_b = ub2 <= u_range_b.0
                || ub2 >= u_range_b.1
                || vb2 <= v_range_b.0
                || vb2 >= v_range_b.1;

            if out_a || out_b {
                break;
            }

            // Check for loop closure.
            if result.len() > 3 && (mid - result[0]).length() < step_size * 2.0 {
                result.push(result[0]);
                break;
            }

            if *direction > 0.0 {
                result.push(mid);
            } else {
                result.insert(0, mid);
            }
            current = mid;
        }
    }

    result
}

/// Project a 3D point onto a parametric surface by grid search.
#[allow(clippy::cast_precision_loss)]
fn project_to_surface(
    surface_fn: &dyn Fn(f64, f64) -> Point3,
    point: Point3,
    u_range: (f64, f64),
    v_range: (f64, f64),
    res: usize,
) -> (f64, f64) {
    let mut best_u = u_range.0;
    let mut best_v = v_range.0;
    let mut best_dist = f64::MAX;

    for i in 0..=res {
        let u = u_range.0 + (u_range.1 - u_range.0) * (i as f64) / (res as f64);
        for j in 0..=res {
            let v = v_range.0 + (v_range.1 - v_range.0) * (j as f64) / (res as f64);
            let p = surface_fn(u, v);
            let dist = (p - point).length_squared();
            if dist < best_dist {
                best_dist = dist;
                best_u = u;
                best_v = v;
            }
        }
    }

    (best_u, best_v)
}

/// Extract closures and parameter ranges for an analytic surface.
#[allow(clippy::type_complexity)]
fn surface_closures<'a>(
    surface: &'a AnalyticSurface<'a>,
) -> (
    Box<dyn Fn(f64, f64) -> Point3 + 'a>,
    Box<dyn Fn(f64, f64) -> Vec3 + 'a>,
    (f64, f64),
    (f64, f64),
) {
    match surface {
        AnalyticSurface::Cylinder(cyl) => (
            Box::new(|u, v| cyl.evaluate(u, v)),
            Box::new(|u, v| cyl.normal(u, v)),
            (0.0, TAU),
            (-1.0, 1.0),
        ),
        AnalyticSurface::Cone(cone) => (
            Box::new(|u, v| cone.evaluate(u, v)),
            Box::new(|u, v| cone.normal(u, v)),
            (0.0, TAU),
            (0.01, 2.0),
        ),
        AnalyticSurface::Sphere(sphere) => (
            Box::new(|u, v| sphere.evaluate(u, v)),
            Box::new(|u, v| sphere.normal(u, v)),
            (0.0, TAU),
            (-FRAC_PI_2, FRAC_PI_2),
        ),
        AnalyticSurface::Torus(torus) => (
            Box::new(|u, v| torus.evaluate(u, v)),
            Box::new(|u, v| torus.normal(u, v)),
            (0.0, TAU),
            (0.0, TAU),
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tolerance::Tolerance;

    #[test]
    fn plane_cylinder_perpendicular() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();

        // Horizontal plane at z=3 -- produces a circle at height 3.
        let curves = intersect_plane_cylinder(&cyl, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        assert!(!curves.is_empty(), "should find intersection curve");
        assert!(
            curves[0].points.len() > 10,
            "should have many sample points"
        );

        let tol = Tolerance::loose();
        for pt in &curves[0].points {
            assert!(
                tol.approx_eq(pt.point.z(), 3.0),
                "z should be ~3.0, got {}",
                pt.point.z()
            );
            let r = pt.point.x().hypot(pt.point.y());
            assert!(tol.approx_eq(r, 2.0), "radius should be ~2.0, got {r}");
        }
    }

    #[test]
    fn plane_sphere_equator() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();

        let curves = intersect_plane_sphere(&sphere, Vec3::new(0.0, 0.0, 1.0), 0.0).unwrap();
        assert!(!curves.is_empty());

        let tol = Tolerance::loose();
        for pt in &curves[0].points {
            assert!(
                tol.approx_eq(pt.point.z(), 0.0),
                "z should be ~0, got {}",
                pt.point.z()
            );
            let r = pt.point.x().hypot(pt.point.y());
            assert!(tol.approx_eq(r, 3.0), "radius should be ~3.0, got {r}");
        }
    }

    #[test]
    fn plane_sphere_no_intersection() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

        let curves = intersect_plane_sphere(&sphere, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        assert!(curves.is_empty());
    }

    #[test]
    fn plane_cone_cross_section() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();

        let curves = intersect_plane_cone(&cone, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        assert!(!curves.is_empty(), "should find intersection with cone");
    }

    #[test]
    fn plane_torus_cross_section() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0).unwrap();

        let curves = intersect_plane_torus(&torus, Vec3::new(0.0, 0.0, 1.0), 0.0).unwrap();
        assert!(
            !curves.is_empty(),
            "should find intersection curves with torus"
        );
    }

    #[test]
    fn dispatch_via_analytic_surface() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let curves = intersect_plane_analytic(
            AnalyticSurface::Cylinder(&cyl),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        )
        .unwrap();
        assert!(!curves.is_empty());
    }
}
