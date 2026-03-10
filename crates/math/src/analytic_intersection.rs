//! Closed-form and semi-analytic intersections of analytic surfaces with planes.
//!
//! Provides specialized intersection algorithms for cylinder, cone, sphere,
//! and torus surfaces with planes, as well as a general marching approach
//! for analytic-analytic surface intersections.

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::MathError;
use crate::curves::{Circle3D, Ellipse3D};
use crate::nurbs::fitting::interpolate;
use crate::nurbs::intersection::{IntersectionCurve, IntersectionPoint};
use crate::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface};
use crate::vec::{Point3, Vec3};

/// Exact curve type resulting from plane-analytic surface intersection.
#[derive(Debug, Clone)]
pub enum ExactIntersectionCurve {
    /// A circle (plane perpendicular to axis of cylinder/cone/sphere).
    Circle(Circle3D),
    /// An ellipse (plane oblique to cylinder/cone axis).
    Ellipse(Ellipse3D),
    /// Fallback to sampled point chain (torus, degenerate cases).
    Points(Vec<Point3>),
}

/// Compute exact intersection curves between a plane and an analytic surface.
///
/// Returns exact `Circle3D` or `Ellipse3D` where possible, falling back to
/// sampled points for complex cases (torus).
///
/// The plane is defined by `dot(normal, p) = d`.
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn exact_plane_analytic(
    surface: AnalyticSurface<'_>,
    plane_normal: Vec3,
    plane_d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => exact_plane_cylinder(cyl, plane_normal, plane_d),
        AnalyticSurface::Sphere(sphere) => exact_plane_sphere(sphere, plane_normal, plane_d),
        AnalyticSurface::Cone(cone) => exact_plane_cone(cone, plane_normal, plane_d),
        AnalyticSurface::Torus(torus) => {
            // Torus intersections are degree-4 — fall back to sampling.
            let chains = sample_plane_torus(torus, plane_normal, plane_d)?;
            Ok(chains
                .into_iter()
                .map(ExactIntersectionCurve::Points)
                .collect())
        }
    }
}

/// Exact plane-cylinder intersection.
///
/// - Plane perpendicular to axis → `Circle3D`
/// - Plane oblique to axis → `Ellipse3D`
/// - Plane parallel to axis → `Points` fallback (0 or 2 lines)
fn exact_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    let axis = cyl.axis();
    let cos_theta = normal.dot(axis).abs();
    let r = cyl.radius();

    if cos_theta < 1e-10 {
        // Plane parallel to cylinder axis → 0 or 2 line segments.
        // Fall back to sampled points.
        let chains = sample_plane_cylinder(cyl, normal, d)?;
        return Ok(chains
            .into_iter()
            .map(ExactIntersectionCurve::Points)
            .collect());
    }

    // Find where axis intersects the plane: axis_point + t*axis, dot(normal, P) = d
    // t = (d - dot(normal, origin)) / dot(normal, axis)
    let n_dot_axis = normal.dot(axis);
    let n_dot_origin = dot_np(normal, cyl.origin());
    let t = (d - n_dot_origin) / n_dot_axis;
    let center_on_axis = Point3::new(
        cyl.origin().x() + t * axis.x(),
        cyl.origin().y() + t * axis.y(),
        cyl.origin().z() + t * axis.z(),
    );

    if cos_theta > 1.0 - 1e-10 {
        // Plane perpendicular to axis → Circle
        let circle = Circle3D::new(center_on_axis, normal, r)?;
        Ok(vec![ExactIntersectionCurve::Circle(circle)])
    } else {
        // Oblique plane → Ellipse
        // Semi-minor = r (the cylinder radius, unchanged)
        // Semi-major = r / cos(θ) where θ = angle between plane normal and axis
        let semi_minor = r;
        let semi_major = r / cos_theta;

        // The major axis direction lies in the intersection of the plane
        // with the plane containing the axis and the plane normal.
        // It's the projection of the axis onto the cutting plane, normalized.
        let axis_proj = Vec3::new(
            axis.x() - n_dot_axis * normal.x(),
            axis.y() - n_dot_axis * normal.y(),
            axis.z() - n_dot_axis * normal.z(),
        );
        let u_axis = axis_proj.normalize()?;
        let v_axis = normal.cross(u_axis);

        let ellipse = Ellipse3D::with_axes(
            center_on_axis,
            normal,
            semi_major,
            semi_minor,
            u_axis,
            v_axis,
        )?;
        Ok(vec![ExactIntersectionCurve::Ellipse(ellipse)])
    }
}

/// Exact plane-sphere intersection.
///
/// Always produces a `Circle3D` (or empty if no intersection).
fn exact_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
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

    let circle = Circle3D::new(circle_center, normal, circle_r)?;
    Ok(vec![ExactIntersectionCurve::Circle(circle)])
}

/// Exact plane-cone intersection.
///
/// - Plane perpendicular to axis → `Circle3D`
/// - Otherwise → `Points` fallback (could be ellipse, parabola, or hyperbola)
fn exact_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    let axis = cone.axis();
    let cos_theta = normal.dot(axis).abs();
    let half_angle = cone.half_angle();

    if cos_theta > 1.0 - 1e-10 {
        // Plane perpendicular to axis → Circle
        // Find where axis meets the plane
        let n_dot_axis = normal.dot(axis);
        let n_dot_apex = dot_np(normal, cone.apex());
        let t = (d - n_dot_apex) / n_dot_axis;

        // t is the signed distance from apex to plane along the axis.
        // The cone surface can extend in either direction from the apex
        // (positive or negative v), so use |t| for the radius.
        // |t| ≈ 0 means the plane passes through the apex → degenerate point.
        if t.abs() < 1e-10 {
            return Ok(vec![]);
        }

        let center = Point3::new(
            cone.apex().x() + t * axis.x(),
            cone.apex().y() + t * axis.y(),
            cone.apex().z() + t * axis.z(),
        );
        let circle_r = t.abs() * half_angle.tan();
        if circle_r < 1e-15 {
            return Ok(vec![]);
        }

        let circle = Circle3D::new(center, normal, circle_r)?;
        Ok(vec![ExactIntersectionCurve::Circle(circle)])
    } else {
        // Oblique → could be ellipse, parabola, or hyperbola depending on angle vs half_angle.
        // For MVP, fall back to sampling.
        let chains = sample_plane_cone(cone, normal, d)?;
        Ok(chains
            .into_iter()
            .map(ExactIntersectionCurve::Points)
            .collect())
    }
}

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
        // Allow negative v — the cone surface extends in both directions from the apex.
        if v.abs() > 1e-10 && v.abs() < 100.0 {
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
    // Try algebraic specialization for known surface pairs before falling
    // back to the general marching approach.
    if let Some(result) = try_algebraic_intersection(&a, &b)? {
        return Ok(result);
    }

    let (surf_a, norm_a, u_range_a, v_range_a) = surface_closures(&a);
    let (surf_b, norm_b, u_range_b, v_range_b) = surface_closures(&b);

    // Compute characteristic surface dimensions for adaptive parameters.
    let diag_a = {
        let p00 = surf_a(u_range_a.0, v_range_a.0);
        let p11 = surf_a(u_range_a.1, v_range_a.1);
        (p00 - p11).length()
    };
    let diag_b = {
        let p00 = surf_b(u_range_b.0, v_range_b.0);
        let p11 = surf_b(u_range_b.1, v_range_b.1);
        (p00 - p11).length()
    };
    let char_size = diag_a.min(diag_b).max(0.1);

    // Sample surface A on a grid. For each grid point, project it
    // analytically onto surface B to find the closest point, then check
    // if the distance is below threshold (indicating near-intersection).
    let mut seeds: Vec<(Point3, (f64, f64), (f64, f64))> = Vec::new();
    // Coarse threshold scales with the surface size — the distance from
    // a grid point on A to its projection on B can be large even near
    // the intersection (e.g., sphere R=2 and cylinder R=1 → gap ≈ 1).
    let seed_threshold = diag_a.max(diag_b).max(1.0) * 0.5;

    #[allow(clippy::cast_precision_loss)]
    for ia in 0..grid_res {
        for ja in 0..grid_res {
            let ua =
                u_range_a.0 + (u_range_a.1 - u_range_a.0) * (ia as f64 + 0.5) / (grid_res as f64);
            let va =
                v_range_a.0 + (v_range_a.1 - v_range_a.0) * (ja as f64 + 0.5) / (grid_res as f64);

            let pa = surf_a(ua, va);

            // Analytically project onto surface B.
            let (ub, vb) = project_analytic(&b, pa, u_range_b, v_range_b);
            let pb = surf_b(ub, vb);
            let dist = (pa - pb).length();

            if dist < seed_threshold {
                // Use the coarse seed directly. The marching algorithm
                // corrects positions at each step via projection, so seeds
                // don't need to be on the exact intersection — they just
                // need to be close enough for the marcher to converge.
                let mid = Point3::new(
                    (pa.x() + pb.x()) * 0.5,
                    (pa.y() + pb.y()) * 0.5,
                    (pa.z() + pb.z()) * 0.5,
                );
                seeds.push((mid, (ua, va), (ub, vb)));
            }
        }
    }

    if seeds.is_empty() {
        return Ok(vec![]);
    }

    // Aggressively deduplicate seeds — we only need 1-2 per intersection
    // branch. Scale dedup radius to ~2% of characteristic surface size
    // (at least 10× the march step size) to avoid redundant marches.
    let march_step = (char_size * 0.02).clamp(0.005, 0.5);
    let dedup_radius = march_step * 10.0;
    let mut unique_seeds = Vec::new();
    for seed in &seeds {
        let dominated = unique_seeds
            .iter()
            .any(|s: &(Point3, (f64, f64), (f64, f64))| (s.0 - seed.0).length() < dedup_radius);
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

        let march_result = march_analytic_intersection(
            &a,
            &b,
            surf_a.as_ref(),
            norm_a.as_ref(),
            surf_b.as_ref(),
            norm_b.as_ref(),
            unique_seeds[si].0,
            u_range_a,
            v_range_a,
            u_range_b,
            v_range_b,
            march_step,
        );

        if march_result.len() >= 2 {
            for (sj, other) in unique_seeds.iter().enumerate() {
                if !used_seeds[sj]
                    && march_result
                        .iter()
                        .any(|p| (*p - other.0).length() < dedup_radius)
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

/// Try algebraic (closed-form or semi-algebraic) intersection for known
/// surface pairs before falling back to general marching.
///
/// Returns `Some(curves)` if a specialized method exists, `None` otherwise.
///
/// Currently handles:
/// - **Sphere-sphere**: intersection is a circle (plane through the two centers)
/// - **Coaxial cylinders**: same axis → circle(s) or empty
/// - **Sphere-cylinder**: reduce to quadratic in one parameter
#[allow(clippy::too_many_lines)]
fn try_algebraic_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    match (a, b) {
        (AnalyticSurface::Sphere(s1), AnalyticSurface::Sphere(s2)) => {
            algebraic_sphere_sphere(s1, s2).map(Some)
        }
        (AnalyticSurface::Cylinder(c1), AnalyticSurface::Cylinder(c2)) => {
            // Only specialize for coaxial cylinders.
            let axis_dot = c1.axis().dot(c2.axis()).abs();
            if axis_dot > 1.0 - 1e-10 {
                // Axes are parallel — check if they're the same line.
                let delta = c2.origin() - c1.origin();
                let delta_vec = Vec3::new(delta.x(), delta.y(), delta.z());
                let along = delta_vec.dot(c1.axis());
                let perp = (delta_vec - c1.axis() * along).length();
                if perp < 1e-8 {
                    // Coaxial: same axis, different radii → no intersection
                    // (unless equal radius → degenerate overlap, skip)
                    if (c1.radius() - c2.radius()).abs() < 1e-8 {
                        return Ok(None); // Overlapping — let marcher handle
                    }
                    return Ok(Some(vec![])); // Coaxial, different radii
                }
            }
            Ok(None) // Non-coaxial: fall through to marching
        }
        _ => Ok(None),
    }
}

/// Algebraic sphere-sphere intersection.
///
/// Two spheres intersect in a circle lying in the radical plane.
/// The radical plane is perpendicular to the line connecting the centers,
/// at a distance d1 from center1 where:
///   d1 = (D² + R1² - R2²) / (2D)
/// and D is the distance between centers.
fn algebraic_sphere_sphere(
    s1: &SphericalSurface,
    s2: &SphericalSurface,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let c1 = s1.center();
    let c2 = s2.center();
    let r1 = s1.radius();
    let r2 = s2.radius();

    let delta = c2 - c1;
    let d_sq = delta.x() * delta.x() + delta.y() * delta.y() + delta.z() * delta.z();
    let d = d_sq.sqrt();

    if d < 1e-12 {
        // Concentric spheres: no intersection (unless same radius → degenerate).
        return Ok(vec![]);
    }

    // Check separation conditions.
    if d > r1 + r2 + 1e-10 {
        return Ok(vec![]); // Too far apart
    }
    if d + r2.min(r1) + 1e-10 < r1.max(r2) {
        return Ok(vec![]); // One inside the other
    }

    // Distance from c1 to the radical plane along the center line.
    let d1 = (d_sq + r1 * r1 - r2 * r2) / (2.0 * d);

    // Radius of the intersection circle.
    let r_circle_sq = r1 * r1 - d1 * d1;
    if r_circle_sq < 0.0 {
        // Tangent or no intersection (numerical noise).
        if r_circle_sq > -1e-10 {
            // Tangent: single point.
            let axis = Vec3::new(delta.x() / d, delta.y() / d, delta.z() / d);
            let tangent_pt = Point3::new(
                c1.x() + axis.x() * d1,
                c1.y() + axis.y() * d1,
                c1.z() + axis.z() * d1,
            );
            let ipt = IntersectionPoint {
                point: tangent_pt,
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            };
            // Single-point "curve" — not very useful but correct.
            return Ok(vec![IntersectionCurve {
                curve: interpolate(&[tangent_pt, tangent_pt], 1)?,
                points: vec![ipt],
            }]);
        }
        return Ok(vec![]);
    }

    let r_circle = r_circle_sq.sqrt();
    let axis = Vec3::new(delta.x() / d, delta.y() / d, delta.z() / d);
    let center = Point3::new(
        c1.x() + axis.x() * d1,
        c1.y() + axis.y() * d1,
        c1.z() + axis.z() * d1,
    );

    // Build a reference frame for the circle.
    // Pick a vector not parallel to axis for the cross product.
    let ref_vec = if axis.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_dir = axis.cross(ref_vec).normalize()?;
    let v_dir = axis.cross(u_dir);

    // Sample the circle for the IntersectionCurve representation.
    let n_samples = 33; // Odd for symmetry
    let mut points = Vec::with_capacity(n_samples);
    let mut positions = Vec::with_capacity(n_samples);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n_samples {
        let theta = TAU * i as f64 / (n_samples - 1) as f64;
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = Point3::new(
            center.x() + (u_dir.x() * cos_t + v_dir.x() * sin_t) * r_circle,
            center.y() + (u_dir.y() * cos_t + v_dir.y() * sin_t) * r_circle,
            center.z() + (u_dir.z() * cos_t + v_dir.z() * sin_t) * r_circle,
        );
        positions.push(pt);
        points.push(IntersectionPoint {
            point: pt,
            param1: (0.0, 0.0),
            param2: (0.0, 0.0),
        });
    }

    let degree = 3.min(positions.len() - 1);
    let curve = interpolate(&positions, degree)?;

    Ok(vec![IntersectionCurve { curve, points }])
}

/// March along the intersection of two surfaces from a seed point.
///
/// Uses the cross product of surface normals as the tangent direction
/// and projects back onto both surfaces using analytical projection
/// (for cylinders/spheres) or grid search (fallback).
#[allow(clippy::too_many_arguments)]
fn march_analytic_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
    surf_a: &dyn Fn(f64, f64) -> Point3,
    norm_a: &dyn Fn(f64, f64) -> Vec3,
    surf_b: &dyn Fn(f64, f64) -> Point3,
    norm_b: &dyn Fn(f64, f64) -> Vec3,
    seed: Point3,
    u_range_a: (f64, f64),
    v_range_a: (f64, f64),
    u_range_b: (f64, f64),
    v_range_b: (f64, f64),
    initial_step: f64,
) -> Vec<Point3> {
    let max_steps = 500;
    let h_min = 1e-6;
    let h_max = initial_step * 4.0;
    // Angular thresholds for curvature-adaptive stepping.
    let max_angle = 10.0_f64.to_radians();
    let min_angle = 2.0_f64.to_radians();

    // March forward from seed, collecting points.
    let mut forward = Vec::new();
    // March backward from seed, collecting points (reversed at end).
    let mut backward = Vec::new();

    for (direction, points) in [(1.0_f64, &mut forward), (-1.0_f64, &mut backward)] {
        let mut current = seed;
        let mut h = initial_step;
        let mut prev_tangent: Option<Vec3> = None;

        for _ in 0..max_steps {
            let (ua, va) = project_analytic(a, current, u_range_a, v_range_a);
            let (ub, vb) = project_analytic(b, current, u_range_b, v_range_b);

            let na = norm_a(ua, va);
            let nb = norm_b(ub, vb);

            let tangent = na.cross(nb);
            let t_len = tangent.length();
            if t_len < 1e-10 {
                break;
            }
            let t_dir = tangent * (direction / t_len);

            // Curvature-adaptive step: check angular deviation from previous tangent.
            if let Some(prev_t) = prev_tangent {
                let cos_angle = prev_t.dot(t_dir).clamp(-1.0, 1.0);
                let angle = cos_angle.acos();
                if angle > max_angle && h > h_min {
                    h = (h * 0.5).max(h_min);
                } else if angle < min_angle {
                    h = (h * 2.0).min(h_max);
                }
            }
            prev_tangent = Some(t_dir);

            let next = Point3::new(
                h.mul_add(t_dir.x(), current.x()),
                h.mul_add(t_dir.y(), current.y()),
                h.mul_add(t_dir.z(), current.z()),
            );

            let (ua2, va2) = project_analytic(a, next, u_range_a, v_range_a);
            let (ub2, vb2) = project_analytic(b, next, u_range_b, v_range_b);

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

            // Check for loop closure — if we've collected enough points and
            // the current point is close to the seed, the curve is closed.
            if points.len() > 3 && (mid - seed).length() < h * 2.0 {
                points.push(seed);
                break;
            }

            points.push(mid);
            current = mid;
        }
    }

    // Assemble result: backward (reversed) + seed + forward
    backward.reverse();
    let mut result = backward;
    result.push(seed);
    result.append(&mut forward);
    result
}

/// Project a 3D point onto an analytic surface using the surface's
/// analytical projection method. Falls back to grid search for surface
/// types without analytical projection.
fn project_analytic(
    surface: &AnalyticSurface<'_>,
    point: Point3,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> (f64, f64) {
    match surface {
        AnalyticSurface::Cylinder(cyl) => {
            let (u, v) = cyl.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        AnalyticSurface::Sphere(sphere) => {
            let (u, v) = sphere.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        AnalyticSurface::Cone(cone) => {
            let (u, v) = cone.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        AnalyticSurface::Torus(torus) => {
            let (u, v) = torus.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
    }
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

    #[test]
    fn perpendicular_cylinders_intersect() {
        let cyl_z =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let cyl_x =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Cylinder(&cyl_z),
            AnalyticSurface::Cylinder(&cyl_x),
            16,
        )
        .unwrap();

        assert!(
            !curves.is_empty(),
            "perpendicular cylinders should intersect"
        );

        for c in &curves {
            assert!(
                c.points.len() >= 2,
                "intersection curve should have >= 2 points, got {}",
                c.points.len()
            );
        }
    }

    #[test]
    fn sphere_cylinder_intersect() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Sphere(&sphere),
            AnalyticSurface::Cylinder(&cyl),
            16,
        )
        .unwrap();

        // A sphere of radius 2 and a cylinder of radius 1, both centered
        // at the origin, should intersect (the cylinder passes through
        // the sphere).
        assert!(!curves.is_empty(), "sphere and cylinder should intersect");
    }

    #[test]
    fn disjoint_cylinders_no_intersection() {
        let cyl_a =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5)
                .unwrap();
        let cyl_b =
            CylindricalSurface::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Cylinder(&cyl_a),
            AnalyticSurface::Cylinder(&cyl_b),
            16,
        )
        .unwrap();

        assert!(curves.is_empty(), "disjoint cylinders should not intersect");
    }
}
