//! Edge sampling and surface normal utilities for fillet operations.

use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

/// Sample a point along an edge curve at normalised parameter `t` ∈ [0, 1].
///
/// For `Line` edges this is simple lerp between start/end.
/// For Circle/Ellipse/NurbsCurve the actual curve geometry is evaluated.
pub(super) fn sample_edge_point(
    curve: &EdgeCurve,
    p_start: Point3,
    p_end: Point3,
    t: f64,
) -> Point3 {
    match curve {
        EdgeCurve::Line => Point3::new(
            p_start.x().mul_add(1.0 - t, p_end.x() * t),
            p_start.y().mul_add(1.0 - t, p_end.y() * t),
            p_start.z().mul_add(1.0 - t, p_end.z() * t),
        ),
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            circle.evaluate(ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ellipse.evaluate(ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            nurbs.evaluate(u0 + (u1 - u0) * t)
        }
    }
}

/// Compute the tangent direction along an edge curve at normalised parameter `t`.
///
/// Returns an unnormalised tangent vector. For `Line` edges this is the constant
/// `p_end - p_start` direction.
pub(super) fn sample_edge_tangent(
    curve: &EdgeCurve,
    p_start: Point3,
    p_end: Point3,
    t: f64,
) -> Vec3 {
    match curve {
        EdgeCurve::Line => p_end - p_start,
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            circle.tangent(ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ellipse.tangent(ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            let u = u0 + (u1 - u0) * t;
            let d = nurbs.derivatives(u, 1);
            d[1]
        }
    }
}

/// Determine the number of v-direction samples needed for an edge curve.
///
/// Line edges need only 2 samples (start + end) for an exact linear surface.
/// Curved edges need more samples to capture the curvature.
pub(super) fn edge_v_samples(curve: &EdgeCurve) -> usize {
    match curve {
        EdgeCurve::Line => 2,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => 9,
        EdgeCurve::NurbsCurve(_) => 7,
    }
}

/// Compute the outward surface normal of a `FaceSurface` at a given 3D point.
///
/// For analytic surfaces this is exact (no parameter-space projection needed).
/// For NURBS surfaces, uses the midpoint normal as an approximation (full
/// point-inversion would be needed for exactness, but this suffices for fillet
/// cross-section geometry where the point is known to lie on the surface).
pub fn face_surface_normal_at(surface: &FaceSurface, point: Point3) -> Option<Vec3> {
    match surface {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Cylinder(cyl) => {
            // Project point onto cylinder axis to find closest axis point,
            // then the normal is the radial direction from axis to point.
            let dp = point - cyl.origin();
            let along_axis = dp.dot(cyl.axis());
            let on_axis = cyl.origin() + cyl.axis() * along_axis;
            (point - on_axis).normalize().ok()
        }
        FaceSurface::Cone(cone) => {
            // For a cone, the normal is perpendicular to the surface.
            // Project point onto axis, compute the radial direction,
            // then rotate by (90° - half_angle) around the tangent.
            let dp = point - cone.apex();
            let along_axis = dp.dot(cone.axis());
            let radial = dp - cone.axis() * along_axis;
            let radial_n = radial.normalize().ok()?;
            let (sin_a, cos_a) = cone.half_angle().sin_cos();
            // Normal = radial * sin(half_angle) - axis * cos(half_angle)
            Some(radial_n * sin_a + cone.axis() * (-cos_a))
        }
        FaceSurface::Sphere(sph) => (point - sph.center()).normalize().ok(),
        FaceSurface::Torus(tor) => {
            // Project point onto the major circle plane to find the closest
            // point on the major circle, then the normal is from the tube
            // center toward the point.
            let dp = point - tor.center();
            let along_axis = dp.dot(tor.z_axis());
            let in_plane = dp - tor.z_axis() * along_axis;
            let ring_dir = in_plane.normalize().ok()?;
            let tube_center = tor.center() + ring_dir * tor.major_radius();
            (point - tube_center).normalize().ok()
        }
        FaceSurface::Nurbs(srf) => {
            // Project the point onto the NURBS surface to find (u, v), then
            // evaluate the exact normal at that parameter.  Tolerance 1e-4 is
            // sufficient here — we only need (u,v) accurate enough for a
            // reliable normal direction, not for position reconstruction.
            // Use a projection tolerance derived from the standard linear
            // tolerance (×1000) — loose enough for normal-direction accuracy
            // without over-iterating, and scale-aware via Tolerance::new().
            let proj_tol = Tolerance::new().linear * 1e3;
            match project_point_to_surface(srf, point, proj_tol) {
                Ok(proj) => srf.normal(proj.u, proj.v).ok(),
                Err(_) => srf.normal(0.5, 0.5).ok(), // fallback to midpoint
            }
        }
    }
}
