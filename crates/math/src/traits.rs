//! Parametric geometry traits for unified curve and surface evaluation.
//!
//! These traits provide a common interface for evaluating both analytic
//! geometry types (circles, cylinders, etc.) and NURBS representations.

use crate::curves::{Circle3D, Ellipse3D};
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::projection::project_point_to_surface;
use crate::nurbs::surface::NurbsSurface;
use crate::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface};
use crate::vec::{Point3, Vec3};

/// Unified interface for parametric surface evaluation.
///
/// Implemented by analytic surfaces ([`CylindricalSurface`], [`ConicalSurface`],
/// [`SphericalSurface`], [`ToroidalSurface`]) and [`NurbsSurface`].
pub trait ParametricSurface {
    /// Evaluate the surface at parameters `(u, v)`.
    fn evaluate(&self, u: f64, v: f64) -> Point3;

    /// Surface normal at parameters `(u, v)`.
    ///
    /// Returns the unit normal. For NURBS surfaces at degenerate points,
    /// implementations should return a best-effort fallback (e.g. `Vec3::Z`).
    fn normal(&self, u: f64, v: f64) -> Vec3;

    /// Project a 3D point onto the surface, returning `(u, v)` parameters.
    fn project_point(&self, point: Point3) -> (f64, f64);

    /// Partial derivative ∂S/∂u at (u, v).
    fn partial_u(&self, u: f64, v: f64) -> Vec3;

    /// Partial derivative ∂S/∂v at (u, v).
    fn partial_v(&self, u: f64, v: f64) -> Vec3;
}

/// Unified interface for parametric curve evaluation.
///
/// Implemented by analytic curves ([`Circle3D`], [`Ellipse3D`]) and [`NurbsCurve`].
pub trait ParametricCurve {
    /// Evaluate the curve at parameter `t`.
    fn evaluate(&self, t: f64) -> Point3;

    /// Tangent vector at parameter `t`.
    fn tangent(&self, t: f64) -> Vec3;

    /// Parameter domain as `(t_min, t_max)`.
    fn domain(&self) -> (f64, f64);
}

// ── ParametricSurface implementations ────────────────────────────────

impl ParametricSurface for CylindricalSurface {
    #[inline]
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate(u, v)
    }

    #[inline]
    fn normal(&self, u: f64, v: f64) -> Vec3 {
        self.normal(u, v)
    }

    #[inline]
    fn project_point(&self, point: Point3) -> (f64, f64) {
        self.project_point(point)
    }

    #[inline]
    fn partial_u(&self, u: f64, _v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        self.x_axis() * (-self.radius() * sin_u) + self.y_axis() * (self.radius() * cos_u)
    }

    #[inline]
    fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
        self.axis()
    }
}

impl ParametricSurface for ConicalSurface {
    #[inline]
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate(u, v)
    }

    #[inline]
    fn normal(&self, u: f64, v: f64) -> Vec3 {
        self.normal(u, v)
    }

    #[inline]
    fn project_point(&self, point: Point3) -> (f64, f64) {
        self.project_point(point)
    }

    #[inline]
    fn partial_u(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let cos_a = self.half_angle().cos();
        self.x_axis() * (-v * cos_a * sin_u) + self.y_axis() * (v * cos_a * cos_u)
    }

    #[inline]
    fn partial_v(&self, u: f64, _v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_a, cos_a) = self.half_angle().sin_cos();
        (self.x_axis() * cos_u + self.y_axis() * sin_u) * cos_a + self.axis() * sin_a
    }
}

impl ParametricSurface for SphericalSurface {
    #[inline]
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate(u, v)
    }

    #[inline]
    fn normal(&self, u: f64, v: f64) -> Vec3 {
        self.normal(u, v)
    }

    #[inline]
    fn project_point(&self, point: Point3) -> (f64, f64) {
        self.project_point(point)
    }

    #[inline]
    fn partial_u(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let cos_v = v.cos();
        self.x_axis() * (-self.radius() * cos_v * sin_u)
            + self.y_axis() * (self.radius() * cos_v * cos_u)
    }

    #[inline]
    fn partial_v(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        self.x_axis() * (-self.radius() * sin_v * cos_u)
            + self.y_axis() * (-self.radius() * sin_v * sin_u)
            + self.z_axis() * (self.radius() * cos_v)
    }
}

impl ParametricSurface for ToroidalSurface {
    #[inline]
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate(u, v)
    }

    #[inline]
    fn normal(&self, u: f64, v: f64) -> Vec3 {
        self.normal(u, v)
    }

    #[inline]
    fn project_point(&self, point: Point3) -> (f64, f64) {
        self.project_point(point)
    }

    #[inline]
    fn partial_u(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let cos_v = v.cos();
        let tube_radius = self.major_radius() + self.minor_radius() * cos_v;
        self.x_axis() * (-tube_radius * sin_u) + self.y_axis() * (tube_radius * cos_u)
    }

    #[inline]
    fn partial_v(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        (self.x_axis() * cos_u + self.y_axis() * sin_u) * (-self.minor_radius() * sin_v)
            + self.z_axis() * (self.minor_radius() * cos_v)
    }
}

impl ParametricSurface for NurbsSurface {
    #[inline]
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate(u, v)
    }

    #[inline]
    fn normal(&self, u: f64, v: f64) -> Vec3 {
        self.normal(u, v)
            .unwrap_or_else(|_| Vec3::new(0.0, 0.0, 1.0))
    }

    #[inline]
    fn project_point(&self, point: Point3) -> (f64, f64) {
        // Use default linear tolerance (1e-7) for the Newton projection.
        if let Ok(proj) = project_point_to_surface(self, point, 1e-7) {
            (proj.u, proj.v)
        } else {
            // Fallback: return domain midpoint if Newton fails.
            let (u0, u1) = self.domain_u();
            let (v0, v1) = self.domain_v();
            ((u0 + u1) * 0.5, (v0 + v1) * 0.5)
        }
    }

    #[inline]
    fn partial_u(&self, u: f64, v: f64) -> Vec3 {
        let d = self.derivatives(u, v, 1);
        d[1][0]
    }

    #[inline]
    fn partial_v(&self, u: f64, v: f64) -> Vec3 {
        let d = self.derivatives(u, v, 1);
        d[0][1]
    }
}

// ── ParametricCurve implementations ─────────────────────────────────

impl ParametricCurve for Circle3D {
    #[inline]
    fn evaluate(&self, t: f64) -> Point3 {
        self.evaluate(t)
    }

    #[inline]
    fn tangent(&self, t: f64) -> Vec3 {
        self.tangent(t)
    }

    #[inline]
    fn domain(&self) -> (f64, f64) {
        (0.0, std::f64::consts::TAU)
    }
}

impl ParametricCurve for Ellipse3D {
    #[inline]
    fn evaluate(&self, t: f64) -> Point3 {
        self.evaluate(t)
    }

    #[inline]
    fn tangent(&self, t: f64) -> Vec3 {
        self.tangent(t)
    }

    #[inline]
    fn domain(&self) -> (f64, f64) {
        (0.0, std::f64::consts::TAU)
    }
}

impl ParametricCurve for NurbsCurve {
    #[inline]
    fn evaluate(&self, t: f64) -> Point3 {
        self.evaluate(t)
    }

    #[inline]
    fn tangent(&self, t: f64) -> Vec3 {
        self.tangent(t).unwrap_or_else(|_| Vec3::new(1.0, 0.0, 0.0))
    }

    #[inline]
    fn domain(&self) -> (f64, f64) {
        self.domain()
    }
}
