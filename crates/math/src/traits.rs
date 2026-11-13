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

    /// Both partials `(∂S/∂u, ∂S/∂v)` at (u, v), as `partial_u` and
    /// `partial_v` give them; a surface that works both out at once
    /// overrides it.
    fn partials(&self, u: f64, v: f64) -> (Vec3, Vec3) {
        (self.partial_u(u, v), self.partial_v(u, v))
    }

    /// The u range `(start, end)` of a surface that closes on itself across
    /// it, so that u wraps round by `end - start`. `None` (the default) for
    /// a surface open in u, or one whose evaluation is already periodic in
    /// u (the analytic surfaces).
    fn closed_u(&self) -> Option<(f64, f64)> {
        None
    }

    /// The v range of a surface that closes on itself across it, as
    /// [`Self::closed_u`] for u.
    fn closed_v(&self) -> Option<(f64, f64)> {
        None
    }
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
        self.first_derivatives(u, v).1
    }

    #[inline]
    fn partial_v(&self, u: f64, v: f64) -> Vec3 {
        self.first_derivatives(u, v).2
    }

    #[inline]
    fn partials(&self, u: f64, v: f64) -> (Vec3, Vec3) {
        let (_, su, sv) = self.first_derivatives(u, v);
        (su, sv)
    }

    fn closed_u(&self) -> Option<(f64, f64)> {
        self.is_periodic_u().then(|| self.domain_u())
    }

    fn closed_v(&self) -> Option<(f64, f64)> {
        self.is_periodic_v().then(|| self.domain_v())
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
