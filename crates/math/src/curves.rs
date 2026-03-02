//! Analytic 3D curve types: lines, circles, and ellipses.
//!
//! These provide exact evaluation (no NURBS approximation) for the
//! most common curve types in CAD. Equivalent to `Geom_Line`,
//! `Geom_Circle`, and `Geom_Ellipse` in `OpenCascade`.

use std::f64::consts::PI;

use crate::MathError;
use crate::vec::{Point3, Vec3};

// ── Line3D ─────────────────────────────────────────────────────────

/// A 3D line defined by origin and direction.
///
/// Parameterized as `P(t) = origin + t * direction`.
#[derive(Debug, Clone)]
pub struct Line3D {
    origin: Point3,
    direction: Vec3,
}

impl Line3D {
    /// Create a new line.
    ///
    /// # Errors
    ///
    /// Returns an error if `direction` is zero-length.
    pub fn new(origin: Point3, direction: Vec3) -> Result<Self, MathError> {
        let len = direction.length();
        if len < 1e-15 {
            return Err(MathError::ZeroVector);
        }
        Ok(Self {
            origin,
            direction: Vec3::new(
                direction.x() / len,
                direction.y() / len,
                direction.z() / len,
            ),
        })
    }

    /// Evaluate the line at parameter `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }

    /// The tangent direction (constant for a line).
    #[must_use]
    pub const fn tangent(&self) -> Vec3 {
        self.direction
    }

    /// Project a point onto the line, returning the parameter.
    #[must_use]
    pub fn project(&self, point: Point3) -> f64 {
        let v = point - self.origin;
        self.direction.dot(v)
    }

    /// Distance from a point to the line.
    #[must_use]
    pub fn distance_to_point(&self, point: Point3) -> f64 {
        let v = point - self.origin;
        let proj = self.direction * self.direction.dot(v);
        (v - proj).length()
    }

    /// The line origin.
    #[must_use]
    pub const fn origin(&self) -> Point3 {
        self.origin
    }

    /// The unit direction.
    #[must_use]
    pub const fn direction(&self) -> Vec3 {
        self.direction
    }
}

// ── Circle3D ───────────────────────────────────────────────────────

/// A 3D circle defined by center, normal (axis), and radius.
///
/// Parameterized as `P(t) = center + radius*(cos(t)*u + sin(t)*v)`
/// where `u` and `v` form an orthonormal basis in the circle plane.
/// `t` ranges from 0 to 2π for a full circle.
#[derive(Debug, Clone)]
pub struct Circle3D {
    center: Point3,
    normal: Vec3,
    radius: f64,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl Circle3D {
    /// Create a new circle.
    ///
    /// # Errors
    ///
    /// Returns an error if `radius` is non-positive or `normal` is zero.
    pub fn new(center: Point3, normal: Vec3, radius: f64) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        let n = normal.normalize()?;

        // Build orthonormal basis in the circle plane.
        let candidate = if n.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = n.cross(candidate).normalize()?;
        let v = n.cross(u);

        Ok(Self {
            center,
            normal: n,
            radius,
            u_axis: u,
            v_axis: v,
        })
    }

    /// Evaluate the circle at angle `t` (radians).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point3 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        self.center + self.u_axis * (self.radius * cos_t) + self.v_axis * (self.radius * sin_t)
    }

    /// Tangent at angle `t` (unit-length).
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec3 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        self.u_axis * (-sin_t) + self.v_axis * cos_t
    }

    /// The circle circumference.
    #[must_use]
    pub fn circumference(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// The circle center.
    #[must_use]
    pub const fn center(&self) -> Point3 {
        self.center
    }

    /// The circle radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// The circle normal (axis direction).
    #[must_use]
    pub const fn normal(&self) -> Vec3 {
        self.normal
    }

    /// Project a point onto the circle, returning the angle parameter.
    #[must_use]
    pub fn project(&self, point: Point3) -> f64 {
        let v = point - self.center;
        let u_comp = self.u_axis.dot(v);
        let v_comp = self.v_axis.dot(v);
        v_comp.atan2(u_comp)
    }
}

// ── Ellipse3D ──────────────────────────────────────────────────────

/// A 3D ellipse defined by center, normal, and two semi-axis lengths.
///
/// Parameterized as `P(t) = center + a*cos(t)*u + b*sin(t)*v`.
#[derive(Debug, Clone)]
pub struct Ellipse3D {
    center: Point3,
    normal: Vec3,
    semi_major: f64,
    semi_minor: f64,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl Ellipse3D {
    /// Create a new ellipse.
    ///
    /// `semi_major` is the larger radius, `semi_minor` the smaller.
    /// The major axis lies along the `u_axis` direction (computed from normal).
    ///
    /// # Errors
    ///
    /// Returns an error if either semi-axis is non-positive.
    pub fn new(
        center: Point3,
        normal: Vec3,
        semi_major: f64,
        semi_minor: f64,
    ) -> Result<Self, MathError> {
        if semi_major <= 0.0 || semi_minor <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: semi_major.min(semi_minor),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if semi_minor > semi_major {
            return Err(MathError::ParameterOutOfRange {
                value: semi_minor,
                min: 0.0,
                max: semi_major,
            });
        }
        let n = normal.normalize()?;
        let candidate = if n.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = n.cross(candidate).normalize()?;
        let v = n.cross(u);

        Ok(Self {
            center,
            normal: n,
            semi_major,
            semi_minor,
            u_axis: u,
            v_axis: v,
        })
    }

    /// Evaluate the ellipse at angle `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point3 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        self.center
            + self.u_axis * (self.semi_major * cos_t)
            + self.v_axis * (self.semi_minor * sin_t)
    }

    /// Tangent at angle `t` (not unit-length).
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec3 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        self.u_axis * (-self.semi_major * sin_t) + self.v_axis * (self.semi_minor * cos_t)
    }

    /// The ellipse center.
    #[must_use]
    pub const fn center(&self) -> Point3 {
        self.center
    }

    /// Semi-major axis length.
    #[must_use]
    pub const fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Semi-minor axis length.
    #[must_use]
    pub const fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// The ellipse normal (axis direction).
    #[must_use]
    pub const fn normal(&self) -> Vec3 {
        self.normal
    }

    /// Approximate circumference using Ramanujan's formula.
    #[must_use]
    pub fn approximate_circumference(&self) -> f64 {
        let a = self.semi_major;
        let b = self.semi_minor;
        let h = (a - b) * (a - b) / ((a + b) * (a + b));
        PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (3.0f64.mul_add(-h, 4.0)).sqrt()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use crate::tolerance::Tolerance;
    use crate::vec::{Point3, Vec3};

    use super::*;

    // ── Line tests ─────────────────────────────────────────────────

    #[test]
    fn line_evaluate() {
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let tol = Tolerance::new();

        let p = line.evaluate(3.0);
        assert!(tol.approx_eq(p.x(), 3.0));
        assert!(tol.approx_eq(p.y(), 0.0));
    }

    #[test]
    fn line_project() {
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let t = line.project(Point3::new(5.0, 3.0, 0.0));
        let tol = Tolerance::new();
        assert!(tol.approx_eq(t, 5.0));
    }

    #[test]
    fn line_distance() {
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let d = line.distance_to_point(Point3::new(5.0, 3.0, 4.0));
        let tol = Tolerance::new();
        assert!(tol.approx_eq(d, 5.0)); // 3-4-5 triangle
    }

    #[test]
    fn line_zero_direction_error() {
        assert!(Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)).is_err());
    }

    // ── Circle tests ───────────────────────────────────────────────

    #[test]
    fn circle_evaluate_at_zero() {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let tol = Tolerance::new();
        let p = circle.evaluate(0.0);
        // Should be at radius distance from center in the plane.
        let dist = (p - circle.center()).length();
        assert!(
            tol.approx_eq(dist, 1.0),
            "point should be on circle, dist={dist}"
        );
    }

    #[test]
    fn circle_evaluate_quarter() {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let tol = Tolerance::new();
        let p = circle.evaluate(FRAC_PI_2);
        let dist = (p - circle.center()).length();
        assert!(tol.approx_eq(dist, 2.0));
    }

    #[test]
    fn circle_circumference() {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

        let tol = Tolerance::new();
        assert!(tol.approx_eq(circle.circumference(), 6.0 * PI));
    }

    #[test]
    fn circle_project_roundtrip() {
        let circle =
            Circle3D::new(Point3::new(1.0, 2.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();

        let tol = Tolerance::new();
        let t_orig = 1.23;
        let p = circle.evaluate(t_orig);
        let t_proj = circle.project(p);
        assert!(
            tol.approx_eq(t_orig, t_proj),
            "project should roundtrip: {t_orig} vs {t_proj}"
        );
    }

    #[test]
    fn circle_zero_radius_error() {
        assert!(Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0).is_err());
    }

    // ── Ellipse tests ──────────────────────────────────────────────

    #[test]
    fn ellipse_evaluate() {
        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
        )
        .unwrap();

        let tol = Tolerance::new();
        // At t=0, should be at (3, 0, 0) direction.
        let p0 = ellipse.evaluate(0.0);
        let dist0 = (p0 - ellipse.center()).length();
        assert!(tol.approx_eq(dist0, 3.0), "at t=0 should be at semi_major");

        // At t=π/2, should be at semi_minor distance.
        let p1 = ellipse.evaluate(FRAC_PI_2);
        let dist1 = (p1 - ellipse.center()).length();
        assert!(
            tol.approx_eq(dist1, 2.0),
            "at t=π/2 should be at semi_minor"
        );
    }

    #[test]
    fn ellipse_circumference_circle() {
        // When semi_major == semi_minor, it's a circle.
        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
        )
        .unwrap();

        let tol = Tolerance::loose();
        let circ = ellipse.approximate_circumference();
        assert!(
            tol.approx_eq(circ, 2.0 * PI),
            "circle circumference should be 2π, got {circ}"
        );
    }

    #[test]
    fn ellipse_zero_axis_error() {
        assert!(
            Ellipse3D::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                0.0,
                1.0
            )
            .is_err()
        );
    }
}
