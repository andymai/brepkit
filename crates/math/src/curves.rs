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

    /// The u-axis direction (major axis in the circle plane).
    #[must_use]
    pub const fn u_axis(&self) -> Vec3 {
        self.u_axis
    }

    /// The v-axis direction (minor axis in the circle plane).
    #[must_use]
    pub const fn v_axis(&self) -> Vec3 {
        self.v_axis
    }

    /// Create a circle with explicit basis vectors (for transform/copy).
    ///
    /// # Errors
    ///
    /// Returns an error if `radius` is non-positive.
    pub fn with_axes(
        center: Point3,
        normal: Vec3,
        radius: f64,
        u_axis: Vec3,
        v_axis: Vec3,
    ) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(Self {
            center,
            normal,
            radius,
            u_axis,
            v_axis,
        })
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

    /// Project a point onto the ellipse, returning the angle parameter.
    #[must_use]
    pub fn project(&self, point: Point3) -> f64 {
        let v = point - self.center;
        let u_comp = self.u_axis.dot(v) / self.semi_major;
        let v_comp = self.v_axis.dot(v) / self.semi_minor;
        v_comp.atan2(u_comp)
    }

    /// The u-axis direction (major axis direction).
    #[must_use]
    pub const fn u_axis(&self) -> Vec3 {
        self.u_axis
    }

    /// The v-axis direction (minor axis direction).
    #[must_use]
    pub const fn v_axis(&self) -> Vec3 {
        self.v_axis
    }

    /// Create an ellipse with explicit basis vectors (for transform/copy).
    ///
    /// # Errors
    ///
    /// Returns an error if either semi-axis is non-positive.
    pub fn with_axes(
        center: Point3,
        normal: Vec3,
        semi_major: f64,
        semi_minor: f64,
        u_axis: Vec3,
        v_axis: Vec3,
    ) -> Result<Self, MathError> {
        if semi_major <= 0.0 || semi_minor <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: semi_major.min(semi_minor),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(Self {
            center,
            normal,
            semi_major,
            semi_minor,
            u_axis,
            v_axis,
        })
    }
}

/// A 3D parabola defined by vertex, axis direction, and focal length.
///
/// Parameterized as `P(t) = vertex + (t²/(4f)) * axis_dir + t * u_axis`
/// where `f` is the focal length and `u_axis` is perpendicular to the axis
/// in the parabola plane.
///
/// The parameter `t` ranges over all reals; `t = 0` is the vertex.
#[derive(Debug, Clone)]
pub struct Parabola3D {
    vertex: Point3,
    axis_dir: Vec3,
    focal_length: f64,
    u_axis: Vec3,
}

impl Parabola3D {
    /// Creates a new parabola.
    ///
    /// `axis_dir` is the direction from vertex toward the interior of the
    /// parabola (the axis of symmetry). `focal_length` is the distance
    /// from vertex to focus.
    ///
    /// # Errors
    /// Returns an error if `focal_length` is not positive or `axis_dir` is zero.
    pub fn new(vertex: Point3, axis_dir: Vec3, focal_length: f64) -> Result<Self, MathError> {
        if focal_length <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: focal_length,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let axis = axis_dir.normalize()?;
        let candidate = if axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = axis.cross(candidate).normalize()?;
        Ok(Self {
            vertex,
            axis_dir: axis,
            focal_length,
            u_axis: u,
        })
    }

    /// Evaluates the parabola at parameter `t`.
    ///
    /// At `t = 0` this returns the vertex.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point3 {
        let along_axis = (t * t) / (4.0 * self.focal_length);
        self.vertex + self.axis_dir * along_axis + self.u_axis * t
    }

    /// Returns the tangent vector at parameter `t`.
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec3 {
        let d_axis = t / (2.0 * self.focal_length);
        self.axis_dir * d_axis + self.u_axis
    }

    /// Returns the curvature at parameter `t`.
    #[must_use]
    pub fn curvature(&self, t: f64) -> f64 {
        let two_f = 2.0 * self.focal_length;
        let ratio = t / two_f;
        let denom = ratio.mul_add(ratio, 1.0);
        1.0 / (two_f * denom.powf(1.5))
    }

    /// Returns the vertex.
    #[must_use]
    pub const fn vertex(&self) -> Point3 {
        self.vertex
    }

    /// Returns the focal length.
    #[must_use]
    pub const fn focal_length(&self) -> f64 {
        self.focal_length
    }

    /// Returns the axis direction (normalized).
    #[must_use]
    pub const fn axis_dir(&self) -> Vec3 {
        self.axis_dir
    }

    /// Returns the focus point.
    #[must_use]
    pub fn focus(&self) -> Point3 {
        self.vertex + self.axis_dir * self.focal_length
    }
}

/// A 3D hyperbola defined by center, axis, and two semi-axis lengths.
///
/// Parameterized as `P(t) = center + a * cosh(t) * u_axis + b * sinh(t) * v_axis`.
///
/// The parameter `t` ranges over all reals; `t = 0` gives the vertex
/// closest to center on the positive branch.
#[derive(Debug, Clone)]
pub struct Hyperbola3D {
    center: Point3,
    normal: Vec3,
    semi_major: f64,
    semi_minor: f64,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl Hyperbola3D {
    /// Creates a new hyperbola.
    ///
    /// `semi_major` is the real semi-axis (distance from center to vertex),
    /// `semi_minor` is the imaginary semi-axis.
    ///
    /// # Errors
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
                min: f64::EPSILON,
                max: f64::MAX,
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

    /// Evaluates the hyperbola at parameter `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point3 {
        self.center
            + self.u_axis * (self.semi_major * t.cosh())
            + self.v_axis * (self.semi_minor * t.sinh())
    }

    /// Returns the tangent vector at parameter `t`.
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec3 {
        self.u_axis * (self.semi_major * t.sinh()) + self.v_axis * (self.semi_minor * t.cosh())
    }

    /// Returns the center.
    #[must_use]
    pub const fn center(&self) -> Point3 {
        self.center
    }

    /// Returns the semi-major axis (real axis).
    #[must_use]
    pub const fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Returns the semi-minor axis (imaginary axis).
    #[must_use]
    pub const fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Returns the normal (axis perpendicular to the hyperbola plane).
    #[must_use]
    pub const fn normal(&self) -> Vec3 {
        self.normal
    }

    /// Returns the eccentricity: `e = sqrt(1 + (b/a)²)`.
    #[must_use]
    pub fn eccentricity(&self) -> f64 {
        let ratio = self.semi_minor / self.semi_major;
        ratio.mul_add(ratio, 1.0).sqrt()
    }

    /// Returns the two foci.
    #[must_use]
    pub fn foci(&self) -> (Point3, Point3) {
        let c = self.semi_major.hypot(self.semi_minor);
        (
            self.center + self.u_axis * c,
            self.center + self.u_axis * (-c),
        )
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

    // ── Parabola tests ──────────────────────────────────────────

    #[test]
    fn parabola_vertex() {
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let v = p.evaluate(0.0);
        assert!(Tolerance::default().approx_eq(v.x(), 0.0));
        assert!(Tolerance::default().approx_eq(v.y(), 0.0));
        assert!(Tolerance::default().approx_eq(v.z(), 0.0));
    }

    #[test]
    fn parabola_symmetry() {
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        // z-component should be the same for +t and -t (symmetric about axis)
        let pos = p.evaluate(2.0);
        let neg = p.evaluate(-2.0);
        assert!(Tolerance::default().approx_eq(pos.z(), neg.z()));
    }

    #[test]
    fn parabola_tangent_at_vertex() {
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let tang = p.tangent(0.0);
        // At vertex, tangent should be purely along u_axis (perpendicular to axis)
        assert!(Tolerance::default().approx_eq(tang.z(), 0.0));
        assert!(tang.length() > 0.5);
    }

    #[test]
    fn parabola_curvature_at_vertex() {
        let f = 2.0;
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), f).unwrap();
        // Curvature at vertex = 1/(2f)
        let k = p.curvature(0.0);
        assert!(Tolerance::default().approx_eq(k, 1.0 / (2.0 * f)));
    }

    #[test]
    fn parabola_focus() {
        let f = 3.0;
        let p = Parabola3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), f).unwrap();
        let focus = p.focus();
        assert!(Tolerance::default().approx_eq(focus.z(), 3.0 + f));
    }

    #[test]
    fn parabola_zero_focal_length_error() {
        assert!(
            Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,).is_err()
        );
    }

    // ── Hyperbola tests ─────────────────────────────────────────

    #[test]
    fn hyperbola_vertex() {
        let h = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
        )
        .unwrap();
        // At t=0: P = center + a*cosh(0)*u + b*sinh(0)*v = center + a*u
        let p = h.evaluate(0.0);
        let dist = (p - h.center()).length();
        assert!(Tolerance::default().approx_eq(dist, 3.0));
    }

    #[test]
    fn hyperbola_tangent_at_vertex() {
        let h = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
        )
        .unwrap();
        // At t=0: tangent = a*sinh(0)*u + b*cosh(0)*v = b*v
        let tang = h.tangent(0.0);
        assert!(Tolerance::default().approx_eq(tang.length(), 2.0));
    }

    #[test]
    fn hyperbola_eccentricity() {
        let h = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            4.0,
        )
        .unwrap();
        // e = sqrt(1 + (b/a)^2) = sqrt(1 + 16/9) = sqrt(25/9) = 5/3
        assert!(Tolerance::default().approx_eq(h.eccentricity(), 5.0 / 3.0));
    }

    #[test]
    fn hyperbola_foci_distance() {
        let a = 3.0;
        let b = 4.0;
        let h =
            Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
        let (f1, f2) = h.foci();
        // Distance from center to each focus = c = sqrt(a^2 + b^2) = 5
        let c = a.hypot(b);
        assert!(Tolerance::default().approx_eq((f1 - h.center()).length(), c));
        assert!(Tolerance::default().approx_eq((f2 - h.center()).length(), c));
    }

    #[test]
    fn hyperbola_zero_axis_error() {
        assert!(
            Hyperbola3D::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                0.0,
                1.0,
            )
            .is_err()
        );
    }

    // ── Line3D::tangent tests ──────────────────────────

    #[test]
    fn line3d_tangent_is_unit_vector() {
        // Even when constructed with a non-unit direction, tangent() must return a unit vector.
        let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0)).unwrap();
        let tang = line.tangent();
        let tol = Tolerance::new();
        assert!(
            tol.approx_eq(tang.length(), 1.0),
            "tangent should be unit length, got {}",
            tang.length()
        );
    }

    #[test]
    fn line3d_tangent_matches_direction() {
        let line = Line3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 5.0)).unwrap();
        let tang = line.tangent();
        let dir = line.direction();
        let tol = Tolerance::new();
        assert!(tol.approx_eq(tang.x(), dir.x()));
        assert!(tol.approx_eq(tang.y(), dir.y()));
        assert!(tol.approx_eq(tang.z(), dir.z()));
    }

    // ── Circle3D::project tests ────────────────────────

    #[test]
    fn circle3d_project_at_pi() {
        // At t=π, evaluate gives the antipodal point; projecting back should return π.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let pt = circle.evaluate(PI);
        let t_proj = circle.project(pt);
        let tol = Tolerance::new();
        // project returns atan2, which is in (-π, π]; -π and π are the same angle
        assert!(tol.approx_eq(t_proj.abs(), PI), "expected ±π, got {t_proj}");
    }

    #[test]
    fn circle3d_project_at_negative_angle() {
        // project should recover a negative angle correctly (atan2 range is (-π, π])
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let t_orig = -1.0_f64;
        let pt = circle.evaluate(t_orig);
        let t_proj = circle.project(pt);
        let tol = Tolerance::new();
        assert!(
            tol.approx_eq(t_orig, t_proj),
            "expected {t_orig}, got {t_proj}"
        );
    }

    #[test]
    fn circle3d_project_at_three_quarter() {
        // t = 3π/2 maps to -π/2 under atan2; round-trip through evaluate→project
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        let t_orig = 3.0 * PI / 2.0;
        let pt = circle.evaluate(t_orig);
        let t_proj = circle.project(pt);
        // atan2 wraps to -π/2, so recovered angle differs by 2π
        let tol = Tolerance::new();
        let diff = (t_proj - t_orig).abs() % (2.0 * PI);
        let diff_wrapped = diff.min(2.0f64.mul_add(PI, -diff));
        assert!(
            diff_wrapped < tol.linear * 1000.0,
            "angle mismatch after wrapping: t_orig={t_orig}, t_proj={t_proj}"
        );
    }

    // ── Parabola3D::evaluate at non-zero t ─────────────

    #[test]
    fn parabola_evaluate_nonzero_t() {
        // P(t) = vertex + (t²/4f)*axis + t*u_axis
        // For axis = (0,0,1), focal_length=1, at t=2:
        //   along_axis = 4/4 = 1   → z-component = 1
        //   u_axis component = 2    → in the plane perpendicular to z
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let pt = p.evaluate(2.0);
        let tol = Tolerance::new();
        // z (along-axis) component
        assert!(
            tol.approx_eq(pt.z(), 1.0),
            "z-component should be 1.0, got {}",
            pt.z()
        );
        // distance from axis in the XY plane should be |t| = 2
        let radial = pt.x().hypot(pt.y());
        assert!(
            tol.approx_eq(radial, 2.0),
            "radial distance should be 2.0, got {radial}"
        );
    }

    #[test]
    fn parabola_evaluate_negative_t() {
        // Symmetry: z-component the same for +t and -t, radial same, u_axis direction flips
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let pos = p.evaluate(3.0);
        let neg = p.evaluate(-3.0);
        let tol = Tolerance::new();
        assert!(tol.approx_eq(pos.z(), neg.z()), "z should match for ±t");
        // The x/y components should be opposite in sign (u_axis flips)
        let radial_pos = pos.x().hypot(pos.y());
        let radial_neg = neg.x().hypot(neg.y());
        assert!(
            tol.approx_eq(radial_pos, radial_neg),
            "radial distance should match"
        );
    }

    #[test]
    fn parabola_tangent_nonzero_t() {
        // tangent(t) = (t / 2f) * axis + u_axis
        // For focal_length=2, axis=(0,0,1), t=4:
        //   d_axis = 4/(2*2) = 1  → z-component of tangent = 1
        //   u_axis component = 1
        let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let tang = p.tangent(4.0);
        let tol = Tolerance::new();
        assert!(
            tol.approx_eq(tang.z(), 1.0),
            "z-component of tangent should be 1.0, got {}",
            tang.z()
        );
        // u_axis is in XY plane, its component should give length 1
        let radial = tang.x().hypot(tang.y());
        assert!(
            tol.approx_eq(radial, 1.0),
            "XY component of tangent should have length 1, got {radial}"
        );
    }

    // ── Hyperbola3D::evaluate away from vertex ─────────

    #[test]
    fn hyperbola_evaluate_nonzero_t() {
        // P(t) = center + a*cosh(t)*u + b*sinh(t)*v
        // The distance from center at parameter t satisfies:
        //   |P - center|² = a²*cosh²(t) + b²*sinh²(t)
        let a = 2.0_f64;
        let b = 1.0_f64;
        let t = 1.0_f64;
        let h =
            Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
        let pt = h.evaluate(t);
        let dist_sq = (pt - h.center()).length_squared();
        let expected_sq = (a * t.cosh()).mul_add(a * t.cosh(), (b * t.sinh()).powi(2));
        assert!(
            (dist_sq - expected_sq).abs() < 1e-10,
            "|P-center|² should be {expected_sq}, got {dist_sq}"
        );
    }

    #[test]
    fn hyperbola_evaluate_large_t_stays_on_curve() {
        // Verify the hyperbolic identity: (x/a)² - (y/b)² = 1
        // where x, y are the in-plane u/v components.
        // We use a finite-difference approximation to recover u/v components.
        let a = 3.0_f64;
        let b = 2.0_f64;
        let h =
            Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
        // The distance from center equals sqrt(a²cosh²(t) + b²sinh²(t))
        for t in [0.5_f64, 1.0, 2.0] {
            let pt = h.evaluate(t);
            let dist = (pt - h.center()).length();
            let expected = (a * t.cosh()).hypot(b * t.sinh());
            assert!(
                (dist - expected).abs() < 1e-10,
                "distance at t={t}: expected {expected}, got {dist}"
            );
        }
    }

    #[test]
    fn hyperbola_tangent_nonzero_t() {
        // Check tangent via finite difference: tangent ≈ (P(t+ε) - P(t-ε)) / (2ε)
        let h = Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1.0,
        )
        .unwrap();
        let t = 1.0_f64;
        let eps = 1e-6;
        let tang_analytic = h.tangent(t);
        let pt_plus = h.evaluate(t + eps);
        let pt_minus = h.evaluate(t - eps);
        let tang_fd_x = (pt_plus.x() - pt_minus.x()) / (2.0 * eps);
        let tang_fd_y = (pt_plus.y() - pt_minus.y()) / (2.0 * eps);
        let tang_fd_z = (pt_plus.z() - pt_minus.z()) / (2.0 * eps);
        let tol = 1e-5;
        assert!(
            (tang_analytic.x() - tang_fd_x).abs() < tol,
            "x: analytic={}, fd={tang_fd_x}",
            tang_analytic.x()
        );
        assert!(
            (tang_analytic.y() - tang_fd_y).abs() < tol,
            "y: analytic={}, fd={tang_fd_y}",
            tang_analytic.y()
        );
        assert!(
            (tang_analytic.z() - tang_fd_z).abs() < tol,
            "z: analytic={}, fd={tang_fd_z}",
            tang_analytic.z()
        );
    }
}
