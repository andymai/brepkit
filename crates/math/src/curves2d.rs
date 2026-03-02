//! 2D analytic curve types for parametric curves on surfaces (pcurves).
//!
//! These curves live in a surface's (u, v) parameter space and are used
//! for exact surface trimming and boolean operations.

use crate::MathError;
use crate::vec::{Point2, Vec2};

/// A 2D line defined by origin and direction.
///
/// Parameterized as `P(t) = origin + t * direction`.
#[derive(Debug, Clone)]
pub struct Line2D {
    origin: Point2,
    direction: Vec2,
}

impl Line2D {
    /// Creates a new 2D line.
    ///
    /// # Errors
    /// Returns `MathError::ZeroVector` if direction has zero length.
    pub fn new(origin: Point2, direction: Vec2) -> Result<Self, MathError> {
        let len = direction.length();
        if len < 1e-15 {
            return Err(MathError::ZeroVector);
        }
        Ok(Self {
            origin,
            direction: Vec2::new(direction.x() / len, direction.y() / len),
        })
    }

    /// Evaluates the line at parameter `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point2 {
        self.origin + self.direction * t
    }

    /// Returns the (constant) tangent direction.
    #[must_use]
    pub const fn tangent(&self, _t: f64) -> Vec2 {
        self.direction
    }

    /// Projects a point onto the line, returning the closest parameter.
    #[must_use]
    pub fn project(&self, point: Point2) -> f64 {
        let v = point - self.origin;
        v.dot(self.direction) / self.direction.length_squared()
    }

    /// Returns the perpendicular distance from a point to the line.
    #[must_use]
    pub fn distance_to_point(&self, point: Point2) -> f64 {
        let t = self.project(point);
        let closest = self.evaluate(t);
        let diff = point - closest;
        diff.length()
    }

    /// Returns the origin point.
    #[must_use]
    pub const fn origin(&self) -> Point2 {
        self.origin
    }

    /// Returns the direction vector.
    #[must_use]
    pub const fn direction(&self) -> Vec2 {
        self.direction
    }
}

/// A 2D circle defined by center and radius.
///
/// Parameterized as `P(t) = center + radius * (cos(t), sin(t))`.
/// `t` ranges from 0 to 2π for a full circle.
#[derive(Debug, Clone)]
pub struct Circle2D {
    center: Point2,
    radius: f64,
}

impl Circle2D {
    /// Creates a new 2D circle.
    ///
    /// # Errors
    /// Returns `MathError::ParameterOutOfRange` if radius is not positive.
    pub fn new(center: Point2, radius: f64) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        Ok(Self { center, radius })
    }

    /// Evaluates the circle at parameter `t` (radians).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point2 {
        self.center + Vec2::new(self.radius * t.cos(), self.radius * t.sin())
    }

    /// Returns the tangent vector at parameter `t`.
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec2 {
        Vec2::new(-self.radius * t.sin(), self.radius * t.cos())
    }

    /// Returns the circumference.
    #[must_use]
    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    /// Projects a point onto the circle, returning the closest parameter.
    #[must_use]
    pub fn project(&self, point: Point2) -> f64 {
        let v = point - self.center;
        v.y().atan2(v.x()).rem_euclid(2.0 * std::f64::consts::PI)
    }

    /// Returns the center.
    #[must_use]
    pub const fn center(&self) -> Point2 {
        self.center
    }

    /// Returns the radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }
}

/// A 2D ellipse defined by center and two semi-axis lengths.
///
/// Parameterized as `P(t) = center + a*cos(t)*u + b*sin(t)*v`
/// where `u = (cos(rotation), sin(rotation))` and `v = (-sin(rotation), cos(rotation))`.
#[derive(Debug, Clone)]
pub struct Ellipse2D {
    center: Point2,
    semi_major: f64,
    semi_minor: f64,
    rotation: f64,
}

impl Ellipse2D {
    /// Creates a new 2D ellipse.
    ///
    /// # Errors
    /// Returns `MathError::ParameterOutOfRange` if either semi-axis is not positive.
    pub fn new(
        center: Point2,
        semi_major: f64,
        semi_minor: f64,
        rotation: f64,
    ) -> Result<Self, MathError> {
        if semi_major <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: semi_major,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        if semi_minor <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: semi_minor,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        if semi_minor > semi_major {
            return Err(MathError::ParameterOutOfRange {
                value: semi_minor,
                min: 0.0,
                max: semi_major,
            });
        }
        Ok(Self {
            center,
            semi_major,
            semi_minor,
            rotation,
        })
    }

    /// Evaluates the ellipse at parameter `t` (radians).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point2 {
        let (sin_r, cos_r) = self.rotation.sin_cos();
        let x = self.semi_major * t.cos();
        let y = self.semi_minor * t.sin();
        self.center + Vec2::new(x.mul_add(cos_r, -(y * sin_r)), x.mul_add(sin_r, y * cos_r))
    }

    /// Returns the tangent vector at parameter `t`.
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec2 {
        let (sin_r, cos_r) = self.rotation.sin_cos();
        let dx = -self.semi_major * t.sin();
        let dy = self.semi_minor * t.cos();
        Vec2::new(
            dx.mul_add(cos_r, -(dy * sin_r)),
            dx.mul_add(sin_r, dy * cos_r),
        )
    }

    /// Returns the center.
    #[must_use]
    pub const fn center(&self) -> Point2 {
        self.center
    }

    /// Returns the semi-major axis length.
    #[must_use]
    pub const fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Returns the semi-minor axis length.
    #[must_use]
    pub const fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Returns the rotation angle.
    #[must_use]
    pub const fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Approximate circumference using Ramanujan's formula.
    #[must_use]
    pub fn approximate_circumference(&self) -> f64 {
        let a = self.semi_major;
        let b = self.semi_minor;
        let h = ((a - b) * (a - b)) / ((a + b) * (a + b));
        std::f64::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + 3.0f64.mul_add(-h, 4.0).sqrt()))
    }
}

/// A 2D NURBS curve for parametric curves on surfaces.
///
/// This is the 2D analogue of `NurbsCurve`, used as pcurves
/// in the surface parameter space.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve2D {
    degree: usize,
    knots: Vec<f64>,
    control_points: Vec<Point2>,
    weights: Vec<f64>,
}

impl NurbsCurve2D {
    /// Creates a new 2D NURBS curve.
    ///
    /// # Errors
    /// Returns an error if knot vector or weights are inconsistent with
    /// the number of control points and degree.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<Point2>,
        weights: Vec<f64>,
    ) -> Result<Self, MathError> {
        let n = control_points.len();
        let expected_knots = n + degree + 1;
        if knots.len() != expected_knots {
            return Err(MathError::InvalidKnotVector {
                expected: expected_knots,
                got: knots.len(),
            });
        }
        if weights.len() != n {
            return Err(MathError::InvalidWeights {
                expected: n,
                got: weights.len(),
            });
        }
        if n == 0 {
            return Err(MathError::EmptyInput);
        }
        Ok(Self {
            degree,
            knots,
            control_points,
            weights,
        })
    }

    /// Evaluates the curve at parameter `u` using De Boor's algorithm.
    #[must_use]
    pub fn evaluate(&self, u: f64) -> Point2 {
        let n = self.control_points.len();
        let p = self.degree;
        let span = crate::nurbs::basis::find_span(n, p, u, &self.knots);
        let basis = crate::nurbs::basis::basis_funs(span, u, p, &self.knots);

        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut ww = 0.0;

        for (j, &basis_val) in basis.iter().enumerate().take(p + 1) {
            let idx = span - p + j;
            let cp = &self.control_points[idx];
            let w = self.weights[idx];
            let bw = basis_val * w;
            wx += bw * cp.x();
            wy += bw * cp.y();
            ww += bw;
        }

        if ww.abs() < f64::EPSILON {
            return self.control_points[0];
        }
        Point2::new(wx / ww, wy / ww)
    }

    /// Returns the first derivative at parameter `u`.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn tangent(&self, param: f64) -> Vec2 {
        let num_pts = self.control_points.len();
        let deg = self.degree;
        let span = crate::nurbs::basis::find_span(num_pts, deg, param, &self.knots);
        let ders = crate::nurbs::basis::ders_basis_funs(span, param, deg, 1, &self.knots);

        let mut curve_pt = Vec2::new(0.0, 0.0);
        let mut curve_deriv = Vec2::new(0.0, 0.0);
        let mut weight_sum = 0.0;
        let mut weight_deriv = 0.0;

        for (j, basis_ders) in ders[0].iter().enumerate().take(deg + 1) {
            let idx = span - deg + j;
            let wi = self.weights[idx];
            let cp = &self.control_points[idx];
            let cp_vec = Vec2::new(cp.x(), cp.y());

            curve_pt = curve_pt + cp_vec * (wi * basis_ders);
            curve_deriv = curve_deriv + cp_vec * (wi * ders[1][j]);
            weight_sum += wi * basis_ders;
            weight_deriv += wi * ders[1][j];
        }

        if weight_sum.abs() < f64::EPSILON {
            return Vec2::new(0.0, 0.0);
        }
        (curve_deriv * weight_sum - curve_pt * weight_deriv) * (1.0 / (weight_sum * weight_sum))
    }

    /// Returns the parameter domain `(u_min, u_max)`.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        )
    }

    /// Returns the degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Returns a reference to the knot vector.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Returns a reference to the control points.
    #[must_use]
    pub fn control_points(&self) -> &[Point2] {
        &self.control_points
    }

    /// Returns a reference to the weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns true if the curve is rational (any weight != 1.0).
    #[must_use]
    pub fn is_rational(&self) -> bool {
        self.weights.iter().any(|w| (*w - 1.0).abs() > f64::EPSILON)
    }

    /// Creates a 2D line segment as a NURBS curve (degree 1).
    ///
    /// # Errors
    /// Returns `MathError::EmptyInput` if called with invalid parameters (should not happen).
    pub fn from_line(start: Point2, end: Point2) -> Result<Self, MathError> {
        Self::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![start, end],
            vec![1.0, 1.0],
        )
    }
}

/// Unified enum for 2D curve types used as pcurves.
#[derive(Debug, Clone)]
pub enum Curve2D {
    /// A straight line in parameter space.
    Line(Line2D),
    /// A circle in parameter space.
    Circle(Circle2D),
    /// An ellipse in parameter space.
    Ellipse(Ellipse2D),
    /// A NURBS curve in parameter space.
    Nurbs(NurbsCurve2D),
}

impl Curve2D {
    /// Evaluates the curve at parameter `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point2 {
        match self {
            Self::Line(c) => c.evaluate(t),
            Self::Circle(c) => c.evaluate(t),
            Self::Ellipse(c) => c.evaluate(t),
            Self::Nurbs(c) => c.evaluate(t),
        }
    }

    /// Returns the tangent at parameter `t`.
    #[must_use]
    pub fn tangent(&self, t: f64) -> Vec2 {
        match self {
            Self::Line(c) => c.tangent(t),
            Self::Circle(c) => c.tangent(t),
            Self::Ellipse(c) => c.tangent(t),
            Self::Nurbs(c) => c.tangent(t),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn point2_approx_eq(a: Point2, b: Point2) -> bool {
        approx_eq(a.x(), b.x()) && approx_eq(a.y(), b.y())
    }

    // ── Line2D tests ──────────────────────────────────

    #[test]
    fn line2d_evaluate() {
        // Direction is normalized, so parameter t is arc-length
        let line = Line2D::new(Point2::new(1.0, 2.0), Vec2::new(3.0, 4.0)).expect("valid line");

        let p = line.evaluate(0.0);
        assert!(point2_approx_eq(p, Point2::new(1.0, 2.0)));

        // direction normalized to (0.6, 0.8), so at t=5 we get (1+3, 2+4) = (4, 6)
        let p = line.evaluate(5.0);
        assert!(point2_approx_eq(p, Point2::new(4.0, 6.0)));

        let p = line.evaluate(2.5);
        assert!(point2_approx_eq(p, Point2::new(2.5, 4.0)));
    }

    #[test]
    fn line2d_project() {
        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).expect("valid line");

        assert!(approx_eq(line.project(Point2::new(5.0, 0.0)), 5.0));
        assert!(approx_eq(line.project(Point2::new(3.0, 7.0)), 3.0));
    }

    #[test]
    fn line2d_distance() {
        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).expect("valid line");
        assert!(approx_eq(
            line.distance_to_point(Point2::new(3.0, 4.0)),
            4.0
        ));
    }

    #[test]
    fn line2d_zero_direction_error() {
        let result = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(0.0, 0.0));
        assert!(result.is_err());
    }

    // ── Circle2D tests ────────────────────────────────

    #[test]
    fn circle2d_evaluate_at_zero() {
        let circle = Circle2D::new(Point2::new(0.0, 0.0), 1.0).expect("valid circle");
        let p = circle.evaluate(0.0);
        assert!(point2_approx_eq(p, Point2::new(1.0, 0.0)));
    }

    #[test]
    fn circle2d_evaluate_quarter() {
        let circle = Circle2D::new(Point2::new(0.0, 0.0), 2.0).expect("valid circle");
        let p = circle.evaluate(PI / 2.0);
        assert!(point2_approx_eq(p, Point2::new(0.0, 2.0)));
    }

    #[test]
    fn circle2d_circumference() {
        let circle = Circle2D::new(Point2::new(0.0, 0.0), 3.0).expect("valid circle");
        assert!(approx_eq(circle.circumference(), 6.0 * PI));
    }

    #[test]
    fn circle2d_project_roundtrip() {
        let circle = Circle2D::new(Point2::new(1.0, 1.0), 5.0).expect("valid circle");
        let t = 1.234;
        let point = circle.evaluate(t);
        let t_proj = circle.project(point);
        assert!(approx_eq(t, t_proj));
    }

    #[test]
    fn circle2d_zero_radius_error() {
        let result = Circle2D::new(Point2::new(0.0, 0.0), 0.0);
        assert!(result.is_err());
    }

    // ── Ellipse2D tests ───────────────────────────────

    #[test]
    fn ellipse2d_evaluate_no_rotation() {
        let ellipse = Ellipse2D::new(Point2::new(0.0, 0.0), 3.0, 2.0, 0.0).expect("valid ellipse");

        let p = ellipse.evaluate(0.0);
        assert!(point2_approx_eq(p, Point2::new(3.0, 0.0)));

        let p = ellipse.evaluate(PI / 2.0);
        assert!(point2_approx_eq(p, Point2::new(0.0, 2.0)));
    }

    #[test]
    fn ellipse2d_evaluate_with_rotation() {
        let ellipse =
            Ellipse2D::new(Point2::new(0.0, 0.0), 3.0, 2.0, PI / 2.0).expect("valid ellipse");
        let p = ellipse.evaluate(0.0);
        assert!(point2_approx_eq(p, Point2::new(0.0, 3.0)));
    }

    #[test]
    fn ellipse2d_circle_circumference() {
        let ellipse = Ellipse2D::new(Point2::new(0.0, 0.0), 5.0, 5.0, 0.0).expect("valid ellipse");
        assert!(approx_eq(
            ellipse.approximate_circumference(),
            2.0 * PI * 5.0
        ));
    }

    #[test]
    fn ellipse2d_zero_axis_error() {
        assert!(Ellipse2D::new(Point2::new(0.0, 0.0), 0.0, 1.0, 0.0).is_err());
        assert!(Ellipse2D::new(Point2::new(0.0, 0.0), 1.0, 0.0, 0.0).is_err());
    }

    // ── NurbsCurve2D tests ────────────────────────────

    #[test]
    fn nurbs2d_line_segment() {
        let curve =
            NurbsCurve2D::from_line(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)).expect("valid");

        assert!(point2_approx_eq(curve.evaluate(0.0), Point2::new(0.0, 0.0)));
        assert!(point2_approx_eq(curve.evaluate(1.0), Point2::new(1.0, 1.0)));
        assert!(point2_approx_eq(curve.evaluate(0.5), Point2::new(0.5, 0.5)));
    }

    #[test]
    fn nurbs2d_quadratic_bezier() {
        let curve = NurbsCurve2D::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.5, 1.0),
                Point2::new(1.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0],
        )
        .expect("valid curve");

        assert!(point2_approx_eq(curve.evaluate(0.0), Point2::new(0.0, 0.0)));
        assert!(point2_approx_eq(curve.evaluate(1.0), Point2::new(1.0, 0.0)));
        assert!(point2_approx_eq(curve.evaluate(0.5), Point2::new(0.5, 0.5)));
    }

    #[test]
    fn nurbs2d_tangent() {
        let curve =
            NurbsCurve2D::from_line(Point2::new(0.0, 0.0), Point2::new(2.0, 3.0)).expect("valid");
        let tangent = curve.tangent(0.5);
        assert!(approx_eq(tangent.x(), 2.0));
        assert!(approx_eq(tangent.y(), 3.0));
    }

    #[test]
    fn nurbs2d_domain() {
        let curve = NurbsCurve2D::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.25, 1.0),
                Point2::new(0.75, 1.0),
                Point2::new(1.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid curve");

        let (u_min, u_max) = curve.domain();
        assert!(approx_eq(u_min, 0.0));
        assert!(approx_eq(u_max, 1.0));
    }

    #[test]
    fn nurbs2d_invalid_knots() {
        let result = NurbsCurve2D::new(
            1,
            vec![0.0, 1.0],
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            vec![1.0, 1.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn nurbs2d_invalid_weights() {
        let result = NurbsCurve2D::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            vec![1.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn nurbs2d_is_rational() {
        let non_rational =
            NurbsCurve2D::from_line(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)).expect("valid");
        assert!(!non_rational.is_rational());

        let rational = NurbsCurve2D::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            vec![1.0, 2.0],
        )
        .expect("valid");
        assert!(rational.is_rational());
    }

    // ── Curve2D enum tests ────────────────────────────

    #[test]
    fn curve2d_evaluate_dispatch() {
        let line =
            Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).expect("valid"));
        assert!(point2_approx_eq(line.evaluate(5.0), Point2::new(5.0, 0.0)));

        let circle = Curve2D::Circle(Circle2D::new(Point2::new(0.0, 0.0), 1.0).expect("valid"));
        assert!(point2_approx_eq(
            circle.evaluate(0.0),
            Point2::new(1.0, 0.0)
        ));
    }
}
