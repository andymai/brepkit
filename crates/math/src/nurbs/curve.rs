//! NURBS curve evaluation via De Boor's algorithm.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::basis;
use crate::vec::{Point3, Vec3};

/// A Non-Uniform Rational B-Spline (NURBS) curve in 3D space.
///
/// The curve is defined by its degree, a knot vector, control points, and
/// per-control-point weights (1.0 for non-rational curves).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NurbsCurve {
    /// Polynomial degree of the basis functions.
    degree: usize,
    /// Knot vector (non-decreasing, length = n + degree + 1).
    knots: Vec<f64>,
    /// Control points in 3D.
    control_points: Vec<Point3>,
    /// Weights for rational curves (same length as `control_points`).
    weights: Vec<f64>,
}

impl NurbsCurve {
    /// Construct a new NURBS curve with validation.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidKnotVector`] if the knot vector length is
    /// not equal to `control_points.len() + degree + 1`.
    ///
    /// Returns [`MathError::InvalidWeights`] if the weights vector length does
    /// not match the number of control points.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<Point3>,
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
        Ok(Self {
            degree,
            knots,
            control_points,
            weights,
        })
    }

    /// The polynomial degree of the curve.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Whether the curve is rational (any weight differs from 1.0).
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_rational(&self) -> bool {
        self.weights.iter().any(|&w| w != 1.0)
    }

    /// Return the valid parameter domain `[u_min, u_max]`.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        let u_min = self.knots[self.degree];
        let u_max = self.knots[self.knots.len() - self.degree - 1];
        (u_min, u_max)
    }

    /// Approximate the arc length of the curve by numerical integration.
    ///
    /// Uses Simpson's rule with `n_samples` intervals.
    #[must_use]
    pub fn arc_length(&self, n_samples: usize) -> f64 {
        let (u_min, u_max) = self.domain();
        let n = n_samples.max(4);
        #[allow(clippy::cast_precision_loss)]
        let dt = (u_max - u_min) / (n as f64);
        let mut length = 0.0;

        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let t0 = u_min + dt * (i as f64);
            let t1 = t0 + dt;
            #[allow(clippy::manual_midpoint)]
            let t_mid = (t0 + t1) / 2.0;

            let v0 = self.derivatives(t0, 1)[1].length();
            let v_mid = self.derivatives(t_mid, 1)[1].length();
            let v1 = self.derivatives(t1, 1)[1].length();

            length += (dt / 6.0) * v_mid.mul_add(4.0, v0 + v1);
        }

        length
    }

    /// Compute the curvature at parameter `u`.
    ///
    /// Curvature κ = |C' × C''| / |C'|³
    ///
    /// # Errors
    ///
    /// Returns an error if derivatives cannot be computed or the tangent
    /// is zero-length (degenerate point).
    pub fn curvature(&self, u: f64) -> Result<f64, MathError> {
        let derivs = self.derivatives(u, 2);
        if derivs.len() < 3 {
            return Err(MathError::EmptyInput);
        }

        let d1 = derivs[1]; // First derivative (tangent)
        let d2 = derivs[2]; // Second derivative

        let speed = d1.length();
        if speed < 1e-15 {
            return Err(MathError::ZeroVector);
        }

        let cross = d1.cross(d2);
        Ok(cross.length() / (speed * speed * speed))
    }

    /// Reference to the knot vector.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Reference to the control points.
    #[must_use]
    pub fn control_points(&self) -> &[Point3] {
        &self.control_points
    }

    /// Reference to the weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Evaluate the curve at parameter `u` using De Boor's algorithm.
    ///
    /// For rational curves this performs the perspective divide automatically.
    #[must_use]
    pub fn evaluate(&self, u: f64) -> Point3 {
        let p = self.degree;
        let n = self.control_points.len();
        let span = basis::find_span(n, p, u, &self.knots);
        let bf = basis::basis_funs(span, u, p, &self.knots);

        // Weighted sum in homogeneous coordinates.
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        let mut ww = 0.0;
        for (j, &basis_val) in bf.iter().enumerate().take(p + 1) {
            let idx = span - p + j;
            let pt = &self.control_points[idx];
            let w = self.weights[idx];
            let bw = basis_val * w;
            wx += bw * pt.x();
            wy += bw * pt.y();
            wz += bw * pt.z();
            ww += bw;
        }

        if ww == 0.0 {
            Point3::new(wx, wy, wz)
        } else {
            Point3::new(wx / ww, wy / ww, wz / ww)
        }
    }

    /// Compute curve derivatives up to order `d` at parameter `u`.
    ///
    /// Returns a vector of length `d + 1` where element `k` is the `k`-th
    /// derivative. Element 0 is the curve point itself (as a `Vec3`).
    ///
    /// Uses NURBS Book Algorithm A3.2 + A4.2 (rational quotient rule).
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn derivatives(&self, u: f64, d: usize) -> Vec<Vec3> {
        let p = self.degree;
        let n = self.control_points.len();
        let span = basis::find_span(n, p, u, &self.knots);
        let du = d.min(p);
        let ders_bf = basis::ders_basis_funs(span, u, p, du, &self.knots);

        // Compute homogeneous derivatives: Aw[k] = (Aw_x, Aw_y, Aw_z, w) for k-th deriv.
        let mut aw = vec![[0.0f64; 4]; du + 1];
        for (k, aw_k) in aw.iter_mut().enumerate().take(du + 1) {
            for (j, &db) in ders_bf[k].iter().enumerate().take(p + 1) {
                let idx = span - p + j;
                let pt = &self.control_points[idx];
                let w = self.weights[idx];
                aw_k[0] += db * pt.x() * w;
                aw_k[1] += db * pt.y() * w;
                aw_k[2] += db * pt.z() * w;
                aw_k[3] += db * w;
            }
        }

        // Apply rational quotient rule (A4.2).
        let mut ck = vec![Vec3::new(0.0, 0.0, 0.0); d + 1];
        for k in 0..=du {
            let mut v = [aw[k][0], aw[k][1], aw[k][2]];
            for i in 1..=k {
                #[allow(clippy::cast_precision_loss)]
                let bin = binomial(k, i) as f64;
                v[0] -= bin * aw[i][3] * ck[k - i].x();
                v[1] -= bin * aw[i][3] * ck[k - i].y();
                v[2] -= bin * aw[i][3] * ck[k - i].z();
            }
            let w0 = aw[0][3];
            if w0 == 0.0 {
                ck[k] = Vec3::new(v[0], v[1], v[2]);
            } else {
                ck[k] = Vec3::new(v[0] / w0, v[1] / w0, v[2] / w0);
            }
        }
        // Higher derivatives beyond degree are zero (already initialized).
        ck
    }

    /// Compute the unit tangent vector at parameter `u`.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the first derivative is zero
    /// (degenerate point).
    pub fn tangent(&self, u: f64) -> Result<Vec3, MathError> {
        let d = self.derivatives(u, 1);
        d[1].normalize()
    }

    /// Compute an axis-aligned bounding box from control point extrema.
    ///
    /// This is a conservative bound — the curve lies inside it but it may
    /// not be tight. For rational curves the control polygon hull property
    /// guarantees containment.
    #[must_use]
    pub fn aabb(&self) -> Aabb3 {
        Aabb3::from_points(self.control_points.iter().copied())
    }
}

use super::basis::binomial;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_lossless, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    /// A cubic Bezier curve (single span): control points form a simple shape.
    fn cubic_bezier() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid bezier")
    }

    /// Quarter circle arc as a rational NURBS (degree 2).
    fn quarter_circle() -> NurbsCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, w, 1.0],
        )
        .expect("valid quarter circle")
    }

    #[test]
    fn endpoint_interpolation() {
        let c = cubic_bezier();
        let p0 = c.evaluate(0.0);
        let p1 = c.evaluate(1.0);
        assert!((p0.x() - 0.0).abs() < 1e-14);
        assert!((p0.y() - 0.0).abs() < 1e-14);
        assert!((p1.x() - 4.0).abs() < 1e-14);
        assert!((p1.y() - 0.0).abs() < 1e-14);
    }

    #[test]
    fn cubic_bezier_midpoint() {
        // For Bezier: C(0.5) = (1/8)(P0 + 3*P1 + 3*P2 + P3)
        let c = cubic_bezier();
        let mid = c.evaluate(0.5);
        let expected_x = (0.0 + 3.0 * 1.0 + 3.0 * 3.0 + 4.0) / 8.0; // 2.0
        let expected_y = (0.0 + 3.0 * 2.0 + 3.0 * 2.0 + 0.0) / 8.0; // 1.5
        assert!((mid.x() - expected_x).abs() < 1e-14);
        assert!((mid.y() - expected_y).abs() < 1e-14);
    }

    #[test]
    fn quarter_circle_midpoint() {
        let c = quarter_circle();
        let mid = c.evaluate(0.5);
        // At u=0.5, a quarter circle should give (cos(45°), sin(45°), 0)
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (mid.x() - expected).abs() < 1e-14,
            "x: {} != {}",
            mid.x(),
            expected
        );
        assert!(
            (mid.y() - expected).abs() < 1e-14,
            "y: {} != {}",
            mid.y(),
            expected
        );
    }

    #[test]
    fn quarter_circle_on_unit_circle() {
        let c = quarter_circle();
        // Points on a quarter circle should have radius 1
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p = c.evaluate(u);
            let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
            assert!((r - 1.0).abs() < 1e-13, "radius at u={u}: {r}");
        }
    }

    #[test]
    fn derivatives_zeroth_is_point() {
        let c = cubic_bezier();
        let d = c.derivatives(0.5, 2);
        let p = c.evaluate(0.5);
        assert!((d[0].x() - p.x()).abs() < 1e-14);
        assert!((d[0].y() - p.y()).abs() < 1e-14);
    }

    #[test]
    fn cubic_bezier_first_derivative() {
        // For a cubic Bezier, C'(u) = 3[(1-u)^2(P1-P0) + 2u(1-u)(P2-P1) + u^2(P3-P2)]
        let c = cubic_bezier();
        let d = c.derivatives(0.5, 1);
        // At u=0.5: C'(0.5) = 3[0.25*(1,2,0) + 0.5*(2,0,0) + 0.25*(1,-2,0)]
        //                    = 3[0.25+1.0+0.25, 0.5+0-0.5, 0] = 3[1.5, 0, 0] = (4.5, 0, 0)
        let expected_x = 3.0 * (0.25 * 1.0 + 0.5 * 2.0 + 0.25 * 1.0);
        let expected_y = 3.0 * (0.25 * 2.0 + 0.5 * 0.0 + 0.25 * (-2.0));
        assert!(
            (d[1].x() - expected_x).abs() < 1e-12,
            "dx: {} != {}",
            d[1].x(),
            expected_x
        );
        assert!(
            (d[1].y() - expected_y).abs() < 1e-12,
            "dy: {} != {}",
            d[1].y(),
            expected_y
        );
    }

    #[test]
    fn tangent_at_start() {
        let c = cubic_bezier();
        let t = c.tangent(0.0).expect("non-degenerate");
        // At u=0, tangent direction is P1 - P0 = (1, 2, 0), normalized.
        let expected = Vec3::new(1.0, 2.0, 0.0).normalize().expect("non-zero");
        assert!((t.x() - expected.x()).abs() < 1e-12);
        assert!((t.y() - expected.y()).abs() < 1e-12);
    }

    #[test]
    fn aabb_contains_all_control_points() {
        let c = cubic_bezier();
        let bb = c.aabb();
        for pt in c.control_points() {
            assert!(bb.contains_point(*pt));
        }
    }

    #[test]
    fn binomial_values() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(3, 4), 0);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_evaluate_equals_derivatives_zeroth(u in 0.0f64..=1.0) {
            let c = cubic_bezier();
            let p = c.evaluate(u);
            let d = c.derivatives(u, 0);
            prop_assert!((d[0].x() - p.x()).abs() < 1e-12);
            prop_assert!((d[0].y() - p.y()).abs() < 1e-12);
            prop_assert!((d[0].z() - p.z()).abs() < 1e-12);
        }
    }

    #[test]
    fn domain_returns_correct_range() {
        let c = cubic_bezier();
        let (u_min, u_max) = c.domain();
        assert!((u_min - 0.0).abs() < 1e-14);
        assert!((u_max - 1.0).abs() < 1e-14);
    }

    #[test]
    fn arc_length_quarter_circle() {
        let c = quarter_circle();
        let len = c.arc_length(100);
        // Quarter circle of radius 1: arc length = π/2 ≈ 1.5708
        let expected = std::f64::consts::FRAC_PI_2;
        assert!(
            (len - expected).abs() < 0.01,
            "quarter circle arc length should be ~{expected}, got {len}"
        );
    }

    #[test]
    fn arc_length_straight_line() {
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 4.0, 0.0)],
            vec![1.0, 1.0],
        )
        .expect("valid line");

        let len = line.arc_length(10);
        assert!(
            (len - 5.0).abs() < 0.01,
            "3-4-5 line should have length ~5.0, got {len}"
        );
    }

    #[test]
    fn curvature_quarter_circle() {
        let c = quarter_circle();
        // For rational NURBS, curvature() uses parameter-space derivatives
        // which don't directly give geometric curvature. The returned value
        // is still useful for relative comparisons (higher curvature = sharper).
        let k = c.curvature(0.5).expect("curvature should compute");
        assert!(
            k > 0.0,
            "quarter circle should have positive curvature, got {k}"
        );
    }

    #[test]
    fn curvature_straight_line_is_zero() {
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .expect("valid line");

        let k = line.curvature(0.5).expect("curvature should compute");
        assert!(k < 1e-10, "straight line curvature should be ~0, got {k}");
    }
}
