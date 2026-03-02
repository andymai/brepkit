//! NURBS curve evaluation via De Boor's algorithm.

use crate::MathError;
use crate::vec::Point3;

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

    /// Whether the curve is rational (any weight differs from 1.0).
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_rational(&self) -> bool {
        self.weights.iter().any(|&w| w != 1.0)
    }

    /// Polynomial degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
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

    /// Find the knot span index for parameter `u`.
    ///
    /// Returns the index `i` such that `knots[i] <= u < knots[i+1]`,
    /// clamped to the valid range for evaluation.
    fn find_span(&self, u: f64) -> usize {
        let n = self.control_points.len();
        let p = self.degree;

        // Clamp to the upper end of the parameter domain.
        if u >= self.knots[n] {
            return n - 1;
        }
        // Clamp to the lower end.
        if u <= self.knots[p] {
            return p;
        }

        // Binary search for the span.
        let mut low = p;
        let mut high = n;
        let mut mid = usize::midpoint(low, high);
        while u < self.knots[mid] || u >= self.knots[mid + 1] {
            if u < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = usize::midpoint(low, high);
        }
        mid
    }

    /// Evaluate the curve at parameter `u` using De Boor's algorithm.
    ///
    /// For rational curves this performs the perspective divide automatically.
    #[must_use]
    pub fn evaluate(&self, u: f64) -> Point3 {
        let p = self.degree;
        let span = self.find_span(u);

        // Build initial homogeneous (weighted) control points for the
        // relevant span: indices span-p ..= span.
        // Store as [wx, wy, wz, w].
        let mut d: Vec<[f64; 4]> = (0..=p)
            .map(|j| {
                let idx = span - p + j;
                let pt = &self.control_points[idx];
                let w = self.weights[idx];
                [pt.x() * w, pt.y() * w, pt.z() * w, w]
            })
            .collect();

        // De Boor triangular computation.
        for r in 1..=p {
            for j in (r..=p).rev() {
                let left = span + j - p;
                let right = span + 1 + j - r;
                let denom = self.knots[right] - self.knots[left];

                // If the knot interval is zero, alpha is 0 (repeated knots).
                let alpha = if denom == 0.0 {
                    0.0
                } else {
                    (u - self.knots[left]) / denom
                };

                let one_minus_alpha = 1.0 - alpha;
                let prev = d[j - 1];
                for (dk, pk) in d[j].iter_mut().zip(prev.iter()) {
                    *dk = one_minus_alpha.mul_add(*pk, alpha * *dk);
                }
            }
        }

        // Perspective divide for rational curves.
        let w = d[p][3];
        if w == 0.0 {
            // Degenerate case: return the un-divided point.
            Point3::new(d[p][0], d[p][1], d[p][2])
        } else {
            Point3::new(d[p][0] / w, d[p][1] / w, d[p][2] / w)
        }
    }
}
