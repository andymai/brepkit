//! NURBS surface data structure.

use crate::MathError;
use crate::vec::Point3;

/// A Non-Uniform Rational B-Spline (NURBS) surface in 3D space.
///
/// The surface is defined by degrees in the u and v directions, two knot
/// vectors, a 2D grid of control points, and matching weights.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NurbsSurface {
    /// Polynomial degree in the u direction.
    degree_u: usize,
    /// Polynomial degree in the v direction.
    degree_v: usize,
    /// Knot vector in the u direction.
    knots_u: Vec<f64>,
    /// Knot vector in the v direction.
    knots_v: Vec<f64>,
    /// Control point grid indexed as `control_points[row_u][col_v]`.
    control_points: Vec<Vec<Point3>>,
    /// Weight grid matching `control_points` dimensions.
    weights: Vec<Vec<f64>>,
}

impl NurbsSurface {
    /// Construct a new NURBS surface with validation.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidControlPointGrid`] if the control point
    /// rows have inconsistent lengths.
    ///
    /// Returns [`MathError::InvalidKnotVector`] if either knot vector has the
    /// wrong length for the given degree and control point count.
    ///
    /// Returns [`MathError::InvalidWeights`] if the weights grid dimensions
    /// do not match the control point grid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        control_points: Vec<Vec<Point3>>,
        weights: Vec<Vec<f64>>,
    ) -> Result<Self, MathError> {
        let n_rows = control_points.len();

        // Validate that all rows have the same length.
        let n_cols = control_points.first().map_or(0, Vec::len);
        for row in &control_points {
            if row.len() != n_cols {
                return Err(MathError::InvalidControlPointGrid {
                    expected_rows: n_rows,
                    expected_cols: n_cols,
                });
            }
        }

        // Validate knot vectors.
        let expected_knots_u = n_rows + degree_u + 1;
        if knots_u.len() != expected_knots_u {
            return Err(MathError::InvalidKnotVector {
                expected: expected_knots_u,
                got: knots_u.len(),
            });
        }

        let expected_knots_v = n_cols + degree_v + 1;
        if knots_v.len() != expected_knots_v {
            return Err(MathError::InvalidKnotVector {
                expected: expected_knots_v,
                got: knots_v.len(),
            });
        }

        // Validate weights grid dimensions.
        if weights.len() != n_rows {
            return Err(MathError::InvalidWeights {
                expected: n_rows,
                got: weights.len(),
            });
        }
        for row in &weights {
            if row.len() != n_cols {
                return Err(MathError::InvalidWeights {
                    expected: n_cols,
                    got: row.len(),
                });
            }
        }

        Ok(Self {
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            control_points,
            weights,
        })
    }

    /// Polynomial degree in the u direction.
    #[must_use]
    pub const fn degree_u(&self) -> usize {
        self.degree_u
    }

    /// Polynomial degree in the v direction.
    #[must_use]
    pub const fn degree_v(&self) -> usize {
        self.degree_v
    }

    /// Knot vector in the u direction.
    #[must_use]
    pub fn knots_u(&self) -> &[f64] {
        &self.knots_u
    }

    /// Knot vector in the v direction.
    #[must_use]
    pub fn knots_v(&self) -> &[f64] {
        &self.knots_v
    }

    /// Reference to the control point grid.
    #[must_use]
    pub fn control_points(&self) -> &[Vec<Point3>] {
        &self.control_points
    }

    /// Reference to the weights grid.
    #[must_use]
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }

    /// Evaluate the surface at parameters `(u, v)`.
    ///
    /// Not yet implemented.
    #[must_use]
    pub fn evaluate(&self, _u: f64, _v: f64) -> Point3 {
        todo!()
    }
}
