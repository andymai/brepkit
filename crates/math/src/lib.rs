//! # brepkit-math
//!
//! Vector math, matrix transforms, NURBS geometry, and exact geometric
//! predicates for the brepkit CAD kernel.
//!
//! This is the foundation layer (L0) with no workspace dependencies.

/// Errors from math operations.
#[derive(Debug, thiserror::Error)]
pub enum MathError {
    /// Knot vector length does not match control points and degree.
    #[error("invalid knot vector: expected {expected} knots, got {got}")]
    InvalidKnotVector {
        /// Expected number of knots.
        expected: usize,
        /// Actual number of knots.
        got: usize,
    },

    /// Weights vector length does not match control points.
    #[error("invalid weights: expected {expected} weights, got {got}")]
    InvalidWeights {
        /// Expected number of weights.
        expected: usize,
        /// Actual number of weights.
        got: usize,
    },

    /// Control point grid dimensions are inconsistent.
    #[error(
        "invalid control point grid: expected {expected_rows}x{expected_cols}, got inconsistent dimensions"
    )]
    InvalidControlPointGrid {
        /// Expected number of rows.
        expected_rows: usize,
        /// Expected number of columns.
        expected_cols: usize,
    },

    /// Cannot normalize a zero-length vector.
    #[error("cannot normalize zero vector")]
    ZeroVector,

    /// Matrix is singular and cannot be inverted.
    #[error("singular matrix cannot be inverted")]
    SingularMatrix,

    /// Input collection is empty where at least one element is required.
    #[error("empty input where at least one element is required")]
    EmptyInput,

    /// Parameter is outside the valid range.
    #[error("parameter {value} out of range [{min}, {max}]")]
    ParameterOutOfRange {
        /// The out-of-range value.
        value: f64,
        /// Lower bound of the valid range.
        min: f64,
        /// Upper bound of the valid range.
        max: f64,
    },

    /// Newton iteration did not converge within the allowed iterations.
    #[error("Newton iteration did not converge after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
    },
}

pub mod aabb;
pub mod analytic_intersection;
pub mod bvh;
pub mod cdt;
pub mod chord;
pub mod convex_hull;
pub mod curves;
pub mod curves2d;
pub mod filtered;
pub mod frame;
pub mod mat;
pub mod nurbs;
pub mod obb;
pub mod plane;
pub mod polygon2d;
pub mod polygon_offset;
pub mod predicates;
pub mod quadrature;
pub mod ray_triangle;
pub mod surfaces;
pub mod tolerance;
pub mod traits;
pub mod vec;

#[cfg(feature = "simd")]
pub mod simd;
