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
}

pub mod mat;
pub mod nurbs;
pub mod predicates;
pub mod tolerance;
pub mod vec;
