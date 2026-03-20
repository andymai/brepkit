//! Distance and extrema computation between geometry primitives.
//!
//! # Result types
//!
//! - [`ExtremaSolution`] — general closest-point result between two entities.
//! - [`CurveProjection`] — closest point on a curve from a query point.
//! - [`SurfaceProjection`] — closest point on a surface from a query point.
//!
//! # Algorithms
//!
//! - [`point_curve`] — point-to-curve projection (analytic fast paths + generic
//!   Newton-Raphson fallback).
//! - [`segment`] — segment-to-segment minimum distance.

pub mod point_curve;
pub mod segment;

pub use point_curve::{point_to_circle, point_to_curve, point_to_line};
pub use segment::segment_segment_distance;

use brepkit_math::vec::Point3;

/// Result of a distance/extrema computation between two geometric entities.
#[derive(Debug, Clone, Copy)]
pub struct ExtremaSolution {
    /// Minimum distance found.
    pub distance: f64,
    /// Closest point on entity A.
    pub point_a: Point3,
    /// Closest point on entity B.
    pub point_b: Point3,
    /// Parameter on entity A at closest point.
    pub param_a: f64,
    /// Parameter on entity B at closest point.
    pub param_b: f64,
}

/// Result of projecting a point onto a curve.
#[derive(Debug, Clone, Copy)]
pub struct CurveProjection {
    /// Distance from point to closest point on curve.
    pub distance: f64,
    /// Closest point on the curve.
    pub point: Point3,
    /// Parameter value at closest point.
    pub parameter: f64,
}

/// Result of projecting a point onto a surface.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceProjection {
    /// Distance from point to closest point on surface.
    pub distance: f64,
    /// Closest point on the surface.
    pub point: Point3,
    /// U parameter at closest point.
    pub u: f64,
    /// V parameter at closest point.
    pub v: f64,
}
