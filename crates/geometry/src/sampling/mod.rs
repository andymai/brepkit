//! Adaptive and uniform curve/surface sampling.
//!
//! # Uniform sampling
//!
//! [`sample_uniform`] and [`sample_uniform_with_params`] divide a parameter
//! range into equal-size steps and evaluate the curve at each step. This is
//! fast but may under-sample highly curved regions.
//!
//! # Deflection-based (adaptive) sampling
//!
//! [`sample_deflection`] uses recursive midpoint subdivision: it measures the
//! perpendicular distance from the true curve point at each interval midpoint
//! to the straight chord between the interval endpoints. If that distance
//! (the "sag") exceeds `max_deflection`, the interval is split in two. This
//! guarantees that every chord's midpoint deviation is within the requested
//! tolerance.
//!
//! # Arc-length parameterized sampling
//!
//! [`sample_arc_length`] places `n` points at approximately equal arc-length
//! spacing. It builds a fine chord-length table (256 segments) and bisects to
//! find the parameter at each target fraction.
//!
//! # Curvature-adaptive sampling
//!
//! [`sample_curvature`] subdivides intervals where the product of curvature
//! and interval arc-length exceeds a tolerance. Produces denser samples in
//! high-curvature regions of a NURBS curve.
//!
//! # Surface grid sampling
//!
//! [`surface_grid`] evaluates a regular N×M grid of points over a parametric
//! surface domain.
//!
//! # Utility
//!
//! [`segments_for_chord_deviation`] (re-exported from `brepkit_math::chord`)
//! computes the number of segments needed to discretize a circular arc of
//! known radius within a given chord-height tolerance. Useful for pre-sizing
//! uniform samples on known-curvature geometry.

pub mod arc_length;
pub mod curvature;
pub mod deflection;
pub mod surface;
pub mod uniform;

pub use arc_length::sample_arc_length;
pub use brepkit_math::chord::segments_for_chord_deviation;
pub use curvature::sample_curvature;
pub use deflection::sample_deflection;
pub use surface::surface_grid;
pub use uniform::{sample_uniform, sample_uniform_with_params};
