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
//! # Utility
//!
//! [`segments_for_chord_deviation`] (re-exported from `brepkit_math::chord`)
//! computes the number of segments needed to discretize a circular arc of
//! known radius within a given chord-height tolerance. Useful for pre-sizing
//! uniform samples on known-curvature geometry.

pub mod deflection;
pub mod uniform;

pub use brepkit_math::chord::segments_for_chord_deviation;
pub use deflection::sample_deflection;
pub use uniform::{sample_uniform, sample_uniform_with_params};
