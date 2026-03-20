//! # brepkit-geometry
//!
//! Geometry algorithms for brepkit B-Rep models.
//!
//! This is layer L0.5, depending only on `brepkit-math`.
//!
//! Three subsystems:
//! - **sampling** — adaptive and uniform curve/surface sampling
//! - **extrema** — distance and extrema computation between geometry primitives
//! - **convert** — geometry type conversion (e.g. analytic ↔ NURBS)

pub mod convert;
pub mod error;
pub mod extrema;
pub mod sampling;

pub use error::GeomError;
