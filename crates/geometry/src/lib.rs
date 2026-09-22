//! # brepkit-geometry
//!
//! Curve and surface sampling, extrema, and analytic/NURBS conversion.
//! Layer L1, depending only on `brepkit-math`.
//!
//! Three subsystems:
//!
//! - [`sampling`]: uniform, deflection-adaptive, arc-length-uniform, and
//!   curvature-adaptive curve sampling, plus surface grids.
//! - [`extrema`]: point-to-curve projection, curve-to-curve and
//!   point-to-surface distance, segment-segment distance, and a Lipschitz
//!   global optimizer.
//! - [`convert`]: analytic geometry to NURBS, and recognition of NURBS back
//!   into analytic curves and surfaces.
//!
//! # Stability
//!
//! This crate is internal. It is published so that `brepkit-operations`
//! resolves from crates.io, not because its API is meant to be called
//! directly. Depend on `brepkit-operations` instead and expect breakage here
//! on any release.
//!
//! # Example
//!
//! Adaptive sampling refines only where the curve bends, so a chord never
//! deviates from the true curve by more than the requested deflection.
//!
//! ```
//! use brepkit_geometry::sampling::deflection::sample_deflection;
//! use brepkit_math::curves::Circle3D;
//! use brepkit_math::vec::{Point3, Vec3};
//!
//! let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0)?;
//!
//! let coarse = sample_deflection(&circle, 0.0, std::f64::consts::TAU, 1.0);
//! let fine = sample_deflection(&circle, 0.0, std::f64::consts::TAU, 0.01);
//!
//! // A tighter deflection budget buys more points on the same arc.
//! assert!(fine.len() > coarse.len());
//! # Ok::<(), brepkit_math::MathError>(())
//! ```
//!
//! # Choosing a sampler
//!
//! The four curve samplers answer different questions, and picking by habit
//! rather than by need is a common source of either ugly output or wasted
//! points:
//!
//! | Sampler | Spaces points by | Use when |
//! |---------|------------------|----------|
//! | [`sampling::uniform`] | parameter | You need a fixed count, or the curve is a line |
//! | [`sampling::deflection`] | chord error | You are tessellating, and want a geometric accuracy guarantee |
//! | [`sampling::arc_length`] | distance along the curve | Points must be evenly spaced in space, as for a sweep or a dashed line |
//! | [`sampling::curvature`] | local curvature | A NURBS curve has tight and flat regions and you want detail only where it bends |
//!
//! Uniform parameter spacing is not uniform spatial spacing. On a NURBS curve
//! with a non-uniform knot vector the two diverge sharply, which is why a
//! sweep built on `uniform` can bunch its sections at one end.
//!
//! # See also
//!
//! - [`brepkit_math`](https://docs.rs/brepkit-math): the curves and surfaces
//!   these algorithms operate on.
//! - [`brepkit_operations`](https://docs.rs/brepkit-operations): the crate to
//!   depend on instead of this one.

pub mod convert;
pub mod error;
pub mod extrema;
pub mod sampling;

pub use error::GeomError;
