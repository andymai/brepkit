//! Geometry type conversion (e.g. analytic curves/surfaces to NURBS and back).
//!
//! # Modules
//!
//! - [`curve_to_nurbs`] — convert [`Circle3D`], [`Ellipse3D`], and line segments
//!   to exact or near-exact rational NURBS curves.
//! - [`surface_to_nurbs`] — convert analytic surfaces (cylinder, sphere, cone,
//!   torus) to NURBS surfaces.
//! - [`recognize_curve`] — identify a NURBS curve as a line or circle arc.
//! - [`recognize_surface`] — identify a NURBS surface as a plane, cylinder, or
//!   sphere.
//!
//! [`Circle3D`]: brepkit_math::curves::Circle3D
//! [`Ellipse3D`]: brepkit_math::curves::Ellipse3D

pub mod curve_to_nurbs;
pub mod recognize_curve;
pub mod recognize_surface;
pub mod surface_to_nurbs;

pub use curve_to_nurbs::{circle_to_nurbs, ellipse_to_nurbs, line_to_nurbs};
pub use recognize_curve::{RecognizedCurve, recognize_curve};
pub use recognize_surface::{RecognizedSurface, recognize_surface};
pub use surface_to_nurbs::{cone_to_nurbs, cylinder_to_nurbs, sphere_to_nurbs, torus_to_nurbs};
