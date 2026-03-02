//! NURBS curve and surface representations.
//!
//! Non-Uniform Rational B-Spline (NURBS) geometry is the standard
//! representation for free-form curves and surfaces in CAD.

pub mod basis;
pub mod curve;
pub mod decompose;
pub mod fitting;
pub mod knot_ops;
pub mod projection;
pub mod surface;

pub use curve::NurbsCurve;
pub use decompose::{curve_degree_elevate, curve_to_bezier_segments};
pub use fitting::{approximate, interpolate};
pub use knot_ops::{
    curve_knot_insert, curve_knot_refine, curve_split, surface_knot_insert_u, surface_knot_insert_v,
};
pub use projection::{
    CurveProjection, SurfaceProjection, project_point_to_curve, project_point_to_surface,
};
pub use surface::NurbsSurface;
