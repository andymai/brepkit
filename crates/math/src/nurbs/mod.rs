//! NURBS curve and surface representations.
//!
//! Non-Uniform Rational B-Spline (NURBS) geometry is the standard
//! representation for free-form curves and surfaces in CAD.

pub mod basis;
pub mod bezier_clip;
pub mod curve;
pub mod decompose;
pub mod fitting;
pub mod intersection;
pub mod knot_ops;
pub mod projection;
pub mod self_intersection;
pub mod surface;
pub mod surface_fitting;

pub use bezier_clip::{CurveCurveHit, curve_curve_intersect};
pub use curve::NurbsCurve;
pub use decompose::{
    BezierPatch, curve_degree_elevate, curve_to_bezier_segments, surface_to_bezier_patches,
};
pub use fitting::{approximate, interpolate};
pub use knot_ops::{
    curve_knot_insert, curve_knot_refine, curve_knot_remove, curve_split, surface_knot_insert_u,
    surface_knot_insert_v,
};
pub use projection::{
    CurveProjection, SurfaceProjection, project_point_to_curve, project_point_to_surface,
};
pub use surface::NurbsSurface;
pub use surface_fitting::interpolate_surface;
