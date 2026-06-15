//! Surface-surface and curve-surface intersection routines.
//!
//! These are the geometric foundations for boolean operations on NURBS solids.
//!
//! ## Algorithms
//!
//! - **Plane-NURBS**: Sample the NURBS surface on a grid, find sign changes of the
//!   signed distance to the plane, trace zero-crossings via linear interpolation,
//!   then refine with Newton iteration.
//! - **NURBS-NURBS**: Subdivision + marching method in (u1,v1,u2,v2) parameter space.
//! - **Line-surface**: Newton iteration from grid-based seed points.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::manual_let_else
)]

mod chaining;
mod curve_surface;
mod line;
mod plane;
mod surface_marching;
mod surface_seeding;

use crate::nurbs::curve::NurbsCurve;
use crate::vec::Point3;

pub use chaining::chain_intersection_points;
pub use curve_surface::{CurveSurfaceHit, intersect_curve_surface};
pub use line::intersect_line_nurbs;
pub use plane::intersect_plane_nurbs;
pub use surface_seeding::intersect_nurbs_nurbs;

/// Maximum work-queue entries for the branch-aware SSI marcher.
const MAX_QUEUE_SIZE: usize = 100;

/// Maximum traced curve segments before stopping branch exploration.
const MAX_SEGMENTS: usize = 50;

/// Maximum branch points detected per march direction.
const MAX_BRANCHES_PER_DIRECTION: usize = 10;

/// Maximum iterations for Newton-type solvers.
///
/// 20 iterations is sufficient for quadratic convergence from reasonable seeds
/// (quadratic convergence achieves ~1e-12 in ~6 iterations from a 1e-1 seed).
/// The limit is generous to handle near-singular cases where convergence slows.
const MAX_NEWTON_ITER: usize = 20;

/// A point on an intersection curve, with parameter values on both surfaces.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionPoint {
    /// 3D position of the intersection.
    pub point: Point3,
    /// Parameter on the first surface (u1, v1) or the curve parameter.
    pub param1: (f64, f64),
    /// Parameter on the second surface (u2, v2).
    pub param2: (f64, f64),
}

/// Result of a surface-surface intersection: a list of intersection curves.
#[derive(Debug, Clone)]
pub struct IntersectionCurve {
    /// The 3D intersection curve as a NURBS.
    pub curve: NurbsCurve,
    /// Sampled points along the curve with parameter values.
    pub points: Vec<IntersectionPoint>,
}

#[cfg(test)]
mod tests;
