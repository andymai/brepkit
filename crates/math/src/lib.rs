//! # brepkit-math
//!
//! Vectors, matrices, NURBS, analytic curves and surfaces, and exact
//! geometric predicates. Layer L0 of the brepkit CAD kernel, with no
//! workspace dependencies.
//!
//! # What is here
//!
//! | Area | Modules |
//! |------|---------|
//! | Linear algebra | [`mod@vec`], [`mat`], [`plane`], [`frame`] |
//! | NURBS | [`nurbs`] (evaluation, knot operations, fitting, projection, intersection) |
//! | Analytic curves | [`curves`], [`curves2d`] |
//! | Analytic surfaces | [`surfaces`], [`analytic_intersection`] |
//! | Robustness | [`tolerance`], [`predicates`], [`filtered`] |
//! | Spatial structures | [`aabb`], [`obb`], [`bvh`], [`cdt`], [`convex_hull`] |
//! | 2D polygons | [`polygon2d`], [`polygon_offset`] |
//!
//! # Points are not vectors
//!
//! [`Point3`](vec::Point3) (a position) and [`Vec3`](vec::Vec3) (a direction)
//! are separate types rather than one three-float struct. Subtracting two
//! points gives a vector; adding a vector to a point gives a point; adding two
//! points is not defined. The distinction is load-bearing under transforms: a
//! [`Mat4`](mat::Mat4) translates a point but must not translate a direction,
//! and conflating the two is a classic source of silently wrong normals.
//!
//! # The tolerance model
//!
//! Floating-point coordinates never compare equal in the way geometry needs.
//! A wire that closes to within a billionth of a millimetre has closed, and a
//! kernel that insists on bit equality will reject every real model. So
//! measured comparisons, distances, angles, and coordinates go through
//! [`Tolerance`](tolerance::Tolerance), which bundles three thresholds.
//! Orientation decisions are the deliberate exception, covered below.
//!
//! | Field | Default | Meaning |
//! |-------|---------|---------|
//! | `linear` | `1e-7` | Distance below which two points are the same point |
//! | `angular` | `1e-12` rad | Angle below which two directions are parallel |
//! | `relative` | `1e-10` | Fraction of the larger operand, for scale-aware comparison |
//!
//! Three presets are provided. [`Tolerance::new`](tolerance::Tolerance::new)
//! is the CAD default above. [`loose`](tolerance::Tolerance::loose)
//! (`1e-4`/`1e-8`/`1e-6`) suits visualization and rough checks.
//! [`tight`](tolerance::Tolerance::tight) (`1e-10`/`1e-15`/`1e-14`) suits
//! high-precision work, at the cost of rejecting geometry that a looser
//! setting would accept.
//!
//! ## Scale-aware by default
//!
//! [`approx_eq`](tolerance::Tolerance::approx_eq) is not a plain epsilon
//! compare. It returns true when
//!
//! ```text
//! |a - b| <= max(linear, relative * max(|a|, |b|))
//! ```
//!
//! The relative term is what keeps the comparison meaningful at any
//! magnitude. Two coordinates near `1e6` differ by more than `1e-7` purely
//! from rounding, and an absolute-only test would call them distinct forever.
//!
//! That scaling is wrong for quantities that are not coordinates. A dot
//! product, a determinant, or anything already normalized should use
//! [`approx_eq_abs`](tolerance::Tolerance::approx_eq_abs), which compares
//! against `linear` alone. Reaching for `approx_eq` on a near-zero dot product
//! works, because the relative term vanishes, but on a large one it silently
//! widens the threshold.
//!
//! ## When exactness is required
//!
//! Some decisions cannot be tolerance-based at all. Whether a point is left
//! of a line, or above a plane, has to be consistent across every call or the
//! algorithm built on it will contradict itself and produce a non-manifold
//! result. The [`predicates`] module provides filtered exact orientation
//! tests ([`orient2d`](predicates::orient2d),
//! [`orient3d`](predicates::orient3d)) that compute in floating point, check
//! whether the error bound admits the answer, and fall back to exact
//! arithmetic only when it does not. They are fast in the common case and
//! never wrong in the degenerate one.
//!
//! ## When tolerance bites
//!
//! Two situations account for most tolerance trouble:
//!
//! - **Geometry far from the origin.** Doubles carry roughly 15 significant
//!   digits. Near a coordinate of `1e7` the gap between representable values
//!   is about `1.9e-9`, so a `1e-7` linear tolerance sits only some 50 times
//!   above the noise floor. Booleans on far-flung parts lose precision well
//!   before they lose correctness. Translate the part near the origin,
//!   operate, and translate back.
//! - **Units much smaller than a millimetre.** The defaults assume millimetre
//!   scale. Modelling in micrometres makes `1e-7` of your unit a distance
//!   the kernel cannot resolve, and distinct points start merging. Model in
//!   millimetres and scale at export.
//!
//! As a rule, keep coordinates roughly within `1e0` to `1e4` in your chosen
//! units and the defaults take care of themselves.
//!
//! # Analytic first, NURBS as the general case
//!
//! Curves and surfaces are enums, not one universal representation.
//! [`Circle3D`](curves::Circle3D) is a circle, not a rational B-spline that
//! happens to be circular. Analytic types get closed-form intersections where
//! a pair admits one (see [`analytic_intersection`]), which is both faster and
//! exact. NURBS is what everything can convert into and what free-form
//! geometry uses: the fallback, not the default.
//!
//! # Example
//!
//! ```
//! use brepkit_math::curves::Circle3D;
//! use brepkit_math::tolerance::Tolerance;
//! use brepkit_math::vec::{Point3, Vec3};
//!
//! let center = Point3::new(0.0, 0.0, 0.0);
//! let circle = Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), 2.0)?;
//!
//! let tol = Tolerance::new();
//! let start = circle.evaluate(0.0);
//! let quarter = circle.evaluate(std::f64::consts::FRAC_PI_2);
//!
//! // Every point sits one radius from the center, in the plane the normal
//! // defines, and a quarter turn is a right angle.
//! assert!(tol.approx_eq((quarter - center).length(), 2.0));
//! assert!(tol.approx_eq_abs(quarter.z(), 0.0));
//! assert!(tol.approx_eq_abs((start - center).dot(quarter - center), 0.0));
//!
//! // Which direction `evaluate(0.0)` points is set by the frame derived from
//! // the normal. Use `Circle3D::new_with_ref` when the seam position matters.
//! # Ok::<(), brepkit_math::MathError>(())
//! ```
//!
//! # See also
//!
//! - [`brepkit_topology`](https://docs.rs/brepkit-topology): the B-Rep
//!   structures these types give shape to.
//! - [`brepkit_operations`](https://docs.rs/brepkit-operations): the modeling
//!   operations most projects call instead of this crate directly.
//! - [brepjs.dev](https://brepjs.dev/concepts/tolerance): the same tolerance
//!   model from the TypeScript side, with guidance on when to heal.

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

    /// Matrix is singular and cannot be inverted.
    #[error("singular matrix cannot be inverted")]
    SingularMatrix,

    /// Input collection is empty where at least one element is required.
    #[error("empty input where at least one element is required")]
    EmptyInput,

    /// Parameter is outside the valid range.
    #[error("parameter {value} out of range [{min}, {max}]")]
    ParameterOutOfRange {
        /// The out-of-range value.
        value: f64,
        /// Lower bound of the valid range.
        min: f64,
        /// Upper bound of the valid range.
        max: f64,
    },

    /// Newton iteration did not converge within the allowed iterations.
    #[error("Newton iteration did not converge after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
    },
}

pub mod aabb;
pub mod analytic_intersection;
pub mod bvh;
pub mod cdt;
pub mod chord;
pub mod convex_hull;
pub mod curves;
pub mod curves2d;
pub mod det_hash;
pub mod filtered;
pub mod frame;
pub mod mat;
pub mod nurbs;
pub mod obb;
pub mod plane;
pub mod polygon2d;
pub mod polygon_boolean;
pub mod polygon_offset;
pub mod predicates;
pub mod quadrature;
pub mod ray_triangle;
pub mod surfaces;
pub mod tolerance;
pub mod traits;
pub mod vec;

#[cfg(feature = "simd")]
pub mod simd;
