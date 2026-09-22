//! # brepkit-sketch
//!
//! 2D parametric geometric constraint solver for sketch-mode design.
//!
//! Provides a production-grade GCS (Geometric Constraint System) with:
//! - **Entities**: Points, Lines, Circles with generational arena handles
//! - **Constraints**: 10 constraint types with analytic Jacobians
//! - **Solver**: DogLeg trust-region (globally convergent)
//! - **DOF analysis**: QR-based rank detection
//!
//! This crate has no workspace dependencies. It solves 2D constraint systems
//! and nothing else, so it can be used independently of the rest of brepkit.
//!
//! # Example
//!
//! ```
//! use brepkit_sketch::{Constraint, GcsSystem, PointData};
//!
//! let mut sys = GcsSystem::new();
//! let anchor = sys.add_point(PointData { x: 0.0, y: 0.0, fixed: true });
//! let free = sys.add_point(PointData { x: 5.0, y: 1.0, fixed: false });
//!
//! sys.add_constraint(Constraint::Distance(anchor, free, 3.0))?;
//! let result = sys.solve(100, 1e-10)?;
//!
//! assert!(result.converged);
//! # Ok::<(), brepkit_sketch::SketchError>(())
//! ```
//!
//! # Degrees of freedom
//!
//! A sketch is rarely fully constrained while you are still drawing it, and
//! knowing *how* under-constrained it is turns a failed solve into a useful
//! message. [`GcsSystem::dof`] reports it:
//!
//! ```
//! use brepkit_sketch::{Constraint, GcsSystem, PointData};
//!
//! let mut sys = GcsSystem::new();
//! let anchor = sys.add_point(PointData { x: 0.0, y: 0.0, fixed: true });
//! let free = sys.add_point(PointData { x: 5.0, y: 1.0, fixed: false });
//!
//! // One free point is two parameters. One distance constraint removes one.
//! sys.add_constraint(Constraint::Distance(anchor, free, 3.0))?;
//!
//! let analysis = sys.dof();
//! assert_eq!(analysis.dof, 1); // still free to slide around the circle
//! # Ok::<(), brepkit_sketch::SketchError>(())
//! ```
//!
//! `dof == 0` means fully constrained. A positive value counts the dimensions
//! the geometry can still move in, which is what a UI shows as a draggable
//! entity. The analysis comes from the rank of the Jacobian rather than from
//! counting constraints, so it stays correct when constraints are redundant:
//! adding the same distance twice consumes a constraint slot but no degree of
//! freedom, and `rank` reflects that while `num_equations` does not.
//!
//! # Why DogLeg
//!
//! Constraint systems are solved by driving a residual vector to zero, and
//! plain Newton-Raphson does that only from a good starting guess. Sketches
//! do not provide one: the user drops a rough shape and expects the solver to
//! find the exact configuration from there. DogLeg is a trust-region method,
//! blending the Gauss-Newton step with a gradient-descent step and shrinking
//! the region when a step overshoots. It converges from far worse initial
//! guesses, which is the difference between a solver that works on a real
//! sketch and one that works on a textbook example.
//!
//! # See also
//!
//! - [`brepkit_operations`](https://docs.rs/brepkit-operations): turning a
//!   solved sketch into B-Rep geometry.
//! - [brepjs.dev](https://brepjs.dev/tasks/sketching): the same solver from
//!   the TypeScript side.

mod gcs;

pub use gcs::{
    ArcData, ArcId, CircleData, CircleId, Constraint, ConstraintEntry, ConstraintId, DofAnalysis,
    GcsSystem, LineData, LineId, PointData, PointId, SolveResult,
};

/// Errors from the sketch constraint solver.
#[derive(Debug, thiserror::Error)]
pub enum SketchError {
    /// A GCS entity handle is invalid or stale (entity was removed).
    #[error("invalid or stale GCS entity handle")]
    InvalidHandle,

    /// Cannot remove a GCS entity that is still referenced by other entities or constraints.
    #[error("GCS entity is still in use by other entities or constraints")]
    EntityInUse,
}
