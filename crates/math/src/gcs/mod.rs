//! Geometric Constraint Solver (GCS) for 2D sketch parametric design.
//!
//! A production-grade solver following the architecture of FreeCAD's PlaneGCS:
//! - **Entities**: Points, Lines, Circles with generational arena handles
//! - **Constraints**: 10 constraint types with analytic Jacobians
//! - **Solver**: DogLeg trust-region (globally convergent)
//! - **DOF analysis**: QR-based rank detection
//! - **CRUD**: Full add/remove/modify with stale-handle detection
//!
//! # Example
//! ```
//! use brepkit_math::gcs::{GcsSystem, PointData, Constraint};
//!
//! let mut sys = GcsSystem::new();
//! let p0 = sys.add_point(PointData { x: 0.0, y: 0.0, fixed: true });
//! let p1 = sys.add_point(PointData { x: 5.0, y: 1.0, fixed: false });
//! sys.add_constraint(Constraint::Distance(p0, p1, 3.0)).unwrap();
//! let result = sys.solve(100, 1e-10).unwrap();
//! assert!(result.converged);
//! ```

mod constraint;
mod dof;
mod entity;
mod qr;
mod solver;
mod system;

pub use constraint::{Constraint, ConstraintEntry, ConstraintId};
pub use dof::DofAnalysis;
pub use entity::{CircleData, CircleId, LineData, LineId, PointData, PointId};
pub use solver::SolveResult;
pub use system::GcsSystem;
