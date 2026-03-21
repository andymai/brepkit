//! Geometric Constraint Solver (GCS) for 2D sketch parametric design.

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
