//! # brepkit-check
//!
//! Topology algorithms for brepkit B-Rep models.
//!
//! This is layer L1.5, depending on `brepkit-math` and `brepkit-topology`.
//!
//! Four subsystems:
//! - **classify** — point-in-solid classification (ray casting + winding numbers)
//! - **validate** — hierarchical shape validation (geometric + topological checks)
//! - **properties** — geometric properties (volume, area, CoM, inertia tensor)
//! - **distance** — minimum distance and extrema between shapes

pub mod classify;
pub mod distance;
pub mod error;
pub mod properties;
pub(crate) mod util;
pub mod validate;

pub use error::CheckError;
