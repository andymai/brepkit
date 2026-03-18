//! # brepkit-algo
//!
//! General Fuse Algorithm (GFA) engine for boolean operations.
//!
//! This is layer L1.5, depending on `brepkit-math` and `brepkit-topology`.
//! The `brepkit-operations` crate delegates boolean, section, and split
//! operations to this crate.
//!
//! # Architecture
//!
//! The GFA follows OCCT's proven approach:
//!
//! 1. **PaveFiller** — intersects all shape pairs, builds pave blocks
//!    (edge segments at intersection points), and populates face info.
//! 2. **Builder** — splits faces using pave block data, classifies
//!    sub-faces relative to opposing solids, assembles result shells.
//! 3. **BOP** — selects faces based on boolean operation type
//!    (fuse/cut/intersect).

pub mod bop;
pub mod builder;
pub mod classifier;
pub mod ds;
pub mod error;
pub mod gfa;
pub mod pave_filler;
