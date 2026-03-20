//! Shape healing for brepkit B-Rep models.
//!
//! This crate provides OCCT-parity shape healing with three layers:
//!
//! - **Analysis** — pure diagnostic queries that detect problems without
//!   modifying the model.
//! - **Fix** — targeted repairs that correct detected issues, tracked
//!   through a [`ReShape`](reshape::ReShape) context for atomic application.
//! - **Upgrade** — geometry decomposition and merging (e.g.
//!   [`unify_same_domain`](upgrade::unify_same_domain)).
//!
//! Additional modules provide curve/surface construction utilities,
//! custom geometry transformations, and a configurable processing pipeline.
//!
//! # Quick start
//!
//! ```ignore
//! use brepkit_heal::fix::{fix_shape, FixConfig};
//!
//! let config = FixConfig::default();
//! let (new_solid, result) = fix_shape(&mut topo, solid_id, &config)?;
//! ```

pub mod analysis;
pub mod construct;
pub mod context;
pub mod custom;
pub mod error;
pub mod fix;
pub mod pipeline;
pub mod reshape;
pub mod status;
pub mod upgrade;

pub use error::HealError;
pub use status::Status;
