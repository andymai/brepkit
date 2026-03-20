//! Shape upgrading — geometry decomposition and merging.
//!
//! The most important operation is [`unify_same_domain`], which merges
//! adjacent faces that share the same underlying surface.

pub mod convert_to_bezier;
pub mod remove_internal_wires;
pub mod shell_sewing;
pub mod split_curve;
pub mod split_surface;
pub mod unify_same_domain;
