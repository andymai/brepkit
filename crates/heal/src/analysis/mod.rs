//! Shape analysis — pure diagnostic queries.
//!
//! Each analyzer takes `&Topology` (immutable) and returns a result
//! struct describing detected issues.  No modifications are made.

pub mod contents;
pub mod curve;
pub mod edge;
pub mod face;
pub mod free_bounds;
pub mod shell;
pub mod surface;
pub mod tolerance;
pub mod wire;
pub mod wire_order;
