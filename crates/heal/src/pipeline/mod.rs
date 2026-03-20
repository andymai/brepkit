//! Configurable healing pipeline.
//!
//! Define a sequence of [`HealOperator`] steps and execute them
//! in order on a solid.  Built-in operators cover the 19 OCCT
//! `ShapeProcess` operations.

pub mod builtin;
pub mod operator;
pub mod process;
pub mod registry;
