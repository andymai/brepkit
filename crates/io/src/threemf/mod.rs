//! 3MF data exchange.

pub mod reader;
pub mod writer;

pub use reader::{read_threemf, read_threemf_solid};
pub use writer::write_threemf;
