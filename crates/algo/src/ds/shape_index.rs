//! Shape reference types for the GFA.

/// Which boolean argument a shape belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    /// Shape from the first argument (solid A).
    A,
    /// Shape from the second argument (solid B).
    B,
}
