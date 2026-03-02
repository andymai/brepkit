//! Compound — a collection of solids treated as a single entity.

use crate::arena;
use crate::solid::SolidId;

/// Typed handle for a [`Compound`] stored in an [`Arena`](crate::Arena).
pub type CompoundId = arena::Id<Compound>;

/// A topological compound: a group of solids.
///
/// Compounds allow treating multiple disjoint solids as a single
/// modelling entity (e.g. the result of a boolean split).
#[derive(Debug, Clone)]
pub struct Compound {
    /// The solids contained in this compound.
    solids: Vec<SolidId>,
}

impl Compound {
    /// Creates a new compound from the given solids.
    #[must_use]
    pub const fn new(solids: Vec<SolidId>) -> Self {
        Self { solids }
    }

    /// Returns the solids in this compound.
    #[must_use]
    pub fn solids(&self) -> &[SolidId] {
        &self.solids
    }
}
