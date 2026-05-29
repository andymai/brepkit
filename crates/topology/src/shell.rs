//! Shell — a connected set of faces forming a surface boundary.

use crate::TopologyError;
use crate::arena;
use crate::face::FaceId;

/// Typed handle for a [`Shell`] stored in an [`Arena`](crate::Arena).
pub type ShellId = arena::Id<Shell>;

/// A topological shell: a connected set of faces.
///
/// A closed shell bounds a volume (solid). An open shell represents
/// a sheet or partial boundary.
#[derive(Debug, Clone)]
pub struct Shell {
    /// The faces that make up this shell.
    faces: Vec<FaceId>,
}

impl Shell {
    /// Creates a new shell from a non-empty list of faces.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::Empty`] if `faces` is empty.
    pub fn new(faces: Vec<FaceId>) -> Result<Self, TopologyError> {
        if faces.is_empty() {
            return Err(TopologyError::Empty { entity: "shell" });
        }
        Ok(Self { faces })
    }

    /// Creates a faceless shell backing an empty-result sentinel.
    ///
    /// A regular shell rejects an empty face list because an ordinary
    /// surface boundary must enclose something. The empty shell exists
    /// only so a boolean whose algebraic outcome is the empty set
    /// (e.g. the intersection of disjoint solids) can be represented as
    /// a valid, queryable solid handle reporting zero faces and zero
    /// volume — distinct from a malformed-input error.
    #[must_use]
    pub const fn empty() -> Self {
        Self { faces: Vec::new() }
    }

    /// Returns `true` when this shell has no faces (the empty-result
    /// sentinel — see [`Shell::empty`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Returns the faces of this shell.
    #[must_use]
    pub fn faces(&self) -> &[FaceId] {
        &self.faces
    }

    /// Returns mutable access to the faces of this shell.
    ///
    /// Allows in-place mutation (reorder, replace) but not removal.
    /// The shell must always contain at least one face.
    pub fn faces_mut(&mut self) -> &mut [FaceId] {
        &mut self.faces
    }
}
