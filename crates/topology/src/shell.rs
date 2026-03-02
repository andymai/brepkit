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

    /// Returns the faces of this shell.
    #[must_use]
    pub fn faces(&self) -> &[FaceId] {
        &self.faces
    }
}
