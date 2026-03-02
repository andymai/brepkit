//! Solid — a volume bounded by closed shells.

use crate::arena;
use crate::shell::ShellId;

/// Typed handle for a [`Solid`] stored in an [`Arena`](crate::Arena).
pub type SolidId = arena::Id<Solid>;

/// A topological solid: a volume bounded by one or more shells.
///
/// The outer shell defines the exterior boundary. Inner shells
/// represent voids (cavities) within the solid.
#[derive(Debug, Clone)]
pub struct Solid {
    /// The outer bounding shell of the solid.
    outer_shell: ShellId,
    /// Inner shells representing voids inside the solid.
    inner_shells: Vec<ShellId>,
}

impl Solid {
    /// Creates a new solid with the given outer shell and optional inner shells.
    #[must_use]
    pub const fn new(outer_shell: ShellId, inner_shells: Vec<ShellId>) -> Self {
        Self {
            outer_shell,
            inner_shells,
        }
    }

    /// Returns the outer bounding shell of this solid.
    #[must_use]
    pub const fn outer_shell(&self) -> ShellId {
        self.outer_shell
    }

    /// Returns the inner shells (voids) of this solid.
    #[must_use]
    pub fn inner_shells(&self) -> &[ShellId] {
        &self.inner_shells
    }
}
