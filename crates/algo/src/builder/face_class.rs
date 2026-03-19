//! Classification types for sub-faces in the boolean result.

/// Classification of a sub-face relative to the opposing solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceClass {
    /// Inside the opposing solid.
    Inside,
    /// Outside the opposing solid.
    Outside,
    /// On the boundary — coplanar with an opposing face, same normal direction.
    /// Assigned by future same-domain detection pass.
    #[allow(dead_code)]
    CoplanarSame,
    /// On the boundary — coplanar with an opposing face, opposite normal direction.
    /// Assigned by future same-domain detection pass.
    #[allow(dead_code)]
    CoplanarOpposite,
    /// Classification not yet determined.
    Unknown,
}
