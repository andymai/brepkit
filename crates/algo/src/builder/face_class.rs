//! Classification types for sub-faces in the boolean result.

/// Classification of a sub-face relative to the opposing solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceClass {
    /// Inside the opposing solid.
    Inside,
    /// Outside the opposing solid.
    Outside,
    /// On the boundary — coplanar with an opposing face, same normal direction.
    /// Assigned by the same-domain detection pass.
    CoplanarSame,
    /// On the boundary — coplanar with an opposing face, opposite normal direction.
    /// Assigned by the same-domain detection pass.
    CoplanarOpposite,
    /// On the boundary of the opposing solid — within geometric tolerance.
    /// Used for faces that touch the opposing solid's surface without
    /// crossing it.
    On,
    /// Classification not yet determined.
    Unknown,
}
