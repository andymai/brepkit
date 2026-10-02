//! Classification types for sub-faces in the boolean result.

/// Classification of a sub-face relative to the opposing solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceClass {
    /// Inside the opposing solid.
    Inside,
    /// Outside the opposing solid.
    Outside,
    /// On the boundary — coincident with the opposing solid's boundary, whose
    /// material lies behind it as the sub-face's own does.
    CoplanarSame,
    /// On the boundary — coincident with the opposing solid's boundary, whose
    /// material lies in front of it (the two solids abut there).
    CoplanarOpposite,
    /// On the boundary of the opposing solid — within geometric tolerance.
    /// Used for faces that touch the opposing solid's surface without
    /// crossing it.
    On,
    /// Classification not yet determined.
    Unknown,
}
