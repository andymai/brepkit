//! Multiply then add, as two rounded operations.

/// `self * a + b`, rounded after the product and again after the sum.
///
/// [`f64::mul_add`] rounds once, which takes a fused multiply-add
/// instruction. WebAssembly has none, and neither has x86-64 at its
/// baseline, so there the call goes out to a software routine that costs
/// several times the two operations it stands in for: more than half the
/// running time of a boolean in the wasm build. Nothing in the kernel
/// needs the single rounding, so its arithmetic uses this instead, and
/// gives the same bits on every target.
pub trait Madd {
    /// `self * a + b`.
    #[must_use]
    fn madd(self, a: Self, b: Self) -> Self;
}

impl Madd for f64 {
    #[inline]
    fn madd(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

impl Madd for f32 {
    #[inline]
    fn madd(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}
