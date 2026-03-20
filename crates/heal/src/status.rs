//! Status bitflags for healing operations.
//!
//! Each analysis or fix operation returns a [`Status`] indicating what
//! happened.  The flags follow a three-tier convention: `OK` means
//! nothing was needed, `DONEn` means specific fixes were applied, and
//! `FAILn` means specific fixes could not be applied.

bitflags::bitflags! {
    /// Outcome of a healing operation, encoded as bit flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Status: u32 {
        /// No action was needed — the entity was already valid.
        const OK    = 0x0001;
        /// Primary fix applied.
        const DONE1 = 0x0002;
        /// Secondary fix applied.
        const DONE2 = 0x0004;
        /// Tertiary fix applied.
        const DONE3 = 0x0008;
        /// Fix 4 applied.
        const DONE4 = 0x0010;
        /// Fix 5 applied.
        const DONE5 = 0x0020;
        /// Fix 6 applied.
        const DONE6 = 0x0040;
        /// Fix 7 applied.
        const DONE7 = 0x0080;
        /// Fix 8 applied.
        const DONE8 = 0x0100;
        /// Primary fix failed.
        const FAIL1 = 0x0200;
        /// Secondary fix failed.
        const FAIL2 = 0x0400;
        /// Tertiary fix failed.
        const FAIL3 = 0x0800;
        /// Fix 4 failed.
        const FAIL4 = 0x1000;
        /// Fix 5 failed.
        const FAIL5 = 0x2000;
        /// Fix 6 failed.
        const FAIL6 = 0x4000;
    }
}

/// Mask covering all DONE flags.
const DONE_MASK: u32 = 0x01FE; // DONE1..DONE8
/// Mask covering all FAIL flags.
const FAIL_MASK: u32 = 0x7E00; // FAIL1..FAIL6

impl Status {
    /// True if any DONE flag is set (at least one fix was applied).
    #[must_use]
    pub fn is_done(self) -> bool {
        self.bits() & DONE_MASK != 0
    }

    /// True if any FAIL flag is set (at least one fix could not be applied).
    #[must_use]
    pub fn is_fail(self) -> bool {
        self.bits() & FAIL_MASK != 0
    }

    /// True if OK and no failures.
    #[must_use]
    pub fn is_ok(self) -> bool {
        self.contains(Self::OK) && !self.is_fail()
    }

    /// Combine two statuses (union of flags, clearing OK if any DONE/FAIL).
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let combined = self | other;
        if combined.is_done() || combined.is_fail() {
            combined - Self::OK
        } else {
            combined
        }
    }
}
