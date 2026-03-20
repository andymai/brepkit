//! Healing context: shared state for all fixers.
//!
//! [`HealContext`] carries tolerance settings, the [`ReShape`] tracker,
//! and diagnostic messages.  It is passed through the fix hierarchy
//! (solid → shell → face → wire → edge) so that all fixers share a
//! consistent view of tolerance and accumulated changes.

use brepkit_math::tolerance::Tolerance;

use crate::reshape::ReShape;
use crate::status::Status;

/// Severity of a diagnostic message emitted during healing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageSeverity {
    /// Informational — no action required.
    Info,
    /// Something unexpected but non-fatal.
    Warning,
    /// A fix could not be applied.
    Error,
}

/// Diagnostic message emitted during a healing operation.
#[derive(Debug, Clone)]
pub struct HealMessage {
    /// Severity level.
    pub severity: MessageSeverity,
    /// Human-readable description.
    pub description: String,
    /// Status flags at the time of the message.
    pub status: Status,
}

/// Shared state for healing operations.
///
/// Instead of a base class with inherited state, every fixer receives
/// this context by mutable reference.
#[derive(Debug)]
pub struct HealContext {
    /// Working tolerance for geometric comparisons.
    pub tolerance: Tolerance,
    /// Upper bound: tolerance will never be grown beyond this.
    pub max_tolerance: f64,
    /// Lower bound: tolerance will never be shrunk below this.
    pub min_tolerance: f64,
    /// Entity replacement/removal tracker.
    pub reshape: ReShape,
    /// Diagnostic messages accumulated during healing.
    pub messages: Vec<HealMessage>,
}

impl HealContext {
    /// Create a context with default tolerances.
    #[must_use]
    pub fn new() -> Self {
        let tol = Tolerance::new();
        Self {
            tolerance: tol,
            max_tolerance: 1.0,   // 1 mm — generous upper bound
            min_tolerance: 1e-10, // sub-nanometre
            reshape: ReShape::new(),
            messages: Vec::new(),
        }
    }

    /// Create a context with a specific linear tolerance.
    #[must_use]
    pub fn with_tolerance(linear: f64) -> Self {
        let mut ctx = Self::new();
        ctx.tolerance.linear = linear;
        ctx
    }

    /// Clamp a tolerance value to the `[min_tolerance, max_tolerance]` range.
    #[must_use]
    pub fn limit_tolerance(&self, tol: f64) -> f64 {
        tol.clamp(self.min_tolerance, self.max_tolerance)
    }

    /// Record a diagnostic message.
    pub fn send_message(&mut self, severity: MessageSeverity, description: String, status: Status) {
        self.messages.push(HealMessage {
            severity,
            description,
            status,
        });
    }

    /// Record an info message.
    pub fn info(&mut self, description: String) {
        self.send_message(MessageSeverity::Info, description, Status::OK);
    }

    /// Record a warning message.
    pub fn warn(&mut self, description: String) {
        self.send_message(MessageSeverity::Warning, description, Status::OK);
    }

    /// Record an error message.
    pub fn error(&mut self, description: String, status: Status) {
        self.send_message(MessageSeverity::Error, description, status);
    }
}

impl Default for HealContext {
    fn default() -> Self {
        Self::new()
    }
}
