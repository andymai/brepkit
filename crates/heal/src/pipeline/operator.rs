//! Heal operator trait for pipeline steps.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::HealError;
use crate::context::HealContext;
use crate::fix::FixResult;

/// A single step in a healing pipeline.
///
/// Each operator transforms a solid and returns the (possibly new)
/// solid ID along with a [`FixResult`] describing what was done.
/// Operators are stateless — all mutable state lives in the
/// [`HealContext`].
///
/// Returning a [`FixResult`] (rather than just the `SolidId`) lets
/// the pipeline driver report accurate status: an operator that
/// mutates topology in place (e.g. via `ReShape` recording, returning
/// the same `SolidId`) can still surface non-zero `actions_taken`
/// instead of being silently classified as a no-op.
pub trait HealOperator: std::fmt::Debug + Send + Sync {
    /// Human-readable name of this operator.
    fn name(&self) -> &'static str;

    /// Execute the operator on a solid.
    ///
    /// Returns the (possibly updated) `SolidId` and a [`FixResult`]
    /// describing how much work was actually done.
    ///
    /// # Errors
    ///
    /// Returns [`HealError`] if the operation fails.
    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError>;
}
