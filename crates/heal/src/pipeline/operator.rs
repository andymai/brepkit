//! Heal operator trait for pipeline steps.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::HealError;
use crate::context::HealContext;

/// A single step in a healing pipeline.
///
/// Each operator transforms a solid and returns the (possibly new)
/// solid ID.  Operators are stateless — all mutable state lives in
/// the [`HealContext`].
pub trait HealOperator: std::fmt::Debug + Send + Sync {
    /// Human-readable name of this operator.
    fn name(&self) -> &'static str;

    /// Execute the operator on a solid.
    ///
    /// # Errors
    ///
    /// Returns [`HealError`] if the operation fails.
    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<SolidId, HealError>;
}
