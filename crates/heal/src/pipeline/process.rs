//! Configurable healing pipeline.
//!
//! [`HealProcess`] executes a sequence of named operators in order.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use super::builtin::register_builtins;
use super::registry::OperatorRegistry;
use crate::HealError;
use crate::context::HealContext;
use crate::fix::FixResult;

/// A configurable sequence of healing operators.
#[derive(Debug)]
pub struct HealProcess {
    steps: Vec<String>,
    registry: OperatorRegistry,
}

impl HealProcess {
    /// Create a new pipeline with all built-in operators registered.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = OperatorRegistry::new();
        register_builtins(&mut registry);
        Self {
            steps: Vec::new(),
            registry,
        }
    }

    /// Add a step to the pipeline by operator name.
    pub fn add_step(&mut self, operator_name: &str) {
        self.steps.push(operator_name.to_string());
    }

    /// Get a reference to the operator registry for custom registration.
    pub fn registry_mut(&mut self) -> &mut OperatorRegistry {
        &mut self.registry
    }

    /// Execute all steps in sequence on the given solid.
    ///
    /// Returns the final solid ID and a [`FixResult`] per step.
    ///
    /// # Errors
    ///
    /// Returns [`HealError`] if any operator fails or a step name is unknown.
    pub fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(SolidId, Vec<FixResult>), HealError> {
        let mut current = solid_id;
        let mut results = Vec::with_capacity(self.steps.len());
        let mut ctx = HealContext::new();

        for step_name in &self.steps {
            let op = self.registry.get(step_name).ok_or_else(|| {
                HealError::InvalidConfig(format!("unknown operator: {step_name}"))
            })?;

            log::info!("heal pipeline: running '{step_name}'");
            let (new_solid, result) = op.execute(topo, current, &mut ctx)?;
            results.push(result);
            current = new_solid;
        }

        Ok((current, results))
    }
}

impl Default for HealProcess {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    #[test]
    fn pipeline_propagates_fix_result_from_operator() {
        // Regression test for the prior SolidId-comparison heuristic:
        // a clean unit-cube run through `direct_faces` shouldn't
        // produce a new SolidId (so the old code would call it OK), and
        // the operator's real `FixResult` should be Status::OK with
        // actions_taken=0 — but propagating it lets future operators
        // that DO modify in-place report non-zero actions correctly.
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let mut process = HealProcess::new();
        process.add_step("direct_faces");
        let (_new_solid, results) = process.execute(&mut topo, solid).unwrap();

        assert_eq!(results.len(), 1, "one step → one result");
        // The result comes from the operator itself, not synthesized
        // from a SolidId comparison. Whether the cube needs orientation
        // fixes is a topology-dependent detail; what matters is that
        // we get *some* status (a non-empty bitflags value) — the old
        // SolidId-comparison heuristic would have returned DONE1 only
        // if the SolidId changed, missing in-place mutations entirely.
        let r = &results[0];
        assert!(
            !r.status.is_empty(),
            "FixResult status should always be set, got empty"
        );
    }
}
