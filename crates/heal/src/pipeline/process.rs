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
    use crate::pipeline::operator::HealOperator;
    use crate::status::Status;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    /// Test-only operator that mutates `topo` in place (so the SolidId
    /// returned matches the input) but reports `actions_taken = 7` and
    /// `Status::DONE1`. Demonstrates the value of the trait change:
    /// the old `SolidId == solid_id ⇒ actions = 0` heuristic would
    /// have lost this signal entirely.
    #[derive(Debug)]
    struct FakeInPlaceOp;

    impl HealOperator for FakeInPlaceOp {
        fn name(&self) -> &'static str {
            "fake_in_place"
        }

        fn execute(
            &self,
            _topo: &mut Topology,
            solid_id: SolidId,
            _ctx: &mut HealContext,
        ) -> Result<(SolidId, FixResult), HealError> {
            Ok((
                solid_id,
                FixResult {
                    status: Status::DONE1,
                    actions_taken: 7,
                },
            ))
        }
    }

    #[test]
    fn pipeline_surfaces_in_place_actions_taken() {
        // Regression test for the prior SolidId-comparison heuristic:
        // an in-place mutation (same SolidId returned) used to be
        // synthesized as `actions_taken = 0` regardless of the
        // operator's actual work. Now the operator's reported count
        // is propagated verbatim.
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let mut process = HealProcess::new();
        process
            .registry_mut()
            .register("fake_in_place", Box::new(FakeInPlaceOp));
        process.add_step("fake_in_place");
        let (new_solid, results) = process.execute(&mut topo, solid).unwrap();

        assert_eq!(new_solid, solid, "in-place op preserves SolidId");
        assert_eq!(results.len(), 1);
        // Critical assertions: the operator's real values (NOT
        // synthesized from solid_id == new_solid).
        assert_eq!(
            results[0].actions_taken, 7,
            "actions_taken should propagate from operator, got {}",
            results[0].actions_taken
        );
        assert!(
            results[0].status.contains(Status::DONE1),
            "DONE1 should propagate, got {:?}",
            results[0].status
        );
    }
}
