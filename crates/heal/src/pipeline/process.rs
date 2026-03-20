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
use crate::status::Status;

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
            let new_solid = op.execute(topo, current, &mut ctx)?;

            // TODO: Status is inferred from SolidId comparison, which is lossy —
            // operators that modify topology in-place (without creating a new
            // solid) will appear as no-ops. Ideally, HealOperator::execute
            // would return (SolidId, FixResult) to report accurate status.
            let changed = new_solid != current;
            results.push(FixResult {
                status: if changed { Status::DONE1 } else { Status::OK },
                actions_taken: usize::from(changed),
            });

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
