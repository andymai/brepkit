//! Operator registry — name-based lookup for pipeline steps.

use std::collections::HashMap;

use super::operator::HealOperator;

/// Registry mapping operator names to implementations.
#[derive(Debug)]
pub struct OperatorRegistry {
    operators: HashMap<String, Box<dyn HealOperator>>,
}

impl OperatorRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    /// Register an operator under a name.
    pub fn register(&mut self, name: &str, op: Box<dyn HealOperator>) {
        self.operators.insert(name.to_string(), op);
    }

    /// Look up an operator by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn HealOperator> {
        self.operators.get(name).map(AsRef::as_ref)
    }

    /// List all registered operator names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.operators.keys().map(String::as_str).collect()
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
