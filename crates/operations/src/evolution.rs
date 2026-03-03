//! Evolution tracking for modeling operations.
//!
//! Records how faces evolve through booleans, fillets, and other operations,
//! enabling downstream consumers to track face provenance (e.g., for applying
//! persistent attributes like color or constraints).

use std::collections::{HashMap, HashSet};

/// Tracks how faces evolve through a modeling operation.
///
/// After a boolean, fillet, or other operation, this map records:
/// - **modified**: input face -> output faces that replace it
/// - **generated**: input face -> new faces created adjacent to it
/// - **deleted**: input faces that were completely removed
#[derive(Debug, Clone, Default)]
pub struct EvolutionMap {
    /// Input face -> output faces that are modified versions of it.
    pub modified: HashMap<usize, Vec<usize>>,
    /// Input face -> new faces generated from it (e.g., blend faces from fillet).
    pub generated: HashMap<usize, Vec<usize>>,
    /// Input faces that were completely removed.
    pub deleted: HashSet<usize>,
}

impl EvolutionMap {
    /// Create an empty evolution map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `input` was modified into `output`.
    pub fn add_modified(&mut self, input: usize, output: usize) {
        self.modified.entry(input).or_default().push(output);
    }

    /// Record that `output` was generated from `input`.
    pub fn add_generated(&mut self, input: usize, output: usize) {
        self.generated.entry(input).or_default().push(output);
    }

    /// Record that `input` was deleted.
    pub fn add_deleted(&mut self, input: usize) {
        self.deleted.insert(input);
    }

    /// Serialize to JSON without serde.
    ///
    /// Produces a JSON object with `modified`, `generated`, and `deleted` fields.
    #[must_use]
    pub fn to_json(&self) -> String {
        let modified_entries: Vec<String> = self
            .modified
            .iter()
            .map(|(k, vs)| {
                let vals: Vec<String> = vs.iter().map(ToString::to_string).collect();
                format!("\"{k}\":[{}]", vals.join(","))
            })
            .collect();

        let generated_entries: Vec<String> = self
            .generated
            .iter()
            .map(|(k, vs)| {
                let vals: Vec<String> = vs.iter().map(ToString::to_string).collect();
                format!("\"{k}\":[{}]", vals.join(","))
            })
            .collect();

        let deleted_vals: Vec<String> = self.deleted.iter().map(ToString::to_string).collect();

        format!(
            "{{\"modified\":{{{}}},\"generated\":{{{}}},\"deleted\":[{}]}}",
            modified_entries.join(","),
            generated_entries.join(","),
            deleted_vals.join(",")
        )
    }
}
