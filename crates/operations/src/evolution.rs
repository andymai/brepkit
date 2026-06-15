//! Evolution tracking for modeling operations.
//!
//! Records how faces evolve through booleans, fillets, and other operations,
//! enabling downstream consumers to track face provenance (e.g., for applying
//! persistent attributes like color or constraints).

use std::collections::{HashMap, HashSet};

use brepkit_math::vec::{Point3, Vec3};

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

/// Build an [`EvolutionMap`] by matching output faces to input faces purely
/// from geometry (face normal + centroid signatures `(index, normal, centroid)`).
///
/// This is operation-agnostic — any op that can snapshot face signatures before
/// and after (booleans, fillets, …) reuses it:
/// - An output face whose normal+centroid is close to an input face is a
///   **modified** version of it (every near-tied input is recorded, so a
///   same-domain merge of two inputs into one output keeps both origins).
/// - An output face matching no input is **generated**, attributed to the
///   nearest input (e.g. a fillet blend face or a boolean intersection face).
/// - An input face matched by no output is **deleted**.
#[must_use]
pub fn build_evolution_by_geometry(
    input_faces: &[(usize, Vec3, Point3)],
    output_faces: &[(usize, Vec3, Point3)],
) -> EvolutionMap {
    let mut evo = EvolutionMap::new();
    let mut matched_inputs: HashSet<usize> = HashSet::new();
    let mut unmatched_outputs: Vec<(usize, Vec3, Point3)> = Vec::new();

    // Normal dot threshold cos(45°) — relaxed because faces split by an
    // operation may shift slightly. Centroid distance² cap is generous.
    let normal_threshold = 0.707;
    let centroid_dist_sq_max = 100.0;

    for &(out_idx, out_normal, out_centroid) in output_faces {
        let mut best_score = f64::NEG_INFINITY;
        let mut matches: Vec<(usize, f64)> = Vec::new();

        for &(in_idx, in_normal, in_centroid) in input_faces {
            let dot = out_normal.dot(in_normal);
            if dot < normal_threshold {
                continue;
            }
            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if dist_sq > centroid_dist_sq_max {
                continue;
            }
            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
            }
            matches.push((in_idx, score));
        }

        if matches.is_empty() {
            unmatched_outputs.push((out_idx, out_normal, out_centroid));
            continue;
        }

        // Accept any near-tied match: two inputs legitimately contributing to
        // one output (e.g. the two halves of a same-domain-merged face).
        let score_tol = 0.05;
        for &(in_idx, score) in &matches {
            if score >= best_score - score_tol {
                evo.add_modified(in_idx, out_idx);
                matched_inputs.insert(in_idx);
            }
        }
    }

    // Unmatched outputs are generated — attribute each to the nearest input.
    for &(out_idx, _out_normal, out_centroid) in &unmatched_outputs {
        let mut best_dist_sq = f64::MAX;
        let mut best_input: Option<usize> = None;
        for &(in_idx, _, in_centroid) in input_faces {
            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_input = Some(in_idx);
            }
        }
        if let Some(in_idx) = best_input {
            evo.add_generated(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        }
    }

    // Any input matched by nothing was deleted.
    for &(in_idx, _, _) in input_faces {
        if !matched_inputs.contains(&in_idx) {
            evo.add_deleted(in_idx);
        }
    }

    evo
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::vec::{Point3, Vec3};

    use super::*;

    #[test]
    fn matcher_classifies_modified_generated_deleted() {
        let pz = Vec3::new(0.0, 0.0, 1.0);
        let nz = Vec3::new(0.0, 0.0, -1.0);
        let px = Vec3::new(1.0, 0.0, 0.0);
        let inputs = [
            (0usize, pz, Point3::new(0.0, 0.0, 0.0)),
            (1usize, nz, Point3::new(0.0, 0.0, -10.0)),
        ];
        let outputs = [
            // Same normal+position as input 0 → modified.
            (100usize, pz, Point3::new(0.0, 0.0, 0.0)),
            // Orthogonal normal, matches nothing → generated, nearest input is 0.
            (200usize, px, Point3::new(1.0, 0.0, 0.0)),
        ];
        let evo = build_evolution_by_geometry(&inputs, &outputs);
        assert_eq!(evo.modified.get(&0), Some(&vec![100]));
        assert_eq!(evo.generated.get(&0), Some(&vec![200]));
        assert!(evo.deleted.contains(&1), "input 1 had no output → deleted");
    }

    #[test]
    fn fillet_evolution_tracks_all_faces() {
        use brepkit_topology::explorer::solid_edges;

        let mut topo = brepkit_topology::Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let inputs = crate::boolean::collect_face_signatures(&topo, cube).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        let filleted = crate::blend_ops::fillet_v2(&mut topo, cube, &[edges[0]], 1.0)
            .unwrap()
            .solid;
        let outputs = crate::boolean::collect_face_signatures(&topo, filleted).unwrap();

        let evo = build_evolution_by_geometry(&inputs, &outputs);

        // The fillet adds a blend face (6 → 7) yet removes no box face.
        assert_eq!(inputs.len(), 6);
        assert_eq!(outputs.len(), 7);
        assert!(
            evo.deleted.is_empty(),
            "no box face is deleted by the fillet"
        );
        assert_eq!(evo.modified.len(), 6, "all six box faces are tracked");

        // Every output face — including the new blend — is attributed to an
        // input, so a downstream face reference always resolves.
        let tracked: HashSet<usize> = evo
            .modified
            .values()
            .chain(evo.generated.values())
            .flatten()
            .copied()
            .collect();
        let output_indices: HashSet<usize> = outputs.iter().map(|&(i, _, _)| i).collect();
        assert_eq!(tracked, output_indices, "every output face is attributed");
    }
}
