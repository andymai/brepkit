//! Checkpoint and sketch state types used by [`super::kernel::BrepKernel`].

use std::rc::Rc;

use brepkit_topology::Topology;

/// A saved snapshot of the kernel state that can be restored.
#[derive(Clone)]
pub struct Checkpoint {
    pub topo: Rc<Topology>,
    pub assemblies: Vec<brepkit_operations::assembly::Assembly>,
    pub sketches: Vec<SketchState>,
}

/// Internal state for an in-progress sketch.
///
/// Stores points and constraints for the legacy index-based JS API.
/// A `GcsSystem` is created on-the-fly during `sketch_solve`.
#[derive(Default, Clone)]
pub struct SketchState {
    /// Legacy point/constraint storage for backward-compat API.
    pub points: Vec<brepkit_operations::sketch::SketchPoint>,
    pub constraints: Vec<brepkit_operations::sketch::Constraint>,
    /// Arc definitions: `(center_idx, start_idx, end_idx)` into points.
    pub arcs: Vec<(usize, usize, usize)>,
    /// Circle definitions: `(center_idx, radius)`, where `center_idx` indexes into `points`.
    pub circles: Vec<(usize, f64)>,
    /// Deferred arc-referencing constraints stored as raw JSON.
    /// These are resolved into real `GcsConstraint` values at solve time
    /// when entity IDs are available.
    pub deferred_constraints: Vec<serde_json::Value>,
}
