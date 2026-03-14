//! Checkpoint and sketch state types used by [`super::kernel::BrepKernel`].

use brepkit_topology::Topology;

/// A saved snapshot of the kernel state that can be restored.
#[derive(Clone)]
pub struct Checkpoint {
    pub topo: Topology,
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
}
