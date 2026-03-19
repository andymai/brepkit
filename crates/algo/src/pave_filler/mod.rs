//! PaveFiller — intersection engine that builds pave blocks.
//!
//! Runs phases in two stages:
//!
//! **Stage 1 — Intersection** (reads `&Topology`, writes `&mut GfaArena`):
//! VV, VE, EE, VF, EF, FF.
//!
//! **Stage 2 — Resolution** (writes `&mut Topology` from arena data):
//! `MakeBlocks`, `MakeSplitEdges`, `MakePCurves`, `FillFaceInfo`.

pub mod fill_face_info;
mod helpers;
pub mod make_blocks;
pub mod make_pcurves;
pub mod make_split_edges;
pub mod phase_ee;
pub mod phase_ef;
pub mod phase_ff;
pub mod phase_ve;
pub mod phase_vf;
pub mod phase_vv;

#[cfg(test)]
mod tests;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// PaveFiller intersects all shape pairs between two solids,
/// building pave blocks and populating the GFA arena.
pub struct PaveFiller<'a> {
    /// The topology containing both solids.
    topo: &'a Topology,
    /// Solid A (first boolean argument).
    solid_a: SolidId,
    /// Solid B (second boolean argument).
    solid_b: SolidId,
    /// Tolerance for geometric comparisons.
    tol: Tolerance,
}

impl<'a> PaveFiller<'a> {
    /// Creates a new `PaveFiller` for two solids.
    #[must_use]
    pub fn new(topo: &'a Topology, solid_a: SolidId, solid_b: SolidId) -> Self {
        Self {
            topo,
            solid_a,
            solid_b,
            tol: Tolerance::default(),
        }
    }

    /// Creates a `PaveFiller` with custom tolerance.
    #[must_use]
    pub fn with_tolerance(
        topo: &'a Topology,
        solid_a: SolidId,
        solid_b: SolidId,
        tol: Tolerance,
    ) -> Self {
        Self {
            topo,
            solid_a,
            solid_b,
            tol,
        }
    }

    /// Run intersection phases (VV through FF), populating the GFA arena.
    ///
    /// This is stage 1 of the pipeline and only reads `&Topology`.
    /// Call [`run_pave_filler`] instead to run both stages.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup or intersection fails.
    pub fn perform(&self, arena: &mut GfaArena) -> Result<(), AlgoError> {
        // Phase 0: Initialize pave blocks for all edges
        self.init_pave_blocks(arena)?;

        // Phase 1: Vertex-vertex coincidence
        phase_vv::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Phase 2: Vertex-on-edge detection
        phase_ve::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Phase 3: Edge-edge intersection
        phase_ee::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Phase 4: Vertex-on-face detection
        phase_vf::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Phase 5: Edge-face intersection
        phase_ef::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Phase 6: Face-face intersection
        phase_ff::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        Ok(())
    }

    /// Initialize pave blocks for all edges of both solids.
    fn init_pave_blocks(&self, arena: &mut GfaArena) -> Result<(), AlgoError> {
        for &solid in &[self.solid_a, self.solid_b] {
            let edges = brepkit_topology::explorer::solid_edges(self.topo, solid)?;
            for edge_id in edges {
                // Skip if already initialized (shared edges between solids)
                if arena.edge_pave_blocks.contains_key(&edge_id) {
                    continue;
                }
                let edge = self.topo.edge(edge_id)?;
                let start_pos = self.topo.vertex(edge.start())?.point();
                let end_pos = self.topo.vertex(edge.end())?.point();
                let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
                arena.init_edge_pave_block(edge_id, edge.start(), t0, edge.end(), t1);
            }
        }
        Ok(())
    }
}

/// Run the complete PaveFiller pipeline (both stages).
///
/// **Stage 1 — Intersection** (reads `&Topology`):
/// Runs VV, VE, EE, VF, EF, FF phases to discover all interferences
/// and populate pave blocks with extra paves.
///
/// **Stage 2 — Resolution** (writes `&mut Topology`):
/// - `MakeBlocks` — splits pave blocks at extra paves
/// - `MakeSplitEdges` — creates new topology edges for leaf pave blocks
/// - `MakePCurves` — builds 2D curves on faces (stub)
/// - `FillFaceInfo` — classifies pave blocks as On/In/Sc per face
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup or intersection fails.
pub fn run_pave_filler(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // Stage 1: Intersection (immutable Topology)
    {
        let filler = PaveFiller::with_tolerance(topo, solid_a, solid_b, tol);
        filler.perform(arena)?;
    }

    // Stage 2: Resolution (mutable Topology)
    make_blocks::perform(arena)?;
    make_split_edges::perform(topo, arena)?;
    make_pcurves::perform(topo, arena)?;
    fill_face_info::perform(topo, arena)?;

    Ok(())
}
