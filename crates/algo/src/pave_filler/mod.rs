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
pub mod force_interf_ee;
mod helpers;
pub mod link_existing;
pub mod make_blocks;
pub mod make_pcurves;
pub mod make_split_edges;
pub mod phase_ee;
pub mod phase_ef;
pub mod phase_ff;
pub mod phase_ff_coplanar;
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
    topo: &'a mut Topology,
    /// Solid A (first boolean argument).
    solid_a: SolidId,
    /// Solid B (second boolean argument).
    solid_b: SolidId,
    /// Tolerance for geometric comparisons.
    tol: Tolerance,
}

impl<'a> PaveFiller<'a> {
    /// Creates a new `PaveFiller` for two solids.
    #[allow(dead_code)]
    pub fn new(topo: &'a mut Topology, solid_a: SolidId, solid_b: SolidId) -> Self {
        Self {
            topo,
            solid_a,
            solid_b,
            tol: Tolerance::default(),
        }
    }

    /// Creates a `PaveFiller` with custom tolerance.
    pub fn with_tolerance(
        topo: &'a mut Topology,
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
    /// Creates new vertices in `Topology` when intersection points have
    /// no nearby existing vertex (EE, EF, FF phases).
    /// Call [`run_pave_filler`] instead to run both stages.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup or intersection fails.
    pub fn perform(&mut self, arena: &mut GfaArena) -> Result<(), AlgoError> {
        self.init_pave_blocks(arena)?;

        phase_vv::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;
        // VV is the only phase that registers same-domain vertices, and
        // `edge_pave_blocks` is fixed at init — so the pave-vertex coincidence
        // index is stable for the remaining phases. Build it once here instead
        // of linear-scanning every pave block per intersection endpoint.
        arena.build_pave_vertex_index(self.topo, self.tol.linear);
        phase_ve::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;
        phase_ee::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;
        phase_vf::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;
        phase_ef::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;
        phase_ff::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

        // Coplanar face splitting: parallel planes are skipped by Phase FF.
        phase_ff_coplanar::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)?;

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
    // Stage 1: Intersection (may create new vertices for EE/EF/FF crossings)
    {
        let mut filler = PaveFiller::with_tolerance(topo, solid_a, solid_b, tol);
        filler.perform(arena)?;
    }

    // Stage 2: Resolution (mutable Topology)
    make_blocks::perform(arena)?;
    force_interf_ee::perform(topo, tol, arena)?;
    link_existing::perform(topo, tol, arena)?;
    make_split_edges::perform(topo, arena)?;
    make_pcurves::perform(topo, arena)?;
    fill_face_info::perform(topo, arena)?;

    Ok(())
}

/// Run the PaveFiller pipeline over **N** source solids for an N-way fuse.
///
/// The Stage-1 intersection phases are inherently pairwise (each section is
/// between the faces of two solids) and, crucially, deposit only geometric
/// split data into the shared `arena` — they carry no `Rank`. So the two-solid
/// phase code is reused verbatim: run each phase for every spatially-interacting
/// source pair into ONE arena, and the paves/sections accumulate correctly. A
/// bbox broad-phase skips non-interacting pairs, keeping the stage O(n·k) for
/// the sparse interaction graphs a fused lattice produces rather than O(n²).
///
/// Phase order preserves the two-solid pipeline's invariant that the
/// pave-vertex coincidence index is built ONCE, after all VV coincidences are
/// registered and before the phases that query it: every pair's VV runs first,
/// then the index is built, then the remaining phases run per pair. Stage-2
/// resolution (which is already solid-agnostic — it reads the accumulated arena)
/// runs once.
///
/// For `sources.len() == 2` this is behaviourally identical to
/// [`run_pave_filler`]. Cut/Intersect are unaffected; this path is fuse-only.
///
/// # Errors
///
/// Returns [`AlgoError`] if `sources` is empty or any stage fails.
pub fn run_pave_filler_n(
    topo: &mut Topology,
    sources: &[SolidId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    if sources.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "N-way pave filler needs at least one source solid".into(),
        ));
    }

    // Stage 1: Intersection over interacting pairs, accumulating into one arena.
    init_pave_blocks_n(topo, sources, arena)?;
    let pairs = interacting_pairs(topo, sources, tol);

    for &(i, j) in &pairs {
        phase_vv::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    // VV is the only phase that registers same-domain vertices and the edge
    // pave blocks are fixed at init, so the coincidence index is stable for the
    // remaining phases — build it once, after every pair's VV (mirrors the
    // two-solid `PaveFiller::perform`).
    arena.build_pave_vertex_index(topo, tol.linear);
    for &(i, j) in &pairs {
        phase_ve::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ee::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_vf::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ef::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ff::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ff_coplanar::perform(topo, sources[i], sources[j], tol, arena)?;
    }

    // Stage 2: Resolution (solid-agnostic — reads the accumulated arena).
    make_blocks::perform(arena)?;
    force_interf_ee::perform(topo, tol, arena)?;
    link_existing::perform(topo, tol, arena)?;
    make_split_edges::perform(topo, arena)?;
    make_pcurves::perform(topo, arena)?;
    fill_face_info::perform(topo, arena)?;

    Ok(())
}

/// Initialize a pave block for every edge across all `sources`, de-duplicating
/// edges shared between solids (coincident walls). Mirrors
/// [`PaveFiller::init_pave_blocks`] but over N solids.
fn init_pave_blocks_n(
    topo: &Topology,
    sources: &[SolidId],
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    for &solid in sources {
        for edge_id in brepkit_topology::explorer::solid_edges(topo, solid)? {
            if arena.edge_pave_blocks.contains_key(&edge_id) {
                continue;
            }
            let edge = topo.edge(edge_id)?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
            arena.init_edge_pave_block(edge_id, edge.start(), t0, edge.end(), t1);
        }
    }
    Ok(())
}

/// Axis-aligned bounding box of a solid from its vertices.
fn solid_aabb(topo: &Topology, solid: SolidId) -> Option<brepkit_math::aabb::Aabb3> {
    let mut pts = Vec::new();
    for vid in brepkit_topology::explorer::solid_vertices(topo, solid).ok()? {
        pts.push(topo.vertex(vid).ok()?.point());
    }
    brepkit_math::aabb::Aabb3::try_from_points(pts)
}

/// Source-index pairs `(i, j)` with `i < j` whose bounding boxes overlap (each
/// expanded by tolerance so coincident-wall pairs are not missed). Two solids
/// with disjoint boxes cannot intersect, so pruning them is result-preserving.
/// A solid without a computable box is treated as interacting with all others
/// (conservative — never drops a real interaction).
fn interacting_pairs(topo: &Topology, sources: &[SolidId], tol: Tolerance) -> Vec<(usize, usize)> {
    let boxes: Vec<Option<brepkit_math::aabb::Aabb3>> =
        sources.iter().map(|&s| solid_aabb(topo, s)).collect();
    let mut pairs = Vec::new();
    for i in 0..sources.len() {
        for j in (i + 1)..sources.len() {
            let interact = match (boxes[i], boxes[j]) {
                (Some(bi), Some(bj)) => bi.expanded(tol.linear).intersects(bj.expanded(tol.linear)),
                _ => true,
            };
            if interact {
                pairs.push((i, j));
            }
        }
    }
    pairs
}
