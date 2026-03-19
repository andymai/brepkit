//! Top-level GFA orchestrator.
//!
//! Runs the complete General Fuse Algorithm pipeline:
//! PaveFiller -> Builder -> BOP -> assemble.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::bop::BooleanOp;
use crate::builder::Builder;
use crate::ds::GfaArena;
use crate::error::AlgoError;
use crate::pave_filler;

/// Run the complete GFA boolean operation with default tolerance.
///
/// This is the single entry point for boolean operations via the GFA.
///
/// # Errors
///
/// Returns [`AlgoError`] if any stage fails.
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<SolidId, AlgoError> {
    boolean_with_tolerance(topo, op, solid_a, solid_b, Tolerance::default())
}

/// Run the complete GFA boolean operation with custom tolerance.
///
/// Stages:
/// 1. **PaveFiller** — intersect shapes, build pave blocks
/// 2. **Builder** — split faces, classify sub-faces
/// 3. **BOP + assemble** — select faces, build result solid
///
/// # Errors
///
/// Returns [`AlgoError`] if any stage fails.
pub fn boolean_with_tolerance(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
) -> Result<SolidId, AlgoError> {
    // Stage 1: PaveFiller — intersection + pave block construction
    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler(topo, solid_a, solid_b, tol, &mut arena)?;

    // Stage 2: Builder — face splitting + classification
    let mut builder = Builder::with_tolerance(std::mem::take(topo), arena, solid_a, solid_b, tol);
    builder.perform()?;

    // Stage 3: BOP selection + assembly
    let (result_topo, result_solid) = builder.build_result(op)?;

    // Restore topology
    *topo = result_topo;

    Ok(result_solid)
}
