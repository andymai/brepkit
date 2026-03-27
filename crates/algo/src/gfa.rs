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
    // Identical-solid fast path: avoid the full GFA pipeline when both
    // operands are the same topology entity.
    if solid_a == solid_b {
        return match op {
            BooleanOp::Fuse | BooleanOp::Intersect => {
                // A ∪ A = A, A ∩ A = A — return the original solid.
                // The caller (operations crate) copies if needed.
                Ok(solid_a)
            }
            BooleanOp::Cut => {
                // A \ A = empty
                Err(AlgoError::AssemblyFailed(
                    "Cut of identical solids produces empty result".into(),
                ))
            }
        };
    }
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
    // Create an isolated shape store with deep-copied input solids.
    // The GFA pipeline operates entirely within the store, avoiding
    // vertex/edge identity conflicts with the caller's topology.
    let mut store = crate::ds::GfaShapeStore::new(topo, solid_a, solid_b)?;

    // Stage 1: PaveFiller — intersection + pave block construction
    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler(
        &mut store.topo,
        store.solid_a,
        store.solid_b,
        tol,
        &mut arena,
    )?;

    // Stage 2: Builder — face splitting + classification
    let mut builder = Builder::with_tolerance(
        std::mem::take(&mut store.topo),
        arena,
        store.solid_a,
        store.solid_b,
        tol,
    );
    builder.perform()?;

    // Stage 3: BOP selection + assembly
    let (store_topo, store_result) = builder.build_result(op)?;
    store.topo = store_topo;

    // Export result solid back to the caller's topology
    let result = store.export_solid(topo, store_result)?;

    Ok(result)
}
