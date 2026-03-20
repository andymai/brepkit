//! Shape fixing — targeted repairs for detected issues.
//!
//! The fix hierarchy mirrors the B-Rep entity tree:
//! `fix_shape` → `fix_solid` → `fix_shell` → `fix_face` → `fix_wire` → `fix_edge`.
//!
//! Each fixer uses analysis results to decide which fixes to apply,
//! controlled by [`FixConfig`] tri-state modes.

pub mod config;
pub mod edge;
pub mod face;
pub mod shell;
pub mod small_face;
pub mod solid;
pub mod split_vertex;
pub mod wire;
pub mod wireframe;

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

pub use config::{FixConfig, FixMode};

use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Result of a fix operation.
#[derive(Debug, Clone)]
pub struct FixResult {
    /// Status flags indicating what was done/failed.
    pub status: Status,
    /// Total number of individual repair actions taken.
    pub actions_taken: usize,
}

impl FixResult {
    /// Create a result indicating nothing was needed.
    #[must_use]
    pub fn ok() -> Self {
        Self {
            status: Status::OK,
            actions_taken: 0,
        }
    }

    /// Merge another result into this one.
    pub fn merge(&mut self, other: &Self) {
        self.status = self.status.merge(other.status);
        self.actions_taken += other.actions_taken;
    }
}

/// Top-level shape fixer — the main entry point for healing.
///
/// Creates a [`HealContext`], runs the full fix hierarchy
/// (solid → shell → face → wire → edge), applies all recorded
/// changes via [`ReShape`](crate::reshape::ReShape), and returns
/// the (possibly updated) solid ID.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail during healing.
pub fn fix_shape(
    topo: &mut Topology,
    solid_id: SolidId,
    config: &FixConfig,
) -> Result<(SolidId, FixResult), HealError> {
    let mut ctx = HealContext::new();
    let result = solid::fix_solid(topo, solid_id, &mut ctx, config)?;

    // Apply all accumulated replacements.
    let new_solid = ctx.reshape.apply(topo, solid_id)?;

    Ok((new_solid, result))
}

/// Top-level shape fixer with custom tolerance.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail during healing.
pub fn fix_shape_with_tolerance(
    topo: &mut Topology,
    solid_id: SolidId,
    config: &FixConfig,
    tolerance: f64,
) -> Result<(SolidId, FixResult), HealError> {
    let mut ctx = HealContext::with_tolerance(tolerance);
    let result = solid::fix_solid(topo, solid_id, &mut ctx, config)?;

    let new_solid = ctx.reshape.apply(topo, solid_id)?;

    Ok((new_solid, result))
}
