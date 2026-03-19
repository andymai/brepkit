//! Phase VV: Vertex-vertex coincidence detection.
//!
//! Finds all vertex pairs (one from each solid) that are spatially
//! coincident within tolerance. Merges them via same-domain mapping.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::ds::{GfaArena, Interference};
use crate::error::AlgoError;

/// Detect coincident vertices between solid A and solid B.
///
/// For every `(va, vb)` pair where `va` belongs to `solid_a` and `vb` to
/// `solid_b`, check if they are within combined tolerance. Coincident
/// pairs are recorded as VV interferences and merged in the same-domain
/// vertex map.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // AABB pre-filter: skip if solids are disjoint
    let bbox_a = crate::classifier::compute_solid_bbox(topo, solid_a)?;
    let bbox_b = crate::classifier::compute_solid_bbox(topo, solid_b)?;
    if !bbox_a
        .expanded(tol.linear)
        .intersects(bbox_b.expanded(tol.linear))
    {
        log::debug!("VV: solids are disjoint, skipping");
        return Ok(());
    }

    let verts_a = brepkit_topology::explorer::solid_vertices(topo, solid_a)?;
    let verts_b = brepkit_topology::explorer::solid_vertices(topo, solid_b)?;

    for &va in &verts_a {
        let vertex_a = topo.vertex(va)?;
        let pos_a = vertex_a.point();
        let tol_a = vertex_a.tolerance();

        for &vb in &verts_b {
            let vertex_b = topo.vertex(vb)?;
            let pos_b = vertex_b.point();
            let tol_b = vertex_b.tolerance();

            // Combined tolerance: sum of vertex tolerances + linear tolerance
            let combined_tol = tol_a + tol_b + tol.linear;
            let dist = (pos_a - pos_b).length();

            if dist <= combined_tol {
                // Record the interference
                arena
                    .interference
                    .vv
                    .push(Interference::VV { v1: va, v2: vb });

                // Merge into same-domain
                arena.merge_vertices(va, vb);

                log::debug!("VV: vertices {va:?} and {vb:?} coincide (dist={dist:.2e})",);
            }
        }
    }

    Ok(())
}
