//! Small face removal — detect and remove degenerate faces.
//!
//! Faces whose outer-wire bounding-box diagonal is smaller than the
//! working tolerance are considered degenerate slivers (common after
//! boolean operations at near-tangent intersections) and are removed
//! via the [`ReShape`](crate::reshape::ReShape) tracker.

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Remove or merge small faces in a solid.
///
/// For each face in the solid's outer shell, computes the bounding-box
/// diagonal of the outer wire's vertex positions. If the diagonal is
/// smaller than `ctx.tolerance.linear`, the face is recorded for removal
/// via [`ctx.reshape.remove_face`](crate::reshape::ReShape::remove_face).
///
/// The removal is not applied immediately — it is deferred until
/// [`ReShape::apply`](crate::reshape::ReShape::apply) is called by the
/// top-level [`fix_shape`](super::fix_shape) function.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn fix_small_faces(
    topo: &mut Topology,
    solid_id: SolidId,
    ctx: &mut HealContext,
    _config: &FixConfig,
) -> Result<FixResult, HealError> {
    let tol = ctx.tolerance.linear;

    let solid_data = topo.solid(solid_id)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut removed_count = 0usize;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;

        // Compute bounding box of the face's outer wire vertices.
        let mut min_pt = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_pt = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for &vid in &[edge.start(), edge.end()] {
                let pos = topo.vertex(vid)?.point();
                min_pt = Vec3::new(
                    min_pt.x().min(pos.x()),
                    min_pt.y().min(pos.y()),
                    min_pt.z().min(pos.z()),
                );
                max_pt = Vec3::new(
                    max_pt.x().max(pos.x()),
                    max_pt.y().max(pos.y()),
                    max_pt.z().max(pos.z()),
                );
            }
        }

        let diagonal = (max_pt - min_pt).length();
        if diagonal < tol {
            ctx.reshape.remove_face(fid);
            removed_count += 1;
        }
    }

    if removed_count > 0 {
        // Guard: don't remove ALL faces from a shell.
        if removed_count >= face_ids.len() {
            ctx.warn(format!(
                "all {} faces are small; skipping removal to preserve shell",
                face_ids.len()
            ));
            return Ok(FixResult::ok());
        }

        ctx.info(format!("marked {removed_count} small faces for removal"));
        Ok(FixResult {
            status: Status::DONE3,
            actions_taken: removed_count,
        })
    } else {
        Ok(FixResult::ok())
    }
}
