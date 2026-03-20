//! Convert analytic geometry to B-spline representation.

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Convert all analytic geometry in a solid to B-Spline representation.
///
/// Returns the number of entities converted.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn convert_solid_to_bspline(
    topo: &mut Topology,
    solid_id: SolidId,
) -> Result<usize, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let converted = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let is_analytic = face.surface().is_analytic();
        if is_analytic {
            // TODO: Convert analytic surface to NURBS.
            // This requires generating control points and knot vectors
            // for each surface type (plane, cylinder, cone, sphere, torus).
            log::debug!(
                "convert_to_bspline: skipping face {fid:?} (analytic→NURBS not yet implemented)"
            );
        }
    }

    // TODO: Convert analytic edge curves to NURBS.

    Ok(converted)
}
