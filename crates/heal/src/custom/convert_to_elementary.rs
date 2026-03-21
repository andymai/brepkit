//! Convert NURBS geometry to analytic (elementary) surfaces where possible.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use brepkit_geometry::convert::{RecognizedSurface, recognize_surface};

use crate::HealError;

/// Try to recognize and replace NURBS surfaces with analytic equivalents.
///
/// Returns the number of surfaces converted.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn convert_to_elementary(
    topo: &mut Topology,
    solid_id: SolidId,
    tolerance: &Tolerance,
) -> Result<usize, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut converted = 0;

    // Snapshot surfaces.
    let surfaces: Vec<(FaceId, FaceSurface)> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| (fid, f.surface().clone())))
        .collect::<Result<Vec<_>, _>>()?;

    for (fid, surface) in &surfaces {
        if let FaceSurface::Nurbs(nurbs) = surface {
            match recognize_surface(nurbs, tolerance.linear) {
                RecognizedSurface::Plane { normal, d } => {
                    let face = topo.face_mut(*fid)?;
                    face.set_surface(FaceSurface::Plane { normal, d });
                    converted += 1;
                }
                RecognizedSurface::Cylinder {
                    origin,
                    axis,
                    radius,
                } => {
                    if let Ok(cyl) =
                        brepkit_math::surfaces::CylindricalSurface::new(origin, axis, radius)
                    {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Cylinder(cyl));
                        converted += 1;
                    }
                }
                RecognizedSurface::Sphere { center, radius } => {
                    if let Ok(sph) = brepkit_math::surfaces::SphericalSurface::new(center, radius) {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Sphere(sph));
                        converted += 1;
                    }
                }
                RecognizedSurface::NotRecognized => {}
            }
        }
    }

    Ok(converted)
}
