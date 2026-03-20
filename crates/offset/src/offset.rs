//! Offset surface construction for each face.

use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::traits::ParametricSurface;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::data::{OffsetData, OffsetFace, OffsetStatus};
use crate::error::OffsetError;

/// Number of sample points along each parameter direction for NURBS fallback.
const NURBS_GRID_SIZE: usize = 16;

/// Interpolation degree for NURBS fallback surfaces.
const NURBS_DEGREE: usize = 3;

/// Construct the offset surface for every non-excluded face.
///
/// # Errors
///
/// Returns [`OffsetError`] if a surface cannot be offset (e.g. collapsed
/// cylinder radius or degenerate cone).
#[allow(clippy::too_many_lines)]
pub fn build_offset_faces(
    topo: &Topology,
    solid: SolidId,
    data: &mut OffsetData,
) -> Result<(), OffsetError> {
    let shell_id = topo.solid(solid)?.outer_shell();
    let faces = topo.shell(shell_id)?.faces().to_vec();

    for face_id in faces {
        // Check if excluded
        if data.excluded_faces.contains(&face_id) {
            let face = topo.face(face_id)?;
            data.offset_faces.insert(
                face_id,
                OffsetFace {
                    original: face_id,
                    surface: face.surface().clone(),
                    distance: 0.0,
                    status: OffsetStatus::Excluded,
                },
            );
            continue;
        }

        let face = topo.face(face_id)?;
        let reversed = face.is_reversed();
        let effective_distance = if reversed {
            -data.distance
        } else {
            data.distance
        };

        let offset_surface = match face.surface() {
            FaceSurface::Plane { normal, d } => FaceSurface::Plane {
                normal: *normal,
                d: d + effective_distance,
            },

            FaceSurface::Cylinder(cyl) => {
                let new_radius = cyl.radius() + effective_distance;
                if new_radius <= 0.0 {
                    return Err(OffsetError::InvalidInput {
                        reason: format!(
                            "cylinder offset collapses: radius {:.6} + offset {effective_distance:.6} <= 0",
                            cyl.radius()
                        ),
                    });
                }
                FaceSurface::Cylinder(CylindricalSurface::new(
                    cyl.origin(),
                    cyl.axis(),
                    new_radius,
                )?)
            }

            FaceSurface::Cone(cone) => {
                let half_angle = cone.half_angle();
                let cos_ha = half_angle.cos();
                if cos_ha.abs() < 1e-15 {
                    return Err(OffsetError::InvalidInput {
                        reason: "cone has degenerate half-angle (cos ≈ 0)".to_string(),
                    });
                }
                // Offset cone: the apex shifts along the axis so that the
                // surface at every v-parameter moves by `effective_distance`
                // along its outward normal. The normal makes angle (π/2 - a)
                // with the axis, so the axial component of the offset is
                // d / cos(a). The sign is negative because offsetting outward
                // moves the apex in the opposite direction of the axis.
                let apex_shift = -effective_distance / cos_ha;
                let new_apex = brepkit_math::vec::Point3::new(
                    cone.apex().x() + apex_shift * cone.axis().x(),
                    cone.apex().y() + apex_shift * cone.axis().y(),
                    cone.apex().z() + apex_shift * cone.axis().z(),
                );
                FaceSurface::Cone(ConicalSurface::new(new_apex, cone.axis(), half_angle)?)
            }

            FaceSurface::Sphere(sph) => {
                let new_radius = sph.radius() + effective_distance;
                if new_radius <= 0.0 {
                    return Err(OffsetError::InvalidInput {
                        reason: format!(
                            "sphere offset collapses: radius {:.6} + offset {effective_distance:.6} <= 0",
                            sph.radius()
                        ),
                    });
                }
                FaceSurface::Sphere(SphericalSurface::new(sph.center(), new_radius)?)
            }

            FaceSurface::Torus(tor) => {
                let new_minor = tor.minor_radius() + effective_distance;
                if new_minor <= 0.0 {
                    return Err(OffsetError::InvalidInput {
                        reason: format!(
                            "torus offset collapses: minor_radius {:.6} + offset {effective_distance:.6} <= 0",
                            tor.minor_radius()
                        ),
                    });
                }
                FaceSurface::Torus(ToroidalSurface::new(
                    tor.center(),
                    tor.major_radius(),
                    new_minor,
                )?)
            }

            FaceSurface::Nurbs(nurbs) => {
                let (u_min, u_max) = nurbs.domain_u();
                let (v_min, v_max) = nurbs.domain_v();

                let mut grid = Vec::with_capacity(NURBS_GRID_SIZE);
                for i in 0..NURBS_GRID_SIZE {
                    let u = u_min + (u_max - u_min) * (i as f64) / (NURBS_GRID_SIZE - 1) as f64;
                    let mut row = Vec::with_capacity(NURBS_GRID_SIZE);
                    for j in 0..NURBS_GRID_SIZE {
                        let v = v_min + (v_max - v_min) * (j as f64) / (NURBS_GRID_SIZE - 1) as f64;
                        let pt = ParametricSurface::evaluate(nurbs, u, v);
                        let n = nurbs.normal(u, v).map_err(|_| OffsetError::InvalidInput {
                            reason: format!("NURBS normal evaluation failed at ({u:.6}, {v:.6})"),
                        })?;
                        row.push(brepkit_math::vec::Point3::new(
                            pt.x() + effective_distance * n.x(),
                            pt.y() + effective_distance * n.y(),
                            pt.z() + effective_distance * n.z(),
                        ));
                    }
                    grid.push(row);
                }

                FaceSurface::Nurbs(interpolate_surface(&grid, NURBS_DEGREE, NURBS_DEGREE)?)
            }
        };

        data.offset_faces.insert(
            face_id,
            OffsetFace {
                original: face_id,
                surface: offset_surface,
                distance: effective_distance,
                status: OffsetStatus::Done,
            },
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions, OffsetStatus};
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;

    fn run_phases_1_2(topo: &Topology, solid: SolidId, distance: f64) -> OffsetData {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        build_offset_faces(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_offset_faces_are_planes() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_2(&topo, solid, 0.5);
        assert_eq!(data.offset_faces.len(), 6);
        for of in data.offset_faces.values() {
            assert_eq!(of.status, OffsetStatus::Done);
            assert!(matches!(of.surface, FaceSurface::Plane { .. }));
        }
    }

    #[test]
    fn box_plane_d_shifted() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_2(&topo, solid, 0.5);
        let tol = Tolerance::new();
        let mut d_values: Vec<f64> = data
            .offset_faces
            .values()
            .filter_map(|of| {
                if let FaceSurface::Plane { d, .. } = &of.surface {
                    Some(*d)
                } else {
                    None
                }
            })
            .collect();
        d_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Unit cube: 3 faces have outward normal in +axis with d=1,
        // and 3 have outward normal in -axis with d=0.
        // Plane offset shifts d by +distance along normal direction:
        //   d=1 faces → d=1.5, d=0 faces → d=0.5.
        // (The d=0 faces have normals pointing in negative axis direction;
        // shifting d by +0.5 moves the plane outward in that direction.)
        assert!(
            d_values.iter().filter(|&&d| tol.approx_eq(d, 0.5)).count() >= 1,
            "expected some d=0.5 faces, got {d_values:?}"
        );
        assert!(
            d_values.iter().filter(|&&d| tol.approx_eq(d, 1.5)).count() >= 1,
            "expected some d=1.5 faces, got {d_values:?}"
        );
    }

    #[test]
    fn excluded_faces_marked() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let shell = topo.solid(solid).unwrap().outer_shell();
        let faces: Vec<_> = topo.shell(shell).unwrap().faces().to_vec();
        let exclude = vec![faces[0]];
        let mut data = OffsetData::new(0.5, OffsetOptions::default(), exclude);
        crate::analyse::analyse_edges(&topo, solid, &mut data).unwrap();
        build_offset_faces(&topo, solid, &mut data).unwrap();

        let excluded_count = data
            .offset_faces
            .values()
            .filter(|of| of.status == OffsetStatus::Excluded)
            .count();
        assert_eq!(excluded_count, 1, "one face should be excluded");
        assert_eq!(data.offset_faces.len(), 6, "all faces should be in the map");
    }
}
