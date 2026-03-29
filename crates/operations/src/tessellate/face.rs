//! Face tessellation dispatcher with UV computation.

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::AnalyticKind;
use super::TriangleMeshUV;
use super::edge_sampling::{plane_axes, segments_for_chord_deviation};
use super::nurbs::{
    compute_angular_range, compute_axial_range, compute_sphere_v_range, compute_v_param_range,
    sphere_analytic_kind, tessellate_nurbs,
};
use super::planar::{tessellate_analytic, tessellate_analytic_with_boundary, tessellate_planar};

/// Tessellate a face and return mesh with per-vertex UV coordinates.
///
/// UV coordinates are the parametric (u, v) values of the surface at each
/// vertex. For planar faces, UVs are computed by projecting onto the face
/// plane axes.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate_with_uvs(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    let face_data = topo.face(face)?;
    let is_reversed = face_data.is_reversed();

    let mut result = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => {
            let mesh = tessellate_planar(topo, face_data, *normal, deflection)?;
            // For planar faces, project onto plane axes to get UVs.
            let (u_axis, v_axis) = plane_axes(*normal);
            let origin = if mesh.positions.is_empty() {
                brepkit_math::vec::Point3::new(0.0, 0.0, 0.0)
            } else {
                mesh.positions[0]
            };
            let uvs = mesh
                .positions
                .iter()
                .map(|p| {
                    let d: brepkit_math::vec::Vec3 = *p - origin;
                    [d.dot(u_axis), d.dot(v_axis)]
                })
                .collect();
            Ok::<_, crate::OperationsError>(TriangleMeshUV { mesh, uvs })
        }
        FaceSurface::Nurbs(surface) => Ok(tessellate_nurbs(surface, deflection)),
        FaceSurface::Cylinder(cyl) => {
            // Check if the boundary is non-standard (e.g., boolean result
            // with arbitrary polyline boundary instead of circles + seams).
            let has_non_standard_boundary = {
                let wire = topo.wire(face_data.outer_wire())?;
                let mut has_nurbs = false;
                let mut all_line = true;
                for oe in wire.edges() {
                    if let Ok(e) = topo.edge(oe.edge()) {
                        match e.curve() {
                            EdgeCurve::NurbsCurve(_) => has_nurbs = true,
                            EdgeCurve::Line => {}
                            _ => all_line = false,
                        }
                    }
                }
                has_nurbs || (all_line && wire.edges().len() > 4)
            };

            if has_non_standard_boundary {
                tessellate_analytic_with_boundary(topo, face_data, cyl, deflection)
            } else {
                let v_range = compute_axial_range(topo, face_data, cyl.origin(), cyl.axis());
                let u_range = compute_angular_range(topo, face_data, |p| cyl.project_point(p));
                let nu =
                    segments_for_chord_deviation(cyl.radius(), u_range.1 - u_range.0, deflection);
                let nv = 1;
                let cyl = cyl.clone();
                Ok(tessellate_analytic(
                    |u, v| cyl.evaluate(u, v),
                    |u, v| cyl.normal(u, v),
                    u_range,
                    v_range,
                    nu,
                    nv,
                    AnalyticKind::General,
                ))
            }
        }
        FaceSurface::Cone(cone) => {
            let v_range = compute_v_param_range(topo, face_data, |p| cone.project_point(p).1);
            let u_range = compute_angular_range(topo, face_data, |p| cone.project_point(p));
            let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
            let nu = segments_for_chord_deviation(
                max_radius.max(0.01),
                u_range.1 - u_range.0,
                deflection,
            );
            let nv = 1;
            let kind = if v_range.0.abs() < 1e-10 {
                AnalyticKind::ConeApex
            } else {
                AnalyticKind::General
            };
            let cone = cone.clone();
            Ok(tessellate_analytic(
                |u, v| cone.evaluate(u, v),
                |u, v| cone.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                kind,
            ))
        }
        FaceSurface::Sphere(sphere) => {
            let u_range = compute_angular_range(topo, face_data, |p| sphere.project_point(p));
            let v_range = compute_sphere_v_range(topo, face_data, sphere);
            let nu =
                segments_for_chord_deviation(sphere.radius(), u_range.1 - u_range.0, deflection);
            let nv =
                segments_for_chord_deviation(sphere.radius(), v_range.1 - v_range.0, deflection);
            let kind = sphere_analytic_kind(v_range);
            let sphere = sphere.clone();
            Ok(tessellate_analytic(
                |u, v| sphere.evaluate(u, v),
                |u, v| sphere.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                kind,
            ))
        }
        FaceSurface::Torus(torus) => {
            let u_range = compute_angular_range(topo, face_data, |p| torus.project_point(p));
            let v_range = (0.0, std::f64::consts::TAU);
            let nu = segments_for_chord_deviation(
                torus.major_radius(),
                u_range.1 - u_range.0,
                deflection,
            );
            let nv = segments_for_chord_deviation(
                torus.minor_radius(),
                v_range.1 - v_range.0,
                deflection,
            );
            let torus = torus.clone();
            Ok(tessellate_analytic(
                |u, v| torus.evaluate(u, v),
                |u, v| torus.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                AnalyticKind::General,
            ))
        }
    }?;

    if is_reversed {
        for n in &mut result.mesh.normals {
            *n = -*n;
        }
        let tri_count = result.mesh.indices.len() / 3;
        for t in 0..tri_count {
            result.mesh.indices.swap(t * 3 + 1, t * 3 + 2);
        }
    }

    Ok(result)
}
