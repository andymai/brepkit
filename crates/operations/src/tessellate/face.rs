//! Face tessellation dispatcher with UV computation.

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::AnalyticKind;
use super::TriangleMeshUV;
use super::edge_sampling::{edge_sample_count, plane_axes, segments_for_chord_deviation_a};
use super::nurbs::{
    compute_angular_range, compute_axial_range, compute_sphere_v_range, compute_torus_v_range,
    compute_v_param_range, sphere_analytic_kind, tessellate_nurbs, tessellate_periodic_nurbs_grid,
};
use super::planar::{tessellate_analytic, tessellate_analytic_with_boundary, tessellate_planar};

/// Diagonal shrink factors for spheres: both u and v are curved
/// simultaneously, so the worst-case chord spans a grid cell's diagonal,
/// whose angular step is `sqrt(2)` times the per-direction step. Sag grows
/// with the square of the step, so the deflection budget must be halved;
/// the angular cap is linear in the step, so it shrinks by `1/sqrt(2)`.
const SPHERE_DIAG_DEFL: f64 = 0.5;
const SPHERE_DIAG_ANG: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Legacy single shrink factor, kept verbatim for the curvature-floored
/// (mesh-boolean) path so its calibrated tessellations stay bit-identical.
const SPHERE_DIAG_LEGACY: f64 = 0.7;

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
    tessellate_with_uvs_a(
        topo,
        face,
        deflection,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
}

/// Tessellate a face (with UVs) using explicit linear and angular tolerances.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate_with_uvs_a(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    tessellate_with_uvs_floor(topo, face, deflection, angular_tol, false)
}

/// Whether a face is its whole surface: no holes, and an outer wire made of
/// closed edges only (a torus's seam pair collapsed onto one vertex), which
/// trims nothing away.
fn covers_whole_domain(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }
    for oe in topo.wire(face_data.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        if edge.start() != edge.end() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether a NURBS face's boundary runs along its surface's domain edges
/// (seams and poles included), so the face is the whole patch and its grid
/// needs no trim. Each edge is tested at twice the density the trimmed
/// mesher would sample it, so a trim this passes would give that mesher the
/// domain's own boundary.
fn bounded_by_domain(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
    angular_tol: f64,
) -> Result<bool, crate::OperationsError> {
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }
    let ((u_lo, u_hi), (v_lo, v_hi)) = (surface.domain_u(), surface.domain_v());
    let (tol_u, tol_v) = (1e-6 * (u_hi - u_lo), 1e-6 * (v_hi - v_lo));
    for oe in topo.wire(face_data.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (start, end) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        let samples = 2 * edge_sample_count(topo, edge, deflection, angular_tol, false).max(4);
        for k in 0..=samples {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (k as f64) / (samples as f64);
            let p = edge.curve().evaluate_with_endpoints(t, start, end);
            let Ok(at) =
                brepkit_math::nurbs::projection::project_point_to_surface(surface, p, 1e-6)
            else {
                return Ok(false);
            };
            let on_edge = (at.u - u_lo).abs() <= tol_u
                || (u_hi - at.u).abs() <= tol_u
                || (at.v - v_lo).abs() <= tol_v
                || (v_hi - at.v).abs() <= tol_v;
            if !on_edge {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Like [`tessellate_with_uvs_a`] with an explicit curvature-floor selector.
///
/// `curvature_floor` keeps the legacy dense sampling on doubly-curved
/// surfaces; the mesh-boolean path passes `true` (its co-refinement
/// robustness and fallback volume accuracy depend on the density), display
/// and export callers pass `false` (the chord formula already bounds sag).
pub(super) fn tessellate_with_uvs_floor(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
    angular_tol: f64,
    curvature_floor: bool,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    let face_data = topo.face(face)?;
    let is_reversed = face_data.is_reversed();

    // A holed curved face goes through the solid mesher's hole-aware paths,
    // and so does a NURBS face, which is a trimmed patch of its surface
    // unless it closes over the whole of a doubly periodic domain. One they
    // cannot take keeps the grid below, which covers the whole surface.
    let holed_analytic = !face_data.inner_wires().is_empty()
        && matches!(
            face_data.surface(),
            FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_)
        );
    let trimmed_nurbs = match face_data.surface() {
        FaceSurface::Nurbs(n) => {
            let whole =
                (n.is_periodic_u() && n.is_periodic_v() && covers_whole_domain(topo, face_data)?)
                    || bounded_by_domain(topo, face_data, n, deflection, angular_tol)?;
            !whole
        }
        _ => false,
    };
    let holed_wall = if holed_analytic || trimmed_nurbs {
        match super::nonplanar::tessellate_holed_face_local(
            topo,
            face,
            face_data,
            deflection,
            angular_tol,
            curvature_floor,
        ) {
            Ok(mesh) if !mesh.mesh.indices.is_empty() => Some(mesh),
            Ok(_) => None,
            Err(e) => {
                log::debug!("holed wall {face:?} falls back to the analytic grid: {e}");
                None
            }
        }
    } else {
        None
    };
    let mut result = if let Some(mesh) = holed_wall {
        Ok(mesh)
    } else {
        match face_data.surface() {
            FaceSurface::Plane { normal, .. } => {
                let mesh = tessellate_planar(topo, face_data, *normal, deflection, angular_tol)?;
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
            FaceSurface::Nurbs(surface)
                if surface.is_periodic_u()
                    && surface.is_periodic_v()
                    && covers_whole_domain(topo, face_data)? =>
            {
                Ok(tessellate_periodic_nurbs_grid(
                    surface,
                    deflection,
                    angular_tol,
                ))
            }
            FaceSurface::Nurbs(surface) => Ok(tessellate_nurbs(surface, deflection, angular_tol)),
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
                    tessellate_analytic_with_boundary(topo, face_data, cyl, deflection, angular_tol)
                } else {
                    let v_range = compute_axial_range(topo, face_data, cyl.origin(), cyl.axis());
                    let u_range = compute_angular_range(topo, face_data, |p| cyl.project_point(p));
                    let nu = segments_for_chord_deviation_a(
                        cyl.radius(),
                        u_range.1 - u_range.0,
                        deflection,
                        angular_tol,
                        false,
                    );
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
                // Boolean results can bound a cone by a winding chain of marched
                // NURBS pieces; the plain analytic sweep below ignores the
                // boundary and skins the full parametric band, so classify
                // meshes lose the wall lobes. Try the locally sampled cycle-rim
                // band first; it declines anything that is not a two-rim band.
                let has_nurbs_boundary = {
                    let wire = topo.wire(face_data.outer_wire())?;
                    wire.edges().iter().any(|oe| {
                        topo.edge(oe.edge())
                            .is_ok_and(|e| matches!(e.curve(), EdgeCurve::NurbsCurve(_)))
                    })
                };
                if has_nurbs_boundary
                    && let Some(band) = super::nonplanar::tessellate_band_face_local(
                        topo,
                        face_data,
                        deflection,
                        angular_tol,
                    )?
                {
                    Ok(band)
                } else {
                    let v_range =
                        compute_v_param_range(topo, face_data, |p| cone.project_point(p).1);
                    let u_range = compute_angular_range(topo, face_data, |p| cone.project_point(p));
                    let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
                    let nu = segments_for_chord_deviation_a(
                        max_radius.max(0.01),
                        u_range.1 - u_range.0,
                        deflection,
                        angular_tol,
                        false,
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
            }
            FaceSurface::Sphere(sphere) => {
                let u_range = compute_angular_range(topo, face_data, |p| sphere.project_point(p));
                let v_range = compute_sphere_v_range(topo, face_data, sphere);
                // Both directions are curved at once; the worst-case sag is along
                // the diagonal, so shrink the step to keep it within tol.
                // Without the curvature floor: every normal-section curvature of a
                // sphere is exactly 1/r, and latitude chords are shorter than
                // great-circle chords at the same angular step, so the diag-shrunk
                // chord formula already bounds the surface sag in both directions.
                let (defl_shrink, ang_shrink) = if curvature_floor {
                    (SPHERE_DIAG_LEGACY, SPHERE_DIAG_LEGACY)
                } else {
                    (SPHERE_DIAG_DEFL, SPHERE_DIAG_ANG)
                };
                let nu = segments_for_chord_deviation_a(
                    sphere.radius(),
                    u_range.1 - u_range.0,
                    deflection * defl_shrink,
                    angular_tol * ang_shrink,
                    curvature_floor,
                );
                let nv = segments_for_chord_deviation_a(
                    sphere.radius(),
                    v_range.1 - v_range.0,
                    deflection * defl_shrink,
                    angular_tol * ang_shrink,
                    curvature_floor,
                );
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
                let v_range = compute_torus_v_range(topo, face_data, torus);
                let nu = segments_for_chord_deviation_a(
                    torus.major_radius(),
                    u_range.1 - u_range.0,
                    deflection,
                    angular_tol,
                    true,
                );
                let nv = segments_for_chord_deviation_a(
                    torus.minor_radius(),
                    v_range.1 - v_range.0,
                    deflection,
                    angular_tol,
                    true,
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
