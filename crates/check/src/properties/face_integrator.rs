//! Per-face Gauss quadrature integration for area, volume, CoM, and inertia.
//!
//! Provides numerical integration of geometric properties over individual
//! faces. Planar faces use polygon fan triangulation; parametric faces
//! (cylinder, cone, sphere, torus, NURBS) use tensor-product Gauss-Legendre
//! quadrature over the UV domain.

use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::CheckError;

/// Contribution of a single face to global geometric properties.
#[derive(Debug, Clone)]
pub struct FaceContribution {
    /// Face area.
    pub area: f64,
    /// Volume contribution: (1/3) integral of P dot N dA.
    pub volume: f64,
    /// Area-weighted centroid x-component.
    pub centroid_x: f64,
    /// Area-weighted centroid y-component.
    pub centroid_y: f64,
    /// Area-weighted centroid z-component.
    pub centroid_z: f64,
}

/// Integrate a face's geometric contribution using Gauss quadrature.
///
/// For planar faces, evaluates via polygon fan triangulation. For
/// parametric surfaces (analytic and NURBS), evaluates the surface and its
/// partial derivatives on a Gauss-point grid over the UV domain derived
/// from the face's boundary vertices.
///
/// # Errors
///
/// Returns an error if topology entities are missing or the face has
/// insufficient geometry for integration.
#[allow(clippy::too_many_lines)]
pub fn integrate_face(
    topo: &Topology,
    face_id: FaceId,
    gauss_order: usize,
) -> Result<FaceContribution, CheckError> {
    let face = topo.face(face_id)?;
    let reversed = face.is_reversed();
    let sign = if reversed { -1.0 } else { 1.0 };

    match face.surface() {
        FaceSurface::Plane { normal, .. } => {
            let effective_normal = if reversed { -*normal } else { *normal };
            integrate_planar_face(topo, face_id, effective_normal)
        }
        FaceSurface::Cylinder(s) => {
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s)?;
            Ok(integrate_parametric(s, u_range, v_range, gauss_order, sign))
        }
        FaceSurface::Cone(s) => {
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s)?;
            Ok(integrate_parametric(s, u_range, v_range, gauss_order, sign))
        }
        FaceSurface::Sphere(s) => {
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s)?;
            Ok(integrate_parametric(s, u_range, v_range, gauss_order, sign))
        }
        FaceSurface::Torus(s) => {
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s)?;
            Ok(integrate_parametric(s, u_range, v_range, gauss_order, sign))
        }
        FaceSurface::Nurbs(s) => {
            let u_range = s.domain_u();
            let v_range = s.domain_v();
            Ok(integrate_parametric(s, u_range, v_range, gauss_order, sign))
        }
    }
}

/// UV domain bounds as `((u_min, u_max), (v_min, v_max))`.
type UvBounds = ((f64, f64), (f64, f64));

/// Compute UV bounds for a parametric face by projecting boundary vertices
/// onto the surface and taking the min/max of the resulting parameters.
fn face_uv_bounds<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
) -> Result<UvBounds, CheckError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        for &vid in &[edge.start(), edge.end()] {
            let pt = topo.vertex(vid)?.point();
            let (u, v) = surface.project_point(pt);
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
    }

    if u_min >= u_max || v_min >= v_max {
        return Err(CheckError::IntegrationFailed(
            "degenerate UV domain for face".into(),
        ));
    }

    Ok(((u_min, u_max), (v_min, v_max)))
}

/// Integrate a planar face using polygon fan triangulation.
fn integrate_planar_face(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
) -> Result<FaceContribution, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(FaceContribution {
            area: 0.0,
            volume: 0.0,
            centroid_x: 0.0,
            centroid_y: 0.0,
            centroid_z: 0.0,
        });
    }

    // Fan triangulation from vertex 0
    let mut area = 0.0;
    let mut vol = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for i in 1..polygon.len() - 1 {
        let (a, b, c) = (polygon[0], polygon[i], polygon[i + 1]);
        let ab = b - a;
        let ac = c - a;
        let cross = Vec3::new(
            ab.y() * ac.z() - ab.z() * ac.y(),
            ab.z() * ac.x() - ab.x() * ac.z(),
            ab.x() * ac.y() - ab.y() * ac.x(),
        );
        let tri_area = cross.length() * 0.5;
        area += tri_area;

        // Volume contribution: (1/3) * centroid dot normal * area
        let centroid = Point3::new(
            (a.x() + b.x() + c.x()) / 3.0,
            (a.y() + b.y() + c.y()) / 3.0,
            (a.z() + b.z() + c.z()) / 3.0,
        );
        let pv = Vec3::new(centroid.x(), centroid.y(), centroid.z());
        vol += pv.dot(normal) * tri_area / 3.0;

        cx += centroid.x() * tri_area;
        cy += centroid.y() * tri_area;
        cz += centroid.z() * tri_area;
    }

    Ok(FaceContribution {
        area,
        volume: vol,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    })
}

/// Integrate a parametric surface using Gauss quadrature over the UV domain.
#[allow(clippy::cast_precision_loss)]
fn integrate_parametric<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
) -> FaceContribution {
    let gauss_pts = gauss_legendre_points(gauss_order);

    let u_scale = (u_range.1 - u_range.0) / 2.0;
    let u_mid = f64::midpoint(u_range.0, u_range.1);
    let v_scale = (v_range.1 - v_range.0) / 2.0;
    let v_mid = f64::midpoint(v_range.0, v_range.1);

    let mut area = 0.0;
    let mut vol = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for gpu in gauss_pts {
        let u = u_scale.mul_add(gpu.x, u_mid);
        for gpv in gauss_pts {
            let v = v_scale.mul_add(gpv.x, v_mid);
            let w = gpu.w * gpv.w * u_scale * v_scale;

            let p = surface.evaluate(u, v);
            let du = surface.partial_u(u, v);
            let dv = surface.partial_v(u, v);

            // Normal = du x dv (unnormalized, includes Jacobian)
            let n = Vec3::new(
                du.y() * dv.z() - du.z() * dv.y(),
                du.z() * dv.x() - du.x() * dv.z(),
                du.x() * dv.y() - du.y() * dv.x(),
            );
            let n_len = n.length();

            area += w * n_len;

            // Volume: (1/3) P dot N (unnormalized N includes Jacobian)
            let pv = Vec3::new(p.x(), p.y(), p.z());
            vol += w * pv.dot(n) / 3.0;

            cx += w * p.x() * n_len;
            cy += w * p.y() * n_len;
            cz += w * p.z() * n_len;
        }
    }

    FaceContribution {
        area,
        volume: vol * sign,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
}
