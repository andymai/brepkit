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
    /// Volume-weighted x-moment: (1/2) integral of x^2 * n_x dA (divergence theorem).
    pub volume_moment_x: f64,
    /// Volume-weighted y-moment: (1/2) integral of y^2 * n_y dA (divergence theorem).
    pub volume_moment_y: f64,
    /// Volume-weighted z-moment: (1/2) integral of z^2 * n_z dA (divergence theorem).
    pub volume_moment_z: f64,
    /// Area-weighted centroid x-component (for surface centroid, not solid CoM).
    pub centroid_x: f64,
    /// Area-weighted centroid y-component (for surface centroid, not solid CoM).
    pub centroid_y: f64,
    /// Area-weighted centroid z-component (for surface centroid, not solid CoM).
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
            let full = (
                (0.0, std::f64::consts::TAU),
                (f64::NEG_INFINITY, f64::INFINITY),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
            ))
        }
        FaceSurface::Cone(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (f64::NEG_INFINITY, f64::INFINITY),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
            ))
        }
        FaceSurface::Sphere(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
            ))
        }
        FaceSurface::Torus(s) => {
            let full = ((0.0, std::f64::consts::TAU), (0.0, std::f64::consts::TAU));
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, true, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
            ))
        }
        FaceSurface::Nurbs(s) => {
            let full = (s.domain_u(), s.domain_v());
            let periodic_u = s.is_periodic_u();
            let periodic_v = s.is_periodic_v();
            let (u_range, v_range) =
                face_uv_bounds(topo, face_id, s, periodic_u, periodic_v, full)?;
            let uv_boundary =
                build_face_uv_boundary(topo, face_id, |p| s.project_point(p), periodic_u)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                periodic_u,
            ))
        }
    }
}

/// UV domain bounds as `((u_min, u_max), (v_min, v_max))`.
type UvBounds = ((f64, f64), (f64, f64));

/// Compute UV bounds for a parametric face by projecting boundary vertices
/// onto the surface and taking the min/max of the resulting parameters.
///
/// For surfaces with periodic u or v coordinates (cylinders, cones, spheres,
/// tori), sequentially unwraps the angular coordinates so that faces straddling
/// the 0/2pi seam produce correct ranges.
///
/// When all projected vertices coincide (e.g. a full-revolution face),
/// `full_domain` is returned instead.
fn face_uv_bounds<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
    periodic_u: bool,
    periodic_v: bool,
    full_domain: UvBounds,
) -> Result<UvBounds, CheckError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;

    // Collect projected UV values for each boundary vertex (in edge order).
    let mut uvs = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        let pt = topo.vertex(vid)?.point();
        uvs.push(surface.project_point(pt));
    }

    if uvs.is_empty() {
        return Err(CheckError::IntegrationFailed(
            "face wire has no edges".into(),
        ));
    }

    // Unwrap periodic coordinates sequentially so seam-straddling faces
    // produce a contiguous range instead of the full [0, 2pi).
    if periodic_u || periodic_v {
        for i in 1..uvs.len() {
            if periodic_u {
                uvs[i].0 = unwrap_angle(uvs[i - 1].0, uvs[i].0);
            }
            if periodic_v {
                uvs[i].1 = unwrap_angle(uvs[i - 1].1, uvs[i].1);
            }
        }
    }

    // Check for coincident vertices (all project to same point) — use full domain.
    let coincident = uvs.len() < 3 || {
        let ref_uv = uvs[0];
        uvs.iter()
            .all(|uv| (uv.0 - ref_uv.0).abs() < 1e-6 && (uv.1 - ref_uv.1).abs() < 1e-6)
    };
    if coincident {
        return Ok(full_domain);
    }

    let u_min = uvs.iter().map(|uv| uv.0).fold(f64::INFINITY, f64::min);
    let u_max = uvs.iter().map(|uv| uv.0).fold(f64::NEG_INFINITY, f64::max);
    let v_min = uvs.iter().map(|uv| uv.1).fold(f64::INFINITY, f64::min);
    let v_max = uvs.iter().map(|uv| uv.1).fold(f64::NEG_INFINITY, f64::max);

    if u_min >= u_max || v_min >= v_max {
        return Err(CheckError::IntegrationFailed(
            "degenerate UV domain for face".into(),
        ));
    }

    Ok(((u_min, u_max), (v_min, v_max)))
}

/// Unwrap a step in a periodic (angular) coordinate to avoid discontinuities.
///
/// Adjusts `next` so that `next - prev` lies in `(-pi, pi]`, keeping the
/// sequence monotonic through the 0/2pi seam.
fn unwrap_angle(prev: f64, next: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let diff = next - prev;
    prev + diff - tau * ((diff + std::f64::consts::PI) / tau).floor()
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
            volume_moment_x: 0.0,
            volume_moment_y: 0.0,
            volume_moment_z: 0.0,
            centroid_x: 0.0,
            centroid_y: 0.0,
            centroid_z: 0.0,
        });
    }

    // Fan triangulation from vertex 0
    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
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

        // Volume moments via divergence theorem: (1/2) integral of x^2 * n_x dA
        // For a planar triangle with constant normal, integral of x^2 over triangle
        // = (area/3) * (x_a^2 + x_b^2 + x_c^2 + x_a*x_b + x_a*x_c + x_b*x_c) / 2
        // Simplified: use (x_a^2 + x_b^2 + x_c^2 + x_a*x_b + x_a*x_c + x_b*x_c)/6
        let avg_x2 = (a.x() * a.x()
            + b.x() * b.x()
            + c.x() * c.x()
            + a.x() * b.x()
            + a.x() * c.x()
            + b.x() * c.x())
            / 6.0;
        let avg_y2 = (a.y() * a.y()
            + b.y() * b.y()
            + c.y() * c.y()
            + a.y() * b.y()
            + a.y() * c.y()
            + b.y() * c.y())
            / 6.0;
        let avg_z2 = (a.z() * a.z()
            + b.z() * b.z()
            + c.z() * c.z()
            + a.z() * b.z()
            + a.z() * c.z()
            + b.z() * c.z())
            / 6.0;
        mx += 0.5 * avg_x2 * normal.x() * tri_area;
        my += 0.5 * avg_y2 * normal.y() * tri_area;
        mz += 0.5 * avg_z2 * normal.z() * tri_area;

        cx += centroid.x() * tri_area;
        cy += centroid.y() * tri_area;
        cz += centroid.z() * tri_area;
    }

    Ok(FaceContribution {
        area,
        volume: vol,
        volume_moment_x: mx,
        volume_moment_y: my,
        volume_moment_z: mz,
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
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
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

            // Volume moments via divergence theorem:
            // CoM_x = (1/2V) surface_integral(x^2 * n_x dA)
            // n already includes Jacobian, so n.x() = N_x * |J|
            mx += w * 0.5 * p.x() * p.x() * n.x();
            my += w * 0.5 * p.y() * p.y() * n.y();
            mz += w * 0.5 * p.z() * p.z() * n.z();

            cx += w * p.x() * n_len;
            cy += w * p.y() * n_len;
            cz += w * p.z() * n_len;
        }
    }

    FaceContribution {
        area,
        volume: vol * sign,
        volume_moment_x: mx * sign,
        volume_moment_y: my * sign,
        volume_moment_z: mz * sign,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
}

/// Dispatch to trimmed or untrimmed parametric integration based on whether
/// a UV boundary polygon is available.
fn integrate_with_trimming<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    uv_boundary: &[(f64, f64)],
    u_periodic: bool,
) -> FaceContribution {
    if uv_boundary.len() >= 3 {
        integrate_parametric_trimmed(
            surface,
            u_range,
            v_range,
            gauss_order,
            sign,
            uv_boundary,
            u_periodic,
        )
    } else {
        integrate_parametric(surface, u_range, v_range, gauss_order, sign)
    }
}

/// Integrate a parametric surface with UV boundary trimming.
///
/// At each Gauss point, checks if the (u,v) coordinate falls inside the
/// face's UV boundary polygon. Points outside are skipped (zero contribution).
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn integrate_parametric_trimmed<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    uv_boundary: &[(f64, f64)],
    u_periodic: bool,
) -> FaceContribution {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let gauss_pts = gauss_legendre_points(gauss_order);
    let u_scale = (u_range.1 - u_range.0) / 2.0;
    let u_mid = f64::midpoint(u_range.0, u_range.1);
    let v_scale = (v_range.1 - v_range.0) / 2.0;
    let v_mid = f64::midpoint(v_range.0, v_range.1);

    // Pre-build UV polygon for containment tests.
    let uv_poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();

    // Pre-compute boundary u-range for periodic shifting.
    let u_bcenter = if u_periodic {
        let bmin = uv_boundary
            .iter()
            .map(|(bu, _)| *bu)
            .fold(f64::INFINITY, f64::min);
        let bmax = uv_boundary
            .iter()
            .map(|(bu, _)| *bu)
            .fold(f64::NEG_INFINITY, f64::max);
        (bmin + bmax) * 0.5
    } else {
        0.0
    };

    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for gpu in gauss_pts {
        let u = u_scale.mul_add(gpu.x, u_mid);
        for gpv in gauss_pts {
            let v = v_scale.mul_add(gpv.x, v_mid);

            // UV containment check for trimmed faces.
            let test_u = if u_periodic {
                let tau = std::f64::consts::TAU;
                let diff = u - u_bcenter;
                u_bcenter + diff - tau * ((diff + std::f64::consts::PI) / tau).floor()
            } else {
                u
            };

            if !point_in_polygon(Point2::new(test_u, v), &uv_poly) {
                continue;
            }

            let w = gpu.w * gpv.w * u_scale * v_scale;
            let p = surface.evaluate(u, v);
            let du = surface.partial_u(u, v);
            let dv = surface.partial_v(u, v);
            let n = Vec3::new(
                du.y() * dv.z() - du.z() * dv.y(),
                du.z() * dv.x() - du.x() * dv.z(),
                du.x() * dv.y() - du.y() * dv.x(),
            );
            let n_len = n.length();

            area += w * n_len;

            let pv = Vec3::new(p.x(), p.y(), p.z());
            vol += w * pv.dot(n) / 3.0;

            mx += w * 0.5 * p.x() * p.x() * n.x();
            my += w * 0.5 * p.y() * p.y() * n.y();
            mz += w * 0.5 * p.z() * p.z() * n.z();

            cx += w * p.x() * n_len;
            cy += w * p.y() * n_len;
            cz += w * p.z() * n_len;
        }
    }

    FaceContribution {
        area,
        volume: vol * sign,
        volume_moment_x: mx * sign,
        volume_moment_y: my * sign,
        volume_moment_z: mz * sign,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
}

/// Build a UV boundary polygon from a face's outer wire.
///
/// Projects each boundary vertex onto the surface to obtain (u, v) coordinates,
/// then unwraps periodic u-coordinates to avoid seam discontinuities.
fn build_face_uv_boundary<F>(
    topo: &Topology,
    face_id: FaceId,
    project: F,
    u_periodic: bool,
) -> Result<Vec<(f64, f64)>, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(vec![]);
    }

    let mut uv: Vec<(f64, f64)> = polygon.iter().map(|&p| project(p)).collect();

    // Unwrap periodic u-coordinates.
    for i in 1..uv.len() {
        if u_periodic {
            uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);
        }
    }

    Ok(uv)
}
