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
use brepkit_topology::edge::EdgeCurve;
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
            let inner_loops =
                build_face_uv_inner_loops(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                &[],
                &inner_loops,
            ))
        }
        FaceSurface::Cone(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (f64::NEG_INFINITY, f64::INFINITY),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            let inner_loops =
                build_face_uv_inner_loops(topo, face_id, |p| s.project_point(p), true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                &[],
                &inner_loops,
            ))
        }
        FaceSurface::Sphere(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, |p| s.project_point(p), true)?;
            let hole_vs = full_revolution_hole_vs(topo, face_id, s);
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                &hole_vs,
                &[],
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
                &[],
                &[],
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
                &[],
                &[],
            ))
        }
    }
}

/// UV domain bounds as `((u_min, u_max), (v_min, v_max))`.
type UvBounds = ((f64, f64), (f64, f64));

/// The v-positions of a face's full-revolution inner wires (holes) on a
/// surface periodic in u.
///
/// A boolean that drills a cylinder through a sphere leaves each spherical
/// band bounded by a latitude circle hole (the tunnel rim). Such a hole wraps
/// the full u-period and sits at a single v, so the band runs from its outer
/// latitude to the hole — not on to the pole. Collecting these lets the
/// integrator clip the band instead of over-integrating the polar cap the hole
/// removed. Each entry is the mean projected v of one full-revolution hole.
fn full_revolution_hole_vs<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
) -> Vec<f64> {
    use std::f64::consts::TAU;
    let Ok(face) = topo.face(face_id) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &wid in face.inner_wires() {
        let Ok(wire) = topo.wire(wid) else { continue };
        let mut us = Vec::new();
        let mut vs = Vec::new();
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            // Oriented traversal: the wire-ordered start vertex is the edge's
            // end when the oriented edge is reversed.
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            let Ok(v) = topo.vertex(vid) else {
                continue;
            };
            let (u, vv) = surface.project_point(v.point());
            us.push(u);
            vs.push(vv);
        }
        if vs.is_empty() {
            continue;
        }
        let v_min = vs.iter().copied().fold(f64::INFINITY, f64::min);
        let v_max = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Constant-v latitude circle.
        if v_max - v_min > 1e-6 {
            continue;
        }
        // Full revolution in u: the unwrapped per-vertex deltas around the
        // CLOSED loop (including the closing step back to the first vertex) sum
        // to ≈ TAU. A single-edge closed circle has one vertex, so also accept
        // holes whose sole edge is a closed circle curve.
        let unwrapped_span = {
            let n = us.len();
            let mut acc = 0.0;
            for i in 0..n {
                let d = us[(i + 1) % n] - us[i];
                acc += d - TAU * ((d + std::f64::consts::PI) / TAU).floor();
            }
            acc.abs()
        };
        let single_closed_circle = wire.edges().len() == 1
            && wire.edges().first().is_some_and(|oe| {
                topo.edge(oe.edge())
                    .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Circle(_)))
            });
        if unwrapped_span >= TAU - 1e-3 || single_closed_circle {
            out.push(0.5 * (v_min + v_max));
        }
    }
    out
}

/// Compute UV bounds for a parametric face by projecting boundary vertices
/// onto the surface and taking the min/max of the resulting parameters.
///
/// For surfaces with periodic u or v coordinates (cylinders, cones, spheres,
/// tori), sequentially unwraps the angular coordinates so that faces straddling
/// the 0/2pi seam produce correct ranges.
///
/// When all projected vertices coincide (e.g. a full-revolution face),
/// `full_domain` is returned instead.
///
/// **Limitation:** Only the outer wire is used for UV bounds. Inner wires
/// (holes) are handled during Gauss integration by the UV containment check
/// in `integrate_parametric_trimmed`, but the current containment only tests
/// against the outer boundary. Faces with holes will over-integrate the hole
/// region. A proper fix requires multi-polygon UV containment (outer minus
/// holes).
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
    let mut u_max = uvs.iter().map(|uv| uv.0).fold(f64::NEG_INFINITY, f64::max);
    let v_min = uvs.iter().map(|uv| uv.1).fold(f64::INFINITY, f64::min);
    let mut v_max = uvs.iter().map(|uv| uv.1).fold(f64::NEG_INFINITY, f64::max);

    // All boundary vertices on the seam of a periodic axis (e.g. a
    // full-revolution lateral face whose circles start/end at the seam)
    // collapse that axis's range to zero — the face actually spans the
    // full period.
    if periodic_u && u_max - u_min < 1e-9 {
        u_max = u_min + (full_domain.0.1 - full_domain.0.0);
    }
    if periodic_v && v_max - v_min < 1e-9 {
        v_max = v_min + (full_domain.1.1 - full_domain.1.0);
    }

    if u_min >= u_max || v_min >= v_max {
        // A degenerate projection (e.g. all boundary vertices on a sphere's
        // pole seam) does not mean an empty face — it means the boundary failed
        // to bound a sub-region, so the face spans the full analytic domain.
        return Ok(full_domain);
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
///
/// Inner wires (holes) are integrated the same way and subtracted from the
/// outer-wire contribution.
fn integrate_planar_face(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
) -> Result<FaceContribution, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    let mut contrib = integrate_planar_polygon(&polygon, normal);

    let face = topo.face(face_id)?;
    let inner: Vec<_> = face.inner_wires().to_vec();
    for wid in inner {
        let hole = crate::util::wire_polygon(topo, wid)?;
        let h = integrate_planar_polygon(&hole, normal);
        contrib.area -= h.area;
        contrib.volume -= h.volume;
        contrib.volume_moment_x -= h.volume_moment_x;
        contrib.volume_moment_y -= h.volume_moment_y;
        contrib.volume_moment_z -= h.volume_moment_z;
        contrib.centroid_x -= h.centroid_x;
        contrib.centroid_y -= h.centroid_y;
        contrib.centroid_z -= h.centroid_z;
    }

    Ok(contrib)
}

/// Integrate a planar polygon's contribution via fan triangulation.
fn integrate_planar_polygon(polygon: &[Point3], normal: Vec3) -> FaceContribution {
    if polygon.len() < 3 {
        return FaceContribution {
            area: 0.0,
            volume: 0.0,
            volume_moment_x: 0.0,
            volume_moment_y: 0.0,
            volume_moment_z: 0.0,
            centroid_x: 0.0,
            centroid_y: 0.0,
            centroid_z: 0.0,
        };
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

    FaceContribution {
        area,
        volume: vol,
        volume_moment_x: mx,
        volume_moment_y: my,
        volume_moment_z: mz,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
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
    // Composite quadrature: tile the domain into patches no larger than ~PI/4
    // so one Gauss rule resolves curved and periodic integrands. A single patch
    // over a torus's full 2*PI period in both u and v under-resolves it (~0.5%
    // error); several patches per period converge to machine precision. The
    // patch count is capped so a long *linear* axis (e.g. a tall cylinder/cone
    // whose v is axial distance) cannot make integration cost scale with model
    // size — its integrand is low-degree, so a bounded number of patches stays
    // exact. Angular axes never exceed 2*PI (= 8 patches), well under the cap.
    const MAX_PATCHES: usize = 16;

    let gauss_pts = gauss_legendre_points(gauss_order);
    let patch = std::f64::consts::FRAC_PI_4;
    let nu = (((u_range.1 - u_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let nv = (((v_range.1 - v_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let du_patch = (u_range.1 - u_range.0) / nu as f64;
    let dv_patch = (v_range.1 - v_range.0) / nv as f64;
    let u_scale = du_patch / 2.0;
    let v_scale = dv_patch / 2.0;

    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for iu in 0..nu {
        let u_mid = du_patch.mul_add(iu as f64, u_range.0) + u_scale;
        for iv in 0..nv {
            let v_mid = dv_patch.mul_add(iv as f64, v_range.0) + v_scale;
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

/// Integrate a full-revolution analytic band, excluding the region enclosed by
/// the interior `hole_loops` (the UV loops of the face's inner wires).
///
/// Used for a cylinder/cone wall that another quadric bores through: the wall
/// wraps the full `u` period (so the outer boundary can't trim it) yet carries
/// closed lens loops that must be cut out of the volume integral. At each
/// Gauss `u`-node the lens occupies one or more `v`-intervals (found exactly
/// from where a vertical line crosses the inner-loop arrangement); the surface
/// is integrated only over the complementary kept `v`-sub-intervals, so the
/// curved lens boundary is resolved exactly rather than by coarse point
/// rejection.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn integrate_band_excluding_holes<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    hole_loops: &[Vec<(f64, f64)>],
) -> FaceContribution {
    const MAX_PATCHES: usize = 16;
    let gauss_pts = gauss_legendre_points(gauss_order);
    let patch = std::f64::consts::FRAC_PI_4;
    let nu = (((u_range.1 - u_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let du_patch = (u_range.1 - u_range.0) / nu as f64;
    let u_scale = du_patch / 2.0;

    // Collect every inner-loop edge as a (u, v) segment, keeping each segment's
    // NATIVE (per-loop-unwrapped) u. The query u is later tested against each
    // segment in every 2π translate, so the band's own u-origin need not match.
    // The lens region is treated as "inside the combined arrangement of ALL
    // these segments" via an even-odd vertical-ray test — NOT per-loop
    // point-in-polygon. This is essential here: each lens ellipse where one
    // cylinder bores through another is a full-u-wrapping sinusoid bounding NO
    // area on its own, yet the two ellipses TOGETHER enclose the lens lobes,
    // which the combined even-odd test resolves. The single wrap segment per
    // loop (Δu ≈ a full period) is dropped; the dense sampling keeps the
    // arrangement intact.
    let tau = std::f64::consts::TAU;
    let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new();
    for loop_uv in hole_loops {
        if loop_uv.len() < 2 {
            continue;
        }
        for i in 0..loop_uv.len() {
            let (u0, v0) = loop_uv[i];
            let (u1, v1) = loop_uv[(i + 1) % loop_uv.len()];
            if (u1 - u0).abs() > tau * 0.5 {
                continue;
            }
            let (mut a_u, mut a_v, mut b_u, mut b_v) = (u0, v0, u1, v1);
            if a_u > b_u {
                std::mem::swap(&mut a_u, &mut b_u);
                std::mem::swap(&mut a_v, &mut b_v);
            }
            segs.push((a_u, a_v, b_u, b_v));
        }
    }
    let seg_u_min = segs.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
    let seg_u_max = segs.iter().map(|s| s.2).fold(f64::NEG_INFINITY, f64::max);

    // The v-values where a vertical line at `u` crosses the inner-loop
    // arrangement, sorted ascending. Even-odd pairing of these gives the lens
    // v-intervals at that u; the kept region is the complement within
    // [v_range.0, v_range.1]. The query u is tested against each segment in
    // every 2π translate that can land in the segments' u-extent, so a loop
    // unwrapped to any window is matched regardless of the band's own u-origin.
    // Exact crossing v's (not a coarse-grid point test) let the v-integration
    // place its Gauss nodes only on the kept sub-intervals, integrating the
    // curved lens boundary exactly.
    let lens_v_crossings = |u: f64| -> Vec<f64> {
        let mut vs: Vec<f64> = Vec::new();
        // Smallest k so u + k*tau >= seg_u_min, then sweep up past seg_u_max.
        let k0 = ((seg_u_min - u) / tau).floor() as i64;
        let k1 = ((seg_u_max - u) / tau).ceil() as i64;
        for k in k0..=k1 {
            let uu = u + (k as f64) * tau;
            for &(a_u, a_v, b_u, b_v) in &segs {
                if uu >= a_u && uu < b_u && (b_u - a_u) > 1e-15 {
                    let t = (uu - a_u) / (b_u - a_u);
                    vs.push(a_v + t * (b_v - a_v));
                }
            }
        }
        vs.sort_by(f64::total_cmp);
        vs
    };

    // Kept v-sub-intervals at `u`: [v0, v1] minus the lens intervals (the
    // even-odd-paired crossing spans).
    let kept_v_intervals = |u: f64, v0: f64, v1: f64| -> Vec<(f64, f64)> {
        let crossings = lens_v_crossings(u);
        // Lens spans = consecutive pairs (c[0],c[1]), (c[2],c[3]), ...
        let mut kept = Vec::new();
        let mut cursor = v0;
        let mut i = 0;
        while i + 1 < crossings.len() {
            let lo = crossings[i].clamp(v0, v1);
            let hi = crossings[i + 1].clamp(v0, v1);
            if lo > cursor {
                kept.push((cursor, lo));
            }
            cursor = cursor.max(hi);
            i += 2;
        }
        if cursor < v1 {
            kept.push((cursor, v1));
        }
        kept
    };

    let mut acc = FaceContribution {
        area: 0.0,
        volume: 0.0,
        volume_moment_x: 0.0,
        volume_moment_y: 0.0,
        volume_moment_z: 0.0,
        centroid_x: 0.0,
        centroid_y: 0.0,
        centroid_z: 0.0,
    };

    for iu in 0..nu {
        let u_mid = du_patch.mul_add(iu as f64, u_range.0) + u_scale;
        for gpu in gauss_pts {
            let u = u_scale.mul_add(gpu.x, u_mid);
            let w_u = gpu.w * u_scale;
            // Integrate v only over the kept sub-intervals at this u, each with
            // its own Gauss rule so the lens-clipped limits are exact.
            for (kv0, kv1) in kept_v_intervals(u, v_range.0, v_range.1) {
                // Sub-tile a long kept interval so one Gauss rule resolves it.
                let kv_len = kv1 - kv0;
                let n_sub = ((kv_len / patch).ceil() as usize).clamp(1, MAX_PATCHES);
                let dv_sub = kv_len / n_sub as f64;
                let v_sub_scale = dv_sub / 2.0;
                for isv in 0..n_sub {
                    let v_sub_mid = dv_sub.mul_add(isv as f64, kv0) + v_sub_scale;
                    for gpv in gauss_pts {
                        let v = v_sub_scale.mul_add(gpv.x, v_sub_mid);
                        let w = w_u * gpv.w * v_sub_scale;
                        let p = surface.evaluate(u, v);
                        let du = surface.partial_u(u, v);
                        let dv = surface.partial_v(u, v);
                        let n = Vec3::new(
                            du.y() * dv.z() - du.z() * dv.y(),
                            du.z() * dv.x() - du.x() * dv.z(),
                            du.x() * dv.y() - du.y() * dv.x(),
                        );
                        let n_len = n.length();
                        acc.area += w * n_len;
                        let pv = Vec3::new(p.x(), p.y(), p.z());
                        acc.volume += w * pv.dot(n) / 3.0;
                        acc.volume_moment_x += w * 0.5 * p.x() * p.x() * n.x();
                        acc.volume_moment_y += w * 0.5 * p.y() * p.y() * n.y();
                        acc.volume_moment_z += w * 0.5 * p.z() * p.z() * n.z();
                        acc.centroid_x += w * p.x() * n_len;
                        acc.centroid_y += w * p.y() * n_len;
                        acc.centroid_z += w * p.z() * n_len;
                    }
                }
            }
        }
    }

    let (area, vol, mx, my, mz, cx, cy, cz) = (
        acc.area,
        acc.volume,
        acc.volume_moment_x,
        acc.volume_moment_y,
        acc.volume_moment_z,
        acc.centroid_x,
        acc.centroid_y,
        acc.centroid_z,
    );

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

/// Angular span (radians) covered by a set of `u` samples, computed as `2π`
/// minus the LARGEST gap between consecutive sorted samples (the wrap gap
/// included). A full revolution returns `≈ 2π`; a partial arc returns its true
/// (shorter) span — robust to a boundary that straddles the seam, unlike a bare
/// max−min of unwrapped values.
fn covered_angular_span(u_iter: impl Iterator<Item = f64>) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut us: Vec<f64> = u_iter.map(|u| u.rem_euclid(tau)).collect();
    if us.len() < 2 {
        return 0.0;
    }
    us.sort_by(f64::total_cmp);
    us.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if us.len() < 2 {
        return 0.0;
    }
    let mut max_gap = us[0] + tau - us[us.len() - 1]; // wrap-around gap
    for w in us.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    tau - max_gap
}

/// Whether the combined even-odd vertical-ray hole-clip is correct for these
/// inner loops.
///
/// The clip pairs ALL inner-loop crossings at a given `u` even-odd into removed
/// v-intervals. That is exact for the Steinmetz lens (its two seam ellipses give
/// a single clean v-interval per `u`) AND for independent holes whose v-ranges
/// are disjoint at every shared `u` (each hole's own crossings pair within its
/// own interval). It is WRONG only when two INDEPENDENT compact holes overlap in
/// BOTH `u` and `v` (their crossings interleave, and even-odd would merge/spawn
/// spurious intervals). Detect and EXCLUDE only that case: a pair of loops that
/// both stay local in `u` (neither wraps the full period — so they are not lens
/// envelopes) yet overlap in both their `u` and `v` bounding boxes. Everything
/// else (the lens; disjoint holes) is safe.
fn combined_evenodd_holes_safe(inner_loops: &[Vec<(f64, f64)>]) -> bool {
    let tau = std::f64::consts::TAU;
    if inner_loops.is_empty() {
        return false;
    }
    // (u_lo, u_hi unwrapped, v_lo, v_hi, wraps_full_period) per loop.
    let boxes: Vec<(f64, f64, f64, f64, bool)> = inner_loops
        .iter()
        .filter(|l| l.len() >= 2)
        .map(|l| {
            let u_lo = l.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let u_hi = l.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let v_lo = l.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let v_hi = l.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let wraps = covered_angular_span(l.iter().map(|p| p.0)) >= tau - 1.0;
            (u_lo, u_hi, v_lo, v_hi, wraps)
        })
        .collect();
    if boxes.len() < 2 {
        return true; // A single hole pairs cleanly.
    }
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            let (a, b) = (boxes[i], boxes[j]);
            // Lens envelopes wrap the period; skip the dangerous-overlap test
            // for them (their interleaving IS the lens the clip handles).
            if a.4 || b.4 {
                continue;
            }
            let u_overlap = a.0.max(b.0) < a.1.min(b.1);
            let v_overlap = a.2.max(b.2) < a.3.min(b.3);
            if u_overlap && v_overlap {
                return false; // Two independent compact holes interleave — defer.
            }
        }
    }
    true
}

/// Absolute shoelace area of a UV polygon. Near-zero means the boundary has
/// collapsed onto a line or point (a degenerate seam/pole projection).
fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        a += x0 * y1 - x1 * y0;
    }
    (a * 0.5).abs()
}

/// Dispatch to trimmed or untrimmed parametric integration based on whether
/// a UV boundary polygon is available.
#[allow(clippy::too_many_arguments)]
fn integrate_with_trimming<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    uv_boundary: &[(f64, f64)],
    u_periodic: bool,
    hole_vs: &[f64],
    inner_loops: &[Vec<(f64, f64)>],
) -> FaceContribution {
    if uv_boundary.len() < 3 {
        return integrate_parametric(surface, u_range, v_range, gauss_order, sign);
    }

    // The dense boundary polygon is the reliable signal for a face's true
    // parametric extent: `face_uv_bounds` samples only sparse edge endpoints and
    // under-spans full-revolution faces (a cone's lateral face reports a narrow
    // u-range though its boundary wraps the full 2pi). A face that wraps the
    // full period in u, or whose boundary collapses onto a seam or pole, cannot
    // be trimmed by a UV polygon — the apex/pole/seam folds the polygon and the
    // point-in-polygon test rejects valid interior samples. Integrate the
    // analytic surface untrimmed over its true domain in those cases.
    let u_min = uv_boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min);
    let v_min = uv_boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    let v_max = uv_boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    // Winding number of the boundary around the periodic u-axis: ±TAU for a
    // face that wraps a full revolution, ~0 for a partially-trimmed face.
    // Computed from shortest signed steps so it is independent of the
    // boundary's discretization (segment count).
    let tau = std::f64::consts::TAU;
    let winding: f64 = (0..uv_boundary.len())
        .map(|i| {
            let d = uv_boundary[(i + 1) % uv_boundary.len()].0 - uv_boundary[i].0;
            d - tau * ((d + std::f64::consts::PI) / tau).floor()
        })
        .sum();
    // The u-extent the boundary actually COVERS. A wall bounded by two full
    // cap circles (each projecting across the whole 0..2π period) plus a seam
    // line nets a winding of ~0 — the two circles wind oppositely — so the
    // winding test alone misses it. A bare max−min is fragile too: a PARTIAL
    // face whose boundary straddles the seam can have max−min ≈ 2π without
    // covering the full period. The robust signal is the complement of the
    // LARGEST angular gap between sorted boundary u-samples: a full revolution
    // has no large gap (covered span ≈ 2π); a partial face's gap is its
    // uncovered arc.
    // `u_covered` (gap-complement) is robust to a boundary that straddles the
    // seam, where a bare max−min would read ≈2π for a partial face. The
    // tolerance (covered ≥ 2π − 1, i.e. largest gap < 1 rad) admits the discrete
    // boundary SAMPLING gap of a genuine full-revolution wall (its two cap
    // circles cover every u) while a truly partial face leaves a large
    // uncovered arc. The winding term still catches a single-loop band (a cone
    // lateral face) whose boundary winds a full ±2π.
    let u_covered = covered_angular_span(uv_boundary.iter().map(|p| p.0));
    let full_revolution = u_periodic && (winding.abs() >= tau - 1e-3 || u_covered >= tau - 1.0);
    let v_degenerate = (v_max - v_min) <= 1e-9;

    // A full-revolution wall (cylinder/cone tube) carrying the STEINMETZ LENS
    // (two mutually-trimmed cylinders): each lens loop is a full-u-wrapping
    // sinusoid, and TOGETHER they enclose the lens by an even-odd combined
    // arrangement. Integrate the whole revolution over the wall's v-extent with
    // those lens v-intervals clipped out. Gated TIGHTLY to that signature:
    // ordinary INDEPENDENT holes (each a local u-arc) would be mis-handled by
    // the combined even-odd test (it would subtract spurious merged intervals),
    // so they fall through to the standard paths below.
    if u_periodic && !v_degenerate && full_revolution && combined_evenodd_holes_safe(inner_loops) {
        return integrate_band_excluding_holes(
            surface,
            (u_min, u_min + tau),
            (v_min, v_max),
            gauss_order,
            sign,
            inner_loops,
        );
    }

    if full_revolution && v_degenerate {
        // Polar cap (e.g. a sphere hemisphere bounded only by one latitude
        // circle): the cap runs from that latitude to a pole. The winding sign
        // (CCW vs CW boundary) selects which pole — the boundary's interior
        // side — so the two hemispheres do not both integrate the whole sphere.
        let v_pole = if winding >= 0.0 { v_range.1 } else { v_range.0 };
        // A full-revolution hole at a latitude between the outer circle and the
        // pole (the drilled-tunnel rim) clips the cap into a band: integrate
        // only from the outer latitude to the hole, not on to the pole.
        let v_far = hole_vs
            .iter()
            .copied()
            // Same side of v_min as the pole (strict same sign → positive
            // product), and not coincident with v_min.
            .filter(|&hv| (hv - v_min) * (v_pole - v_min) > 0.0 && (hv - v_min).abs() > 1e-9)
            .min_by(|a, b| (a - v_min).abs().total_cmp(&(b - v_min).abs()))
            .unwrap_or(v_pole);
        let v_dom = (v_min.min(v_far), v_min.max(v_far));
        integrate_parametric(surface, (u_min, u_min + tau), v_dom, gauss_order, sign)
    } else if full_revolution {
        // Full-revolution band (cone/cylinder) with no interior holes (the
        // holed case is handled by the early return above): integrate the whole
        // revolution over the band's v-extent.
        integrate_parametric(
            surface,
            (u_min, u_min + tau),
            (v_min, v_max),
            gauss_order,
            sign,
        )
    } else if polygon_area(uv_boundary) <= 1e-12 {
        // Collapsed polygon (e.g. a closed torus whose seam projects to a
        // point): trust the analytic full-domain range from `face_uv_bounds`.
        integrate_parametric(surface, u_range, v_range, gauss_order, sign)
    } else {
        integrate_parametric_trimmed(
            surface,
            u_range,
            v_range,
            gauss_order,
            sign,
            uv_boundary,
            u_periodic,
        )
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

    let uv_poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();

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

    for i in 1..uv.len() {
        if u_periodic {
            uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);
        }
    }

    Ok(uv)
}

/// Build a UV polygon per inner wire (hole) of a face, projecting each loop's
/// densely-sampled boundary onto the surface and unwrapping periodic u so a
/// seam-crossing hole (e.g. the closed ellipse where one cylinder bores
/// through another) stays a simple, non-self-crossing loop in its own
/// u-window.
fn build_face_uv_inner_loops<F>(
    topo: &Topology,
    face_id: FaceId,
    project: F,
    u_periodic: bool,
) -> Result<Vec<Vec<(f64, f64)>>, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    let face = topo.face(face_id)?;
    let mut loops = Vec::new();
    for &wid in face.inner_wires() {
        let polygon = crate::util::wire_polygon(topo, wid)?;
        if polygon.len() < 3 {
            continue;
        }
        let mut uv: Vec<(f64, f64)> = polygon.iter().map(|&p| project(p)).collect();
        if u_periodic {
            for i in 1..uv.len() {
                uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);
            }
        }
        loops.push(uv);
    }
    Ok(loops)
}

#[cfg(test)]
mod gate_tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::{combined_evenodd_holes_safe, covered_angular_span};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    fn ellipse_uv(v_mid: f64, amp: f64, phase: f64, n: usize) -> Vec<(f64, f64)> {
        // v = v_mid + amp·cos(u − phase) over a full revolution in u (a lens
        // envelope's projected sinusoid).
        (0..=n)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let u = TAU * (k as f64) / (n as f64);
                (u, v_mid + amp * (u - phase).cos())
            })
            .collect()
    }

    fn local_loop(u_c: f64, v_c: f64, r: f64, n: usize) -> Vec<(f64, f64)> {
        // A small compact circular hole centred at (u_c, v_c).
        (0..=n)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let t = TAU * (k as f64) / (n as f64);
                (u_c + r * t.cos(), v_c + r * t.sin())
            })
            .collect()
    }

    #[test]
    fn covered_span_full_vs_partial() {
        // A full revolution (32 evenly-spaced samples): covered span ≈ 2π.
        let full = (0..32).map(|k| TAU * f64::from(k) / 32.0);
        assert!((covered_angular_span(full) - TAU).abs() < 0.3);
        // A quarter arc near the seam [-π/4, π/4]: covered span ≈ π/2, NOT 2π
        // (a bare max−min of these seam-straddling values would read ≈2π via
        // the rem_euclid wrap, but the gap-complement correctly reports ~π/2).
        let quarter = (0..=8).map(|k| -FRAC_PI_4 + FRAC_PI_2 * f64::from(k) / 8.0);
        assert!(
            covered_angular_span(quarter) < PI,
            "a quarter arc is not a full revolution"
        );
    }

    #[test]
    fn combined_evenodd_safe_for_lens_unsafe_for_overlapping_independent() {
        // Steinmetz lens: two full-u-wrapping envelopes → SAFE.
        let lens = vec![
            ellipse_uv(10.0, 3.0, 0.0, 64),
            ellipse_uv(10.0, -3.0, 0.0, 64),
        ];
        assert!(
            combined_evenodd_holes_safe(&lens),
            "lens envelopes are safe"
        );

        // Two independent compact holes that OVERLAP in both u and v → UNSAFE
        // (combined even-odd would interleave them).
        let overlapping = vec![local_loop(1.0, 5.0, 0.4, 32), local_loop(1.3, 5.1, 0.4, 32)];
        assert!(
            !combined_evenodd_holes_safe(&overlapping),
            "overlapping independent holes must defer"
        );

        // Two independent compact holes that are DISJOINT in u → SAFE (each
        // pairs within its own interval).
        let disjoint = vec![local_loop(1.0, 5.0, 0.3, 32), local_loop(4.0, 5.0, 0.3, 32)];
        assert!(
            combined_evenodd_holes_safe(&disjoint),
            "u-disjoint holes pair cleanly"
        );

        // A single hole is always safe.
        assert!(combined_evenodd_holes_safe(&[local_loop(
            1.0, 5.0, 0.3, 32
        )]));
        // No holes → not applicable.
        assert!(!combined_evenodd_holes_safe(&[]));
    }
}
