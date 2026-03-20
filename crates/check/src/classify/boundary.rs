//! UV boundary polygon construction and containment tests for face trimming.
//!
//! Provides the core algorithms for determining whether a ray-surface hit
//! point falls within a face's trimming boundary, using UV-space projection
//! for analytic surfaces and 3D polygon containment for surfaces with
//! pole singularities (spheres).

#![allow(dead_code)]

use std::f64::consts::PI;

use smallvec::SmallVec;

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::CheckError;
use crate::classify::ray_surface;
use crate::util::{face_polygon, point_in_polygon_3d};

// ---------------------------------------------------------------------------
// Tolerance constants
// ---------------------------------------------------------------------------

/// Near-zero threshold for floating-point denominators and discriminants.
const NEAR_ZERO: f64 = 1e-15;

/// Minimum positive ray parameter to count as a forward hit.
const RAY_T_MIN: f64 = 1e-12;

/// Threshold for half-space sign test (negative side rejection).
const HALF_SPACE_EPS: f64 = 1e-10;

/// Near-zero threshold for degenerate vector length.
const DEGENERATE_LEN: f64 = 1e-30;

/// Threshold for coincident vertex detection (squared distance).
const COINCIDENT_SQ: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Angular unwrapping
// ---------------------------------------------------------------------------

/// Unwrap a step in a periodic (angular) coordinate so the difference
/// lies in `[-PI, PI)`.
///
/// Given the previous unwrapped value `prev` and the next raw value `next`,
/// returns the next value adjusted so the step is continuous.
#[inline]
fn unwrap_angle(prev: f64, next: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let diff = next - prev;
    prev + diff - tau * ((diff + PI) / tau).floor()
}

// ---------------------------------------------------------------------------
// UV boundary construction
// ---------------------------------------------------------------------------

/// Build a UV boundary polygon from 3D face boundary vertices,
/// with proper unwrapping of periodic coordinates.
///
/// `v_periodic`: whether the v-coordinate is periodic (e.g. torus). Cylinder
/// and cone have linear v (height / distance), so only u is unwrapped for them.
fn build_uv_boundary<F>(verts: &[Point3], project: &F, v_periodic: bool) -> Vec<(f64, f64)>
where
    F: Fn(Point3) -> (f64, f64),
{
    let mut uv: Vec<(f64, f64)> = verts.iter().map(|&p| project(p)).collect();

    for i in 1..uv.len() {
        // u is always periodic (angular coordinate for all analytic surfaces).
        uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);

        // v is periodic only for doubly-periodic surfaces (torus).
        if v_periodic {
            uv[i].1 = unwrap_angle(uv[i - 1].1, uv[i].1);
        }
    }

    uv
}

// ---------------------------------------------------------------------------
// UV containment test
// ---------------------------------------------------------------------------

/// Test if a (u,v) point is inside the UV boundary polygon.
///
/// Adjusts the test point's u coordinate (and v when periodic) to lie within
/// the unwrapped polygon's coordinate range before testing.
fn point_in_uv_boundary(
    hit_u: f64,
    hit_v: f64,
    uv_boundary: &[(f64, f64)],
    v_periodic: bool,
) -> bool {
    // Find the u range of the unwrapped boundary.
    let u_min = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::INFINITY, f64::min);
    let u_max = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::NEG_INFINITY, f64::max);
    let u_center = (u_min + u_max) * 0.5;

    // Shift hit_u to be closest to the polygon's u center.
    let hu = unwrap_angle(u_center, hit_u);

    // For doubly-periodic surfaces (torus), also shift hit_v.
    let hv = if v_periodic {
        let v_min = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::INFINITY, f64::min);
        let v_max = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let v_center = (v_min + v_max) * 0.5;
        unwrap_angle(v_center, hit_v)
    } else {
        hit_v
    };

    let poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();
    let test = Point2::new(hu, hv);
    point_in_polygon(test, &poly)
}

// ---------------------------------------------------------------------------
// Polygon normal (Newell's method)
// ---------------------------------------------------------------------------

/// Compute the normal of a polygon via Newell's method.
///
/// Returns a unit-length normal, or `(0,0,1)` for degenerate polygons.
pub fn polygon_normal(verts: &[Point3]) -> Vec3 {
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    let n = verts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = verts[i];
        let vj = verts[j];
        nx += (vi.y() - vj.y()) * (vi.z() + vj.z());
        ny += (vi.z() - vj.z()) * (vi.x() + vj.x());
        nz += (vi.x() - vj.x()) * (vi.y() + vj.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < DEGENERATE_LEN {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(nx / len, ny / len, nz / len)
    }
}

// ---------------------------------------------------------------------------
// Crossing counters
// ---------------------------------------------------------------------------

/// Count crossings for analytic (non-planar) faces using UV containment.
///
/// Given ray parameter roots (where the ray hits the infinite surface),
/// checks whether each hit point falls within the face's trimming boundary
/// by projecting to the surface's (u,v) parameter space.
///
/// If the face boundary is degenerate (all vertices coincide, as in a full
/// torus face with seam edges), every positive-t root is counted as a crossing.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
fn count_analytic_crossings<F>(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &SmallVec<[f64; 4]>,
    project: F,
    v_periodic: bool,
) -> Result<u32, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    if roots.is_empty() {
        return Ok(0);
    }

    // Build UV boundary polygon from face wire.
    let verts = face_polygon(topo, face_id)?;

    // Detect degenerate boundary: a "full-surface" face whose wire has fewer
    // than 3 distinct vertices. Every positive-t root is a crossing.
    let is_full_surface = verts.len() < 3 || {
        let ref_pt = verts[0];
        verts
            .iter()
            .all(|v| (*v - ref_pt).length_squared() < COINCIDENT_SQ)
    };
    if is_full_surface {
        #[allow(clippy::cast_possible_truncation)]
        return Ok(roots.iter().filter(|&&t| t > RAY_T_MIN).count() as u32);
    }

    let uv_boundary = build_uv_boundary(&verts, &project, v_periodic);

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;
        let (hit_u, hit_v) = project(hit);

        if point_in_uv_boundary(hit_u, hit_v, &uv_boundary, v_periodic) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Count crossings using 3D polygon containment (for faces with planar
/// boundaries, e.g. sphere hemispheres where UV projection has pole
/// singularities).
///
/// The polygon normal (from Newell's method) indicates which side of the
/// boundary plane the face extends into. A hit point must be on that side
/// AND project inside the boundary polygon.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
fn count_3d_polygon_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &SmallVec<[f64; 4]>,
) -> Result<u32, CheckError> {
    if roots.is_empty() {
        return Ok(0);
    }

    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }
    let mut normal = polygon_normal(&verts);
    // If the face is reversed, the surface normal is flipped.
    let face = topo.face(face_id)?;
    if face.is_reversed() {
        normal = -normal;
    }
    let ref_pt = verts[0];

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;

        // The hit must be on the face's side of the boundary plane.
        let side = (hit - ref_pt).dot(normal);
        if side < -HALF_SPACE_EPS {
            continue;
        }

        if point_in_polygon_3d(&hit, &verts, &normal) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Count ray crossings for a single face, dispatching by surface type.
///
/// For plane faces, uses direct ray-plane + 3D polygon containment.
/// For analytic curved faces, uses ray-surface intersection + UV containment.
/// For sphere faces, uses 3D polygon containment (avoids UV pole singularity).
/// For NURBS faces, uses line-surface intersection.
///
/// # Errors
///
/// Returns an error if topology lookups or intersection computations fail.
#[allow(clippy::too_many_lines)]
pub fn count_face_ray_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
) -> Result<u32, CheckError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            ray_plane_crossings(topo, face_id, origin, direction, *normal, *d)
        }
        FaceSurface::Cylinder(cyl) => {
            let cyl = cyl.clone();
            let roots = ray_surface::ray_cylinder(origin, direction, &cyl);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cyl.project_point(p),
                false,
            )
        }
        FaceSurface::Cone(cone) => {
            let cone = cone.clone();
            let roots = ray_surface::ray_cone(origin, direction, &cone);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cone.project_point(p),
                false,
            )
        }
        FaceSurface::Sphere(sph) => {
            // Sphere boundaries are planar (equator, small circles), so
            // point_in_polygon_3d works. UV projection fails at poles.
            let sph = sph.clone();
            let roots = ray_surface::ray_sphere(origin, direction, &sph);
            count_3d_polygon_crossings(topo, face_id, origin, direction, &roots)
        }
        FaceSurface::Torus(tor) => {
            let tor = tor.clone();
            let roots = ray_surface::ray_torus(origin, direction, &tor);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| tor.project_point(p),
                true,
            )
        }
        FaceSurface::Nurbs(surface) => {
            ray_crossings_nurbs(topo, face_id, origin, direction, surface)
        }
    }
}

/// Ray-plane intersection with point-in-polygon boundary test.
fn ray_plane_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    normal: Vec3,
    d: f64,
) -> Result<u32, CheckError> {
    let t = match ray_surface::ray_plane(origin, direction, normal, d) {
        Some(t) => t,
        None => return Ok(0),
    };

    let hit = origin + direction * t;
    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }

    if point_in_polygon_3d(&hit, &verts, &normal) {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Count ray crossings for a NURBS face using ray-surface intersection.
fn ray_crossings_nurbs(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<u32, CheckError> {
    let hits = ray_surface::ray_nurbs(origin, direction, surface, 20)?;
    if hits.is_empty() {
        return Ok(0);
    }

    // Build UV boundary from face wire vertices.
    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        // Full-surface face — every forward hit is a crossing.
        #[allow(clippy::cast_possible_truncation)]
        return Ok(hits.len() as u32);
    }

    let project = |p: Point3| -> (f64, f64) { surface.project_point(p) };
    let uv_boundary = build_uv_boundary(&verts, &project, false);

    let mut crossings = 0u32;
    for (_, hit_u, hit_v) in &hits {
        if point_in_uv_boundary(*hit_u, *hit_v, &uv_boundary, false) {
            crossings += 1;
        }
    }

    Ok(crossings)
}
